"""Canonicalize values and publish equivalent typed CPU execution bodies."""

import json
from pathlib import Path
from dataclasses import asdict

from scar.contract import (
    ElementwiseEntry, ExecutableIR, FragmentModule, Plan, ScalarModule, Step, copy_artifact, local_member,
    open_graph, read_artifact, write_json, write_plan,
)

from .native import rewrite_native_regions
from .portable import canonical_values, rewrite_portable


def _structural_schedule(graph):
    """Retain forward dependencies across nested regions as templates."""
    dependencies = {operation.id: {} for operation in graph.operations}
    boundaries = {region.id for region in graph.regions}
    known = dependencies.keys() | boundaries
    for edge in graph.edges:
        if (edge.kind in ('DATA', 'CONTROL') and edge.source != edge.target
                and edge.source in known and edge.target in known):
            dependencies.setdefault(edge.source, {})
            dependencies.setdefault(edge.target, {})[edge.source] = None
    return tuple(Step(key, depends_on=tuple(inputs)) for key, inputs in dependencies.items())


def _rank_two_shape(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(size) is not int or size <= 0 for size in value)):
        raise ValueError('input shape must be an exact positive rank-two shape')
    return list(value)


def _input_shape(value, portable):
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_bytes())
    if type(value) is not dict:
        return _rank_two_shape(value)
    entry = next((row for row in portable['functions'] if row['name'] == portable['entry']), None)
    if entry is None or not value or value.keys() != set(entry['parameters']):
        raise ValueError('input shapes must name the original Python array parameters')
    shapes = {tuple(_rank_two_shape(shape)) for shape in value.values()}
    if len(shapes) != 1:
        raise ValueError('the portable array profile requires one common shape')
    return list(shapes.pop())


def _seal_fragments(graph, reference, product, root):
    material = json.loads(read_artifact(reference))
    version = material.get('schema_version')
    seals = material.pop('region_seals', None) if version == 2 else None
    if (set(material) != {'schema', 'schema_version', 'configuration', 'fragments', 'exports'}
            or material['schema'] != 'scar.native-units-ir' or version not in (1, 2)
            or (version == 2 and type(seals) is not dict)):
        raise ValueError('invalid frontend fragment material')
    regions, rewrites = rewrite_native_regions(graph, material['fragments'], seals)
    material['schema_version'] = 1
    members = {}
    for name, file in material['fragments'].items():
        identity = file['sha256']
        owned = members.get(identity)
        if owned is None:
            copied = copy_artifact(file, product / 'blobs' / identity)
            members[identity] = owned = asdict(local_member(copied, root))
        material['fragments'][name] = owned
    manifest = write_json(product / 'fragments.json', material)
    member = local_member(manifest, root)
    return FragmentModule(member, regions), FragmentModule(member), rewrites


def _seal_cases(target, product, root):
    if not target.get('cases'):
        return {}
    baseline = dict(target['baseline'])
    executable = copy_artifact(baseline.pop('receipt'), product / 'original', executable=True)
    baseline['executable'] = asdict(local_member(executable, root))
    baseline['argv'] = baseline['argv'][1:]
    cases = []
    for index, original in enumerate(target['cases']):
        case = dict(original)
        case.pop('stdin', None)
        if receipt := case.pop('input_receipt', None):
            copied = copy_artifact(receipt, product / f'case-{index}.stdin')
            case['input'] = asdict(local_member(copied, root))
        cases.append(case)
    return dict(baseline=baseline, cases=cases)


def _lowering(graph, product, root, *, input_shape, trip_count):
    cost = dict(basis='UNKNOWN', original_duration_ns=None, trial_duration_ns=None)
    decision = dict(status='KEEP_ORIGINAL', deployment_authorized=False,
                    cost=cost, timing_scope='REGION_TEMPLATES_UNPRICED')
    fragments = graph.evidence.get('native_units')
    scalar_functions = graph.evidence.get('scalar_functions')
    if fragments or scalar_functions:
        if input_shape is not None or trip_count is not None:
            raise ValueError('array specialization applies to an elementwise entry')
        bodies, original, rewrites = (), (), ()
        if fragments:
            transformed, unchanged, rewrites = _seal_fragments(graph, fragments, product, root)
            bodies, original = (transformed,), (unchanged,)
        if scalar_functions:
            scalar = ScalarModule(tuple(scalar_functions))
            bodies, original = (*bodies, scalar), (*original, scalar)
        if rewrites:
            decision.update(status='STRUCTURAL_TRIAL', reason='GUARDED_COUNTED_REGION_REWRITE')
        return ExecutableIR(bodies), ExecutableIR(original), decision, rewrites

    portable = graph.evidence.get('portable')
    if not portable or input_shape is None or trip_count is None:
        reason = 'NO_EXECUTABLE_BODY' if not portable else 'INPUT_SHAPE_AND_TRIP_COUNT_REQUIRED'
        return ExecutableIR(), None, dict(decision, reason=reason,
            detail=graph.evidence.get('native_units_frontier')), ()
    shape = _input_shape(input_shape, portable)
    if type(trip_count) is not int or not 0 < trip_count < 2**63:
        raise ValueError('trip count must be a positive signed 64-bit integer')
    entry = next(row for row in portable['functions'] if row['name'] == portable['entry'])
    input_bytes = 8 * shape[0] * shape[1] * len(entry['parameters'])
    if input_bytes + 8 * shape[0] * shape[1] >= 2**63:
        raise ValueError('float64 input payload exceeds signed 64-bit size bounds')
    program, rewrites = rewrite_portable(graph, portable, trip_count)
    if program is None:
        return ExecutableIR(), None, dict(decision, reason='ARRAY_PROGRAM_GRAPH_LINK_UNRESOLVED'), ()
    decision.update(status='STRUCTURAL_TRIAL', reason='ARRAY_ENTRY_WITH_EXPLICIT_SPECIALIZATION',
                    reference_scope='ORIGINAL_LOGICAL_COMPUTE_IR', input_payload_bytes=input_bytes,
                    lowering_rewritten=program != portable)
    selected = ExecutableIR((ElementwiseEntry(program, tuple(shape), trip_count),))
    original = ExecutableIR((ElementwiseEntry(portable, tuple(shape), trip_count),))
    return selected, original, decision, rewrites


def plan(source, out, *, input_shape=None, trip_count=None):
    """Open one frontend product and publish one legal, sealed lowering plan."""
    reference, graph = open_graph(source)
    path = Path(out)
    if path.is_dir() or path.suffix != '.json':
        path /= 'plan.json'
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    product = path.parent / f'{path.stem}.payload'
    lowering, original, decision, changes = _lowering(graph, product, path.parent,
        input_shape=input_shape, trip_count=trip_count)
    schedule = _structural_schedule(graph)
    validation = _seal_cases(graph.target, product, path.parent) if lowering.bodies else {}
    result = Plan(reference, validation, (*canonical_values(graph), *changes), schedule, lowering, decision,
                  reference_lowering=original, resources=graph.resources)
    write_plan(path, result)
    return result

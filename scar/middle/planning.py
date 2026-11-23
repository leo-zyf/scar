"""Canonicalize exact values, retain dependencies and bind CPU work."""

import json
import math
from pathlib import Path
from dataclasses import asdict

from scar.contract import (
    ElementwiseEntry, ExecutableIR, FragmentModule, Plan, Step, copy_artifact, local_member,
    open_graph, read_artifact, write_json, write_plan,
)


def _canonical_values(graph):
    representatives, rewrites = {}, []
    for operation in graph.operations:
        if operation.kind != 'CONSTANT' or operation.effects != ('PURE',):
            continue
        dtype = operation.attributes.get('dtype')
        if type(dtype) is not str or 'value' not in operation.attributes:
            continue
        value = operation.attributes['value']
        if type(value) is float and not math.isfinite(value):
            continue
        if type(value) not in (bool, int, float, str, type(None)):
            continue
        # Exact payload equality leaves object identity and operation execution intact.
        payload = value.hex() if type(value) is float else value
        key = operation.region, dtype, type(value), payload
        canonical = representatives.setdefault(key, operation.id)
        if canonical != operation.id:
            rewrites.append(dict(rule='CANONICAL_LITERAL_VALUE', operation=operation.id,
                                 canonical=canonical, region=operation.region, dtype=dtype,
                                 scope='TYPED_PAYLOAD', source_edit=False))
    return tuple(rewrites)


def _cpu_resources(graph):
    cpus = {}
    for resource in graph.resources:
        if resource.kind != 'CPU' or resource.capacity != 1:
            continue
        logical_id = resource.attributes.get('logical_cpu')
        if type(logical_id) is not int or logical_id < 0:
            raise ValueError('CPU resources must name their logical CPU')
        if logical_id in cpus:
            raise ValueError('logical CPU has more than one resource identity')
        cpus[logical_id] = resource
    return tuple(sorted(cpus.items()))


def _structural_schedule(graph, cpu):
    """Retain local precedence once per template, including loops as regions."""
    if cpu is None:
        return ()
    operations = {operation.id: operation.region for operation in graph.operations}
    dependencies = {key: {} for key in operations}
    for edge in graph.edges:
        if (edge.kind in ('DATA', 'CONTROL') and edge.source != edge.target
                and edge.source in operations and edge.target in operations
                and operations[edge.source] == operations[edge.target]):
            dependencies[edge.target][edge.source] = None
    return tuple(Step(key, cpu.id, depends_on=tuple(dependencies[key]))
                 for region in graph.regions for key in region.operations)


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


def _cpu_cohort(cpus, workers):
    # Place distinct physical cores first, within one NUMA domain when possible.
    domains = {}
    for cpu, resource in cpus:
        domains.setdefault(resource.attributes.get('numa_node'), []).append((cpu, resource))
    available = [group for group in domains.values() if len(group) >= workers]
    group = min(available, key=lambda rows: rows[0][0]) if available else cpus
    first, siblings, physical = [], [], set()
    for cpu, resource in group:
        package, core = (resource.attributes.get(field) for field in ('package_id', 'core_id'))
        key = (package, core) if package is not None and core is not None else ('logical', cpu)
        (siblings if key in physical else first).append(cpu)
        physical.add(key)
    return tuple((first + siblings)[:workers])


def _seal_fragments(reference, product, root):
    material = json.loads(read_artifact(reference))
    if (set(material) != {'schema', 'schema_version', 'configuration', 'fragments', 'exports'}
            or material['schema'] != 'scar.native-units-ir' or material['schema_version'] != 1):
        raise ValueError('invalid frontend fragment material')
    members = {}
    for name, file in material['fragments'].items():
        identity = file['sha256']
        owned = members.get(identity)
        if owned is None:
            copied = copy_artifact(file, product / 'blobs' / identity)
            members[identity] = owned = asdict(local_member(copied, root))
        material['fragments'][name] = owned
    manifest = write_json(product / 'fragments.json', material)
    return FragmentModule(local_member(manifest, root))


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


def _lowering(graph, cpus, product, root, *, input_shape, trip_count, threads):
    cost = dict(basis='UNKNOWN', original_duration_ns=None, trial_duration_ns=None)
    decision = dict(status='KEEP_ORIGINAL', deployment_authorized=False,
                    cost=cost, timing_scope='REGION_TEMPLATES_UNPRICED')
    if reference := graph.evidence.get('native_units'):
        if any(value is not None for value in (input_shape, trip_count, threads)):
            raise ValueError('array specialization applies to an elementwise entry')
        return ExecutableIR((_seal_fragments(reference, product, root),)), decision

    portable = graph.evidence.get('portable')
    if not portable or input_shape is None or trip_count is None:
        reason = 'NO_EXECUTABLE_BODY' if not portable else 'INPUT_SHAPE_AND_TRIP_COUNT_REQUIRED'
        return ExecutableIR(), dict(decision, reason=reason, detail=graph.evidence.get('native_units_frontier'))
    shape = _input_shape(input_shape, portable)
    if type(trip_count) is not int or not 0 < trip_count < 2**63:
        raise ValueError('trip count must be a positive signed 64-bit integer')
    if 8 * shape[0] * shape[1] >= 2**63:
        raise ValueError('float64 input payload exceeds signed 64-bit size bounds')
    workers = 1 if threads is None else threads
    if type(workers) is not int or workers < 1 or workers > len(cpus):
        raise ValueError('CPU thread count must fit the captured CPU resources')
    body = ElementwiseEntry(portable, tuple(shape), trip_count)
    return ExecutableIR((body,), _cpu_cohort(cpus, workers)), dict(
        decision, status='STRUCTURAL_TRIAL', reason='ARRAY_ENTRY_WITH_EXPLICIT_SPECIALIZATION',
        input_payload_bytes=8 * shape[0] * shape[1])


def plan(source, out, *, input_shape=None, trip_count=None, threads=None):
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
    rewrites = _canonical_values(graph)
    cpus = _cpu_resources(graph)
    lowering, decision = _lowering(graph, cpus, product, path.parent,
                                   input_shape=input_shape, trip_count=trip_count, threads=threads)
    selected_cpu = dict(cpus)[lowering.cpu_ids[0]] if lowering.cpu_ids else cpus[0][1] if cpus else None
    schedule = _structural_schedule(graph, selected_cpu)
    validation = _seal_cases(graph.target, product, path.parent) if lowering.bodies else {}
    result = Plan(reference, validation, rewrites, schedule, lowering, decision)
    write_plan(path, result)
    return result

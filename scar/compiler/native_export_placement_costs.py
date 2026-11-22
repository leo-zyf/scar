"""Native export placement costs."""

from dataclasses import dataclass, replace
import json
import sys
from pathlib import Path, PurePosixPath
from scar.compiler.calibrate_native_device import replay_native_device_calibration
from scar.compiler.compile_exported_aten_cuda import ACTIVITY_BINDING, PLACEMENT_RECORDS
from scar.compiler.native_device_pair import (
    CUDA_SCHEMA,
    _identity,
    _manifest,
    _member,
    CPU_SCHEMA,
    _package,
    _ports,
    _selection,
)
from scar.ir.implementation_plan import (
    digest,
    problem_document,
    problem_from_dict,
    validate_problem,
)
from scar.ir.v2.schemas import canonical_json
from scar.optimization.constructive_planner import construct_implementations
from scar.optimization.joint_planner import verify_execution_plan
from scar.optimization.native_device_activity_costs import _queue_order
from scar.compiler.aten_export_hybrid_plan import (
    HOST_RESOURCE,
    _PAGEABLE_HOST,
    _RETURN,
    plan_placed_hybrid_export,
)
from scar.compiler.aten_export_hybrid_source import qualify_hybrid_export
from scar.compiler.tensor_execution import (
    operation_costs,
    replay_operation_costs,
    scheduled_tensor_body,
)
from scar.ir.program import codec
from scar.ir.record_codec import encode
from scar.optimization.implementation_costs import CostQuery, query_key
from scar.compiler.aten_export_device_placement import placement_choices, placement_proposal
from scar.optimization.native_device_placement_search import placement_assignment


# Admit native CUDA service costs for one archived physical placement.
@dataclass(frozen=True, slots=True)
class NativeHybridPlacementCosts:
    record: dict
    original_problem: dict
    original_schedule: dict
    bindings: dict
    repriced_problem: dict
    repriced_schedule: dict
    emitted_activities: tuple[dict, ...] = ()


def _receipt_status(observation, summary):
    if observation['phase'] != 'COMPLETE':
        return observation['phase']
    if summary['unknown_activities'] or not summary['known_activity_count']:
        return 'INCOMPLETE_NATIVE_ACTIVITY_COSTS'
    if summary['schedule_status'] != 'FEASIBLE_WITHIN_MODEL':
        return 'NO_VERIFIED_NATIVE_SERVICE_SCHEDULE'
    return 'EXACT_OBSERVED_TREATMENT'


def load_native_hybrid_placement_costs(calibration, *, calibration_sha256, cpu_project, cuda_project):
    """Replay locked artifacts and retain their positive or negative observation."""
    calibration = Path(calibration).resolve(strict=True)
    report = replay_native_device_calibration(calibration, calibration_sha256=calibration_sha256,
                                              cpu_project=cpu_project, cuda_project=cuda_project)
    pair = report['pair']
    cuda = Path(cuda_project).resolve(strict=True)
    manifest = _manifest(cuda, pair['cuda']['manifest_sha256'], CUDA_SCHEMA)
    placement_record = PLACEMENT_RECORDS.get(manifest['status'])
    if placement_record is None:
        raise ValueError('empirical placement costs require a compiled typed placement archive')
    proposal = json.loads(_member(cuda, manifest, f'plan/{placement_record}.json'))
    original_problem = json.loads(_member(cuda, manifest, 'plan/problem.json'))
    original_schedule = json.loads(_member(cuda, manifest, 'plan/schedule.json'))
    bindings = json.loads(_member(cuda, manifest, 'plan/bindings.json'))
    observation = json.loads(_member(calibration.parent, report, 'observation.json'))
    repriced_problem = json.loads(_member(calibration.parent, report, 'problem.json'))
    repriced_schedule = json.loads(_member(calibration.parent, report, 'schedule.json'))

    reason = _receipt_status(observation, report['summary'])
    activities = ()
    if reason == 'EXACT_OBSERVED_TREATMENT':
        activity_binding = json.loads(_member(cuda, manifest, ACTIVITY_BINDING))
        activities = tuple(activity_binding['activities'])
    record = dict(calibration_sha256=calibration_sha256,
        calibration_files=report['files'], activity_binding=_identity(manifest['files'][ACTIVITY_BINDING]),
        cpu_manifest_sha256=pair['cpu']['manifest_sha256'], cuda_manifest_sha256=pair['cuda']['manifest_sha256'],
        cpu_profile=pair['cpu_profile'], symbolic_profile=pair['symbolic_profile'],
        source=pair['source'], program=pair['program'], profile_id=original_problem['target']['sha256'],
        placement=proposal['selected_placement'], device_ordinal=bindings['device'],
        inputs=[_identity(row) for row in observation['inputs']],
        references=[_identity(row) for row in observation['references']],
        context=observation['context'], status=observation['phase'], reason=reason,
        cost_available=reason == 'EXACT_OBSERVED_TREATMENT')
    return NativeHybridPlacementCosts(record, original_problem, original_schedule, bindings,
                                      repriced_problem, repriced_schedule, activities)


def _relative(path):
    if type(path) is not str or not path or '\\' in path:
        return False
    parsed = PurePosixPath(path)
    return (not parsed.is_absolute() and parsed.as_posix() == path
            and all(part != '..' for part in parsed.parts) and bool(parsed.parts))


def _anchor(record):
    return record.get('sha256'), record.get('size_bytes')


def _relative_refs(value, paths):
    if isinstance(value, list):
        return [_relative_refs(item, paths) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _relative_refs(item, paths) for key, item in value.items()}
    original_paths = paths.get(_anchor(value))
    if original_paths is not None:
        original_path, expected_path = original_paths
        if 'path' in value and value['path'] == expected_path:
            result['path'] = original_path
        if value.get('uri') == 'declared:' + expected_path:
            result['uri'] = 'declared:' + original_path
    return result


def _reference_paths(original, expected):
    original_program, = original['source_refs']
    expected_program, = expected['source_refs']
    original_target, expected_target = original['target'], expected['target']
    old_uri, new_uri = original_target['uri'], expected_target['uri']
    if not old_uri.startswith('declared:') or not new_uri.startswith('declared:'):
        return None
    anchors = ((original_program, expected_program, original_program['path'], expected_program['path']),
               (original_target, expected_target, old_uri.removeprefix('declared:'), new_uri.removeprefix('declared:')))
    if any(_identity(old) != _identity(new) or not (_relative(old_path) and _relative(new_path))
           for old, new, old_path, new_path in anchors):
        return None
    return {_anchor(old): (old_path, new_path) for old, _, old_path, new_path in anchors}


def _same_schedule(original, expected, original_digest, expected_digest):
    return (original['problem_sha256'] == original_digest and expected['problem_sha256'] == expected_digest
            and canonical_json({key: value for key, value in original.items()
                                if key not in ('elapsed_seconds', 'problem_sha256')})
            == canonical_json({key: value for key, value in expected.items()
                               if key not in ('elapsed_seconds', 'problem_sha256')}))


def native_hybrid_placement_cost_matches(evidence, placement, built, *, profile_id, input_identities,
                                         reference_identities):
    """Match an exact treatment, including a negative no-device receipt."""
    record = evidence.record
    if (placement != record['placement'] or profile_id != record['profile_id']
            or [_identity(row) for row in input_identities] != record['inputs']
            or [_identity(row) for row in reference_identities] != record['references']):
        return False
    expected, expected_schedule, bindings = built
    if canonical_json(bindings) != canonical_json(evidence.bindings):
        return False
    original = evidence.original_problem
    if len(expected['source_refs']) != 1:
        return False
    paths = _reference_paths(original, expected)
    return (paths is not None and canonical_json(original) == canonical_json(_relative_refs(expected, paths))
            and _same_schedule(evidence.original_schedule, expected_schedule,
                               digest(problem_from_dict(original)), digest(problem_from_dict(expected))))


def match_native_hybrid_placement_costs(evidence, placement, built, *, profile_id, input_identities,
                                        reference_identities, host_dispatch=False):
    """Return only measured costs and the corresponding locked lane/wait schedule."""
    if not evidence.record['cost_available'] or not native_hybrid_placement_cost_matches(
            evidence, placement, built, profile_id=profile_id, input_identities=input_identities,
            reference_identities=reference_identities):
        return None
    original = problem_from_dict(built[0])
    measured = problem_from_dict(evidence.repriced_problem)
    operations = tuple(replace(operation, options=(replace(operation.options[0], cost=priced.options[0].cost),))
                       for operation, priced in zip(original.operations, measured.operations, strict=True))
    rebased = replace(original, operations=operations, assumptions=measured.assumptions,
                      timing_constraints=measured.timing_constraints)
    if host_dispatch:
        reference = dict(path=ACTIVITY_BINDING, **evidence.record['activity_binding'])
        if not evidence.emitted_activities:
            raise ValueError('host dispatch ordering requires a replayed native activity binding')
        constraints = _queue_order(evidence.emitted_activities, reference, host_dispatch=True)
        rebased = replace(rebased, timing_constraints=constraints,
                          assumptions=(*rebased.assumptions, 'SYNCHRONOUS_HOST_CALLS_GATE_LATER_GPU_DISPATCH'))
        schedule = construct_implementations(rebased, baseline_only=True)
        if schedule['status'] != 'FEASIBLE_WITHIN_MODEL':
            return None
    else:
        # Legacy point estimates retain the archived diagnostic time table.
        schedule = dict(evidence.repriced_schedule, problem_sha256=digest(rebased))
    schedule.pop('elapsed_seconds', None)
    verify_execution_plan(rebased, schedule)
    return problem_document(rebased), schedule, built[2]


# Bind native CPU body calibration to its exact pageable placement.
class CpuPlacementCostUnavailable(ValueError):
    """The matching native project contains no calibration for this profile."""


def _serial_modes(problem, expected):
    if ([row.id for row in problem.operations] != expected or problem.activities or problem.timing_constraints
            or any(edge.transfers for edge in problem.data_edges)):
        raise ValueError('CPU timing requires the complete serial body without extra execution or copies')
    modes = {}
    for index, row in enumerate(problem.operations):
        if (len(row.options) != 1 or row.baseline_option != row.options[0].id
                or row.predecessors != tuple(expected[max(0, index - 1):index])
                or row.options[0].workspace_bytes or row.options[0].unresolved_resources):
            raise ValueError('CPU timing requires the exact source-order execution')
        modes[row.id] = row.options[0]
    end = modes.pop(_RETURN)
    if (end.lowering != 'REGION_VALUE_HANDOFF' or end.parameters or end.output_locations
            or end.demands or end.cost.estimate_ns != 0):
        raise ValueError('CPU operator timing excludes file output publication')
    return modes, end


def _edges(problem, projection, memory):
    sizes = {row['id']: row['nbytes'] for row in [*projection['inputs'], *projection['operations']]}
    produced = {row['id'] for row in projection['operations']}
    ports = {(arg['value'], row['id']) for row in projection['operations']
             for arg in row['arguments'] if 'value' in arg}
    ports.update((name, _RETURN) for name in projection['outputs'])
    edges = {(edge.value, edge.consumer): edge for edge in problem.data_edges}
    if (len(edges) != len(problem.data_edges) or edges.keys() != ports
            or any((edge.nbytes, edge.producer, edge.external_location) !=
                   (sizes[name], name if name in produced else None, None if name in produced else memory)
                   for (name, _), edge in edges.items())):
        raise ValueError('CPU timing requires the same pageable dataflow and file ports')
    return edges


def _placement_modes(problem, projection, order):
    if type(problem) is dict:
        problem = problem_from_dict(problem)
    else:
        validate_problem(problem)
    resources = {row.id: (row.kind, row.capacity) for row in problem.resources}
    if (problem.source_refs != (projection['program'],) or resources.keys() != {_PAGEABLE_HOST, HOST_RESOURCE}
            or resources[HOST_RESOURCE] != ('EXECUTION', 1) or resources[_PAGEABLE_HOST][0] != 'MEMORY'
            or problem.reserved_memory != {_PAGEABLE_HOST: sum(row['nbytes'] for row in projection['inputs'])}):
        raise ValueError('CPU timing is scoped to the exact all-pageable CPU placement')
    modes, end = _serial_modes(problem, order)
    edges = _edges(problem, projection, _PAGEABLE_HOST)
    shapes = {row['id']: row['shape'] for row in [*projection['inputs'], *projection['operations']]}
    for row in projection['operations']:
        name = row['id']
        arguments = [dict(edge=edges[arg['value'], name].id, shape=shapes[arg['value']])
                     if 'value' in arg else dict(scalar=arg['scalar']) for arg in row['arguments']]
        operands = [arg['value'] for arg in row['arguments'] if 'value' in arg]
        option = modes[name]
        if (option.lowering != 'NATIVE_ATEN_CALL' or option.demands != {HOST_RESOURCE: 1}
                or option.parameters != dict(schema=row['schema'], arguments=arguments, shape=row['shape'],
                                             completion='HOST_SYNCHRONOUS')
                or option.input_locations != dict.fromkeys(operands, _PAGEABLE_HOST)
                or option.output_locations != {name: _PAGEABLE_HOST}):
            raise ValueError('CPU timing does not apply to a different operator ABI, layout or host out variant')
    if end.input_locations != dict.fromkeys(projection['outputs'], _PAGEABLE_HOST):
        raise ValueError('CPU timing requires the same ordered file output values')


def _native_costs(document, schedule, recipe, execution, projection, order, observation, evidence):
    scheduled = scheduled_tensor_body(document, schedule, recipe)
    if (scheduled['operations'] != recipe['operations']
            or scheduled['release_after'] != execution['storage']['release_after']):
        raise ValueError('native CPU cost schedule changes the emitted operation order or lifetimes')
    problem = problem_from_dict(document)
    modes, end = _serial_modes(problem, order)
    memory, = [row.id for row in problem.resources if row.kind == 'MEMORY']
    if (problem.source_refs != (execution['program_graph'],) or encode(problem.target) != execution['target']
            or memory != 'host-tensor-storage-budget'):
        raise ValueError('native CPU cost schedule differs from the bound source, target or host storage')
    _edges(problem, projection, memory)
    demands = {row.id: 1 for row in problem.resources if row.kind == 'EXECUTION'}
    for row in projection['operations']:
        name = row['id']
        mode = modes[name]
        operands = [arg['value'] for arg in row['arguments'] if 'value' in arg]
        if (mode.lowering != 'NATIVE_ATEN_CALL' or not demands or mode.demands != demands
                or mode.parameters != dict(cpu_ids=execution['cpu_ids'], completion='SYNCHRONOUS_RETURN')
                or mode.input_locations != dict.fromkeys(operands, memory) or mode.output_locations != {name: memory}):
            raise ValueError('native CPU cost schedule must use serial functional ATen calls on one CPU')
    if end.input_locations != dict.fromkeys(projection['outputs'], memory):
        raise ValueError('native CPU cost schedule return differs from the measured outputs')
    replay_operation_costs(problem, recipe, observation, evidence)
    return operation_costs(observation, recipe['operations'], evidence)


def admit_pageable_cpu_costs(root, *, cpu_manifest_sha256, cpu_profile, projection, problem, inputs, references):
    """Replay operator costs from an already verified CPU project and ordered case receipts."""
    root = Path(root).resolve(strict=True)
    manifest = _manifest(root, cpu_manifest_sha256, CPU_SCHEMA)

    def record(name):
        return json.loads(_member(root, manifest, name))

    def reference(name):
        return dict(path=name, **_identity(manifest['files'][name]))

    family = manifest.get('profiles')
    if (family is None and cpu_profile is not None) or (family is not None and cpu_profile not in family):
        raise ValueError('CPU timing requires exactly one published shape specialization')
    base = 'plan' if cpu_profile is None else f'plan/profiles/{cpu_profile}'
    execution = record(f'{base}/execution.json')
    _, _, execution = _selection(root, manifest, cpu_profile,
        dict(symbolic_profile=execution['symbolic_profile'], program=projection['program']))
    cpus = execution['cpu_ids']
    if (len(cpus) != 1 or type(cpus[0]) is not int or execution['schedule'] != 'SOURCE_DATA_ORDER'
            or execution.get('placement') is not None):
        raise ValueError('CPU timing requires one core and the native serial path')
    if manifest['selected'] == 'CPU_PROFILE_PLACEMENTS' and any(
            record(f'plan/profiles/{label}/execution.json').get('placement') is not None for label in family):
        raise ValueError('CPU timing was collected before the shape family executable was rebuilt for ready placement')
    recipe = record(f'{base}/recipe.json')
    graph = codec.loads(_member(root, manifest, execution['program_graph']['path']).decode())
    qualified = qualify_hybrid_export(recipe, graph=graph, program_ref=projection['program'],
        input_kinds={row['id']: row['kind'] for row in execution['inputs']}, allow_cuda_only=True)
    if qualified != projection:
        raise ValueError('native CPU timing differs from the requested typed operations or representation')
    order = [*(row['id'] for row in projection['operations']), _RETURN]
    _placement_modes(problem, projection, order)

    ports = _ports(execution['inputs'])
    supplied, outputs = list(map(_identity, inputs)), list(map(_identity, references))
    users = [row for row in ports if row['kind'] == 'INPUT']
    sizes = {row['id']: row['nbytes'] for row in projection['operations']}
    if ([row['size_bytes'] for row in supplied] != [row['size_bytes'] for row in users]
            or [row['size_bytes'] for row in outputs] != [sizes[name] for name in projection['outputs']]):
        raise ValueError('declared CPU case does not match the ordered input and output byte extents')
    calibration = 'validation/calibration.json' if cpu_profile is None else f'{base}/calibration.json'
    schedule = f'{base}/schedule.json'
    present = manifest['files'].keys() & {calibration, schedule}
    if not present and execution['timing_basis'] == 'UNMEASURED_MODEL_ENTRY':
        raise CpuPlacementCostUnavailable('the exact native CPU profile has no operator calibration')
    if present != {calibration, schedule} or execution['timing_basis'] != 'MEASURED_CPU_BODY_ONE_FIXED_SHAPE':
        raise ValueError('CPU timing requires a locked calibration and the matching serial schedule')
    observation = record(calibration)
    if (observation['schema'] != 'scar.tensor-body-observation' or observation['schema_version'] != 1
            or observation['repeats'] != 1 or observation['discarded_warmups'] != 1
            or observation['cpu_ids'] != cpus or list(map(_identity, observation['inputs'])) != supplied
            or list(map(_identity, observation['outputs'])) != outputs):
        raise ValueError('native CPU calibration input or output identities differ from the declared case')
    physical = f'{base}/problem.json'
    evidence = reference(calibration)
    costs = _native_costs(record(physical), record(schedule), recipe, execution, projection, order,
                          observation, evidence)
    treatment = dict(schema='scar.native-export-pageable-cpu-cost-treatment', schema_version=1,
        package=_package(manifest, cpu_manifest_sha256), profile=cpu_profile,
        source=_identity(manifest['source']), program=_identity(projection['program']),
        symbolic_profile=execution['symbolic_profile'], cpu_ids=cpus, target=execution['target'],
        binding=reference(f'{base}/execution.json'), dependencies=reference('plan/dependencies.json'),
        lowering='NATIVE_ATEN_CALL', memory=_PAGEABLE_HOST, file_abi='ORDERED_CONTIGUOUS_F32',
        byte_order=sys.byteorder, input_ports=ports, inputs=supplied, outputs=outputs,
        calibration=evidence, physical_problem=reference(physical), physical_schedule=reference(schedule),
        timing_scope='SERIAL_OPERATOR_SERVICE_EXCLUDES_SETUP_AND_FILE_IO',
        warmups=observation['discarded_warmups'], samples=len(observation['samples']))
    key = query_key(CostQuery('EXACT_PAGEABLE_CPU_EXPORT_TREATMENT', treatment))
    treatment['cost_source'] = key
    priced = {}
    for row in projection['operations']:
        name = row['id']
        cost = costs[name]
        context = CostQuery('EXACT_PAGEABLE_CPU_EXPORT_OPERATOR_SERVICE', dict(treatment=key, operation=row['operation']))
        priced[name] = replace(cost, context=query_key(context), source_refs=(*cost.source_refs, dict(treatment=key)))
    return priced, treatment


# Build and replay one source-qualified CUDA placement for calibration.
PLACEMENT_TREATMENT_SCHEMA = 'scar.native-export-placement-treatment'


def plan_device_treatment(projection, model, *, model_ref, ordinal):
    """Form the physical plan for one requested non-CPU placement ordinal."""
    choices = placement_choices(projection)
    placement = placement_assignment(choices, ordinal)
    if 'CUDA' not in placement.values():
        raise ValueError('a calibration treatment requires a CUDA placement ordinal')
    problem, schedule, bindings = plan_placed_hybrid_export(projection, model, model_ref=model_ref,
                                                            placement=placement)
    schedule.pop('elapsed_seconds', None)
    report = placement_proposal(projection, choices, None, dict(placement=placement))
    del report['analysis']
    report.update(schema=PLACEMENT_TREATMENT_SCHEMA, ordinal=ordinal,
                  program=projection['program'], declared_model=model_ref,
                  problem_sha256=schedule['problem_sha256'],
                  decision='EXPLICIT_CALIBRATION_TREATMENT', target_program_executed=False,
                  execution_authorized=False, model_optimality_proven=False, physical_optimality_proven=False)
    return problem, schedule, bindings, report


def replay_device_treatment(projection, model, *, model_ref, problem, schedule, bindings, placement_treatment):
    """Rebuild the selected source assignment and verify its archived certificate."""
    expected = plan_device_treatment(projection, model, model_ref=model_ref,
                                     ordinal=placement_treatment.get('ordinal'))
    stable_schedule = {key: value for key, value in schedule.items() if key != 'elapsed_seconds'}
    if canonical_json(expected) != canonical_json((problem, stable_schedule, bindings, placement_treatment)):
        raise ValueError('CUDA calibration treatment differs from its replayed source placement or schedule')

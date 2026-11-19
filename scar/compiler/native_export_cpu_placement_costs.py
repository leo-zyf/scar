"""Bind native CPU body calibration to its exact pageable placement."""

from dataclasses import replace
import json
from pathlib import Path
import sys

from scar.compiler.aten_export_hybrid_plan import HOST_RESOURCE, _PAGEABLE_HOST, _RETURN
from scar.compiler.aten_export_hybrid_source import qualify_hybrid_export
from scar.compiler.native_device_pair import CPU_SCHEMA, _identity, _manifest, _member, _package, _ports, _selection
from scar.compiler.tensor_execution import operation_costs, replay_operation_costs, scheduled_tensor_body
from scar.ir.implementation_plan import problem_from_dict, validate_problem
from scar.ir.program import codec
from scar.ir.record_codec import encode
from scar.optimization.implementation_costs import CostQuery, query_key


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

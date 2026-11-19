"""Plan a source-qualified CPU and CUDA graph under an explicit offline model."""

from collections import defaultdict

from scar.compiler.aten_export_cuda_plan import (
    _COMPUTE, _DEVICE, _DOWNLOAD, _GPU, _HOST, _RETURN, _UPLOAD, _matrix, _profile_target,
)
from scar.compiler.aten_export_hybrid_source import CPU_SCHEMAS, SCHEMA as SOURCE_SCHEMA
from scar.compiler.aten_native_recipe import SCHEMA as RECIPE_SCHEMA, SCHEMA_VERSION as RECIPE_VERSION
from scar.ir.implementation_plan import (
    CostEstimate, ImplementationOption, PlanningDataEdge, PlanningOperation, PlanningProblem, PlanningResource,
    TransferOption, problem_document, problem_from_dict,
)
from scar.optimization.aten_export_rewrite import FUNCTIONAL_EXPORT_SCHEMAS
from scar.optimization.aten_tensor_lifetime import tensor_footprint
from scar.optimization.constructive_planner import construct_implementations
from scar.optimization.joint_planner import verify_execution_plan
from scar.optimization.material_routes import expand_shared_material_route


OFFLINE_PROFILE = 'scar.aten-export-hybrid-offline-profile'
PLACEMENT_PROFILE = 'scar.aten-export-placement-offline-profile'
HOST_RESOURCE = 'cpu:aten'
HOST_LOWERING = 'LIBTORCH_ATEN_CPU_OUT_F32_V1'
_PAGEABLE_HOST = 'cpu:pageable-host'
_MM = FUNCTIONAL_EXPORT_SCHEMAS['aten.mm.default']
_GEMM = 'CUBLAS_SGEMM_F32_COLUMN_MAJOR_V1'


def _operator_mode(row, definitions, edge_ids, origin, profile_ref, model, cost, *, placement, host):
    shape = _matrix(row)
    proof = (*origin, profile_ref)
    schema = row['schema']
    if schema == _MM:
        left, right = row['left'], row['right']
        if left == right or row['arguments'] != [{'value': left}, {'value': right}]:
            raise ValueError('qualified MM requires two distinct ordered input matrices')
        rows, reduction = definitions[left][1]
        other_reduction, columns = definitions[right][1]
        if other_reduction != reduction or shape != [rows, columns] or row['reduction'] != reduction:
            raise ValueError('MM dimensions differ from the hybrid source ports')
    if placement == 'CUDA' and schema == _MM:
        # Row-major A and B share their bytes with column-major A^T and B^T.
        return ImplementationOption('sgemm', _GEMM, proof, {_COMPUTE: 1, _GPU: 1},
            {_DEVICE: model['workspace_bytes']}, dict.fromkeys(edge_ids, _DEVICE), {row['result']: _DEVICE},
            cost('sgemm'), dict(left_edge=edge_ids[right], right_edge=edge_ids[left], rows=columns, columns=rows,
                                reduction=reduction, completion='SAME_STREAM'))
    if placement != 'CPU' or (schema not in CPU_SCHEMAS and schema != _MM):
        raise ValueError('hybrid source operator is outside the qualified CPU and CUDA placement contracts')
    operands = [dict(edge=edge_ids[arg['value']], shape=definitions[arg['value']][1]) if 'value' in arg
                else dict(scalar=arg['scalar']) for arg in row['arguments']]
    lowering = HOST_LOWERING if host == _HOST else 'NATIVE_ATEN_CALL'
    return ImplementationOption('host', lowering, proof, {HOST_RESOURCE: 1}, {},
        dict.fromkeys(edge_ids, host), {row['result']: host}, cost('cpu_mm' if schema == _MM else 'cpu_aten'),
        dict(schema=schema, arguments=operands, shape=shape, completion='HOST_SYNCHRONOUS'))


def plan_hybrid_export(projection, model, *, model_ref):
    """Insert shared host/device crossings and plan both providers in one graph.

    The model supplies synthetic costs for offline C++ generation. The
    resulting certificate leaves device performance and deployment unproven.
    """
    target = _profile_target(model, model_ref, schema=OFFLINE_PROFILE, extra_durations=('cpu_aten',))
    return _certify_placement(*_build_placement(projection, model, model_ref, target, None))


def plan_placed_hybrid_export(projection, model, *, model_ref, placement):
    """Lower an explicitly selected typed placement to a fixed physical problem."""
    return _certify_placement(*_explicit_placement(projection, model, model_ref, placement))


def build_placed_hybrid_problem(projection, model, *, model_ref, placement):
    """Return placements and material routes before any timing-based scheduling."""
    problem, bindings = _explicit_placement(projection, model, model_ref, placement)
    return problem_document(problem), None, bindings


def _explicit_placement(projection, model, model_ref, placement):
    host_bytes = model.get('host_memory_bytes') if type(model) is dict else None
    if type(host_bytes) is not int or not 0 < host_bytes < 2**63:
        raise ValueError('the placement profile requires bounded ordinary host memory')
    target = _profile_target({key: value for key, value in model.items() if key != 'host_memory_bytes'},
                             model_ref, schema=PLACEMENT_PROFILE, extra_durations=('cpu_aten', 'cpu_mm'))
    if type(placement) is not dict:
        raise ValueError('an explicit typed CPU/CUDA placement is required')
    return _build_placement(projection, model, model_ref, target, placement)


def _certify_placement(problem, bindings):
    schedule = construct_implementations(problem)
    verify_execution_plan(problem, schedule)
    return problem_document(problem), schedule, bindings


def _build_placement(projection, model, model_ref, target, placement):
    if (projection.get('schema') != SOURCE_SCHEMA or type(projection.get('schema_version')) is not int
            or projection['schema_version'] != 1 or projection.get('layout') != 'CONTIGUOUS_ROW_MAJOR'
            or projection.get('dtype') != 'FLOAT32'
            or projection.get('ownership') != 'READ_ONLY_ENTRY_VALUES_AND_FRESH_RESULTS'
            or not projection['operations'] or not projection['outputs'] or not projection['inputs']):
        raise ValueError('hybrid lowering needs a source-qualified functional matrix graph')
    explicit = placement is not None
    native = {row['id']: row['placement'] for row in projection['operations']}
    if explicit:
        if (placement.keys() != native.keys() or any(value not in ('CPU', 'CUDA') for value in placement.values())
                or any(row['schema'] != _MM and placement[row['id']] != 'CPU' for row in projection['operations'])):
            raise ValueError('only qualified MM nodes have a selectable CPU/CUDA placement')
        if any(row['schema'] == _MM and row['placement'] != 'CUDA' for row in projection['operations']):
            raise ValueError('the original hybrid source projection must retain the qualified MM modes')
    elif set(native.values()) != {'CPU', 'CUDA'} or any(row['placement'] == 'CPU' and row['schema'] == _MM
                                                       for row in projection['operations']):
        raise ValueError('hybrid placement requires both CPU and CUDA operations and an explicit CPU-MM profile')
    choices = placement if explicit else native
    cuda = 'CUDA' in choices.values()
    host = _HOST if cuda else _PAGEABLE_HOST
    program = projection['program']
    qualification = 'SYNTHETIC_PLACEMENT_OFFLINE' if explicit else 'SYNTHETIC_HYBRID_OFFLINE'
    profile_ref = dict(model_ref, qualification=qualification + '_PROFILE')
    pool_rows = (
        (_HOST, 'MEMORY', model['pinned_host_bytes']), (_DEVICE, 'MEMORY', model['device_bytes']),
        (_UPLOAD, 'CHANNEL', 1), (_COMPUTE, 'CHANNEL', 1), (_DOWNLOAD, 'CHANNEL', 1),
        (_GPU, 'EXECUTION', 1)) if cuda else ((_PAGEABLE_HOST, 'MEMORY', model['host_memory_bytes']),)
    resources = tuple(PlanningResource(name, kind, capacity, (profile_ref,)) for name, kind, capacity in (
        *pool_rows, (HOST_RESOURCE, 'EXECUTION', 1)))

    def cost(key):
        context = qualification + '_' + key.upper()
        if explicit and key in ('cpu_mm', 'cpu_aten'):
            variant = 'PINNED_OUT' if cuda else 'PAGEABLE_CALL'
            context = f'{context}_{variant}'
        return CostEstimate(model['model_durations_ns'][key], None, None, 'DECLARED_MODEL',
                            context, (profile_ref,))

    inputs = [row['id'] for row in projection['inputs']]
    definitions = {row['id']: (None, _matrix(row), row['nbytes'], host) for row in projection['inputs']}
    if len(definitions) != len(inputs):
        raise ValueError('hybrid placement requires unique inputs')
    operations, edges, demand, crossings = [], [], set(), defaultdict(list)

    def edge(name, consumer, edge_id, destination, origin):
        producer, _, nbytes, location = definitions[name]
        demand.add(name)
        if location != destination:
            crossings[name, destination].append(edge_id)
        edges.append(PlanningDataEdge(edge_id, name, producer, consumer, nbytes,
                                      location if producer is None else None, source_refs=origin))

    for row in projection['operations']:
        operation, result = row['id'], row['result']
        names = dict.fromkeys(arg['value'] for arg in row['arguments'] if 'value' in arg)
        if result in definitions or operation == _RETURN or not names or names.keys() - definitions.keys():
            raise ValueError('hybrid operation requires defined operands and a fresh result')
        origin = (dict(program, operation=row['operation']),)
        ports = {name: f'{operation}:input:{index}' for index, name in enumerate(names)}
        mode = _operator_mode(row, definitions, ports, origin, profile_ref, model, cost,
                              placement=choices[operation], host=host)
        for name, edge_id in ports.items():
            edge(name, operation, edge_id, mode.input_locations[name], origin)
        predecessors = (operations[-1].id,) if not cuda and operations else ()
        operations.append(PlanningOperation(operation, origin, predecessors, (mode,), mode.id))
        definitions[result] = (operation, row['shape'], row['nbytes'], mode.output_locations[result])

    outputs = tuple(dict.fromkeys(projection['outputs']))
    for name in outputs:
        if name not in definitions or definitions[name][0] is None:
            raise ValueError('hybrid entry returns only produced matrix values')
        edge(name, _RETURN, f'{_RETURN}:{name}', host, (program,))
    terminal = ImplementationOption('return', 'CUDA_COPY_RETURN' if cuda else 'REGION_VALUE_HANDOFF',
                                    (program,), {}, {}, dict.fromkeys(outputs, host),
                                    {}, CostEstimate(0, 0, 0, 'DECLARED_MODEL', 'LOGICAL_ENDPOINT'), {})
    predecessors = () if cuda else (operations[-1].id,)
    operations.append(PlanningOperation(_RETURN, (program,), predecessors, (terminal,), terminal.id))
    if demand != definitions.keys():
        raise ValueError('hybrid file ABI requires each input and produced value to have a consumer')

    if not cuda:
        return _plan_pageable_host(projection, model, target, resources, operations, edges, inputs)

    # The executor retains each buffer and the cuBLAS workspace until return.
    backing = {_HOST: 0, _DEVICE: model['workspace_bytes']}
    for _, _, nbytes, location in definitions.values():
        backing[location] += nbytes
    for name, destination in crossings:
        backing[destination] += definitions[name][2]
    if backing[_HOST] > model['pinned_host_bytes'] or backing[_DEVICE] > model['device_bytes']:
        if explicit:
            from scar.optimization.native_device_placement_search import PlacementCapacityExceeded

            raise PlacementCapacityExceeded('hybrid file ABI backing and workspace exceed the declared capacity')
        raise ValueError('hybrid file ABI backing and workspace exceed the declared capacity')
    problem = PlanningProblem(target, (program,), resources, tuple(operations), tuple(edges), assumptions=(
        'OFFLINE_SYNTHETIC_ONLY', 'ACTUAL_DEVICE_TIMING_UNKNOWN', 'NORMAL_COMPLETION_FP32_MATRIX_GRAPH',
        'SINGLE_SYNCHRONOUS_HOST_ATEN_LANE', 'ROW_MAJOR_STORAGE_VIA_TRANSPOSED_COLUMN_MAJOR_SGEMM',
        'ONE_COPY_PER_LOGICAL_VALUE_AND_DESTINATION', 'ALL_FILE_ABI_BUFFERS_AND_PROVIDER_WORKSPACE_RETAINED_THROUGH_RETURN'))
    upload = TransferOption('h2d', _HOST, _DEVICE, {_UPLOAD: 1}, cost('h2d'), 'CUDA_MEMCPY_ASYNC', (profile_ref,))
    download = TransferOption('d2h', _DEVICE, _HOST, {_DOWNLOAD: 1}, cost('d2h'), 'CUDA_MEMCPY_ASYNC', (profile_ref,))
    for (_, destination), edge_ids in crossings.items():
        problem = expand_shared_material_route(problem, edge_ids, (upload if destination == _DEVICE else download,))
    published = {row.id: row.value for row in problem.data_edges if row.consumer == _RETURN}
    bindings = dict(device=model['device'], memory_kinds={_HOST: 'PinnedHost', _DEVICE: 'Device'},
        stream_channels=[_UPLOAD, _COMPUTE, _DOWNLOAD], compute_resource=_GPU, host_resource=HOST_RESOURCE,
        input_values=inputs, output_values=[published[f'{_RETURN}:{name}'] for name in projection['outputs']])
    return problem, bindings


def _plan_pageable_host(projection, model, target, resources, operations, edges, inputs):
    from scar.optimization.native_device_placement_search import PlacementCapacityExceeded

    budget = model['host_memory_bytes']
    footprint = tensor_footprint(dict(projection, schema=RECIPE_SCHEMA, schema_version=RECIPE_VERSION),
        input_bytes=[item['nbytes'] for item in projection['inputs']],
        output_bytes=[item['nbytes'] for item in projection['operations']], candidates=())
    if footprint['serial_peak_bytes'] > budget:
        raise PlacementCapacityExceeded('pageable host Tensor backing exceeds the declared capacity')
    problem = PlanningProblem(target, (projection['program'],), resources, tuple(operations), tuple(edges), assumptions=(
        'OFFLINE_SYNTHETIC_ONLY', 'ACTUAL_DEVICE_TIMING_UNKNOWN', 'NORMAL_COMPLETION_FP32_MATRIX_GRAPH',
        'SINGLE_SYNCHRONOUS_HOST_ATEN_LANE', 'PAGEABLE_HOST_SOURCE_ORDER_ATEN_CALLS',
        'HOST_TENSOR_BACKING_EXCLUDES_PROVIDER_WORKSPACE'),
        reserved_memory={_PAGEABLE_HOST: footprint['persistent_input_bytes']})
    return problem, dict(deployment='CPU', host_resource=HOST_RESOURCE, input_values=inputs,
                         output_values=list(projection['outputs']))


def replay_hybrid_export(projection, model, *, model_ref, problem, schedule, bindings):
    """Regenerate the placement and verify its archived physical timeline."""
    expected_problem, expected_schedule, expected_bindings = plan_hybrid_export(projection, model, model_ref=model_ref)
    if (problem != expected_problem or bindings != expected_bindings
            or {key: value for key, value in schedule.items() if key != 'elapsed_seconds'} !=
            {key: value for key, value in expected_schedule.items() if key != 'elapsed_seconds'}):
        raise ValueError('archived hybrid plan or file material bindings differ from the source replay')
    return verify_execution_plan(problem_from_dict(problem), schedule)

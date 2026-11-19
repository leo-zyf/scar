"""Plan a source-qualified CPU and CUDA graph under an explicit offline model."""

from collections import defaultdict

from scar.compiler.aten_export_cuda_plan import (
    _COMPUTE, _DEVICE, _DOWNLOAD, _GPU, _HOST, _RETURN, _UPLOAD, _matrix, _profile_target,
)
from scar.compiler.aten_export_hybrid_source import CPU_SCHEMAS, SCHEMA as SOURCE_SCHEMA
from scar.ir.implementation_plan import (
    CostEstimate, ImplementationOption, PlanningDataEdge, PlanningOperation, PlanningProblem, PlanningResource,
    TransferOption, problem_document, problem_from_dict,
)
from scar.optimization.aten_export_rewrite import FUNCTIONAL_EXPORT_SCHEMAS
from scar.optimization.constructive_planner import construct_implementations
from scar.optimization.joint_planner import verify_execution_plan
from scar.optimization.material_routes import expand_shared_material_route


OFFLINE_PROFILE = 'scar.aten-export-hybrid-offline-profile'
HOST_RESOURCE = 'cpu:aten'
HOST_LOWERING = 'LIBTORCH_ATEN_CPU_OUT_F32_V1'
_MM = FUNCTIONAL_EXPORT_SCHEMAS['aten.mm.default']
_GEMM = 'CUBLAS_SGEMM_F32_COLUMN_MAJOR_V1'


def _operator_mode(row, definitions, edge_ids, origin, profile_ref, model, cost):
    shape = _matrix(row)
    proof = (*origin, profile_ref)
    schema = row['schema']
    if row['placement'] == 'CUDA' and schema == _MM:
        left, right = row['left'], row['right']
        if left == right or row['arguments'] != [{'value': left}, {'value': right}]:
            raise ValueError('CUDA GEMM requires two distinct ordered input matrices')
        rows, reduction = definitions[left][1]
        other_reduction, columns = definitions[right][1]
        if other_reduction != reduction or shape != [rows, columns] or row['reduction'] != reduction:
            raise ValueError('CUDA GEMM dimensions differ from the hybrid source ports')
        # Row-major A and B share their bytes with column-major A^T and B^T.
        return ImplementationOption('sgemm', _GEMM, proof, {_COMPUTE: 1, _GPU: 1},
            {_DEVICE: model['workspace_bytes']}, dict.fromkeys(edge_ids, _DEVICE), {row['result']: _DEVICE},
            cost('sgemm'), dict(left_edge=edge_ids[right], right_edge=edge_ids[left], rows=columns, columns=rows,
                                reduction=reduction, completion='SAME_STREAM'))
    if row['placement'] != 'CPU' or schema not in CPU_SCHEMAS:
        raise ValueError('hybrid source operator is outside the qualified CPU and CUDA placement contracts')
    operands = [dict(edge=edge_ids[arg['value']], shape=definitions[arg['value']][1]) if 'value' in arg
                else dict(scalar=arg['scalar']) for arg in row['arguments']]
    return ImplementationOption('host', HOST_LOWERING, proof, {HOST_RESOURCE: 1}, {},
        dict.fromkeys(edge_ids, _HOST), {row['result']: _HOST}, cost('cpu_aten'),
        dict(schema=schema, arguments=operands, shape=shape, completion='HOST_SYNCHRONOUS'))


def plan_hybrid_export(projection, model, *, model_ref):
    """Insert shared host/device crossings and plan both providers in one graph.

    The model supplies synthetic costs for offline C++ generation. The
    resulting certificate leaves device performance and deployment unproven.
    """
    target = _profile_target(model, model_ref, schema=OFFLINE_PROFILE, extra_durations=('cpu_aten',))
    if (projection.get('schema') != SOURCE_SCHEMA or type(projection.get('schema_version')) is not int
            or projection['schema_version'] != 1 or projection.get('layout') != 'CONTIGUOUS_ROW_MAJOR'
            or projection.get('dtype') != 'FLOAT32'
            or projection.get('ownership') != 'READ_ONLY_ENTRY_VALUES_AND_FRESH_RESULTS'
            or not projection['operations'] or not projection['outputs'] or not projection['inputs']):
        raise ValueError('hybrid lowering needs a source-qualified functional matrix graph')
    program = projection['program']
    profile_ref = dict(model_ref, qualification='SYNTHETIC_HYBRID_OFFLINE_PROFILE')
    resources = tuple(PlanningResource(name, kind, capacity, (profile_ref,)) for name, kind, capacity in (
        (_HOST, 'MEMORY', model['pinned_host_bytes']), (_DEVICE, 'MEMORY', model['device_bytes']),
        (_UPLOAD, 'CHANNEL', 1), (_COMPUTE, 'CHANNEL', 1), (_DOWNLOAD, 'CHANNEL', 1),
        (_GPU, 'EXECUTION', 1), (HOST_RESOURCE, 'EXECUTION', 1)))

    def cost(key):
        return CostEstimate(model['model_durations_ns'][key], None, None, 'DECLARED_MODEL',
                            'SYNTHETIC_HYBRID_OFFLINE_' + key.upper(), (profile_ref,))

    inputs = [row['id'] for row in projection['inputs']]
    definitions = {row['id']: (None, _matrix(row), row['nbytes'], _HOST) for row in projection['inputs']}
    if len(definitions) != len(inputs) or {row['placement'] for row in projection['operations']} != {'CPU', 'CUDA'}:
        raise ValueError('hybrid placement requires unique inputs and both CPU and CUDA operations')
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
        mode = _operator_mode(row, definitions, ports, origin, profile_ref, model, cost)
        for name, edge_id in ports.items():
            edge(name, operation, edge_id, mode.input_locations[name], origin)
        operations.append(PlanningOperation(operation, origin, (), (mode,), mode.id))
        definitions[result] = (operation, row['shape'], row['nbytes'], mode.output_locations[result])

    outputs = tuple(dict.fromkeys(projection['outputs']))
    for name in outputs:
        if name not in definitions or definitions[name][0] is None:
            raise ValueError('hybrid entry returns only produced matrix values')
        edge(name, _RETURN, f'{_RETURN}:{name}', _HOST, (program,))
    terminal = ImplementationOption('return', 'CUDA_COPY_RETURN', (program,), {}, {}, dict.fromkeys(outputs, _HOST),
                                    {}, CostEstimate(0, 0, 0, 'DECLARED_MODEL', 'LOGICAL_ENDPOINT'), {})
    operations.append(PlanningOperation(_RETURN, (program,), (), (terminal,), terminal.id))
    if demand != definitions.keys():
        raise ValueError('hybrid file ABI requires each input and produced value to have a consumer')

    # The executor retains each buffer and the cuBLAS workspace until return.
    backing = {_HOST: 0, _DEVICE: model['workspace_bytes']}
    for _, _, nbytes, location in definitions.values():
        backing[location] += nbytes
    for name, destination in crossings:
        backing[destination] += definitions[name][2]
    if backing[_HOST] > model['pinned_host_bytes'] or backing[_DEVICE] > model['device_bytes']:
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
    schedule = construct_implementations(problem)
    verify_execution_plan(problem, schedule)
    return problem_document(problem), schedule, bindings


def replay_hybrid_export(projection, model, *, model_ref, problem, schedule, bindings):
    """Regenerate the placement and verify its archived physical timeline."""
    expected_problem, expected_schedule, expected_bindings = plan_hybrid_export(projection, model, model_ref=model_ref)
    if (problem != expected_problem or bindings != expected_bindings
            or {key: value for key, value in schedule.items() if key != 'elapsed_seconds'} !=
            {key: value for key, value in expected_schedule.items() if key != 'elapsed_seconds'}):
        raise ValueError('archived hybrid plan or file material bindings differ from the source replay')
    return verify_execution_plan(problem_from_dict(problem), schedule)

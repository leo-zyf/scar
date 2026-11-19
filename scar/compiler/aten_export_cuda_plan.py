"""Plan a source-qualified FP32 matrix graph under a synthetic CUDA model."""

from collections import defaultdict
from pathlib import PurePosixPath

from scar.compiler.aten_export_cuda_source import SCHEMA as SOURCE_SCHEMA
from scar.ir.implementation_plan import (
    CostEstimate, ImplementationOption, PlanningDataEdge, PlanningOperation,
    PlanningProblem, PlanningResource, TransferOption, problem_document, problem_from_dict,
)
from scar.ir.program.providers import ProviderArtifact
from scar.optimization.constructive_planner import construct_implementations
from scar.optimization.joint_planner import verify_execution_plan
from scar.optimization.material_routes import expand_material_route, expand_shared_material_route


OFFLINE_PROFILE = 'scar.aten-export-cuda-offline-profile'
_HOST, _DEVICE = 'cuda:pinned-host', 'cuda:device-memory'
_UPLOAD, _COMPUTE, _DOWNLOAD = 'cuda:copy-in', 'cuda:compute-stream', 'cuda:copy-out'
_GPU = 'cuda:compute'
_RETURN = '$return'


def _profile_target(model, reference, *, schema=OFFLINE_PROFILE, extra_durations=()):
    keys = {'schema', 'schema_version', 'device', 'pinned_host_bytes', 'device_bytes',
            'workspace_bytes', 'model_durations_ns'}
    if (type(model) is not dict or model.keys() != keys or model['schema'] != schema
            or type(model['schema_version']) is not int or model['schema_version'] != 1):
        raise ValueError('an explicit synthetic CUDA offline profile is required')
    amounts = [model[key] for key in ('pinned_host_bytes', 'device_bytes', 'workspace_bytes')]
    durations = model['model_durations_ns']
    if (type(model['device']) is not int or not 0 <= model['device'] < 2**31
            or any(type(size) is not int or not 0 < size < 2**63 for size in amounts) or amounts[-1] < 16384
            or type(durations) is not dict or set(durations) != {'h2d', 'sgemm', 'd2h', *extra_durations}
            or any(type(ns) is not int or not 0 < ns < 2**60 for ns in durations.values())):
        raise ValueError('invalid CUDA offline device, storage, workspace or synthetic duration')
    if type(reference) is not dict or set(reference) != {'path', 'sha256', 'size_bytes'}:
        raise ValueError('the CUDA offline profile must reference its archived JSON file')
    path = PurePosixPath(reference['path'])
    if path.is_absolute() or not path.parts or '..' in path.parts:
        raise ValueError('the CUDA offline profile path must be relative to its project')
    return ProviderArtifact('declared:' + path.as_posix(), reference['sha256'],
                            reference['size_bytes'], 'scar.target-hardware-graph')


def _matrix(row):
    shape = row['shape']
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(extent) is not int or not 0 < extent < 2**31 for extent in shape)
            or type(row['nbytes']) is not int or row['nbytes'] != 4 * shape[0] * shape[1]
            or row['nbytes'] >= 2**63):
        raise ValueError('CUDA source matrix extent or byte count is outside the provider ABI')
    return shape


def plan_cuda_export(projection, model, *, model_ref):
    """Return a replayable CUDA implementation, schedule and ordered file materials.

    Synthetic durations qualify an offline schedule for C++ generation. The
    planner certificate never authorizes selection over the deployed CPU path.
    """
    target = _profile_target(model, model_ref)
    if (projection.get('schema') != SOURCE_SCHEMA or type(projection.get('schema_version')) is not int
            or projection['schema_version'] != 1 or projection.get('layout') != 'CONTIGUOUS_ROW_MAJOR'
            or projection.get('dtype') != 'FLOAT32'
            or projection.get('ownership') != 'READ_ONLY_ENTRY_VALUES_AND_FRESH_RESULTS'
            or not projection['operations'] or not projection['outputs'] or not projection['inputs']):
        raise ValueError('CUDA lowering needs a source-qualified functional matrix graph')
    program = projection['program']
    profile_ref = dict(model_ref, qualification='SYNTHETIC_CUDA_OFFLINE_PROFILE')
    resources = tuple(PlanningResource(name, kind, capacity, (profile_ref,)) for name, kind, capacity in (
        (_HOST, 'MEMORY', model['pinned_host_bytes']), (_DEVICE, 'MEMORY', model['device_bytes']),
        (_UPLOAD, 'CHANNEL', 1), (_COMPUTE, 'CHANNEL', 1), (_DOWNLOAD, 'CHANNEL', 1), (_GPU, 'EXECUTION', 1)))

    def cost(key):
        ns = model['model_durations_ns'][key]
        return CostEstimate(ns, None, None, 'DECLARED_MODEL', 'SYNTHETIC_CUDA_OFFLINE_' + key.upper(), (profile_ref,))

    definitions, inputs = {}, []
    for item in projection['inputs']:
        name = item['id']
        if name in definitions:
            raise ValueError('CUDA source input ports must be distinct')
        inputs.append(name)
        definitions[name] = (None, _matrix(item), item['nbytes'])

    operations, data, uploads = [], [], defaultdict(list)
    for item in projection['operations']:
        operation, result = item['id'], item['result']
        shape = _matrix(item)
        left, right = item['left'], item['right']
        if (result in definitions or operation == _RETURN or left == right
                or left not in definitions or right not in definitions):
            raise ValueError('CUDA GEMM requires distinct valid inputs and one fresh output')
        m, k = definitions[left][1]
        right_k, n = definitions[right][1]
        if right_k != k or shape != [m, n] or item['reduction'] != k:
            raise ValueError('CUDA GEMM matrix geometry does not match its qualified source inputs')
        origin = (dict(program, operation=item['operation']),)
        left_edge, right_edge = f'{operation}:left', f'{operation}:right'
        for name, edge_id in ((left, left_edge), (right, right_edge)):
            producer, _, size = definitions[name]
            data.append(PlanningDataEdge(edge_id, name, producer, operation, size,
                                         _HOST if producer is None else None, source_refs=origin))
            if producer is None:
                uploads[name].append(edge_id)
        # The same row-major bytes describe each transpose in column-major
        # form, so C^T = B^T A^T produces the original row-major output bytes.
        gemm = ImplementationOption('sgemm', 'CUBLAS_SGEMM_F32_COLUMN_MAJOR_V1', (*origin, profile_ref),
            {_COMPUTE: 1, _GPU: 1}, {_DEVICE: model['workspace_bytes']},
            {left: _DEVICE, right: _DEVICE}, {result: _DEVICE}, cost('sgemm'),
            dict(left_edge=right_edge, right_edge=left_edge, rows=n, columns=m, reduction=k, completion='SAME_STREAM'))
        operations.append(PlanningOperation(operation, origin, (), (gemm,), gemm.id))
        definitions[result] = (operation, shape, item['nbytes'])

    outputs = tuple(dict.fromkeys(projection['outputs']))
    if set(uploads) != set(inputs) or any(name not in definitions or definitions[name][0] is None for name in outputs):
        raise ValueError('CUDA file ABI requires used inputs and produced output values')
    for value in outputs:
        producer, _, size = definitions[value]
        data.append(PlanningDataEdge(f'{_RETURN}:{value}', value, producer, _RETURN, size, None, source_refs=(program,)))
    terminal = ImplementationOption('return', 'CUDA_COPY_RETURN', (program,), {}, {}, dict.fromkeys(outputs, _HOST), {},
                                    CostEstimate(0, 0, 0, 'DECLARED_MODEL', 'LOGICAL_ENDPOINT'), {})
    operations.append(PlanningOperation(_RETURN, (program,), (), (terminal,), terminal.id))

    # The native ABI retains every allocation through return. The shared
    # timeline also checks incremental lifetimes, which can have a lower peak.
    pinned = sum(definitions[name][2] for name in (*inputs, *outputs))
    device = sum(size for _, _, size in definitions.values()) + model['workspace_bytes']
    if pinned > model['pinned_host_bytes'] or device > model['device_bytes']:
        raise ValueError('CUDA file ABI backing and workspace exceed the declared capacity')

    problem = PlanningProblem(target, (program,), resources, tuple(operations), tuple(data), assumptions=(
        'OFFLINE_SYNTHETIC_ONLY', 'ACTUAL_DEVICE_TIMING_UNKNOWN', 'NORMAL_COMPLETION_FP32_MATRIX_GRAPH',
        'ROW_MAJOR_STORAGE_VIA_TRANSPOSED_COLUMN_MAJOR_SGEMM', 'ONE_UPLOAD_PER_REACHABLE_EXTERNAL_VALUE',
        'ALL_FILE_ABI_BUFFERS_AND_PROVIDER_WORKSPACE_RETAINED_THROUGH_RETURN'))
    upload = TransferOption('h2d', _HOST, _DEVICE, {_UPLOAD: 1}, cost('h2d'), 'CUDA_MEMCPY_ASYNC', (profile_ref,))
    download = TransferOption('d2h', _DEVICE, _HOST, {_DOWNLOAD: 1}, cost('d2h'), 'CUDA_MEMCPY_ASYNC', (profile_ref,))
    for name in inputs:
        problem = expand_shared_material_route(problem, uploads[name], (upload,))
    for name in outputs:
        problem = expand_material_route(problem, f'{_RETURN}:{name}', (download,))
    exported_materials = {edge.id: edge.value for edge in problem.data_edges if edge.consumer == _RETURN}
    bindings = dict(device=model['device'], memory_kinds={_HOST: 'PinnedHost', _DEVICE: 'Device'},
                    stream_channels=[_UPLOAD, _COMPUTE, _DOWNLOAD], compute_resource=_GPU, input_values=inputs,
                    output_values=[exported_materials[f'{_RETURN}:{name}'] for name in projection['outputs']])
    schedule = construct_implementations(problem)
    verify_execution_plan(problem, schedule)
    return problem_document(problem), schedule, bindings


def replay_cuda_export(projection, model, *, model_ref, problem, schedule, bindings):
    """Rebuild the declared CUDA plan and verify the archived timeline independently."""
    expected_problem, expected_schedule, expected_bindings = plan_cuda_export(projection, model, model_ref=model_ref)
    if (problem != expected_problem or bindings != expected_bindings
            or {key: value for key, value in schedule.items() if key != 'elapsed_seconds'} !=
            {key: value for key, value in expected_schedule.items() if key != 'elapsed_seconds'}):
        raise ValueError('archived CUDA plan or file material bindings differ from the source replay')
    return verify_execution_plan(problem_from_dict(problem), schedule)

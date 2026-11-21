"""Plan one NVIDIA CPU/CUDA FP32 entry with explicit host and device placements."""

from dataclasses import asdict, replace
from pathlib import Path

from scar.compiler.aten_export_cuda_plan import _profile_target
from scar.compiler.aten_export_device_placement import placement_choices, placement_proposal
from scar.compiler.aten_export_hybrid_plan import _MM, _build_placement, _certify_placement
from scar.compiler.aten_export_hybrid_source import qualify_hybrid_export
from scar.ir.execution_recipe import read_artifact
from scar.ir.implementation_plan import problem_document, problem_from_dict
from scar.ir.program.providers import ProviderArtifact
from scar.optimization.constructive_planner import construct_implementations
from scar.optimization.joint_planner import verify_execution_plan
from scar.optimization.native_device_placement_search import search_native_device_placements


OFFLINE_PROFILE = 'scar.aten-export-nvhpc-hybrid-offline-profile'
PLACEMENT_PROFILE = 'scar.aten-export-nvhpc-placement-offline-profile'
HOST_RESOURCE = 'cpu:nvhpc'
HOST_LOWERING = 'NVHPC_CBLAS_SGEMM_F32_ROW_MAJOR_V1'
HOST_PROVIDER = 'NVIDIA_HPC_CBLAS_LP64'
_CPU_FIELDS = frozenset(('host_cpu_ids', 'host_target', 'placement'))
_SEARCH_FIELDS = frozenset(('host_cpu_ids', 'host_target', 'host_memory_bytes', 'max_placements', 'host_calibration'))


def qualify_nvhpc_hybrid_export(recipe, shapes=None, *, graph, program_ref, input_kinds):
    """Admit only MM operations shared by the locked CPU and CUDA libraries."""
    projection = qualify_hybrid_export(recipe, shapes, graph=graph, program_ref=program_ref,
                                       input_kinds=input_kinds, allow_cuda_only=True)
    if any(row['schema'] != _MM for row in projection['operations']):
        raise ValueError('same-stack NVIDIA hybrid lowering requires the exact functional FP32 MM overload')
    return projection


def _host(model):
    cpus = model.get('host_cpu_ids')
    if type(cpus) is not list or len(cpus) != 1 or type(cpus[0]) is not int or not 0 <= cpus[0] < 1024:
        raise ValueError('NVIDIA hybrid requires one explicit host hardware thread')
    binding = dict(host_resource=HOST_RESOURCE, host_provider=HOST_PROVIDER, host_lowering=HOST_LOWERING,
        host_cpu_ids=cpus, host_threads=1, host_explicit_workspace_bytes=0, host_provider_workspace_status='UNKNOWN')
    if 'host_target' in model:
        from scar.compiler.plan_aten_boundary import _target

        reference = model['host_target']
        if type(reference) is not dict or reference.get('schema') != 'scar.target-hardware-graph':
            raise ValueError('NVIDIA hybrid host target must be an original hardware graph receipt')
        read_artifact(reference)
        target, resources = _target(Path(reference['uri']).parent, cpus, allow_cpu_subset=True)
        if asdict(target) != reference:
            raise ValueError('NVIDIA hybrid host resource differs from its recorded hardware graph')
        binding.update(host_target=reference, host_topology_resource_ids=[row.id for row in resources[cpus[0]]])
    return binding


def plan_nvhpc_hybrid_export(projection, model, *, model_ref):
    """Build and cold-verify the offline timeline of the declared mixed route."""
    if type(model) is not dict or type(model.get('placement')) is not dict:
        raise ValueError('NVIDIA hybrid profile requires a complete CPU/CUDA placement map')
    if set(model['placement'].values()) != {'CPU', 'CUDA'}:
        raise ValueError('NVIDIA hybrid profile requires both CPU and CUDA operations')
    base = {key: value for key, value in model.items() if key not in _CPU_FIELDS}
    target = _profile_target(base, model_ref, schema=OFFLINE_PROFILE, extra_durations=('cpu_mm',))
    return _certify_placement(*_build_placement(projection, base, model_ref, target, model['placement'],
                                                nvhpc_host=_host(model)))


def replay_nvhpc_hybrid_export(projection, model, *, model_ref, problem, schedule, bindings):
    """Recompute provider modes, crossings, resource bounds and the scheduled order."""
    expected_problem, expected_schedule, expected_bindings = plan_nvhpc_hybrid_export(
        projection, model, model_ref=model_ref)
    if (problem != expected_problem or bindings != expected_bindings
            or {key: value for key, value in schedule.items() if key != 'elapsed_seconds'} !=
            {key: value for key, value in expected_schedule.items() if key != 'elapsed_seconds'}):
        raise ValueError('NVIDIA hybrid plan differs from replayed MM operations and resource constraints')
    return verify_execution_plan(problem_from_dict(problem), schedule)


def _pageable_calibration(projection, calibration, host):
    from scar.compiler.operator_stack import load_operator_stack
    from scar.optimization.nvhpc_sgemm_cost import sgemm_body_costs

    if (type(calibration) is not dict or set(calibration) != {'operator_stack', 'observations'}
            or type(calibration['operator_stack']) is not dict or type(calibration['observations']) is not list
            or not calibration['observations'] or 'host_target' not in host):
        raise ValueError('pageable SGEMM calibration requires an operator stack, observations and a pinned host target')
    stack, sdk = load_operator_stack(calibration['operator_stack'])
    costs = sgemm_body_costs(projection, calibration['observations'], stack=sdk,
        target=ProviderArtifact(**host['host_target']), cpus=host['host_cpu_ids'])
    points = {id_: cost if cost.estimate_ns is None else replace(cost, basis='DECLARED_MODEL',
        context='OFFLINE_POINT_FROM_MATCHED_WARM_PAGEABLE_CPU_SGEMM') for id_, cost in costs.items()}
    receipt = dict(operator_stack=stack, observations=calibration['observations'],
        scope='WARM_PAGEABLE_ROW_MAJOR_FP32_SGEMM_POINT_FORECAST',
        pageable_cpu_points_ns={id_: cost.estimate_ns for id_, cost in points.items()},
        physical_duration_bounds_proven=False, pinned_host_points_admitted=False)
    return points, receipt


def _apply_pageable_points(problem, bindings, points):
    def priced(operation):
        if operation.id not in points:
            return operation
        option, = operation.options
        return replace(operation, options=(replace(option, cost=points[operation.id]),))

    problem = replace(problem, operations=tuple(priced(operation) for operation in problem.operations))
    if any(cost.estimate_ns is None for cost in points.values()):
        return problem_document(problem), construct_implementations(problem), bindings
    return _certify_placement(problem, bindings)


def propose_nvhpc_placement(projection, model, *, model_ref):
    """Compare CPU, CUDA and mixed placements under one declared FP32 body model."""
    if type(model) is not dict or type(model.get('host_memory_bytes')) is not int or not 0 < model['host_memory_bytes'] < 2**63:
        raise ValueError('NVIDIA placement requires a bounded pageable host memory capacity')
    if any(row['schema'] != _MM for row in projection['operations']):
        raise ValueError('NVIDIA placement requires exact FP32 MM eligibility for every operation')
    base = {key: value for key, value in model.items() if key not in _SEARCH_FIELDS}
    target = _profile_target(base, model_ref, schema=PLACEMENT_PROFILE, extra_durations=('cpu_mm', 'cpu_mm_pageable'))
    host = _host(model)
    choices = placement_choices(projection)
    physical = dict(base, host_memory_bytes=model['host_memory_bytes'])
    costs = base['model_durations_ns']
    pageable = dict(physical, model_durations_ns=dict(costs, cpu_mm=costs['cpu_mm_pageable']))
    calibration = None if 'host_calibration' not in model else _pageable_calibration(
        projection, model['host_calibration'], host)

    def build(placement):
        cuda = 'CUDA' in placement.values()
        selected = physical if cuda else pageable
        problem, bindings = _build_placement(projection, selected, model_ref, target, placement, nvhpc_host=host)
        if calibration is not None and not cuda:
            return _apply_pageable_points(problem, bindings, calibration[0])
        return _certify_placement(problem, bindings)

    analysis, selected = search_native_device_placements(choices, build, max_placements=model.get('max_placements'))
    report = placement_proposal(projection, choices, analysis, selected)
    report.update(schema='scar.nvhpc-placement-proposal', model=model_ref, program=projection['program'],
        timing_scope='LOADED_HOST_MATRICES_THROUGH_HOST_RESULT_READY', default_backend='NO_OP',
        hardware=dict(cpu=dict(status='PINNED_TOPOLOGY' if 'host_target' in host else 'DECLARED_CPU_MASK', **host),
                      cuda=dict(status='RUNTIME_PREFLIGHT_REQUIRED', model_ordinal=base['device'])),
        runtime_requirements=['MATCHED_OPERATOR_STACK', 'MATCHED_CUDA_DEVICE_IDENTITY_FOR_GPU_ROUTES',
            'SUFFICIENT_LIVE_HOST_AND_DEVICE_STORAGE', 'SAME_ENTRY_NUMERICAL_AND_PUBLICATION_VALIDATION'],
        execution_authorized=False)
    if calibration is not None:
        report['host_calibration'] = calibration[1]
    return report, selected


def plan_nvhpc_placement(projection, model, *, model_ref):
    """Nominate a CUDA-containing plan for offline emission after full-domain comparison."""
    report, selected = propose_nvhpc_placement(projection, model, model_ref=model_ref)
    if selected is None or 'CUDA' not in selected['placement'].values():
        raise ValueError('declared NVIDIA placement retained CPU or remains unresolved')
    return selected['problem'], selected['schedule'], selected['bindings'], report


def replay_nvhpc_placement(projection, model, *, model_ref, problem, schedule, bindings, placement_search):
    """Re-run stable finite placement search and independently verify the selected timeline."""
    expected_problem, expected_schedule, expected_bindings, expected = plan_nvhpc_placement(
        projection, model, model_ref=model_ref)
    if (expected != placement_search or expected_problem != problem or expected_bindings != bindings
            or {key: value for key, value in schedule.items() if key != 'elapsed_seconds'} != expected_schedule):
        raise ValueError('NVIDIA selected placement or timing plan differs from finite-domain replay')
    return verify_execution_plan(problem_from_dict(problem), schedule)

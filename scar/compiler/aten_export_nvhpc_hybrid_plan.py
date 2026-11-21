"""Plan one NVIDIA CPU/CUDA FP32 entry with explicit host and device placements."""

from dataclasses import asdict
from pathlib import Path

from scar.compiler.aten_export_cuda_plan import _profile_target
from scar.compiler.aten_export_hybrid_plan import _MM, _build_placement, _certify_placement
from scar.compiler.aten_export_hybrid_source import qualify_hybrid_export
from scar.ir.execution_recipe import read_artifact
from scar.ir.implementation_plan import problem_from_dict
from scar.optimization.joint_planner import verify_execution_plan


OFFLINE_PROFILE = 'scar.aten-export-nvhpc-hybrid-offline-profile'
HOST_RESOURCE = 'cpu:nvhpc'
HOST_LOWERING = 'NVHPC_CBLAS_SGEMM_F32_ROW_MAJOR_V1'
HOST_PROVIDER = 'NVIDIA_HPC_CBLAS_LP64'
_CPU_FIELDS = frozenset(('host_cpu_ids', 'host_target', 'placement'))


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

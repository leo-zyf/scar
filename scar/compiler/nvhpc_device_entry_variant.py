"""Derive an original matrix-file entry's NVIDIA device candidate before building it."""

from pathlib import Path

from scar.compiler.aten_export_cuda_plan import OFFLINE_PROFILE as CUDA_PROFILE
from scar.compiler.aten_export_nvhpc_hybrid_plan import OFFLINE_PROFILE as HYBRID_PROFILE
from scar.compiler.aten_export_nvhpc_hybrid_plan import PLACEMENT_PROFILE, propose_nvhpc_placement
from scar.compiler.compile_exported_aten_cuda import (
    _MODEL_PATH, _check_host_calibration_stack, _native_sources, _original_entry, _pipeline,
)
from scar.compiler.nvhpc_entry_variant import matrix_entry_inputs
from scar.compiler.operator_stack import load_operator_stack, nvidia_stack_build
from scar.compiler.tensor_program_graph import straight_line_graph, tensor_type
from scar.ir.execution_recipe import read_artifact
from scar.optimization.resource_solver import semantic_sha256


FAMILY = 'NVIDIA_DEVICE_MATRIX_FILE_ENTRY'
SCHEMA = 'scar.nvhpc-device-source-variants'
PROFILE_SCHEMAS = (CUDA_PROFILE, HYBRID_PROFILE, PLACEMENT_PROFILE)


class _Frontier(ValueError):
    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason = reason


def _require(condition, reason, detail):
    if not condition:
        raise _Frontier(reason, detail)


def _source_graph(scope):
    recipe, source = _original_entry(scope['entry'])
    _require((source['source']['path'], source['source']['sha256']) == (scope['entry'], scope['entry_sha256']),
             'ORIGINAL_F4_ENTRY_CHANGED', 'matrix source bytes differ from the original F4 entry')
    projection = source['projection']
    shapes = {row['id']: row['shape'] for row in (*projection['inputs'], *projection['operations'])}
    common = scope.get('input_shapes', {}).get('common_shape')
    _require(common is None or all(row['shape'] == common for row in projection['inputs']),
             'ORIGINAL_F4_INPUT_SHAPES_DIFFER', 'matrix file input shapes differ from the declared original scope')
    graph = straight_line_graph(recipe, {name: tensor_type(shape) for name, shape in shapes.items()})
    kinds = dict.fromkeys((row['id'] for row in recipe['inputs']), 'INPUT')
    return recipe, source, shapes, graph, kinds


def _check_host_scope(scope, profile):
    if 'host_target' in profile:
        _require(profile['host_target'] == scope['target'], 'DEVICE_HOST_TARGET_DIFFERS',
                 'device profile host graph differs from the original F4 target')
    if 'host_cpu_ids' in profile:
        cpus = profile['host_cpu_ids']
        _require(type(cpus) is list and all(type(cpu) is int for cpu in cpus)
                 and set(cpus) <= set(scope['process_affinity_cpu_ids']), 'DEVICE_HOST_COHORT_DIFFERS',
                 'device profile host CPUs are outside the original invocation affinity')
        if 'host_target' not in profile:
            from scar.compiler.plan_aten_boundary import _target

            _target(Path(scope['target']['uri']).parent, cpus, allow_cpu_subset=True)


def derive_nvhpc_device_entry(scope, profile_ref):
    """Return the offline device nominee, with exact materials and an open release gate."""
    result = dict(schema=SCHEMA, schema_version=1, family=FAMILY, scope_sha256=semantic_sha256(scope),
        options=[], coverage_frontier=[], compiled_source_entry=False, target_executed=False)
    try:
        _require(all(scope.get(key) for key in ('program_sha256', 'source_admission_sha256',
                                               'entry_obligations_sha256')),
                 'ORIGINAL_F4_ENTRY_OBLIGATIONS_REQUIRED', 'original F4 entry, admission and effect obligations are required')
        _require(scope.get('boundary') == 'PROCESS_LAUNCH_THROUGH_EXIT' and scope.get('operator_stack') == 'NVIDIA_HPC'
                 and scope.get('native_output_contract') == 'STANDALONE_CPP',
                 'DEVICE_ENTRY_SCOPE_DIFFERS', 'pinned NVIDIA standalone whole-entry scope required')
        target = read_artifact(scope['stack_target'])
        _require(target.get('target') == scope['target'], 'DEVICE_HARDWARE_TARGET_DIFFERS',
                 'operator stack is not bound to the original F4 hardware graph')
        _require(target.get('cuda_provider_installed') is True, 'PINNED_CUDA_PROVIDER_UNAVAILABLE',
                 'offline planning needs the locked CUDA provider files')
        stack, sdk = load_operator_stack(target['stack'])
        _require(type(profile_ref) is dict and profile_ref.get('schema') in PROFILE_SCHEMAS,
                 'NVIDIA_OFFLINE_PROFILE_REQUIRED', 'a supported NVIDIA offline profile receipt is required')
        profile = read_artifact(profile_ref)
        _check_host_calibration_stack(profile, stack)
        _check_host_scope(scope, profile)
        profile_member = dict(path=_MODEL_PATH, sha256=profile_ref['sha256'], size_bytes=profile_ref['size_bytes'])
        recipe, source, shapes, graph, kinds = _source_graph(scope)
        files = matrix_entry_inputs(source, scope['argv'])
        qualify, plan, _, status = _pipeline(profile)
        program = source['program']
        projection = qualify(recipe, shapes, graph=graph, program_ref=program, input_kinds=kinds)
        placement_search = None
        if profile_ref['schema'] == PLACEMENT_PROFILE:
            placement_search, selected = propose_nvhpc_placement(projection, profile, model_ref=profile_member)
            if selected is None or 'CUDA' not in selected['placement'].values():
                result['placement_search'] = placement_search
                reason = ('OFFLINE_PLACEMENT_RETAINED_CPU' if placement_search['analysis']['placement_domain_complete']
                          else 'OFFLINE_PLACEMENT_DOMAIN_OPEN')
                raise _Frontier(reason, 'the finite placement model nominated no CUDA-containing plan')
            problem, schedule, bindings = (selected[key] for key in ('problem', 'schedule', 'bindings'))
        else:
            problem, schedule, bindings = plan(projection, profile, model_ref=profile_member)
        route = 'HYBRID' if 'host_resource' in bindings else 'CUDA'
        nvidia_stack_build(sdk, route)
        schedule = {key: value for key, value in schedule.items() if key != 'elapsed_seconds'}
        allocations = _native_sources(projection, problem, schedule, bindings).reserved_payload_bytes
        payload = sum(allocations.values())
        budget = scope['memory_budget_bytes']
        _require(type(budget) is int and 0 < payload <= budget, 'DEVICE_ENTRY_STORAGE_BUDGET_EXCEEDED',
                 'retained host and device payloads exceed the whole-entry graph-held storage budget')
        device = bindings['device']
        observed_device = dict(evidence=target['cuda_device_evidence'], device_ordinal=device,
                               ordinal_bound=device in target['bound_cuda_ordinals'])
        emission = dict(source_contract=source, operator_stack=stack, profile=profile_ref, program=program,
            problem=problem, schedule=schedule, bindings=bindings, placement_search=placement_search, files=files,
            numerical_contract=scope.get('numerical_contract', {'kind': 'EXACT_BYTES'}),
            source_f4_program_sha256=scope['program_sha256'], source_effect_obligations_sha256=scope['entry_obligations_sha256'],
            route=route, artifact_status=status, observed_device=observed_device,
            graph_storage=dict(reserved_payload_bytes=allocations, retained_bytes=payload, budget_bytes=budget),
            forecast=dict(scope='LOADED_HOST_MATRICES_THROUGH_HOST_RESULT_READY', declared_ns=schedule['makespan_ns'],
                          observed_device_time_ns=None, physical_duration_bound_proven=False))
        option_id = 'planned:'+semantic_sha256(dict(scope_sha256=result['scope_sha256'], plan=emission))
        result['options'].append(dict(id=option_id, legal_status='UNRESOLVED',
            reason='CUDA_RUNTIME_AND_ORIGINAL_ENTRY_EFFECTS_REQUIRE_VALIDATION', implementation_family=FAMILY,
            proof=dict(emission_plan=emission)))
    except _Frontier as error:
        result['coverage_frontier'].append(dict(reason=error.reason, detail=str(error)))
    except (OSError, SyntaxError, ValueError, KeyError, TypeError) as error:
        result['coverage_frontier'].append(dict(reason='SOURCE_DEVICE_MATRIX_PROFILE_OPEN', detail=str(error)))
    return result

"""Emit and verify the retained NVIDIA device plan for an original Python entry."""

import json
from pathlib import Path
from subprocess import SubprocessError

from scar.codegen.native_provider_bundle import _content
from scar.compiler.compile_exported_aten_cuda import SCHEMA, compile_exported_aten_cuda
from scar.compiler.entry_delivery import HANDOFF, NVHPC_DEVICE_ROUTE, replay_entry_delivery
from scar.compiler.nvhpc_entry_variant import matrix_entry_inputs
from scar.ir.execution_recipe import artifact_reference, read_artifact


def _plan_scope(plan, scope):
    source = plan['source_contract']['source']
    stack_target = read_artifact(scope['stack_target'])
    numerics = scope.get('numerical_contract', {'kind': 'EXACT_BYTES'})
    if (source['path'] != scope['entry'] or source['sha256'] != scope['entry_sha256']
            or stack_target['target'] != scope['target'] or stack_target['stack'] != plan['operator_stack']
            or plan['source_f4_program_sha256'] != scope['program_sha256']
            or plan['source_effect_obligations_sha256'] != scope['entry_obligations_sha256']
            or plan['source_contract']['program'] != plan['program'] or plan['numerical_contract'] != numerics
            or matrix_entry_inputs(plan['source_contract'], scope['argv']) != plan['files']):
        raise ValueError('selected NVIDIA device plan differs from the original entry scope')
    read_artifact(plan['profile'])


def _check_emission(project, manifest, plan):
    def record(name):
        return json.loads((project/'plan'/name).read_bytes())

    profile = plan['profile']
    emitted_schedule = record('schedule.json')
    emitted_schedule.pop('elapsed_seconds', None)
    placement_path = project/'plan/placement-search.json'
    emitted_placement = record(placement_path.name) if placement_path.exists() else None
    checks = {
        'artifact kind': (manifest['status'], plan['artifact_status']),
        'operator stack': (manifest['operator_stack']['reference'], plan['operator_stack']),
        'device route': (manifest['operator_stack']['linkage']['route'], plan['route']),
        'profile bytes': (_content(project/'plan/cuda-model.json'),
            {key: profile[key] for key in ('sha256', 'size_bytes')}),
        'program graph': (dict(path='plan/program.json', **_content(project/'plan/program.json')), plan['program']),
        'source contract': (record('source-entry.json'), plan['source_contract']),
        'physical problem': (record('problem.json'), plan['problem']),
        'schedule': (emitted_schedule, plan['schedule']),
        'material bindings': (record('bindings.json'), plan['bindings']),
        'placement search': (emitted_placement, plan['placement_search']),
    }
    differences = [name for name, (actual, expected) in checks.items() if actual != expected]
    if differences:
        raise ValueError('emitted NVIDIA device project differs from selected ' + ', '.join(differences))


def emit_nvhpc_device_entry(option, scope, out, *, plan_only, verify_entry):
    """Build only the retained placement and use the original-process device gate."""
    plan = option['proof']['emission_plan']
    result = dict(schema=HANDOFF, schema_version=1, selected_id=option['id'], scope=scope,
        final_action='NATIVE_COMPILATION_DEFERRED', reason='plan-only preserved the NVIDIA device emission plan',
        selected_emission=plan, native_compiled=False, target_executed=False, validation_attempted=False,
        operator_stack=plan['operator_stack'], qualification=None, validation=None, launch_authority=None,
        deployment_authorized=False)
    try:
        _plan_scope(plan, scope)
    except (OSError, ValueError) as error:
        return dict(result, final_action='KEEP_ORIGINAL', reason='NVIDIA device plan scope failed: ' + str(error))
    if plan_only:
        return result

    project = Path(out)/'native-entry'
    profile = plan['profile']
    try:
        compile_exported_aten_cuda(source_entry=scope['entry'], operator_stack=plan['operator_stack'],
            cuda_model=profile['uri'], cuda_model_sha256=profile['sha256'], out=project)
        reference = artifact_reference(project/'artifact.json', SCHEMA)
        manifest = read_artifact(reference)
    except (OSError, ValueError, SubprocessError) as error:
        return dict(result, final_action='KEEP_ORIGINAL', reason='NVIDIA device emission failed: ' + str(error))
    result.update(native_compiled=True, native=dict(artifact=reference, project=str(project)))
    try:
        _check_emission(project, manifest, plan)
    except (OSError, ValueError) as error:
        return dict(result, final_action='KEEP_ORIGINAL', reason=str(error))

    candidate = dict(id=option['id'], route=NVHPC_DEVICE_ROUTE, artifact=reference, validation=None)
    admitted = replay_entry_delivery(scope, selected_id=option['id'], candidate=candidate)
    result.update(admitted)
    if admitted['final_action'] != 'CANDIDATE_VALIDATION_REQUIRED':
        return result
    evidence = admitted['qualification']['evidence']
    if evidence['file_inputs'] != plan['files']:
        return dict(result, final_action='KEEP_ORIGINAL', reason='original matrix inputs changed before device trial')
    storage = plan['graph_storage']
    kinds = plan['bindings']['memory_kinds']
    payload = storage['reserved_payload_bytes']
    device = sum(size for pool, size in payload.items() if kinds[pool] == 'Device')
    host = sum(size for pool, size in payload.items() if kinds[pool] != 'Device')
    if (storage['retained_bytes'] != host + device or storage['budget_bytes'] != scope['memory_budget_bytes']
            or (evidence['host_payload_bytes'], evidence['device_payload_bytes']) != (host, device)):
        return dict(result, final_action='KEEP_ORIGINAL', reason='emitted NVIDIA payload differs from the selected memory plan')
    if not verify_entry:
        return result

    from scar.compiler.nvhpc_source_device_validation import validate_source_device

    validation = Path(out)/'native-validation'
    try:
        observed = validate_source_device(scope, evidence, validation)
        candidate['validation'] = artifact_reference(validation/'result.json', observed['schema'])
        replay = replay_entry_delivery(scope, selected_id=option['id'], candidate=candidate)
    except (OSError, ValueError, SubprocessError) as error:
        return dict(result, final_action='KEEP_ORIGINAL', validation_attempted=True, target_executed=None,
                    reason='NVIDIA device trial validation failed: ' + str(error))
    return dict(result, **replay, validation_attempted=True, target_executed=bool(observed['runs']),
                trial_observation=candidate['validation'], device_trial_phase=observed['phase'])

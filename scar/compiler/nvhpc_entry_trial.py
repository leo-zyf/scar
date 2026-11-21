"""Emit and verify a source-derived whole-entry CPU trial on its declared scope."""

import json
from pathlib import Path
from subprocess import SubprocessError

from scar.compiler.compile_exported_nvhpc import SCHEMA, compile_exported_nvhpc
from scar.compiler.entry_delivery import HANDOFF, replay_entry_delivery
from scar.ir.execution_recipe import artifact_reference


def emit_nvhpc_entry(option, scope, out, *, plan_only, verify_entry):
    """Cold replay the derived fixed graph, then materialize the requested C++ candidate."""
    plan = option['proof']['emission_plan']
    result = dict(schema=HANDOFF, schema_version=1, selected_id=option['id'], scope=scope,
        final_action='NATIVE_COMPILATION_DEFERRED', reason='plan-only preserved the C++ emission plan',
        selected_emission=plan, native_compiled=False, target_executed=False,
        operator_stack=plan['operator_stack'], qualification=None, validation=None, deployment_authorized=False)
    if plan_only:
        return result
    project = out/'native-entry'
    stack = plan['operator_stack']
    numerics = plan['numerical_contract']
    try:
        compile_exported_nvhpc(source_entry=scope['entry'], stack=stack['uri'], stack_sha256=stack['sha256'],
            target=Path(scope['target']['uri']).parent, cpus=plan['cpu_ids'], threads=plan['threads'],
            tensor_budget_bytes=scope['memory_budget_bytes'], out=project,
            sgemm_observations=[row['uri'] for row in plan['sgemm_observations']],
            atol=numerics.get('atol'), rtol=numerics.get('rtol'))
    except (OSError, ValueError, SubprocessError) as error:
        return dict(result, final_action='KEEP_ORIGINAL', reason='NVHPC emission failed: ' + str(error))
    reference = artifact_reference(project/'artifact.json', SCHEMA)
    candidate = dict(id=option['id'], route='NVHPC_SOURCE_ENTRY', artifact=reference, validation=None)
    result.update(native_compiled=True, native=dict(artifact=reference, project=str(project)))
    checks = {'source-entry': plan['source_contract'], 'problem': plan['problem']}
    if 'body_forecast' in plan:
        checks['body-forecast'] = plan['body_forecast']
    if any(json.loads((project/f'plan/{name}.json').read_bytes()) != expected
           for name, expected in checks.items()):
        return dict(result, final_action='KEEP_ORIGINAL', reason='emitted NVIDIA plans differ from the chosen proof')
    admitted = replay_entry_delivery(scope, selected_id=option['id'], candidate=candidate)
    result.update(admitted)
    if admitted['final_action'] != 'CANDIDATE_VALIDATION_REQUIRED':
        return result
    if admitted['qualification']['evidence']['file_inputs'] != plan['files']:
        return dict(result, final_action='KEEP_ORIGINAL', reason='original matrix inputs changed before the trial')
    if not verify_entry:
        return result

    from scar.compiler.nvhpc_source_validation import validate_source_process

    input_names = [row['value'] for row in plan['source_contract']['inputs']]
    arguments = [f'{name}={record["uri"]}' for name, record in zip(input_names, plan['files'], strict=True)]
    validation = out/'native-validation'
    try:
        observed = validate_source_process(project, reference['sha256'], arguments, validation, entry_scope=scope)
    except (OSError, ValueError, SubprocessError) as error:
        return dict(result, final_action='KEEP_ORIGINAL', target_executed=None,
                    reason='NVHPC trial process validation failed: ' + str(error))
    candidate['validation'] = artifact_reference(validation/'result.json', observed['schema'])
    replay = replay_entry_delivery(scope, selected_id=option['id'], candidate=candidate)
    return dict(result, **replay, target_executed=True, trial_observation=candidate['validation'])

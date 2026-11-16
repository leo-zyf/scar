"""Trial delivery of a source-replayed canonical copy-component choice.

The original ABI1 artifact and its guards remain the executable. This sidecar
binds a component-model decision to it without authorizing production use.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from scar.codegen import stateful as base_stateful, window_order
from scar.compiler.stateful_model import stateful_digest, stateful_from_dict
from scar.ir.activity_costs import activity_cost_document
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.temporal_resources import projection_document
from scar.ir.versioned_schedule import schedule_document
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.native-copy-choice-trial'
CHOICE_SCHEMA = 'scar.copy-window-choice-preparation'


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _directory(path):
    path = Path(path).resolve()
    return path if path.is_dir() else path.parent


def _reference(path):
    path = Path(path).resolve()
    if path.is_dir():
        path /= 'result.json'
    return artifact_reference(path, CHOICE_SCHEMA)


def _qualified(reference):
    from scar.compiler.choose_copy_window import replay_copy_window_choice
    report, choice, original_costs, replacement_costs, context = replay_copy_window_choice(reference['uri'])
    _need(read_artifact(reference) == report
          and (report.get('schema'), report.get('schema_version')) == (CHOICE_SCHEMA, 1),
          'original copy-choice preparation changed during complete replay')
    _need((choice.get('schema'), choice.get('schema_version')) == ('scar.copy-window-choice', 1)
          and choice.get('selected_variant') in ('ORIGINAL', 'FUSED'),
          'one supported source-replayed copy-component choice required')
    expected_decisions = {'ORIGINAL': ('KEEP_ORIGINAL_WITHIN_COMPONENT_MODEL', 'KEEP_ORIGINAL_ON_TIE'),
                          'FUSED': ('TRIAL_REWRITE_WITHIN_COMPONENT_MODEL',)}
    _need(choice.get('decision') in expected_decisions[choice['selected_variant']],
          'selected copy variant and canonical component-model decision differ')
    _need(report['rewrite'] == context['rewrite_reference']
          and read_artifact(report['choice']) == choice
          and read_artifact(report['original_costs']) == activity_cost_document(original_costs)
          and read_artifact(report['replacement_costs']) == activity_cost_document(replacement_costs)
          and read_artifact(report['calibration']) == context['calibration_document'],
          'copy choice/source/calibration/component publications differ from complete replay')
    certificate = context['certificate']
    _need(certificate['parent_schedule_sha256'] == semantic_sha256(schedule_document(context['original_schedule']))
          and certificate['rewritten_schedule_sha256'] == semantic_sha256(schedule_document(context['rewritten_schedule'])),
          'copy choice does not retain its original and replacement graphs')
    return dict(reference=reference, report=report, choice=choice, context=context,
                original_costs=original_costs, replacement_costs=replacement_costs)


def _child_metadata(directory):
    path = directory/'artifact.json'
    value = json.loads(path.read_bytes())
    _need(value.get('schema_version') == 1 and value.get('stateful_abi_version') == 1,
          'original child artifact schema/ABI1 metadata required')
    for key, name in value['files'].items():
        _need(type(name) is str and Path(name).name == name,
              'child artifact files must remain local basenames')
        raw = (directory/name).read_bytes()
        _need(sha256(raw).hexdigest() == value['sha256'][key],
              'copy choice child artifact checksum differs: ' + key)
    return value


def _validate_child(qualified, directory, metadata):
    """Check branch/source identities before dispatch to the original loader."""
    context = qualified['context']; recipe = context['recipe']
    variant = qualified['choice']['selected_variant']
    expected_schema = base_stateful.SCHEMA if variant == 'ORIGINAL' else window_order.CERTIFIED_SCHEMA
    _need(metadata.get('schema') == expected_schema,
          'selected copy variant differs from the actual child artifact schema')
    region_document = read_artifact(recipe['artifacts']['region'])
    plan = read_artifact(recipe['artifacts']['stage_plan'])
    _need(json.loads((directory/metadata['files']['region']).read_bytes()) == region_document
          and json.loads((directory/metadata['files']['plan']).read_bytes()) == plan,
          'copy choice child must retain the exact original recipe region and base plan')
    region = stateful_from_dict(region_document)
    _need(metadata['region_sha256'] == stateful_digest(region)
          and metadata['provider_runtime'] == recipe['implementation']['backend_capability_record']['provider_runtime']
          and metadata['compiler'] == recipe['implementation']['backend_capability_record']['compiler'],
          'copy choice child source/provider/compiler differs from the admitted recipe')
    core = dict(region_sha256=metadata['region_sha256'], plan_sha256=metadata['plan_sha256'],
                provider_runtime=metadata['provider_runtime'])
    _need(semantic_sha256(core) == metadata['contract_sha256'], 'copy choice child core identity differs')
    if variant == 'ORIGINAL':
        _need(metadata['plan_sha256'] == semantic_sha256(plan)
              and metadata['implementation_sha256'] == base_stateful.implementation_sha256(),
              'original copy choice must retain the exact base backend and plan')
        admitted, actual_plan, binding = base_stateful._admit(region_document, plan)
        _need((directory/metadata['files']['source']).read_bytes()
              == base_stateful._source(admitted, actual_plan, binding, core).encode(),
              'original copy choice source differs from exact recipe replay')
    else:
        rewrite = metadata.get('rewrite', {})
        rewrite_report = read_artifact(context['rewrite_reference'])
        graph_hashes = rewrite.get('graph_semantic_hashes', {})
        _need(rewrite.get('preparation') == context['rewrite_reference']
              and rewrite.get('certificate') == rewrite_report['certificate']
              and rewrite.get('replacement_task_id') == context['certificate']['replacement_task_id']
              and graph_hashes.get('schedule') == semantic_sha256(schedule_document(context['rewritten_schedule']))
              and graph_hashes.get('resources') == semantic_sha256(projection_document(context['rewritten_resources']))
              and graph_hashes.get('certificate') == semantic_sha256(context['certificate'])
              and graph_hashes.get('costs') == semantic_sha256(read_artifact(rewrite_report['costs'])),
              'fused copy choice native artifact differs from its exact rewritten graph/certificate')
        _need(metadata['base_plan_sha256'] == semantic_sha256(plan)
              and metadata['execution_plan_sha256'] == metadata['plan_sha256']
              and semantic_sha256(metadata['execution_plan']) == metadata['plan_sha256'],
              'fused copy choice execution plan/core differs from its certified base plan')
    return metadata


def _receipt(qualified, native_directory, metadata):
    context = qualified['context']; reference = qualified['reference']
    native_ref = artifact_reference(native_directory/'artifact.json', metadata['schema'])
    return dict(schema=SCHEMA, schema_version=1, stateful_abi_version=1,
        choice_preparation=reference, choice=qualified['report']['choice'],
        choice_sha256=semantic_sha256(qualified['choice']),
        selected_variant=qualified['choice']['selected_variant'],
        native_directory='native', native_metadata=native_ref,
        native_identity={key: metadata[key] for key in (
            'build_id', 'region_sha256', 'plan_sha256', 'contract_sha256')},
        native_core_sha256=metadata['contract_sha256'],
        native_source=artifact_reference(native_directory/metadata['files']['source'], 'scar.generated-cpp-source'),
        native_library=artifact_reference(native_directory/metadata['files']['library'], 'scar.native-library'),
        parent_recipe_id=context['recipe']['recipe_id'], rewrite=context['rewrite_reference'],
        original_schedule_sha256=semantic_sha256(schedule_document(context['original_schedule'])),
        rewritten_schedule_sha256=semantic_sha256(schedule_document(context['rewritten_schedule'])),
        certificate_sha256=semantic_sha256(context['certificate']),
        original_cost_table_sha256=semantic_sha256(activity_cost_document(qualified['original_costs'])),
        replacement_cost_table_sha256=semantic_sha256(activity_cost_document(qualified['replacement_costs'])),
        source_rule=artifact_reference(Path(__file__).resolve(), 'scar.native-copy-choice-source-rule'),
        choice_authority='CANONICAL_COMPONENT_MODEL_ONLY_NOT_PHYSICAL_OR_FULL_CALL_OPTIMUM',
        execution_mode='EXPLICIT_TRIAL', default_execution_decision='NO_OP',
        fallback_policy='EXISTING_PRECOMMIT_GUARD_AND_ORIGINAL_ONCE_NO_ADDED_WRAPPER_RETRY',
        fallback_installed_by_this_wrapper=False, cost_fully_measured=False,
        whole_call_cost_measured=False, performance_measured=False,
        physical_optimality_proven=False, production_execution_authorized=False,
        deployment_authorized=False)


def _attach(artifact, reference):
    # The child files and metadata stay unchanged. StatefulArtifact.report
    # already returns a detached copy; only this in-memory sidecar is added.
    artifact._metadata = {**artifact._metadata, 'copy_choice_receipt': deepcopy(reference)}
    return artifact


def build_copy_choice_trial(choiceprep, out):
    """Compile only the model-selected implementation as an explicit trial."""
    directory = Path(out).resolve()
    _need(not directory.exists(), 'fresh native copy-choice trial output required')
    qualified = _qualified(_reference(choiceprep))
    directory.mkdir(parents=True)
    native = directory/'native'
    context = qualified['context']
    if qualified['choice']['selected_variant'] == 'FUSED':
        artifact = window_order.build_certified_window_fusion(context['rewrite_reference']['uri'], native)
    else:
        recipe = context['recipe']
        artifact = base_stateful.build(read_artifact(recipe['artifacts']['region']), native,
            plan=read_artifact(recipe['artifacts']['stage_plan']))
    _need(artifact.directory.resolve() == native, 'copy choice builder returned a different child directory')
    metadata = _validate_child(qualified, native, _child_metadata(native))
    receipt = _receipt(qualified, native, metadata)
    (directory/'artifact.json').write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False)+'\n')
    read_artifact(qualified['reference'])
    return _attach(artifact, artifact_reference(directory/'artifact.json', SCHEMA))


def load_copy_choice_trial(out):
    """Replay the complete choice and correct child source gate before loading."""
    directory = _directory(out)
    reference = artifact_reference(directory/'artifact.json', SCHEMA)
    receipt = read_artifact(reference)
    _need((receipt.get('schema'), receipt.get('schema_version'), receipt.get('stateful_abi_version'))
          == (SCHEMA, 1, 1) and receipt.get('native_directory') == 'native',
          'versioned trial-only copy-choice wrapper required')
    qualified = _qualified(receipt['choice_preparation'])
    native = directory/'native'
    _need(native.is_dir() and not native.is_symlink(), 'original local native child directory required')
    metadata = _validate_child(qualified, native, _child_metadata(native))
    _need(receipt == _receipt(qualified, native, metadata),
          'native copy-choice receipt differs from full choice/source/child replay')
    # Both loaders independently validate source and embedded ABI identities
    # before calling CDLL; neither path may be selected by wrapper text alone.
    artifact = (window_order.load_certified_window_fusion(native)
        if qualified['choice']['selected_variant'] == 'FUSED' else base_stateful.load(native))
    read_artifact(reference); read_artifact(qualified['reference'])
    return _attach(artifact, reference)


__all__ = ['SCHEMA', 'build_copy_choice_trial', 'load_copy_choice_trial']

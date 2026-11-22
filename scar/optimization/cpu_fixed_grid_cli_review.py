"""Compare fixed-grid byte-profile evidence without changing the entry ranking."""

from pathlib import Path

from scar.analysis.replay_transaction import replay_transaction
from scar.compiler.aten_entry_delivery import BINDING_FILE
from scar.ir.execution_recipe import read_artifact
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.cpu-fixed-grid-cli-review'
_BRIDGE = 'scar.f4-cpu-fixed-grid-cli-effect-bridge'
_DELIVERY = 'scar.cpu-fixed-grid-cli-delivery'
_FAMILY = 'scar.cpu-fixed-grid-source-provider'
_PROVEN_ORDER = 'PROVEN_SCOPED_NORMAL_SUCCESS'
_PROVEN_BRIDGE = 'PROVEN_SCOPED_NORMAL_SUCCESS_SOURCE_CLI_EFFECT_BRIDGE'
_BYTE_VALIDATION = ('exact_stdout', 'exact_stderr', 'exact_exit_status', 'bitwise_final_file',
                    'matched_retained_original')


def _require(condition, detail):
    if not condition:
        raise ValueError('CPU fixed-grid CLI review: ' + detail)


def _option(option, scope):
    plan = option['proof']['emission_plan']
    contract = plan['contract']
    scope_sha = semantic_sha256(scope)
    identity = 'planned:' + semantic_sha256(dict(scope_sha256=scope_sha, plan=plan))
    _require(option['id'] == identity and option['implementation_family'] == 'CATALOG_CPU_ENTRY'
             and plan['provider_family'] == plan['authority']['schema'] == _FAMILY
             and plan['catalog'] == scope['cpu_provider_catalog'] and plan['target'] == scope['target']
             and plan['entry']['uri'] == scope['entry'] and plan['entry']['sha256'] == scope['entry_sha256']
             and plan['argv'] == scope['argv'] and plan['source_f4_program_sha256'] == scope['program_sha256']
             and plan['effect_obligations_sha256'] == scope['entry_obligations_sha256']
             and set(plan['context']['scope_matches']) == {'cwd', 'affinity', 'environment', 'resolved_interpreter'}
             and all(plan['context']['scope_matches'].values())
             and contract['publication'] == dict(output_index=2, output_contract='EXACT_FINAL_F32_FILE_DISTINCT_STDOUT_CONTRACTS'),
             'selected option must be the exact original fixed-grid catalog emission plan')
    return plan, contract, scope_sha


def _bind(option, scope, plan, contract, bridge, delivery):
    main = read_artifact(delivery['main_chain'])
    domain = read_artifact(delivery['domain'])
    old_options = read_artifact(main['variant_discovery'])['options']
    _require(main['domain'] == delivery['domain'] and domain['scope'] == scope
             and main['entry_obligations'] == bridge['entry_obligations']
             and any(row['id'] == option['id'] and row['proof']['emission_plan'] == plan for row in old_options)
             and delivery['retained_catalog_pair'] == scope['cpu_paired_observation']
             and bridge['program']['sha256'] == scope['program_sha256']
             and bridge['entry_obligations']['sha256'] == scope['entry_obligations_sha256']
             and bridge['provider'] == plan['authority'] == delivery['provider']
             and bridge['source_project'] == contract['project'] == delivery['parent_project']
             and bridge['source'] == plan['entry'] == delivery['source']
             and bridge['target'] == plan['target'] == delivery['target']
             and bridge['argv'] == scope['argv'] == delivery['argv'],
             'F4 bridge, catalog plan and source CLI witness refer to different scopes')
    context = plan['context']
    _require(delivery['cpu_ids'] == context['cpu_ids'] == scope['process_affinity_cpu_ids']
             and delivery['working_directory'] == context['cwd'] == scope['working_directory']
             and delivery['environment_sha256'] == scope['environment_sha256']
             and delivery['native_placement'] == contract['body_placement']
             and delivery['steps'] == contract['steps'],
             'source CLI witness changes original launch, trip count or body placement')
    project_ref = delivery['project']
    project = read_artifact(project_ref)
    binding = read_artifact(dict(uri=str(Path(project_ref['uri']).parent / BINDING_FILE),
        schema='scar.aten-source-entry-binding', **project['files'][BINDING_FILE]))
    storage = contract['storage']
    expected_storage = dict(scope=storage['scope'], budget_bytes=scope['memory_budget_bytes'],
                            peak_bytes=storage['peak_tensor_storage_bytes'], excluded=storage['excluded_bytes'])
    _require(project['source_entry_cli']['parent_manifest_sha256'] == contract['project']['sha256']
             and binding['source_contract_sha256'] == bridge['source_contract_sha256']
             and binding['placement'] == delivery['native_placement'] and binding['target'] == scope['target']
             and type(storage['budget_bytes']) is int and type(expected_storage['peak_bytes']) is int
             and 0 < expected_storage['peak_bytes'] <= storage['budget_bytes'] == scope['memory_budget_bytes']
             and delivery['storage'] == binding['storage'] == expected_storage
             and len(delivery['file_inputs']) == len(contract['inputs'])
             and all(native['path'] == original['uri']
                     and all(native[key] == original[key] for key in ('size_bytes', 'sha256'))
                     for native, original in zip(delivery['file_inputs'], contract['inputs'], strict=True)),
             'source CLI witness changes the parent body, source ABI, file inputs or tensor budget')


def derive_cpu_fixed_grid_cli_review(option, scope, bridge_ref, delivery_ref):
    """Bind a normal-success source CLI byte profile to its original catalog option."""
    from scar.analysis.cpu_fixed_grid_cli_effect_bridge import verify_cpu_fixed_grid_cli_effect_bridge
    from scar.compiler.cpu_fixed_grid_cli_delivery import verify_cpu_fixed_grid_cli_delivery

    _require(bridge_ref['schema'] == _BRIDGE and delivery_ref['schema'] == _DELIVERY,
             'scoped source CLI effect and delivery receipts required')
    plan, contract, scope_sha = _option(option, scope)
    with replay_transaction():
        bridge = verify_cpu_fixed_grid_cli_effect_bridge(bridge_ref)
        delivery = verify_cpu_fixed_grid_cli_delivery(delivery_ref)
        _bind(option, scope, plan, contract, bridge, delivery)
    _require(delivery['status'] == 'SCOPED_SOURCE_CLI_VALIDATED'
             and delivery['deployment_authorized'] is False
             and delivery['whole_program_equivalence_proven'] is False
             and all(delivery['validation'].get(key) is True for key in _BYTE_VALIDATION),
             'validation must preserve source stdout, stderr, exit and complete final file bytes')
    order = bridge['normal_completion_order']
    same_order = order['status'] == _PROVEN_ORDER
    value_routes_complete = not bridge['value_route_frontier']
    scoped_effects = bridge['status'] == _PROVEN_BRIDGE and same_order and value_routes_complete
    measured = delivery['source_cli_performance_measured'] is True
    timing = delivery['selection']
    measured_gate = (measured and timing is not None and timing['verdict'] == 'ADVANCE_TO_RELEASE_REVIEW'
                     and delivery['delivery_gate']['verdict'] == 'ADVANCE_TO_RELEASE_REVIEW')
    eligible = scoped_effects and measured_gate
    metadata = delivery['validation'].get('exact_publication_mode')
    return dict(
        schema=SCHEMA, schema_version=1, option_id=option['id'], scope_sha256=scope_sha,
        bridge=bridge_ref, delivery=delivery_ref, source=plan['entry'], argv=plan['argv'],
        native_executable=delivery['native_executable'],
        status='SCOPED_PROFILE_REVIEW_ELIGIBLE' if eligible else 'SCOPED_PROFILE_REVIEW_OPEN',
        scoped_profile_review_eligible=eligible, scoped_release_review_eligible=eligible and metadata is True,
        review_scope='FRESH_OUTPUT_NORMAL_SUCCESS_STDOUT_STDERR_EXIT_AND_FILE_BYTES',
        source_cli=dict(project=delivery['project'], parent_project=contract['project'],
            placement=delivery['native_placement'], storage=delivery['storage'],
            observation=delivery['observation'], validation=delivery['validation'], timing=timing,
            measured_gain_gate_passed=measured_gate, f4_write_before_stdout_proven=same_order,
            f4_value_routes_proven=value_routes_complete,
            publication_modes=delivery.get('publication_modes'), exact_publication_mode=metadata),
        legacy_final_tensor=dict(comparison=delivery['retained_catalog_pair'],
            publication_contract=contract['publication']['output_contract'], applicable_to_source_cli_review=False),
        coverage_frontier=[
            dict(role='SOURCE_EFFECT_ORDER', status=order['status'], evidence='#/normal_completion_order'),
            dict(role='SOURCE_EFFECT_DATA', status='SCOPED_VALUE_ROUTES' if value_routes_complete else 'OPEN',
                 evidence='#/value_route_frontier'),
            dict(role='POSIX_PUBLICATION_MODE',
                 status='SCOPED_MODE_MATCH' if metadata is True else 'OBSERVED_MISMATCH' if metadata is False else 'UNOBSERVED'),
            dict(role='OTHER_FILE_AND_FAILURE_EFFECTS', status='OPEN',
                 blockers=['FULL_FILE_METADATA_AND_OBSERVER_EFFECTS', 'IO_FAILURE_AND_PARTIAL_PUBLICATION']),
            dict(role='STARTUP_EFFECTS', status='OPEN', blockers=['ORIGINAL_IMPORT_AND_EXTERNAL_EFFECT_EQUIVALENCE']),
            dict(role='FORWARD_COST', status='OPEN', blockers=['COMPLETE_HARDWARE_MODEL_PROCESS_LAUNCH_THROUGH_EXIT'])],
        complete_forward_cost_proven=False, original_ranking_changed=False, whole_entry_equivalence_proven=False,
        general_release_authorized=False, deployment_authorized=False, execution_default='NO_OP', target_executed=False)

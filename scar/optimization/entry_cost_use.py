"""Admit an unchanged whole-entry estimate for an explicit requested use.

Declared analysis and qualified target timing are distinct uses. Qualification
replays the existing frozen-policy evaluator and existing native/source reader;
it does not fit a model, adjust a policy, load native code or authorize use of
physical resources. Rejection preserves the point estimate and adds missing
terms to the active planning cost.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import CostEstimate
from scar.ir.record_codec import decode, encode
from scar.optimization.deployment_cost import _digest
from scar.optimization.implementation_costs import CostQuery, query_key
from scar.optimization.joint_planner import _known


SCHEMA = 'scar.entry-cost-use'
EXPLICIT_DECLARED_ANALYSIS = 'EXPLICIT_DECLARED_ANALYSIS'
QUALIFIED_TARGET_TIMING = 'QUALIFIED_TARGET_TIMING'
_ERRORS = (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _reference(value):
    if type(value) is dict:
        return deepcopy(value)
    path = Path(value).resolve()
    schema = json.loads(path.read_bytes())['schema']
    return artifact_reference(path, schema)


def _accepted_profile_refs(entry):
    """Keep the exact offered artifact identity, not only its decoded digest."""
    result = []
    admitted = entry['admitted_profile_sha256']
    for offer in entry['profiles']:
        if not offer['admitted'] or offer['profile_sha256'] != admitted:
            continue
        source = offer['source'] or {}
        if source.get('kind') == 'FACT_REFERENCE':
            fact = source['reference']
            if fact['locator'] == '#':
                reference = deepcopy(fact['artifact'])
            else:
                continue
        elif source.get('kind') == 'PATH':
            reference = _reference(source['path'])
            _need(reference['sha256'] == source['sha256']
                and reference['size_bytes'] == source['bytes'], 'offered profile file changed')
        else:
            # An inline model can remain a declared mathematical input, but
            # cannot acquire a pinned target-timing identity from this gate.
            continue
        if reference not in result:
            result.append(reference)
    return result


def _entry_header(entry):
    _need(type(entry) is dict and
        (entry.get('schema'), entry.get('schema_version')) == ('scar.stateful-entry-cost', 1),
        'existing whole-stateful-entry cost report required')
    cost = decode(CostEstimate, entry['cost'])
    query = entry['query']
    _need(query['scope'] == 'ONE_ORIGINAL_METHOD_CALL_SITE'
        and query['context'] == entry['context']
        and entry['context']['candidate_id'] == entry['candidate_id']
        and cost.context == query_key(CostQuery(query['scope'], query['context'])),
        'entry estimate and its exact original query/context differ')
    return cost


def _replay_entry(entry, profile_refs):
    from scar.optimization.stateful_cpu_choices import _domain, _cost, stateful_cpu_cost_context
    from scar.optimization.stateful_entry_cost import _verify_entry_profile_model

    _need(profile_refs, 'a pinned admitted whole profile is required for qualified timing')
    domains = [row['artifact'] for row in entry['source_refs'] if type(row) is dict
        and type(row.get('artifact')) is dict
        and row['artifact'].get('schema') == 'scar.stateful-cpu-domain']
    _need(domains and all(row == domains[0] for row in domains),
        'one original stateful domain artifact is required')
    domain = _domain(read_artifact(domains[0]))
    context = stateful_cpu_cost_context(domain, entry['candidate_id'])
    _need(_digest(context) == _digest(entry['context'])
        and domain['domain_id'] == entry['domain_id']
        and _digest(domain) == entry['domain_sha256'],
        'entry source/compiler/shape/guard domain differs from cold replay')
    for reference in profile_refs:
        document = read_artifact(reference)
        cost, issues, replayed = _cost(dict(artifact=reference, locator='#'), context)
        _need(cost is not None and not issues and _digest(document) == entry['admitted_profile_sha256']
            and replayed == document, 'accepted profile differs from exact-context cost replay')
        _verify_entry_profile_model(document)
        original = entry['cost']
        _need(cost['tick_seconds'] == 1e-9 and cost['duration_ticks'] == original['estimate_ns']
            and original['basis'] == 'DECLARED_MODEL'
            and original['lower_ns'] is None and original['upper_ns'] is None
            and cost['derivation'] == entry['derivation'],
            'original entry duration, nanosecond unit or derivation changed')
    return dict(domain=domains[0], domain_id=domain['domain_id'],
        context_sha256=_digest(context), candidate_id=entry['candidate_id'])


def _replay_qualification(reference):
    from scar.optimization.stateful_entry_cost import evaluate_stateful_entry_holdout

    saved = read_artifact(reference)
    _need((saved.get('schema'), saved.get('schema_version')) ==
        ('scar.stateful-entry-model-qualification', 1), 'existing frozen-policy qualification receipt required')
    replayed = evaluate_stateful_entry_holdout(saved['model'], saved['holdout'], saved['policy'])
    _need(replayed == saved, 'qualification receipt differs from existing evaluator cold replay')
    return replayed


def _replay_native(entry, qualification, native_reference):
    """Reuse the actual serial stateful source/plan/byte admission path."""
    from scar.compiler.controlled_resources import derive_controlled_resource_contract

    observed = read_artifact(qualification['holdout'])
    if (observed['schema'], observed['schema_version']) == ('scar.real-caller-cost-holdout', 1):
        specialization = observed['specialization']
        expected_native = observed['native_artifact']
    elif (observed['schema'], observed['schema_version']) == ('scar.stateful-method-holdout', 2):
        design = read_artifact(observed['design'])
        specialization = design['request']['specialization']
        expected_native = design['native_artifact']
    else:
        raise ValueError('known original holdout/native protocol required')
    _need(native_reference == expected_native, 'requested native manifest differs from the qualified holdout')
    contract = derive_controlled_resource_contract('STATEFUL_SPECIALIZATION', specialization)
    identity = contract['identity']
    native = read_artifact(native_reference)
    _need(contract['implementation']['native_manifest'] == native_reference
        and identity['candidate_id'] == entry['candidate_id']
        and identity['domain_id'] == entry['domain_id']
        and identity['region_sha256'] == entry['context']['region_sha256'],
        'actual native source, plan or candidate differs from the entry cost')
    profile_backend = read_artifact(identity['domain'])['backend_capabilities']['serial']
    _need(_digest(profile_backend) == entry['context']['backend']['descriptor_sha256']
        and native['compiler'] == profile_backend['compiler']
        and native['provider_runtime'] == profile_backend['provider_runtime']
        and native['implementation_sha256'] == profile_backend['implementation_sha256'],
        'actual native compiler/provider/guard implementation differs from cost applicability')
    result=dict(native_manifest=native_reference, native_build_id=native['build_id'],
        actual_plan=identity['stage_plan'], actual_plan_sha256=identity['stage_plan_sha256'],
        generated_source=contract['implementation']['native_members']['source'],
        source_and_plan_replayed=True, runtime_abi_table_admitted_here=False,
        controlled_contract_sha256=_digest(contract),
        scope='EXACT_RETAINED_NATIVE_SOURCE_PLAN_COMPILER_SHAPE_GUARDS_AND_HOLDOUT_CONTEXT')
    if qualification.get('entry_protocol') is not None:
        from scar.analysis.stateful_entry_protocol import derive_stateful_entry_protocol
        protocol=derive_stateful_entry_protocol(native_reference,
            trial=True,exclusive_inputs=True,selection=None)
        _need(_digest(protocol)==_digest(qualification['entry_protocol'])
            and _digest(protocol)==_digest((entry.get('derivation') or {}).get('entry_protocol')),
            'actual native guarded entry protocol differs from the qualified model')
        result['entry_protocol']=protocol
    return result


def admit_entry_cost_for_use(entry_report, *, use, qualification_ref=None, native_manifest_ref=None,
                             requested_entry_protocol=None):
    """Return original and active CostEstimate wires for the requested use.

    DECLARED analysis preserves the existing cost exactly. Qualified use is
    available only after existing policy/model/holdout and actual native-source
    admission agree. Missing/rejected evidence adds missing terms to the active
    estimate, keeping the original point, basis, context and bounds intact.
    """
    _need(use in (EXPLICIT_DECLARED_ANALYSIS, QUALIFIED_TARGET_TIMING), 'known explicit entry-cost use required')
    original = _entry_header(entry_report)
    frontier, profiles, qualification, native_binding, entry_binding = [], [], None, None, None
    references = dict(qualification=None, native_manifest=None)
    derivation=entry_report.get('derivation') or {}
    canonical=derivation.get('basis')=='MATCHED_CANONICAL_GUARDED_ENTRY_POINT_MODEL'
    protocol_binding=None

    def missing(field, reason):
        frontier.append(dict(field=field, reason=reason))

    try:
        profiles = _accepted_profile_refs(entry_report)
    except _ERRORS as error:
        missing('profile_identity', 'ACCEPTED_PROFILE_REFERENCE_UNAVAILABLE: ' + str(error))
    if use == QUALIFIED_TARGET_TIMING:
        if native_manifest_ref is not None:
            try:
                references['native_manifest'] = _reference(native_manifest_ref)
            except _ERRORS as error:
                missing('native_manifest', 'ACTUAL_NATIVE_MANIFEST_REFERENCE_UNAVAILABLE: ' + str(error))
        if not _known(original):
            missing('entry_cost', 'ORIGINAL_ENTRY_COST_IS_INCOMPLETE')
        try:
            entry_binding = _replay_entry(entry_report, profiles)
        except _ERRORS as error:
            missing('entry_binding', 'ENTRY_CONTEXT_OR_PROFILE_REPLAY_FAILED: ' + str(error))
        if qualification_ref is None:
            missing('qualification', 'QUALIFICATION_REFERENCE_REQUIRED_FOR_TARGET_TIMING')
        else:
            try:
                references['qualification'] = _reference(qualification_ref)
                qualification = _replay_qualification(references['qualification'])
                _need(qualification['original_declared_profile_retained_for_analysis'] in profiles,
                    'qualification model names a different original profile artifact')
                if qualification['status'] != 'HOLDOUT_CONTEXT_QUALIFIED':
                    missing('qualification', 'EXISTING_FROZEN_POLICY_REJECTED_PREDICTION')
                    for reason in qualification['qualification_frontiers']:
                        missing('qualification', reason)
                elif qualification['qualified_profile'] not in profiles:
                    missing('qualification', 'QUALIFIED_PROFILE_DIFFERS_FROM_ACCEPTED_ENTRY_PROFILE')
                _need(qualification['predicted_native_ns'] == original.estimate_ns,
                    'frozen qualified prediction differs from original entry estimate')
            except _ERRORS as error:
                missing('qualification', 'QUALIFICATION_COLD_REPLAY_OR_PROFILE_BINDING_FAILED: ' + str(error))
        if native_manifest_ref is None:
            missing('native_manifest', 'ACTUAL_NATIVE_MANIFEST_REQUIRED_FOR_TARGET_TIMING')
        elif qualification is not None and references['native_manifest'] is not None:
            try:
                native_binding = _replay_native(entry_report, qualification, references['native_manifest'])
            except _ERRORS as error:
                missing('native_binding', 'ACTUAL_NATIVE_SOURCE_PLAN_BINDING_FAILED: ' + str(error))
        else:
            missing('native_binding', 'NATIVE_BINDING_REQUIRES_A_REPLAYED_QUALIFICATION_CONTEXT')
        if canonical or qualification is not None and qualification.get('entry_protocol') is not None:
            protocol=derivation.get('entry_protocol')
            if type(protocol) is not dict:
                missing('entry_protocol','CANONICAL_ENTRY_PROTOCOL_MISSING_FROM_ORIGINAL_DERIVATION')
            if type(requested_entry_protocol) is not dict:
                missing('entry_protocol','EXACT_REQUESTED_ENTRY_PROTOCOL_REQUIRED_FOR_CANONICAL_TIMING')
            elif _digest(requested_entry_protocol)!=_digest(protocol):
                missing('entry_protocol','REQUESTED_ENTRY_PROTOCOL_DIFFERS_FROM_CANONICAL_MODEL')
            if qualification is not None and _digest(qualification.get('entry_protocol'))!=_digest(protocol):
                missing('entry_protocol','HOLDOUT_QUALIFICATION_ENTRY_PROTOCOL_DIFFERS_FROM_CANONICAL_MODEL')
            protocol_binding=dict(model_entry_protocol=deepcopy(protocol),
                requested_entry_protocol=deepcopy(requested_entry_protocol),
                exact_request_matches=type(requested_entry_protocol) is dict
                    and _digest(requested_entry_protocol)==_digest(protocol),
                applies_only_to_exact_entry_protocol=True,compact_or_other_trial_configuration_qualified=False)
    qualified = use == QUALIFIED_TARGET_TIMING and not frontier and _known(original)
    active = original
    if use == QUALIFIED_TARGET_TIMING and not qualified:
        terms = tuple(dict.fromkeys((*original.missing_terms,
            'COST_NOT_QUALIFIED_FOR_REQUESTED_USE', *(row['reason'] for row in frontier))))
        active = replace(original, missing_terms=terms)
    result=dict(schema=SCHEMA, schema_version=1, use=use, requested_use=use,
        entry_report_sha256=_digest(entry_report),
        entry_report_reference=dict(schema=entry_report['schema'], semantic_sha256=_digest(entry_report), locator='#'),
        original_cost=deepcopy(entry_report['cost']), active_cost=encode(active),
        cost_available_for_use=_known(active), timing_qualified=qualified,
        status='QUALIFIED_TARGET_TIMING' if qualified else 'EXPLICIT_DECLARED_ANALYSIS_ONLY'
            if use == EXPLICIT_DECLARED_ANALYSIS else 'UNRESOLVED_QUALIFIED_TARGET_TIMING',
        original_query=deepcopy(entry_report['query']), context_sha256=_digest(entry_report['context']),
        accepted_profile_refs=deepcopy(profiles), accepted_profile_semantic_sha256=entry_report['admitted_profile_sha256'],
        entry_binding=entry_binding, qualification_reference=references['qualification'],
        qualification_replay=qualification, native_manifest_reference=references['native_manifest'],
        native_binding=native_binding, frontier=frontier,
        original_numeric_estimate_and_policy_unchanged=True,
        entry_report_admission='PARENT_STATEFUL_ENTRY_COST_REPLAY_REQUIRED_FOR_DECLARED_ANALYSIS',
        original_resource_obligations_unchanged=True,
        qualification_is_not_physical_resource_or_deployment_authority=True,
        target_samples_fitted=False, runtime_guards_executed=False,
        target_program_executed=False, native_loaded=False, compiled=False,
        execution_authorized=False, deployment_authorized=False, default_backend='NO_OP')
    if canonical or requested_entry_protocol is not None or protocol_binding is not None:
        result.update(requested_entry_protocol=deepcopy(requested_entry_protocol),
            entry_protocol_binding=protocol_binding)
    return result


def verify_entry_cost_use(report, entry_report, *, use, qualification_ref=None, native_manifest_ref=None,
                          requested_entry_protocol=None):
    expected = admit_entry_cost_for_use(entry_report, use=use,
        qualification_ref=qualification_ref, native_manifest_ref=native_manifest_ref,
        requested_entry_protocol=requested_entry_protocol)
    _need(_digest(expected) == _digest(report), 'entry-cost use differs from exact evidence and requested-use replay')
    return expected


__all__ = ['SCHEMA', 'EXPLICIT_DECLARED_ANALYSIS', 'QUALIFIED_TARGET_TIMING',
    'admit_entry_cost_for_use', 'verify_entry_cost_use']

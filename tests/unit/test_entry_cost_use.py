"""A timing-use gate replays qualification and never edits declared inputs."""
from copy import deepcopy
import json

from scar.ir.execution_recipe import artifact_reference
from scar.ir.implementation_plan import CostEstimate
from scar.ir.record_codec import decode, encode
from scar.optimization import entry_cost_use as gate
from scar.optimization import stateful_entry_cost
from scar.optimization.deployment_cost import _digest
from scar.optimization.implementation_costs import CostQuery, query_key
from scar.optimization.joint_planner import _known


def test_rejected_or_forged_qualification_keeps_declared_cost_but_blocks_timing_use(tmp_path, monkeypatch):
    def publish(name, document):
        path = tmp_path/name
        path.write_text(json.dumps(document, sort_keys=True))
        return artifact_reference(path, document['schema'])

    context = dict(candidate_id='declared-metadata-candidate')
    profile = dict(schema='scar.stateful-cpu-cost-profile', schema_version=1, context=context)
    profile_ref = publish('profile.json', profile)
    query = dict(scope='ONE_ORIGINAL_METHOD_CALL_SITE', context=context)
    original = encode(CostEstimate(123, None, None, 'DECLARED_MODEL',
        query_key(CostQuery(query['scope'], context))))
    entry = dict(schema='scar.stateful-entry-cost', schema_version=1,
        candidate_id=context['candidate_id'], context=context, query=query, cost=original,
        admitted_profile_sha256=_digest(profile), profiles=[dict(admitted=True,
            profile_sha256=_digest(profile), source=dict(kind='FACT_REFERENCE',
                reference=dict(artifact=profile_ref, locator='#')))])
    before = deepcopy(entry)
    # The gate test isolates requested-use policy from the independently tested
    # source/profile and native readers. No synthetic row claims observations.
    monkeypatch.setattr(gate, '_replay_entry', lambda entry, profiles:dict(source_fixture=True))
    references = {key:publish(key+'.json', dict(schema='metadata-fixture-'+key))
        for key in ('model', 'holdout', 'policy')}
    rejected = dict(schema='scar.stateful-entry-model-qualification', schema_version=1,
        **references, status='HOLDOUT_PREDICTION_REJECTED', predicted_native_ns=123,
        qualified_profile=None, original_declared_profile_retained_for_analysis=profile_ref,
        qualification_frontiers=['FROZEN_PREDICTION_ERROR_THRESHOLD_EXCEEDED'])
    calls = []

    def replay(model, holdout, policy):
        calls.append((model, holdout, policy))
        return deepcopy(rejected)

    monkeypatch.setattr(stateful_entry_cost, 'evaluate_stateful_entry_holdout', replay)
    analysis = gate.admit_entry_cost_for_use(entry, use=gate.EXPLICIT_DECLARED_ANALYSIS)
    assert analysis['active_cost'] == original and analysis['cost_available_for_use']
    assert not analysis['timing_qualified'] and not calls
    forged = {**rejected, 'status':'HOLDOUT_CONTEXT_QUALIFIED', 'qualified_profile':profile_ref}
    forged_ref = publish('forged-qualification.json', forged)
    denied = gate.admit_entry_cost_for_use(entry, use=gate.QUALIFIED_TARGET_TIMING,
        qualification_ref=forged_ref)
    assert any('differs from existing evaluator cold replay' in row['reason'] for row in denied['frontier'])
    rejected_ref = publish('rejected-qualification.json', rejected)
    denied = gate.admit_entry_cost_for_use(entry, use=gate.QUALIFIED_TARGET_TIMING,
        qualification_ref=rejected_ref)
    active = decode(CostEstimate, denied['active_cost'])
    assert denied['original_cost'] == original and active.estimate_ns == 123
    assert active.basis == 'DECLARED_MODEL' and active.context == original['context']
    assert 'COST_NOT_QUALIFIED_FOR_REQUESTED_USE' in active.missing_terms
    assert not _known(active) and not denied['cost_available_for_use'] and not denied['timing_qualified']
    assert denied['qualification_replay']['qualified_profile'] is None
    assert len(calls) == 2 and entry == before
    assert denied['entry_report_sha256'] == _digest(before)
    assert not denied['deployment_authorized']

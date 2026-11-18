"""Requested protocol is a separate gate from source/holdout/native admission."""
from copy import deepcopy
import json

from scar.ir.execution_recipe import artifact_reference
from scar.ir.implementation_plan import CostEstimate
from scar.ir.record_codec import decode,encode
from scar.optimization import entry_cost_use as gate
from scar.optimization.deployment_cost import _digest
from scar.optimization.implementation_costs import CostQuery,query_key


def test_canonical_timing_requires_exact_requested_protocol_and_preserves_estimate(tmp_path,monkeypatch):
    def publish(name,document):
        path=tmp_path/name;path.write_text(json.dumps(document,sort_keys=True))
        return artifact_reference(path,document['schema'])
    protocol=dict(schema='metadata-entry-protocol',family='ARTIFACT_ADAPTER_GUARD',
        configuration=dict(trial=True,exclusive_inputs=True,selection=None))
    context=dict(candidate_id='metadata-candidate')
    derivation=dict(basis='MATCHED_CANONICAL_GUARDED_ENTRY_POINT_MODEL',entry_protocol=protocol)
    profile=dict(schema='scar.stateful-cpu-cost-profile',schema_version=1,
        context=context,derivation=derivation)
    profile_ref=publish('profile.json',profile)
    query=dict(scope='ONE_ORIGINAL_METHOD_CALL_SITE',context=context)
    original=encode(CostEstimate(123,None,None,'DECLARED_MODEL',
        query_key(CostQuery(query['scope'],context))))
    entry=dict(schema='scar.stateful-entry-cost',schema_version=1,candidate_id=context['candidate_id'],
        context=context,query=query,cost=original,derivation=derivation,
        admitted_profile_sha256=_digest(profile),profiles=[dict(admitted=True,
            profile_sha256=_digest(profile),source=dict(kind='FACT_REFERENCE',
                reference=dict(artifact=profile_ref,locator='#')))])
    qualification_ref=publish('qualification.json',dict(schema='metadata-qualification'))
    native_ref=publish('native.json',dict(schema='metadata-native'))
    # Only the requested-use policy is isolated here. These fixtures are not
    # target observations and do not replace the real readers in production.
    monkeypatch.setattr(gate,'_replay_entry',lambda *a:dict(metadata_fixture=True))
    monkeypatch.setattr(gate,'_replay_qualification',lambda *a:dict(
        status='HOLDOUT_CONTEXT_QUALIFIED',predicted_native_ns=123,qualification_frontiers=[],
        qualified_profile=profile_ref,original_declared_profile_retained_for_analysis=profile_ref,
        entry_protocol=deepcopy(protocol)))
    monkeypatch.setattr(gate,'_replay_native',lambda *a:dict(metadata_fixture=True))
    before=deepcopy(entry)
    arguments=dict(use=gate.QUALIFIED_TARGET_TIMING,
        qualification_ref=qualification_ref,native_manifest_ref=native_ref)
    denied=gate.admit_entry_cost_for_use(entry,**arguments)
    assert not denied['timing_qualified'] and denied['original_cost']==original
    assert decode(CostEstimate,denied['active_cost']).estimate_ns==123
    assert any('EXACT_REQUESTED_ENTRY_PROTOCOL_REQUIRED' in row['reason'] for row in denied['frontier'])
    for wrong in (dict(protocol,family='COMPACT'),
            dict(protocol,configuration=dict(protocol['configuration'],trial=False)),
            dict(protocol,configuration=dict(protocol['configuration'],trial=1))):
        denied=gate.admit_entry_cost_for_use(entry,**arguments,requested_entry_protocol=wrong)
        assert not denied['timing_qualified'] and denied['original_cost']==original
        assert decode(CostEstimate,denied['active_cost']).estimate_ns==123
        assert any('REQUESTED_ENTRY_PROTOCOL_DIFFERS' in row['reason'] for row in denied['frontier'])
    accepted=gate.admit_entry_cost_for_use(entry,**arguments,requested_entry_protocol=protocol)
    assert accepted['timing_qualified'] and accepted['active_cost']==original
    assert gate.verify_entry_cost_use(accepted,entry,**arguments,requested_entry_protocol=protocol)==accepted
    analysis=gate.admit_entry_cost_for_use(entry,use=gate.EXPLICIT_DECLARED_ANALYSIS)
    assert analysis['active_cost']==original and not analysis['timing_qualified']
    assert entry==before and not accepted['deployment_authorized']

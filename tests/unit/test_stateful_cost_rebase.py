"""Small declared component costs move only across verified evidence aliases.

No fixture duration is a hardware observation. The actual domain replay and
cost consumer run over a small retained hardware/region, without target code.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scar.ir.execution_recipe import artifact_reference
from scar.optimization.stateful_cost_rebase import rebase_stateful_cpu_cost,verify_rebased_stateful_cpu_cost,PROOF
from scar.optimization.stateful_cpu_model import _cost, stateful_cpu_cost_context, build_stateful_cpu_domain
from scar.optimization.stateful_cpu_model import stateful_cpu_demand, predict_stateful_cpu_model
from test_cpu_hardware_placement import hardware
from test_stateful_recipe import setup_stateful_recipe


def test_rebase_preserves_declared_cost_and_rejects_context_or_proof_tamper(hardware,tmp_path):
    from scar.analysis.target_graph import build_target_graph
    inventory=json.loads((tmp_path/'inventory.json').read_bytes())
    inventory['platform']=dict(machine='x86_64',system='linux')
    path=tmp_path/'platform.json';path.write_text(json.dumps(inventory))
    hardware=tmp_path/'platform-target';build_target_graph(hardware,inventory=path)
    _,region,_=setup_stateful_recipe(hardware,tmp_path)
    options=dict(input_shapes={'left':[3,5],'right':[3,5]},memory_budget_bytes=4096,max_threads=1)
    source_domain=build_stateful_cpu_domain(region,hardware,**options,out=tmp_path/'source')
    destination=build_stateful_cpu_domain(region,hardware,**options,out=tmp_path/'target-domain')
    source=tmp_path/'source/domain.json';target=tmp_path/'target-domain/domain.json'
    candidate=source_domain['candidates'][0]['id'];demand=stateful_cpu_demand(source_domain,candidate)
    terms={term['key']:term for term in demand['terms']}
    calibration=dict(schema='scar.stateful-resource-calibration',schema_version=1,
        resource_context=demand['resource_context'],target_workload_executed=False,target_variants_ranked=False,
        components=[dict(component=term['component'],parameters=term['parameters'],median_ns=index+1,
            basis='DECLARED_COMPONENT_MODEL',scope='SYNTHETIC_UNIT_MODEL_NOT_HARDWARE_TIMING')
            for index,term in enumerate(terms.values())])
    raw=tmp_path/'calibration.json';raw.write_text(json.dumps(calibration))
    model=predict_stateful_cpu_model(source_domain,candidate,[raw],out=tmp_path/'model')
    profile=Path(model['profile']);original=json.loads(profile.read_bytes())
    expected=stateful_cpu_cost_context(destination,candidate)
    assert _cost(original,expected)[0]is None
    result=rebase_stateful_cpu_cost(profile,source,target,candidate_id=candidate,out=tmp_path/'rebase')
    rebased=json.loads(Path(result['profile']).read_bytes())
    assert rebased['estimate']==original['estimate']and rebased['derivation']==original['derivation']
    assert rebased['context_evidence']==original['context_evidence']
    cost,issues,_=_cost(rebased,expected)
    assert not issues and cost['duration_ticks']==original['estimate']['duration_ticks']
    assert verify_rebased_stateful_cpu_cost(result['profile'],expected_context=expected)['profile']==rebased
    changed=deepcopy(rebased);changed['context']['memory_budget_bytes']+=8
    assert _cost(changed,changed['context'])[0]is None
    proof=json.loads(Path(result['proof']['uri']).read_bytes())
    proof['context_rebindings'][0]['target']['pointer']='#/request'
    wrong=tmp_path/'wrong-proof.json';wrong.write_text(json.dumps(proof))
    changed=deepcopy(rebased);changed['rebase_proof']=artifact_reference(wrong,PROOF)
    assert _cost(changed,expected)[0]is None
    changed=deepcopy(rebased);changed['source_profile']['sha256']='0'*64
    assert _cost(changed,expected)[0]is None
    changed=deepcopy(rebased);changed['source_profile']=changed['rebase_proof']
    assert _cost(changed,expected)[0]is None
    # The legacy consumer accepts explicitly declared extensions; the rebase
    # adapter must reject an unrecognized dependency rather than dropping it.
    facts_path=Path(original['context_evidence'][0]['artifact']['uri'])
    facts=json.loads(facts_path.read_bytes());facts['derivation']['unknown_model_dependency']='unmodeled'
    unsupported=tmp_path/'unknown-facts.json';unsupported.write_text(json.dumps(facts))
    reference=artifact_reference(unsupported,facts['schema'])
    unknown=deepcopy(original);unknown['derivation']['unknown_model_dependency']='unmodeled'
    for fact in (*unknown['context_evidence'],*unknown['estimate']['evidence'],*unknown['derivation']['evidence']):
        fact['artifact']=reference
    unknown_path=tmp_path/'unknown-profile.json';unknown_path.write_text(json.dumps(unknown))
    with pytest.raises(ValueError,match='unsupported primitive-composition'):
        rebase_stateful_cpu_cost(unknown_path,source,target,candidate_id=candidate,out=tmp_path/'unsupported')

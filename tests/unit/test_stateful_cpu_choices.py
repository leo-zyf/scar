"""Finite CPU-domain coverage and exact sourced-model choice, never target trials."""
from copy import deepcopy
import json

import pytest

from test_cpu_hardware_placement import hardware
from test_stateful_recipe import setup_stateful_recipe
from scar.ir.execution_recipe import artifact_reference
from scar.optimization.stateful_cpu_choices import (build_stateful_cpu_domain,
    stateful_cpu_cost_context,plan_stateful_cpu_choices,verify_stateful_cpu_choices,selected_stateful_plan)


def test_every_worker_count_and_missing_noop_cost_prevent_unjustified_selection(hardware,tmp_path,monkeypatch):
    from scar.analysis.target_graph import build_target_graph
    inventory=json.loads((tmp_path/'inventory.json').read_bytes())
    inventory['platform']=dict(machine='x86_64',system='linux')
    path=tmp_path/'explicit-platform.json';path.write_text(json.dumps(inventory))
    hardware=tmp_path/'explicit-platform-target';build_target_graph(hardware,inventory=path)
    _,region,_=setup_stateful_recipe(hardware,tmp_path)
    domain=build_stateful_cpu_domain(region,hardware,input_shapes={'left':[3,5],'right':[3,5]},
        memory_budget_bytes=4096,out=tmp_path/'domain')
    for executor in {r['binding']['executor_id'] for r in domain['candidates']}:
        assert {r['worker_count'] for r in domain['candidates'] if r['binding']['executor_id']==executor}=={1,2,3,4}
    assert len({tuple(r['binding']['cpu_ids']) for r in domain['candidates']})==len(domain['candidates'])
    assert not domain['enumeration']['all_cpu_subsets_enumerated']
    assert len(domain['retained_resource_roles'])==len(domain['all_resources'])
    assert all(r['capability_applicability']['status']=='SUPPORTED_UNDER_GUARDS' and not r['lowering_admitted']
               for r in domain['candidates'])
    assert not domain['hardware_context']['quota_is_worker_cap']
    unknown=plan_stateful_cpu_choices(domain,out=tmp_path/'unknown')
    assert unknown['status']=='UNRESOLVED' and unknown['selected_id'] is None
    assert unknown['decision']=='NO_OP' and len(unknown['variants'])==len(domain['candidates'])+1
    assert verify_stateful_cpu_choices(unknown)==unknown and selected_stateful_plan(unknown) is None
    assert unknown['solver']['global_lower_bound_ticks']==0  # relaxation, not an imputed measured cost
    facts=dict(schema='test.stateful-forward-declarations',contexts={},estimates={},derivations={})
    for row in domain['candidates']+[dict(id='NO_OP')]:
        key=row['id'];facts['contexts'][key]=stateful_cpu_cost_context(domain,key)
        # Declared unit-model data: it is intentionally not hardware timing.
        duration=200 if key=='NO_OP' else 100+10*abs(row['worker_count']-3)+(row['numa_node'] or 0)
        facts['estimates'][key]=dict(duration_ticks=duration,tick_seconds=1e-9)
        facts['derivations'][key]=dict(origin='DECLARED_FORWARD_MODEL',target_variant_trials=False,
            assumptions=['SYNTHETIC_UNIT_MODEL_NOT_REAL_HARDWARE_PERFORMANCE'])
    path=tmp_path/'declarations.json';path.write_text(json.dumps(facts))
    artifact=artifact_reference(path,facts['schema'])
    def ref(pointer): return dict(artifact=artifact,locator=pointer)
    profiles={}
    for key in facts['contexts']:
        profiles[key]=dict(schema='scar.stateful-cpu-cost-profile',schema_version=1,context=facts['contexts'][key],
            context_evidence=[ref('#/contexts/'+key)],
            estimate={**facts['estimates'][key],'evidence':[ref('#/estimates/'+key)]},
            derivation={**facts['derivations'][key],'evidence':[ref('#/derivations/'+key)]})
    baseline=profiles.pop('NO_OP')
    incomplete=plan_stateful_cpu_choices(domain,cost_profiles=profiles,out=tmp_path/'missing-original')
    assert incomplete['status']=='UNRESOLVED' and incomplete['selected_id'] is None
    chosen=plan_stateful_cpu_choices(domain,cost_profiles=profiles,no_op_profile=baseline,out=tmp_path/'complete')
    assert chosen['complete_cost_domain'] and chosen['solver']['gap_ticks']==0
    assert chosen['solver']['solution']['makespan_ticks']==100
    selected=next(row for row in domain['candidates'] if row['id']==chosen['selected_id'])
    assert selected['worker_count']==3 and selected['numa_node']==0
    assert verify_stateful_cpu_choices(chosen)==chosen
    actual_region,plan=selected_stateful_plan(chosen)
    assert actual_region==region and plan['cpu_threads']==3
    assert not chosen['physical_optimality_proven'] and not chosen['target_variants_executed']
    bad=deepcopy(chosen);bad['selected_id']='NO_OP'
    with pytest.raises(ValueError,match='complete replay'): verify_stateful_cpu_choices(bad)
    with monkeypatch.context()as context:
        context.setenv('OMP_DYNAMIC','TRUE')
        with pytest.raises(ValueError,match='retained hardware/source replay'):
            verify_stateful_cpu_choices(chosen)

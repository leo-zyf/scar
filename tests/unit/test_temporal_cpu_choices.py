"""Complete independently sourced costs, never work/N, authorize a model choice."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from test_cpu_hardware_placement import hardware
from scar.compiler.model import ArrayRegion, Expr, SourceRef
from scar.compiler.loop_admission import InvariantLoopProof
from scar.ir.execution_recipe import artifact_reference
from scar.optimization.temporal_cpu_choices import (build_temporal_cpu_domain,
    preview_temporal_cpu_choice, verify_temporal_cpu_preview, temporal_cpu_cost_demand,
    temporal_cpu_cost_context, plan_temporal_cpu_choices, match_temporal_primitive_family)
from scar.optimization.temporal_cpu_choices import plan_selected_temporal_cpu_choice,verify_selected_temporal_cpu_choice


def test_hardware_masks_exact_chunks_unknown_costs_and_sourced_forward_choice(hardware,tmp_path,monkeypatch):
    source=tmp_path/'original.py';source.write_text('def leaf(s,x):\n    return s+x*x\n')
    digest=sha256(source.read_bytes()).hexdigest();key='pycode-v2:'+'1'*64;unit='fixture'
    original=SourceRef(str(source),digest,'leaf',1,0,key,(dict(kind='source_identity',
        method='COMPILE_ONLY_RETAINED_F0',semantic_authority='ORIGINAL_F4_PRODUCTION_PLANS_AND_UNIT_FACTS',code_key=key,unit=unit),))
    body=ArrayRegion('r',original,('s','x'),(Expr('s','input',attrs=dict(name='s')),
        Expr('x','input',attrs=dict(name='x')),Expr('fixed','mul',('x','x')),Expr('next','add',('s','fixed'))),'next')
    def witness(role,**kw):return dict(kind='original_f4',unit=unit,role=role,**kw)
    proof=InvariantLoopProof('invariant-array-loop:fixture',original,('seed','input','count'),body,
        's','count',dict(s='seed',x='input'),'leaf',(
            witness('builtin_range',trip_parameter_index=2),
            witness('lexical_carry_write',outer_parameter_index=0,leaf_formal_index=0),
            witness('invariant_array_leaf_call',outer_parameter_indices=[0,1],leaf_region_id='r',call_inputs=[{},{}]),
            witness('single_shared_invariant_loop',loop_unrolled=False)))
    domain=build_temporal_cpu_domain(proof,hardware,input_shapes=dict(s=[5,3],x=[5,3]),
        trip_count=10,memory_budget_bytes=128,max_threads=2)
    assert len(domain['candidates'])==4 and {row['cpu_threads'] for row in domain['candidates']}=={1,2}
    assert domain['original_no_op_available'] and len(domain['all_resources'])>4
    unknown=plan_temporal_cpu_choices(domain)
    assert unknown['decision']=='NO_OP' and not unknown['complete_cost_domain']
    preview=preview_temporal_cpu_choice(domain,cpu_threads=2)
    assert preview['resource_binding']['cpu_ids']==(0,1) and preview['cpu_threads']==2
    assert [(row['row_begin'],row['row_end']) for row in preview['worker_chunks']]==[(0,3),(3,5)]
    assert preview['logical_work']['arithmetic_operations']['mul']==15
    assert preview['logical_work']['arithmetic_operations']['add']==150
    assert preview['profitability_status']=='UNPROVEN_NO_SOURCED_TIMING_MODEL'
    assert verify_temporal_cpu_preview(proof,hardware,preview)==preview
    bad=deepcopy(preview);bad['worker_chunks'][0]['row_end']=4
    with pytest.raises(ValueError,match='hardware/backend replay'):verify_temporal_cpu_preview(proof,hardware,bad)
    facts=[]
    for index,row in enumerate(domain['candidates']):
        demand=temporal_cpu_cost_demand(domain,row['id'])
        components=[dict(component=term['component'],parameters=term['parameters'],basis='DECLARED_COMPONENT_MODEL',
            median_ns=10 if term['composition']=='SERIAL' else 60 if row['cpu_threads']==1 else
                15 if row['resource_binding']['cpu_ids']==(4,5) else 25) for term in demand['terms']]
        path=tmp_path/f'facts-{index}.json';path.write_text(json.dumps(dict(schema='scar.temporal-resource-calibration',schema_version=1,
            resource_context=demand['resource_context'],components=components,target_workload_executed=False,target_variants_ranked=False)))
        facts.append(path)
    context=temporal_cpu_cost_context(domain,'NO_OP');base=tmp_path/'baseline-facts.json'
    base.write_text(json.dumps(dict(schema='scar.unit-cost-facts',context=context,duration_ns=200)))
    ref=artifact_reference(base,'scar.unit-cost-facts');profile=tmp_path/'original-cost.json'
    profile.write_text(json.dumps(dict(schema='scar.temporal-cpu-cost-profile',schema_version=1,context=context,duration_ns=200,
        context_evidence=[dict(artifact=ref,locator='#/context')],cost_evidence=[dict(artifact=ref,locator='#/duration_ns')],
        target_variants_ranked=False)))
    choice=plan_temporal_cpu_choices(domain,resource_facts=facts,no_op_profile=profile)
    assert choice['complete_cost_domain'] and choice['decision']=='SELECT_MODEL_IMPLEMENTATION_FOR_LOWERING'
    selected=next(row for row in choice['variants'] if row['id']==choice['selected_id'])
    # Four serial components cost 40; the winning second-NUMA chunks cost
    # max(15,15), neither sum(15,15) nor inferred serial-work/2.
    assert selected['cpu_threads']==2 and selected['model']['duration_ns']==55
    assert selected['resource_binding']['cpu_ids']==(4,5)
    planned=plan_selected_temporal_cpu_choice(domain,resource_facts=facts,no_op_profile=profile)
    assert planned['schedule']['resource_binding']['cpu_ids']==(4,5)
    assert planned['schedule']['candidate_id']==selected['id']
    assert planned['schedule']['predicted_complete_call_ns']==55 and planned['schedule']['original_complete_call_ns']==200
    assert planned['schedule']['profitability_status']=='UNPROVEN_NO_SOURCED_TIMING_MODEL'
    assert not planned['schedule']['full_entry_performance_qualified']
    assert verify_selected_temporal_cpu_choice(proof,hardware,planned['schedule'])==planned['schedule']
    forged=deepcopy(planned['schedule']);forged['resource_binding']['cpu_ids']=(0,1)
    with pytest.raises(ValueError,match='full-domain cost/hardware replay'):
        verify_selected_temporal_cpu_choice(proof,hardware,forged)
    partial=plan_temporal_cpu_choices(domain,resource_facts=facts[:-1],no_op_profile=profile)
    assert partial['decision']=='NO_OP' and partial['status']=='UNRESOLVED_COST_DOMAIN'
    assert plan_selected_temporal_cpu_choice(domain,resource_facts=facts[:-1],no_op_profile=profile)['schedule'] is None
    assert match_temporal_primitive_family(domain['body'])['family']=='FIXED_SQUARE_ADD_RECURRENCE'
    reordered=deepcopy(domain['body']);reordered['nodes'][-1]['inputs'].reverse()
    with pytest.raises(ValueError,match='no supported canonical'):
        match_temporal_primitive_family(reordered)
    # A team-size cap must not discard a higher-ID sibling CPU from the
    # executor's mask universe before SMT representatives are constructed.
    from scar.analysis.target_graph import TargetHardwareGraph,build_target_graph
    inventory=deepcopy(TargetHardwareGraph(hardware).inventory)
    for processor in inventory['cpu']['processors']:processor['core_id']%=2
    inventory_path=tmp_path/'smt-inventory.json';inventory_path.write_text(json.dumps(inventory))
    smt=tmp_path/'smt-target';build_target_graph(smt,inventory=inventory_path)
    smt_domain=build_temporal_cpu_domain(proof,smt,input_shapes=dict(s=[5,3],x=[5,3]),
        trip_count=10,memory_budget_bytes=128,max_threads=2)
    sibling=next(row for row in smt_domain['candidates'] if row['resource_binding']['cpu_ids']==(0,2))
    assert sibling['smt_siblings_beyond_first']==1 and sibling['cpu_threads']==2
    # OpenMP controls change the applicability of old component facts even
    # when the four common thread-count variables stay identical.
    monkeypatch.setenv('OMP_DYNAMIC','TRUE')
    changed=build_temporal_cpu_domain(proof,hardware,input_shapes=dict(s=[5,3],x=[5,3]),
        trip_count=10,memory_budget_bytes=128,max_threads=2)
    assert changed['live_constraints']['native_thread_environment']['OMP_DYNAMIC']=='TRUE'
    stale=plan_temporal_cpu_choices(changed,resource_facts=facts)
    assert not any(row['model']['complete'] for row in stale['variants'])

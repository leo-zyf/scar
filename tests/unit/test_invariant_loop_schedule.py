"""Logical dominance and bounded storage never manufacture CPU throughput."""
from copy import deepcopy
from hashlib import sha256

import pytest

from test_cpu_hardware_placement import hardware
from scar.compiler.model import ArrayRegion, Expr, SourceRef
from scar.compiler.invariant_loop_frontend import InvariantLoopProof
from scar.ir.program import target_hardware as H
from scar.ir.record_codec import decode
from scar.optimization.invariant_loop_schedule import plan_invariant_loop_schedule, verify_invariant_loop_schedule


def test_static_work_dominance_retains_hardware_and_unknown_profitability(hardware, tmp_path):
    source=tmp_path/'arbitrary.py'; source.write_text('def body(state, fixed):\n    return state+fixed*fixed\n')
    fingerprint=sha256(source.read_bytes()).hexdigest(); code='pycode-v2:'+'1'*64; unit='fixture-unit'
    identity=dict(kind='source_identity',method='COMPILE_ONLY_RETAINED_F0',
        semantic_authority='ORIGINAL_F4_PRODUCTION_PLANS_AND_UNIT_FACTS',code_key=code,unit=unit)
    body_source=SourceRef(str(source),fingerprint,'body',1,0,code,(identity,))
    body=ArrayRegion('fixture-body',body_source,('state','fixed'),(
        Expr('carry','input',attrs={'name':'state'}), Expr('external','input',attrs={'name':'fixed'}),
        Expr('invariant','mul',('external','external')), Expr('next','add',('carry','invariant'))),'next')
    def witness(role,**kw): return dict(role=role,kind='original_f4',unit=unit,**kw)
    proof=InvariantLoopProof('invariant-array-loop:fixture',body_source,('seed','input','count'),body,
        'state','count',{'state':'seed','fixed':'input'},'body',(
            witness('builtin_range',trip_parameter_index=2),
            witness('lexical_carry_write',outer_parameter_index=0,leaf_formal_index=0),
            witness('invariant_array_leaf_call',outer_parameter_indices=[0,1],leaf_region_id=body.id,call_inputs=[{},{}]),
            witness('single_shared_invariant_loop',loop_unrolled=False)))
    shapes={name:[2,3] for name in body.parameters}
    plan=plan_invariant_loop_schedule(proof,hardware,input_shapes=shapes,trip_count=100,memory_budget_bytes=48)
    assert plan['selected_kind']=='REGISTER_TEMPORAL_FUSION'
    assert plan['status']=='STRUCTURAL_RESOURCE_DOMINANCE_UNDER_GUARDS'
    assert plan['profitability_status']=='UNPROVEN_NO_SOURCED_TIMING_MODEL'
    assert not plan['selected_for_use'] and not plan['physical_optimality_proven']
    assert plan['resource_binding']['cpu_ids']==(0,) and plan['memory_policy']=='UNBOUND'
    assert len(plan['all_resources'])>1 and not plan['source_loop_replacement_authorized']
    variants={row['kind']:row for row in plan['variants']}
    register=variants['REGISTER_TEMPORAL_FUSION']; keep=variants['KEEP_IN_BODY']
    assert register['arithmetic_operations']['mul']==6 and register['arithmetic_operations']['add']==600
    assert keep['arithmetic_operations']['mul']==600
    assert register['logical_array_load_elements']==12 and register['logical_array_store_elements']==6
    assert register['budget_feasible'] and not variants['MATERIALIZE_INVARIANT']['budget_feasible']
    assert all(row['weakly_dominates'] for row in plan['dominance'])
    assert plan['resource_schedule']['schema_version']==2
    assert len(plan['resource_schedule']['tasks'])==3 and not plan['loop_unrolled']
    for task in plan['resource_schedule']['tasks'][:2]:
        assert {str(decode(H.TargetResourceRef,row).resource) for row in task['resource_refs']} == {
            plan['resource_binding']['executor_id'], plan['resource_binding']['memory_domain_id']}
    assert plan['external_input_size_authority']=='LOGICAL_INPUT_PAYLOAD_NOT_STORAGE_SPAN_PROOF'
    assert 'EXACT_C_CONTIGUOUS_INPUTS_AT_RUNTIME' in plan['required_contract']
    assert plan['resource_schedule']['proof']['physical_memory_bound_scope']=='PRIVATE_CONTIGUOUS_OUTPUT_PAYLOAD_ONLY'
    assert verify_invariant_loop_schedule(proof,hardware,plan)==plan
    forged=deepcopy(plan); forged['variants'][-1]['arithmetic_operations']['add']=0
    with pytest.raises(ValueError,match='dependency/hardware replay'):
        verify_invariant_loop_schedule(proof,hardware,forged)
    rejected=plan_invariant_loop_schedule(proof,hardware,input_shapes=shapes,trip_count=100,memory_budget_bytes=8)
    assert rejected['selected_kind']=='NO_OP' and rejected['resource_schedule'] is None
    zero=plan_invariant_loop_schedule(proof,hardware,input_shapes=shapes,trip_count=0,memory_budget_bytes=8)
    assert zero['selected_kind']=='NO_OP' and zero['status']=='ORIGINAL_ZERO_TRIP_ALIAS'
    assert zero['variants'][-1]['arithmetic_operations']['mul']==0
    assert zero['resource_schedule']['tasks'][0]['kind']=='EXPORT'

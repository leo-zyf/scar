"""Resource planning for an independently replayed composite F4 loop proof.

Only generic logical-count and versioned-storage helpers are shared with the
single-callee planner. Composite proofs keep their own type, schema, fusion
replay and complete function-global guard obligations.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict

from scar.compiler.pipeline_loop_frontend import validate_composite_invariant_loop_proof
from scar.compiler.invariant_loop import plan_invariant_loop
from scar.ir.program import target_hardware as H
from scar.ir.record_codec import encode
from scar.optimization.hardware import _target_materials,propose_cpu_allocations,validate_cpu_placement
from scar.optimization.invariant_loop_schedule import _ARITHMETIC,_live,_counts,_work,_dominance,_schedule
from scar.optimization.resource_solver import semantic_sha256


def plan_composite_invariant_loop_schedule(proof,target,*,catalog,input_shapes,trip_count,memory_budget_bytes,cpu_threads=1):
    """Replay actual F4/fusion/global routes and derive one serial placement."""
    proof=validate_composite_invariant_loop_proof(proof,catalog=catalog)
    if type(cpu_threads) is not int or cpu_threads!=1:raise ValueError('composite temporal profile currently requires one CPU')
    if type(trip_count) is not int or not -(2**63)<=trip_count<2**63:raise ValueError('exact signed int64 trip count required')
    if type(memory_budget_bytes) is not int or memory_budget_bytes<8:raise ValueError('explicit incremental payload budget required')
    if type(input_shapes) is not dict or set(input_shapes)!=set(proof.body.parameters):
        raise ValueError('all original fused-body array input shapes required')
    shapes={name:list(shape) for name,shape in input_shapes.items()}
    if any(len(shape)!=2 or any(type(n) is not int or n<=0 for n in shape) for shape in shapes.values()):
        raise ValueError('positive exact rank-two shapes required')
    shape=shapes[proof.carry_parameter]
    if any(value!=shape for value in shapes.values()):raise ValueError('equal-shape elementwise composite inputs required')
    interface=dict(carry_parameter=proof.carry_parameter,
        readonly_parameters=[name for name in proof.body.parameters if name!=proof.carry_parameter],
        original_composite_f4_proof=asdict(proof),runtime_global_routes_required=True)
    cut=plan_invariant_loop(proof.body,carry_parameter=proof.carry_parameter,
        loop_interface=interface,memory_budget_bytes=memory_budget_bytes)
    graph,snapshot,resources,connections=_target_materials(target)
    bindings=propose_cpu_allocations(graph,memory_budget_bytes=memory_budget_bytes,max_threads=1)
    if not bindings:raise ValueError('retained CPU/host-memory allocation unavailable')
    placement=min(bindings,key=lambda row:(row.cpu_ids,row.executor_id,row.memory_domain_id))
    validate_cpu_placement(placement);budget=placement.memory_budget_bytes
    nodes=_live(asdict(proof.body));elements=shape[0]*shape[1];nbytes=elements*8;steps=max(trip_count,0)
    loads=sum(node['op']=='input' for node in nodes)
    variants=[_work('KEEP_IN_BODY',_counts(nodes,elements,steps),elements*steps*loads,
        elements*steps,steps,2 if steps else 0,nbytes)]
    materialized=Counter();preloads=0
    for region in cut['hoisted_regions']:
        live=_live(region);materialized.update(_counts(live,elements));preloads+=elements*sum(n['op']=='input' for n in live)
    residual=_live(cut['residual_region']);materialized.update(_counts(residual,elements,steps))
    if not steps:materialized.clear();preloads=0
    variants.append(_work('MATERIALIZE_INVARIANT',{op:materialized[op] for op in _ARITHMETIC},
        preloads+elements*steps*sum(n['op']=='input' for n in residual),
        elements*(len(cut['hoisted_roots'])+steps) if steps else 0,
        len(cut['hoisted_roots'])+steps if steps else 0,len(cut['hoisted_roots'])+2 if steps else 0,nbytes))
    invariant={node['id'] for node in nodes if node['op'] in _ARITHMETIC
        and proof.carry_parameter not in cut['dependencies'][node['id']]}
    operations={op:sum(elements*(1 if node['id'] in invariant else steps) for node in nodes if node['op']==op)
        if steps else 0 for op in _ARITHMETIC}
    variants.append(_work('REGISTER_TEMPORAL_FUSION',operations,elements*loads if steps else 0,
        elements if steps else 0,1 if steps else 0,1 if steps else 0,nbytes))
    for variant in variants:variant['budget_feasible']=variant['incremental_array_payload_bytes']<=budget
    winner=variants[-1];dominance=[_dominance(winner,other) for other in variants[:-1]]
    selected=('NO_OP' if not steps else 'REGISTER_TEMPORAL_FUSION' if winner['budget_feasible']
        and all(row['weakly_dominates'] for row in dominance) else 'NO_OP')
    schedule=None
    if selected=='REGISTER_TEMPORAL_FUSION' or not steps:
        schedule=_schedule(proof,placement,resources,snapshot,shape,steps,budget,selected,winner)
        schedule['proof'].update(source_proof_kind=proof.profile,
            synthetic_body_is_original_callable=False,root_callee_source=asdict(proof.root_callee_source),
            all_function_global_routes_required=True)
    result=dict(schema='scar.composite-invariant-loop-resource-plan',schema_version=1,
        proof_id=proof.id,proof_sha256=semantic_sha256(asdict(proof)),body_sha256=cut['body_sha256'],
        body_identity=proof.body_identity,root_callee_source=asdict(proof.root_callee_source),
        function_global_guards=[asdict(guard) for guard in proof.nested_callee_guards],
        original_proof_fact_locators=[f'#/origins/{i}' for i,row in enumerate(proof.origins)
            if row.get('role') in ('single_shared_invariant_loop','invariant_array_leaf_call','lexical_carry_write')],
        target_snapshot=encode(snapshot),target_path=str(graph.path),
        all_resources=[encode(H.TargetResourceRef(snapshot,resource.id)) for resource in resources],
        hardware_connections_retained_count=len(connections),resource_binding=asdict(placement),cpu_threads=1,
        input_shapes=shapes,trip_count=trip_count,memory_budget_bytes=memory_budget_bytes,
        effective_incremental_budget_bytes=budget,logical_input_payload_bytes=len(proof.body.parameters)*nbytes,
        input_layout_obligation='EXACT_C_CONTIGUOUS_INPUTS_AT_RUNTIME',
        external_input_size_authority='LOGICAL_INPUT_PAYLOAD_NOT_STORAGE_SPAN_PROOF',
        external_storage_alias_identity_proven=False,memory_policy='UNBOUND',numa_page_placement_proven=False,
        placement_policy='LOWEST_CPU_ID_WITHIN_RETAINED_EXECUTOR_AND_CURRENT_PROCESS_AFFINITY',
        variants=variants,dominance=dominance,selected_kind=selected,
        status='STRUCTURAL_COMPOSITE_RESOURCE_DOMINANCE_UNDER_GUARDS' if selected=='REGISTER_TEMPORAL_FUSION'
            else 'ORIGINAL_ZERO_TRIP_ALIAS' if not steps else 'INCREMENTAL_PAYLOAD_BUDGET_INFEASIBLE',
        profitability_status='UNPROVEN_NO_SOURCED_TIMING_MODEL',duration_seconds=None,
        comparison_scope='THREE_DERIVED_PLACEMENTS_OF_SAME_CERTIFIED_FUSED_SCALAR_DAG_NOT_PYTHON_TIME',
        invariant_nodes=[node['id'] for node in nodes if node['id'] in invariant],
        body_dependencies=cut['dependencies'],resource_schedule=schedule,loop_unrolled=False,
        required_contract=[*cut['Q'],'EXACT_C_CONTIGUOUS_INPUTS_AT_RUNTIME','ALL_ORIGINAL_FUNCTION_GLOBAL_ROUTES_MATCH'],
        runtime_guards_required=True,original_f4_replay_required=True,composition_replay_required=True,
        cost_authority='ORIGINAL_F4_COMPOSITION_AND_DECLARED_SHAPE_TRIP_COUNT',
        model_exclusions=['PYTHON_CALL_AND_GUARD_COST','REGISTER_SPILLS_STACK_AND_MACHINE_INSTRUCTIONS',
            'FINITE_SCANS_AND_ALLOCATION_OVERHEAD','CACHE_NUMA_SHARED_CONTENTION'],
        source_loop_replacement_authorized=False,target_variants_executed=False,
        whole_program_makespan_optimized=False,physical_optimality_proven=False,selected_for_use=False)
    result['id']='composite-invariant-loop-resource-plan:'+semantic_sha256(result)
    return result


def verify_composite_invariant_loop_schedule(proof,target,plan,*,catalog):
    expected=plan_composite_invariant_loop_schedule(proof,target,catalog=catalog,input_shapes=plan['input_shapes'],
        trip_count=plan['trip_count'],memory_budget_bytes=plan['memory_budget_bytes'],cpu_threads=plan['cpu_threads'])
    if semantic_sha256(expected)!=semantic_sha256(plan):raise ValueError('composite source/fusion/hardware schedule replay differs')
    return deepcopy(expected)


__all__=['plan_composite_invariant_loop_schedule','verify_composite_invariant_loop_schedule']

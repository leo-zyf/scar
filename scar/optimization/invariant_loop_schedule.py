"""Forward logical-work placement for an original invariant-loop proof.

The ordered expression DAG and declared loop count determine counts, not time.
The abstract winner is conditional on the original callable/array/effect
contract; it is not a claim about register allocation, DRAM traffic or speed.
"""
from collections import Counter
from dataclasses import asdict
from copy import deepcopy

from scar.compiler.invariant_loop import plan_invariant_loop
from scar.compiler.invariant_loop_frontend import validate_invariant_loop_proof
from scar.ir.execution_recipe import artifact_reference
from scar.ir.program import target_hardware as H
from scar.ir.record_codec import encode
from scar.ir.versioned_schedule import (VersionedResourceSchedule, VersionedLogicalValue,
    VersionedMaterialization, StorageSlot, StorageRegion, VersionedTask, VersionedTaskKind as K,
    schedule_document, validate_versioned_schedule)
from scar.optimization.hardware import _target_materials, propose_cpu_allocations, validate_cpu_placement
from scar.optimization.resource_solver import semantic_sha256

_ARITHMETIC = ('add', 'sub', 'mul', 'div', 'neg')


def _live(region):
    nodes = {node['id']: node for node in region['nodes']}; live = set(); pending = [region['output']]
    while pending:
        key = pending.pop()
        if key in live: continue
        live.add(key); pending.extend(nodes[key]['inputs'])
    return [node for node in region['nodes'] if node['id'] in live]


def _counts(nodes, elements, repeats=1):
    counts = Counter(node['op'] for node in nodes if node['op'] in _ARITHMETIC)
    return {op: elements * repeats * counts[op] for op in _ARITHMETIC}


def _work(kind, operations, loads, stores, kernels, payloads, nbytes):
    return dict(kind=kind, arithmetic_operations=operations, logical_array_load_elements=loads,
        logical_array_store_elements=stores, logical_array_load_bytes=8*loads, logical_array_store_bytes=8*stores,
        full_array_kernel_completions=kernels, fixed_private_array_payloads=payloads,
        incremental_array_payload_bytes=payloads*nbytes, logical_bytes_are_dram_bytes=False,
        machine_instructions_counted=False, guard_and_descriptor_scan_work_counted=False,
        allocator_metadata_thread_stack_and_possible_register_spills_counted=False,
        duration_seconds=None, wall_clock_profitability='UNPROVEN')


def _dominance(left, right):
    dimensions = [(f'arithmetic.{op}', left['arithmetic_operations'][op], right['arithmetic_operations'][op])
        for op in _ARITHMETIC]
    dimensions += [(key, left[key], right[key]) for key in ('logical_array_load_elements',
        'logical_array_store_elements', 'full_array_kernel_completions', 'incremental_array_payload_bytes')]
    return dict(dominated=right['kind'], weakly_dominates=all(a <= b for _, a, b in dimensions),
        strictly_improves=[key for key, a, b in dimensions if a < b],
        worsens=[key for key, a, b in dimensions if a > b],
        scope='EXACT_DECLARED_LOGICAL_WORK_AND_INCREMENTAL_ARRAY_PAYLOAD_ONLY')


def _schedule(proof, placement, resources, snapshot, shape, steps, budget, selected, counts):
    elements=shape[0]*shape[1]; nbytes=8*elements; values=[]; materials=[]; slots=[]; tasks=[]
    refs=tuple(encode(H.TargetResourceRef(snapshot, resource.id)) for resource in resources)
    selected_refs=tuple(encode(H.TargetResourceRef(snapshot, resource.id)) for resource in resources
        if str(resource.id) in (placement.executor_id, placement.memory_domain_id))
    if len(selected_refs)!=2:
        raise ValueError('selected CPU executor and host memory must retain their typed resource references')
    # These are abstract logical contiguous input regions under a runtime
    # layout obligation. shape-product bytes do not bound strided storage
    # spans or prove distinct physical StorageIDs for different parameters.
    for parameter in proof.body.parameters:
        value='input:'+parameter; slot='caller:'+parameter; material='material:'+parameter
        values.append(VersionedLogicalValue(value, attrs=dict(parameter=parameter,
            identity='CALL_SCOPED_INPUT_PORT_NOT_CONTENT_EQUALITY_OR_STORAGE_ALIAS_PROOF')))
        slots.append(StorageSlot(slot, 'caller-host-memory', nbytes, ownership='EXTERNAL'))
        materials.append(VersionedMaterialization(material, value, StorageRegion(slot, 0, nbytes), 0, external_ready=True))
    if steps <= 0:
        tasks.append(VersionedTask('export:original-carry', K.EXPORT,
            reads=('material:'+proof.carry_parameter,), capability='CPU_EXPORT',
            attrs=dict(alias='ORIGINAL_CARRY_WITHOUT_EVALUATING_LOOP_BODY')))
    else:
        values.append(VersionedLogicalValue('result:final', 'compute:temporal',
            source_refs=(asdict(proof.body.source),), attrs=dict(fresh=True, iteration_scope='FINAL_CARRY')))
        slots.append(StorageSlot('output', placement.memory_domain_id, nbytes, allocated_by='allocate:output', resource_refs=selected_refs))
        materials.append(VersionedMaterialization('material:final', 'result:final', StorageRegion('output', 0, nbytes),
            0, ready_task='compute:temporal'))
        tasks.extend((VersionedTask('allocate:output', K.ALLOCATE, allocates=('output',),
                capability='CPU_ALLOCATE', resource_refs=selected_refs),
            VersionedTask('compute:temporal', K.COMPUTE,
                reads=tuple('material:'+name for name in proof.body.parameters), writes=('material:final',),
                depends_on=('allocate:output',), capability='CPU_SCALAR_DAG', resource_refs=selected_refs,
                source_refs=(asdict(proof.body.source),),
                attrs=dict(kind=selected, spatial_shape=list(shape), temporal_loop=dict(trip_count=steps,
                    unrolled=False, invariant_placement='PER_SPATIAL_ELEMENT_BEFORE_TEMPORAL_LOOP',
                    carry_placement='LOCAL_SCALAR_RECURRENCE'), logical_work=counts)),
            VersionedTask('export:final', K.EXPORT, reads=('material:final',),
                depends_on=('compute:temporal',), capability='CPU_EXPORT')))
    external=len(proof.body.parameters)*nbytes
    schedule=VersionedResourceSchedule(proof.id, tuple(values), tuple(materials), tuple(slots), tuple(tasks),
        resource_refs=refs, budget_bytes=external+budget,
        budget_scope='ABSTRACT_LOGICAL_INPUT_SLOTS_PLUS_DECLARED_PRIVATE_OUTPUT_PAYLOAD_NOT_WHOLE_PHYSICAL_MEMORY_BOUND',
        budget_refs=(dict(caller_incremental_budget_bytes=budget, logical_external_input_bytes=external),),
        proof=dict(original_loop=proof.id, loop_unrolled=False, source_loop_replacement_authorized=False,
            source_profile='PURE_EQUAL_SHAPE_ELEMENTWISE_NO_INTERMEDIATE_ESCAPE_UNDER_RUNTIME_Q',
            scalar_liveness_is_not_register_allocation_proof=True,
            external_input_layout_obligation='C_CONTIGUOUS',
            external_input_size_authority='LOGICAL_INPUT_PAYLOAD_NOT_STORAGE_SPAN_PROOF',
            physical_memory_bound_scope='PRIVATE_CONTIGUOUS_OUTPUT_PAYLOAD_ONLY'), executable=False)
    validate_versioned_schedule(schedule)
    return schedule_document(schedule)


def plan_invariant_loop_schedule(proof, target, *, input_shapes, trip_count, memory_budget_bytes, cpu_threads=1):
    """Derive placement from a typed F4 proof and full retained hardware graph.

    ``input_shapes`` is keyed by original LEAF parameter names. The typed
    proof must have been discovered/read through the caller's F4 session;
    serialized proof paths are resolved by that session, not trusted here.
    The private-array budget excludes external inputs, stack and allocator
    metadata. Nonpositive counts preserve the original carry alias.
    """
    proof=validate_invariant_loop_proof(proof)
    if cpu_threads != 1 or type(cpu_threads) is not int:
        raise ValueError('current register-temporal lowering supports an explicit single CPU only')
    if type(trip_count) is not int or not -(2**63) <= trip_count < 2**63:
        raise ValueError('exact signed int64 trip count required')
    if type(memory_budget_bytes) is not int or memory_budget_bytes < 8:
        raise ValueError('explicit incremental array payload budget of at least eight bytes required')
    if type(input_shapes) is not dict or set(input_shapes) != set(proof.body.parameters):
        raise ValueError('original leaf parameter shape map required')
    shapes={name:list(shape) for name, shape in input_shapes.items()}
    if any(len(shape)!=2 or any(type(n) is not int or n <= 0 for n in shape) for shape in shapes.values()):
        raise ValueError('positive rank-two exact input shapes required')
    shape=shapes[proof.carry_parameter]
    if any(value != shape for value in shapes.values()): raise ValueError('all original elementwise inputs require equal shapes')
    interface=dict(carry_parameter=proof.carry_parameter,
        readonly_parameters=[name for name in proof.body.parameters if name != proof.carry_parameter],
        original_f4_proof=asdict(proof))
    cut=plan_invariant_loop(proof.body, carry_parameter=proof.carry_parameter, loop_interface=interface,
        memory_budget_bytes=memory_budget_bytes)
    graph, snapshot, resources, connections=_target_materials(target)
    bindings=propose_cpu_allocations(graph, memory_budget_bytes=memory_budget_bytes, max_threads=1)
    if not bindings: raise ValueError('no retained CPU/host-memory allocation within current affinity')
    placement=min(bindings, key=lambda row:(row.cpu_ids, row.executor_id, row.memory_domain_id))
    validate_cpu_placement(placement)
    effective_budget=placement.memory_budget_bytes
    body=asdict(proof.body); nodes=_live(body); elements=shape[0]*shape[1]; nbytes=8*elements; steps=max(trip_count, 0)
    original=_counts(nodes, elements, steps)
    input_loads=sum(node['op']=='input' for node in nodes)
    variants=[_work('KEEP_IN_BODY', original, elements*steps*input_loads, elements*steps,
        steps, 2 if steps else 0, nbytes)]
    materialized=Counter(); hoisted_loads=0
    for region in cut['hoisted_regions']:
        live=_live(region); materialized.update(_counts(live, elements))
        hoisted_loads+=elements*sum(node['op']=='input' for node in live)
    residual=_live(cut['residual_region']); materialized.update(_counts(residual, elements, steps))
    if not steps: materialized.clear(); hoisted_loads=0
    variants.append(_work('MATERIALIZE_INVARIANT', {op:materialized[op] for op in _ARITHMETIC},
        hoisted_loads+elements*steps*sum(node['op']=='input' for node in residual),
        elements*(len(cut['hoisted_roots'])+steps) if steps else 0,
        len(cut['hoisted_roots'])+steps if steps else 0, len(cut['hoisted_roots'])+2 if steps else 0, nbytes))
    invariants={node['id'] for node in nodes if node['op'] in _ARITHMETIC
        and proof.carry_parameter not in cut['dependencies'][node['id']]}
    register={op:sum(elements*(1 if node['id'] in invariants else steps) for node in nodes if node['op']==op)
        if steps else 0 for op in _ARITHMETIC}
    variants.append(_work('REGISTER_TEMPORAL_FUSION', register, elements*input_loads if steps else 0,
        elements if steps else 0, 1 if steps else 0, 1 if steps else 0, nbytes))
    for variant in variants: variant['budget_feasible']=variant['incremental_array_payload_bytes'] <= effective_budget
    winner=variants[-1]; dominance=[_dominance(winner, other) for other in variants[:-1]]
    selected=('NO_OP' if not steps else 'REGISTER_TEMPORAL_FUSION' if winner['budget_feasible']
        and all(row['weakly_dominates'] for row in dominance) else 'NO_OP')
    schedule=None
    if selected=='REGISTER_TEMPORAL_FUSION' or not steps:
        schedule=_schedule(proof, placement, resources, snapshot, shape, steps, effective_budget, selected, winner)
    result=dict(schema='scar.invariant-loop-resource-plan', schema_version=1,
        proof_id=proof.id, proof_sha256=semantic_sha256(asdict(proof)), body_sha256=cut['body_sha256'],
        target_snapshot=encode(snapshot), target_path=str(graph.path),
        all_resources=[encode(H.TargetResourceRef(snapshot, resource.id)) for resource in resources],
        hardware_connections_retained_count=len(connections),
        input_shapes=shapes, trip_count=trip_count, memory_budget_bytes=memory_budget_bytes,
        effective_incremental_budget_bytes=effective_budget,
        logical_input_payload_bytes=len(proof.body.parameters)*nbytes,
        external_input_size_authority='LOGICAL_INPUT_PAYLOAD_NOT_STORAGE_SPAN_PROOF',
        input_layout_obligation='EXACT_C_CONTIGUOUS_INPUTS_AT_RUNTIME',
        external_storage_alias_identity_proven=False,
        resource_binding=asdict(placement), cpu_threads=1,
        placement_policy='LOWEST_CPU_ID_WITHIN_RETAINED_EXECUTOR_AND_CURRENT_PROCESS_AFFINITY',
        memory_policy='UNBOUND', numa_page_placement_proven=False,
        variants=variants, selected_kind=selected, dominance=dominance,
        status='STRUCTURAL_RESOURCE_DOMINANCE_UNDER_GUARDS' if selected=='REGISTER_TEMPORAL_FUSION' else
            'ORIGINAL_ZERO_TRIP_ALIAS' if not steps else 'INCREMENTAL_PAYLOAD_BUDGET_INFEASIBLE',
        profitability_status='UNPROVEN_NO_SOURCED_TIMING_MODEL', duration_seconds=None,
        invariant_nodes=[node['id'] for node in nodes if node['id'] in invariants],
        body_dependencies=cut['dependencies'], loop_unrolled=False, resource_schedule=schedule,
        required_contract=[*cut['Q'],'EXACT_C_CONTIGUOUS_INPUTS_AT_RUNTIME'], runtime_guards_required=True,
        cost_authority='STATIC_ORIGINAL_EXPR_DEPENDENCIES_DECLARED_SHAPE_AND_TRIP_COUNT',
        model_exclusions=['INPUT_FINITE_SCANS_AND_FP_GUARDS', 'DESCRIPTOR_ABI_AND_ALLOCATION_OVERHEAD',
            'REGISTER_SPILLS_STACK_BYTES_AND_MACHINE_SIMD_INSTRUCTIONS', 'CACHE_NUMA_AND_SHARED_RESOURCE_CONTENTION'],
        original_f4_replay_required=True, source_loop_replacement_authorized=False,
        target_variants_executed=False, whole_program_makespan_optimized=False,
        physical_optimality_proven=False, selected_for_use=False)
    result['id']='invariant-loop-resource-plan:'+semantic_sha256(result)
    return result


def verify_invariant_loop_schedule(proof, target, plan):
    """Replay every source/hardware/count/budget choice; never run a target."""
    expected=plan_invariant_loop_schedule(proof, target, input_shapes=plan['input_shapes'],
        trip_count=plan['trip_count'], memory_budget_bytes=plan['memory_budget_bytes'], cpu_threads=plan['cpu_threads'])
    if semantic_sha256(expected)!=semantic_sha256(plan):
        raise ValueError('invariant loop resource plan differs from original dependency/hardware replay')
    return deepcopy(expected)

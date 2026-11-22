"""Fixed-capacity pools that introduce no ordering beyond the source RAW DAG.

A slot may be reused only when the original RAW graph already places its old
producer and every old consumer strictly before the next producer. Source
indices identify calls; their interval last-use positions are not DAG lifetime
proofs. The bounded exact objective is fixed payload capacity, never runtime.
"""
from copy import deepcopy
from pathlib import Path

from scar.compiler.pipeline_frontend import pipeline_digest
from scar.ir.record_codec import loads
from .pipeline_memory import _pipeline, _metadata
from .resource_solver import semantic_sha256


SCHEMA = 'scar.pipeline-partial-order-memory-plan'
OPTIMAL = 'OPTIMAL_WITHIN_RAW_PARTIAL_ORDER_POOL_MODEL'
INFEASIBLE = 'INFEASIBLE_WITHIN_RAW_PARTIAL_ORDER_POOL_MODEL'


def _raw(pipeline):
    producers = {call.output: index for index, call in enumerate(pipeline.calls)}
    predecessors = [set() for _ in pipeline.calls]
    for index, call in enumerate(pipeline.calls):
        predecessors[index] = {producers[value] for value in call.inputs if value in producers}
        if any(parent >= index for parent in predecessors[index]):
            raise ValueError('original RAW producers must precede their consumers')
    # Reachability is computed once, using original topological indices and
    # Python integer bitsets. No execution order or storage edge participates.
    ancestors = []
    for parents in predecessors:
        mask = 0
        for parent in parents:
            mask |= ancestors[parent] | (1 << parent)
        ancestors.append(mask)
    return predecessors, ancestors


def _before(old, new, ancestors):
    writer = new['producer_index']
    events = [old['producer_index'], *old['consumer_indices']]
    return all(event != writer and bool(ancestors[writer] & (1 << event)) for event in events)


def _interference(items, ancestors):
    conflicts = [0] * len(items)
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            if not (_before(items[left], items[right], ancestors)
                    or _before(items[right], items[left], ancestors)):
                conflicts[left] |= 1 << right
                conflicts[right] |= 1 << left
    return conflicts


def _solve(items, conflicts, budget):
    """Weighted coloring: each color costs max(payload) of its members."""
    order = sorted(range(len(items)),
        key=lambda i: (-conflicts[i].bit_count(), -items[i]['nbytes'], i))
    # Every pair in each constructed clique interferes. Its summed payload is
    # therefore a valid lower bound, without claiming a maximum clique solver.
    lower = 0
    for start in order[:16]:
        clique_mask = 1 << start
        weight = items[start]['nbytes']
        for index in order:
            if index != start and conflicts[index] & clique_mask == clique_mask:
                clique_mask |= 1 << index
                weight += items[index]['nbytes']
        lower = max(lower, weight)
    greedy, initial = [], [-1] * len(items)
    for index in order:
        size = items[index]['nbytes']
        choices = [(max(capacity, size) - capacity, slot)
                   for slot, (capacity, members) in enumerate(greedy)
                   if not members & conflicts[index]]
        choices.append((size, len(greedy)))
        _, slot = min(choices)
        if slot == len(greedy):
            greedy.append((size, 1 << index))
        else:
            capacity, members = greedy[slot]
            greedy[slot] = (max(capacity, size), members | (1 << index))
        initial[index] = slot
    best_slots, best_assignment = tuple(greedy), tuple(initial)
    best = sum(capacity for capacity, _ in best_slots)
    stack, seen, explored = [(0, (), ())], set(), 0
    while stack and explored < budget and best > lower:
        depth, slots, assignment = stack.pop()
        signature = (depth, tuple(sorted(slots)))
        if signature in seen:
            continue
        seen.add(signature)
        explored += 1
        cost = sum(capacity for capacity, _ in slots)
        if max(cost, lower) >= best:
            continue
        if depth == len(order):
            result = [-1] * len(items)
            for index, slot in zip(order, assignment):
                result[index] = slot
            best, best_slots, best_assignment = cost, slots, tuple(result)
            continue
        index = order[depth]
        size = items[index]['nbytes']
        choices = [(max(capacity, size) - capacity, slot)
                   for slot, (capacity, members) in enumerate(slots)
                   if not members & conflicts[index]]
        choices.append((size, len(slots)))
        successors = []
        for _, slot in sorted(choices):
            updated = list(slots)
            if slot == len(slots):
                updated.append((size, 1 << index))
            else:
                capacity, members = slots[slot]
                updated[slot] = (max(capacity, size), members | (1 << index))
            successors.append((depth + 1, tuple(updated), assignment + (slot,)))
        stack.extend(reversed(successors))
    complete = best == lower or not stack
    return best_slots, best_assignment, best if complete else lower, explored, complete


def _reuse_proofs(pipeline, records, values, ancestors):
    metadata = {value['id']: value for value in values}
    proofs = []
    for slot in records:
        members = sorted((metadata[value] for value in slot['values']),
                         key=lambda value: value['producer_index'])
        for index, new in enumerate(members):
            for old in members[:index]:
                if not _before(old, new, ancestors):
                    raise ValueError('slot assignment lacks original strict RAW happens-before')
                events = sorted(set([old['producer_index'], *old['consumer_indices']]))
                proofs.append(dict(slot_id=slot['id'], previous_value=old['id'], next_value=new['id'],
                    old_producer=old['producer_call'],
                    all_old_consumers=[pipeline.calls[i].id for i in old['consumer_indices']],
                    new_producer=new['producer_call'],
                    strict_raw_happens_before_events=[pipeline.calls[i].id for i in events],
                    new_producer_is_not_old_consumer=True,
                    relation_authority='#/raw_predecessors',
                    storage_order_already_implied_by_source_raw=True))
    return proofs


def plan_pipeline_partial_order_memory(pipeline, *, input_shapes, memory_budget_bytes,
                                        search_budget=100000):
    """Minimize the fixed pool under source-RAW-only reuse legality."""
    pipeline = _pipeline(pipeline)
    if memory_budget_bytes is not None and (type(memory_budget_bytes) is not int or memory_budget_bytes < 0):
        raise ValueError('nonnegative incremental payload budget required')
    if type(search_budget) is not int or search_budget < 1:
        raise ValueError('positive coloring search budget required')
    values, calls, declared, frontier = _metadata(pipeline, input_shapes)
    if memory_budget_bytes is None:
        frontier.append(dict(field='memory_budget_bytes', reason='explicit incremental payload budget required'))
    parents, ancestors = _raw(pipeline)
    final = next(value for value in values if value['id'] == pipeline.output)
    for value in values:
        value['lifetime_basis'] = 'ORIGINAL_RAW_PRODUCER_AND_ALL_CONSUMERS'
        value['source_index_last_use_role'] = 'SERIAL_ORDER_DIAGNOSTIC_NOT_PARTIAL_ORDER_LIFETIME_PROOF'
    report = dict(schema=SCHEMA, schema_version=1, pipeline_id=pipeline.id,
        pipeline_sha256=pipeline_digest(pipeline), input_shapes=declared,
        memory_budget_bytes=memory_budget_bytes, search_budget=search_budget,
        status='UNRESOLVED', optimality='NOT_ESTABLISHED', frontier=frontier,
        values=values, calls=calls, assignment=None, workspace_slots=[],
        total_workspace_bytes=None, final_bytes=final['nbytes'], peak_allocation_bytes=None,
        lower_bound_workspace_bytes=None, lower_bound_peak_allocation_bytes=None,
        gap_bytes=None, budget_satisfied=False, states_explored=0, search_budget_exhausted=False,
        objective='MINIMIZE_SUM_OF_FIXED_WORKSPACE_SLOT_CAPACITIES',
        optimality_scope='SOURCE_RAW_PARTIAL_ORDER_PREALLOCATED_HOST_UNBOUND_PAYLOAD_POOL_ONLY',
        memory_policy='UNBOUND', slot_allocation='ENTRY_ONCE_FIXED_CAPACITY_HELD_THROUGH_EXIT',
        logical_release='REUSE_PERMISSION_ONLY_NOT_PHYSICAL_FREE',
        budget_scope='INCREMENTAL_PAYLOAD_WORKSPACE_CAPACITIES_PLUS_FRESH_FINAL',
        budget_excludes=['EXTERNAL_INPUTS', 'ALLOCATOR_METADATA_AND_ALIGNMENT_OVERHEAD',
                         'THREAD_SCRATCH', 'PROCESS_RSS'],
        physical_optimality_proven=False, placement_optimized=False, copy_tasks_inserted=False,
        call_order_changed=False, duration_predicted=False, target_program_executed=False,
        raw_predecessors={call.id: [pipeline.calls[i].id for i in sorted(parents[index])]
                          for index, call in enumerate(pipeline.calls)},
        reachability_algorithm='ORIGINAL_TOPOLOGICAL_CALL_INDEX_ANCESTOR_INTEGER_BITSETS',
        interference_basis='NOT_STRICT_RAW_ORDERED_AFTER_OLD_PRODUCER_AND_ALL_OLD_CONSUMERS',
        reuse_proofs=[], additional_storage_order_reachability=False,
        proof=dict(storage_may_not_create_new_call_happens_before=True,
            current_consumer_cannot_be_overwritten=True, last_use_index_not_legality_authority=True,
            complete_search_means_fixed_pool_capacity_optimum_not_fastest_execution=True,
            hardware_duration_or_physical_residency_inferred=False),
        runtime_guards=['EXACT_NUMPY_NDARRAY_FLOAT64_CPU_2D', 'SOURCE_AND_GLOBAL_CALLEE_IDENTITIES_MATCH',
            'EXACT_MODELED_INPUT_AND_VALUE_SHAPES', 'FRESH_FINAL_DOES_NOT_ALIAS_INPUTS_OR_WORKSPACE',
            'WORKSPACE_AND_FINAL_PAYLOAD_WITHIN_BUDGET'],
        external_assumptions=['SYNCHRONOUS_CPU_EXECUTION', 'NO_FOREIGN_CONCURRENT_INPUT_WRITERS'])
    if frontier:
        return report
    items = [value for value in values if value['ownership'] == 'WORKSPACE_LOGICAL_VALUE']
    conflicts = _interference(items, ancestors)
    slots, indices, lower, explored, complete = _solve(items, conflicts, search_budget)
    assignment = {value: dict(kind='EXTERNAL_INPUT', input_index=index)
                  for index, value in enumerate(pipeline.input_values)}
    assignment[pipeline.output] = dict(kind='FINAL_OUTPUT')
    for value, slot in zip(items, indices):
        assignment[value['id']] = dict(kind='WORKSPACE', slot_id='slot:' + str(slot))
    records = [dict(id='slot:' + str(slot), capacity_bytes=capacity,
        values=[value['id'] for value, assigned in zip(items, indices) if assigned == slot])
        for slot, (capacity, _) in enumerate(slots)]
    workspace = sum(capacity for capacity, _ in slots)
    peak = workspace + final['nbytes']
    proved_infeasible = lower + final['nbytes'] > memory_budget_bytes
    report.update(status=INFEASIBLE if proved_infeasible else OPTIMAL if complete else 'SEARCH_BUDGET_EXHAUSTED',
        optimality='RAW_PARTIAL_ORDER_FIXED_POOL_MODEL_OPTIMUM_ESTABLISHED' if complete else
            'BUDGET_INFEASIBILITY_ESTABLISHED' if proved_infeasible else 'NOT_ESTABLISHED',
        assignment=assignment, workspace_slots=records, total_workspace_bytes=workspace,
        peak_allocation_bytes=peak, lower_bound_workspace_bytes=lower,
        lower_bound_peak_allocation_bytes=lower + final['nbytes'], gap_bytes=workspace - lower,
        budget_satisfied=peak <= memory_budget_bytes, states_explored=explored,
        search_budget_exhausted=not complete,
        reuse_proofs=_reuse_proofs(pipeline, records, values, ancestors))
    return report


def verify_pipeline_partial_order_memory(pipeline, report):
    """Recompute original metadata, RAW closure, coloring, slots and budget."""
    value = loads(Path(report).read_bytes()) if isinstance(report, (str, Path)) else deepcopy(report)
    if (value.get('schema') != SCHEMA or type(value.get('schema_version')) is not int
            or value['schema_version'] != 1):
        raise ValueError('versioned RAW partial-order memory plan required')
    expected = plan_pipeline_partial_order_memory(pipeline, input_shapes=value['input_shapes'],
        memory_budget_bytes=value['memory_budget_bytes'], search_budget=value['search_budget'])
    if semantic_sha256(expected) != semantic_sha256(value):
        raise ValueError('partial-order pool differs from original RAW/shape/interference/search/budget replay')
    return value


__all__ = ['SCHEMA', 'OPTIMAL', 'INFEASIBLE', 'plan_pipeline_partial_order_memory',
           'verify_pipeline_partial_order_memory']

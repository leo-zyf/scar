"""Conditional cost-subject coordinates over an unchanged source time graph.

Groups enclose consecutive source activities. They do not divide collective
costs, place unknown execution work, or instantiate residuals as zero.
"""
from collections import defaultdict

from scar.ir.activity_costs import (CostBasis, CostCoverage, activity_cost_document,
    validate_activity_cost_table)
from scar.ir.record_codec import encode
from scar.ir.temporal_resources import projection_document, projection_from_dict
from scar.ir.versioned_schedule import (TemporalMode, TemporalScopeKind,
    TemporalOccupancyKind as O, TemporalEventKind as E, schedule_document,
    validate_versioned_schedule)
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.activity-time-projection'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def project_activity_time(schedule, resources, table, accounting, *, source_refs=()):
    """Project admitted modern standalone inputs without solving or executing.

    Parent publication readers own source/hardware re-admission. This pure
    adapter validates their typed identities, accounting, and original source
    order; it retains the entire parent graph by digest and original ID.
    """
    validate_versioned_schedule(schedule)
    resources = projection_from_dict(projection_document(resources))
    cost_summary = validate_activity_cost_table(table, schedule, resources)
    temporal = schedule.temporal
    _need(temporal.mode is TemporalMode.SEMANTIC and not any(
        scope.kind is TemporalScopeKind.LOOP for scope in temporal.scopes),
        'one semantic standalone source scope required; loops are not unrolled')
    envelope = schedule.proof.get('implementation_envelope', {})
    _need('implementation_scope_cleanup' in schedule.proof
        and envelope.get('invocation_scope') == 'ONE_STANDALONE_STATEFUL_METHOD_CALL_NOT_REPEAT_ADAPTER'
        and envelope.get('phase_order') == [task.id for task in schedule.tasks],
        'complete original modern standalone fixed source phase order required')
    _need(type(accounting) is dict and (accounting.get('schema'), accounting.get('schema_version'))
          == ('scar.stateful-scope-accounting', 1) and accounting.get('summary') == cost_summary,
          'matching source-backed scope accounting and cost summary required')
    subjects = {row.subject: row for row in table.costs}
    _need(len(accounting['active_subjects']) == len(subjects)
          and set(accounting['active_subjects']) == set(subjects)
          and accounting['numeric_unknown_subjects'] == [r.subject for r in table.costs if r.basis is CostBasis.UNKNOWN],
          'scope accounting must preserve every original final cost subject')
    parent = accounting['scope_schedule']
    _need(accounting['source_certificate'] == dict(artifact=parent, locator='#/proof/implementation_scope_cleanup')
          and any(ref.get('artifact') == parent and ref.get('locator')
                  == '#/proof/implementation_scope_cleanup/milestone' for ref in table.source_refs),
          'original saved scope/source certificate reference required')
    _need(resources.budget.bytes == schedule.budget_bytes and resources.budget.scope == schedule.budget_scope
          and resources.budget.source_refs == schedule.budget_refs,
          'original budget bytes, scope and references cannot be reinterpreted')
    _need(type(source_refs) is tuple and all(type(row) is dict for row in source_refs),
          'one immutable shared parent-reference index required')
    activities = {row.id: row for row in temporal.activities}
    by_task = {row.task: row for row in temporal.activities}
    _need(len(by_task) == len(temporal.activities) and set(by_task) == {t.id for t in schedule.tasks},
          'one synchronous original lifecycle per source task required')
    adjacency = defaultdict(set)
    for constraint in temporal.constraints:
        _need(constraint.iteration_distance == 0, 'recurrence needs a separate shared-loop adapter')
        if constraint.min_delay_ticks >= 0:
            adjacency[constraint.before].add(constraint.after)
    for activity in temporal.activities:
        adjacency[activity.start].add(activity.completion)
    def reaches(before, after):
        pending = [before]; seen = {before}
        while pending:
            point = pending.pop()
            if point == after:
                return True
            for target in adjacency[point] - seen:
                seen.add(target); pending.append(target)
        return False
    ordered = [by_task[task.id] for task in schedule.tasks]
    _need(all(reaches(a.completion, b.start) for a, b in zip(ordered, ordered[1:])),
          'original temporal happens-before must prove the fixed source order')
    subject_of_activity = {name: row.subject for row in table.costs for name in row.activities}
    runs = []; seen_subjects = set()
    for activity in ordered:
        subject = subject_of_activity[activity.id]
        if not runs or runs[-1][0] != subject:
            _need(subject not in seen_subjects, 'noncontiguous collective cost group cannot be split or moved')
            seen_subjects.add(subject); runs.append((subject, []))
        runs[-1][1].append(activity)
    execution = {row.id: row for row in temporal.occupancies if row.kind is O.EXECUTION_DEMAND}
    bindings = {row.occupancy: row for row in resources.executions}
    _need(len(bindings) == len(resources.executions) and set(bindings) == set(execution),
          'every original execution occupancy needs exactly one hardware binding')
    known = {}
    for key, occupancy in execution.items():
        binding = bindings[key]
        _need((binding.resource, binding.units) == (occupancy.resource, occupancy.units),
              'original execution demand and qualified resource binding differ')
        exact = binding.exact_thread
        known[key] = (exact is not None and exact.snapshot == resources.target
            and str(exact.resource) == occupancy.resource and type(occupancy.units) is int and occupancy.units > 0
            and type(temporal.resource_capacities.get(occupancy.resource)) is int
            and temporal.resource_capacities[occupancy.resource] >= occupancy.units)
    storage_occupancies = {row.id: row for row in temporal.occupancies if row.kind is O.PHYSICAL_SLOT_BYTES}
    _need({row.occupancy for row in resources.storages} == set(storage_occupancies)
          and len(resources.storages) == len(storage_occupancies), 'one binding per original physical slot occupancy required')
    slots = {row.id: row for row in schedule.storage_slots}
    for binding in resources.storages:
        occupancy = storage_occupancies[binding.occupancy]
        _need(binding.slot == occupancy.storage_slot and binding.payload_bytes == slots[binding.slot].capacity_bytes,
              'original owned physical slot capacity differs')
    by_activity_occupancy = defaultdict(list)
    for occupancy in execution.values():
        by_activity_occupancy[occupancy.activity].append(occupancy)
    event_ids_by_task = defaultdict(list)
    for event in temporal.events:
        if event.task is not None:
            event_ids_by_task[event.task].append(event)
    groups = []; group_by_task = {}; group_by_event = {}; cursor = 0; known_sum = 0
    for subject, members in runs:
        cost = subjects[subject]; identities = {row.id for row in members}
        _need(identities == set(cost.activities) and len({row.scope for row in members}) == 1,
              'collective subject must retain one complete original body scope')
        identity = 'cost-scope:' + semantic_sha256(dict(schedule=table.source_schedule_sha256, subject=subject))[:32]
        events = []; task_ids = [row.task for row in members]
        for member in members:
            group_by_task[member.task] = identity
            for event in event_ids_by_task[member.task]:
                group_by_event[event.id] = identity
                events.append(dict(event=event.id, kind=event.kind.value, task=member.task,
                    position=('ENTRY_BOUNDARY' if event.id == members[0].start else
                              'EXIT_BOUNDARY' if event.id == members[-1].completion else 'INTERNAL_OFFSET_UNKNOWN')))
        occupancies = [row for member in members for row in by_activity_occupancy[member.id]]
        marker = cost.basis is CostBasis.SOURCE_CONTROL_MILESTONE
        _need(occupancies or marker, 'ordinary work cannot acquire an empty/free resource demand')
        demands = None
        status = 'UNKNOWN_EXECUTION_DEMAND'
        if marker:
            _need(not occupancies, 'source milestone must not reserve execution resources')
            demands = {}; status = 'SOURCE_CONTROL_MILESTONE_NO_EXECUTION_DEMAND'
        elif all(known[row.id] for row in occupancies):
            demands = {}
            for row in occupancies:
                demands[row.resource] = max(demands.get(row.resource, 0), row.units)
            status = 'CONSERVATIVE_KNOWN_EXACT_RESOURCE_MAXIMA'
        estimate = cost.estimate_ticks
        start = cursor
        if estimate is None or cursor is None:
            cursor = None
        else:
            cursor += estimate
        if estimate is not None:
            known_sum += estimate
        residual = None
        if cost.coverage is CostCoverage.COMPONENT_ONLY:
            residual = dict(symbol='delta:' + identity, signed_unknown=True, lower_ticks=None, upper_ticks=None,
                categories=list(cost.missing_terms) or ['COMPONENT_ONLY_TARGET_IMPLEMENTATION_DELTA'],
                additional_work=False, cancellation_assumed=False, set_to_zero=False)
        groups.append(dict(id=identity, subject=subject, activities=[row.id for row in members], task_ids=task_ids,
            scope=members[0].scope, entry_event=members[0].start, exit_event=members[-1].completion,
            events=events, cost=encode(cost), component_model_coordinate=dict(start_ticks=start, end_ticks=cursor),
            residual=residual, resources=dict(status=status, demands=demands,
                occupancies=[row.id for row in occupancies],
                unknown_occupancies=[row.id for row in occupancies if not known[row.id]],
                physical_reservation_proven=False), storage=dict(reserve_slots=[], release_slots=[]), produced_ready=[]))
    by_group = {row['id']: row for row in groups}
    owned = [row for row in schedule.storage_slots if row.ownership == 'SCHEDULE']
    external = [row for row in schedule.storage_slots if row.ownership == 'EXTERNAL']
    memory_slots = []
    for slot in owned:
        _need(slot.allocated_by in group_by_task and (slot.released_by is None or slot.released_by in group_by_task),
              'owned slot lifetime needs original allocation/release task references')
        allocate = group_by_task[slot.allocated_by]
        release = None if slot.released_by is None else group_by_task[slot.released_by]
        by_group[allocate]['storage']['reserve_slots'].append(slot.id)
        if release is not None:
            by_group[release]['storage']['release_slots'].append(slot.id)
        memory_slots.append(dict(slot=slot.id, capacity_bytes=slot.capacity_bytes, reserve_group=allocate,
            release_group=release, retained_through_scope_exit=release is None))
    external_bytes = sum(row.capacity_bytes for row in external)
    live = set(); peak = 0; boundaries = []
    for group in groups:
        reserve = group['storage']['reserve_slots']; release = group['storage']['release_slots']
        _need(not live.intersection(reserve), 'one physical slot cannot be reserved twice')
        before = sum(slots[name].capacity_bytes for name in live); live.update(reserve)
        during = sum(slots[name].capacity_bytes for name in live); peak = max(peak, during)
        _need(set(release) <= live, 'physical slot released before its enclosing reservation')
        live.difference_update(release)
        boundaries.append(dict(group=group['id'], owned_before_bytes=before, owned_during_bytes=during,
            owned_after_bytes=sum(slots[name].capacity_bytes for name in live),
            total_payload_during_bytes=during + external_bytes))
    for material in schedule.materializations:
        if material.ready_task is None:
            continue
        group = by_group[group_by_task[material.ready_task]]
        ready = [event.id for event in temporal.events if event.kind is E.READY and event.materialization == material.id]
        group['produced_ready'].append(dict(materialization=material.id, value=material.value, slot=material.region.slot,
            ready_task=material.ready_task, original_ready_events=ready,
            conservatively_ready_at_event=group['exit_event'], original_ready_time_equality_claimed=False))
    budget_scope = schedule.budget_scope
    supported_budget = budget_scope == 'ALL_REGION_ARRAY_PAYLOAD_EXTERNAL_STATE_PLUS_FIXED_WORKSPACE'
    satisfied = (peak + external_bytes <= schedule.budget_bytes
                 if supported_budget and schedule.budget_bytes is not None else None)
    lifted = []
    for constraint in temporal.constraints:
        before = group_by_event.get(constraint.before); after = group_by_event.get(constraint.after)
        lifted.append(dict(constraint=constraint.id, before_event=constraint.before, after_event=constraint.after,
            before_group=before, after_group=after,
            conservative_before_event=by_group[before]['exit_event'] if before and before != after else constraint.before,
            conservative_after_event=by_group[after]['entry_event'] if after and before != after else constraint.after,
            original_min_delay_ticks=constraint.min_delay_ticks, original_max_delay_ticks=constraint.max_delay_ticks,
            embedding_witness_established=False))
    roots = [row for row in temporal.scopes if row.parent is None]
    _need(len(roots) == 1, 'one original standalone root scope required')
    known_groups = sum(row['resources']['demands'] is not None for row in groups)
    summary = dict(groups=len(groups), activities=len(activities), component_model_sum_ticks=cursor,
        known_component_model_subtotal_ticks=known_sum, physical_scope_latency=None,
        conservative_payload_budget_satisfied=satisfied, known_resource_groups=known_groups,
        unknown_resource_groups=len(groups)-known_groups, known_execution_occupancies=sum(known.values()),
        unknown_execution_occupancies=len(known)-sum(known.values()),
        source_control_milestones=sum(row.basis is CostBasis.SOURCE_CONTROL_MILESTONE for row in table.costs),
        residual_symbols=sum(row['residual'] is not None for row in groups),
        solver_ready=False, execution_authorized=False)
    return dict(schema=SCHEMA, schema_version=1,
        parents=dict(schedule_sha256=table.source_schedule_sha256,
            resource_projection_sha256=table.resource_projection_sha256,
            cost_table_sha256=semantic_sha256(activity_cost_document(table)),
            accounting_sha256=semantic_sha256(accounting), scope_schedule=parent),
        source_refs=encode(source_refs), entry_event=roots[0].entry, exit_event=roots[0].exit,
        groups=groups, constraints=lifted, accounting_frontiers=list(accounting['source_frontiers']),
        references=dict(events=[row.id for row in temporal.events], constraints=[row.id for row in temporal.constraints],
            values=[row.id for row in schedule.values], materializations=[row.id for row in schedule.materializations],
            slots=[row.id for row in schedule.storage_slots], occupancies=[row.id for row in temporal.occupancies]),
        external_ready_materializations=[row.id for row in schedule.materializations if row.external_ready],
        memory=dict(slots=memory_slots, group_boundaries=boundaries, peak_owned_bytes=peak,
            external_slots=[row.id for row in external], external_payload_bytes=external_bytes,
            peak_total_payload_bytes=peak+external_bytes, budget_bytes=schedule.budget_bytes,
            budget_scope=budget_scope, budget_refs=encode(schedule.budget_refs), budget_satisfied=satisfied,
            exact_original_internal_times=False, policy='GROUP_ENTRY_RESERVE_GROUP_EXIT_RELEASE_READY',
            numa_residency_proven=False, process_rss_bound_proven=False),
        summary=summary, tick=dict(numerator=table.tick_numerator, denominator=table.tick_denominator),
        source_order_fixed=True, group_costs_divided=False, physical_scope_latency=None,
        residuals_set_to_zero=False, empirical_ranges_are_guaranteed_bounds=False,
        embedding_witness_established=False, resource_feasibility_proven=False,
        mathematical_schedule_optimality_proven=False, physical_optimality_proven=False,
        solver_invoked=False, solver_ready=False, native_compiled=False, target_program_executed=False,
        deployment_authorized=False)


__all__ = ['SCHEMA', 'project_activity_time']

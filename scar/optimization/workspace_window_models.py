"""Adapt one indexed workspace body to existing conditional window models.

This is a per-iteration scheduling quotient, not a linearized program clock.
The original graph owns all data, recurrence, storage and boundary work. Only
the existing closed-window planner invokes the existing task-only solver.
"""
from copy import deepcopy

from scar.ir.activity_costs import (CostBasis, CostCoverage, activity_cost_document,
    validate_activity_cost_table)
from scar.ir.record_codec import encode
from scar.ir.temporal_resources import OriginalBudget, projection_document, projection_from_dict
from scar.ir.versioned_schedule import (TemporalMode, TemporalScopeKind as S,
    TemporalOccupancyKind as O, schedule_document, validate_versioned_schedule)
from scar.optimization.closed_window_planner import plan_closed_window_models
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.workspace-window-models'
PROJECTION_SCHEMA = 'scar.workspace-local-component-projection'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _project(schedule, resources, table, accounting, windows):
    source = validate_versioned_schedule(schedule)
    resources = projection_from_dict(projection_document(resources))
    summary = validate_activity_cost_table(table, schedule, resources)
    temporal = schedule.temporal
    _need(schedule_document(schedule)['schema_version'] == 4 and temporal.mode is TemporalMode.SEMANTIC
          and not schedule.executable, 'source-qualified non-executable workspace v4 time graph required')
    roots = [row for row in temporal.scopes if row.kind is S.ROOT]
    loops = [row for row in temporal.scopes if row.kind is S.LOOP]
    _need(len(roots) == len(loops) == 1 and len(temporal.scopes) == 2
          and loops[0].parent == roots[0].id and type(loops[0].trip_count) is int
          and loops[0].trip_count > 0, 'one fixed positive shared loop and original root required')
    root, loop = roots[0], loops[0]
    scope = dict(id=loop.id, kind='LOOP', iteration_symbol=loop.iteration_symbol,
        trip_count=loop.trip_count, occurrence_count=loop.trip_count, symbolic_same_i=True)
    digest = semantic_sha256(schedule_document(schedule))
    resource_digest = semantic_sha256(projection_document(resources))
    _need(accounting.get('schema') == 'scar.workspace-cost-accounting'
          and accounting.get('schema_version') == 1 and accounting.get('summary') == summary
          and accounting.get('source_schedule_sha256') == digest
          and accounting.get('resource_projection_sha256') == resource_digest
          and accounting.get('loop_scope') == loop.id and accounting.get('trip_count') == loop.trip_count,
          'one original workspace cost ledger/source/resource/loop context required')
    _need(windows.get('schema') == 'scar.closed-window-projection' and windows.get('schema_version') == 1
          and windows['source']['schedule_sha256'] == digest
          and windows['source']['resources_sha256'] == resource_digest
          and windows.get('scope') == scope,
          'source-qualified same-iteration windows must retain the original workspace context')
    certificate = schedule.proof.get('workspace_time', {})
    _need(certificate.get('schema') == 'scar.workspace-time-projection-certificate'
          and certificate.get('trip_count') == loop.trip_count,
          'original shared workspace time certificate required')
    activities = {row.id: row for row in temporal.activities}
    by_task = {row.task: row for row in temporal.activities}
    tasks = {row.id: row for row in schedule.tasks}
    _need(len(by_task) == len(activities) and set(by_task) == set(tasks),
          'one original synchronous lifecycle per workspace task required')
    ledger = {row['subject']: row for row in accounting['subjects']}
    _need(len(ledger) == len(accounting['subjects']) == len(table.costs)
          and set(ledger) == {row.subject for row in table.costs}, 'complete original cost subject ledger required')
    execution = {row.id: row for row in temporal.occupancies if row.kind is O.EXECUTION_DEMAND}
    bindings = {row.occupancy: row for row in resources.executions}
    _need(len(bindings) == len(resources.executions) and set(bindings) == set(execution),
          'complete source execution occupancy binding coverage required')
    by_activity = {key: [] for key in activities}; known = {}
    for key, occupancy in execution.items():
        binding = bindings[key]
        _need((binding.resource, binding.units) == (occupancy.resource, occupancy.units),
              'source occupancy and qualified resource binding differ')
        by_activity[occupancy.activity].append(occupancy)
        exact = binding.exact_thread; capacity = temporal.resource_capacities.get(occupancy.resource)
        known[key] = (exact is not None and exact.snapshot == resources.target
            and binding.candidate_threads == (exact,) and str(exact.resource) == occupancy.resource
            and type(occupancy.units) is int and occupancy.units > 0
            and type(capacity) is int and capacity >= occupancy.units)
    groups = []
    for cost in table.costs:
        record = ledger[cost.subject]; members = [activities[key] for key in cost.activities]
        containing = members[0].scope; occurrences = loop.trip_count if containing == loop.id else 1
        _need(containing in (root.id, loop.id) and record['scope'] == containing
              and tuple(record['activities']) == cost.activities
              and record['task_ids'] == [row.task for row in members]
              and type(record['occurrence_count']) is int and record['occurrence_count'] == occurrences
              and record['estimate_per_occurrence_ticks'] == cost.estimate_ticks,
              'cost groups must retain original scope, task identity, estimate and occurrence count')
        residual = None
        if cost.coverage is CostCoverage.COMPONENT_ONLY:
            residual = record['residual']
            _need(residual.get('signed_unknown') is True and residual.get('lower_ticks') is None
                  and residual.get('upper_ticks') is None and residual.get('set_to_zero') is False
                  and residual.get('cancellation_assumed') is False
                  and (containing != loop.id or residual.get('across_iteration_equality_assumed') is False),
                  'component residuals must remain unknown and separately indexed by iteration')
        occupancies = [row for member in members for row in by_activity[member.id]]
        demands = None; status = 'UNKNOWN_EXECUTION_DEMAND'
        if cost.basis is CostBasis.SOURCE_CONTROL_MILESTONE:
            _need(not occupancies, 'source control milestone must not reserve execution resources')
            demands = {}; status = 'SOURCE_CONTROL_MILESTONE_NO_EXECUTION_DEMAND'
        elif len(members) == 1 and occupancies and all(known[row.id] for row in occupancies):
            demands = {row.resource: row.units for row in occupancies}
            status = 'SOURCE_QUALIFIED_FIXED_EXACT_RESOURCES'
        groups.append(dict(id='workspace-local-group:' + semantic_sha256(dict(
            schedule=digest, subject=cost.subject))[:32], subject=cost.subject,
            activities=list(cost.activities), scope=containing, occurrence_count=occurrences,
            cost=encode(cost), residual=deepcopy(residual), resources=dict(status=status, demands=demands,
                occupancies=[row.id for row in occupancies],
                unknown_occupancies=[row.id for row in occupancies if not known[row.id]],
                grouped_resource_timeline_disaggregated=False, physical_reservation_proven=False)))
    _need(resources.budget == OriginalBudget(schedule.budget_bytes, schedule.budget_scope, schedule.budget_refs),
          'original full payload budget and references cannot be reinterpreted')
    slots = {row.id: row for row in schedule.storage_slots}
    owned = {row.id: row for row in schedule.storage_slots if row.ownership == 'SCHEDULE'}
    physical = {row.id: row for row in temporal.occupancies if row.kind is O.PHYSICAL_SLOT_BYTES}
    storages = {row.occupancy: row for row in resources.storages}
    _need(len(storages) == len(resources.storages) == len(physical) == len(owned)
          and set(storages) == set(physical)
          and {row.storage_slot for row in physical.values()} == set(owned),
          'one parent-held physical occupancy per original owned slot required')
    for key, occupancy in physical.items():
        binding = storages[key]; slot = owned[occupancy.storage_slot]
        _need(occupancy.scope == root.id and binding.slot == slot.id
              and binding.payload_bytes == occupancy.units == slot.capacity_bytes
              and by_task[slot.allocated_by].scope == by_task[slot.released_by].scope == root.id,
              'workspace storage must retain its original once-only root allocation/release')
    workspace_bytes = sum(row.capacity_bytes for row in owned.values())
    external_bytes = sum(row.capacity_bytes for row in slots.values() if row.ownership == 'EXTERNAL')
    supported_budget = schedule.budget_scope == 'ALL_REGION_ARRAY_PAYLOAD_EXTERNAL_STATE_PLUS_FIXED_WORKSPACE'
    satisfied = workspace_bytes + external_bytes <= schedule.budget_bytes if (
        supported_budget and schedule.budget_bytes is not None) else None
    seen = set(); positions = {row.id: index for index, row in enumerate(schedule.tasks)}
    for window in windows['windows']:
        ids = window['task_ids']; members = set(ids)
        _need(ids and len(ids) == len(members) and not seen.intersection(members)
              and members <= set(tasks) and window.get('scope') == scope
              and all(by_task[key].scope == loop.id for key in ids)
              and ids == sorted(ids, key=positions.__getitem__),
              'windows must be disjoint source-ordered task subsets in one symbolic iteration')
        seen.update(members)
        _need(window['memory_budget'] == encode(resources.budget), 'window must borrow the unchanged full parent budget')
        for borrowed in window['borrowed_slots']:
            slot = borrowed['slot']
            _need(slot['id'] in slots and slot == encode(slots[slot['id']])
                  and borrowed.get('newly_allocated') is False and borrowed.get('lifetime_changed') is False,
                  'window cannot invent, reallocate or release borrowed parent storage')
    return dict(schema=PROJECTION_SCHEMA, schema_version=1,
        parents=dict(schedule_sha256=digest, resource_projection_sha256=resource_digest,
            cost_table_sha256=semantic_sha256(activity_cost_document(table)),
            accounting_sha256=semantic_sha256(accounting)), scope=scope, groups=groups,
        memory=dict(budget_satisfied=satisfied, budget=encode(resources.budget),
            owned_slot_ids=list(owned), parent_held_workspace_bytes=workspace_bytes,
            external_payload_bytes=external_bytes, total_parent_payload_bytes=workspace_bytes + external_bytes,
            parent_allocation_and_release_unchanged=True, new_window_allocations=[],
            physical_memory_available_budget_inferred=False),
        tick_numerator=table.tick_numerator, tick_denominator=table.tick_denominator,
        original_temporal_activity_count=source['temporal']['activities'],
        unknown_parent_services_excluded_from_local_objective_not_priced_zero=True,
        global_linear_timeline_constructed=False, iterations_unrolled=0,
        source_and_hardware_admission_required=True, solver_authorized=False,
        execution_authorized=False, deployment_authorized=False)


def plan_workspace_window_models(schedule, resources, table, accounting, windows, *, search_budget=10000):
    """Reuse the original window solver on one fixed-pool, same-i quotient."""
    projection = _project(schedule, resources, table, accounting, windows)
    report = plan_closed_window_models(schedule, resources, table, projection, windows,
                                      search_budget=search_budget)
    scope = projection['scope']
    for row in report['windows']:
        proof_available = row['source_order_trial_supported']
        row.update(scope=deepcopy(scope), occurrence_count=scope['trip_count'],
            component_model_durations_are_per_iteration=True, source_order_proof_available=proof_available,
            source_order_trial_supported=False, shared_workspace_lowering_required=True,
            normal_sequence_duration_ticks=None)
        for proposal in row['candidate_transforms']:
            proposal['standalone_lowering_entry'] = proposal.pop('lowering_entry')
            proposal['lowering_entry'] = None
            proposal['shared_workspace_lowering_required'] = True
    return dict(**{key: value for key, value in report.items() if key not in ('schema', 'scope')},
        schema=SCHEMA, model_scope=report['scope'], loop_scope=deepcopy(scope),
        component_projection=projection, current_domain='ORIGINAL_EXACT_CPU_BINDINGS_AND_ORIGINAL_FIXED_POOL_ONLY',
        memory_layout_domain_expanded=False, physical_slot_lifetimes_changed=False,
        workspace_executable_trial_windows=0, iterations_unrolled=0,
        normal_sequence_duration_ticks=None, finite_order_is_complete_physical_timeline=False,
        unknown_parent_services_priced_zero=False, global_model_optimum_proven=False)


__all__ = ['SCHEMA', 'PROJECTION_SCHEMA', 'plan_workspace_window_models']

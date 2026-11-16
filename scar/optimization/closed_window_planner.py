"""Finite, conditional component models inside source-closed parent windows.

The task-only solver receives a scheduling quotient, never the parent v3
execution graph. Original data and time facts stay attached to the window;
nominal component optimization does not solve unknown physical residuals.
"""
from scar.ir.activity_costs import CostBasis, validate_activity_cost_table
from scar.ir.versioned_schedule import TemporalConstraintKind, VersionedTaskKind
from scar.optimization.resource_solver import solve_resource_schedule, semantic_sha256
from scar.optimization.schedule import ResourceSchedule, Task, TaskKind, schedule_document

SCHEMA = 'scar.closed-window-plans'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _fusion_proposal(window, tasks, baseline):
    """Propose a source-checkable control rewrite, independently of cost selection."""
    ids = window['task_ids']
    if len(ids) < 2:
        return []
    shapes = [tasks[key].attrs.get('shape') for key in ids]
    shape = shapes[0]
    if (type(shape) not in (list, tuple) or len(shape) != 2
            or any(type(n) is not int or n < 1 for n in shape)
            or any(list(other or ()) != list(shape) for other in shapes)
            or any(tasks[key].kind is not VersionedTaskKind.SNAPSHOT
                or tasks[key].capability != 'CPU_HOST_SNAPSHOT'
                or tasks[key].attrs.get('stage_kind') != 'SNAPSHOT_ROOT' for key in ids)):
        return []
    pairs = {(p['before'], p['after']): p for p in window['pairs']}
    pair_refs = []
    for i, before in enumerate(ids):
        for after in ids[i+1:]:
            pair = pairs.get((before, after))
            if pair is None or pair['status'] != 'PAIRWISE_DATA_INDEPENDENT':
                return []
            proof = pair['proof']
            if (proof['remaining_obligations'] or proof['storage_region_conflicts']
                    or proof['producer_consumer'] or proof['world_or_token_conflict']
                    or proof['non_sequence_dependencies'] or not proof['source_closed_contract']):
                return []
            pair_refs.append(dict(before=before, after=after))
    kind = 'FUSE_IDENTICAL_INDEPENDENT_RAW_COPY_LOOPS'
    return [dict(id='control-fusion:'+semantic_sha256(dict(window=window['id'], tasks=ids,
            kind=kind, domain=list(shape)))[:32], kind=kind, task_ids=ids,
        iteration_domain=list(shape), source_pair_refs=pair_refs,
        transformation=dict(loop_nests_before=len(ids), loop_nests_after=1,
            inner_copy_statements_unchanged=True, arithmetic_changed=False,
            new_workers=0, placement_allocation_and_anchors_unchanged=True),
        applicability='SOURCE_LOOP_SKELETON_AND_ALL_PRIVATE_BUFFER_PROOFS_RECHECKED_AT_LOWERING',
        cost=dict(status='FUSED_COMPONENT_COST_UNMEASURED',
            original_component_sum_ticks=baseline, fused_component_ticks=None,
            original_component_cost_sum_reused=False, physical_profitability_proven=False),
        lowering_entry='scar.codegen.window_order.build_window_fusion_trial',
        validation_required='EXPLICIT_NATIVE_REGION_OUTPUT_AND_VISIBLE_EFFECT_COMPARISON',
        automatic_selected=False, deployment_authorized=False)]


def plan_closed_window_models(schedule, resources, table, time_projection, windows, *, search_budget=10000):
    """Solve only known fixed-placement nominal models; preserve all frontiers."""
    validate_activity_cost_table(table, schedule, resources)
    _need(windows.get('schema') == 'scar.closed-window-projection'
        and windows['source']['schedule_sha256'] == table.source_schedule_sha256
        and windows['source']['resources_sha256'] == table.resource_projection_sha256
        and time_projection['parents']['schedule_sha256'] == table.source_schedule_sha256,
        'one qualified original window/time/cost/resource context required')
    _need(type(search_budget) is int and 1 <= search_budget <= 1000000, 'bounded local search budget required')
    tasks = {row.id: row for row in schedule.tasks}
    activities = {row.task: row for row in schedule.temporal.activities}
    events = {row.id: row for row in schedule.temporal.events}
    costs = {name: row for row in table.costs for name in row.activities}
    groups = {name: row for row in time_projection['groups'] for name in row['activities']}
    results = []
    for window in windows['windows']:
        ids = window['task_ids']; members = set(ids); issues = []; relaxed = set(); proofs = []
        for pair in window['pairs']:
            if pair['status'] != 'PAIRWISE_DATA_INDEPENDENT':
                continue
            proof = pair['proof']
            _need(not proof['remaining_obligations'] and not proof['storage_region_conflicts']
                and not proof['producer_consumer'] and not proof['world_or_token_conflict']
                and not proof['non_sequence_dependencies'] and proof['source_closed_contract'] is not None,
                'pairwise independence must retain every original data/effect proof')
            _need(all(tasks[key].kind is VersionedTaskKind.SNAPSHOT for key in (pair['before'], pair['after'])),
                'current source reordering authority is the reviewed raw snapshot body')
            relaxed.add((pair['before'], pair['after'])); proofs.append(pair)
        durations = {}; placements = {}; capacities = {}; residuals = []; local_tasks = []
        for identity in ids:
            task, activity = tasks[identity], activities[identity]
            cost = costs[activity.id]; group = groups[activity.id]
            if (cost.activities != (activity.id,) or cost.basis is not CostBasis.DECLARED_COMPONENT_MODEL
                    or type(cost.estimate_ticks) is not int):
                issues.append(dict(task=identity, reason='NO_UNSPLIT_SINGLE_ACTIVITY_COMPONENT_COST'))
            else:
                durations[identity] = cost.estimate_ticks
            demand = group['resources']['demands']
            if demand is None or not demand:
                issues.append(dict(task=identity, reason='EXACT_LOCAL_EXECUTION_RESOURCE_DEMAND_UNKNOWN'))
            else:
                placements[identity] = dict(demand)
                for resource, units in demand.items():
                    capacity = schedule.temporal.resource_capacities.get(resource)
                    if type(capacity) is not int or capacity < units:
                        issues.append(dict(task=identity, reason='SOURCE_RESOURCE_CAPACITY_UNKNOWN'))
                    else:
                        capacities[resource] = capacity
            if activity.submission_return is not None:
                issues.append(dict(task=identity, reason='ASYNC_LIFECYCLE_REQUIRES_FULL_TEMPORAL_ADAPTER'))
            residuals.append(dict(task=identity, subject=cost.subject, residual=group['residual'],
                component_refs=group['cost']['component_refs'], conditions=list(cost.conditions)))
            dependencies = tuple(key for key in task.depends_on
                if key in members and (key, identity) not in relaxed)
            local_tasks.append(Task(identity, TaskKind.COMPUTE, depends_on=dependencies,
                attrs=dict(parent_task=identity, parent_activity=activity.id,
                    scheduling_quotient_only=True, parent_data_not_duplicated=True)))
        removed_constraints = []
        for constraint in schedule.temporal.constraints:
            before = events[constraint.before]; after = events[constraint.after]
            if before.task not in members or after.task not in members or before.task == after.task:
                continue
            pair = (before.task, after.task)
            if constraint.iteration_distance:
                issues.append(dict(constraint=constraint.id, reason='RECURRENCE_NOT_PROJECTED'))
            elif pair in relaxed:
                if (constraint.kind is not TemporalConstraintKind.CAUSAL or constraint.min_delay_ticks != 0
                        or constraint.max_delay_ticks is not None
                        or constraint.before != activities[before.task].completion
                        or constraint.after != activities[after.task].start):
                    issues.append(dict(constraint=constraint.id, reason='NON_SEQUENCE_TEMPORAL_CONSTRAINT_CANNOT_BE_RELAXED'))
                else:
                    removed_constraints.append(constraint.id)
            elif (constraint.min_delay_ticks != 0 or constraint.max_delay_ticks is not None
                    or before.task not in next(t.depends_on for t in local_tasks if t.id == after.task)):
                issues.append(dict(constraint=constraint.id, reason='TEMPORAL_RELATION_NOT_REPRESENTED_BY_LOCAL_MODEL'))
        if time_projection['memory']['budget_satisfied'] is not True:
            issues.append(dict(reason='ORIGINAL_FIXED_POOL_BUDGET_NOT_QUALIFIED'))
        if not relaxed:
            issues.append(dict(reason='NO_SOURCE_PROVEN_REORDERABLE_OPERATION_PAIR'))
        model = ResourceSchedule('local-window:'+semantic_sha256(dict(
            parent=table.source_schedule_sha256, window=window['id']))[:32], (), (), tuple(local_tasks),
            call_scope=schedule.call_scope+':'+window['id'],
            proof=dict(parent_schedule_sha256=table.source_schedule_sha256,
                parent_window=window['id'], borrowed_pool_is_not_reallocated=True,
                data_dependencies_and_storage_validated_in_original_parent=True,
                original_task_ids=ids, source_pair_proofs=proofs,
                relaxed_parent_temporal_constraints=removed_constraints), executable=False)
        solver = None
        if not issues:
            solver = solve_resource_schedule(model, task_costs=durations, task_placements=placements,
                resource_capacities=capacities, memory_capacities={}, materialization_sizes={},
                memory_locations={}, search_budget=search_budget,
                model_instantiation=dict(parent_schedule_sha256=table.source_schedule_sha256,
                    window=window['id'], objective_scope='NOMINAL_CANONICAL_COMPONENT_COST_ONLY',
                    physical_residuals_unknown=True))
        baseline = sum(durations.values()) if len(durations) == len(ids) else None
        # A tied point model retains the original order. Unmodeled residuals
        # cannot provide an implicit tie breaker or authorize a transformation.
        decision = ('KEEP_ORIGINAL_ON_COMPONENT_MODEL_TIE' if solver is not None
            and solver.get('status') == 'OPTIMAL_WITHIN_MODEL' and solver['solution']['makespan_ticks'] == baseline
            else 'PROPOSED_COMPONENT_MODEL_ONLY' if solver is not None and solver.get('solution') else 'UNRESOLVED_LOCAL_MODEL')
        results.append(dict(window_id=window['id'], task_ids=ids, source_task_order=ids,
            status=decision, frontier=issues, local_model=schedule_document(model),
            task_component_costs=durations, task_placements=placements, resource_capacities=capacities,
            source_pair_proofs=proofs, relaxed_parent_constraints=removed_constraints,
            fixed_pool=window['memory_budget'], borrowed_slots=[row['slot']['id'] for row in window['borrowed_slots']],
            original_parent_allocation_and_release_unchanged=True,
            component_model_baseline_ticks=baseline, solver=solver, residuals=residuals,
            original_window_frontiers=window['frontiers'],
            new_placement_or_worker_contexts_qualified=False, physical_window_latency=None,
            residuals_set_to_zero=False, whole_scope_optimality_proven=False,
            transformation_authorized=False, execution_authorized=False,
            candidate_transforms=_fusion_proposal(window, tasks, baseline),
            source_order_trial_supported=bool(relaxed)))
    return dict(schema=SCHEMA, schema_version=1, windows=results, search_budget=search_budget,
        source_schedule_sha256=table.source_schedule_sha256,
        parent_resource_frontiers=windows['parent_resource_frontiers'],
        original_parent_events_values_storage_retained=True,
        scope='LOCAL_FIXED_PLACEMENT_CONDITIONAL_COMPONENT_MODELS',
        summary=dict(windows=len(results), local_models_solved=sum(row['solver'] is not None
            and row['solver'].get('solution') is not None for row in results),
            structural_reorder_windows=sum(row['source_order_trial_supported'] for row in results),
            control_fusion_proposals=sum(len(row['candidate_transforms']) for row in results),
            unresolved_windows=sum(bool(row['frontier']) for row in results),
            selected_transforms=0, whole_scope_physical_latency=None,
            physical_optimality_proven=False, deployment_authorized=False),
        global_solver_invoked=False, parent_temporal_graph_dropped=False,
        physical_optimality_proven=False, deployment_authorized=False)


__all__ = ['plan_closed_window_models']

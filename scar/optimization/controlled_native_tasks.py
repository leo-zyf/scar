"""Refine one admitted controlled activation into its fixed native recipe.

This is an analysis projection over existing task/value/storage identities.
It neither changes native order nor allocates logical versions again.  The
enclosing controller remains responsible for cold admission of all parents.
"""
from copy import deepcopy

from scar.analysis.controlled_native_time import bind_controlled_native_time
from scar.ir.implementation_plan import (
    CostEstimate, ImplementationOption, PlanningOperation, PlanningProblem,
    PlanningResource, ResourceObligation, digest, validate_problem,
)
from scar.ir.record_codec import decode, encode
from scar.ir.versioned_schedule import schedule_document
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.controlled-native-task-projection'
SCOPE = 'FIXED_NATIVE_RECIPE_TASKS'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _reference(schema, sha, locator='#'):
    return dict(schema=schema, semantic_sha256=sha, locator=locator)


def _unique(rows):
    result = {}
    for row in rows:
        result.setdefault(semantic_sha256(encode(row)), row)
    return tuple(result.values())


def _parent(problem, contract):
    original = validate_problem(problem)
    _need(contract.get('kind') == 'STATEFUL_SPECIALIZATION',
        'fixed native task projection requires a stateful specialization')
    _need(len(original.operations) == 1 and not original.data_edges
        and not original.activities and not original.timing_constraints,
        'one validated complete controlled activation required')
    operation, = original.operations
    _need(len(operation.options) == 1 and not operation.predecessors,
        'one already selected controlled implementation required')
    mode, = operation.options
    sha = semantic_sha256(contract)
    caller = 'controlled:caller-lane:'+sha
    parameters = mode.parameters
    _need(encode(original.target) == contract['target']
        and parameters.get('analysis_scope') == 'SCAR_CONTROLLED_SYNCHRONOUS_REGION_RESOURCES'
        and parameters.get('controlled_scope_qualified') is True
        and parameters['controlled_resource_contract']['semantic_sha256'] == sha
        and parameters['controlled_resource_contract']['parent_admission'] == contract['parent_admission']
        and parameters['controlled_execution'] == contract['execution']
        and parameters['controlled_memory'] == contract['memory'],
        'controlled problem and resource contract identities or scopes differ')
    source = parameters['source_region']; context = source['complete_entry_context']
    identity = contract['identity']; native = contract['execution']['native']
    _need(source['candidate_id'] == identity['candidate_id']
        and context['candidate_id'] == identity['candidate_id']
        and context['region_sha256'] == identity['region_sha256']
        and native['cpu_threads'] == 1 and native['creates_worker_threads'] is False
        and native['asynchronous'] is False,
        'same source candidate and one synchronous native worker required')
    _need(mode.demands == {caller:1} and not mode.workspace_bytes
        and not mode.input_locations and not mode.output_locations,
        'controlled caller participation and one scope-held payload required')
    resources = {row.id:row for row in original.resources}
    _need(caller in resources and resources[caller].kind == 'EXECUTION'
        and resources[caller].capacity == 1
        and all(row.kind == 'MEMORY' for row in original.resources if row.id != caller),
        'hardware inventory cannot substitute for controlled lane capacity')
    memory = contract['memory']; budget = memory['budget_bytes']
    pool = 'controlled:payload-budget:'+sha
    expected = {} if not budget else {pool:memory['included_payload_bytes']}
    _need(original.reserved_memory == expected
        and {row.id for row in original.resources if row.kind == 'MEMORY'} ==
            (set() if not budget else {pool})
        and (not budget or resources[pool].capacity == budget),
        'original scoped payload must be reserved exactly once')
    return original, mode


def project_controlled_native_tasks(controlled_problem, contract, correspondence,
        stage_context, *, cost_groups=()):
    """Return ``(PlanningProblem, receipt)`` without executing native code.

    Costs are supplied after their producer's independent verification.  A
    group is one joint cost, never a distribution over its member tasks.
    Retained missing_terms continue to prevent the existing planner's _known
    predicate from treating a conditional observation as a qualified duration.
    """
    original, parent_mode = _parent(controlled_problem, contract)
    schedule, time_binding = bind_controlled_native_time(stage_context, contract, correspondence)
    timed_document = schedule_document(schedule)
    context_sha = semantic_sha256(stage_context)
    contract_sha = semantic_sha256(contract)
    parent_sha = digest(original)
    parent_ref = _reference('scar.implementation-planning-problem', parent_sha)
    contract_ref = _reference(contract['schema'], contract_sha)
    context_ref = _reference(stage_context['schema'], context_sha)
    schedule_ref = _reference('scar.resource-schedule',
        semantic_sha256(timed_document))
    task_ids = [task.id for task in schedule.tasks]
    positions = {name:index for index,name in enumerate(task_ids)}
    task_bindings = {row['task']:row for row in time_binding['task_bindings']}
    _need(len(task_bindings) == len(task_ids) and set(task_bindings) == set(task_ids),
        'one controlled execution binding for each original task required')
    # This adapter has no loop expansion or numeric event-offset arithmetic.
    # Preserve exact event records in the receipt; refuse a stronger profile.
    temporal = schedule.temporal
    _need(temporal.mode.value == 'SEMANTIC' and len(temporal.scopes) == 1,
        'one semantic synchronous ROOT scope required')
    scope, = temporal.scopes
    _need(scope.kind.value == 'ROOT' and scope.id == schedule.call_scope
        and scope.parent is None and scope.iteration_symbol is None
        and scope.trip_count is None and scope.trip_count_symbol is None
        and not scope.content_carries and not scope.iteration_local_values
        and not scope.iteration_local_materializations
        and all(e.scope == scope.id and e.timestamp_ticks is None for e in temporal.events)
        and all(a.scope == scope.id for a in temporal.activities)
        and all(o.scope == scope.id for o in temporal.occupancies),
        'iteration, timestamps or non-current scope need an explicit refinement')
    _need(all(c.min_delay_ticks == 0 and c.max_delay_ticks in (None, 0)
        and c.iteration_distance == 0 and c.loop_scope is None
        and c.before_iteration is None and c.after_iteration is None
        for c in schedule.temporal.constraints),
        'nonzero or repeated temporal constraints need an explicit refinement')
    _need(all(a.duration_lower_ticks is None and a.duration_upper_ticks is None
        for a in schedule.temporal.activities),
        'existing timed durations need an explicit cost refinement')

    groups = deepcopy(list(cost_groups))
    group_by_task = {}; group_ids = set()
    for group in groups:
        _need(type(group) is dict and type(group.get('id')) is str and group['id']
            and group['id'] not in group_ids, 'unique nonempty cost group identity required')
        names = group.get('task_ids')
        _need(type(names) is list and names and all(type(name) is str and name in positions for name in names)
            and len(set(names)) == len(names), 'cost group must name distinct original tasks')
        start = positions[names[0]]
        _need(names == task_ids[start:start+len(names)],
            'joint costs require contiguous tasks in original native order')
        _need(not set(names).intersection(group_by_task), 'cost groups must not overlap')
        _need(type(group.get('source_refs')) is list and group['source_refs']
            and all(type(ref) is dict and ref for ref in group['source_refs']),
            'joint task cost must retain its source references')
        group['cost'] = encode(decode(CostEstimate, group['cost']))
        group_ids.add(group['id'])
        for name in names:
            group_by_task[name] = group

    partitions = []; task_to_operation = {}
    for task in schedule.tasks:
        if task.id in task_to_operation:
            continue
        group = group_by_task.get(task.id)
        names = [task.id] if group is None else group['task_ids']
        op_id = ('controlled-native-task:' if group is None else 'controlled-native-group:')+semantic_sha256(
            dict(parent=parent_sha, contract=contract_sha, task_ids=names,
                cost_group=None if group is None else group['id']))
        partitions.append((op_id, names, group))
        task_to_operation.update({name:op_id for name in names})

    dependencies = []
    def edge(before, after, kind, reference):
        _need(before in positions and after in positions,
            'native dependency must name original tasks')
        _need(positions[before] <= positions[after],
            'native task projection cannot reverse original recipe order')
        left, right = task_to_operation[before], task_to_operation[after]
        dependencies.append(dict(before=before, after=after, kind=kind, source_ref=reference,
            before_operation=left, after_operation=right,
            disposition='INTERNAL_TO_JOINT_OPERATION' if left == right else 'PREDECESSOR'))

    for index, task in enumerate(schedule.tasks):
        for parent in task.depends_on:
            edge(parent, task.id, 'ORIGINAL_TASK_DEPENDENCY',
                _reference(stage_context['schema'], context_sha,
                    '#/versioned_schedule/tasks/'+str(index)+'/depends_on'))
    for index, order in enumerate(schedule.order_constraints):
        edge(order.before, order.after, order.kind,
            _reference(stage_context['schema'], context_sha,
                '#/versioned_schedule/order_constraints/'+str(index)))
    for index, relation in enumerate(stage_context['dependency_edges']):
        edge(relation['before'], relation['after'], relation['kind'],
            _reference(stage_context['schema'], context_sha, '#/dependency_edges/'+str(index)))
    # Readiness is causal even when it is represented by an event rather than
    # an explicit task.depends_on. Logical values incur no new byte leases.
    for index, material in enumerate(schedule.materializations):
        if material.ready_task is not None:
            for task in schedule.tasks:
                if material.id in task.reads:
                    edge(material.ready_task, task.id, 'MATERIALIZATION_READY_BEFORE_CONSUMPTION',
                        _reference(stage_context['schema'], context_sha,
                            '#/versioned_schedule/materializations/'+str(index)))

    # Each original semantic event must map to an actual task endpoint or the
    # enclosing activation boundary. A READY alias is not a new duration.
    materials = {row.id:row for row in schedule.materializations}
    endpoints = {}
    for event in temporal.events:
        kind = event.kind.value
        if kind == 'SCOPE_ENTRY' and event.id == scope.entry:
            endpoint = (None, 'ROOT_START')
        elif kind == 'SCOPE_EXIT' and event.id == scope.exit:
            endpoint = (None, 'ROOT_END')
        elif kind in ('START', 'COMPLETION') and event.task in positions:
            endpoint = (event.task, 'START' if kind == 'START' else 'END')
        elif kind == 'READY' and event.materialization in materials:
            material = materials[event.materialization]
            _need(event.task == material.ready_task,
                'readiness event differs from original material producer')
            endpoint = ((None, 'ROOT_START') if material.external_ready else (material.ready_task, 'END'))
        else:
            raise ValueError('unsupported original temporal event needs an explicit refinement')
        endpoints[event.id] = endpoint
    temporal_bindings = []
    for index, constraint in enumerate(temporal.constraints):
        before, after = endpoints[constraint.before], endpoints[constraint.after]
        if before == after:
            disposition = 'SAME_ORIGINAL_ENDPOINT'
        elif constraint.max_delay_ticks is not None:
            raise ValueError('finite cross-endpoint delay cannot be dropped during group contraction')
        elif before == (None, 'ROOT_START') or after == (None, 'ROOT_END'):
            _need(before != (None, 'ROOT_END') and after != (None, 'ROOT_START'),
                'root boundary constraint reverses the native activation')
            disposition = 'ENCLOSING_ACTIVATION_BOUNDARY'
        elif before[0] == after[0] and before[1:] == ('START',) and after[1:] == ('END',):
            disposition = 'NONNEGATIVE_TASK_DURATION'
        else:
            _need(before[1] == 'END' and after[1] == 'START'
                and before[0] in positions and after[0] in positions,
                'temporal relation is not a synchronous task predecessor')
            edge(before[0], after[0], 'ORIGINAL_TEMPORAL_PREDECESSOR',
                _reference('scar.resource-schedule', schedule_ref['semantic_sha256'],
                    '#/temporal/constraints/'+str(index)))
            disposition = ('INTERNAL_TO_JOINT_OPERATION' if task_to_operation[before[0]] ==
                task_to_operation[after[0]] else 'PREDECESSOR')
        temporal_bindings.append(dict(constraint=constraint.id, disposition=disposition))

    operation_order = {row[0]:i for i,row in enumerate(partitions)}
    predecessor_sets = {row[0]:set() for row in partitions}
    for relation in dependencies:
        if relation['disposition'] == 'PREDECESSOR':
            predecessor_sets[relation['after_operation']].add(relation['before_operation'])
    resources = list(original.resources)
    native_lane = time_binding['resources']['native_lane']
    if native_lane is not None:
        _need(native_lane['id'] not in {row.id for row in resources},
            'native lane conflicts with a controlled parent resource')
        resources.append(PlanningResource(native_lane['id'], 'EXECUTION', native_lane['capacity'],
            tuple(deepcopy(native_lane['source_refs']))))
    operations = []; mappings = []; frontier = []
    for op_id, names, group in partitions:
        refs = tuple(_reference(stage_context['schema'], context_sha,
            '#/versioned_schedule/tasks/'+str(positions[name])) for name in names)
        demands = {}; obligations = list(parent_mode.unresolved_resources)
        for name in names:
            binding = task_bindings[name]
            for resource, units in binding['execution_demands'].items():
                demands[resource] = max(demands.get(resource, 0), units)
            obligations.extend(decode(ResourceObligation, row) for row in binding['unresolved_resources'])
        obligations = _unique(obligations)
        if group is None:
            cost = CostEstimate(None, None, None, 'UNKNOWN',
                'fixed-native-task:'+semantic_sha256(dict(context=context_sha, tasks=names)),
                refs, ('NO_SOURCED_FIXED_NATIVE_TASK_DURATION',))
        else:
            cost = decode(CostEstimate, group['cost'])
            refs += tuple(deepcopy(group['source_refs']))
        option = ImplementationOption(op_id+':fixed', 'EXISTING_FIXED_NATIVE_RECIPE_ANALYSIS',
            (parent_ref, contract_ref, *refs), demands, {}, {}, {}, cost,
            dict(analysis_scope=SCOPE, parent_problem=parent_ref, timed_schedule=schedule_ref,
                task_ids=list(names), cost_group=None if group is None else group['id'],
                joint_cost_charged_once=group is not None, per_task_duration_inferred=False,
                original_native_order_preserved=True, reordering_authorized=False,
                lowering_authorized=False, deployment_authorized=False), obligations)
        operations.append(PlanningOperation(op_id, refs,
            tuple(sorted(predecessor_sets[op_id], key=operation_order.__getitem__)), (option,), option.id))
        mappings.append(dict(operation=op_id, option=option.id, task_ids=list(names),
            cost_group=None if group is None else group['id']))
        if cost.estimate_ns is None or cost.basis == 'UNKNOWN' or cost.missing_terms:
            frontier.append(dict(field='operations/'+op_id+'/cost', task_ids=list(names),
                reason='UNRESOLVED_FIXED_NATIVE_DURATION', missing_terms=list(cost.missing_terms),
                point_estimate_retained=cost.estimate_ns, source_refs=encode(cost.source_refs)))
        for obligation in obligations:
            frontier.append(dict(operation=op_id, **encode(obligation)))

    result = validate_problem(PlanningProblem(original.target,
        original.source_refs+(contract_ref, context_ref, parent_ref), tuple(resources), tuple(operations), (),
        normal_completion_only=original.normal_completion_only,
        assumptions=original.assumptions+('FIXED_NATIVE_RECIPE_TASKS_ONLY_NOT_WHOLE_GUARDED_ENTRY',
            'ORIGINAL_TASK_AND_STORAGE_ORDER_RETAINED_NO_REORDERING_PERMISSION',
            'SCOPE_HELD_PAYLOAD_RESERVED_ONCE_NOT_PER_LOGICAL_VERSION',
            'CONTROLLED_LANE_PARTICIPATION_IS_NOT_TOTAL_CPU_DEMAND'),
        reserved_memory=deepcopy(original.reserved_memory)))
    # Helper/guard/FFI residuals are outside this smaller clock boundary,
    # never implicit zero or a leftover obtained by subtracting medians.
    frontier.append(dict(field='whole_entry_boundary',
        reason='GUARDS_DISPATCH_FFI_HELPERS_AND_UNTIMED_TASK_SERVICES_NOT_CLOSED_BY_GROUP_COSTS',
        scope=SCOPE, cost_ns=None, source_refs=[parent_ref, contract_ref]))
    report = dict(schema=SCHEMA, schema_version=1, planning_scope=SCOPE,
        controlled_problem_sha256=parent_sha, contract_sha256=contract_sha,
        correspondence_sha256=semantic_sha256(correspondence), stage_context_sha256=context_sha,
        projected_problem_sha256=digest(result), cost_groups_sha256=semantic_sha256(groups),
        time_binding=time_binding, timed_schedule=timed_document,
        task_to_operation=task_to_operation, operation_bindings=mappings,
        dependency_bindings=dependencies, temporal_constraint_bindings=temporal_bindings,
        original_graph_indexes=dict(values='#/timed_schedule/values',
            materializations='#/timed_schedule/materializations', storage_slots='#/timed_schedule/storage_slots',
            tasks='#/timed_schedule/tasks', temporal='#/timed_schedule/temporal',
            source_actions=dict(semantic_sha256=semantic_sha256(correspondence), locator='#/stages'),
            source_expressions=dict(semantic_sha256=semantic_sha256(correspondence), locator='#/expression_stages')),
        memory_accounting=dict(reserved_memory=deepcopy(original.reserved_memory),
            included_payload_bytes=contract['memory']['included_payload_bytes'],
            budget_scope=contract['memory']['budget_scope'], charged_once=True,
            materialization_bytes_added=0, logical_lifetime_releases_payload=False,
            private_pool_lifetime='NATIVE_ENTRY_THROUGH_NATIVE_RETURN',
            physical_reservation_proven=False, budget_excludes=deepcopy(contract['memory']['budget_excludes'])),
        parent_whole_entry_cost=dict(reference=_reference('scar.implementation-planning-problem',
            parent_sha, '#/operations/0/options/0/cost'), cost_sha256=semantic_sha256(encode(parent_mode.cost)),
            charged_in_task_problem=False, divided_among_tasks=False),
        coverage=dict(original_tasks=len(task_ids), projected_operations=len(operations),
            cost_groups=len(groups), task_dependencies=sum(len(task.depends_on) for task in schedule.tasks),
            original_order_constraints=len(schedule.order_constraints),
            native_dependency_edges=len(stage_context['dependency_edges']),
            dependency_witnesses=len(dependencies),
            contracted_internal_witnesses=sum(row['disposition']=='INTERNAL_TO_JOINT_OPERATION' for row in dependencies),
            projected_predecessors=sum(len(op.predecessors) for op in operations),
            materializations=len(schedule.materializations), storage_slots=len(schedule.storage_slots)),
        frontier=frontier, data_edges_created=False, logical_bytes_are_dram_traffic=False,
        temporal_event_records_retained=True, numeric_internal_group_offsets_inferred=False,
        root_entry_exit='ENCLOSING_ACTIVATION_BOUNDARIES_NOT_EXTRA_TIMED_OPERATIONS',
        input_materials_require_controller_cold_admission=True,
        original_native_order_preserved=True, original_whole_entry_problem_unchanged=True,
        reordering_authorized=False, lowering_authorized=False, deployment_authorized=False,
        physical_feasibility_proven=False, physical_cpu_reservation=False,
        full_call_cost_closed=False, timing_qualified=False,
        target_program_executed=False, native_loaded=False, compiled=False)
    report['id'] = 'controlled-native-tasks:'+semantic_sha256(report)
    return result, report


def verify_controlled_native_tasks(controlled_problem, contract, correspondence,
        stage_context, problem, report, *, cost_groups=()):
    """Recompute both projections and reject changed input/output receipts."""
    expected, expected_report = project_controlled_native_tasks(controlled_problem,
        contract, correspondence, stage_context, cost_groups=cost_groups)
    _need(digest(validate_problem(problem)) == digest(expected) and report == expected_report,
        'controlled native task projection differs from complete metadata replay')
    return expected


__all__ = ['project_controlled_native_tasks', 'verify_controlled_native_tasks']

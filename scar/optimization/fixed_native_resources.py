"""Check lane capacity and storage lifetime in the admitted serial native order."""
from collections import defaultdict

from scar.ir.implementation_plan import digest, validate_problem
from scar.ir.versioned_schedule import schedule_from_dict
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.controlled-native-resource-certificate'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _peak(changes):
    occupancy = peak = 0
    for point in sorted(changes):
        occupancy += changes[point]
        _require(occupancy >= 0, 'fixed recipe released an unoccupied resource')
        peak = max(peak, occupancy)
    _require(occupancy == 0, 'fixed recipe retains resources after returning')
    return peak


def _ordered_events(schedule):
    model = schedule.temporal
    scope, = model.scopes
    positions = {task.id: 2 * index for index, task in enumerate(schedule.tasks)}
    materials = {row.id: row for row in schedule.materializations}
    points = {scope.entry: -1, scope.exit: 2 * len(positions)}
    for event in model.events:
        kind = event.kind.value
        _require(event.timestamp_ticks is None and event.scope == scope.id,
            'fixed order witness requires one untimed synchronous scope')
        if kind in ('START', 'COMPLETION'):
            points[event.id] = positions[event.task] + (kind == 'COMPLETION')
        elif kind == 'READY':
            material = materials[event.materialization]
            points[event.id] = -1 if material.external_ready else positions[material.ready_task] + 1
        else:
            _require(event.id in points, 'unsupported fixed native event endpoint')
    for relation in model.constraints:
        start = points[relation.before]
        end = points[relation.after]
        _require(relation.min_delay_ticks == 0 and not relation.iteration_distance
            and relation.loop_scope is None and start <= end
            and (relation.max_delay_ticks is None or relation.max_delay_ticks == 0 and start == end),
            'original fixed native order does not realize its semantic time relations')
    return points


def derive_fixed_native_resources(problem, projection, contract):
    """Model the existing normal-success order using event ordinals and byte leases.

    The controller provides the source-replayed recipe, problem and contract.
    Ordinals describe ordering only. Unknown numeric costs stay on the timed
    problem, and this certificate cannot admit a different native schedule.
    """
    problem = validate_problem(problem)
    contract_sha = semantic_sha256(contract)
    _require(projection['projected_problem_sha256'] == digest(problem)
        and projection['contract_sha256'] == contract_sha
        and projection['original_native_order_preserved'] is True
        and projection['reordering_authorized'] is False,
        'fixed resource witness needs the source-replayed task projection')
    if any(option.unresolved_resources for op in problem.operations for option in op.options):
        return None
    time = projection['time_binding']
    lane = time['resources']['native_lane']
    if time['frontier'] or lane is None:
        return None
    schedule = schedule_from_dict(projection['timed_schedule'])
    task_order = [task.id for task in schedule.tasks]
    positions = {task: index for index, task in enumerate(task_order)}
    _require(all(positions[dep] < positions[task.id] for task in schedule.tasks for dep in task.depends_on),
        'native recipe is not in an executable predecessor order')
    binding_order = [task for row in projection['operation_bindings'] for task in row['task_ids']]
    op_positions = {op.id: index for index, op in enumerate(problem.operations)}
    _require(binding_order == task_order
        and [row['operation'] for row in projection['operation_bindings']] == list(op_positions)
        and all(op_positions[dep] < op_positions[op.id] for op in problem.operations for dep in op.predecessors),
        'fixed order differs from the projected implementation problem')

    points = _ordered_events(schedule)
    model = schedule.temporal
    activities = {activity.id: activity.task for activity in model.activities}
    changes, private_changes = defaultdict(lambda: defaultdict(int)), defaultdict(int)
    occupied_slots, demands = set(), {}
    for occupancy in model.occupancies:
        start, end = points[occupancy.start], points[occupancy.end]
        units = occupancy.units
        _require(type(units) is int and 0 <= units and start < end,
            'fixed resource witness requires closed positive-length occupancy intervals')
        changes[occupancy.resource][start] += units
        changes[occupancy.resource][end] -= units
        if occupancy.storage_slot is not None:
            occupied_slots.add(occupancy.storage_slot)
            private_changes[start] += units
            private_changes[end] -= units
        else:
            demands[activities[occupancy.activity], occupancy.resource] = units
    peaks = {name: _peak(values) for name, values in changes.items()}
    _require(demands == {(row['task'], resource): units for row in time['task_bindings']
        for resource, units in row['execution_demands'].items()},
        'execution occupancy differs from its original task participation')
    _require(all(peak <= model.resource_capacities[name] for name, peak in peaks.items()),
        'fixed order exceeds a controlled execution lane or slot capacity')

    memory = contract['memory']
    slots = {slot.id: slot.capacity_bytes for slot in schedule.storage_slots if slot.ownership == 'SCHEDULE'}
    borrowed = sum(slot.capacity_bytes for slot in schedule.storage_slots if slot.ownership == 'EXTERNAL')
    private_peak = _peak(private_changes)
    payload = private_peak + memory['fresh_output_bytes'] + (borrowed if memory['borrowed_included_in_budget'] else 0)
    _require(occupied_slots == slots.keys()
        and slots == {slot['id']: slot['capacity_bytes'] for slot in memory['private_slots']}
        and borrowed == memory['borrowed_payload_bytes']
        and private_peak <= memory['private_workspace_bytes'] == sum(slots.values())
        and payload <= memory['included_payload_bytes'] <= memory['budget_bytes']
        and sum(problem.reserved_memory.values()) == memory['included_payload_bytes'],
        'original fixed storage lifetimes differ from scoped workspace accounting')
    affinity = contract['execution']['affinity_stage']
    _require(lane['cpu_ids'] == affinity['cpu_ids'] and len(lane['cpu_ids']) == peaks[lane['id']] == 1,
        'fixed native lane differs from its single-CPU affinity contract')
    implementation = contract['implementation']
    identity = contract['identity']
    return dict(schema=SCHEMA, schema_version=1, status='VERIFIED_FIXED_NATIVE_ORDER',
        contract_sha256=contract_sha, projected_problem_sha256=digest(problem),
        task_order=task_order, event_order_kind='ORDINAL_NORMAL_SUCCESS',
        semantic_constraints_checked=len(model.constraints), resource_peaks=peaks,
        native_manifest=implementation['native_manifest'], native_stage_plan=identity['stage_plan'],
        generated_source=implementation['native_members']['source'], candidate=identity['candidate'],
        native_cpu_ids=affinity['cpu_ids'], native_affinity_task_ids=[row['task']
            for row in time['task_bindings'] if row['native_lane'] == lane['id']],
        private_slots=slots, private_payload_bytes=private_peak, borrowed_payload_bytes=borrowed,
        scoped_payload_bytes=payload, budget_bytes=memory['budget_bytes'],
        numeric_durations_proven=False, physical_reservation_proven=False, numa_residency_proven=False,
        native_reordering_authorized=False, deployment_authorized=False)

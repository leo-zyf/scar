"""Construct schedules over the existing typed implementation planning model.

Critical-path priorities, local mode/route choices and capacity calendars are
heuristics.  Endpoint propagation is an integer graph algorithm, not a solver.
The existing timeline reconstruction and independent verifier remain the sole
authority for storage accounting and final feasibility.
"""
from collections import defaultdict, deque
import time

from scar.ir.implementation_plan import PLAN_SCHEMA, digest, validate_problem
from scar.optimization.joint_planner import _frontiers, _known, _qualified_mode, _timeline, verify_execution_plan


class _Stopped(Exception):
    def __init__(self, reason, **details):
        self.details = dict(reason=reason, **details)


def _present(when, selected):
    return not when or any(c.operation in selected and selected[c.operation].id == c.option for c in when)


def _event(task, endpoint):
    return (*task, endpoint)


def _topology(problem):
    predecessors = {o.id:set(o.predecessors) for o in problem.operations}
    for edge in problem.data_edges:
        if edge.producer is not None:
            predecessors[edge.consumer].add(edge.producer)
    successors = {o.id:[] for o in problem.operations}
    for after, before in predecessors.items():
        for name in before:
            successors[name].append(after)
    pending = {name:len(deps) for name, deps in predecessors.items()}
    ready = deque(name for name in pending if not pending[name])
    order = []
    while ready:
        name = ready.popleft(); order.append(name)
        for child in successors[name]:
            pending[child] -= 1
            if not pending[child]: ready.append(child)
    return predecessors, successors, order


def _fit(release, duration, demands, calendar, capacities):
    if not duration: return release
    if any(units > capacities[r] for r, units in demands.items()): return None
    candidates = {release}
    for resource in demands:
        candidates.update(end for _, end, _ in calendar.get(resource, ()) if end >= release)
    for start in sorted(candidates):
        end = start + duration
        fits = True
        for resource, units in demands.items():
            records = []
            for left, right, occupied in calendar.get(resource, ()):
                if left < end and start < right:
                    records.extend(((max(left, start), occupied), (min(right, end), -occupied)))
            current = units
            for _, delta in sorted(records):
                current += delta
                if current > capacities[resource]:
                    fits = False; break
            if not fits: break
        if fits: return start
    return None


def _occupy(calendar, start, end, demands):
    if start != end:
        for resource, units in demands.items():
            if units: calendar.setdefault(resource, []).append((start, end, units))


def _minimum_transfer(edge, choices):
    sources = ({edge.external_location} if edge.producer is None else
        {m.output_locations[edge.value] for m in choices[edge.producer]})
    destinations = {m.input_locations[edge.value] for m in choices[edge.consumer]}
    if sources & destinations: return 0
    costs = [t.cost.estimate_ns for t in edge.transfers if _known(t.cost)
        and t.source_memory in sources and t.destination_memory in destinations]
    # Omitting an unavailable communication cost is a relaxation for the bound,
    # never a zero-cost route offered to the constructor.
    return min(costs, default=0)


def _bounds(problem, choices, predecessors, order, capacities):
    if any(not modes for modes in choices.values()): return None, {}, {}
    durations = {name:min(m.cost.estimate_ns for m in modes) for name, modes in choices.items()}
    delays = {name:{p:0 for p in deps} for name, deps in predecessors.items()}
    roots = {name:0 for name in choices}
    for edge in problem.data_edges:
        delay = _minimum_transfer(edge, choices)
        if edge.producer is None:
            roots[edge.consumer] = max(roots[edge.consumer], delay)
        else:
            delays[edge.consumer][edge.producer] = max(delays[edge.consumer][edge.producer], delay)
    ends = {}
    for name in order:
        ends[name] = durations[name] + max([roots[name],
            *(ends[p] + delay for p, delay in delays[name].items())])
    rank = {}
    for name in reversed(order):
        tails = [rank[after] + delays[after][name] for after in order if name in delays[after]]
        rank[name] = durations[name] + max(tails, default=0)
    energy = {r:0 for r in capacities}
    for modes in choices.values():
        for resource in energy:
            energy[resource] += min(m.cost.estimate_ns *
                (m.demands.get(resource, 0) + m.workspace_bytes.get(resource, 0)) for m in modes)
    unconditional = 0
    for activity in problem.activities:
        if not activity.when and _known(activity.cost):
            unconditional = max(unconditional, activity.cost.estimate_ns)
            for resource, units in activity.demands.items():
                energy[resource] += units * activity.cost.estimate_ns
    workload = max(((amount + capacities[r] - 1) // capacities[r]
        for r, amount in energy.items() if amount and capacities[r] > 0), default=0)
    return max(max(ends.values()), unconditional, workload), rank, dict(
        critical_path_ns=max(ends.values()), unconditional_activity_ns=unconditional,
        resource_energy_lower_bound_ns=workload,
        basis='MINIMUM_KNOWN_MODE_DURATIONS_AND_MATCHING_TRANSFER_RELAXATION_PLUS_PER_RESOURCE_MINIMUM_ENERGY')


def _choose(problem, choices, predecessors, successors, rank, capacities, check):
    indexes = {o.id:i for i, o in enumerate(problem.operations)}
    operations = {o.id:o for o in problem.operations}
    incoming = defaultdict(list)
    for edge in problem.data_edges: incoming[edge.consumer].append(edge)
    selected, routes, ends, calendar, priority_order = {}, {}, {}, {}, []
    ready = {name for name, deps in predecessors.items() if not deps}
    while ready:
        check()
        name = min(ready, key=lambda n:(-rank[n], indexes[n])); ready.remove(name)
        best = None
        for position, mode in enumerate(choices[name]):
            check()
            probe = {r:list(rows) for r, rows in calendar.items()}
            release = max((ends[p] for p in predecessors[name]), default=0)
            picked, possible = {}, True
            for edge in incoming[name]:
                source = (edge.external_location if edge.producer is None else
                    selected[edge.producer].output_locations[edge.value])
                destination = mode.input_locations[edge.value]
                available = 0 if edge.producer is None else ends[edge.producer]
                if source == destination:
                    release = max(release, available); continue
                options = []
                for index, route in enumerate(edge.transfers):
                    if not _known(route.cost) or (route.source_memory, route.destination_memory) != (source, destination):
                        continue
                    start = _fit(available, route.cost.estimate_ns, route.demands, probe, capacities)
                    if start is not None: options.append((start + route.cost.estimate_ns, index, start, route))
                if not options:
                    possible = False; break
                end, _, start, route = min(options, key=lambda row:row[:2])
                picked[edge.id] = route; release = max(release, end)
                _occupy(probe, start, end, route.demands)
            if not possible: continue
            demands = {**mode.demands, **mode.workspace_bytes}
            start = _fit(release, mode.cost.estimate_ns, demands, probe, capacities)
            if start is None: continue
            end = start + mode.cost.estimate_ns
            activation = sum(a.cost.estimate_ns for a in problem.activities if _known(a.cost)
                and not _present(a.when, selected) and _present(a.when, {**selected, name:mode}))
            score = (end + activation, end, mode.id != operations[name].baseline_option, position)
            if best is None or score < best[0]:
                _occupy(probe, start, end, demands)
                best = score, mode, picked, end, probe
        if best is None:
            raise _Stopped('GREEDY_MODE_OR_ROUTE_DEAD_END', operation=name,
                meaning='A different earlier choice may still be feasible; no infeasibility is established.')
        _, selected[name], picked, ends[name], calendar = best
        routes.update(picked); priority_order.append(name)
        for child in successors[name]:
            if child not in selected and predecessors[child] <= selected.keys(): ready.add(child)
    return selected, routes, priority_order


def _temporal(problem, selected, routes, rank, priority_order):
    tasks, edges, lease_events = {}, [], {}
    order = {name:i for i, name in enumerate(priority_order)}
    def task(key, duration, demands, priority, wait=False):
        tasks[key] = dict(duration=duration, demands=demands, priority=priority, wait=wait)
        start, end = _event(key, 'START'), _event(key, 'END')
        edges.append((start, end, duration))
        if not wait: edges.append((end, start, -duration))
    names = {}
    for operation in problem.operations:
        mode = selected[operation.id]; key = ('operation', operation.id); names[operation.id] = key
        task(key, mode.cost.estimate_ns, mode.demands, rank[operation.id])
        for pool in mode.workspace_bytes:
            lease_events['workspace:' + operation.id + ':' + pool] = (_event(key, 'START'), _event(key, 'END'))
        for before in operation.predecessors:
            edges.append((_event(('operation', before), 'END'), _event(key, 'START'), 0))
    for activity in problem.activities:
        if not _present(activity.when, selected): continue
        if not _known(activity.cost): raise _Stopped('ACTIVE_ACTIVITY_COST_UNKNOWN', activity=activity.id)
        key = ('activity', activity.id); names[activity.id] = key
        priority = max((rank[c.operation] for c in activity.when), default=max(rank.values()))
        task(key, activity.cost.estimate_ns, activity.demands, priority, activity.duration_kind == 'WAIT_TAIL')
    for edge in problem.data_edges:
        consumer = ('operation', edge.consumer)
        producer = None if edge.producer is None else ('operation', edge.producer)
        source = edge.external_location if producer is None else selected[edge.producer].output_locations[edge.value]
        destination = selected[edge.consumer].input_locations[edge.value]
        if source == destination:
            if producer is not None: edges.append((_event(producer, 'END'), _event(consumer, 'START'), 0))
            last = _event(consumer, 'END')
        else:
            route = routes[edge.id]; key = ('transfer', edge.id)
            task(key, route.cost.estimate_ns, route.demands, rank[edge.consumer])
            if producer is not None: edges.append((_event(producer, 'END'), _event(key, 'START'), 0))
            edges.append((_event(key, 'END'), _event(consumer, 'START'), 0))
            lease_events['input-copy:' + edge.id] = (_event(key, 'START'), _event(consumer, 'END'))
            last = _event(key, 'END')
        if producer is not None:
            release = ('release', edge.value, 'END')
            # This only names the max-of-uses endpoint for calendar ordering;
            # actual bytes and lifetimes are reconstructed by _timeline.
            edges.extend(((_event(producer, 'END'), release, 0), (last, release, 0)))
            lease_events['output:' + edge.value] = (_event(producer, 'START'), release)
    for constraint in problem.timing_constraints:
        if not _present(constraint.when, selected) or constraint.before not in names or constraint.after not in names:
            continue
        before = _event(names[constraint.before], constraint.before_endpoint)
        after = _event(names[constraint.after], constraint.after_endpoint)
        edges.append((before, after, constraint.minimum_ns))
        if constraint.maximum_ns is not None: edges.append((after, before, -constraint.maximum_ns))
    nodes = {_event(key, endpoint) for key in tasks for endpoint in ('START', 'END')}
    nodes.update(point for left, right, _ in edges for point in (left, right))
    return tasks, edges, nodes, lease_events, order


def _propagate(nodes, edges, check):
    times = {node:0 for node in nodes}
    for _ in range(len(nodes)):
        check(); changed = False
        for before, after, delay in edges:
            value = times[before] + delay
            if times[after] < value:
                times[after] = value; changed = True
        if not changed: return times
    return None  # Positive cycle for these choices/orderings, not a domain proof.


def _rows(problem, selected, routes, tasks, times, report):
    value = dict(report)
    def interval(key):
        return dict(start_ns=times[_event(key, 'START')], end_ns=times[_event(key, 'END')])
    value['selected'] = [dict(operation=o.id, option=selected[o.id].id,
        **interval(('operation', o.id))) for o in problem.operations]
    value['activities'] = [dict(activity=a.id, **interval(('activity', a.id)))
        for a in problem.activities if ('activity', a.id) in tasks]
    if not problem.activities and not problem.timing_constraints: value.pop('activities', None)
    value['transfers'] = [dict(edge=e.id, option=routes[e.id].id,
        source_memory=routes[e.id].source_memory, destination_memory=routes[e.id].destination_memory,
        **interval(('transfer', e.id))) for e in problem.data_edges if e.id in routes]
    value['makespan_ns'] = max(row['end_ns'] for rows in
        (value['selected'], value['transfers'], value.get('activities', [])) for row in rows)
    execution, leases = _timeline(problem, value)
    value['memory_leases'] = leases
    return value, execution, leases


def _calendar(tasks, times, execution, leases, lease_events):
    anchors = defaultdict(deque)
    for key, task in tasks.items():
        start, end = _event(key, 'START'), _event(key, 'END')
        for resource, units in task['demands'].items():
            anchors[(resource, times[start], times[end], units)].append((start, end, task['priority']))
    records = []
    for resource, start, end, units in execution:
        left, right, priority = anchors[(resource, start, end, units)].popleft()
        records.append(dict(resource=resource, start=start, end=end, units=units,
            left=left, right=right, priority=priority, label='execution:' + ':'.join(left[:2])))
    for row in leases:
        endpoints = lease_events.get(row['id'])
        left, right = endpoints if endpoints is not None else (None, None)
        if endpoints is not None:
            if (times[left], times[right]) != (row['start_ns'], row['end_ns']):
                raise ValueError('constructive lifetime endpoints differ from independent timeline')
        priority = tasks[left[:2]]['priority'] if left is not None else float('inf')
        records.append(dict(resource=row['memory'], start=row['start_ns'], end=row['end_ns'],
            units=row['bytes'], left=left, right=right, priority=priority, label=row['id']))
    return records


def _conflict(records, resources):
    candidates = []
    for resource in sorted(resources):
        events = defaultdict(list)
        for index, row in enumerate(records):
            if row['resource'] == resource and row['units'] and row['start'] < row['end']:
                events[row['start']].append((1, index)); events[row['end']].append((-1, index))
        active, units = set(), 0
        for instant in sorted(events):
            for sign, index in sorted(events[instant]):
                units += sign * records[index]['units']
                if sign < 0: active.remove(index)
                else: active.add(index)
            if units > resources[resource].capacity:
                candidates.append((instant, resource, sorted(active))); break
    return min(candidates, default=None)


def construct_implementations(problem, *, baseline_only=False, max_repairs=10000, max_seconds=30):
    """Return a verifier-compatible PLAN_SCHEMA using no solver or target run.

    Ready operations use optimistic critical-path priorities. Modes/routes use
    local earliest finish, with a first-activation host-service penalty. Exact
    endpoint propagation then repairs capacity conflicts by adding scheduling
    orders. A greedy dead end does not establish infeasibility. All successful
    plans pass the existing independent resource/storage/timing verifier.
    """
    if (type(baseline_only) is not bool or type(max_repairs) is not int or max_repairs < 1
            or type(max_seconds) not in (int, float) or not 0 < max_seconds <= 3600):
        raise ValueError('explicit baseline flag, positive repair budget and bounded construction time required')
    started = time.monotonic(); problem = validate_problem(problem)
    frontiers = _frontiers(problem, baseline_only)
    extended = bool(problem.activities or problem.timing_constraints)
    report = dict(schema=PLAN_SCHEMA, schema_version=2 if extended else 1, problem_sha256=digest(problem),
        status='UNRESOLVED', solver_status='NOT_RUN_CONSTRUCTIVE_HEURISTIC',
        domain_complete=not frontiers, frontiers=frontiers,
        optimization_domain='BASELINE_IMPLEMENTATIONS_ONLY' if baseline_only else 'ALL_DECLARED_IMPLEMENTATION_OPTIONS',
        tie_break='BASELINE_THEN_DECLARED_ORDER_AT_EQUAL_LOCAL_EARLIEST_FINISH_SCORE',
        baseline_only=baseline_only, selected=[], transfers=[], memory_leases=[], makespan_ns=None,
        model_lower_bound_ns=None, relative_gap=None, known_subdomain_lower_bound_ns=None,
        known_subdomain_gap=None, model_optimality_proven=False, physical_optimality_proven=False,
        cost_basis='EXPLICIT_CONTEXT_POINT_ESTIMATES_NOT_PHYSICAL_TIME_GUARANTEES',
        memory_accounting='SCOPED_INCREMENTAL_PRIVATE_PAYLOAD_EXTERNAL_INPUTS_BORROWED',
        external_input_bytes_not_charged=sum({e.value:e.nbytes for e in problem.data_edges if e.producer is None}.values()),
        solver=dict(name='CONSTRUCTIVE_CRITICAL_PATH_CAPACITY_CALENDARS', solver_called=False,
            max_repairs=max_repairs, max_seconds=max_seconds),
        target_program_executed=False, execution_authorized=False, deployment_authorized=False,
        default_backend='NO_OP', construction_frontier=[], scheduling_constraints=[],
        construction=dict(mode_selection='READY_CRITICAL_PATH_LOCAL_EFT_PLUS_FIRST_HOST_ACTIVATION_SERVICE',
            memory_policy='FINITE_CAPACITIES_AND_INDEPENDENT_TIMELINE_LEASES',
            endpoint_policy='INTEGER_DIFFERENCE_PROPAGATION_FIXED_EQUALITIES_AND_WAIT_TAIL',
            repair_attempts=0, added_resource_orders=0))
    if extended: report['activities'] = []
    def check():
        if time.monotonic() - started > max_seconds: raise _Stopped('CONSTRUCTION_TIME_BUDGET_EXHAUSTED')
    try:
        resources = {r.id:r for r in problem.resources}
        capacities = {r.id:r.capacity - problem.reserved_memory.get(r.id, 0) for r in problem.resources}
        if any(capacity < 0 for capacity in capacities.values()):
            raise _Stopped('RESERVED_MEMORY_EXCEEDS_DECLARED_CAPACITY')
        forbidden = set()
        for activity in problem.activities:
            blocked = (not _known(activity.cost) or bool(activity.cost.estimate_ns)
                and any(units > resources[r].capacity for r, units in activity.demands.items()))
            if blocked:
                if not activity.when: raise _Stopped('UNAVAILABLE_UNCONDITIONAL_ACTIVITY', activity=activity.id)
                forbidden.update((c.operation, c.option) for c in activity.when)
        choices = {o.id:[m for m in o.options if _qualified_mode(m)
            and (not baseline_only or m.id == o.baseline_option) and (o.id, m.id) not in forbidden
            and (not m.cost.estimate_ns or all(units <= capacities[r]
                for r, units in {**m.demands, **m.workspace_bytes}.items()))] for o in problem.operations}
        predecessors, successors, order = _topology(problem)
        lower, rank, bound = _bounds(problem, choices, predecessors, order, capacities)
        if lower is None: raise _Stopped('NO_COMPLETE_KNOWN_MODE_DOMAIN')
        report.update(known_subdomain_lower_bound_ns=lower, lower_bound_details=bound,
            lower_bound_scope='ALL_KNOWN_COST_OPTIONS_WITH_UNAVAILABLE_ACTIVITY_CHOICES_EXCLUDED',
            model_lower_bound_ns=None if frontiers else lower)
        selected, routes, priority_order = _choose(problem, choices, predecessors, successors, rank, capacities, check)
        report['construction']['chosen_options'] = {name:mode.id for name, mode in selected.items()}
        report['construction']['chosen_routes'] = {name:route.id for name, route in routes.items()}
        tasks, constraints, nodes, lease_events, _ = _temporal(problem, selected, routes, rank, priority_order)
        times = _propagate(nodes, constraints, check)
        if times is None: raise _Stopped('CHOSEN_MODE_ENDPOINT_CONSTRAINTS_CONFLICT')
        report['fixed_selection_lower_bound_ns'] = max(times[_event(key, 'END')] for key in tasks)
        report['fixed_selection_bound_scope'] = 'CHOSEN_MODES_ROUTES_AND_ACTIVITY_PRESENCE_ENDPOINT_RELAXATION_ONLY'
        report['construction']['priority_order'] = priority_order
        seen = set(constraints)
        while True:
            check()
            candidate, execution, leases = _rows(problem, selected, routes, tasks, times, report)
            records = _calendar(tasks, times, execution, leases, lease_events)
            conflict = _conflict(records, resources)
            if conflict is None:
                total = candidate['makespan_ns']
                if total < lower:
                    raise ValueError('constructive lower bound exceeds the independently reconstructed schedule')
                candidate.update(status='FEASIBLE_PARTIAL_DOMAIN' if frontiers else 'FEASIBLE_WITHIN_MODEL',
                    relative_gap=None if frontiers else (total - lower) / max(1, total),
                    known_subdomain_gap=(total - lower) / max(1, total),
                    lower_bound_matched=total == lower, elapsed_seconds=time.monotonic() - started)
                candidate['verification'] = verify_execution_plan(problem, candidate)
                return candidate
            instant, resource, active = conflict
            delayed = sorted(active, key=lambda i:(records[i]['priority'], -records[i]['start'], records[i]['label']))
            repaired = False
            for victim in delayed:
                row = records[victim]
                if row['left'] is None: continue
                blockers = sorted((i for i in active if i != victim),
                    key=lambda i:(records[i]['end'], -records[i]['priority'], records[i]['label']))
                for blocker in blockers:
                    before = records[blocker]
                    if before['right'] is None: continue
                    edge = (before['right'], row['left'], 0)
                    if edge in seen: continue
                    if report['construction']['repair_attempts'] >= max_repairs:
                        raise _Stopped('CAPACITY_ORDER_BUDGET_EXHAUSTED', resource=resource, time_ns=instant)
                    report['construction']['repair_attempts'] += 1
                    proposed = _propagate(nodes, [*constraints, edge], check)
                    if proposed is None: continue
                    constraints.append(edge); seen.add(edge); times = proposed; repaired = True
                    report['construction']['added_resource_orders'] += 1
                    report['scheduling_constraints'].append(dict(resource=resource,
                        before=list(before['right']), after=list(row['left']), minimum_ns=0,
                        reason='CAPACITY_CALENDAR_ORDER_NOT_SOURCE_SEMANTIC_DEPENDENCY'))
                    break
                if repaired: break
            if not repaired:
                raise _Stopped('CAPACITY_CALENDAR_DEAD_END', resource=resource, time_ns=instant,
                    intervals=[records[i]['label'] for i in active],
                    meaning='Alternative modes, routes or resource orders may be feasible; no infeasibility is established.')
    except _Stopped as stopped:
        report['construction_frontier'].append(stopped.details)
        report['elapsed_seconds'] = time.monotonic() - started
        return report


__all__ = ['construct_implementations']

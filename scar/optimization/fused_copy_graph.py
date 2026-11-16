"""A source-window-qualified replacement task and synchronous temporal graph.

The caller replays the original source/window publications. This module checks
that finite selection against the original graph and preserves every external
port. It grants no execution or cost authority to the replacement.
"""
from copy import deepcopy
from dataclasses import replace

from scar.ir.record_codec import encode
from scar.ir.versioned_schedule import (VersionedResourceSchedule, VersionedTask,
    VersionedTaskKind as K, TemporalMode, TemporalScopeKind, TemporalEvent,
    TemporalEventKind as E, TemporalActivity, TemporalDurationBasis as D,
    TemporalOccupancy, TemporalOccupancyKind as O, TemporalConstraintKind as C,
    schedule_document, validate_versioned_schedule)
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.fused-copy-graph-certificate'
KIND = 'FUSE_IDENTICAL_INDEPENDENT_RAW_COPY_LOOPS'
_WINDOW_FIELDS = ('id', 'task_ids', 'activity_ids', 'left_anchor', 'right_anchor', 'live_inputs',
    'live_outputs', 'preserved_unconsumed_outputs', 'borrowed_slots', 'memory_budget',
    'resource_refs', 'original_order_refs', 'pairs', 'frontiers')
_PROPOSAL_FIELDS = ('id', 'kind', 'task_ids', 'iteration_domain', 'source_pair_refs', 'transformation')


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def derive_fused_copy_graph(schedule, window, proposal):
    """Replace a complete contiguous raw-copy window, without unrolling it."""
    _need(type(schedule) is VersionedResourceSchedule, 'original typed versioned schedule required')
    validate_versioned_schedule(schedule)
    temporal = schedule.temporal
    _need(temporal is not None and temporal.mode is TemporalMode.SEMANTIC
          and not any(row.kind is TemporalScopeKind.LOOP for row in temporal.scopes),
          'one synchronous semantic source scope required')
    parent_sha = semantic_sha256(schedule_document(schedule))
    _need(type(window) is dict and type(proposal) is dict and proposal.get('kind') == KIND,
          'original closed-copy window and graph-derived fusion proposal required')
    ids = list(window['task_ids']); selected = set(ids)
    tasks = {row.id: row for row in schedule.tasks}; positions = {row.id: i for i, row in enumerate(schedule.tasks)}
    _need(len(ids) >= 2 and len(selected) == len(ids) and selected <= set(tasks)
          and ids == list(proposal['task_ids']), 'one complete distinct original copy window required')
    first = positions[ids[0]]
    _need([positions[key] for key in ids] == list(range(first, first+len(ids))),
          'copy window must be complete and consecutive in the original task graph')
    _need(window['left_anchor'] == (schedule.tasks[first-1].id if first else None)
          and window['right_anchor'] == (schedule.tasks[first+len(ids)].id if first+len(ids) < len(tasks) else None),
          'original complete window anchors differ')
    activities = {row.task: row for row in temporal.activities}
    _need(len(activities) == len(temporal.activities) == len(tasks)
          and set(activities) == set(tasks) and window['activity_ids'] == [activities[key].id for key in ids],
          'one original synchronous lifecycle per task/window required')
    shape = list(tasks[ids[0]].attrs.get('shape', ()))
    _need(len(shape) == 2 and all(type(n) is int and n > 0 for n in shape)
          and list(proposal['iteration_domain']) == shape, 'one positive original rank-two iteration domain required')
    extent = shape[0] * shape[1] * 8
    materials = {row.id: row for row in schedule.materializations}; slots = {row.id: row for row in schedule.storage_slots}
    reads = []; writes = []
    for key in ids:
        task = tasks[key]
        _need(task.kind is K.SNAPSHOT and task.capability == 'CPU_HOST_SNAPSHOT'
            and task.attrs.get('stage_kind') == 'SNAPSHOT_ROOT' and list(task.attrs.get('shape', ())) == shape
            and len(task.reads) == len(task.writes) == 1 and task.source_refs
            and not (task.world_before or task.world_after or task.validates or task.allocates or task.releases),
            'only complete source-backed snapshots without other effects may fuse')
        source, target = materials[task.reads[0]], materials[task.writes[0]]
        _need(source.value == target.value and source.region.nbytes == target.region.nbytes == extent
            and source.region.offset_bytes == target.region.offset_bytes == 0
            and target.ready_task == key and slots[target.region.slot].ownership == 'SCHEDULE'
            and slots[source.region.slot].device == slots[target.region.slot].device == 'cpu',
            'original full CPU float64 copy ports and private destination required')
        _need(source.ready_task not in selected, 'window-internal material consumer cannot fuse in this profile')
        reads.append(source.id); writes.append(target.id)
    _need(len(set(reads)) == len(reads) and len(set(writes)) == len(writes), 'distinct original copy-pair ports required')
    _need(not any(value.producer in selected for value in schedule.values),
          'snapshot window cannot own a new logical value producer')
    def overlap(left, right):
        a, b = materials[left].region, materials[right].region
        return a.slot == b.slot and a.nbytes and b.nbytes and a.offset_bytes < b.offset_bytes+b.nbytes and b.offset_bytes < a.offset_bytes+a.nbytes
    _need(not any(overlap(a, b) for a in writes for b in reads)
          and not any(overlap(a, b) for i, a in enumerate(writes) for b in writes[i+1:]),
          'all private destinations must be disjoint from every source and destination')
    pair_rows = {(row['before'], row['after']): row for row in window['pairs']}
    pairs = {(before, after) for i, before in enumerate(ids) for after in ids[i+1:]}
    _need(len(pair_rows) == len(window['pairs']) and set(pair_rows) == pairs,
          'all original window pairs need unique independence proofs')
    _need({(row['before'], row['after']) for row in proposal['source_pair_refs']} == pairs,
          'proposal must reference every original independent pair')
    allowed_dependencies = set(); allowed_orders = set()
    for pair in pairs:
        row = pair_rows[pair]; proof = row['proof']
        _need(row['status'] == 'PAIRWISE_DATA_INDEPENDENT' and proof['source_closed_contract']
              and not (proof['remaining_obligations'] or proof['storage_region_conflicts']
                  or proof['producer_consumer'] or proof['world_or_token_conflict'] or proof['non_sequence_dependencies'])
              and proof['same_original_buffers_and_no_arithmetic_rewrite'] is True,
              'original source/data/effect independence remains incomplete')
        for ref in proof['candidate_source_order_edges']:
            _need((ref['before'], ref['after']) == pair and ref['parent_document'] == 'schedule',
                  'source sequence proof cites a different task pair')
            pointer = ref['source_locator']
            if pointer.startswith('#/tasks/'):
                allowed_dependencies.add((pair[0], pair[1], pointer))
            elif pointer.startswith('#/order_constraints/'):
                allowed_orders.add((pair[0], pair[1], pointer))
            else:
                raise ValueError('unknown original source sequence proof locator')
    hashes = [ref['source_schedule_sha256'] for row in window['resource_refs'] for binding in row['bindings']
              for ref in binding.get('source_refs', ()) if 'source_schedule_sha256' in ref]
    if 'parent_schedule_sha256' in window:
        hashes.append(window['parent_schedule_sha256'])
    _need(hashes and all(value == parent_sha for value in hashes), 'window evidence refers to a different original schedule hash')
    _need(window['memory_budget'] == dict(bytes=schedule.budget_bytes, scope=schedule.budget_scope, source_refs=encode(schedule.budget_refs)),
          'original window budget cannot change')
    for row in window['borrowed_slots']:
        _need(row['slot'] == encode(slots[row['slot']['id']]) and row['newly_allocated'] is False
              and row['lifetime_changed'] is False, 'original borrowed storage identity/lifetime differs')
    consumers = {name: [t.id for t in schedule.tasks if name in t.reads] for name in writes}
    ports = window['live_inputs'] + window['live_outputs'] + window['preserved_unconsumed_outputs']
    _need([row['materialization'] for row in window['live_inputs']] == reads
          and set(row['materialization'] for row in window['live_outputs']) == {name for name in writes if consumers[name]}
          and set(row['materialization'] for row in window['preserved_unconsumed_outputs']) == {name for name in writes if not consumers[name]},
          'complete original data input/output/unused-output interface required')
    for row in ports:
        original = materials[row['materialization']]
        _need(all(row[name] == value for name, value in dict(value=original.value, version=original.version,
            region=encode(original.region), ready_task=original.ready_task, external_ready=original.external_ready).items()),
            'original materialization port changed')
    lifecycle = [activities[key] for key in ids]
    _need(len({row.scope for row in lifecycle}) == 1 and all(row.submission_return is None for row in lifecycle),
          'asynchronous or cross-scope copy lifecycles cannot fuse')
    selected_activity_ids = {row.id for row in lifecycle}
    demand_rows = [row for row in temporal.occupancies if row.kind is O.EXECUTION_DEMAND and row.activity in selected_activity_ids]
    _need(len(demand_rows) == len(ids) and {row.activity for row in demand_rows} == selected_activity_ids
          and len({row.resource for row in demand_rows}) == 1 and all(row.units == 1 for row in demand_rows)
          and all(row.start == activities[task].start and row.end == activities[task].completion
              for task in ids for row in demand_rows if row.activity == activities[task].id),
          'one same-CPU unit-one full lifecycle execution occupancy per copy required')
    cpu = demand_rows[0].resource
    bindings = {row['task_id']: row['bindings'] for row in window['resource_refs']}
    _need(set(bindings) == selected and all(len(bindings[key]) == 1
        and bindings[key][0]['occupancy'] in {o.id for o in demand_rows if o.activity == activities[key].id}
        and bindings[key][0]['resource'] == cpu and bindings[key][0]['units'] == 1
        and bindings[key][0]['exact_thread'] is not None
        and bindings[key][0]['exact_thread']['resource']['wire'] == cpu for key in ids),
        'original exact same-CPU window binding required')
    seed = dict(parent=parent_sha, window=window['id'], proposal=proposal['id'], tasks=ids)
    def identity(role): return 'fused-copy:' + semantic_sha256(dict(**seed, role=role))[:32]
    fused_id = identity('task'); start = identity('start'); completion = identity('completion'); activity_id = identity('activity')
    _need(fused_id not in tasks, 'replacement task identity collides with original graph')
    parent_ref = dict(schedule_sha256=parent_sha, locator='#/proof')
    refs = tuple(dict(parent_schedule_sha256=parent_sha, locator='#/tasks/'+str(positions[key]), source_task=key) for key in ids)
    copy_pairs = [dict(read=a, write=b, source_task=key) for key, a, b in zip(ids, reads, writes)]
    external_dependencies = []
    for key in ids:
        for index, dependency in enumerate(tasks[key].depends_on):
            if dependency in selected:
                _need((dependency, key, '#/tasks/'+str(positions[key])+'/depends_on/'+str(index)) in allowed_dependencies,
                      'internal dependency is not a proved removable SOURCE_ORDER edge')
            elif dependency not in external_dependencies:
                external_dependencies.append(dependency)
    fused = VersionedTask(fused_id, K.COPY, reads=tuple(reads), writes=tuple(writes),
        depends_on=tuple(external_dependencies), capability='CPU_HOST_COPY_REGION', source_refs=refs,
        resource_refs=tuple(dict(source_task=key, parent_schedule_sha256=parent_sha,
            source_locator='#/tasks/'+str(positions[key])+'/resource_refs') for key in ids),
        attrs=dict(stage_kind='FUSED_SNAPSHOT_ROOT', dtype='float64', shape=shape, copy_pairs=copy_pairs,
            child_task_ids=ids, composite_expansion=[dict(task=key, parent_schedule_sha256=parent_sha,
                locator='#/tasks/'+str(positions[key])) for key in ids]))
    task_map = {key: fused_id if key in selected else key for key in tasks}
    def retarget(names): return tuple(dict.fromkeys(task_map[name] for name in names))
    rewritten_tasks = []
    for task in schedule.tasks:
        if task.id == ids[0]: rewritten_tasks.append(fused)
        if task.id not in selected:
            rewritten_tasks.append(replace(task, depends_on=retarget(task.depends_on), validates=retarget(task.validates)))
    removed_orders = []; orders = []
    for index, edge in enumerate(schedule.order_constraints):
        if edge.before in selected and edge.after in selected:
            _need(edge.kind == 'SOURCE_ORDER' and (edge.before, edge.after, '#/order_constraints/'+str(index)) in allowed_orders,
                  'internal non-source order cannot be removed')
            removed_orders.append(index)
        else:
            orders.append(replace(edge, before=task_map[edge.before], after=task_map[edge.after]))
    updated_materials = tuple(replace(row, ready_task=fused_id) if row.ready_task in selected else row for row in schedule.materializations)
    old_events = {row.id: row for row in temporal.events}; event_map = {key: key for key in old_events}
    new_ready = {name: identity('ready:'+name) for name in writes}
    for activity in lifecycle:
        event_map[activity.start] = start; event_map[activity.completion] = completion
    for event in temporal.events:
        if event.task in selected and event.kind is E.READY:
            _need(event.materialization in new_ready, 'selected READY event lacks an actual copy output')
            event_map[event.id] = new_ready[event.materialization]
        elif event.task in selected:
            _need(event.kind in (E.START, E.COMPLETION), 'unhandled selected asynchronous/lifecycle event')
    new_events = [TemporalEvent(start, E.START, lifecycle[0].scope, task=fused_id, source_refs=refs),
                  TemporalEvent(completion, E.COMPLETION, lifecycle[0].scope, task=fused_id, source_refs=refs)]
    new_events += [TemporalEvent(new_ready[name], E.READY, lifecycle[0].scope, task=fused_id,
        materialization=name, source_refs=refs) for name in writes]
    events = tuple(row for row in temporal.events if row.task not in selected) + tuple(new_events)
    activity_map = {row.id: activity_id if row.task in selected else row.id for row in temporal.activities}
    new_activity = TemporalActivity(activity_id, fused_id, lifecycle[0].scope, start, completion,
        ready_events=tuple(new_ready.values()), duration_basis=D.UNKNOWN, source_refs=refs)
    rewritten_activities = tuple(row for row in temporal.activities if row.task not in selected) + (new_activity,)
    removed_constraints = []; constraints = []
    for edge in temporal.constraints:
        before, after = old_events[edge.before], old_events[edge.after]
        if before.task in selected and after.task in selected and before.task != after.task:
            _need((before.task, after.task) in pairs and before.task in tasks[after.task].depends_on
                and edge.kind is C.CAUSAL
                and edge.before == activities[before.task].completion and edge.after == activities[after.task].start
                and edge.min_delay_ticks == 0 and edge.max_delay_ticks is None
                and edge.iteration_distance == 0 and edge.loop_scope is None,
                'internal timing is not a closed removable pairwise SOURCE_ORDER lifecycle edge')
            removed_constraints.append(edge.id); continue
        if before.task in selected or after.task in selected:
            same = before.task == after.task and before.task in selected
            readiness = (same and edge.kind is C.READINESS and before.kind is E.COMPLETION
                         and after.kind is E.READY and after.materialization in new_ready)
            _need(edge.iteration_distance == 0 and edge.loop_scope is None and edge.min_delay_ticks == 0
                and (edge.max_delay_ticks is None or readiness and edge.max_delay_ticks == 0),
                'unclosed selected boundary/internal temporal relation cannot be retargeted')
        constraints.append(replace(edge, before=event_map[edge.before], after=event_map[edge.after]))
    new_demand = TemporalOccupancy(identity('execution'), O.EXECUTION_DEMAND, cpu, start, completion,
        lifecycle[0].scope, units=1, activity=activity_id, source_refs=refs)
    old_demands = {row.id for row in demand_rows}
    occupancy_map = {row.id: new_demand.id if row.id in old_demands else row.id for row in temporal.occupancies}
    occupancies = tuple(row for row in temporal.occupancies if row.id not in old_demands) + (new_demand,)
    proof = dict(rewrite_parent=parent_ref, source_window_rewrite=dict(kind=KIND, window_id=window['id'],
        proposal_id=proposal['id'], child_task_ids=ids, replacement_task_id=fused_id,
        original_source_phase_order_not_reused=True, execution_authorized=False))
    if 'final_root_commit_identity' in schedule.proof:
        proof['final_root_commit_identity'] = deepcopy(schedule.proof['final_root_commit_identity'])
        for key, row in proof['final_root_commit_identity'].items():
            original = row.get('initial_snapshot')
            if original in selected:
                row['initial_snapshot'] = fused_id
                row['original_initial_snapshot'] = original
            row['parent_proof_ref'] = dict(schedule_sha256=parent_sha,
                locator='#/proof/final_root_commit_identity/'+key.replace('~', '~0').replace('/', '~1'))
    if 'implementation_scope_cleanup' in schedule.proof:
        original = schedule.proof['implementation_scope_cleanup']
        proof['implementation_scope_cleanup'] = dict(schema=original['schema'], schema_version=original['schema_version'],
            milestone=deepcopy(original['milestone']), reused_unmodified_milestone_only=True,
            parent_proof_ref=dict(schedule_sha256=parent_sha, locator='#/proof/implementation_scope_cleanup'))
    temporal_result = replace(temporal, events=events, activities=rewritten_activities, occupancies=occupancies,
        constraints=tuple(constraints), source_refs=(*temporal.source_refs, dict(parent_schedule_sha256=parent_sha,
            transformation=KIND, replacement_task=fused_id)))
    rewritten = replace(schedule, tasks=tuple(rewritten_tasks), materializations=updated_materials,
        order_constraints=tuple(orders), proof=proof, executable=False, temporal=temporal_result)
    validate_versioned_schedule(rewritten)
    certificate = dict(schema=SCHEMA, schema_version=1, parent_schedule_sha256=parent_sha,
        rewritten_schedule_sha256=semantic_sha256(schedule_document(rewritten)),
        window_id=window['id'], proposal_id=proposal['id'], original_task_ids=ids,
        replacement_task_id=fused_id, shape=shape, task_map=task_map, event_map=event_map,
        activity_map=activity_map, occupancy_map=occupancy_map,
        removed_order_constraints=removed_orders, removed_temporal_constraints=removed_constraints,
        selection=dict(window={name: deepcopy(window[name]) for name in _WINDOW_FIELDS},
            proposal={name: deepcopy(proposal[name]) for name in _PROPOSAL_FIELDS}),
        external_ports=dict(reads=reads, writes=writes, input_events=[event_map[e.id] for e in temporal.events
            if e.kind is E.READY and e.materialization in reads], output_events=list(new_ready.values()),
            left_anchor=window['left_anchor'], right_anchor=window['right_anchor']),
        preserved_ids=dict(values=[row.id for row in schedule.values], materializations=[row.id for row in schedule.materializations],
            storage_slots=[row.id for row in schedule.storage_slots], scopes=[row.id for row in temporal.scopes],
            unchanged_activity_ids=[row.id for row in temporal.activities if row.task not in selected]),
        parent_refs=[parent_ref], original_frontiers=deepcopy(window['frontiers']),
        physical_storage_occupancies_unchanged=True, source_window_reader_admission_required=True,
        source_semantics_replayed_by_this_adapter=False, original_costs_reused_for_replacement=False,
        replacement_duration_known=False, execution_authorized=False, profitability_proven=False,
        deployment_authorized=False, physical_optimality_proven=False)
    return rewritten, certificate


def verify_fused_copy_graph(original, rewritten, certificate):
    """Independently rederive structure; source-window admission stays upstream."""
    _need(type(certificate) is dict and (certificate.get('schema'), certificate.get('schema_version')) == (SCHEMA, 1),
          'versioned fused-copy graph certificate required')
    selection = certificate['selection']
    expected, replay = derive_fused_copy_graph(original, selection['window'], selection['proposal'])
    _need(schedule_document(expected) == schedule_document(rewritten)
          and semantic_sha256(replay) == semantic_sha256(certificate), 'replacement graph/certificate differs from complete structural replay')
    return certificate


__all__ = ['SCHEMA', 'KIND', 'derive_fused_copy_graph', 'verify_fused_copy_graph']

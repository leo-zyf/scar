"""Indexed normal-completion time graph for an admitted workspace sequence.

The original data/storage DAG remains a single iteration template. Outer
allocation and release are different tasks with whole-sequence lifetimes.
This is a semantic projection, not timestamps or an executable schedule.
"""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

from scar.ir.versioned_schedule import (
    VersionedTask as Task, VersionedTaskKind as K,
    TemporalMode, TemporalEventKind as E, TemporalScopeKind as S,
    TemporalOccupancyKind as O, TemporalConstraintKind as C,
    TemporalEvent, TemporalScope, TemporalActivity, TemporalOccupancy,
    TemporalConstraint, TemporalModel, TemporalIterationSelection,
    TemporalIterationSelector as Select, TemporalPathCondition, TemporalContentCarry,
    schedule_from_dict, validate_versioned_schedule,
)
from scar.optimization.resource_solver import semantic_sha256


def derive_workspace_time(plan, *, native_manifest, native_metadata, cpp_source, workspace_reference):
    """Consume independently admitted source/implementation records, never run them."""
    if ((plan.get('schema'), plan.get('schema_version')) != ('scar.stateful-workspace-reuse-plan', 1)
            or native_metadata['workspace_plan_sha256'] != semantic_sha256(plan)
            or native_metadata['trip_count'] != plan['trip_count']):
        raise ValueError('matching admitted workspace graph and native source required')
    original = schedule_from_dict(plan['shared_step'])
    count = plan['trip_count']
    if type(count) is not int or count < 0:
        raise ValueError('exact nonnegative iteration count required')
    refs = (dict(artifact=workspace_reference, locator='#'),
            dict(artifact=native_manifest, locator='#/source_mapping'),
            dict(artifact=cpp_source, locator='#'))
    proof = deepcopy(original.proof)
    certificate = dict(schema='scar.workspace-time-projection-certificate', schema_version=1,
        source_workspace_plan_sha256=semantic_sha256(plan), native_manifest=native_manifest,
        cpp_source=cpp_source, trip_count=count, normal_completion_only=True,
        source_f4_authority=deepcopy(plan['source_f4_authority']),
        content_coordinates='(iteration_i, original_local_epoch)',
        fixed_capacity_does_not_imply_constant_content=True,
        failure_contract=dict(artifact=workspace_reference, locator='#/execution_contract/failure'),
        failure_path_timing_modeled=False, python_packing_lock_and_ffi_modeled=False,
        source_math_unchanged=True, native_lowering_authorized=False)
    proof['workspace_time'] = certificate
    root = original.call_scope
    identity = lambda *parts: 'workspace-time:'+sha256(repr((original.region_id, parts)).encode()).hexdigest()[:32]
    events, activities, occupancies, constraints = [], [], [], []
    def event(key, kind, scope, **extra):
        name = identity('event', *key)
        events.append(TemporalEvent(name, kind, scope, source_refs=refs, **extra))
        return name
    root_entry = event(('root', 'entry'), E.SCOPE_ENTRY, root)
    root_exit = event(('root', 'exit'), E.SCOPE_EXIT, root)
    scopes = [TemporalScope(root, S.ROOT, root_entry, root_exit, source_refs=refs)]
    if count == 0:
        # Declared source inputs remain in the parent proof, not active reads.
        certificate.update(task_map={}, body_task_ids=[], outer_task_ids=[],
            activity_occurrences=[], storage_occurrences=[], zero_trip_no_fields_or_allocation=True)
        model = TemporalModel(TemporalMode.SEMANTIC, tuple(events), (), (), (), tuple(scopes),
            source_refs=refs, path_condition=TemporalPathCondition.NORMAL_COMPLETE)
        result = replace(original, values=(), materializations=(), storage_slots=(), tasks=(),
            order_constraints=(), proof=proof, executable=False, temporal=model)
        validate_versioned_schedule(result)
        return result
    loop = identity('loop')
    entry = event(('loop', 'entry'), E.SCOPE_ENTRY, loop)
    exit = event(('loop', 'exit'), E.SCOPE_EXIT, loop)
    values = {row.id: row for row in original.values}
    original_tasks = {row.id: row for row in original.tasks}
    carries = []
    for index, row in enumerate(plan['loop_model']['inputs']):
        if row['role'] != 'ORDERED_ARRAY_CONTENT_CARRY':
            continue
        incoming = [m for m in original.materializations if m.external_ready
                    and values[m.value].attrs.get('input_id') == row['input_id']]
        if len(incoming) != 1:
            raise ValueError('one exact original entry materialization per carried input required')
        outgoing = [m for m in original.materializations if m.region == incoming[0].region
            and m.ready_task is not None and original_tasks[m.ready_task].kind is K.COMMIT]
        if len(outgoing) != 1:
            raise ValueError('one original committed materialization for the next iteration required')
        carries.append(TemporalContentCarry(outgoing[0].id, incoming[0].id,
            source_refs=(dict(artifact=workspace_reference, locator='#/loop_model/inputs/'+str(index)+'/backedge'),)))
    scopes.append(TemporalScope(loop, S.LOOP, entry, exit, parent=root, iteration_symbol='i',
        trip_count=count, source_refs=refs,
        iteration_local_values=tuple(row.id for row in original.values),
        iteration_local_materializations=tuple(row.id for row in original.materializations),
        content_carries=tuple(carries)))
    boundary = plan['loop_region']['storage_boundary_map']
    remap = {row[key]: row[target] for row in boundary for key, target in
             (('original_allocate', 'outer_allocate'), ('original_release', 'outer_release'))}
    body = [row for row in original.tasks if row.kind not in (K.ALLOCATE, K.RELEASE)]
    fences = [row for row in body if row.kind is K.SUCCESS_FENCE]
    commits = [row for row in body if row.kind is K.COMMIT]
    if len(fences) != 1 or not commits:
        raise ValueError('one existing complete precommit fence and committed state required')
    outer_pre = 'workspace:outer-preflight'
    iteration_pre = 'workspace:iteration-preflight'
    restore = 'workspace:iteration-validate-restore'
    progress = 'workspace:iteration-exit-progress'
    def phase(name, kind, dependencies, effects, *, post=False, reads=()):
        return Task(name, K.IMPLEMENTATION_PHASE, reads=tuple(reads), depends_on=tuple(dependencies),
            capability='CPU_IMPLEMENTATION_PHASE', source_refs=refs,
            attrs=dict(phase_kind=kind, actor='CALLER_UNBOUND',
                failure_domain='POSTCOMMIT' if post else 'PRECOMMIT',
                may_fail=False if post else 'UNKNOWN', blocks='UNKNOWN',
                phase_effects=list(effects), normal_completion_only=True,
                qualification='ADMITTED_NATIVE_SOURCE_ORDER_NOT_OBSERVED_TIME'))
    tasks = [phase(outer_pre, 'WORKSPACE_ABI_TABLE_AND_PAYLOAD_PREFLIGHT', (),
        ('ABI_SHAPE_ALIAS_PROVIDER_AND_BUDGET_READS', 'EXACT_COUNT_CHECK'))]
    allocations = []
    for row in boundary:
        prior = original_tasks[row['original_allocate']]
        predecessor = allocations[-1].id if allocations else outer_pre
        allocations.append(replace(prior, id=remap[prior.id], depends_on=(predecessor,)))
    tasks.extend(allocations)
    tasks.append(phase(iteration_pre, 'ORIGINAL_PER_STEP_PREFLIGHT_AND_CONTEXT',
        (allocations[-1].id if allocations else outer_pre,),
        ('SCALAR_GUARDS', 'INPUT_FINITE_SCANS', 'AFFINITY_FENV_AND_TEAM_CONTEXT'),
        reads=tuple(row.id for row in original.materializations if row.external_ready)))
    for row in body:
        dependencies = tuple(dict.fromkeys((*(remap.get(key, key) for key in row.depends_on), iteration_pre)))
        tasks.append(replace(row, depends_on=dependencies,
            validates=tuple(remap.get(key, key) for key in row.validates)))
    fence = next(row for row in tasks if row.kind is K.SUCCESS_FENCE)
    tasks.append(phase(restore, 'ORIGINAL_PER_STEP_VALIDATE_AND_RESTORE', fence.depends_on,
        ('NUMERIC_GUARDS', 'AFFINITY_FENV_RESTORE_AND_CHECK', 'TEAM_COMPLETION')))
    precommit = tuple(row.id for row in tasks if row.kind in (K.ALLOCATE, K.SNAPSHOT, K.COMPUTE,
        K.STATE_WRITE, K.COPY) or row.kind is K.IMPLEMENTATION_PHASE)
    tasks = [replace(row, depends_on=tuple(dict.fromkeys((*row.depends_on, restore))),
        validates=precommit) if row.id == fence.id else row for row in tasks]
    tasks.append(phase(progress, 'ORIGINAL_HELPER_CLEANUP_STEP_RETURN_AND_PROGRESS',
        tuple(row.id for row in body),
        ('HELPER_DESTRUCTION', 'STEP_RETURN', 'SUCCESSFUL_COMMITTED_CURSOR_STORE'), post=True))
    releases = []
    for row in reversed(boundary):
        prior = original_tasks[row['original_release']]
        dependencies = tuple(dict.fromkeys((*(remap.get(key, key) for key in prior.depends_on),
            releases[-1].id if releases else progress)))
        releases.append(replace(prior, id=remap[prior.id], depends_on=dependencies))
    tasks.extend(releases)
    # Keep source order for body presentation; restore must precede its fence.
    rank = {row.id: i for i, row in enumerate(tasks)}
    rank[restore] = rank[fence.id]-0.5
    tasks.sort(key=lambda row: rank[row.id])
    body_ids = {row.id for row in body} | {iteration_pre, restore, progress}
    scope_of = {row.id: loop if row.id in body_ids else root for row in tasks}
    slots = tuple(replace(row, allocated_by=remap.get(row.allocated_by, row.allocated_by),
        released_by=remap.get(row.released_by, row.released_by)) for row in original.storage_slots)
    orders = tuple(replace(row, before=remap.get(row.before, row.before),
        after=remap.get(row.after, row.after)) for row in original.order_constraints)
    result = replace(original, tasks=tuple(tasks), storage_slots=slots, order_constraints=orders,
        proof=proof, executable=False, temporal=None)
    validate_versioned_schedule(result)
    starts, completes = {}, {}
    for row in tasks:
        starts[row.id] = event(('task', row.id, 'start'), E.START, scope_of[row.id], task=row.id)
        completes[row.id] = event(('task', row.id, 'complete'), E.COMPLETION, scope_of[row.id], task=row.id)
    def edge(before, after, kind, key, **extra):
        constraints.append(TemporalConstraint(identity('constraint', *key), before, after, kind,
            source_refs=refs, **extra))
    ready = {}; material_ready = {}
    for material in original.materializations:
        name = event(('material', material.id), E.READY, loop,
            task=material.ready_task, materialization=material.id)
        material_ready[material.id] = name
        edge(entry if material.external_ready else completes[material.ready_task], name,
            C.READINESS, ('material', material.id), max_delay_ticks=0)
        if material.ready_task is not None:
            ready.setdefault(material.ready_task, []).append(name)
        for task in tasks:
            if material.id in task.reads:
                edge(name, starts[task.id], C.READINESS, ('read', material.id, task.id))
    capacities = {}
    for task in tasks:
        scope = scope_of[task.id]
        activity = identity('activity', task.id)
        activities.append(TemporalActivity(activity, task.id, scope, starts[task.id], completes[task.id],
            ready_events=tuple(ready.get(task.id, ())), source_refs=(*refs, *task.source_refs)))
        for dependency in task.depends_on:
            selection = {}
            if scope_of[dependency] != scope:
                selection = (dict(after_iteration=TemporalIterationSelection(loop, Select.FIRST))
                    if scope == loop else dict(before_iteration=TemporalIterationSelection(loop, Select.LAST_NORMAL)))
            edge(completes[dependency], starts[task.id], C.CAUSAL,
                ('dependency', dependency, task.id), **selection)
        resource = identity('unknown-service', task.id)
        capacities[resource] = None
        occupancies.append(TemporalOccupancy(identity('execution', task.id), O.EXECUTION_DEMAND,
            resource, starts[task.id], completes[task.id], scope,
            activity=activity, source_refs=refs))
    for slot in slots:
        if slot.ownership == 'SCHEDULE':
            resource = identity('slot-resource', slot.id)
            capacities[resource] = slot.capacity_bytes
            occupancies.append(TemporalOccupancy(identity('slot-occupancy', slot.id), O.PHYSICAL_SLOT_BYTES,
                resource, starts[slot.allocated_by], completes[slot.released_by], root,
                units=slot.capacity_bytes, storage_slot=slot.id, source_refs=refs))
    edge(exit, entry, C.LOOP_CARRIED, ('recurrence',), iteration_distance=1, loop_scope=loop)
    materials = {row.id: row for row in original.materializations}
    for carry in carries:
        edge(completes[materials[carry.before].ready_task], material_ready[carry.after],
            C.LOOP_CARRIED, ('content-carry', carry.before, carry.after),
            iteration_distance=carry.iteration_distance, loop_scope=loop)
    edge(completes[allocations[-1].id] if allocations else completes[outer_pre], entry,
        C.STORAGE_LIFETIME, ('first-entry',), after_iteration=TemporalIterationSelection(loop, Select.FIRST))
    if releases:
        edge(exit, starts[releases[0].id], C.STORAGE_LIFETIME, ('last-exit',),
            before_iteration=TemporalIterationSelection(loop, Select.LAST_NORMAL))
    certificate.update(task_map={row.id: remap.get(row.id, row.id) for row in original.tasks},
        body_task_ids=[row.id for row in tasks if scope_of[row.id] == loop],
        outer_task_ids=[row.id for row in tasks if scope_of[row.id] == root],
        activity_occurrences=[dict(activity=row.id, scope=row.scope,
            count=count if row.scope == loop else 1,
            meaning='PER_ITERATION_TEMPLATE' if row.scope == loop else 'ONCE_PER_NATIVE_SEQUENCE') for row in activities],
        storage_occurrences=[dict(slot=row.id, physical_count=1,
            loop_content_instances=count, capacity_bytes=row.capacity_bytes) for row in slots if row.ownership == 'SCHEDULE'],
        allocation_count_per_positive_sequence=len(allocations),
        release_count_per_positive_sequence=len(releases),
        body_exit_includes_helper_cleanup_and_progress=True,
        costs_estimated=False, unknown_costs_are_not_zero=True)
    model = TemporalModel(TemporalMode.SEMANTIC, tuple(events), tuple(activities), tuple(occupancies),
        tuple(constraints), tuple(scopes), resource_capacities=capacities, source_refs=refs,
        path_condition=TemporalPathCondition.NORMAL_COMPLETE)
    result = replace(result, temporal=model)
    validate_versioned_schedule(result)
    return result


__all__ = ['derive_workspace_time']

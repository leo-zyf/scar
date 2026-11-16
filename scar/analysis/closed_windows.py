"""Project source-closed islands without removing any original graph edge.

Pairwise data independence is a local proof under the existing input contract,
not permission to move work across an opaque boundary or to deploy workers.
"""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from scar.analysis.implementation_scope import derive_stateful_scope_envelope
from scar.analysis.target_graph import TargetHardwareGraph
from scar.ir.execution_recipe import artifact_reference, read_artifact, validate_recipe
from scar.ir.record_codec import encode
from scar.ir.temporal_resources import (ResourceProjection, projection_document,
    validate_resource_projection)
from scar.ir.versioned_schedule import (TemporalMode, TemporalScopeKind,
    VersionedTaskKind as K, schedule_document, schedule_from_dict,
    validate_versioned_schedule)
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.closed-window-projection'
_CLOSED = {K.SNAPSHOT: ('CPU_HOST_SNAPSHOT',),
           K.COMPUTE: ('CPU_SCALAR_DAG', 'CPU_SCALAR_SETUP'),
           K.STATE_WRITE: ('CPU_STATE_WRITE',)}


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _closed(task):
    return (task.capability in _CLOSED.get(task.kind, ())
            and task.attrs.get('stage_kind') != 'PROVIDER')


def _source_contracts():
    """Admit the reviewed snapshot loop and synchronous serial bind statement."""
    from scar.codegen.boundary_cuts import _fragment, _serial_lines
    root = Path(__file__).resolve().parents[1]
    renderer_path = root/'codegen/stateful.py'; renderer = renderer_path.read_text()
    snapshot, receipt = _fragment(renderer_path, renderer,
        "            if kind=='SNAPSHOT_ROOT':", "            elif kind=='COMPUTE':")
    expected = (
        "            if kind=='SNAPSHOT_ROOT':\n"
        "                target=stage['target']; source=f'static_cast<const double *>(arrays[{arrays.index(target)}].data)'\n"
        "                inner=[f'{payload(output)}[row * {cols} + col] = {source}[row * {cols} + col];']\n")
    _need(snapshot == expected, 'closed raw snapshot source contract changed')
    loop, loop_receipt = _fragment(renderer_path, renderer,
        "            if kind!='PROVIDER':", '            if output and root:')
    expected_loop = (
        "            if kind!='PROVIDER':\n"
        "                body=([\"#pragma omp for schedule(static)\"] if parallel else [])+[f'for (int64_t row = 0; row < {rows}; ++row) {{',f'    for (int64_t col = 0; col < {cols}; ++col) {{',*['        '+line for line in inner],'    }','}']\n")
    _need(loop == expected_loop, 'closed snapshot iteration source contract changed')
    bind, check, serial = _serial_lines(renderer_path, renderer)
    loop_path = root/'codegen/loop_v1.cpp.in'; helper = loop_path.read_text()
    _, bind_helper = _fragment(loop_path, helper, '    bool bind(int cpu) noexcept {',
        '    bool restore() noexcept {\n        if (!changed_)')
    return dict(rule=artifact_reference(Path(__file__).resolve(), 'scar.closed-window-source-rule'),
        snapshot=dict(contract='SERIAL_RAW_DOUBLE_LOAD_STORE_TO_FRESH_PRIVATE_SLOT_NO_DISPATCH_OR_FP_ARITHMETIC',
            source=receipt, iteration_source=loop_receipt, conditional_on_original_valid_stable_input_contract=True,
            no_task_local_numeric_guard_or_fp_status_mutation=True),
        serial_bind=dict(renderer=serial, helper=bind_helper,
            bind_statement=bind, final_bound_check=check,
            normal_return_means_calling_tid_synchronous_bind_completed=True,
            background_work_or_future_affinity_preservation_proven=False))


def _admit(schedule, resources, recipe):
    _need(type(resources) is ResourceProjection, 'qualified typed resource projection required')
    validate_versioned_schedule(schedule)
    _need(schedule.temporal is not None and schedule.temporal.mode is TemporalMode.SEMANTIC
          and not any(s.kind is TemporalScopeKind.LOOP for s in schedule.temporal.scopes),
          'one original standalone semantic implementation scope required')
    recipe = validate_recipe(recipe, verify_artifacts=False)
    _need(recipe['control'].get('pool_release_order') == 'CPP_STD_ARRAY_REVERSE_DESTRUCTION'
          and recipe['control'].get('spatial_partition') == 'SERIAL_ROWS',
          'modern source-replayed serial physical-release recipe required')
    replay = derive_stateful_scope_envelope(schedule_from_dict(recipe['task_graph']), recipe)
    _need(replace(schedule, temporal=None) == replay,
          'complete implementation scope differs from current source replay')
    order = replay.proof['implementation_envelope']['phase_order']
    _need(order == [t.id for t in schedule.tasks], 'complete original implementation phase order required')
    activities = {a.task: a for a in schedule.temporal.activities}
    _need(len(activities) == len(schedule.temporal.activities) == len(schedule.tasks)
          and set(activities) == set(order), 'one original activity per complete source task required')
    target_ref = encode(resources.target)
    _need(target_ref == recipe['artifacts']['target_snapshot'],
          'source recipe and qualified resources have different targets')
    graph = TargetHardwareGraph(Path(target_ref['uri']).parent)
    _need(artifact_reference(graph.path/'result.json', 'scar.target-hardware-graph') == target_ref
          and read_artifact(target_ref) == graph.report, 'original target snapshot differs')
    validate_resource_projection(resources, schedule, resources=graph.resources())
    plan = read_artifact(recipe['artifacts']['stage_plan'])
    region = read_artifact(recipe['artifacts']['region'])
    sources = [region['source'], *(row['source'] for row in region.get('callees', ()))]
    for row in sources:
        _need(sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256'],
              'original source bytes changed after caller admission')
    for role, ref in recipe['artifacts'].items():
        if role.startswith(('source_python:', 'source_artifact:')):
            read_artifact(ref, json_document=False)
    return recipe, plan, activities, _source_contracts()


def derive_closed_windows(schedule, resources, recipe, *, source_refs=()):
    """Return structural windows and explicit boundary/frontier obligations.

    The caller must first cold-replay its time/resource preparation receipts.
    This layer replays the complete source scope and retained inventory without
    repeating the F4 frontend, CPU cost domain, or scope-demand derivation.
    """
    recipe, plan, activities, contracts = _admit(schedule, resources, recipe)
    _need(type(source_refs) in (list, tuple) and all(type(r) is dict for r in source_refs),
          'source references must retain original qualified receipt dictionaries')
    tasks = {t.id: t for t in schedule.tasks}; positions = {t.id: i for i, t in enumerate(schedule.tasks)}
    materials = {m.id: m for m in schedule.materializations}
    slots = {s.id: s for s in schedule.storage_slots}
    consumers = {name: [] for name in materials}
    for task in schedule.tasks:
        for name in task.reads:
            consumers[name].append(task.id)
    stage_rows = {row['id']: row for row in plan['stages']}
    stage_positions = {row['id']: i for i, row in enumerate(plan['stages'])}
    scalar_producers = {row['root_expr']: row['id'] for row in plan['stages'] if row['kind'] == 'SCALAR'}
    binding_rows = {row.occupancy: i for i, row in enumerate(resources.executions)}
    storage_rows = {row.slot: i for i, row in enumerate(resources.storages)}
    demands = {a.id: [] for a in activities.values()}
    for occupancy in schedule.temporal.occupancies:
        if occupancy.activity is not None:
            demands[occupancy.activity].append(occupancy)
    task_constraints = {name: [] for name in tasks}
    events = {e.id: e for e in schedule.temporal.events}
    for i, edge in enumerate(schedule.temporal.constraints):
        for task_id in {events[edge.before].task, events[edge.after].task} - {None}:
            task_constraints[task_id].append(i)
    assumptions = recipe['semantic_boundary'].get('external_assumptions', ())
    required_inputs = [dict(condition=value, authority='ORIGINAL_EXTERNAL_ASSUMPTION', proven=False,
        parent_document='recipe', source_ref='#/semantic_boundary/external_assumptions/'+str(i))
        for i, value in enumerate(assumptions)]

    def demand_refs(task):
        return [dict(occupancy=o.id, parent_document='resources',
            binding_locator='#/executions/'+str(binding_rows[o.id]),
            units=o.units, resource=o.resource, exact_thread=encode(resources.executions[binding_rows[o.id]].exact_thread),
            source_refs=encode(resources.executions[binding_rows[o.id]].source_refs))
            for o in demands[activities[task.id].id]]

    def material_port(name):
        material = materials[name]
        return dict(materialization=name, value=material.value, version=material.version,
            region=encode(material.region), ready_task=material.ready_task,
            external_ready=material.external_ready,
                parent_document='schedule',
                source_locator='#/materializations/'+str(next(i for i, m in enumerate(schedule.materializations) if m.id == name)))

    def overlaps(left, right):
        a, b = materials[left].region, materials[right].region
        return a.slot == b.slot and a.nbytes > 0 and b.nbytes > 0 and (
            a.offset_bytes < b.offset_bytes+b.nbytes and b.offset_bytes < a.offset_bytes+a.nbytes)

    def hazards(left, right):
        result = []
        for kind, first, second in (('RAW', left.writes, right.reads),
                                    ('WAR', left.reads, right.writes), ('WAW', left.writes, right.writes)):
            for a in first:
                for b in second:
                    if overlaps(a, b):
                        result.append(dict(kind=kind, before_material=a, after_material=b,
                            slot=materials[a].region.slot))
        return result

    # This is an evidence projection of existing storage conflicts, not added
    # source edges or a claim that disjoint logical IDs imply disjoint storage.
    all_hazards = []
    ordered = list(schedule.tasks)
    for i, left in enumerate(ordered):
        for right in ordered[i+1:]:
            for row in hazards(left, right):
                all_hazards.append(dict(before=left.id, after=right.id, **row))

    def semantic_predecessors(task):
        result = {materials[m].ready_task for m in task.reads if materials[m].ready_task is not None}
        stage = stage_rows.get(task.id)
        if stage is not None:
            result.update(scalar_producers[s] for s in stage['scalar_inputs'] if s in scalar_producers)
        return result

    def pair(left, right):
        storage_conflicts = hazards(left, right)
        producers = sorted(({left.id} & semantic_predecessors(right)) | ({right.id} & semantic_predecessors(left)))
        worlds = bool(left.world_before or left.world_after or right.world_before or right.world_after)
        order_edges = []
        other_dependencies = []
        for before, after in ((left, right), (right, left)):
            for i, dependency in enumerate(after.depends_on):
                if dependency != before.id:
                    continue
                stage = stage_rows.get(after.id)
                previous = stage_positions.get(after.id, 0)-1
                is_sequence = (stage is not None and previous >= 0
                    and plan['stages'][previous]['id'] == before.id
                    and before.id not in semantic_predecessors(after))
                row = dict(before=before.id, after=after.id,
                    parent_document='schedule',
                    source_locator='#/tasks/'+str(positions[after.id])+'/depends_on/'+str(i))
                (order_edges if is_sequence else other_dependencies).append(row)
        for i, edge in enumerate(schedule.order_constraints):
            if {edge.before, edge.after} == {left.id, right.id}:
                row = dict(before=edge.before, after=edge.after,
                    parent_document='schedule', source_locator='#/order_constraints/'+str(i))
                (order_edges if edge.kind == 'SOURCE_ORDER' else other_dependencies).append(row)
        snapshot = all(t.kind is K.SNAPSHOT and t.capability == 'CPU_HOST_SNAPSHOT'
            and t.attrs.get('stage_kind') == 'SNAPSHOT_ROOT' and not t.validates
            and len(t.writes) == 1 and slots[materials[t.writes[0]].region.slot].ownership == 'SCHEDULE'
            for t in (left, right))
        reasons = []
        if storage_conflicts: reasons.append('OVERLAPPING_PHYSICAL_REGION_RAW_WAR_OR_WAW')
        if producers: reasons.append('CONTENT_OR_SCALAR_PRODUCER_CONSUMER')
        if worlds: reasons.append('ORIGINAL_WORLD_TOKEN_ORDER_NOT_DISCHARGED')
        if other_dependencies: reasons.append('ORIGINAL_NON_SEQUENCE_DEPENDENCY_NOT_DISCHARGED')
        if not snapshot: reasons.append('NUMERIC_GUARD_FP_OR_EXCEPTION_EFFECT_COMMUTATION_NOT_PROVEN')
        status = ('PAIRWISE_DATA_INDEPENDENT' if not reasons else
                  'DATA_INDEPENDENT_BUT_EFFECT_ORDER_UNPROVEN' if not (
                      storage_conflicts or producers or worlds or other_dependencies) else 'ORDER_DEPENDENT_OR_UNPROVEN')
        return dict(before=left.id, after=right.id, status=status,
            proof=dict(storage_region_conflicts=storage_conflicts, producer_consumer=producers,
                world_or_token_conflict=worlds, non_sequence_dependencies=other_dependencies,
                candidate_source_order_edges=order_edges, original_edges_removed=False,
                source_closed_contract='#/source_contracts/snapshot' if snapshot else None,
                requirements=deepcopy(required_inputs), remaining_obligations=reasons,
                same_original_buffers_and_no_arithmetic_rewrite=True,
                parallel_execution_or_movement_authorized=False))

    backend_obligations = (
        'JOIN_ALL_NEW_WORKERS_BEFORE_ENTERING_ORIGINAL_ANCHOR',
        'RESUME_ORIGINAL_CALLING_TID_WITH_REQUIRED_FP_ENVIRONMENT_AND_TLS',
        'NO_BORROWED_POINTER_OR_UNCOMMITTED_OUTPUT_ESCAPES_NEW_WORKERS',
        'AGGREGATE_FAILURES_BEFORE_ORIGINAL_SUCCESS_FENCE_AND_EXTERNAL_COMMIT')
    anchors = []
    for task in ordered:
        if _closed(task):
            continue
        phase = task.attrs.get('phase_kind')
        proven = [dict(condition='ORIGINAL_ANCHOR_IDENTITY_AND_SOURCE_ORDER_PRESERVED', authority='SOURCE_SCOPE_REPLAY')]
        required = [*deepcopy(required_inputs),
            dict(condition='ANCHOR_RETURNS_WITH_NO_OUTSTANDING_READERS_OR_WRITERS_OF_BORROWED_PORTS',
                authority='REQUIRED_UNPROVEN_NOT_IMPLIED_BY_RETURN_VALUE', proven=False),
            dict(condition='NO_BORROWED_PORT_ESCAPE_OR_ASYNC_CALLBACK_USE', authority='REQUIRED_UNPROVEN', proven=False),
            dict(condition='CALLER_FP_TLS_AND_EXECUTION_DOMAIN_VALID_AT_NEXT_WINDOW', authority='REQUIRED_UNPROVEN', proven=False)]
        required.extend(dict(condition=value, authority='REQUIRED_UNPROVEN_ENVIRONMENT_Q', proven=False)
            for value in ('NO_FOREIGN_CALLING_TID_AFFINITY_MUTATION_WITHIN_NEW_WINDOW',
                          'SELECTED_CPU_ONLINE_AND_EFFECTIVE_CPUSET_STABLE_WITHIN_NEW_WINDOW'))
        if phase == 'TEAM_ENTER':
            _need(task.attrs.get('phase_effects') == ['BIND_CALLER_TO_SELECTED_SINGLE_CPU'],
                  'original source serial bind phase differs')
            proven.append(dict(condition='SCAR_SYNCHRONOUS_CALLING_TID_BIND_COMPLETED_ON_NORMAL_RETURN',
                source_ref='#/source_contracts/serial_bind', authority='EXACT_SOURCE_STATEMENT_AND_HELPER'))
        provider = task.attrs.get('stage_kind') == 'PROVIDER'
        if provider:
            required.extend([dict(condition='PROVIDER_INTERNAL_HELPERS_QUIESCENT_AND_PORTS_NOT_RETAINED',
                authority='REQUIRED_UNPROVEN_NO_REQUIRES_PYAPI_INFERENCE', proven=False),
                dict(condition='PROVIDER_RETURN_DOES_NOT_ESTABLISH_SELECTED_CPU_DOMAIN',
                     authority='FRONTIER_KEEP_PARENT_UNKNOWN', proven=False)])
        anchors.append(dict(task_id=task.id, activity_id=activities[task.id].id,
            kind=task.kind.value, phase_kind=phase, source_refs=encode(task.source_refs),
            resource_refs=demand_refs(task), source_order_index=positions[task.id],
            boundary_contract=dict(proven=proven, required_unproven=required,
                backend_obligations=[dict(condition=value, status='BACKEND_OBLIGATION') for value in backend_obligations],
                outstanding_effects_closed=False, new_window_ready_authorized=False)))

    groups = []; pending = []
    for task in ordered:
        if _closed(task):
            pending.append(task)
        elif pending:
            groups.append(pending); pending = []
    if pending:
        groups.append(pending)
    windows = []
    for index, group in enumerate(groups):
        members = {t.id for t in group}
        read_ids = list(dict.fromkeys(m for t in group for m in t.reads))
        write_ids = list(dict.fromkeys(m for t in group for m in t.writes))
        inputs = [m for m in read_ids if materials[m].ready_task not in members]
        outputs = [m for m in write_ids if any(c not in members for c in consumers[m])]
        slot_ids = list(dict.fromkeys(materials[m].region.slot for m in (*read_ids, *write_ids)))
        left_index, right_index = positions[group[0].id]-1, positions[group[-1].id]+1
        left = ordered[left_index].id if left_index >= 0 else None
        right = ordered[right_index].id if right_index < len(ordered) else None
        pairs = [pair(a, b) for i, a in enumerate(group) for b in group[i+1:]]
        frontiers = [dict(reason='BOUNDARY_CONTRACT_NOT_DISCHARGED', anchor=left),
            dict(reason='BACKEND_JOIN_CALLER_FP_TLS_AND_NO_ESCAPE_OBLIGATIONS', anchor=right)]
        if left is not None and tasks[left].attrs.get('stage_kind') == 'PROVIDER':
            frontiers.append(dict(reason='OPAQUE_PROVIDER_OUTSTANDING_EFFECTS_AND_CPU_DOMAIN_UNPROVEN',
                anchor=left, next_window_forced_ready=False))
        windows.append(dict(id='closed-window:'+str(index), task_ids=[t.id for t in group],
            activity_ids=[activities[t.id].id for t in group], left_anchor=left, right_anchor=right,
            live_inputs=[material_port(m) for m in inputs], live_outputs=[material_port(m) for m in outputs],
            preserved_unconsumed_outputs=[material_port(m) for m in write_ids if m not in outputs
                and not consumers[m]],
            borrowed_slots=[dict(slot=encode(slots[name]),
                parent_document='resources',
                binding_locator='#/storages/'+str(storage_rows[name]) if name in storage_rows else None,
                newly_allocated=False, lifetime_changed=False) for name in slot_ids],
            memory_budget=encode(resources.budget), resource_refs=[dict(task_id=t.id, bindings=demand_refs(t)) for t in group],
            original_order_refs=dict(parent_document='schedule',
                task_dependencies=[dict(task_id=t.id, source_locator='#/tasks/'+str(positions[t.id])+'/depends_on') for t in group],
                order_constraints=[i for i, e in enumerate(schedule.order_constraints) if e.before in members or e.after in members],
                temporal_constraints=sorted({i for t in group for i in task_constraints[t.id]})),
            storage_hazards=[i for i, h in enumerate(all_hazards) if h['before'] in members or h['after'] in members],
            pairs=pairs, frontiers=frontiers,
            original_source_order_preserved=True, complete_lowering_authority=False))
    result = dict(schema=SCHEMA, schema_version=1,
        source=dict(recipe_id=recipe['recipe_id'], source_refs=encode(tuple(source_refs)),
            schedule_sha256=semantic_sha256(schedule_document(schedule)),
            resources_sha256=semantic_sha256(projection_document(resources)), target=encode(resources.target),
            artifacts=deepcopy(recipe['artifacts']), source_scope=deepcopy(schedule.proof['implementation_scope_cleanup'])),
        source_contracts=contracts, windows=windows, anchors=anchors,
        parent_documents=dict(schedule=dict(semantic_sha256=semantic_sha256(schedule_document(schedule))),
            resources=dict(semantic_sha256=semantic_sha256(projection_document(resources)),
                source_refs=encode(resources.source_refs)),
            recipe=dict(recipe_id=recipe['recipe_id'])),
        original_storage_hazards=all_hazards,
        original_constraints=dict(order_constraint_count=len(schedule.order_constraints),
            temporal_constraint_count=len(schedule.temporal.constraints),
            all_order_constraint_ordinals=list(range(len(schedule.order_constraints))),
            all_temporal_constraint_ordinals=list(range(len(schedule.temporal.constraints))),
            source_schedule_unchanged=True),
        parent_resource_frontiers=encode(resources.unresolved),
        summary=dict(task_count=len(tasks), activity_count=len(activities), window_count=len(windows),
            anchor_count=len(anchors), eligible_structure_count=sum(any(
                p['status'] == 'PAIRWISE_DATA_INDEPENDENT' for p in w['pairs']) for w in windows),
            pairwise_data_independent_count=sum(p['status'] == 'PAIRWISE_DATA_INDEPENDENT'
                for w in windows for p in w['pairs']),
            original_edges_removed=0, original_tasks_removed=0,
            complete_lowering_authority=False, solver_authorized=False,
            native_execution_authorized=False, physical_optimality_proven=False))
    return result


__all__ = ['SCHEMA', 'derive_closed_windows']

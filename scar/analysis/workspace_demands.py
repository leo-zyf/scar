"""Source-qualified serial CPU domains for a shared workspace loop template.

The selected domain is established afresh per successful iteration preflight.
Opaque work invalidates forward placement knowledge, including for later closed
work. This does not measure threads, reserve a CPU, infer costs or execute code.
"""
from dataclasses import replace
from pathlib import Path

from scar.analysis.target_graph import TargetHardwareGraph
from scar.analysis.workspace_time import derive_workspace_time
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program import target_hardware as H
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import encode
from scar.ir.versioned_schedule import (TemporalMode, TemporalPathCondition,
    TemporalScopeKind as SK, TemporalOccupancyKind as OK, VersionedTaskKind as K,
    schedule_document, validate_versioned_schedule)
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.workspace-cpu-demand-source-proof'


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def derive_workspace_demands(schedule, step_plan, native_metadata, *, native_reference,
        cpp_reference, workspace_plan_reference, parent_schedule_reference, target):
    """Refine an admitted baseline; deterministic replay does not open a compiler."""
    from scar.analysis.workspace_source import workspace_serial_source_contract
    validate_versioned_schedule(schedule)
    parent = schedule_document(schedule)
    temporal = schedule.temporal
    _need(parent['schema_version'] == 4 and temporal.mode is TemporalMode.SEMANTIC
          and temporal.path_condition is TemporalPathCondition.NORMAL_COMPLETE,
          'normal-complete semantic workspace v4 baseline required')
    _need(not any(row.get('schema') == SCHEMA for row in temporal.source_refs),
          'workspace demands must derive from their original unrefined baseline')
    _need(read_artifact(parent_schedule_reference) == parent,
          'parent schedule reference differs from the exact workspace baseline')
    _need(read_artifact(native_reference) == native_metadata,
          'original native workspace metadata reference differs')
    workspace = read_artifact(workspace_plan_reference)
    base_reference = native_metadata['base_native_reference']
    base = read_artifact(base_reference)
    plan_ref = artifact_reference(Path(base_reference['uri']).parent/base['files']['plan'],
                                  'scar.stateful-stage-plan')
    _need(read_artifact(plan_ref) == encode(step_plan)
          and semantic_sha256(step_plan) == workspace['step_plan_sha256'],
          'source-qualified step plan differs from original workspace/base plan')
    replay = derive_workspace_time(workspace, native_manifest=native_reference,
        native_metadata=native_metadata, cpp_source=cpp_reference, workspace_reference=workspace_plan_reference)
    _need(replay == schedule, 'workspace baseline differs from deterministic original source projection')
    contract = workspace_serial_source_contract(native_metadata, step_plan,
        native_reference=native_reference, cpp_reference=cpp_reference)
    cpus = contract['cpu_ids']; binding = step_plan['resource_binding']
    _need(type(cpus) in (tuple, list) and len(cpus) == 1 and type(cpus[0]) is int
          and step_plan['cpu_threads'] == 1 and list(binding['cpu_ids']) == list(cpus),
          'one original serial selected CPU required')
    if type(target) is TargetHardwareGraph:
        graph = target
    else:
        path = Path(target).resolve(); graph = TargetHardwareGraph(path.parent if path.is_file() else path)
    snapshot = artifact_reference(graph.path/'result.json', 'scar.target-hardware-graph')
    _need(snapshot == step_plan['resource_context']['target_snapshot']
          and binding['target_sha256'] == snapshot['sha256']
          and Path(binding['target_path']).resolve() == graph.path.resolve(),
          'selected workspace CPU and retained target snapshot differ')
    inventory = graph.resources()
    threads = [r for r in inventory if type(r) is H.HardwareThread and r.cpu_id == cpus[0]]
    executors = [r for r in inventory if type(r) is H.CpuExecutor
                 and str(r.id) == binding['executor_id'] and cpus[0] in r.cpu_ids]
    _need(len(threads) == len(executors) == 1 and threads[0].online is not False,
          'original selected thread/executor is missing or offline')
    thread = threads[0]; selected_resource = str(thread.id)
    _need(any(e.kind is H.ConnectionKind.CONTAINS and e.source == executors[0].id
              and e.target == thread.id for e in graph.connections()),
          'selected executor/thread has no retained containment edge')
    activities = {a.task: a for a in temporal.activities}
    _need(len(activities) == len(temporal.activities) and set(activities) == {t.id for t in schedule.tasks},
          'one synchronous workspace lifecycle per original task required')
    loops = [s for s in temporal.scopes if s.kind is SK.LOOP]
    _need(len(loops) == int(workspace['trip_count'] > 0), 'one active shared iteration template required')
    loop = loops[0] if loops else None
    def phase(kind):
        rows = [t for t in schedule.tasks if t.kind is K.IMPLEMENTATION_PHASE and t.attrs.get('phase_kind') == kind]
        _need(len(rows) == 1 and activities[rows[0].id].scope == loop.id,
              'one original loop-local '+kind+' phase required')
        return rows[0]
    enter = phase(contract['enter_phase_kind']) if loop else None
    restore = phase(contract['restore_phase_kind']) if loop else None
    _need(contract['enter_fact_at'] == 'COMPLETION' and contract['restore_kill_at'] == 'START',
          'source bind/restore lifecycle anchors differ')
    adjacency = {}
    for edge in temporal.constraints:
        if not edge.iteration_distance and edge.before_iteration is None and edge.after_iteration is None:
            adjacency.setdefault(edge.before, set()).add(edge.after)
    for a in activities.values():
        adjacency.setdefault(a.start, set()).add(a.completion)
    def reaches(before, after):
        pending = [before]; seen = {before}
        while pending:
            point = pending.pop()
            if point == after:
                return True
            for following in adjacency.get(point, ()):
                if following not in seen:
                    seen.add(following); pending.append(following)
        return False
    if loop:
        _need(reaches(activities[enter.id].completion, activities[restore.id].start),
              'successful per-iteration bind/restore happens-before window missing')
    by_activity = {}
    for occupancy in temporal.occupancies:
        if occupancy.kind is OK.EXECUTION_DEMAND:
            by_activity.setdefault(occupancy.activity, []).append(occupancy)
    closed_rules = contract['closed_stages']
    state = 'ORIGINAL_CALLER_DOMAIN'; replacements = {}; rows = []; trace = []
    capacities = dict(temporal.resource_capacities)
    for task in schedule.tasks:
        a = activities[task.id]; before = state
        body = loop is not None and a.scope == loop.id
        entering, restoring = task is enter, task is restore
        opaque = (task.attrs.get('stage_kind') in contract['opaque_stage_kinds']
                  or task.capability in contract['opaque_capabilities'])
        matches = [r for r in closed_rules if r['kind'] == task.kind.value
                   and r['capability'] == task.capability and r['stage_kind'] == task.attrs.get('stage_kind')]
        _need(len(matches) <= 1, 'ambiguous closed serial source-stage rule')
        closed = bool(matches)
        if closed:
            _need(matches[0]['whole_thread_units'] == 1 and matches[0]['affinity_preserving'] is True,
                  'closed stage lacks whole-thread/preservation source contract')
        if restoring:
            state = 'RESTORED_ORIGINAL_DOMAIN'
        elif body and opaque:
            state = 'OPAQUE_UNKNOWN'
        elif body and not entering and not closed and state == 'SELECTED_SINGLETON':
            state = 'OPAQUE_UNKNOWN'
        exact = body and closed and not opaque and state == 'SELECTED_SINGLETON' and not entering and not restoring
        if exact:
            _need(reaches(activities[enter.id].completion, a.start)
                  and reaches(a.completion, activities[restore.id].start),
                  'selected iteration stage is not inside source bind/restore bounds')
        occupancies = by_activity.get(a.id, ())
        if getattr(a.duration_basis, 'value', None) == 'SOURCE_CONTROL_MILESTONE':
            _need(not occupancies, 'source milestone cannot acquire physical execution demand')
        else:
            _need(len(occupancies) == 1, 'one original execution occupancy per workspace task required')
            old = occupancies[0]
            _need(old.start == a.start and old.end == a.completion and old.units is None
                  and temporal.resource_capacities[old.resource] is None,
                  'baseline execution lifecycle or UNKNOWN demand was changed')
            resource, units = (selected_resource if exact else old.resource), (1 if closed and not opaque else None)
            replacements[old.id] = (resource, units)
            rows.append(dict(occupancy=old.id, task=task.id, activity=a.id, scope=a.scope,
                domain_state=state if body else 'ORIGINAL_CALLER_DOMAIN',
                cpu_ids=[cpus[0]] if exact else None, exact_cpu_id=cpus[0] if exact else None,
                resource=resource, units=units,
                unit_authority='CLOSED_SERIAL_SOURCE_STAGE' if units == 1 else 'WHOLE_SCOPE_THREAD_DEMAND_UNRESOLVED',
                caller_participation_units=1, caller_participation_is_total_scope_demand=False,
                transition_scope=entering or restoring))
            if exact:
                _need(capacities.get(selected_resource) in (None, 1), 'selected thread capacity conflicts')
                capacities[selected_resource] = 1
        if entering:
            state = 'SELECTED_SINGLETON'
        trace.append(dict(task=task.id, scope=a.scope, before=before, after=state,
            opaque_preservation_unproven=opaque, selected_cpu_proven=exact,
            reason='SUCCESSFUL_PER_ITERATION_BIND' if entering else 'RESTORE_START_KILLS_SELECTED_DOMAIN'
                if restoring else 'OPAQUE_SCOPE_KILLS_FORWARD_KNOWLEDGE' if opaque
                else 'CLOSED_SERIAL_STAGE' if closed else 'UNRESOLVED_WHOLE_SCOPE'))
    proof = dict(schema=SCHEMA, schema_version=1, parent_schedule=parent_schedule_reference,
        parent_schedule_sha256=semantic_sha256(parent), target=snapshot, stage_plan=plan_ref,
        native_manifest=native_reference, cpp_source=cpp_reference, workspace_plan=workspace_plan_reference,
        source_rule=artifact_reference(Path(__file__).resolve(), 'scar.source-demand-rule'),
        source_contract=contract, selected_cpu_id=cpus[0],
        selected_thread=encode(H.TargetResourceRef(ProviderArtifact(**snapshot), thread.id)),
        occupancies=rows, transition_trace=trace,
        iteration_qualification='REBIND_AFTER_EACH_SUCCESSFUL_PER_STEP_CONTEXT_NOT_INHERITED_FROM_PREVIOUS_ITERATION',
        conditions=['NORMAL_COMPLETION_ONLY', 'ORIGINAL_SERIAL_SOURCE_AND_SELECTED_TARGET_UNCHANGED',
                    'OPAQUE_STAGE_AFFINITY_EFFECTS_INVALIDATE_FORWARD_KNOWLEDGE'],
        actual_cpu_observation=False, global_reservation=False, cost_inferred=False,
        native_execution_authorized=False)
    proof_index = len(temporal.source_refs)
    positions = {row['occupancy']: i for i, row in enumerate(rows)}
    occupancies = tuple(replace(o, resource=replacements[o.id][0], units=replacements[o.id][1],
        source_refs=(*o.source_refs, dict(schema=SCHEMA,
            locator='#/temporal/source_refs/'+str(proof_index)+'/occupancies/'+str(positions[o.id]))))
        if o.id in replacements else o for o in temporal.occupancies)
    result = replace(schedule, temporal=replace(temporal, occupancies=occupancies,
        resource_capacities=capacities, source_refs=(*temporal.source_refs, proof)))
    validate_versioned_schedule(result)
    read_artifact(snapshot)
    return result


__all__ = ['SCHEMA', 'derive_workspace_demands']

"""Replay source-qualified serial CPU demands on retained semantic time graphs."""

from dataclasses import replace
from pathlib import Path

from scar.analysis.schedule_time import attach_task_time
from scar.analysis.target_graph import TargetHardwareGraph
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program import target_hardware as H
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import encode
from scar.ir.versioned_schedule import (
    TemporalMode, TemporalOccupancyKind as OK, TemporalPathCondition, TemporalScopeKind,
    VersionedTaskKind as K, schedule_document, schedule_from_dict, validate_versioned_schedule,
)
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.serial-scope-demand-source-proof'
WORKSPACE_PROOF = 'scar.workspace-cpu-demand-source-proof'
AFFINITY_PROOF = 'scar.serial-affinity-demand-source-proof'
_SOURCE_SCHEMA = 'scar.source-demand-rule'
# Exact predecessors may be used only in a complete semantic replay.
_PREVIOUS_RULES = {
    SCHEMA: ('scope_demands.py', 'c8d94031b6b04ce38261c62a815bdd2348063e4048a8f9649345fa7f047debb5', 16075),
    WORKSPACE_PROOF: ('workspace_demands.py', '99509e01e62b4de814a48ad956c8bccb02898e50c6835b95a443613fe3b36251', 12369),
}


_PHASE_FRAGMENTS = (
    ('metadata_prefix', 'stateful_v1.cpp.in',
     '    if (array_count != @ARRAY_COUNT@ || scalar_count != @SCALAR_COUNT@ ||',
     '    std::array<std::unique_ptr<double[]>, @SLOT_COUNT@> slots;',
     'dade9e51ecbd724d7f408a092c6236c204853347dd7461018c2d876e3e76ff17'),
    ('array_helpers', 'stateful_v1.cpp.in',
     'bool valid_array(const scar_array_v1 &a, int64_t rows, int64_t cols) {', '\nstruct Scalar {',
     '5b651079bac958113d3ad109bb9d30288537f9386b84cda4f25376b4f9cf6a25'),
    ('array_guard_renderer', 'stateful.py',
     "    guards=[f'    if (!valid_array(arrays[", '\n    finite=',
     '9fcc2ce7c2541ac018ebd04550aa2f803d3d481fd19312df771080d372a26b8e'),
    ('finite_helper', 'stateful_v1.cpp.in', 'bool finite(double value) {', 'bool valid_array(',
     '9238433de045e1b20fcfb1c2450488f556596cc5ab3440122d88077a02aca1cd'),
    ('finite_scan_renderer', 'stateful.py', "    finite=[f'    for (int64_t i = 0;", '\n    replacements=dict(',
     '09b9cef05a17f7670cc2ec279b0b0d4c7feadeac56c7f9ab319849981b514740'),
    ('native_return', 'stateful_v1.cpp.in', '@COMMITS@', '\nconst scar_stateful_table_v1 table',
     '6a658e3afc72044c052820ffc3f9be877fabbc53c588ed12852af362839f6571'),
)
_PHASE_EFFECTS = {
    'NATIVE_METADATA_GUARD': ['READ_NATIVE_ABI_DESCRIPTORS', 'CHECK_ARRAY_PROVIDER_ALIAS_AND_PAYLOAD_METADATA'],
    'INPUT_FINITE_SCAN_SERIAL': ['READ_ORIGINAL_EXTERNAL_ARRAY_CONTENT', 'CHECK_EACH_FLOAT64_VALUE_IS_FINITE'],
    'NATIVE_RETURN': ['RETURN_NATIVE_SUCCESS_STATUS'],
}


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _recipe_reference(schedule, recipe):
    refs = {}
    for row in schedule.temporal.source_refs:
        ref = row.get('artifact')
        if type(ref) is dict and ref.get('schema') == recipe['schema']:
            refs[semantic_sha256(ref)] = ref
    _need(len(refs) == 1, 'one original recipe ArtifactRef required for source scope demand replay')
    reference = next(iter(refs.values()))
    _need(read_artifact(reference) == recipe, 'source scope recipe reference differs from original recipe')
    return reference


def _closed_phase_sources(schedule):
    from scar.codegen.boundary_cuts import _fragment
    directory = Path(__file__).resolve().parents[1]/'codegen'
    fragments = {}
    for name, filename, start, end, expected in _PHASE_FRAGMENTS:
        path = directory/filename
        _, receipt = _fragment(path, path.read_text(), start, end)
        _need(receipt['extracted_sha256'] == expected,
              'closed serial phase source contract changed: ' + name)
        fragments[name] = receipt
    source = schedule.proof['implementation_scope_cleanup']['source']
    _need(source['fragments']['commit_and_native_return']['extracted_sha256']
              == fragments['native_return']['extracted_sha256']
          and source['helper_cleanup_before_pool_release'] is True
          and source['reverse_raii_order'] == ['FloatingScope', 'ThreadAffinity', 'WORKSPACE_POOL_UNIQUE_PTR_ARRAY'],
          'literal native return needs the original separated helper/pool cleanup scope')
    return dict(
        NATIVE_METADATA_GUARD=dict(fragments={key: fragments[key] for key in (
            'metadata_prefix', 'array_helpers', 'array_guard_renderer')},
            contract='SERIAL_ABI_AND_BOUNDS_CHECKS_NO_ALLOCATION_PROVIDER_OS_OR_VIRTUAL_DISPATCH'),
        INPUT_FINITE_SCAN_SERIAL=dict(fragments={key: fragments[key] for key in (
            'finite_helper', 'finite_scan_renderer')},
            contract='SERIAL_RAW_FLOAT64_READ_AND_FINITE_BIT_TEST_NO_EXTERNAL_DISPATCH'),
        NATIVE_RETURN=dict(fragments={'native_return': fragments['native_return']},
            contract='LITERAL_SUCCESS_STATUS_AFTER_SEPARATELY_MODELED_RAII_AND_POOL_RELEASE_NOT_FFI_HANDOFF'))


def _selected_thread(target, binding, expected):
    if type(target) is TargetHardwareGraph:
        graph = target
    else:
        path = Path(target).resolve()
        graph = TargetHardwareGraph(path.parent if path.is_file() else path)
    snapshot = artifact_reference(graph.path / 'result.json', 'scar.target-hardware-graph')
    _need(snapshot == expected and binding['target_sha256'] == snapshot['sha256']
          and Path(binding['target_path']).resolve() == graph.path.resolve(),
          'selected CPU and original target snapshot differ')
    cpu, = binding['cpu_ids']
    inventory = graph.resources()
    threads = [row for row in inventory if type(row) is H.HardwareThread and row.cpu_id == cpu]
    executors = [row for row in inventory if type(row) is H.CpuExecutor
                 and str(row.id) == binding['executor_id'] and cpu in row.cpu_ids]
    _need(len(threads) == len(executors) == 1 and threads[0].online is not False,
          'original selected thread/executor is missing or offline')
    thread = threads[0]
    _need(any(edge.kind is H.ConnectionKind.CONTAINS and edge.source == executors[0].id
              and edge.target == thread.id for edge in graph.connections()),
          'original selected executor/thread containment is missing')
    return snapshot, thread


def _execution_indexes(schedule):
    temporal = schedule.temporal
    activities = {row.task: row for row in temporal.activities}
    _need(len(activities) == len(temporal.activities) and set(activities) == {task.id for task in schedule.tasks},
          'one original synchronous activity per source task required')
    demands = {}
    for occupancy in temporal.occupancies:
        if occupancy.kind is OK.EXECUTION_DEMAND:
            demands.setdefault(occupancy.activity, []).append(occupancy)
    return activities, demands


def _phase(schedule, kind):
    rows = [task for task in schedule.tasks if task.kind is K.IMPLEMENTATION_PHASE
            and task.attrs.get('phase_kind') == kind]
    _need(len(rows) == 1, 'one original ' + kind + ' phase required')
    return rows[0]


def _reachable(adjacency, seed):
    pending = [seed]
    seen = {seed}
    while pending:
        for event in adjacency.get(pending.pop(), ()):
            if event not in seen:
                seen.add(event)
                pending.append(event)
    return seen


def _binding_window(temporal, bound, restored, *, indexed):
    """Index same-iteration edges and solve each side of the source window once."""
    forward, backward = {}, {}
    edges = ((edge.before, edge.after) for edge in temporal.constraints
             if not edge.iteration_distance and (
                 edge.before_iteration is None and edge.after_iteration is None if indexed
                 else edge.min_delay_ticks >= 0))
    for before, after in edges:
        forward.setdefault(before, set()).add(after)
        backward.setdefault(after, set()).add(before)
    for activity in temporal.activities:
        forward.setdefault(activity.start, set()).add(activity.completion)
        backward.setdefault(activity.completion, set()).add(activity.start)
    after_bind = _reachable(forward, bound.completion)
    _need(restored.start in after_bind, 'original successful bind/restore happens-before window is missing')
    return after_bind, _reachable(backward, restored.start)


def _inside_window(activity, window):
    after_bind, before_restore = window
    return activity.start in after_bind and activity.completion in before_restore


def _source_rule(schema, recorded):
    owner = Path(__file__).resolve()
    active = artifact_reference(owner, _SOURCE_SCHEMA)
    if recorded is None or recorded == active:
        return active
    predecessor, digest, size = _PREVIOUS_RULES[schema]
    older = dict(uri=str(owner.with_name(predecessor)), schema=_SOURCE_SCHEMA, sha256=digest, size_bytes=size)
    _need(recorded == older, 'source demand rule differs from its active or exact predecessor receipt')
    return older


def demand_proof(schedule, schema):
    if schedule.temporal is None:
        return None
    rows = [(i, row) for i, row in enumerate(schedule.temporal.source_refs) if row.get('schema') == schema]
    _need(len(rows) <= 1, 'one original demand proof required for ' + schema)
    return rows[0] if rows else None


def derive_serial_scope_demands(schedule, recipe, target, *, source_rule=None):
    """Refine closed serial stages and retain per-occupancy CPU-domain gaps."""
    from scar.analysis.implementation_scope import derive_stateful_scope_envelope
    validate_versioned_schedule(schedule)
    _need(schedule.temporal is not None and schedule.temporal.mode is TemporalMode.SEMANTIC,
          'original semantic scope time graph required')
    _need(not any(scope.kind is TemporalScopeKind.LOOP for scope in schedule.temporal.scopes),
          'standalone serial scope demands cannot describe a repeat adapter')
    original = schedule_from_dict(recipe['task_graph'])
    replay = derive_stateful_scope_envelope(original, recipe)
    _need(replace(schedule, temporal=None) == replay,
          'complete source scope envelope differs from original replay')
    phase_sources = _closed_phase_sources(replay)
    _need(recipe['control'].get('spatial_partition') == 'SERIAL_ROWS',
          'source scope demand supports only the original serial renderer')
    recipe_ref = _recipe_reference(schedule, recipe)
    plan = read_artifact(recipe['artifacts']['stage_plan']); binding = plan['resource_binding']
    cpus = binding['cpu_ids']
    _need(type(plan['cpu_threads']) is int and plan['cpu_threads'] == 1
          and type(cpus) in (list, tuple) and len(cpus) == 1 and type(cpus[0]) is int,
          'one original selected CPU required')
    snapshot, thread = _selected_thread(target, binding, recipe['artifacts']['target_snapshot'])
    selected_resource = str(thread.id)
    temporal = schedule.temporal
    activities, by_activity = _execution_indexes(schedule)
    enter, restore, worker_exit = (_phase(schedule, name) for name in ('TEAM_ENTER', 'CALLER_RESTORE', 'WORKER_EXIT'))
    _need(enter.attrs['phase_effects'] == ['BIND_CALLER_TO_SELECTED_SINGLE_CPU']
          and worker_exit.attrs['phase_effects'] == ['CHECK_CALLER_REMAINS_ON_SELECTED_CPU'],
          'original serial bind/check source contract differs')
    window = _binding_window(temporal, activities[enter.id], activities[restore.id], indexed=False)
    # Reuse the original semantic builder's anonymous service IDs. This also
    # permits a second derivation to check the same proof without trusting it.
    canonical = attach_task_time(replay)
    canonical_activities = {a.id: a.task for a in canonical.temporal.activities}
    canonical_rows = {canonical_activities[o.activity]: o for o in canonical.temporal.occupancies
                      if o.kind is OK.EXECUTION_DEMAND}
    capacities = dict(temporal.resource_capacities); replacements = {}; rows = []; trace = []
    state = 'ORIGINAL_CALLER_DOMAIN'; inside = False
    for task in schedule.tasks:
        activity = activities[task.id]; before = state
        transition = task.id in (enter.id, restore.id)
        provider = task.attrs.get('stage_kind') == 'PROVIDER'
        closed_stage = (task.kind is K.SNAPSHOT and task.capability == 'CPU_HOST_SNAPSHOT'
            or task.kind is K.STATE_WRITE and task.capability == 'CPU_STATE_WRITE'
            or task.kind is K.COMPUTE and task.capability in ('CPU_SCALAR_DAG', 'CPU_SCALAR_SETUP')
               and not provider)
        phase_kind = task.attrs.get('phase_kind')
        closed_phase = task.kind is K.IMPLEMENTATION_PHASE and phase_kind in _PHASE_EFFECTS
        if closed_phase:
            _need(task.capability == 'CPU_IMPLEMENTATION_PHASE'
                  and task.attrs.get('phase_effects') == _PHASE_EFFECTS[phase_kind],
                  'closed serial phase effects differ from original source contract')
        closed = closed_stage or closed_phase
        if provider:
            state = 'OPAQUE_UNKNOWN'
        elif inside and not transition and not closed and task.id != worker_exit.id:
            state = 'OPAQUE_UNKNOWN'
        occupancies = by_activity.get(activity.id, ())
        milestone = getattr(activity.duration_basis, 'value', None) == 'SOURCE_CONTROL_MILESTONE'
        if milestone:
            _need(task.kind is K.SUCCESS_FENCE and not occupancies,
                  'source control milestone must not acquire an execution demand')
        else:
            _need(len(occupancies) == 1 and task.id in canonical_rows,
                  'one original execution occupancy per non-milestone source task required')
            old = occupancies[0]; anonymous = canonical_rows[task.id]
            _need(old.id == anonymous.id and old.start == activity.start and old.end == activity.completion,
                  'original source execution occupancy identity/lifecycle differs')
            exact = state == 'SELECTED_SINGLETON' and not transition
            if exact:
                _need(_inside_window(activity, window),
                      'selected scope activity lacks successful-bind/restore bounds')
            resource = selected_resource if exact else anonymous.resource
            units = 1 if closed else None
            _need(old.units in (None, 1) if closed else old.units is None,
                  'existing whole-scope units conflict with the closed serial source contract')
            _need(old.resource in (anonymous.resource, resource)
                  and (old.resource == selected_resource or temporal.resource_capacities[old.resource] is None),
                  'existing execution placement conflicts with source scope domain')
            replacements[old.id] = (resource, units)
            rows.append(dict(occupancy=old.id, task=task.id, activity=activity.id,
                domain_state='OPAQUE_UNKNOWN' if task.id == restore.id else state,
                cpu_ids=[cpus[0]] if exact else None, exact_cpu_id=cpus[0] if exact else None,
                resource=resource, units=units,
                unit_authority=('CLOSED_SOURCE_SERIAL_STAGE' if closed_stage else 'CLOSED_SOURCE_SERIAL_PHASE'
                                if closed_phase else 'WHOLE_SCOPE_INTERNAL_THREADS_UNRESOLVED'),
                phase_source_basis=phase_kind if closed_phase else None,
                caller_participation_units=1, caller_participation_is_total_scope_demand=False,
                transition_scope=transition))
            if exact:
                _need(capacities.get(selected_resource) in (None, 1), 'selected thread capacity conflicts')
                capacities[selected_resource] = 1
        if task.id == enter.id:
            state = 'SELECTED_SINGLETON'; inside = True
        elif task.id == restore.id:
            state = 'RESTORED_ORIGINAL_DOMAIN'; inside = False
        trace.append(dict(task=task.id, before=before, after=state,
            opaque_affinity_preservation_unproven=provider, transition_scope=transition))
    proof = dict(schema=SCHEMA, schema_version=1, recipe=recipe_ref, recipe_id=recipe['recipe_id'],
        target=snapshot, stage_plan=recipe['artifacts']['stage_plan'],
        source_rule=_source_rule(SCHEMA, source_rule),
        source_scope_sha256=semantic_sha256(schedule_document(replay)),
        selected_cpu_id=cpus[0], selected_thread=dict(snapshot=snapshot, resource=encode(thread.id)),
        selected_thread_evidence=encode(thread.evidence),
        original_caller_cpu_ids=None, original_caller_domain_authority='NO_SOURCE_FACT_NOT_PLANNING_AFFINITY',
        closed_phase_source_basis=phase_sources,
        occupancies=rows, transition_trace=trace,
        source_qualification='SUCCESSFUL_STANDALONE_SERIAL_NORMAL_PATH',
        restore_is_forward_only=True, unknown_scope_threads_are_not_caller_participation=True,
        actual_cpu_observation=False, global_reservation=False, cost_inferred=False,
        native_execution_authorized=False)
    def prior(refs):
        return tuple(row for row in refs if row.get('schema') != SCHEMA)
    positions = {row['occupancy']: i for i, row in enumerate(rows)}
    proof_index = len(prior(temporal.source_refs))
    occupancies = tuple(replace(o, resource=replacements[o.id][0], units=replacements[o.id][1],
        source_refs=(*prior(o.source_refs), dict(schema=SCHEMA,
            locator='#/temporal/source_refs/'+str(proof_index)+'/occupancies/'+str(positions[o.id]))))
        if o.id in replacements else o for o in temporal.occupancies)
    result = replace(schedule, temporal=replace(temporal, occupancies=occupancies,
        resource_capacities=capacities, source_refs=(*prior(temporal.source_refs), proof)))
    validate_versioned_schedule(result); read_artifact(snapshot); read_artifact(recipe_ref)
    return result


def derive_serial_affinity_demands(schedule,recipe,target):
    """Refine only unknown calling-thread execution demand inside a proved pin."""
    from scar.analysis.implementation_envelope import derive_stateful_envelope

    validate_versioned_schedule(schedule)
    if schedule.temporal is None or schedule.temporal.mode is not TemporalMode.SEMANTIC:
        raise ValueError('original semantic implementation time graph required')
    replay=derive_stateful_envelope(schedule_from_dict(recipe['task_graph']),recipe)
    if replace(schedule,temporal=None)!=replay:
        raise ValueError('complete standalone implementation envelope differs from original replay')
    if recipe['control'].get('spatial_partition')!='SERIAL_ROWS':
        raise ValueError('only the original single-worker serial renderer is supported')
    plan=read_artifact(recipe['artifacts']['stage_plan']);binding=plan['resource_binding']
    cpus=binding['cpu_ids']
    if (type(plan['cpu_threads'])is not int or plan['cpu_threads']!=1 or type(cpus)not in(list,tuple)or len(cpus)!=1
            or type(cpus[0])is not int):raise ValueError('one original selected CPU required')
    snapshot, thread = _selected_thread(target, binding, recipe['artifacts']['target_snapshot'])
    resource = str(thread.id)
    temporal = schedule.temporal
    activities, by_activity = _execution_indexes(schedule)
    enter, restore, exit = (_phase(schedule, name) for name in ('TEAM_ENTER', 'CALLER_RESTORE', 'WORKER_EXIT'))
    _need(enter.attrs['phase_effects'] == ['BIND_CALLER_TO_SELECTED_SINGLE_CPU'],
          'serial successful-bind contract differs')
    window = _binding_window(temporal, activities[enter.id], activities[restore.id], indexed=False)
    selected={};trace=[];inside=False;state='UNKNOWN'
    closed={K.SNAPSHOT:'CPU_HOST_SNAPSHOT',K.STATE_WRITE:'CPU_STATE_WRITE'}
    for task in schedule.tasks:
        if task.id==enter.id:
            inside=True;state='BOUND';trace.append(dict(task=task.id,after=state,reason='SUCCESSFUL_ORIGINAL_SERIAL_BIND'));continue
        if task.id==restore.id:
            inside=False;trace.append(dict(task=task.id,before=state,after='UNKNOWN',reason='CALLER_RESTORE_TRANSITION'));break
        if not inside:continue
        activity=activities[task.id]
        if not _inside_window(activity, window):
            raise ValueError('activity lacks original bind-completion/restore-start temporal bounds')
        allowed=(task.kind in closed and closed[task.kind]==task.capability or task.kind is K.COMPUTE
            and task.capability in('CPU_SCALAR_DAG','CPU_SCALAR_SETUP')or task.id==exit.id)
        if task.attrs.get('stage_kind')=='PROVIDER':allowed=False
        before=state
        if not allowed:
            state='UNKNOWN';reason='OPAQUE_SCOPE_HAS_NO_AFFINITY_PRESERVATION_CONTRACT'
        elif state!='BOUND':reason='NO_REBIND_AFTER_OPAQUE_SCOPE'
        else:
            rows=by_activity.get(activity.id,())
            if len(rows)!=1:raise ValueError('one original execution occupancy per closed activity required')
            old=rows[0]
            if old.resource==resource:
                if old.units not in(None,1):raise ValueError('original selected-thread demand conflicts with serial proof')
            elif old.units is not None or temporal.resource_capacities[old.resource]is not None:
                raise ValueError('cannot overwrite an existing different execution resource demand')
            selected[old.id]=task.id;reason='CLOSED_SOURCE_WORK_PRESERVES_SELECTED_CALLING_THREAD'
        trace.append(dict(task=task.id,before=before,after=state,reason=reason,mapped=reason.startswith('CLOSED_SOURCE')))
    if temporal.resource_capacities.get(resource)not in(None,1):
        raise ValueError('original hardware-thread capacity conflicts with unit serial demand')
    envelope=schedule.proof['implementation_envelope']
    evidence=dict(schema=AFFINITY_PROOF,schema_version=1,
        recipe_id=recipe['recipe_id'],backend_source_pins=envelope['backend_source_pins'],
        selected_thread=dict(snapshot=snapshot,resource=encode(thread.id)),thread_evidence=encode(thread.evidence),
        stage_plan=recipe['artifacts']['stage_plan'],selected_cpu_id=cpus[0],
        bind_completion=activities[enter.id].completion,restore_start=activities[restore.id].start,
        transition_trace=trace,source_qualification='SUCCESSFUL_STANDALONE_SERIAL_NORMAL_PATH',
        opaque_affinity_effects_invalidate_forward_knowledge=True,global_reservation=False,
        actual_cpu_observation=False,cost_inferred=False,native_execution_authorized=False)
    def prior_refs(refs):return tuple(r for r in refs if r.get('schema')!=evidence['schema'])
    occupancies=tuple(replace(o,resource=resource,units=1,source_refs=(*prior_refs(o.source_refs),evidence))
        if o.id in selected else replace(o,source_refs=prior_refs(o.source_refs))for o in temporal.occupancies)
    capacities=dict(temporal.resource_capacities)
    if selected:capacities[resource]=1
    result=replace(schedule,temporal=replace(temporal,occupancies=occupancies,resource_capacities=capacities,
        source_refs=(*prior_refs(temporal.source_refs),evidence)))
    validate_versioned_schedule(result);read_artifact(snapshot)
    return result


def derive_workspace_demands(schedule, step_plan, native_metadata, *, native_reference,
        cpp_reference, workspace_plan_reference, parent_schedule_reference, target, source_rule=None):
    """Refine an admitted baseline; deterministic replay does not open a compiler."""
    from scar.analysis.workspace_source import workspace_serial_source_contract
    from scar.analysis.workspace_time import derive_workspace_time

    validate_versioned_schedule(schedule)
    parent = schedule_document(schedule)
    temporal = schedule.temporal
    _need(parent['schema_version'] == 4 and temporal.mode is TemporalMode.SEMANTIC
          and temporal.path_condition is TemporalPathCondition.NORMAL_COMPLETE,
          'normal-complete semantic workspace v4 baseline required')
    _need(not any(row.get('schema') == WORKSPACE_PROOF for row in temporal.source_refs),
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
    snapshot, thread = _selected_thread(target, binding, step_plan['resource_context']['target_snapshot'])
    selected_resource = str(thread.id)
    activities, by_activity = _execution_indexes(schedule)
    loops = [scope for scope in temporal.scopes if scope.kind is TemporalScopeKind.LOOP]
    _need(len(loops) == int(workspace['trip_count'] > 0), 'one active shared iteration template required')
    loop = loops[0] if loops else None
    _need(contract['enter_fact_at'] == 'COMPLETION' and contract['restore_kill_at'] == 'START',
          'source bind/restore lifecycle anchors differ')
    enter = restore = window = None
    if loop is not None:
        enter, restore = (_phase(schedule, contract[key]) for key in ('enter_phase_kind', 'restore_phase_kind'))
        _need(activities[enter.id].scope == activities[restore.id].scope == loop.id,
              'original bind/restore phases must belong to the shared loop scope')
        window = _binding_window(temporal, activities[enter.id], activities[restore.id], indexed=True)
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
            _need(_inside_window(a, window),
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
    proof = dict(schema=WORKSPACE_PROOF, schema_version=1, parent_schedule=parent_schedule_reference,
        parent_schedule_sha256=semantic_sha256(parent), target=snapshot, stage_plan=plan_ref,
        native_manifest=native_reference, cpp_source=cpp_reference, workspace_plan=workspace_plan_reference,
        source_rule=_source_rule(WORKSPACE_PROOF, source_rule),
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
        source_refs=(*o.source_refs, dict(schema=WORKSPACE_PROOF,
            locator='#/temporal/source_refs/'+str(proof_index)+'/occupancies/'+str(positions[o.id]))))
        if o.id in replacements else o for o in temporal.occupancies)
    result = replace(schedule, temporal=replace(temporal, occupancies=occupancies,
        resource_capacities=capacities, source_refs=(*temporal.source_refs, proof)))
    validate_versioned_schedule(result)
    read_artifact(snapshot)
    return result


__all__ = ['SCHEMA', 'WORKSPACE_PROOF', 'demand_proof', 'derive_serial_scope_demands',
           'derive_serial_affinity_demands', 'derive_workspace_demands']

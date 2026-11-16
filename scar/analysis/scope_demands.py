"""Separate serial source demand from knowledge of calling-thread placement.

This refines the existing semantic occupancies. A synchronous caller's
participation does not prove a provider's complete internal thread demand.
"""
from dataclasses import replace
from pathlib import Path

from scar.analysis.schedule_time import attach_task_time
from scar.analysis.target_graph import TargetHardwareGraph
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program import target_hardware as H
from scar.ir.record_codec import encode
from scar.ir.versioned_schedule import (TemporalMode, TemporalScopeKind, TemporalOccupancyKind as OK,
    VersionedTaskKind as K, schedule_document, schedule_from_dict, validate_versioned_schedule)
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.serial-scope-demand-source-proof'

# Reviewed closed fragments of this backend, rather than an actor-name
# heuristic. A changed fragment needs a new source-contract review; a hash
# only identifies the exact text to which this rule applies.
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


def derive_serial_scope_demands(schedule, recipe, target):
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
    if type(target) is TargetHardwareGraph:
        graph = target
    else:
        path = Path(target).resolve(); graph = TargetHardwareGraph(path.parent if path.is_file() else path)
    snapshot = artifact_reference(graph.path/'result.json', 'scar.target-hardware-graph')
    _need(snapshot == recipe['artifacts']['target_snapshot']
          and binding['target_sha256'] == snapshot['sha256']
          and Path(binding['target_path']).resolve() == graph.path.resolve(),
          'scope selected CPU and original target snapshot differ')
    resources = graph.resources()
    threads = [row for row in resources if type(row) is H.HardwareThread and row.cpu_id == cpus[0]]
    executors = [row for row in resources if type(row) is H.CpuExecutor
                 and str(row.id) == binding['executor_id'] and cpus[0] in row.cpu_ids]
    _need(len(threads) == len(executors) == 1 and threads[0].online is not False,
          'original selected thread/executor is missing or offline')
    thread = threads[0]; selected_resource = str(thread.id)
    _need(any(edge.kind is H.ConnectionKind.CONTAINS and edge.source == executors[0].id
              and edge.target == thread.id for edge in graph.connections()),
          'original selected executor/thread containment is missing')
    temporal = schedule.temporal; activities = {row.task: row for row in temporal.activities}
    _need(len(activities) == len(temporal.activities) and set(activities) == {t.id for t in schedule.tasks},
          'one original synchronous activity per source task required')
    def phase(name):
        rows = [t for t in schedule.tasks if t.kind is K.IMPLEMENTATION_PHASE
                and t.attrs.get('phase_kind') == name]
        _need(len(rows) == 1, 'one original ' + name + ' phase required')
        return rows[0]
    enter, restore, worker_exit = phase('TEAM_ENTER'), phase('CALLER_RESTORE'), phase('WORKER_EXIT')
    _need(enter.attrs['phase_effects'] == ['BIND_CALLER_TO_SELECTED_SINGLE_CPU']
          and worker_exit.attrs['phase_effects'] == ['CHECK_CALLER_REMAINS_ON_SELECTED_CPU'],
          'original serial bind/check source contract differs')
    adjacency = {}
    for edge in temporal.constraints:
        if not edge.iteration_distance and edge.min_delay_ticks >= 0:
            adjacency.setdefault(edge.before, set()).add(edge.after)
    for activity in temporal.activities:
        adjacency.setdefault(activity.start, set()).add(activity.completion)
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
    _need(reaches(activities[enter.id].completion, activities[restore.id].start),
          'original successful-bind/restore temporal window is missing')
    # Reuse the original semantic builder's anonymous service IDs. This also
    # permits a second derivation to check the same proof without trusting it.
    canonical = attach_task_time(replay)
    canonical_activities = {a.id: a.task for a in canonical.temporal.activities}
    canonical_rows = {canonical_activities[o.activity]: o for o in canonical.temporal.occupancies
                      if o.kind is OK.EXECUTION_DEMAND}
    by_activity = {}
    for occupancy in temporal.occupancies:
        if occupancy.kind is OK.EXECUTION_DEMAND:
            by_activity.setdefault(occupancy.activity, []).append(occupancy)
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
                _need(reaches(activities[enter.id].completion, activity.start)
                      and reaches(activity.completion, activities[restore.id].start),
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
        source_rule=artifact_reference(Path(__file__).resolve(), 'scar.source-demand-rule'),
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


__all__ = ['SCHEMA', 'derive_serial_scope_demands']

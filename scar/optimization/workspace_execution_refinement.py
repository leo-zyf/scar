"""Pure structural refinement of an admitted workspace graph to its backend.

The publication reader must first admit current source, inventory, probes and
original caller proofs. This module rederives the graph/domain transformation
in memory; it never replaces that byte/source admission with proof flags.
"""
from dataclasses import replace

from scar.analysis.workspace_time import derive_workspace_time
from scar.ir.activity_costs import activity_cost_document, validate_activity_cost_table
from scar.ir.record_codec import encode
from scar.ir.temporal_resources import OriginalBudget, projection_document, projection_from_dict
from scar.ir.versioned_schedule import (VersionedTaskKind as K, TemporalScopeKind as S,
    TemporalOccupancyKind as O, schedule_document, validate_versioned_schedule)
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.workspace-execution-refinement'
_CPU_PROOF = 'scar.workspace-cpu-demand-source-proof'


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _replay_domains(baseline, schedule, proof, step_plan):
    """Reconstruct the complete admitted serial domain transform, without I/O."""
    temporal = baseline.temporal
    contract = proof['source_contract']
    _need((contract.get('schema'), contract.get('schema_version')) == ('scar.workspace-serial-source-contract', 1)
          and contract['resource_binding'] == step_plan['resource_binding']
          and contract['step_plan_sha256'] == semantic_sha256(step_plan)
          and contract['native_reference'] == proof['native_manifest']
          and contract['cpp_reference'] == proof['cpp_source'] and contract['target'] == proof['target']
          and contract['enter_fact_at'] == 'COMPLETION' and contract['restore_kill_at'] == 'START'
          and contract['commit_whole_thread_units'] is None and contract['commit_exact_selected_cpu'] is False,
          'original serial source contract and exact step plan required')
    cpus = contract['cpu_ids']; selected = proof['selected_thread']
    _need(type(cpus) is list and len(cpus) == 1 and type(cpus[0]) is int and cpus[0] >= 0
          and cpus == list(step_plan['resource_binding']['cpu_ids']) and step_plan['cpu_threads'] == 1
          and proof['selected_cpu_id'] == cpus[0] and selected['snapshot'] == proof['target'],
          'one retained serial selected thread and target required')
    resource = selected['resource']['wire']
    activities = {a.task: a for a in temporal.activities}
    _need(len(activities) == len(temporal.activities) and set(activities) == {t.id for t in baseline.tasks},
          'complete original single-activity workspace lifecycles required')
    loops = [s for s in temporal.scopes if s.kind is S.LOOP]
    loop = loops[0] if loops else None
    def phase(kind):
        rows = [t for t in baseline.tasks if t.kind is K.IMPLEMENTATION_PHASE and t.attrs.get('phase_kind') == kind]
        _need(len(rows) == 1 and activities[rows[0].id].scope == loop.id,
              'one original shared-loop source phase required')
        return rows[0]
    enter = phase(contract['enter_phase_kind']) if loop else None
    restore = phase(contract['restore_phase_kind']) if loop else None
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
    by_activity = {}
    for o in temporal.occupancies:
        if o.kind is O.EXECUTION_DEMAND:
            by_activity.setdefault(o.activity, []).append(o)
    replacements = {}; domains = []; trace = []; capacities = dict(temporal.resource_capacities)
    state = 'ORIGINAL_CALLER_DOMAIN'
    for task in baseline.tasks:
        a = activities[task.id]; before = state
        body = loop is not None and a.scope == loop.id
        entering, restoring = task is enter, task is restore
        opaque = (task.attrs.get('stage_kind') in contract['opaque_stage_kinds']
                  or task.capability in contract['opaque_capabilities'])
        rules = [r for r in contract['closed_stages'] if r['kind'] == task.kind.value
                 and r['capability'] == task.capability and r['stage_kind'] == task.attrs.get('stage_kind')]
        _need(len(rules) <= 1 and all(r['whole_thread_units'] == 1 and r['affinity_preserving'] is True for r in rules),
              'closed serial source stage rule is ambiguous or incomplete')
        closed = bool(rules)
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
                  'closed stage is outside the original successful bind/restore window')
        occupancies = by_activity.get(a.id, ())
        if a.duration_basis.value == 'SOURCE_CONTROL_MILESTONE':
            _need(not occupancies, 'source milestone cannot acquire physical execution demand')
        else:
            _need(len(occupancies) == 1, 'one original service occupancy per workspace task required')
            old = occupancies[0]
            _need(old.units is None and temporal.resource_capacities[old.resource] is None
                  and (old.start, old.end) == (a.start, a.completion),
                  'baseline service identity/UNKNOWN demand differs')
            assigned, units = (resource if exact else old.resource), (1 if closed and not opaque else None)
            replacements[old.id] = (assigned, units)
            domains.append(dict(occupancy=old.id, task=task.id, activity=a.id, scope=a.scope,
                domain_state=state if body else 'ORIGINAL_CALLER_DOMAIN',
                cpu_ids=cpus if exact else None, exact_cpu_id=cpus[0] if exact else None,
                resource=assigned, units=units,
                unit_authority='CLOSED_SERIAL_SOURCE_STAGE' if units == 1 else 'WHOLE_SCOPE_THREAD_DEMAND_UNRESOLVED',
                caller_participation_units=1, caller_participation_is_total_scope_demand=False,
                transition_scope=entering or restoring))
            if exact:
                _need(capacities.get(resource) in (None, 1), 'selected thread capacity conflicts')
                capacities[resource] = 1
        if entering:
            state = 'SELECTED_SINGLETON'
        trace.append(dict(task=task.id, scope=a.scope, before=before, after=state,
            opaque_preservation_unproven=opaque, selected_cpu_proven=exact,
            reason='SUCCESSFUL_PER_ITERATION_BIND' if entering else 'RESTORE_START_KILLS_SELECTED_DOMAIN'
                if restoring else 'OPAQUE_SCOPE_KILLS_FORWARD_KNOWLEDGE' if opaque
                else 'CLOSED_SERIAL_STAGE' if closed else 'UNRESOLVED_WHOLE_SCOPE'))
    _need(proof['occupancies'] == domains and proof['transition_trace'] == trace
          and proof['parent_schedule_sha256'] == semantic_sha256(schedule_document(baseline))
          and all(proof.get(key) is False for key in ('actual_cpu_observation', 'global_reservation',
                  'cost_inferred', 'native_execution_authorized')),
          'CPU proof differs from mechanical per-iteration source-state propagation')
    positions = {row['occupancy']: i for i, row in enumerate(domains)}
    ordinal = len(temporal.source_refs)
    occupancies = tuple(replace(o, resource=replacements[o.id][0], units=replacements[o.id][1],
        source_refs=(*o.source_refs, dict(schema=_CPU_PROOF,
            locator='#/temporal/source_refs/'+str(ordinal)+'/occupancies/'+str(positions[o.id]))))
        if o.id in replacements else o for o in temporal.occupancies)
    expected = replace(baseline, temporal=replace(temporal, occupancies=occupancies,
        resource_capacities=capacities, source_refs=(*temporal.source_refs, proof)))
    _need(expected == schedule, 'workspace graph differs from the complete baseline/domain refinement')


def derive_workspace_execution_refinement(schedule, resources, costs, accounting, workspace_plan,
        native_metadata, *, step_plan, baseline_schedule, source_refs=()):
    """Map the unchanged shared body to its backend; no source admission or I/O."""
    _need(type(source_refs) is tuple and all(type(row) is dict for row in source_refs),
          'immutable caller-admitted source reference index required')
    validate_versioned_schedule(schedule)
    _need(schedule_document(schedule)['schema_version'] == 4 and not schedule.executable,
          'non-executable source-qualified v4 workspace graph required')
    proofs = [r for r in schedule.temporal.source_refs if r.get('schema') == _CPU_PROOF]
    _need(len(proofs) == 1 and proofs[0].get('schema_version') == 1,
          'one current source-qualified workspace CPU proof required')
    proof = proofs[0]; plan_sha = semantic_sha256(workspace_plan)
    _need(workspace_plan['schema'] == 'scar.stateful-workspace-reuse-plan'
          and native_metadata['schema'] == 'scar.native-stateful-workspace-repeat'
          and native_metadata['workspace_plan_sha256'] == plan_sha
          and native_metadata['trip_count'] == workspace_plan['trip_count']
          and native_metadata['base_ids']['plan_sha256'] == workspace_plan['step_plan_sha256'] == semantic_sha256(step_plan)
          and native_metadata['base_ids']['region_sha256'] == workspace_plan['step_region_sha256'],
          'exact workspace/step/region/native identities differ')
    execution = native_metadata['execution_plan']
    _need(native_metadata['execution_plan_sha256'] == semantic_sha256(execution)
          and native_metadata['build_id'] == semantic_sha256(native_metadata['identity'])
          and execution == native_metadata['identity']['execution_plan']
          and execution['workspace_plan_sha256'] == plan_sha and execution['base_ids'] == native_metadata['base_ids']
          and execution['base_native_reference'] == native_metadata['base_native_reference']
          and execution['preparation'] == native_metadata['preparation']
          and execution['trip_count'] == workspace_plan['trip_count']
          and native_metadata['sha256']['source'] == proof['cpp_source']['sha256']
              == native_metadata['identity']['source_sha256']
          and native_metadata['source_mapping'] == native_metadata['identity']['source_mapping']
          and native_metadata['allocations_hoisted'] is True and native_metadata['teams_hoisted'] is False
          and native_metadata['native_math_changed'] is False,
          'original workspace implementation blueprint differs')
    baseline = derive_workspace_time(workspace_plan, native_manifest=proof['native_manifest'],
        native_metadata=native_metadata, cpp_source=proof['cpp_source'], workspace_reference=proof['workspace_plan'])
    _need(baseline == baseline_schedule, 'parent baseline differs from pure original workspace projection')
    _replay_domains(baseline, schedule, proof, step_plan)
    resources = projection_from_dict(projection_document(resources))
    digest = semantic_sha256(schedule_document(schedule))
    _need(encode(resources.target) == proof['target'] == step_plan['resource_context']['target_snapshot']
          and resources.budget == OriginalBudget(schedule.budget_bytes, schedule.budget_scope, schedule.budget_refs),
          'original hardware target or payload budget differs')
    domains = {r['occupancy']: r for r in proof['occupancies']}
    physical = {o.storage_slot: o for o in schedule.temporal.occupancies if o.kind is O.PHYSICAL_SLOT_BYTES}
    _need(len(resources.executions) == len(domains) and {r.occupancy for r in resources.executions} == set(domains),
          'complete per-activity hardware binding coverage required')
    for binding in resources.executions:
        domain = domains[binding.occupancy]
        _need((binding.resource, binding.units) == (domain['resource'], domain['units']),
              'hardware execution binding differs from source domain')
        if domain['exact_cpu_id'] is None:
            _need(binding.exact_thread is None and not binding.candidate_threads,
                  'unknown source CPU cannot acquire an invented binding')
        else:
            _need(encode(binding.exact_thread) == proof['selected_thread']
                  and binding.candidate_threads == (binding.exact_thread,),
                  'known source CPU must retain its exact original singleton thread')
    _need(len(resources.storages) == len(physical)
          and {r.slot for r in resources.storages} == set(physical)
          and all(r.occupancy == physical[r.slot].id and r.payload_bytes == physical[r.slot].units
                  and r.exact_memory is None for r in resources.storages),
          'one unbound physical lifetime per original workspace slot required')
    summary = validate_activity_cost_table(costs, schedule, resources)
    _need(accounting.get('schema') == 'scar.workspace-cost-accounting'
          and accounting['summary'] == summary and accounting['source_schedule_sha256'] == digest
          and accounting['resource_projection_sha256'] == costs.resource_projection_sha256
          and accounting['workspace_plan'] == proof['workspace_plan']
          and accounting['normal_sequence_duration_ticks'] is None and accounting['physical_scope_latency'] is None
          and accounting['unknown_duration_imputed'] is False and accounting['residuals_cancelled'] is False,
          'conditional workspace cost accounting differs')
    temporal = schedule.temporal; activities = {a.id: a for a in temporal.activities}
    loops = [s for s in temporal.scopes if s.kind is S.LOOP]
    count = workspace_plan['trip_count']; loop = loops[0] if loops else None
    _need(len(loops) == int(count > 0) and accounting['trip_count'] == count
          and accounting['loop_scope'] == (None if loop is None else loop.id),
          'one shared loop body and original exact trip count required')
    ledger = {r['subject']: r for r in accounting['subjects']}
    _need(len(ledger) == len(accounting['subjects']) == len(costs.costs), 'complete unique cost subject ledger required')
    for row in costs.costs:
        record = ledger[row.subject]; scope = activities[row.activities[0]].scope
        _need(record['activities'] == list(row.activities)
              and record['task_ids'] == [activities[a].task for a in row.activities]
              and record['scope'] == scope and record['occurrence_count'] == (count if loop and scope == loop.id else 1)
              and record['estimate_per_occurrence_ticks'] == row.estimate_ticks,
              'cost subject occurrence or original task mapping differs')
    tasks = {t.id: t for t in schedule.tasks}
    owned = [s for s in schedule.storage_slots if s.ownership == 'SCHEDULE']
    order = workspace_plan['allocation_order']; reverse = workspace_plan['release_order']
    _need(reverse == list(reversed(order)) and [s.id for s in owned] == order,
          'fixed workspace declaration and reverse destruction order required')
    allocations = [t for t in schedule.tasks if t.kind is K.ALLOCATE]
    releases = [t for t in schedule.tasks if t.kind is K.RELEASE]
    _need([s for t in allocations for s in t.allocates] == order
          and [s for t in releases for s in t.releases] == reverse,
          'outer allocation/reverse-release sequence differs')
    source_mode = workspace_plan['source_f4_authority']['mode']
    _need(source_mode in ('ORIGINAL_SOURCE_LOOP_PLAN_BOUND', 'EXPLICIT_FIXED_INPUT_COMPOSITION'),
          'explicit source-loop or composition authority required')
    mapping = schedule.proof['workspace_time']
    certificate = dict(schema=SCHEMA, schema_version=1,
        inputs=dict(source_schedule_sha256=digest, baseline_schedule_sha256=semantic_sha256(schedule_document(baseline)),
            resource_projection_sha256=costs.resource_projection_sha256,
            cost_table_sha256=semantic_sha256(activity_cost_document(costs)),
            accounting_sha256=semantic_sha256(accounting), workspace_plan_sha256=plan_sha,
            step_plan_sha256=semantic_sha256(step_plan), source_refs=encode(source_refs)),
        native_blueprint=dict(manifest=proof['native_manifest'], cpp_source=proof['cpp_source'],
            workspace_plan_sha256=plan_sha, execution_plan_sha256=native_metadata['execution_plan_sha256'],
            base_ids=native_metadata['base_ids'], source_mapping=native_metadata['source_mapping']),
        execution_scope='WORKSPACE_LIFETIME_ONLY', trip_count=count,
        task_map=[dict(task=t.id, original_step_task=next((old for old, new in mapping['task_map'].items() if new == t.id), None),
            kind=t.kind.value, source_refs=encode(t.source_refs)) for t in schedule.tasks],
        activity_map=[dict(activity=a.id, task=a.task, scope=a.scope, start=a.start, completion=a.completion,
            ready_events=list(a.ready_events), occurrences=count if loop and a.scope == loop.id else 1)
            for a in temporal.activities],
        slot_map=[dict(slot=s.id, capacity_bytes=s.capacity_bytes, allocation=s.allocated_by, release=s.released_by,
            occupancy=physical[s.id].id, start=physical[s.id].start, end=physical[s.id].end,
            physical_count=1, contents='PER_ITERATION_LOCAL_EPOCH_NOT_AN_INVARIANT_VALUE') for s in owned],
        body=dict(scope=None if loop is None else loop.id, task_ids=mapping['body_task_ids'],
            shared_template_count=int(count > 0), iterations_unrolled=False,
            materialization_templates=[] if loop is None else list(loop.iteration_local_materializations),
            content_carries=[] if loop is None else encode(loop.content_carries)),
        outer=dict(task_ids=mapping['outer_task_ids'], allocation_order=order, release_order=reverse,
            selected_boundaries=[dict(id=e.id, before=e.before, after=e.after,
                before_iteration=encode(e.before_iteration), after_iteration=encode(e.after_iteration),
                source_refs=encode(e.source_refs)) for e in temporal.constraints
                if e.before_iteration is not None or e.after_iteration is not None]),
        preserved=dict(original_body_order=True, per_step_guards=True, finite_scans=True, teams_and_context=True,
            transactional_commit_boundaries=True, completed_cursor=True, source_math=True,
            failure_contract=workspace_plan['execution_contract']['failure'], failure_path_timing_modeled=False,
            unknown_cpu_activities_reordered=False, finite_scan_or_team_hoisted=False),
        source_authority=dict(mode=source_mode,
            caller_replacement='REQUIRES_ORIGINAL_F4_GUARD_AND_COMPACT_EXPORT' if source_mode == 'ORIGINAL_SOURCE_LOOP_PLAN_BOUND'
                else 'FORBIDDEN_FOR_FIXED_INPUT_COMPOSITION', automatic_caller_replacement_authorized=False),
        summary=dict(activities=len(activities), tasks=len(tasks), owned_slots=len(owned),
            body_template_count=int(count > 0), content_carries=0 if loop is None else len(loop.content_carries),
            known_component_subjects=summary['known_component_subjects'], unknown_subjects=summary['unknown_subjects'],
            unknown_cpu_domains=sum(r['cpu_ids'] is None for r in domains.values())),
        source_admission_performed_by_pure_function=False, current_source_and_inventory_admission_required=True,
        fresh_cpp_source_required=True, graph_lowering_checked=True, native_libraries_loaded=False,
        target_program_executed=False, cost_status='UNKNOWN', trial_only=True, selected_for_use=False,
        default_runtime_decision='NO_OP', normal_sequence_duration_ticks=None, physical_scope_latency=None,
        residuals_cancelled=False, profitability_proven=False, physical_optimality_proven=False,
        solver_authorized=False, execution_authorized=False, deployment_authorized=False)
    return encode(certificate)


def verify_workspace_execution_refinement(schedule, resources, costs, accounting, workspace_plan,
        native_metadata, certificate, *, step_plan, baseline_schedule, source_refs=()):
    expected = derive_workspace_execution_refinement(schedule, resources, costs, accounting,
        workspace_plan, native_metadata, step_plan=step_plan, baseline_schedule=baseline_schedule, source_refs=source_refs)
    _need(certificate == expected, 'workspace execution refinement differs from complete structural replay')
    return expected


__all__ = ['SCHEMA', 'derive_workspace_execution_refinement', 'verify_workspace_execution_refinement']

"""Conditional primitive costs on one source-qualified shared workspace body.

The publication reader admits source, hardware, calibration and probe bytes.
This pure bridge checks their supplied identities and original raw statistics;
it never reads files, measures work, or copies prices from another cost table.
"""
from collections import defaultdict
import math

from scar.ir.activity_costs import (ActivityCost, ActivityCostTable, CostBasis,
    CostCoverage, CostInterval, IntervalKind, validate_activity_cost_table)
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import encode
from scar.ir.temporal_resources import projection_document, projection_from_dict
from scar.ir.versioned_schedule import (TemporalScopeKind as S, TemporalOccupancyKind as O,
    VersionedTaskKind as K, schedule_document, validate_versioned_schedule)
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.workspace-cost-accounting'
_DEMAND_SCHEMA = 'scar.workspace-cpu-demand-source-proof'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _reference(value, schema):
    ref = ProviderArtifact(**value)
    _need(ref.schema == schema, 'original ' + schema + ' reference required')
    return ref


def _primitive_records(evidence, resources, source_proof):
    """Reuse the existing raw-sample checks with caller-admitted byte material."""
    from scar.optimization.stateful_cpu_model import stateful_resource_context, component_key
    from scar.optimization.temporal_activity_costs import _samples
    domain = evidence['domain']; candidate_id = evidence['candidate_id']
    actual = evidence['actual_step_plan']; backend = evidence['actual_backend']
    _need(source_proof.get('stage_plan') == evidence['actual_step_plan_reference'],
          'actual step plan must retain its source-replayed workspace plan reference')
    _need(backend.get('implementation_sha256')
          == source_proof.get('source_contract', {}).get('base_implementation_sha256')
          and type(backend.get('implementation_sha256')) is str,
          'actual base backend must match the source-qualified workspace implementation')
    candidate = next((row for row in domain['candidates'] if row['id'] == candidate_id), None)
    _need(candidate is not None, 'explicit original canonical CPU candidate required')
    context = stateful_resource_context(domain, candidate_id)
    compatible = (domain['target_snapshot'] == encode(resources.target)
        and backend == domain['backend_capabilities']['serial']
        and candidate['legal_under_guards'] is True
        and candidate['worker_count'] == actual['cpu_threads'] == 1
        and all(candidate['binding'][key] == actual['resource_binding'][key] for key in
            ('cpu_ids', 'executor_id', 'memory_domain_id', 'memory_policy', 'numa_node', 'target_sha256')))
    records = {}; rejected = []; session = dict(sources={}, validated_probes=set(), statistics={})
    for publication in evidence['calibrations']:
        reference = publication['reference']; document = publication['document']
        _reference(reference, 'scar.stateful-resource-calibration')
        _need((document.get('schema'), document.get('schema_version'))
              == ('scar.stateful-resource-calibration', 1), 'versioned canonical primitive document required')
        if not compatible or document.get('resource_context') != context:
            rejected.append(dict(artifact=reference,
                reason='ACTUAL_WORKSPACE_AND_CANONICAL_PRIMITIVE_RESOURCE_CONTEXT_DIFFER', cost_charged=False))
            continue
        source_key = semantic_sha256(reference)
        session['sources'][source_key] = document
        # The caller cold-replayed these exact probe source/library/header pins.
        # Preloading only avoids that I/O here; _samples still rechecks all raw
        # values, measurement flags, statistics and primitive signature fields.
        session['validated_probes'].add(source_key)
        for index, record in enumerate(document['components']):
            if record.get('component') != 'ROW_COPY_STAGE':
                continue
            _need(record.get('basis') == 'DIRECT_COMPONENT_TIMING'
                  and all(record.get(key) is not False for key in
                      ('selectable', 'precision_admitted', 'exportable_as_complete_cost')),
                  'only admitted direct canonical copy samples may price workspace activities')
            parameters = record['parameters']; key = component_key('ROW_COPY_STAGE', parameters)
            _need(key not in records, 'ambiguous canonical copy primitive publications')
            fact = dict(artifact=reference, locator='#/components/' + str(index))
            stats, conditions = _samples(fact, dict(component='ROW_COPY_STAGE', parameters=parameters,
                key=key), session)
            records[key] = (fact, stats, conditions)
    return actual, context, records, rejected


def _snapshot_parameters(task, plan, schedule):
    """Match the original full-root contiguous source/destination copy contract."""
    pointer = task.attrs.get('source_stage_ref', {}).get('pointer', '')
    _need(pointer.startswith('#/stages/') and pointer.rsplit('/', 1)[-1].isdigit(),
          'snapshot must retain its exact original source-stage pointer')
    index = int(pointer.rsplit('/', 1)[-1]); stages = plan['stages']
    _need(index < len(stages), 'snapshot source-stage index outside original plan')
    stage = stages[index]; shape = stage['shape']
    _need(task.id == stage['id'] and stage['kind'] == 'SNAPSHOT_ROOT'
          and task.kind is K.SNAPSHOT and task.capability == 'CPU_HOST_SNAPSHOT'
          and task.attrs.get('stage_kind') == stage['kind'] and task.attrs.get('shape') == shape
          and task.attrs.get('cpu_threads') == plan['cpu_threads']
          and task.attrs.get('executor_id') == plan['resource_binding']['executor_id']
          and task.attrs.get('source_stage_ref') == dict(artifact_role='stage_plan', pointer=pointer)
          and shape == plan['input_shapes'][stage['target']] and len(task.reads) == len(task.writes) == 1,
          'snapshot kind, root input, geometry and stage occurrence must match original source')
    materials = {row.id: row for row in schedule.materializations}
    values = {row.id: row for row in schedule.values}
    source, destination = materials[task.reads[0]], materials[task.writes[0]]
    size = math.prod(shape) * 8
    _need(source.external_ready and source.ready_task is None
          and values[source.value].attrs.get('input_id') == stage['target']
          and source.region.slot != destination.region.slot and destination.ready_task == task.id
          and source.region.offset_bytes == destination.region.offset_bytes == 0
          and source.region.nbytes == destination.region.nbytes == size,
          'full-root float64 copy must preserve original distinct materializations and exact spans')
    return dict(rows=shape[0], cols=shape[1], source_stride=shape[1], source_offset=0,
                destination_stride=shape[1], destination_offset=0, workers=1)


def derive_workspace_costs(schedule, resources, *, primitive_evidence,
                          workspace_time_ref, workspace_plan_ref, source_refs=()):
    """Price matching copy templates; account once/loop occurrences separately.

    ``primitive_evidence`` is None, or a caller-qualified dictionary containing
    domain, candidate_id, actual_step_plan, actual_step_plan_reference,
    actual_backend and calibrations[{reference,document}]. Source/probe byte
    admission remains the publication reader's responsibility.
    """
    validate_versioned_schedule(schedule)
    _need(type(source_refs) is tuple and all(type(row) is dict for row in source_refs),
          'immutable source reference index required')
    resources = projection_from_dict(projection_document(resources))
    digest = semantic_sha256(schedule_document(schedule))
    _need(resources.source_schedule_sha256 == digest, 'workspace resource projection and time graph differ')
    time_ref = ProviderArtifact(**workspace_time_ref)
    _need(time_ref.schema in ('scar.workspace-time-preparation', 'scar.workspace-demand-time-preparation')
          and (primitive_evidence is None or time_ref.schema == 'scar.workspace-demand-time-preparation'),
          'primitive costs require the current source-refined workspace time reference')
    _reference(workspace_plan_ref, 'scar.stateful-workspace-reuse-plan')
    temporal = schedule.temporal
    _need(temporal is not None and any(row.get('artifact') == workspace_plan_ref for row in temporal.source_refs),
          'workspace time must retain the exact workspace plan reference')
    certificate = schedule.proof.get('workspace_time', {})
    loops = [row for row in temporal.scopes if row.kind is S.LOOP]
    roots = [row for row in temporal.scopes if row.kind is S.ROOT]
    _need(certificate.get('schema') == 'scar.workspace-time-projection-certificate'
          and len(loops) == len(roots) == 1 and loops[0].parent == roots[0].id
          and type(loops[0].trip_count) is int and loops[0].trip_count > 0
          and certificate['trip_count'] == loops[0].trip_count,
          'one source-qualified fixed positive shared workspace loop required')
    loop, root = loops[0], roots[0]; count = loop.trip_count
    tasks = {row.id: row for row in schedule.tasks}; by_task = {row.task: row for row in temporal.activities}
    _need(len(by_task) == len(temporal.activities) and set(by_task) == set(tasks),
          'one complete source lifecycle per workspace task required')
    occurrences = certificate['activity_occurrences']
    _need(len(occurrences) == len(by_task) and {row['activity'] for row in occurrences}
          == {row.id for row in temporal.activities}, 'complete workspace activity occurrence certificate required')
    for record in occurrences:
        activity = next(row for row in temporal.activities if row.id == record['activity'])
        _need(activity.scope in (root.id, loop.id) and record['scope'] == activity.scope
              and type(record['count']) is int and record['count'] == (count if activity.scope == loop.id else 1)
              and record['meaning'] == ('PER_ITERATION_TEMPLATE' if activity.scope == loop.id
                                       else 'ONCE_PER_NATIVE_SEQUENCE'),
              'once/iteration occurrence count differs from original activity scope')
    bindings = {row.occupancy: row for row in resources.executions}
    execution = [row for row in temporal.occupancies if row.kind is O.EXECUTION_DEMAND]
    _need(len(bindings) == len(resources.executions) == len(execution)
          and set(bindings) == {row.id for row in execution}, 'complete original execution binding coverage required')
    by_activity = defaultdict(list)
    for occupancy in execution:
        binding = bindings[occupancy.id]
        _need((binding.resource, binding.units) == (occupancy.resource, occupancy.units),
              'workspace execution units/resource differ from original temporal occupancy')
        by_activity[occupancy.activity].append(binding)
    proof_rows = [row for row in temporal.source_refs if row.get('schema') == _DEMAND_SCHEMA]
    plan = context = None; records = {}; rejected = []
    if primitive_evidence is not None:
        _need(type(primitive_evidence) is dict and len(proof_rows) == 1,
              'primitive evidence requires one source-qualified workspace CPU domain proof')
        source_proof = proof_rows[0]
        _need(source_proof.get('schema_version') == 1
              and source_proof.get('target') == encode(resources.target)
              and source_proof.get('workspace_plan') == workspace_plan_ref
              and source_proof.get('native_manifest') == certificate.get('native_manifest'),
              'workspace primitive source proof target/plan differ')
        plan, context, records, rejected = _primitive_records(primitive_evidence, resources, source_proof)
        domains = {row['occupancy']: row for row in source_proof['occupancies']}
    costs = []; ledger = []; pooled = set(); known_subtotal = 0; matched = []; unmatched = []
    for task in schedule.tasks:
        if task.id in pooled:
            continue
        selected = [task]
        if task.kind in (K.ALLOCATE, K.RELEASE):
            selected = [row for row in schedule.tasks if row.kind is task.kind]
            pooled.update(row.id for row in selected)
        members = tuple(by_task[row.id].id for row in selected)
        scope = by_task[task.id].scope
        _need(all(by_task[row.id].scope == scope for row in selected), 'collective pool cannot mix once and loop scopes')
        _need(task.kind not in (K.ALLOCATE, K.RELEASE) or scope == root.id,
              'workspace pool allocation/release must be outside the shared body')
        multiplicity = count if scope == loop.id else 1
        subject = 'cost:workspace:' + semantic_sha256(dict(schedule=digest, activities=members))[:32]
        cost = ActivityCost(subject, members, CostBasis.UNKNOWN,
            missing_terms=('NO_MATCHING_SOURCE_QUALIFIED_WORKSPACE_COMPONENT_COST',))
        if task.kind is K.SNAPSHOT and plan is not None:
            from scar.optimization.stateful_cpu_model import component_key
            parameters = _snapshot_parameters(task, plan, schedule)
            key = component_key('ROW_COPY_STAGE', parameters)
            candidates = by_activity[members[0]]
            binding = candidates[0] if len(candidates) == 1 else None
            fact = records.get(key)
            reason = 'NO_EXACT_CONTEXT_AND_GEOMETRY_COPY_PRIMITIVE'
            if binding is not None and fact is not None:
                domain = domains.get(binding.occupancy, {})
                exact = binding.exact_thread; workers = context['worker_cpu_ids']
                applicable = (exact is not None and exact.snapshot == resources.target
                    and binding.candidate_threads == (exact,) and binding.units == 1
                    and binding.resource == str(exact.resource) and len(workers) == 1
                    and domain.get('exact_cpu_id') == workers[0] and domain.get('cpu_ids') == workers
                    and (domain.get('task'), domain.get('activity'), domain.get('scope'))
                        == (task.id, members[0], scope)
                    and (domain.get('resource'), domain.get('units')) == (binding.resource, 1))
                if applicable:
                    reference, stats, conditions = fact
                    cost = ActivityCost(subject, members, CostBasis.DECLARED_COMPONENT_MODEL,
                        estimate_ticks=math.ceil(stats['median_ns']),
                        interval=CostInterval(math.floor(stats['min_ns']), math.ceil(stats['max_ns']),
                            IntervalKind.EMPIRICAL_PREDICTION),
                        component_refs=(ProviderFactReference(ProviderArtifact(**reference['artifact']), reference['locator']),),
                        charge_keys=('workspace-stage-occurrence:' + semantic_sha256(dict(
                            schedule=digest, task=task.id, scope=scope)),), coverage=CostCoverage.COMPONENT_ONLY,
                        conditions=(*conditions, 'CANONICAL_COPY_SAMPLE_IS_NOT_OBSERVED_WORKSPACE_STAGE_LATENCY'),
                        missing_terms=('WORKSPACE_LIFETIME_CACHE_ALLOCATOR_AND_DISPATCH_DELTA_NOT_VALIDATED',))
                    matched.append(dict(task=task.id, subject=subject, signature=parameters, fact=reference))
                else:
                    reason = 'SOURCE_PROVEN_EXACT_CALLING_THREAD_OR_WHOLE_THREAD_UNITS_MISMATCH'
            if cost.basis is CostBasis.UNKNOWN:
                unmatched.append(dict(task=task.id, signature=parameters, reason=reason, cost_charged=False))
        if cost.estimate_ticks is not None:
            known_subtotal += multiplicity * cost.estimate_ticks
        costs.append(cost)
        ledger.append(dict(subject=subject, activities=list(members), task_ids=[row.id for row in selected],
            scope=scope, occurrence_count=multiplicity,
            role='PER_ITERATION_TEMPLATE' if scope == loop.id else 'ONCE_PER_NATIVE_SEQUENCE',
            estimate_per_occurrence_ticks=cost.estimate_ticks,
            residual=dict(signed_unknown=True, lower_ticks=None, upper_ticks=None, set_to_zero=False,
                aggregation='SUM_OVER_ITERATIONS_WITH_DISTINCT_I_INDEX' if scope == loop.id else 'ONE_OUTER_RESIDUAL',
                across_iteration_equality_assumed=False, cancellation_assumed=False)))
    refs = (*source_refs, dict(artifact=workspace_time_ref, locator='#/schedule'),
            dict(artifact=workspace_plan_ref, locator='#'))
    table = ActivityCostTable(digest, semantic_sha256(projection_document(resources)), resources.target,
        tuple(costs), source_refs=refs, unmapped_terms=(dict(kind='WORKSPACE_OUTER_BOUNDARY_UNMODELED',
            missing_terms=['PYTHON_PACKING_LOCK_AND_FFI', 'FAILURE_PATH_TIMING'], cost_charged=False),))
    summary = validate_activity_cost_table(table, schedule, resources)
    allocations = [row for row in schedule.tasks if row.kind is K.ALLOCATE]
    releases = [row for row in schedule.tasks if row.kind is K.RELEASE]
    allocated_slots = sum(len(row.allocates) for row in allocations)
    released_slots = sum(len(row.releases) for row in releases)
    accounting = dict(schema=SCHEMA, schema_version=1, source_schedule_sha256=digest,
        resource_projection_sha256=table.resource_projection_sha256,
        workspace_time=workspace_time_ref, workspace_plan=workspace_plan_ref,
        loop_scope=loop.id, trip_count=count, subjects=ledger, matched_primitives=matched,
        unmatched_primitives=unmatched,
        rejected_calibrations=rejected, known_component_model_subtotal_ticks=known_subtotal,
        normal_sequence_duration_ticks=None, unknown_duration_imputed=False,
        pool_accounting=dict(original_per_iteration_allocation_events=count*allocated_slots,
            original_per_iteration_release_events=count*released_slots,
            outer_allocation_events=allocated_slots, outer_release_events=released_slots,
            outer_allocation_task_occurrences=len(allocations), outer_release_task_occurrences=len(releases),
            outer_pool_subjects_charged_once=True, pool_cost_split_by_slot=False,
            original_body_pool_subjects_retained=False, physical_slots_multiplied_by_trip_count=False),
        summary=summary, physical_scope_latency=None, source_component_bytes_admitted_by_caller_required=True,
        sample_ranges_are_guaranteed_bounds=False, residuals_cancelled=False,
        complete_call_cost=False, solver_authorized=False, execution_authorized=False,
        physical_optimality_proven=False, deployment_authorized=False)
    if primitive_evidence is not None:
        accounting['canonical_source_bridge'] = dict(kind='EXACT_CONTEXT_RAW_ROOT_COPY_PRIMITIVES_ONLY',
            canonical_domain_id=primitive_evidence['domain']['domain_id'],
            actual_step_plan=primitive_evidence['actual_step_plan_reference'],
            actual_step_plan_sha256=semantic_sha256(primitive_evidence['actual_step_plan']),
            actual_backend_sha256=semantic_sha256(primitive_evidence['actual_backend']),
            canonical_resource_context_sha256=semantic_sha256(context),
            original_region_identity_required=False,
            actual_source_stage_geometry_and_materializations_checked=True)
    return table, accounting


__all__ = ['SCHEMA', 'derive_workspace_costs']

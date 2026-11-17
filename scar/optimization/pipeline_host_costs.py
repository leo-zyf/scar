"""Observed costs of instrumented source fragments, never clean service prices."""
import statistics

from scar.analysis.pipeline_host_samples import SCHEMA as OBSERVATIONS_SCHEMA, summarize_host_samples
from scar.ir.activity_costs import (ActivityCost, ActivityCostTable, CostBasis,
    CostCoverage, CostInterval, IntervalKind, validate_activity_cost_table)
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import decode, encode
from scar.ir.temporal_resources import projection_document, projection_from_dict
from scar.ir.versioned_schedule import TemporalMode, schedule_document, validate_versioned_schedule
from scar.optimization.resource_solver import semantic_sha256

_CONDITIONS = ('OBSERVED_SCOPE_NOT_FUTURE_SERVICE_ESTIMATE',
    'INSTRUMENTED_SCOPE_NOT_INTRINSIC_CLEAN_COST', 'NONADDITIVE_ACROSS_THREADS',
    'COMPLETE_SUBJECT_IS_ONLY_THIS_INSTRUMENTED_CUT_FRAGMENT',
    'EMPIRICAL_RANGE_IS_NOT_A_MATHEMATICAL_OR_PHYSICAL_BOUND',
    'OBSERVED_ABI_ROWS_REPLACE_NOT_ADD_EXISTING_LEAF_SERVICE_PROFILES',
    'KNOWN_ELAPSED_SCOPE_DOES_NOT_PROVE_KNOWN_CPU_DEMAND')


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _token(event):
    rows = [row for row in event.source_refs if 'probe_cut' in row]
    _need(len(rows) <= 1, 'one exact original cut token per observed event required')
    if not rows:
        return None
    row = rows[0]; value = row['probe_cut']
    _need(type(value) in (tuple, list) and len(value) == 3
          and value[0] in ('caller', 'lane', 'call') and type(value[2]) is str
          and row.get('clock') == 'CLOCK_MONOTONIC' and row.get('instrumentation_subtracted') is False,
          'original same-clock probe cut token required')
    return tuple(value)


def _value(trace, token):
    group, owner, name = token
    if group == 'caller':
        _need(owner is None, 'caller cut cannot acquire a worker identity')
        cuts = trace['caller']
    else:
        key, rows = ('cpu_id', trace['lanes']) if group == 'lane' else ('call_id', trace['calls'])
        selected = [row for row in rows if row[key] == owner]
        _need(len(selected) == 1, 'cut must select one original trace actor/call')
        cuts = selected[0]['cuts']
    _need(name in cuts and cuts[name]['valid'] is True, 'original event cut is absent or invalid')
    return cuts[name]['ns']


def _missing(role, activity, start, end):
    if role == 'SYMBOLIC_SLOT_MEMBER_NOT_SEPARATELY_CHARGEABLE':
        return ('SYMBOLIC_SLOT_MEMBER_AGGREGATE_CHARGED_ONCE_NOT_SEPARATE_WORK',)
    if role.startswith('ASYNC_LAUNCH'):
        return ('ASYNCHRONOUS_START_LATENCY_IS_NOT_FIXED_CPU_SERVICE',)
    if role.startswith('WAIT_INCLUSIVE_JOIN'):
        return ('JOIN_INCLUDES_CONCURRENT_WORK_AND_WAIT_NOT_FIXED_CPU_SERVICE',)
    if role.startswith('EXIT_'):
        return ('WORKER_EXIT_AND_DISPATCH_WAIT_HAS_NO_EXCLUSIVE_OBSERVED_BOUNDARIES',)
    if role == 'PUBLISH' or activity.ready_events:
        return ('PUBLICATION_RELEASE_OR_READY_ENDPOINT_IS_ONLY_BRACKETED_NOT_OBSERVED',)
    if start is None or end is None:
        return ('UNOBSERVED_PYTHON_CONTROL_OR_SYMBOLIC_EVENT_ENDPOINT',)
    return ()


def derive_pipeline_host_costs(semantic_schedule, resource_projection, observations, *, observations_ref):
    """Recompute each admitted raw interval and preserve all unknown activities.

    Source/probe/native byte admission belongs to the publication reader. This
    pure bridge independently checks trace protocol, exact event cut mappings,
    raw sample statistics and cost ownership. It does not import leaf prices.
    """
    validate_versioned_schedule(semantic_schedule)
    model = semantic_schedule.temporal
    _need(model is not None and model.mode is TemporalMode.SEMANTIC
          and all(event.timestamp_ticks is None for event in model.events),
          'original semantic host graph required, not one observed sample graph')
    resources = projection_from_dict(projection_document(resource_projection))
    reference = observations_ref if type(observations_ref) is ProviderArtifact else decode(ProviderArtifact, observations_ref)
    _need(reference.schema == OBSERVATIONS_SCHEMA, 'original host observations reference required')
    summarize_host_samples(observations)
    proof = semantic_schedule.proof.get('pipeline_host_envelope', {})
    _need((proof.get('schema'), proof.get('schema_version')) == ('scar.pipeline-host-envelope', 1)
          and proof.get('trace_sha256') is None
          and any(row.get('artifact') == encode(reference) for row in model.source_refs)
          and any(row.get('artifact') == observations['probe'] for row in model.source_refs),
          'semantic host source must retain these exact raw observations and probe parents')
    phases = {row['activity']:row for row in proof['phases']}
    _need(len(phases) == len(proof['phases']) == len(model.activities)
          and set(phases) == {a.id for a in model.activities}, 'complete original source phase ownership required')
    events = {event.id:event for event in model.events}
    by_task = {}
    for activity in model.activities:
        by_task.setdefault(activity.task, []).append(activity)
    groups = proof['physical_cost_groups']; symbolic = set(); aggregate = set()
    for group in groups:
        _need(group['aggregate_activity'] in phases and group['aggregate_activity'] not in aggregate
              and group['additive_charge_count'] == 1
              and group['per_slot_cost_or_exact_timestamps_inferred'] is False,
              'each original pool must have one aggregate cost owner')
        aggregate.add(group['aggregate_activity'])
        for task in group['symbolic_original_tasks']:
            members = by_task.get(task, ())
            _need(len(members) == 1 and phases[members[0].id]['role']
                  == 'SYMBOLIC_SLOT_MEMBER_NOT_SEPARATELY_CHARGEABLE' and members[0].id not in symbolic,
                  'symbolic pool members must not acquire duplicate independent prices')
            symbolic.add(members[0].id)
    digest = semantic_sha256(schedule_document(semantic_schedule))
    rows = []; witnesses = []; unknown = []
    facts = (ProviderFactReference(reference, '#/samples'),)
    for activity in model.activities:
        phase = phases[activity.id]; role = phase['role']
        _need((phase['task'], phase['start'], phase['completion'])
              == (activity.task, activity.start, activity.completion), 'source activity phase mapping differs')
        start, end = _token(events[activity.start]), _token(events[activity.completion])
        missing = _missing(role, activity, start, end)
        subject = 'instrumented-host:'+activity.id
        if not missing:
            values = [_value(sample['trace'], end)-_value(sample['trace'], start)
                      for sample in observations['samples']]
            _need(all(type(v) is int and v >= 0 for v in values), 'same-clock fragment interval is negative')
            median = statistics.median(values)
            if type(median) is float and not median.is_integer():
                missing = ('FRACTIONAL_NANOSECOND_MEDIAN_HAS_NO_INTEGER_TICK_REPRESENTATION',)
        if missing:
            rows.append(ActivityCost(subject, (activity.id,), CostBasis.UNKNOWN,
                component_refs=facts, conditions=_CONDITIONS, missing_terms=missing))
            unknown.append(dict(subject=subject, activity=activity.id, role=role,
                missing_terms=list(missing), numeric_cost_imputed=False))
            continue
        point, lower, upper = int(median), min(values), max(values)
        rows.append(ActivityCost(subject, (activity.id,), CostBasis.OBSERVED_SCOPE, point,
            CostInterval(lower, upper, IntervalKind.EMPIRICAL_PREDICTION), facts,
            ('host-observation:'+digest+':'+activity.id,), CostCoverage.COMPLETE_SUBJECT,
            _CONDITIONS, ()))
        witnesses.append(dict(schema='scar.observed-activity-cost-proof', schema_version=1,
            subject=subject, activities=[activity.id], start_event=activity.start,
            completion_event=activity.completion, scope=activity.scope, observation=encode(reference),
            samples_locator='#/samples', sample_count=len(values),
            interpretation='INSTRUMENTED_SCOPE_OBSERVATION_NOT_INTRINSIC_SERVICE',
            statistics=dict(estimate_ticks=point, lower_ticks=lower, upper_ticks=upper)))
    _need(symbolic.isdisjoint({row.activities[0] for row in rows if row.basis is CostBasis.OBSERVED_SCOPE}),
          'pool member price duplicates its aggregate observation')
    table = ActivityCostTable(digest, semantic_sha256(projection_document(resources)), resources.target,
        tuple(rows), source_refs=(dict(artifact=encode(reference), locator='#',
            source_admission_required_upstream=True, whole_scope_cost_closed=False,
            observations_are_not_clean_service_estimates=True), *witnesses),
        unmapped_terms=(dict(kind='INSTRUMENTATION_AND_FUTURE_CLEAN_IMPLEMENTATION_DELTA',
            conditions=list(observations.get('conditions', ())), residuals=list(observations.get('residuals', ())),
            cost_charged=False, delta_set_to_zero=False),
            dict(kind='UNOBSERVED_OUTER_PYTHON_GUARDS_ALLOCATION_FFI_AND_RETURN', cost_charged=False),
            dict(kind='UNKNOWN_WAIT_AND_BRACKETED_PUBLICATION_SCOPES', subjects=unknown, cost_charged=False),
            dict(kind='EXISTING_LEAF_PROFILE_NOT_ADDED', original_abi_observed_once=True, cost_charged=False)))
    validate_activity_cost_table(table, semantic_schedule, resources)
    return table


__all__ = ['derive_pipeline_host_costs']

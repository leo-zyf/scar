"""Choose between two certified copy graphs inside a canonical window model.

The publication reader separately qualifies source, compiled probes and raw
measurements. This pure bridge rechecks graph/resource/cost identities and
never turns a component-model preference into a runtime deployment decision.
"""
from dataclasses import replace
import statistics

from scar.ir.activity_costs import (ActivityCost, CostBasis, CostCoverage, CostInterval,
    IntervalKind, activity_cost_document, validate_activity_cost_table)
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import encode
from scar.ir.temporal_resources import projection_document, projection_from_dict
from scar.ir.versioned_schedule import TemporalOccupancyKind as O, schedule_document
from scar.optimization.fused_copy_graph import verify_fused_copy_graph
from scar.optimization.resource_solver import semantic_sha256
from scar.planner.choice import FiniteMedianOption, choose_finite_medians

SCHEMA = 'scar.copy-window-choice'
CALIBRATION_SCHEMA = 'scar.copy-window-calibration'
_CLOCK = dict(api='clock_gettime', clock_id='CLOCK_MONOTONIC', unit='ns')
_DISCHARGED = 'FUSED_COPY_COMPONENT_COST_NOT_CALIBRATED'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _strings(value, name):
    _need(type(value) is list and value and all(type(row) is str and row for row in value)
          and len(set(value)) == len(value), 'unique nonempty canonical copy-window ' + name + ' required')
    return tuple(value)


def choose_copy_window(original_schedule, original_resources, original_table,
                       rewritten_schedule, rewritten_resources, rewritten_table,
                       certificate, calibration_reference, calibration_document):
    """Return two newly priced tables and a preference, with default NO_OP."""
    verify_fused_copy_graph(original_schedule, rewritten_schedule, certificate)
    original_resources = projection_from_dict(projection_document(original_resources))
    rewritten_resources = projection_from_dict(projection_document(rewritten_resources))
    validate_activity_cost_table(original_table, original_schedule, original_resources)
    validate_activity_cost_table(rewritten_table, rewritten_schedule, rewritten_resources)
    old_sha = semantic_sha256(schedule_document(original_schedule))
    new_sha = semantic_sha256(schedule_document(rewritten_schedule)); cert_sha = semantic_sha256(certificate)
    old_resource_sha = semantic_sha256(projection_document(original_resources))
    new_resource_sha = semantic_sha256(projection_document(rewritten_resources))
    _need((certificate['parent_schedule_sha256'], certificate['rewritten_schedule_sha256']) == (old_sha, new_sha)
          and original_resources.target == rewritten_resources.target
          and original_resources.storages == rewritten_resources.storages
          and original_resources.budget == rewritten_resources.budget,
          'certified original/replacement graph, hardware and unchanged storage/budget required')
    reference = ProviderArtifact(**calibration_reference); document = calibration_document
    _need(reference.schema == CALIBRATION_SCHEMA and type(document) is dict
          and (document.get('schema'), document.get('schema_version')) == (CALIBRATION_SCHEMA, 1),
          'reader-qualified versioned copy-window calibration required')
    _need((document.get('parent_schedule_sha256'), document.get('rewritten_schedule_sha256'),
           document.get('certificate_sha256')) == (old_sha, new_sha, cert_sha),
          'calibration must retain the exact two graphs and complete rewrite certificate')
    rewrite = ProviderArtifact(**document['rewrite'])
    _need(rewrite.schema == 'scar.window-rewrite-preparation', 'original actual rewrite publication reference required')
    _need(document.get('clock') == _CLOCK, 'one shared monotonic nanosecond calibration clock required')
    reps, warmup = document.get('repetitions'), document.get('warmup')
    _need(type(reps) is int and 3 <= reps <= 101 and reps % 2 == 1
          and type(warmup) is int and warmup >= 0, 'odd repeated whole-window samples and explicit warmup required')
    samples = document.get('samples')
    _need(type(samples) is list and len(samples) == reps, 'all paired canonical window samples required')
    for index, row in enumerate(samples):
        _need(type(row.get('index')) is int and row['index'] == index
              and row.get('order') in (['ORIGINAL', 'FUSED'], ['FUSED', 'ORIGINAL'])
              and row.get('checksums_equal') is True
              and all(type(row.get(name)) is int and row[name] >= 0 for name in ('original_ns', 'fused_ns')),
              'ordered paired integer measurements with equal canonical outputs required')
    stats = {}
    for variant in ('original', 'fused'):
        values = [row[variant+'_ns'] for row in samples]
        stats[variant] = dict(median_ns=statistics.median(values), min_ns=min(values), max_ns=max(values),
                              stdev_ns=statistics.stdev(values))
    _need(document.get('statistics') == stats, 'canonical whole-window statistics differ from retained paired samples')
    conditions = _strings(document.get('conditions'), 'conditions')
    missing = _strings(document.get('missing_terms'), 'residuals')
    original_task_ids = certificate['original_task_ids']; replacement = certificate['replacement_task_id']
    old_activities = tuple(next(row.id for row in original_schedule.temporal.activities if row.task == name)
                           for name in original_task_ids)
    new_activities = tuple(row.id for row in rewritten_schedule.temporal.activities if row.task == replacement)
    _need(len(new_activities) == 1, 'one replacement copy activity required')
    old_occupancies = [row for row in original_schedule.temporal.occupancies
                       if row.kind is O.EXECUTION_DEMAND and row.activity in old_activities]
    old_bindings = {row.occupancy: row for row in original_resources.executions}
    new_bindings = {row.occupancy: row for row in rewritten_resources.executions}
    _need(len(old_occupancies) == len(old_activities), 'one original execution binding per copied task required')
    old_exec = {row.id: row for row in original_schedule.temporal.occupancies if row.kind is O.EXECUTION_DEMAND}
    new_exec = {row.id: row for row in rewritten_schedule.temporal.occupancies if row.kind is O.EXECUTION_DEMAND}
    _need(set(old_bindings) == set(old_exec) and set(new_bindings) == set(new_exec)
          and {certificate['occupancy_map'][name] for name in old_bindings} == set(new_bindings),
          'complete original and replacement execution-binding coverage required')
    _need(all((row.resource, row.units) == (old_exec[name].resource, old_exec[name].units) for name, row in old_bindings.items())
          and all((row.resource, row.units) == (new_exec[name].resource, new_exec[name].units) for name, row in new_bindings.items()),
          'execution binding and temporal occupancy resource/units differ')
    selected = [old_bindings[row.id] for row in old_occupancies]
    exact = selected[0].exact_thread
    _need(exact is not None and exact.snapshot == original_resources.target
          and all(row.exact_thread == exact and row.candidate_threads == (exact,) and row.units == 1
                  and row.resource == str(exact.resource) for row in selected),
          'all original copies must retain one exact unit-one calling thread')
    selected_occupancy_ids = {row.id for row in old_occupancies}
    merged = {certificate['occupancy_map'][name] for name in selected_occupancy_ids}
    _need(len(merged) == 1, 'one certified replacement execution occupancy required')
    fused_binding = new_bindings[next(iter(merged))]
    _need((fused_binding.resource, fused_binding.exact_thread, fused_binding.candidate_threads,
            fused_binding.core_refs, fused_binding.units)
          == (selected[0].resource, exact, (exact,), selected[0].core_refs, 1),
          'replacement must retain the exact original calling-thread allocation')
    for occupancy, binding in old_bindings.items():
        if occupancy not in selected_occupancy_ids:
            _need(certificate['occupancy_map'][occupancy] == occupancy and new_bindings[occupancy] == binding,
                  'unmodified exact or unknown execution bindings cannot change')
    context = document['resource_context']
    _need(context.get('target_snapshot') == encode(original_resources.target)
          and context.get('exact_thread_refs') == [encode(exact)]
          and type(context.get('cpu_ids')) is list and len(context['cpu_ids']) == 1
          and type(context['cpu_ids'][0]) is int and context['cpu_ids'][0] >= 0,
          'original inventory-qualified CPU number and exact hardware reference required; labels are not parsed')
    old_selected = set(old_activities); new_selected = set(new_activities)
    retired_old = [row for row in original_table.costs if old_selected.intersection(row.activities)]
    retired_new = [row for row in rewritten_table.costs if new_selected.intersection(row.activities)]
    _need(len(retired_old) == len(old_activities)
          and all(len(row.activities) == 1 and set(row.activities) <= old_selected for row in retired_old)
          and {a for row in retired_old for a in row.activities} == old_selected,
          'the original complete window must replace whole singleton subjects without splitting a collective cost')
    _need(len(retired_new) == 1 and retired_new[0].activities == new_activities
          and retired_new[0].basis is CostBasis.UNKNOWN,
          'the replacement must have one complete previously UNKNOWN copy subject')
    other_old = [row for row in original_table.costs if row not in retired_old]
    other_new = [row for row in rewritten_table.costs if row not in retired_new]
    _need(other_old == other_new, 'all other conditional component and source milestone rows must remain unchanged')
    unmapped_discharges = []
    def reprice(table, retired, activities, variant, schedule_sha):
        prior_missing = [value for row in retired for value in row.missing_terms if value != _DISCHARGED]
        residuals = tuple(dict.fromkeys((*missing, *prior_missing)))
        applicability = tuple(dict.fromkeys((*conditions, *(value for row in retired for value in row.conditions))))
        subject = 'cost:copy-window:' + semantic_sha256(dict(source=calibration_reference,
            variant=variant, schedule=schedule_sha, activities=activities))[:32]
        charge = 'copy-window-occurrence:' + semantic_sha256(dict(source=calibration_reference,
            variant=variant, schedule=schedule_sha, activities=activities))
        stat = stats[variant]
        row = ActivityCost(subject, activities, CostBasis.DECLARED_COMPONENT_MODEL,
            estimate_ticks=stat['median_ns'], interval=CostInterval(stat['min_ns'], stat['max_ns'], IntervalKind.EMPIRICAL_PREDICTION),
            component_refs=(ProviderFactReference(reference, '#/statistics/'+variant),),
            charge_keys=(charge,), coverage=CostCoverage.COMPONENT_ONLY,
            conditions=applicability, missing_terms=residuals)
        retired_ids = {value.subject for value in retired}; costs = []; inserted = False
        for value in table.costs:
            if value.subject in retired_ids:
                if not inserted:
                    costs.append(row); inserted = True
            else:
                costs.append(value)
        source = dict(schema='scar.copy-window-cost-lineage', schema_version=1,
            calibration=calibration_reference, variant=variant.upper(), locator='#/statistics/'+variant,
            retired_subjects=[value.subject for value in retired], replacement_subject=subject,
            old_component_sum_used=False, remaining_context_unvalidated=True)
        unmapped = []
        parent_table_sha = semantic_sha256(activity_cost_document(table))
        for index, record in enumerate(table.unmapped_terms):
            if (variant == 'fused' and record.get('kind') == 'FUSED_COPY_IMPLEMENTATION_CONTEXT_RESIDUAL'
                    and record.get('certificate_sha256') == cert_sha
                    and _DISCHARGED in record.get('missing_terms', ())):
                lineage = dict(calibration=dict(artifact=calibration_reference, locator='#/statistics/fused'),
                    parent_cost_table_sha256=parent_table_sha, parent_unmapped_term_ordinal=index,
                    parent_locator='#/unmapped_terms/'+str(index),
                    parent_record_sha256=semantic_sha256(record), discharged_missing_terms=[_DISCHARGED])
                changed = dict(record, missing_terms=[term for term in record['missing_terms'] if term != _DISCHARGED],
                    evidence_gap_discharge=lineage)
                unmapped.append(changed)
                unmapped_discharges.append(dict(**lineage, original_source_record=encode(record),
                    updated_record_sha256=semantic_sha256(changed)))
            else:
                unmapped.append(record)
        return replace(table, costs=tuple(costs), source_refs=(*table.source_refs, source),
            unmapped_terms=tuple(unmapped)), row
    original_costs, original_row = reprice(original_table, retired_old, old_activities, 'original', old_sha)
    replacement_costs, fused_row = reprice(rewritten_table, retired_new, new_activities, 'fused', new_sha)
    summaries = dict(original=validate_activity_cost_table(original_costs, original_schedule, original_resources),
                     fused=validate_activity_cost_table(replacement_costs, rewritten_schedule, rewritten_resources))
    original_stat, fused_stat = stats['original'], stats['fused']
    comparison_scope = 'ONE_COMPLETE_CANONICAL_COPY_WINDOW'
    comparison_context = dict(calibration=calibration_reference, certificate_sha256=cert_sha,
                              resource_context=encode(context))
    finite_choice = choose_finite_medians([
        FiniteMedianOption('ORIGINAL', comparison_scope, comparison_context,
                          original_stat['median_ns'], original_row.component_refs),
        FiniteMedianOption('FUSED', comparison_scope, comparison_context,
                          fused_stat['median_ns'], fused_row.component_refs)],
        original_id='ORIGINAL', scope=comparison_scope, context=comparison_context,
        tick_numerator=1, tick_denominator=1_000_000_000)
    _need(finite_choice['status'] == 'FINITE_MODEL_MINIMUM',
          'complete reader-qualified canonical window medians required for copy choice')
    selected_variant = finite_choice['selected_id']
    decision = ('TRIAL_REWRITE_WITHIN_COMPONENT_MODEL' if selected_variant == 'FUSED' else
                'KEEP_ORIGINAL_ON_TIE' if len(finite_choice['tied_minimum_ids']) > 1 else
                'KEEP_ORIGINAL_WITHIN_COMPONENT_MODEL')
    separation = ('FUSED_RANGE_BELOW_ORIGINAL' if fused_stat['max_ns'] < original_stat['min_ns'] else
                  'ORIGINAL_RANGE_BELOW_FUSED' if original_stat['max_ns'] < fused_stat['min_ns'] else
                  'EMPIRICAL_RANGES_OVERLAP_OR_TOUCH')
    old_table_sha = semantic_sha256(activity_cost_document(original_table))
    new_table_sha = semantic_sha256(activity_cost_document(rewritten_table))
    def retired_refs(table, digest, rows):
        return [dict(subject=row.subject, table_sha256=digest, locator='#/costs/'+str(table.costs.index(row)),
            record_sha256=semantic_sha256(encode(row)), retired_charge_keys=list(row.charge_keys),
            retired_missing_terms=list(row.missing_terms)) for row in rows]
    alternatives = []
    for variant, graph_sha, resource_sha, table_result, row, stat in (
            ('ORIGINAL', old_sha, old_resource_sha, original_costs, original_row, original_stat),
            ('FUSED', new_sha, new_resource_sha, replacement_costs, fused_row, fused_stat)):
        alternatives.append(dict(variant=variant, schedule_sha256=graph_sha, resource_projection_sha256=resource_sha,
            cost_table_sha256=semantic_sha256(activity_cost_document(table_result)), subject=row.subject,
            activities=list(row.activities), statistics=stat, cost_scope='ONE_COMPLETE_CANONICAL_COPY_WINDOW',
            basis=row.basis.value, coverage=row.coverage.value, conditions=list(row.conditions),
            residuals=list(row.missing_terms), physical_window_latency=None))
    choice = dict(schema=SCHEMA, schema_version=1, selected_variant=selected_variant, decision=decision,
        objective='MINIMIZE_RETAINED_CANONICAL_COMPLETE_WINDOW_MEDIAN_NS_AT_ONE_FIXED_CPU',
        alternatives=alternatives, finite_choice=finite_choice,
        empirical_separation=separation, summaries=summaries,
        inputs=dict(calibration=calibration_reference, rewrite=encode(rewrite), certificate_sha256=cert_sha,
            parent_schedule_sha256=old_sha, rewritten_schedule_sha256=new_sha,
            parent_resource_projection_sha256=old_resource_sha, rewritten_resource_projection_sha256=new_resource_sha,
            original_cost_table_sha256=old_table_sha, rewritten_input_cost_table_sha256=new_table_sha),
        source_maps=dict(certificate_sha256=cert_sha,
            task_map=certificate['task_map'], event_map=certificate['event_map'],
            activity_map=certificate['activity_map'], occupancy_map=certificate['occupancy_map']),
        retired_costs=dict(original=retired_refs(original_table, old_table_sha, retired_old),
            fused=retired_refs(rewritten_table, new_table_sha, retired_new)),
        discharged_evidence_gap=dict(term=_DISCHARGED,
            retired_from=[row.subject for row in retired_new if _DISCHARGED in row.missing_terms],
            reason='SOURCE_QUALIFIED_CANONICAL_WHOLE_WINDOW_COMPONENT_NOW_MEASURED_TARGET_DELTA_REMAINS',
            source_lineage=unmapped_discharges),
        resource_context=encode(context), sample_indices=[row['index'] for row in samples],
        conditions=list(conditions), missing_terms=list(missing),
        calibration_reader_admission_required=True, calibration_source_semantics_replayed_by_this_adapter=False,
        scope='TWO_CERTIFIED_GRAPH_COMPONENT_MODELS_ONLY_NOT_GLOBAL_TARGET_VARIANT_SEARCH',
        prior_component_cost_sum_used=False, unrelated_unmapped_source_records_unchanged=True,
        existing_context_residuals_preserved=True,
        matching_evidence_gap_field_discharged=bool(unmapped_discharges),
        unmodeled_residuals_cancelled=False, empirical_ranges_are_physical_guarantees=False,
        whole_scope_latency=None, global_v3_solver_invoked=False, physical_optimality_proven=False,
        physical_profitability_proven=False, deployment_authorized=False,
        automatic_runtime_rewrite_authorized=False, default_runtime_decision='NO_OP')
    return original_costs, replacement_costs, choice


__all__ = ['SCHEMA', 'choose_copy_window']

"""Source-qualified exclusive workspace boundary component predictions.

Raw same-clock cuts partition each canonical trace. Loop phase statistics
pool the retained (repetition, iteration) observations; they are not a trace
total, an observed target activity, or guaranteed latency bounds.
"""
from copy import deepcopy
from dataclasses import replace
import math
import statistics

from scar.ir.activity_costs import (ActivityCost, CostBasis, CostCoverage, CostInterval,
    IntervalKind, validate_activity_cost_table)
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import encode
from scar.ir.versioned_schedule import VersionedTaskKind as K
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.workspace-boundary-calibration'
PREFIX = ('native_enter', 'outer_preflight_done', 'pool_allocate_done')
ITERATION = ('step_enter', 'bind_done', 'restore_begin', 'restore_done', 'step_returned', 'progress_done')
SUFFIX = ('loop_done', 'before_pool_release', 'after_pool_release')


def _phase(name, scope, section, start, end, **selector):
    return dict(id=name, scope=scope, start=dict(section=section, offset=start),
                end=dict(section=section, offset=end), **selector)


PHASE_DEFINITIONS = (
    _phase('outer_preflight', 'ROOT', 'prefix', 0, 1,
           phase_kind='WORKSPACE_ABI_TABLE_AND_PAYLOAD_PREFLIGHT'),
    _phase('pool_allocate', 'ROOT', 'prefix', 1, 2, task_kind='ALLOCATE'),
    _phase('iteration_preflight_context', 'LOOP', 'iteration', 0, 1,
           phase_kind='ORIGINAL_PER_STEP_PREFLIGHT_AND_CONTEXT'),
    _phase('iteration_validate_restore', 'LOOP', 'iteration', 2, 3,
           phase_kind='ORIGINAL_PER_STEP_VALIDATE_AND_RESTORE'),
    _phase('iteration_cleanup_return_progress', 'LOOP', 'iteration', 3, 5,
           phase_kind='ORIGINAL_HELPER_CLEANUP_STEP_RETURN_AND_PROGRESS'),
    _phase('pool_release', 'ROOT', 'suffix', 1, 2, task_kind='RELEASE'),
)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _strings(value, name):
    _need(type(value) in (list, tuple) and value
          and all(type(row) is str and row for row in value) and len(set(value)) == len(value),
          'unique nonempty workspace boundary ' + name + ' required')
    return tuple(value)


def cut_layout(trip_count):
    _need(type(trip_count) is int and trip_count > 0, 'fixed positive boundary probe trip count required')
    return dict(prefix=list(PREFIX), iteration=list(ITERATION), suffix=list(SUFFIX),
                iteration_count=trip_count, timestamps_per_sample=6 * trip_count + 6)


def _indices(phase, count, iteration=0):
    section = phase['start']['section']
    base = 0 if section == 'prefix' else 3 + 6 * count if section == 'suffix' else 3 + 6 * iteration
    return base + phase['start']['offset'], base + phase['end']['offset']


def boundary_statistics(document):
    """Recompute retained per-phase samples and every excluded trace interval."""
    _need((document.get('schema'), document.get('schema_version')) == (SCHEMA, 1),
          'versioned workspace boundary calibration required')
    count = document.get('trip_count'); layout = cut_layout(count)
    _need(document.get('cut_layout') == layout and document.get('phase_definitions') == list(PHASE_DEFINITIONS),
          'exact workspace cuts and six nonoverlapping source phase definitions required')
    _need(document.get('clock') == dict(api='clock_gettime', clock_id='CLOCK_MONOTONIC', unit='ns'),
          'one original monotonic nanosecond clock required')
    _strings(document.get('conditions'), 'conditions'); _strings(document.get('missing_terms'), 'residual terms')
    repetitions, warmup = document.get('repetitions'), document.get('warmup')
    _need(type(repetitions) is int and 3 <= repetitions <= 10000
          and type(warmup) is int and 1 <= warmup <= 10000,
          'bounded warm repeated workspace cut matrices required')
    samples = document.get('samples')
    _need(type(samples) is list and len(samples) == repetitions, 'all original workspace cut samples required')
    columns = [[] for _ in PHASE_DEFINITIONS]; matrices = [[] for _ in PHASE_DEFINITIONS]; partition = []
    for index, sample in enumerate(samples):
        cuts = sample.get('timestamps_ns')
        _need(type(sample.get('index')) is int and sample['index'] == index
              and type(sample.get('status')) is int and sample['status'] == 0
              and all(sample.get(key) is True for key in
                  ('caller_affinity_restored', 'caller_rounding_restored', 'input_arrays_unchanged')),
              'unique ordered successful samples with original input/context restoration required')
        _need(type(cuts) is list and len(cuts) == layout['timestamps_per_sample']
              and all(type(value) is int and value >= 0 for value in cuts)
              and all(a <= b for a, b in zip(cuts, cuts[1:])),
              'complete monotonic integer cuts required; negative or missing intervals rejected')
        adjacent = [b - a for a, b in zip(cuts, cuts[1:])]
        covered = set(); charged = 0
        for ordinal, phase in enumerate(PHASE_DEFINITIONS):
            values = []
            for iteration in range(count if phase['scope'] == 'LOOP' else 1):
                start, end = _indices(phase, count, iteration)
                indices = set(range(start, end))
                _need(not covered.intersection(indices), 'boundary phase cut intervals overlap')
                covered.update(indices)
                value = cuts[end] - cuts[start]; values.append(value); charged += value
            columns[ordinal].extend(values)
            if phase['scope'] == 'LOOP':
                matrices[ordinal].append(values)
        omitted = [dict(start_cut=i, end_cut=i + 1, duration_ns=value)
                   for i, value in enumerate(adjacent) if i not in covered]
        residual = sum(row['duration_ns'] for row in omitted)
        total = cuts[-1] - cuts[0]
        _need(sum(adjacent) == total == charged + residual,
              'boundary phases plus retained gaps must exactly partition each original trace')
        partition.append(dict(index=index, adjacent_intervals_ns=adjacent, outer_duration_ns=total,
            covered_boundary_ns=charged, unmapped_gap_ns=residual, unmapped_intervals=omitted))
    phases = []
    for ordinal, (phase, values) in enumerate(zip(PHASE_DEFINITIONS, columns)):
        record = dict(phase=phase['id'], scope=phase['scope'], samples_ns=values,
            median_ns=statistics.median(values), min_ns=min(values), max_ns=max(values),
            stdev_ns=statistics.stdev(values))
        if phase['scope'] == 'LOOP':
            record['per_iteration_samples_matrix'] = matrices[ordinal]
        phases.append(record)
    result = dict(phases=phases, trace_partition=dict(samples=partition,
        samplewise_partition_exact=True, phase_median_sum_is_trace_duration=False),
        loop_statistics_basis='CANONICAL_REPETITION_BY_ITERATION_SAMPLES_NOT_SEQUENCE_TOTAL',
        sample_ranges_are_guaranteed_bounds=False)
    if 'statistics' in document:
        _need(document['statistics'] == result, 'workspace boundary statistics differ from retained original cuts')
    return result


def _context(document, accounting, schedule, resources):
    certificate = schedule.proof['workspace_time']
    _need(document.get('source_time') == accounting['workspace_time']
          and document['source_time'].get('schema') == 'scar.workspace-demand-time-preparation'
          and document.get('workspace_plan') == accounting['workspace_plan']
          and document.get('native_manifest') == certificate['native_manifest']
          and document.get('cpp_source') == certificate['cpp_source']
          and document.get('trip_count') == accounting['trip_count'],
          'workspace boundary source/time/plan/native/C++/iteration context differs')
    proofs = [row for row in schedule.temporal.source_refs
              if row.get('schema') == 'scar.workspace-cpu-demand-source-proof']
    _need(len(proofs) == 1, 'one original source-qualified workspace CPU proof required')
    proof = proofs[0]; contract = proof['source_contract']; context = document['resource_context']
    _need(set(context) == {'target', 'selected_cpu_ids', 'original_caller_cpu_ids', 'step_plan',
                          'step_plan_sha256', 'input_shapes', 'base_implementation_sha256', 'compiler'},
          'exact supported workspace boundary resource context fields required')
    materials = {row.id: row for row in schedule.materializations}
    inputs = {materials[name].value for task in schedule.tasks for name in task.reads
              if materials[name].external_ready}
    shapes = {row.attrs['input_id']: row.attrs['shape'] for row in schedule.values
              if row.id in inputs and 'input_id' in row.attrs}
    _need(context.get('target') == encode(resources.target)
          and context.get('selected_cpu_ids') == contract['cpu_ids']
          and len(contract['cpu_ids']) == 1 and context.get('step_plan') == proof['stage_plan']
          and context.get('step_plan_sha256') == contract['step_plan_sha256']
          and context.get('input_shapes') == shapes
          and context.get('base_implementation_sha256') == contract['base_implementation_sha256']
          and context.get('compiler') == document['probe']['compiler'],
          'boundary component shape/layout/backend/compiler/selected CPU context differs')
    caller = context.get('original_caller_cpu_ids')
    _need(type(caller) is list and caller and len(set(caller)) == len(caller)
          and all(type(cpu) is int and cpu >= 0 for cpu in caller),
          'retained canonical caller affinity context required; it is not target placement proof')


def apply_workspace_boundary_costs(table, accounting, schedule, resources, evidence, *, source_refs=()):
    """Replace six complete UNKNOWN groups after source/probe cold admission."""
    if not evidence:
        return table, accounting
    summary = validate_activity_cost_table(table, schedule, resources)
    _need(accounting.get('schema') == 'scar.workspace-cost-accounting'
          and accounting.get('schema_version') == 1 and accounting.get('summary') == summary
          and accounting.get('source_schedule_sha256') == table.source_schedule_sha256
          and accounting.get('resource_projection_sha256') == table.resource_projection_sha256,
          'original workspace cost table/accounting identity required')
    _need((table.tick_numerator, table.tick_denominator) == (1, 10**9), 'nanosecond component table required')
    _need(type(evidence) in (tuple, list) and type(source_refs) is tuple,
          'source-qualified boundary publication list and immutable references required')
    by_task = {a.task: a for a in schedule.temporal.activities}
    _need(len(by_task) == len(schedule.temporal.activities), 'one original source lifecycle per task required')
    scopes = {row.id: row for row in schedule.temporal.scopes}
    costs = list(table.costs); ledger = deepcopy(accounting['subjects']); mappings = []; facts = []; residuals = []
    current = {row.subject: row for row in costs}
    _need(len(ledger) == len(current) and {row['subject'] for row in ledger} == set(current)
          and all(tuple(row['activities']) == current[row['subject']].activities
                  and row['estimate_per_occurrence_ticks'] == current[row['subject']].estimate_ticks
                  for row in ledger), 'workspace ledger must preserve every original cost subject and estimate')
    seen = set(); covered = set()
    for item in evidence:
        reference, document = item['reference'], item['document']
        artifact = ProviderArtifact(**reference)
        _need(artifact.schema == SCHEMA and semantic_sha256(reference) not in seen,
              'unique original workspace boundary calibration publication required')
        seen.add(semantic_sha256(reference))
        statistics_document = boundary_statistics(document)
        _need('statistics' in document, 'published original boundary statistics required before cost mapping')
        _context(document, accounting, schedule, resources)
        conditions = _strings(document['conditions'], 'conditions'); missing = _strings(document['missing_terms'], 'residual terms')
        phase_mapping = []
        for ordinal, phase in enumerate(PHASE_DEFINITIONS):
            selected = [task for task in schedule.tasks if
                (task.kind.value == phase['task_kind'] if 'task_kind' in phase else
                 task.kind is K.IMPLEMENTATION_PHASE and task.attrs.get('phase_kind') == phase['phase_kind'])]
            _need(selected and ('task_kind' in phase or len(selected) == 1),
                  'one complete original workspace source boundary group required')
            positions = [schedule.tasks.index(task) for task in selected]
            _need(positions == list(range(positions[0], positions[0] + len(positions))),
                  'a whole pool boundary must retain contiguous original source tasks')
            members = tuple(by_task[task.id].id for task in selected)
            _need(not covered.intersection(members), 'boundary phase activity occurrence charged twice')
            covered.update(members)
            scope_ids = {by_task[task.id].scope for task in selected}
            _need(len(scope_ids) == 1 and scopes[next(iter(scope_ids))].kind.value == phase['scope'],
                  'boundary group cannot split a pool or mix once and loop scopes')
            old = [row for row in costs if set(members).intersection(row.activities)]
            _need(old and all(row.basis is CostBasis.UNKNOWN and set(row.activities) <= set(members) for row in old)
                  and {a for row in old for a in row.activities} == set(members),
                  'boundary evidence may replace only complete UNKNOWN subjects; known copy costs remain untouched')
            retired = {row.subject for row in old}
            prior = [row for row in ledger if row['subject'] in retired]
            expected_count = accounting['trip_count'] if phase['scope'] == 'LOOP' else 1
            _need(prior and all(row['occurrence_count'] == expected_count for row in prior),
                  'boundary ledger must retain original once/iteration occurrence counts')
            stats = statistics_document['phases'][ordinal]
            subject = 'cost:workspace-boundary:' + semantic_sha256(dict(source=reference,
                schedule=table.source_schedule_sha256, phase=phase['id'], activities=members))[:32]
            row = ActivityCost(subject, members, CostBasis.DECLARED_COMPONENT_MODEL,
                estimate_ticks=math.ceil(stats['median_ns']), interval=CostInterval(
                    stats['min_ns'], stats['max_ns'], IntervalKind.EMPIRICAL_PREDICTION),
                component_refs=(ProviderFactReference(artifact, '#/statistics/phases/' + str(ordinal)),
                                ProviderFactReference(artifact, '#/samples')),
                charge_keys=('workspace-boundary-occurrence:' + semantic_sha256(dict(
                    schedule=table.source_schedule_sha256, phase=phase['id'], activities=members)),),
                coverage=CostCoverage.COMPONENT_ONLY,
                conditions=(*conditions, 'ITERATION_SAMPLES_PREDICT_ONE_COMPONENT_OCCURRENCE_NOT_SEQUENCE_TOTAL'),
                missing_terms=tuple(dict.fromkeys((*missing,
                    'ITERATION_INDEX_CACHE_PROVIDER_ALLOCATOR_AND_INSTRUMENTATION_DELTA_NOT_VALIDATED'))))
            position = min(costs.index(record) for record in old)
            costs = [record for record in costs if record.subject not in retired]; costs.insert(position, row)
            replacement = dict(prior[0], subject=subject, activities=list(members),
                task_ids=[task.id for task in selected], estimate_per_occurrence_ticks=row.estimate_ticks)
            ledger_position = min(ledger.index(record) for record in prior)
            ledger = [record for record in ledger if record['subject'] not in retired]
            ledger.insert(ledger_position, replacement)
            phase_mapping.append(dict(phase=phase['id'], subject=subject, activities=list(members),
                task_ids=[task.id for task in selected], occurrence_count=expected_count,
                replaced_unknown_subjects=sorted(retired), statistics_locator='#/statistics/phases/' + str(ordinal)))
        mappings.append(dict(source=reference, phases=phase_mapping,
            retained_raw_iteration_matrix=True, per_iteration_residuals_distinct=True,
            source_qualified_component_predictions_not_target_observations=True))
        facts.append(dict(artifact=reference, locator='#/statistics'))
        residuals.append(dict(kind='CANONICAL_WORKSPACE_BOUNDARY_TRACE_GAPS_AND_TARGET_DELTA',
            source=reference, missing_terms=list(missing), original_cut_gaps='#/statistics/trace_partition',
            physical_total_inferred=False, cost_charged=False))
    result = replace(table, costs=tuple(costs), source_refs=(*table.source_refs, *source_refs, *facts),
        unmapped_terms=(*table.unmapped_terms, *residuals))
    updated = deepcopy(accounting); updated['subjects'] = ledger
    updated['known_component_model_subtotal_ticks'] = sum(
        row['occurrence_count'] * row['estimate_per_occurrence_ticks']
        for row in ledger if row['estimate_per_occurrence_ticks'] is not None)
    updated['summary'] = validate_activity_cost_table(result, schedule, resources)
    updated['boundary_components'] = dict(schema='scar.workspace-boundary-cost-derivation', schema_version=1,
        mappings=mappings, source_refs=[*source_refs, *facts], residuals=residuals,
        existing_known_subjects_preserved=True, whole_trace_from_phase_medians=False,
        loop_costs_are_per_occurrence=True, physical_scope_latency=None,
        complete_call_cost=False, unknown_duration_imputed=False, deployment_authorized=False)
    return result, updated


__all__ = ['SCHEMA', 'PREFIX', 'ITERATION', 'SUFFIX', 'PHASE_DEFINITIONS', 'cut_layout',
           'boundary_statistics', 'apply_workspace_boundary_costs']

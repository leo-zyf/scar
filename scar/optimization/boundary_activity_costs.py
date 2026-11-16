"""Exclusive canonical boundary cuts mapped to original implementation phases.

The shared call is a canonical no-target-DAG experiment. Its phase samples are
conditional component models; neither their sum nor their sample range proves
actual target call latency. Existing numeric primitive subjects remain intact.
"""
from dataclasses import replace
import math
import statistics

from scar.ir.activity_costs import (ActivityCost, CostBasis, CostCoverage,
    CostInterval, IntervalKind, validate_activity_cost_table)
from scar.ir.execution_recipe import artifact_reference
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.versioned_schedule import VersionedTaskKind as K
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.stateful-boundary-cut-calibration'
CUTS = ('outer_enter', 'native_enter', 'metadata_ready', 'capture_ready',
        'binding_ready', 'worker_checked', 'restore_ready', 'outer_return')


def _strings(value, name):
    if (type(value) not in (tuple, list) or not value
            or any(type(item) is not str or not item for item in value)
            or len(set(value)) != len(value)):
        raise ValueError('unique nonempty boundary ' + name + ' required')
    return tuple(value)


def project_boundary_activity_costs(table, schedule, resources, *, domain, candidate_id,
                                    calibrations=()):
    """Replace complete UNKNOWN subjects by independently replayed cut models."""
    from scar.compiler.calibrate_boundary_cuts import read_boundary_cuts
    from scar.optimization.stateful_cpu_model import stateful_resource_context
    validate_activity_cost_table(table, schedule, resources)
    if table.tick_numerator != 1 or table.tick_denominator != 10**9:
        raise ValueError('boundary component model requires original nanosecond table ticks')
    if 'implementation_envelope' not in schedule.proof:
        raise ValueError('boundary cuts require the source-replayed standalone implementation envelope')
    candidate = next(row for row in domain['candidates'] if row['id'] == candidate_id)
    if candidate['worker_count'] != 1:
        raise ValueError('boundary cut contract currently requires the original serial candidate')
    by_task = {}
    for activity in schedule.temporal.activities:
        if activity.task in by_task:
            raise ValueError('boundary cut adapter needs one synchronous lifecycle per original task')
        by_task[activity.task] = activity.id
    costs = list(table.costs); mappings = []; facts = []; residuals = []; seen = set()
    for value in calibrations:
        reference = value if type(value) is dict else artifact_reference(value, SCHEMA)
        identity = semantic_sha256(reference)
        if identity in seen:
            raise ValueError('one boundary calibration publication cannot be charged twice')
        seen.add(identity)
        document = read_boundary_cuts(reference, domain=domain, candidate_id=candidate_id)
        if (document.get('schema'), document.get('schema_version')) != (SCHEMA, 1):
            raise ValueError('source-replayed boundary cut calibration required')
        if document['resource_context'] != stateful_resource_context(domain, candidate_id):
            raise ValueError('boundary cuts differ from the original candidate resource context')
        if tuple(document['cuts']) != CUTS:
            raise ValueError('original shared monotonic boundary cut order differs')
        conditions = _strings(document['conditions'], 'applicability conditions')
        missing = _strings(document['missing_terms'], 'residual terms')
        samples = document['samples']; sample_ids = set()
        if type(samples) is not list or len(samples) < 3:
            raise ValueError('retained repeated boundary cut vectors required')
        for sample in samples:
            sample_id, timestamps = sample['id'], sample['timestamps_ns']
            if type(sample_id) is not str or not sample_id or sample_id in sample_ids:
                raise ValueError('unique shared boundary sample identity required')
            sample_ids.add(sample_id)
            if (type(timestamps) is not list or len(timestamps) != len(CUTS)
                    or any(type(t) is not int or t < 0 for t in timestamps)
                    or any(a > b for a, b in zip(timestamps, timestamps[1:]))):
                raise ValueError('one original monotonic integer cut vector required per sample')
        phases = document['phases']
        if type(phases) is not list or len(phases) != len(CUTS) - 1:
            raise ValueError('one exclusive phase per adjacent original cut pair required')
        phase_ids = set(); covered_kinds = set(); phase_rows = []
        for index, phase in enumerate(phases):
            phase_id = phase['id']
            if type(phase_id) is not str or not phase_id or phase_id in phase_ids:
                raise ValueError('unique original boundary phase identity required')
            phase_ids.add(phase_id)
            if (phase['start'], phase['end']) != CUTS[index:index+2]:
                raise ValueError('boundary samples must retain contiguous exclusive cut intervals')
            kinds = _strings(phase['phase_kinds'], 'implementation phase kinds')
            if covered_kinds.intersection(kinds):
                raise ValueError('one implementation phase cannot be charged by two cut intervals')
            covered_kinds.update(kinds)
            selected = [task for task in schedule.tasks if task.kind is K.IMPLEMENTATION_PHASE
                        and task.attrs.get('phase_kind') in kinds]
            if tuple(task.attrs['phase_kind'] for task in selected) != kinds:
                raise ValueError('boundary kinds must match unique original implementation phases in order')
            positions = [schedule.tasks.index(task) for task in selected]
            if positions != list(range(positions[0], positions[0] + len(positions))):
                raise ValueError('one boundary subject must preserve a continuous original phase group')
            activities = tuple(by_task[task.id] for task in selected); activity_set = set(activities)
            old = [row for row in costs if activity_set.intersection(row.activities)]
            if (any(row.basis is not CostBasis.UNKNOWN or not set(row.activities) <= activity_set for row in old)
                    or set(a for row in old for a in row.activities) != activity_set):
                raise ValueError('boundary evidence may only replace complete UNKNOWN subjects; primitive facts remain charged once')
            values = [sample['timestamps_ns'][index+1] - sample['timestamps_ns'][index] for sample in samples]
            stats = dict(median_ns=statistics.median(values), min_ns=min(values), max_ns=max(values),
                         stdev_ns=statistics.stdev(values), repetitions=len(values))
            subject = 'cost:boundary:' + semantic_sha256(dict(source=reference, phase=phase_id,
                schedule=table.source_schedule_sha256, activities=activities))[:32]
            charge = 'boundary-occurrence:' + semantic_sha256(dict(source=reference, phase=phase_id, activities=activities))
            artifact = ProviderArtifact(**reference)
            row = ActivityCost(subject, activities, CostBasis.DECLARED_COMPONENT_MODEL,
                estimate_ticks=math.ceil(stats['median_ns']),
                interval=CostInterval(stats['min_ns'], stats['max_ns'], IntervalKind.EMPIRICAL_PREDICTION),
                component_refs=(ProviderFactReference(artifact, '#/phases/'+str(index)),
                                ProviderFactReference(artifact, '#/samples')),
                charge_keys=(charge,), coverage=CostCoverage.COMPONENT_ONLY,
                conditions=conditions, missing_terms=missing)
            replaced = {record.subject for record in old}
            costs = [record for record in costs if record.subject not in replaced]
            costs.append(row)
            phase_rows.append(dict(subject=subject, phase=phase_id, activities=activities,
                task_ids=[task.id for task in selected], statistics=stats,
                replaced_unknown_subjects=sorted(replaced)))
        mappings.append(dict(source=reference, shared_sample_ids=[sample['id'] for sample in samples],
            phases=phase_rows, exclusive_adjacent_intervals=True, whole_call_charged=False,
            target_activity_observed=False, samplewise_cut_differences=True,
            median_subtraction_used=False, residual_terms=missing))
        facts.append(dict(artifact=reference, locator='#/phases'))
        residuals.append(dict(source=reference, kind='CANONICAL_BOUNDARY_MODEL_RESIDUAL',
            missing_terms=list(missing), cost_charged=False))
    result = replace(table, costs=tuple(costs), source_refs=(*table.source_refs, *facts),
                     unmapped_terms=(*table.unmapped_terms, *residuals))
    summary = validate_activity_cost_table(result, schedule, resources)
    return result, dict(protocol='SOURCE_REPLAYED_EXCLUSIVE_CANONICAL_BOUNDARY_CUTS',
        mapping=mappings, summary=summary, existing_known_subjects_preserved=True,
        complete_call_cost=False, native_execution_authorized=False)


__all__ = ['project_boundary_activity_costs']

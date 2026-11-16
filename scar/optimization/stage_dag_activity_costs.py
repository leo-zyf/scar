"""Conditional stage-DAG samples for exact original activity occurrences.

The independent reader replays the source, emitted stage and measurement
context. This adapter only replaces complete UNKNOWN subjects; canonical
stage samples never become observations of the target's activity or call.
"""
from dataclasses import replace
import math
import statistics

from scar.ir.activity_costs import (ActivityCost, CostBasis, CostCoverage,
    CostInterval, IntervalKind, validate_activity_cost_table)
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import encode
from scar.ir.versioned_schedule import VersionedTaskKind as K
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.stateful-stage-dag-calibration'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _strings(value, name):
    _need(type(value) is list and value and all(type(row) is str and row for row in value)
          and len(value) == len(set(value)), 'unique nonempty stage-DAG ' + name + ' required')
    return tuple(value)


def project_stage_dag_activity_costs(table, schedule, resources, *, domain, candidate_id,
                                    calibrations=()):
    """Bind reader-qualified full stage signatures to one original CPU task."""
    from scar.compiler.calibrate_stage_dag import read_stage_dag_calibration
    from scar.optimization.stateful_cpu_model import stateful_resource_context
    validate_activity_cost_table(table, schedule, resources)
    _need((table.tick_numerator, table.tick_denominator) == (1, 10**9),
          'stage-DAG component models require original nanosecond table ticks')
    candidate = next((row for row in domain['candidates'] if row['id'] == candidate_id), None)
    _need(candidate is not None and candidate['worker_count'] == 1 and candidate['legal_under_guards'],
          'stage-DAG component model requires an admitted serial candidate')
    _need(encode(resources.target) == domain['target_snapshot'],
          'stage-DAG activities and original candidate target differ')
    plan = read_artifact(domain['base_stage_plan'])
    by_task = {}
    for activity in schedule.temporal.activities:
        _need(activity.task not in by_task,
              'stage-DAG adapter needs one synchronous lifecycle per original task')
        by_task[activity.task] = activity.id
    costs = list(table.costs); seen_publications = set(); seen_stages = set()
    mappings = []; source_refs = []; residuals = []
    for value in calibrations:
        reference = value if type(value) is dict else artifact_reference(value, SCHEMA)
        publication = semantic_sha256(reference)
        _need(publication not in seen_publications,
              'one stage-DAG calibration publication cannot be charged twice')
        seen_publications.add(publication)
        document = read_stage_dag_calibration(reference, domain=domain, candidate_id=candidate_id)
        _need((document.get('schema'), document.get('schema_version')) == (SCHEMA, 1),
              'source-replayed stage-DAG calibration required')
        _need(document.get('resource_context') == stateful_resource_context(domain, candidate_id),
              'stage-DAG calibration differs from original candidate resource context')
        _need(document.get('component_topology_specialized') is True
              and document.get('target_fragment_compiled') is True
              and document.get('full_target_program_executed') is False
              and document.get('whole_target_call_measured') is False,
              'stage-DAG samples must preserve the specialized-fragment-only measurement scope')
        conditions = _strings(document.get('conditions'), 'applicability conditions')
        missing = _strings(document.get('missing_terms'), 'residual terms')
        components = document.get('components')
        _need(type(components) is list and components, 'retained stage-DAG components required')
        repetitions, warmup = document.get('repetitions'), document.get('warmup')
        _need(type(repetitions) is int and 3 <= repetitions <= 10000
              and type(warmup) is int and 1 <= warmup <= 10000,
              'warm repeated stage-DAG samples required')
        rows = []
        for component_index, component in enumerate(components):
            stage_index = component.get('stage_index')
            _need(type(stage_index) is int and 0 <= stage_index < len(plan['stages']),
                  'original stage-DAG index required')
            _need(stage_index not in seen_stages, 'one original stage cannot have duplicate stage-DAG publications')
            seen_stages.add(stage_index)
            stage = plan['stages'][stage_index]
            pointer = '#/stages/' + str(stage_index)
            stage_ref = dict(artifact=domain['base_stage_plan'], locator=pointer)
            _need(stage['index'] == stage_index and stage['kind'] in ('COMPUTE', 'SCALAR')
                  and component.get('stage_ref') == stage_ref,
                  'stage-DAG samples must retain the exact original COMPUTE/SCALAR stage')
            signature = component.get('signature')
            _need(type(signature) is dict and signature,
                  'reader-replayed complete stage-DAG signature required')
            selected = [task for task in schedule.tasks
                        if task.attrs.get('source_stage_ref', {}).get('pointer') == pointer]
            _need(len(selected) == 1, 'stage-DAG sample must match exactly one source-stage task')
            task = selected[0]
            _need(task.kind is K.COMPUTE and task.id == stage['id']
                  and task.attrs.get('source_stage_ref') == dict(artifact_role='stage_plan', pointer=pointer)
                  and task.attrs.get('stage_kind') == stage['kind']
                  and task.attrs.get('original_root_expression') == stage['root_expr']
                  and task.attrs.get('shape') == stage['shape'],
                  'stage-DAG task differs from original source-stage kind/root/shape/identity')
            _need(task.id in by_task, 'original stage task has no activity lifecycle')
            activities = (by_task[task.id],); activity_set = set(activities)
            old = [row for row in costs if activity_set.intersection(row.activities)]
            _need(old and all(row.basis is CostBasis.UNKNOWN and set(row.activities) <= activity_set for row in old)
                  and {key for row in old for key in row.activities} == activity_set,
                  'stage-DAG evidence may only replace complete UNKNOWN subjects; known costs remain charged once')
            samples = component.get('samples_ns')
            _need(type(samples) is list and len(samples) == repetitions
                  and all(type(sample) is int and sample >= 0 for sample in samples),
                  'retained nonnegative integer stage-DAG samples required')
            stats = dict(median_ns=statistics.median(samples), min_ns=min(samples),
                         max_ns=max(samples), stdev_ns=statistics.stdev(samples))
            _need(component.get('statistics') == stats,
                  'stage-DAG statistics differ from original retained samples')
            occurrence = dict(domain=domain['domain_id'], candidate=candidate_id,
                task=task.id, activity=activities[0], source=reference, stage=stage_ref)
            subject = 'cost:stage-dag:' + semantic_sha256(dict(
                occurrence=occurrence, schedule=table.source_schedule_sha256))[:32]
            artifact = ProviderArtifact(**reference)
            row = ActivityCost(subject, activities, CostBasis.DECLARED_COMPONENT_MODEL,
                estimate_ticks=math.ceil(stats['median_ns']),
                interval=CostInterval(stats['min_ns'], stats['max_ns'], IntervalKind.EMPIRICAL_PREDICTION),
                component_refs=(ProviderFactReference(artifact, '#/components/' + str(component_index)),
                    ProviderFactReference(ProviderArtifact(**domain['base_stage_plan']), pointer)),
                charge_keys=('stage-dag-occurrence:' + semantic_sha256(occurrence),),
                coverage=CostCoverage.COMPONENT_ONLY, conditions=conditions, missing_terms=missing)
            replaced = {record.subject for record in old}
            costs = [record for record in costs if record.subject not in replaced]
            costs.append(row)
            rows.append(dict(subject=subject, activities=activities, task_ids=[task.id],
                stage_index=stage_index, stage_ref=stage_ref, signature=signature,
                statistics=stats, replaced_unknown_subjects=sorted(replaced)))
        mappings.append(dict(source=reference, stages=rows,
            full_source_stage_signatures_replayed_by_reader=True,
            component_topology_specialized=True, target_fragment_compiled=True,
            full_target_program_executed=False,
            target_activity_observed=False, whole_call_charged=False,
            residual_terms=missing))
        source_refs.append(dict(artifact=reference, locator='#/components'))
        residuals.append(dict(source=reference, kind='CANONICAL_STAGE_DAG_MODEL_RESIDUAL',
            missing_terms=list(missing), cost_charged=False))
    result = replace(table, costs=tuple(costs), source_refs=(*table.source_refs, *source_refs),
                     unmapped_terms=(*table.unmapped_terms, *residuals))
    summary = validate_activity_cost_table(result, schedule, resources)
    return result, dict(protocol='SOURCE_REPLAYED_SERIAL_STAGE_DAG_SAMPLES', mapping=mappings,
        summary=summary, existing_known_subjects_preserved=True,
        primitive_unresolved_terms_are_not_final_activity_coverage=True,
        complete_call_cost=False, sample_ranges_are_guaranteed_bounds=False,
        native_execution_authorized=False)


__all__ = ['project_stage_dag_activity_costs']

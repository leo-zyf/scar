"""Source-backed work ownership and nonadditive control milestone accounting.

A ledger relates old descriptive terms to final subjects. It neither adds the
old coarse estimates nor turns implementation/model residuals into new work.
"""
from dataclasses import replace

from scar.ir.activity_costs import (ActivityCost, CostBasis, CostCoverage,
    validate_activity_cost_table)
from scar.ir.execution_recipe import read_artifact
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.versioned_schedule import VersionedTaskKind as K, schedule_document, schedule_from_dict
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.stateful-scope-accounting'
_COARSE = {
    'GUARDED_METHOD_AND_NATIVE_ARTIFACT_BOUNDARY': ('PYTHON_ENTRY_GUARD_PACKING',
        'NATIVE_METADATA_GUARD', 'NATIVE_RETURN', 'PYTHON_STATUS_RETURN'),
    'CALLER_WORKER_AFFINITY_FENV_TEAM': ('CALLER_CAPTURE_SCALAR_GUARD', 'TEAM_ENTER'),
    'SUCCESS_CHECK_AND_CALLER_RESTORATION': ('WORKER_EXIT', 'CALLER_RESTORE', 'NATIVE_HELPER_CLEANUP'),
}
_CONDITIONS = ('SOURCE_REPLAYED_STANDALONE_SYNCHRONOUS_NORMAL_SUCCESS_PROFILE',
    'PRECOMMIT_CHECK_WORK_OWNED_BY_ORIGINAL_CALLER_RESTORE',
    'NO_ADDITIONAL_NATIVE_FENCE_STATEMENT_OR_SERVICE',
    'CONTROL_MILESTONE_IS_NOT_A_MEASUREMENT_OR_PHYSICAL_ZERO_LATENCY_BOUND')


def project_scope_accounting(table, schedule, resources, recipe, demand, *, mapping,
                             boundary_derivation=None, dag_derivation=None, scope_schedule_ref=None):
    """Replay the exact modern source graph, then qualify its semantic fence."""
    from scar.analysis.implementation_scope import derive_stateful_scope_envelope
    validate_activity_cost_table(table, schedule, resources)
    replay = derive_stateful_scope_envelope(schedule_from_dict(recipe['task_graph']), recipe)
    if schedule_document(replace(schedule, temporal=None)) != schedule_document(replay):
        raise ValueError('scope accounting differs from complete original cleanup/source milestone replay')
    if (type(scope_schedule_ref) is not dict or scope_schedule_ref.get('schema') != 'scar.resource-schedule'
            or read_artifact(scope_schedule_ref) != schedule_document(schedule)):
        raise ValueError('original complete saved scope schedule and source certificate reference required')
    proof = replay.proof['implementation_scope_cleanup']; certificate = proof['milestone']
    source = proof['source']
    if (certificate['basis'] != 'SOURCE_PROVEN_ZERO_ADDITIONAL_SERVICE'
            or source.get('no_runtime_statement_between_restore_checks_and_commit') is not True
            or source.get('recipe_fence_is_semantic_gate_not_additional_emitted_work') is not True):
        raise ValueError('source replay has no nonadditive precommit control milestone')
    tasks = {row.id: row for row in schedule.tasks}
    fence = tasks[certificate['task_id']]; restore = tasks[certificate['restore_task_id']]
    if (fence.kind is not K.SUCCESS_FENCE or restore.attrs.get('phase_kind') != 'CALLER_RESTORE'
            or schedule.tasks.index(fence) != schedule.tasks.index(restore) + 1):
        raise ValueError('original restore/fence normal-success boundary differs')
    activities = [row for row in schedule.temporal.activities if row.task == fence.id]
    if len(activities) != 1:
        raise ValueError('one original semantic fence lifecycle required')
    activity = activities[0].id
    old = [row for row in table.costs if activity in row.activities]
    if (len(old) != 1 or old[0].basis is not CostBasis.UNKNOWN or old[0].activities != (activity,)):
        raise ValueError('source milestone may only replace its complete UNKNOWN subject')
    pointer = '#/proof/implementation_scope_cleanup/milestone'
    marker = ActivityCost('cost:source-milestone:' + semantic_sha256(dict(
        schedule=table.source_schedule_sha256, task=fence.id))[:32], (activity,),
        CostBasis.SOURCE_CONTROL_MILESTONE, estimate_ticks=0, coverage=CostCoverage.COMPLETE_SUBJECT,
        component_refs=(ProviderFactReference(ProviderArtifact(**scope_schedule_ref), pointer),),
        conditions=_CONDITIONS)
    result = replace(table, costs=tuple(marker if row is old[0] else row for row in table.costs),
        source_refs=(*table.source_refs, dict(artifact=scope_schedule_ref, locator=pointer)))
    summary = validate_activity_cost_table(result, schedule, resources)
    by_subject = {row.subject: row for row in result.costs}
    by_activity = {name: row.subject for row in result.costs for name in row.activities}
    activity_by_task = {row.task: row.id for row in schedule.temporal.activities}
    replacements = {}
    for derivation, field in ((boundary_derivation, 'phases'), (dag_derivation, 'stages')):
        if derivation is None:
            continue
        for group in derivation['mapping']:
            for row in group[field]:
                for original in row['replaced_unknown_subjects']:
                    if original in replacements:
                        raise ValueError('one original subject has ambiguous source-model replacements')
                    replacements[original] = row['subject']
    def resolve(subject):
        chain = [subject]
        while subject not in by_subject and subject in replacements:
            subject = replacements[subject]
            if subject in chain:
                raise ValueError('source cost replacement graph contains a cycle')
            chain.append(subject)
        return subject if subject in by_subject else None, chain
    original_mapping = {row['source_term']: row for row in mapping}
    records = []
    for term in demand['terms']:
        row = dict(term=term['id'], component=term['component'], original_source=term['source_ref'],
            cost_charged_separately=False, legacy_numeric_estimate_added=False)
        if term['component'] in _COARSE:
            kinds = _COARSE[term['component']]
            selected = [task for task in schedule.tasks if task.kind is K.IMPLEMENTATION_PHASE
                        and task.attrs.get('phase_kind') in kinds]
            if {task.attrs['phase_kind'] for task in selected} != set(kinds):
                row.update(disposition='SOURCE_FRONTIER', reason='ORIGINAL_COARSE_SCOPE_HAS_NO_COMPLETE_SOURCE_PHASE_OWNERS')
            else:
                row.update(disposition='SUPERSEDED_COARSE_DESCRIPTION', phase_kinds=list(kinds),
                    task_ids=[task.id for task in selected],
                    active_subjects=list(dict.fromkeys(by_activity[activity_by_task[task.id]] for task in selected)),
                    relation='SOURCE_WORK_DESCRIPTION_REPLACED_NOT_NUMERIC_SCOPE_EQUALITY')
        elif term['id'] in original_mapping:
            original = original_mapping[term['id']]
            subject, chain = resolve(original['subject'])
            if subject is None:
                row.update(disposition='SOURCE_FRONTIER', reason='ORIGINAL_COMPONENT_HAS_NO_FINAL_SUBJECT', replacement_chain=chain)
            else:
                row.update(disposition='ADDITIVE_COMPONENT', active_subjects=[subject],
                    replacement_chain=chain, activities=list(by_subject[subject].activities),
                    legacy_component_signature_has_canonical_sample=(len(chain) == 1 and by_subject[subject].basis is CostBasis.DECLARED_COMPONENT_MODEL))
        else:
            row.update(disposition='SOURCE_FRONTIER', reason='ORIGINAL_TERM_HAS_NO_SOURCE_COMPONENT_OR_PHASE_MAPPING')
        records.append(row)
    residuals = [dict(source=row.get('source'), kind=row['kind'], missing_terms=row['missing_terms'],
                      additional_work=False, cost_charged=False)
                 for row in result.unmapped_terms if 'missing_terms' in row and 'kind' in row]
    subject_residuals = {}
    for row in result.costs:
        for term in row.missing_terms:
            subject_residuals.setdefault(term, []).append(row.subject)
    ledger = dict(schema=SCHEMA, schema_version=1, scope_schedule=scope_schedule_ref,
        source_certificate=dict(artifact=scope_schedule_ref, locator='#/proof/implementation_scope_cleanup'),
        milestone=dict(subject=marker.subject, activity=activity, task_id=fence.id,
            restore_task_id=restore.id, additional_service=0, measured_zero=False, physical_zero_latency_claimed=False),
        terms=records, active_subjects=list(by_subject), residuals=residuals,
        subject_residuals=[dict(term=term, subjects=subjects, additional_work=False, cost_charged=False)
                           for term, subjects in subject_residuals.items()],
        numeric_unknown_subjects=[row.subject for row in result.costs if row.basis is CostBasis.UNKNOWN],
        source_frontiers=[row['term'] for row in records if row['disposition'] == 'SOURCE_FRONTIER'],
        summary=summary, original_demand_terms_accounted=len(records) == len(demand['terms']),
        original_activity_ownership_exactly_once=True, coarse_terms_are_nonadditive_descriptions=True,
        coarse_scope_numeric_equality_claimed=False, residuals_are_additional_work=False,
        complete_target_implementation_cost=False, physical_cost_closed=False,
        physical_optimality_proven=False, solver_authorized=False, execution_authorized=False)
    read_artifact(scope_schedule_ref)
    return result, ledger


__all__ = ['SCHEMA', 'project_scope_accounting']

"""Expose source-proven helper cleanup in the existing implementation graph."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from scar.analysis.implementation_envelope import derive_stateful_envelope
from scar.codegen.scope_boundary_probe import stateful_scope_source_evidence
from scar.ir.execution_recipe import artifact_reference
from scar.ir.versioned_schedule import VersionedTask as Task, VersionedTaskKind as K, validate_versioned_schedule


def derive_stateful_scope_envelope(schedule, recipe):
    """Return the untimed normal-success graph, never a new executable recipe."""
    refined = derive_stateful_envelope(schedule, recipe)
    evidence = stateful_scope_source_evidence()
    rows = refined.tasks
    fences = [row for row in rows if row.kind is K.SUCCESS_FENCE]
    restores = [row for row in rows if row.kind is K.IMPLEMENTATION_PHASE
                and row.attrs.get('phase_kind') == 'CALLER_RESTORE']
    commits = [row for row in rows if row.kind is K.COMMIT]
    releases = [row for row in rows if row.kind is K.RELEASE]
    if len(fences) != 1 or len(restores) != 1 or not commits or not releases:
        raise ValueError('one original precommit fence/restore and explicit commit/release phases required')
    release_order = recipe['control'].get('pool_release_order')
    physical_order = 'CPP_STD_ARRAY_REVERSE_DESTRUCTION'
    if ('pool_release_order' in recipe['control'] and release_order != physical_order
            or len(releases) > 1 and release_order != physical_order):
        raise ValueError('modern scope requires the original C++ reverse physical pool release order')
    if release_order == physical_order:
        expected = list(reversed([slot.id for slot in refined.storage_slots if slot.ownership == 'SCHEDULE']))
        actual = [slot for task in releases for slot in task.releases]
        if (any(len(task.releases) != 1 for task in releases) or actual != expected
                or refined.proof.get('pool_release_order') != physical_order
                or refined.proof.get('physical_release_slot_order') != expected):
            raise ValueError('source pool declaration order and reverse physical release graph differ')
    fence, restore = fences[0], restores[0]
    if (rows.index(restore)+1 != rows.index(fence)
            or rows.index(fence)+1 != rows.index(commits[0])
            or rows.index(commits[-1])+1 != rows.index(releases[0])):
        raise ValueError('original restore/fence/commit/release normal-success source order changed')
    identity = 'implementation:native_helper_cleanup'
    if any(row.id == identity for row in rows):
        raise ValueError('helper cleanup phase already exists')
    path = Path(__file__).resolve()
    reference = artifact_reference(path, 'scar.backend-implementation-source')
    cleanup = Task(identity, K.IMPLEMENTATION_PHASE, depends_on=tuple(row.id for row in commits),
        capability='CPU_IMPLEMENTATION_PHASE', source_refs=(dict(artifact=reference), *restore.source_refs),
        attrs=dict(phase_kind='NATIVE_HELPER_CLEANUP', actor='CALLER_UNBOUND',
            failure_domain='POSTCOMMIT', may_fail=False, blocks='UNKNOWN',
            phase_effects=['DESTROY_SUCCESSFULLY_RESTORED_FLOATING_SCOPE',
                'DESTROY_SUCCESSFULLY_RESTORED_THREAD_AFFINITY', 'FREE_CALLER_AFFINITY_METADATA_MASKS'],
            normal_success_only=True, pool_release_excluded=True))
    output = []
    for row in rows:
        if row.id == releases[0].id:
            output.append(cleanup)
            row = replace(row, depends_on=tuple(dict.fromkeys((*row.depends_on, identity))))
        output.append(row)
    proof = deepcopy(refined.proof)
    envelope = proof['implementation_envelope']
    envelope['phase_order'] = [row.id for row in output]
    envelope['inserted_phase_ids'].append(identity)
    proof['implementation_scope_cleanup'] = dict(
        schema='scar.stateful-implementation-scope-proof', schema_version=1,
        cleanup_task_id=identity, source=evidence, source_adapter=reference,
        reverse_raii_order=evidence['reverse_raii_order'],
        pool_release_order=release_order,
        collective_release_internal_order_unexpanded=release_order is None,
        milestone=dict(task_id=fence.id, restore_task_id=restore.id,
            basis='SOURCE_PROVEN_ZERO_ADDITIONAL_SERVICE',
            restore_checks_executed_before_commit=True,
            following_precommit_text_is_comments_only=True,
            fence_is_original_recipe_semantic_gate=True,
            no_additional_emitted_fence_statement=True,
            measurement_claimed=False),
        all_original_tasks_data_storage_and_happens_before_preserved=True,
        helper_cleanup_after_all_commits_before_first_pool_release=True,
        normal_completion_only=True, costs_inferred=False, native_lowering_authorized=False)
    result = replace(refined, tasks=tuple(output), proof=proof, temporal=None, executable=False)
    validate_versioned_schedule(result)
    return result


__all__ = ['derive_stateful_scope_envelope']

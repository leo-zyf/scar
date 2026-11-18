"""Elide entry snapshots inside one guarded transactional stateful step."""
import argparse
import json
from pathlib import Path

from scar.compiler.stateful_model import stateful_digest, validate_stateful_region
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.versioned_schedule import VersionedTaskKind as K, validate_versioned_schedule
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.stateful_schedule import _normalized
from scar.optimization.workspace_guard_properties import _initialization


SCHEMA = 'scar.workspace-step-snapshot-elision-proof'
_GUARDS = frozenset(('EXACT_NDARRAY_FLOAT64_CPU_RANK2', 'C_CONTIGUOUS_NATIVE_FLOAT64_ARRAY_ROOTS',
    'DISTINCT_ARRAY_INPUTS_HAVE_DISJOINT_STORAGE', 'NO_FOREIGN_CONCURRENT_INPUT_WRITERS',
    'SYNCHRONOUS_CPU_EXECUTION', 'ORDERED_TRANSACTIONAL_SHADOW_WRITES_BEFORE_COMMIT',
    'ALL_SHAPES_ALIAS_BOUNDS_AND_WORKSPACE_CHECKED_BEFORE_EXTERNAL_WRITE'))
_STAGES = frozenset(('SNAPSHOT_ROOT', 'SCALAR', 'COMPUTE', 'PROVIDER', 'WRITE_SHADOW', 'COMMIT'))


def _need(condition, detail):
    if not condition:
        raise ValueError(detail)


def _borrowed_read(stage, old, new, rectangle, expressions=()):
    return dict(stage_id=stage['id'], stage_index=stage['index'], kind=stage['kind'],
        expressions=list(expressions), old_materialization=old, new_materialization=new,
        rectangle=rectangle, footprint=('EXACT_WRITE_OPERAND_RECTANGLE' if stage['kind'] == 'WRITE_SHADOW'
            else 'CONSERVATIVE_ROOT_BOUND'))


def _root(target, array_index, graph, stage_plan, actions, nodes, reasons):
    stages = stage_plan['stages']
    by_stage = {stage['id']: stage for stage in stages}
    tasks = {task.id: task for task in graph.tasks}
    materials = {material.id: material for material in graph.materializations}
    slots = {slot.id: slot for slot in graph.storage_slots}
    shadow = stage_plan['shadows'][target]
    slot_id = stage_plan['assignment'][shadow]['slot_id']
    shape = stage_plan['input_shapes'][target]
    full_rectangle = [[0, size] for size in shape]
    size = shape[0] * shape[1] * 8
    snapshots = [stage for stage in stages if stage['kind'] == 'SNAPSHOT_ROOT' and stage['target'] == target]
    _need(len(snapshots) == 1, 'original root needs one exact snapshot stage')
    snapshot = snapshots[0]
    snap_task = tasks[snapshot['id']]
    _need(snap_task.kind is K.SNAPSHOT and len(snap_task.reads) == len(snap_task.writes) == 1
        and snapshot['shape'] == shape and snapshot['output_payload'] == shadow and size > 0,
        'original root/snapshot stage geometry differs')
    entry_id, shadow_id = snap_task.reads[0], snap_task.writes[0]
    entry, initial_shadow = materials[entry_id], materials[shadow_id]
    _need(entry.external_ready and not initial_shadow.external_ready and initial_shadow.ready_task == snapshot['id']
        and (entry.value, entry.version) == (initial_shadow.value, initial_shadow.version)
        and slots[entry.region.slot].ownership == 'EXTERNAL' and slots[slot_id].ownership == 'SCHEDULE'
        and entry.region.slot != slot_id and initial_shadow.region.slot == slot_id
        and entry.region.offset_bytes == initial_shadow.region.offset_bytes == 0
        and entry.region.nbytes == initial_shadow.region.nbytes == size,
        'original snapshot must materialize the same entry logical value into private storage')

    writes = [stage for stage in stages if stage['kind'] == 'WRITE_SHADOW' and stage['target'] == target]
    commits = [stage for stage in stages if stage['kind'] == 'COMMIT' and stage['target'] == target]
    row = dict(input_id=target, array_index=array_index, shape=shape, shadow_payload=shadow, shadow_slot=slot_id,
        decision='KEEP', snapshot_stage=snapshot['id'], snapshot_stage_index=snapshot['index'], snapshot_bytes=size,
        entry_materialization=entry_id, entry_shadow_materialization=shadow_id, borrowed_reads=[],
        write_stage=None, write_stage_index=None, write_op=None, write_rectangle=None, write_materialization=None,
        commit_stage=None, commit_stage_index=None, commit_rectangle=None, conditions=[], rejections=list(reasons))
    if len(writes) != 1 or len(commits) != 1:
        row['rejections'].append('ONE_SHADOW_WRITE_AND_ONE_COMMIT_REQUIRED')
        return row

    write, commit = writes[0], commits[0]
    write_task, commit_task = tasks[write['id']], tasks[commit['id']]
    bounds = write['normalized_bounds']
    action_ids = write['source_action_ids']
    action = actions.get(action_ids[0]) if len(action_ids) == 1 else None
    rhs = stage_plan['materialized_exprs'].get(write['root_expr'])
    if not (action is not None and action.target == target and action.op == write['op']
            and write['op'] in ('assign', 'add') and rhs is not None
            and action.value == write['root_expr']
            and commit['source_action_ids'] == action_ids
            and _normalized(action.bounds, shape)[0] == bounds == commit['normalized_bounds']
            and all(lo < hi for lo, hi in bounds) and snapshot['index'] < write['index'] < commit['index']
            and write_task.kind is K.STATE_WRITE and commit_task.kind is K.COMMIT
            and write['input_payloads'] == [rhs, shadow] and commit['input_payloads'] == [shadow]
            and len(write_task.writes) == 1 and len(commit_task.writes) == 1
            and len(write_task.reads) == 2 and shadow_id in write_task.reads):
        row['rejections'].append('ORIGINAL_ROI_TRANSACTION_OR_EXPLICIT_RHS_NOT_CLOSED')
        return row
    rhs_id, = (material for material in write_task.reads if material != shadow_id)
    written, final = materials[write_task.writes[0]], materials[commit_task.writes[0]]
    if not (materials[rhs_id].region.slot != slot_id and materials[rhs_id].ready_task in by_stage
            and by_stage[materials[rhs_id].ready_task]['index'] < write['index']
            and written.ready_task == write['id'] and written.region == initial_shadow.region
            and written.value != entry.value and written.version > initial_shadow.version
            and commit_task.reads == (entry_id, written.id) and final.region == entry.region
            and (final.value, final.version) == (written.value, written.version)
            and not final.external_ready and final.ready_task == commit['id']):
        row['rejections'].append('SHADOW_RHS_MATERIALIZATION_OR_COMMIT_ROUTE_NOT_CLOSED')
        return row

    reads = []
    for task in graph.tasks:
        affected_reads = tuple(key for key in task.reads if materials[key].region.slot == slot_id)
        affected_writes = tuple(key for key in task.writes if materials[key].region.slot == slot_id)
        if task.id == write['id']:
            if write['op'] == 'add':
                reads.append(_borrowed_read(write, shadow_id, entry_id, bounds))
            continue
        if task.id in (snapshot['id'], commit['id']):
            continue
        if affected_writes:
            row['rejections'].append('SHADOW_SLOT_HAS_ANOTHER_CONTENT_WRITER')
        if not affected_reads:
            continue
        stage = by_stage.get(task.id)
        if stage is None or not snapshot['index'] < stage['index'] < write['index']:
            row['rejections'].append('SHADOW_READ_OUTSIDE_ENTRY_VERSION_WINDOW')
            continue
        if stage['kind'] == 'PROVIDER':
            row['rejections'].append('PROVIDER_BORROWED_ROOT_READONLY_AND_NO_ESCAPE_UNPROVEN')
            continue
        typed_reads = [read for read in stage['array_reads'] if read['input_id'] == target]
        expressions = [read.get('expression') for read in typed_reads]
        if (stage['kind'] != 'COMPUTE' or task.kind is not K.COMPUTE or affected_reads != (shadow_id,)
                or not expressions or any(item not in stage['expr_ids'] for item in expressions)
                or any(read['epoch'] != 0 or read['ready_after'] != snapshot['id'] or read['storage'] != 'SHADOW'
                    for read in typed_reads)
                or any(nodes[item].op != 'input' or nodes[item].attrs != {'kind':'array','name':target,'epoch':0}
                    for item in expressions)):
            row['rejections'].append('ENTRY_VERSION_COMPUTE_READ_NOT_CLOSED')
            continue
        reads.append(_borrowed_read(stage, shadow_id, entry_id, full_rectangle, expressions))

    row.update(write_stage=write['id'], write_stage_index=write['index'], write_op=write['op'],
        write_rectangle=bounds, write_materialization=written.id, commit_stage=commit['id'],
        commit_stage_index=commit['index'], commit_rectangle=commit['normalized_bounds'])
    row['rejections'] = list(dict.fromkeys(row['rejections']))
    if not row['rejections']:
        row.update(decision='ELIDE', borrowed_reads=reads, shadow_after_write_validity='WRITE_RECTANGLE_ONLY',
            conditions=['ORIGINAL_GUARDED_STEP_TRANSACTION_AND_QUIESCENCE',
                'BORROWED_EXTERNAL_ENTRY_UNCHANGED_UNTIL_SUCCESS_COMMIT',
                'EXACT_ORIGINAL_SHADOW_ROI_WRITE_COMPLETES_BEFORE_ANY_ROI_READ'])
    return row


def derive_step_snapshot_elision(region, step_plan, workspace_plan, native_metadata, *, source_refs=()):
    """Derive a bounded rewrite; physical storage outside the written ROI stays undefined."""
    region = validate_stateful_region(region)
    region_digest, step_digest, workspace_digest = (stateful_digest(region), semantic_sha256(step_plan),
        semantic_sha256(workspace_plan))
    _need((native_metadata.get('schema'), native_metadata.get('schema_version')) == ('scar.native-stateful-workspace-repeat', 1)
        and native_metadata['base_ids']['plan_sha256'] == workspace_plan['step_plan_sha256'] == step_digest
        and native_metadata['base_ids']['region_sha256'] == workspace_plan['step_region_sha256'] == region_digest
        and native_metadata['workspace_plan_sha256'] == workspace_digest
        and step_plan['region_sha256'] == region_digest and step_plan['region_id'] == region.id,
        'source region, step plan, versioned workspace and original native differ')
    graph, checked_reads = _initialization(workspace_plan)
    validate_versioned_schedule(graph)
    transaction = step_plan['transaction']
    stages = step_plan['stages']
    reasons = []
    if (step_plan['cpu_threads'] != 1 or native_metadata['cpu_threads'] != 1
            or not _GUARDS <= set(step_plan['guard_contract'])
            or not {'SYNCHRONOUS_CPU_EXECUTION', 'NO_FOREIGN_CONCURRENT_STATE_READERS_OR_WRITERS'}
                <= set(step_plan['external_assumptions'])
            or step_plan['alias_contract'] != 'PAIRWISE_DISJOINT_ARRAY_INPUT_STORAGE_AND_PRIVATE_WORKSPACE'):
        reasons.append('ORIGINAL_SERIAL_ISOLATION_AND_ALIAS_CONTRACT_REQUIRED')
    if not (transaction['external_writes_before_success'] is False
            and transaction['no_failing_operations_or_GuardFailure_after_first_external_commit'] is True
            and transaction['commit'] == 'ORIGINAL_WRITE_ROIS_IN_SOURCE_ORDER_FROM_FINAL_PRIVATE_SHADOWS'
            and transaction['failure'] == 'ORIGINAL_METHOD_ONCE_FROM_UNCHANGED_EXTERNAL_STATE'):
        reasons.append('ORIGINAL_FAILURE_AND_COMMIT_TRANSACTION_REQUIRED')
    stage_kinds = [stage['kind'] for stage in stages]
    _need('COMMIT' in stage_kinds, 'original step requires a transactional COMMIT tail')
    if (set(stage_kinds[stage_kinds.index('COMMIT'):]) != {'COMMIT'}
            or not set(stage_kinds) <= _STAGES or any(task.kind is K.EXPORT for task in graph.tasks)):
        reasons.append('OPAQUE_OR_POST_COMMIT_PAYLOAD_EFFECT')
    actions = {action.id:action for action in region.actions if action.kind == 'write'}
    nodes = {node.id:node for node in region.nodes}
    arrays = [row.id for row in region.inputs if row.kind == 'array']
    roots = [_root(target, index, graph, step_plan, actions, nodes, reasons)
        for index, target in enumerate(arrays) if target in step_plan['shadows']]
    original = sum(row['snapshot_bytes'] for row in roots)
    elided = sum(row['snapshot_bytes'] for row in roots if row['decision'] == 'ELIDE')
    return dict(schema=SCHEMA, schema_version=1, status='PROVEN_CONDITIONAL' if elided else 'REJECTED',
        parents=dict(region_sha256=region_digest, step_plan_sha256=step_digest, workspace_plan_sha256=workspace_digest,
            native_execution_plan_sha256=native_metadata['execution_plan_sha256'],
            native_source_sha256=native_metadata['sha256']['source'], source_refs=list(source_refs)),
        scope='ONE_TRANSACTIONAL_STEP_NO_INTERSTEP_CONTENT_CARRY', roots=roots,
        original_private_reads_replayed=checked_reads,
        work_counts=dict(basis='SOURCE_SNAPSHOT_COPY_PAYLOAD_NOT_OBSERVED_DRAM_TRAFFIC',
            original_snapshot_bytes=original, remaining_snapshot_bytes=original-elided, elided_snapshot_bytes=elided),
        original_guard_contract=step_plan['guard_contract'], original_external_assumptions=step_plan['external_assumptions'],
        original_finite_scans_arithmetic_providers_affinity_fenv_commit_and_fallback_preserved=True,
        private_capacity_reduced=False, provider_borrowing_widened=False, whole_shadow_defined=False,
        source_effects_closed=False, transformation_authorized=False, deployment_authorized=False,
        target_executed=False)


def replay_step_snapshot_elision(native, *, source_refs=()):
    """Cold-admit original native source and workspace without loading a DLL."""
    from scar.codegen.stateful_workspace import inspect_workspace_repeat
    path = Path(native).resolve()
    manifest = path if path.is_file() else path/'artifact.json'
    reference = artifact_reference(manifest, 'scar.native-stateful-workspace-repeat')
    metadata, (_, region, step, workspace, _) = inspect_workspace_repeat(manifest)
    _need(read_artifact(reference) == metadata, 'original native manifest changed during replay')
    refs = [dict(role='original_native', artifact=reference)]
    for key, schema in [('region','scar.stateful-array-region'), ('step_plan','scar.stateful-stage-plan'),
            ('workspace_plan','scar.stateful-workspace-reuse-plan')]:
        ref = artifact_reference(manifest.parent/metadata['files'][key], schema)
        _need(ref['sha256'] == metadata['sha256'][key], 'original '+key+' changed during replay')
        refs.append(dict(role=key, artifact=ref))
    return derive_step_snapshot_elision(region, step, workspace, metadata, source_refs=tuple(refs)+source_refs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', required=True)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--out', type=Path)
    operation.add_argument('--check', type=Path)
    args = parser.parse_args()
    proof = replay_step_snapshot_elision(args.native)
    if args.check:
        _need(json.loads(args.check.read_bytes()) == proof, 'snapshot-elision proof differs from original replay')
    else:
        with args.out.open('x') as stream:
            json.dump(proof, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(status=proof['status'], work_counts=proof['work_counts'],
        root_decisions=[(row['input_id'],row['decision']) for row in proof['roots']])))


if __name__ == '__main__':
    main()

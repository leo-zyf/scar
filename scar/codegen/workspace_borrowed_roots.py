"""Render one certified current-call borrow without changing the session ABI."""

from hashlib import sha256

from scar.optimization.resource_solver import semantic_sha256

_CLONE = '__attribute__((noinline, noipa))\nint32_t run_with_workspace('
_WORKSPACE_GETTER = 'extern "C" SCAR_EXPORT_V1 const char *scar_workspace_plan_sha256_v1()'
_SESSION_GETTER = 'extern "C" SCAR_EXPORT_V1 const char *scar_workspace_session_plan_sha256_v1() noexcept'
_SESSION_STEP = 'extern "C" SCAR_EXPORT_V1 int32_t scar_workspace_session_step_v1('
_SESSION_CALL = '''        const int32_t status = run_with_workspace(arrays, array_count, scalars,
            scalar_count, providers, provider_count, session->slots, first_iteration);'''
_SESSION_CALL_REWRITE = '''        const int32_t status = first_iteration
            ? run_with_borrowed_roots(arrays, array_count, scalars,
                scalar_count, providers, provider_count, session->slots)
            : run_with_workspace(arrays, array_count, scalars,
                scalar_count, providers, provider_count, session->slots, false);'''


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _digest(text):
    return sha256(text.encode()).hexdigest()


def _span(capture):
    return '\n'.join(' '*12+line for line in capture.split('\n'))


def _root_edits(record, step, captured):
    slots = {slot['id']:index for index, slot in enumerate(step['workspace_slots'])}
    stages = {stage['id']:stage for stage in step['stages']}
    snapshot = stages[record['snapshot_stage']]
    write = stages[record['write_stage']]
    commit = stages[record['commit_stage']]
    root, array_index = record['input_id'], record['array_index']
    rows, cols = record['shape']
    slot = slots[record['shadow_slot']]
    before = f'slots[{slot}].get()'
    borrowed = f'static_cast<const double *>(arrays[{array_index}].data)'
    (r0, r1), (c0, c1) = record['write_rectangle']
    _need(snapshot['kind'] == 'SNAPSHOT_ROOT' and snapshot['target'] == root
        and snapshot['index'] == record['snapshot_stage_index']
        and list(snapshot['shape']) == [rows, cols] == list(step['input_shapes'][root])
        and snapshot['output_payload'] == record['shadow_payload']
        and step['assignment'][record['shadow_payload']]['slot_id'] == record['shadow_slot']
        and write['kind'] == 'WRITE_SHADOW' and write['target'] == root
        and write['index'] == record['write_stage_index'] and write['op'] == record['write_op']
        and write['normalized_bounds'] == record['write_rectangle']
        and commit['kind'] == 'COMMIT' and commit['target'] == root
        and commit['index'] == record['commit_stage_index']
        and commit['normalized_bounds'] == record['commit_rectangle'] == record['write_rectangle']
        and 0 <= r0 < r1 <= rows and 0 <= c0 < c1 <= cols,
        'borrowed root stage IDs, geometry or private slot differ from C++ generation')
    original = captured[snapshot['index']]
    copy_line = f'        {before}[row * {cols} + col] = {borrowed}[row * {cols} + col];'
    _need(original.split('\n') == [f'for (int64_t row = 0; row < {rows}; ++row) {{',
        f'    for (int64_t col = 0; col < {cols}; ++col) {{', copy_line, '    }', '}'],
        'one exact original root-copy loop is required')
    edits = {snapshot['index']: []}
    seen = set()
    for read in record['borrowed_reads']:
        stage = stages[read['stage_id']]
        index = stage['index']
        _need(index == read['stage_index'] and stage['kind'] == read['kind']
              and snapshot['index'] < index <= write['index'] and index not in seen,
              'borrowed read must be unique and precede or equal the complete write')
        seen.add(index)
        text = captured[index]
        if stage['kind'] == 'COMPUTE':
            _need(text.count(before) > 0 and all(before not in line or 'const double ' in line
                and before not in line.split(' = ', 1)[0] for line in text.split('\n')),
                'borrowed roots may only replace read operands in an admitted compute stage')
            edits[index] = [(before, borrowed, text.count(before))]
        elif stage['id'] == write['id']:
            target_subscript = f'[({r0} + row) * {cols} + {c0} + col]'
            target = before+target_subscript
            _need(write['shape'] == [r1-r0, c1-c0] and text.count(before) == 1
                and write['op'] in ('assign', 'add'),
                'uninitialized scratch requires a complete assignment or an old-value add')
            if write['op'] == 'add':
                old = target+' += '
                _need(text.count(old) == 1, 'one exact old-value addition required')
                edits[index] = [(old, target+' = '+borrowed+target_subscript+' + ', 1)]
            else:
                _need(text.count(target+' = ') == 1, 'one exact complete assignment required')
        else:
            raise ValueError('unsupported borrowed root read kind')
    for index, text in captured.items():
        if before in text:
            _need(index in edits or index == write['index'] and write['op'] == 'assign',
                  'private root shadow has an unaccounted stage use')
    _need(write['op'] != 'add' or write['index'] in edits, 'old-value addition lacks a borrowed read')
    return edits


def render_borrowed_roots(source, session, workspace_source, workspace, step, capture, proof, proof_refs):
    """Add a dirty-step helper with only the consumer-verified root routes."""
    raw = source.decode()
    workspace_raw = workspace_source.decode()
    mapping = workspace['source_mapping']
    _need(_digest(raw) == session['source']['sha256']
          and _digest(workspace_raw) == workspace['sha256']['source']
          and workspace_raw.count(_CLONE) == workspace_raw.count(_WORKSPACE_GETTER) == 1,
          'exact source-replayed workspace clone and existing session are required')
    segment = workspace_raw[workspace_raw.index(_CLONE):workspace_raw.index(_WORKSPACE_GETTER)]
    _need(segment.endswith('\n}\n\n'), 'workspace helper namespace differs from the original renderer')
    clone = segment[:-len('\n}\n\n')]
    _need(_digest(clone) == mapping['cloned_run_sha256'], 'workspace clone differs from its original source map')
    captured = {row['index']:row['body'] for row in capture['stages']}
    edits = {}
    selected = [row for row in proof['roots'] if row['decision'] == 'ELIDE']
    _need(bool(selected), 'no source-qualified snapshot may be removed')
    for record in selected:
        for index, stage_edits in _root_edits(record, step, captured).items():
            edits.setdefault(index, []).extend(stage_edits)
    changes = []
    rewritten = clone
    for index, stage_edits in sorted(edits.items()):
        original = captured[index]
        changed = original if stage_edits else ''
        for old, new, count in stage_edits:
            _need(changed.count(old) == count, 'borrowed root expression span changed during rendering')
            changed = changed.replace(old, new)
        before = _span(original)+'\n'
        after = _span(changed)+'\n' if changed else ''
        _need(rewritten.count(before) == clone.count(before) == 1,
              'one original source span per changed stage required')
        rewritten = rewritten.replace(before, after, 1)
        changes.append(dict(stage_index=index, original_sha256=_digest(before), replacement_sha256=_digest(after)))

    body = rewritten.replace('int32_t run_with_workspace(', 'int32_t run_with_borrowed_roots(', 1)
    execution = dict(schema='scar.workspace-borrowed-root-execution-plan', schema_version=1,
        original_execution_plan_sha256=session['execution_plan_sha256'], proof_refs=proof_refs,
        selected_inputs=[row['input_id'] for row in selected], changes=changes,
        numerical_operation_order_preserved=True, existing_clean_step_path_unchanged=True)
    identity = semantic_sha256(execution)
    old_getter = _SESSION_GETTER+' {\n    return "'+session['execution_plan_sha256']+'";\n}'
    new_getter = _SESSION_GETTER+' {\n    return "'+identity+'";\n}'
    helper = 'namespace {\n'+body+'\n}\n\n'
    _need(raw.count(_SESSION_STEP) == raw.count(_SESSION_CALL) == raw.count(old_getter) == 1,
          'original session control or execution identity differs')
    generated = raw.replace(_SESSION_STEP, helper+_SESSION_STEP, 1).replace(
        _SESSION_CALL, _SESSION_CALL_REWRITE, 1).replace(old_getter, new_getter, 1)
    _need(generated.replace(helper, '', 1).replace(_SESSION_CALL_REWRITE, _SESSION_CALL, 1).replace(
        new_getter, old_getter, 1).encode() == source,
        'borrowed-root helper must preserve all other original session source bytes')
    return generated.encode(), execution

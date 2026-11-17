"""Instrument retained serial stage groups in their original native run frame.

All original stages and commits still execute on explicitly supplied ABI
ports. This renderer never initializes ports, compiles, loads or runs code.
"""
from hashlib import sha256
import json

from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.native-stage-group-source'
READ_SYMBOL = 'scar_native_stage_group_read_v1'
IDENTITY_SYMBOL = 'scar_native_stage_group_identity_v1'
_PREFIX = 'scar_native_stage_group_'


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def normalize_stage_groups(available_stage_indices, stage_indices=None, *, groups=None):
    """Normalize explicit ordered contiguous groups without adding any stage."""
    indices = list(available_stage_indices)
    _need(indices and all(type(i) is int for i in indices) and len(set(indices)) == len(indices),
        'distinct available stage indices required')
    _need(stage_indices is None or groups is None, 'choose stage_indices or groups, not both')
    groups = ([indices if stage_indices is None else stage_indices] if groups is None else groups)
    _need(type(groups) in (tuple, list) and groups, 'nonempty explicit stage groups required')
    groups = list(groups); previous = -1
    for group in groups:
        _need(type(group) in (tuple, list) and group and all(type(i) is int and i in indices for i in group),
            'every group must name supplied original non-COMMIT stages')
        positions = [indices.index(i) for i in group]
        _need(positions == list(range(positions[0], positions[-1]+1)) and positions[0] > previous,
            'stage groups must be contiguous, nonoverlapping and in original order')
        previous = positions[-1]
    return [list(group) for group in groups]


def render_native_stage_group(native_manifest, stage_indices=None, *, groups=None, stage_capture=None):
    """Return (observer C++ source, JSON provenance); no execution is performed.

    Groups are disjoint contiguous non-COMMIT stage sequences in original
    order. With neither argument, observe every non-COMMIT stage as one group.
    Required dependencies execute in the unchanged prefix, never fake ports.
    """
    from scar.codegen import stateful
    from scar.codegen.stateful_stage_capture import capture_stateful_stage_sources
    from scar.compiler.model import CpuPlacement
    from scar.compiler.native_lowering_view import derive_stateful_lowering_view
    from scar.compiler.stateful_model import stateful_from_dict

    view = derive_stateful_lowering_view(native_manifest, stage_capture=stage_capture)
    original = read_artifact(view['source'], json_document=False).decode('utf-8')
    manifest = read_artifact(view['native_manifest'])
    region = stateful_from_dict(read_artifact(view['region']))
    plan = read_artifact(view['plan'])
    binding = dict(plan['resource_binding'])
    binding['cpu_ids'], binding['evidence'] = tuple(binding['cpu_ids']), tuple(binding.get('evidence', ()))
    core = {name:manifest[name] for name in ('region_sha256', 'plan_sha256', 'provider_runtime')}
    capture = capture_stateful_stage_sources(region, plan, CpuPlacement(**binding), core)
    _need(capture['full_source_sha256'] == view['source']['sha256'], 'actual source changed during group rendering')
    records = capture['stages']; stages = [row for row in plan['stages'] if row['kind'] != 'COMMIT']
    indices = [row['index'] for row in stages]
    _need(indices and [row['index'] for row in records] == indices, 'complete original non-COMMIT capture required')
    groups = normalize_stage_groups(indices, stage_indices, groups=groups)
    # Reconstruct the entire original emitted program, not an isolated body
    # search. Repeated stage bodies then have positions proven by concatenation.
    bodies = ['\n'.join('            '+line for line in row['body'].split('\n')) if row['body'] else ''
        for row in records]
    program = '\n'.join(body for body in bodies if body)
    _need(program and original.count(program) == 1, 'one intact complete original stage-program span required')
    program_start = original.index(program); cursor = program_start; body_ranges = {}
    for index, body in zip(indices, bodies):
        body_ranges[index] = (cursor, cursor+len(body))
        if body:
            cursor += len(body)+1
    known = set()
    for stage in plan['stages']:
        _need(set(stage['depends_on']) <= known, 'stage has an unsupplied or out-of-order dependency')
        known.add(stage['id'])
    record_by_index = {row['index']:row for row in records}
    stage_by_index = {row['index']:row for row in stages}
    order = {node.id:i for i,node in enumerate(region.nodes)}
    spec = dict(native_manifest=view['native_manifest'], source=view['source'], groups=groups,
        renderer=artifact_reference(__file__, 'scar.python-source'), protocol='ORIGINAL_FRAME_GROUP_CUTS_V1')
    identity = stateful._digest(spec)
    count = 2*len(groups)
    _need(_PREFIX not in original and original.count('namespace {') == 1, 'observer symbol/namespace collision')
    declarations = '''#include <time.h>
namespace {
thread_local uint64_t scar_native_stage_group_cuts_v1[@COUNT@] = {};
thread_local bool scar_native_stage_group_complete_v1 = false;
inline bool scar_native_stage_group_clock_v1(uint64_t index) {
    timespec stamp;
    if (clock_gettime(CLOCK_MONOTONIC, &stamp) != 0 || stamp.tv_sec < 0 ||
        stamp.tv_nsec < 0 || stamp.tv_nsec >= 1000000000L) return false;
    const uint64_t seconds = static_cast<uint64_t>(stamp.tv_sec);
    if (seconds > (UINT64_MAX-static_cast<uint64_t>(stamp.tv_nsec))/1000000000ULL) return false;
    scar_native_stage_group_cuts_v1[index] = seconds*1000000000ULL+static_cast<uint64_t>(stamp.tv_nsec);
    return true;
}
}
'''.replace('@COUNT@', str(count))
    signature = 'int32_t run(const scar_array_v1 *arrays, uint64_t array_count,'
    _need(original.count(signature) == 1, 'one original run frame required')
    run_start = original.index(signature)
    entry = original.index(') {\n', run_start)+4
    success = '\n    return 0;\n}\nconst scar_stateful_table_v1 table'
    _need(original.count(success) == 1, 'one original normal-return/table boundary required')
    edits = [(original.index('namespace {'), 'observer declarations', declarations),
        (entry, 'reset observation only', '    for (auto &cut : scar_native_stage_group_cuts_v1) cut = 0;\n'
            '    scar_native_stage_group_complete_v1 = false;\n'),
        (original.index(success)+1, 'normal completion only', '    scar_native_stage_group_complete_v1 = true;\n')]

    def fence(ready):
        # Keep existing scalar values on the proper side of a clock. There are
        # no saved-port reloads or new scalar computations; constraints can
        # nevertheless change register pressure and optimization decisions.
        lines = ['            asm volatile("" : : "g"(numeric_guard) : "memory");']
        for key in ready:
            name = 'scalar'+str(order[key])
            lines.append(f'            asm volatile("" : : "g"({name}.value), "g"({name}.integer) : "memory");')
        return '\n'.join(lines)

    observed = []
    for number, group in enumerate(groups):
        emitted = [i for i in group if body_ranges[i][0] != body_ranges[i][1]]
        _need(emitted, 'a timed group must contain emitted work, not an empty supplied stage')
        first, last = emitted[0], emitted[-1]
        begin, end = body_ranges[first][0], body_ranges[last][1]
        entry_state, exit_state = record_by_index[group[0]], record_by_index[group[-1]]
        edits += [(begin, 'group '+str(number)+' begin', fence(entry_state['entry_scalar_ready'])+
            f'\n            if (!scar_native_stage_group_clock_v1({2*number})) return 3;\n'),
            (end, 'group '+str(number)+' end', '\n'+fence(exit_state['exit_scalar_ready'])+
                f'\n            if (!scar_native_stage_group_clock_v1({2*number+1})) return 3;')]
        members = {stage_by_index[i]['id'] for i in group}
        observed.append(dict(index=number, stage_indices=group, begin_cut=2*number, end_cut=2*number+1,
            entry_available=entry_state['entry_available'], entry_scalar_ready=entry_state['entry_scalar_ready'],
            exit_scalar_ready=exit_state['exit_scalar_ready'], entry_views=entry_state['views'],
            external_stage_dependencies=sorted({d for i in group for d in stage_by_index[i]['depends_on']}-members),
            untimed_prefix_stage_indices=indices[:indices.index(group[0])],
            untimed_suffix_stage_indices=indices[indices.index(group[-1])+1:]))
    suffix = '''
extern "C" SCAR_EXPORT_V1 const char *scar_native_stage_group_identity_v1() { return "@IDENTITY@"; }
extern "C" SCAR_EXPORT_V1 int32_t scar_native_stage_group_read_v1(uint64_t *out, uint64_t count) {
    if (!out || count != @COUNT@ || !scar_native_stage_group_complete_v1) return 1;
    for (uint64_t i = 0; i < count; ++i) out[i] = scar_native_stage_group_cuts_v1[i];
    return 0;
}
'''.replace('@IDENTITY@', identity).replace('@COUNT@', str(count))
    edits.append((len(original), 'diagnostic exports', suffix))
    parts, insertions, cursor, size = [], [], 0, 0
    for position, role, text in sorted(edits, key=lambda row:row[0]):
        _need(position >= cursor, 'overlapping observer insertion')
        retained = original[cursor:position]; parts += [retained, text]; size += len(retained)
        insertions.append(dict(char_start=size, char_end_exclusive=size+len(text), role=role,
            sha256=sha256(text.encode()).hexdigest()))
        size += len(text); cursor = position
    source = ''.join(parts)+original[cursor:]
    restored = source
    for insertion in reversed(insertions):
        a,b = insertion['char_start'], insertion['char_end_exclusive']
        restored = restored[:a]+restored[b:]
    _need(restored == original, 'removing observer insertions must restore every original source byte')
    mapped = []
    for stage, body in zip(stages, bodies):
        start,end = body_ranges[stage['index']]
        spans = []
        if body:
            start += sum(len(text) for pos, _, text in edits if pos <= start)
            end += sum(len(text) for pos, _, text in edits if pos < end)
            _need(source[start:end] == body, 'observer changed an original stage body')
            spans = [dict(start_line=source.count('\n', 0, start)+1,
                end_line=source.count('\n', 0, end-1)+1)]
        mapped.append(dict(index=stage['index'], id=stage['id'], kind=stage['kind'],
            source_action_ids=stage['source_action_ids'], expr_ids=stage['expr_ids'],
            source_line_spans=spans, body_sha256=record_by_index[stage['index']]['body_sha256'],
            mapping_status='PROVEN_BY_UNIQUE_COMPLETE_PROGRAM_CONCATENATION', spans_are_candidates=False))
    provenance = dict(schema=SCHEMA, schema_version=1, parent_view=view, specification=spec,
        observer_identity=identity, read_symbol=READ_SYMBOL, identity_symbol=IDENTITY_SYMBOL,
        read_signature='int32_t(uint64_t*,uint64_t)', identity_signature='const char*()', cut_count=count,
        clock=dict(api='clock_gettime', clock_id='CLOCK_MONOTONIC', unit='INTEGER_NANOSECONDS'),
        generated_source_sha256=sha256(source.encode()).hexdigest(), groups=observed, stages=mapped,
        insertions=insertions, all_original_source_bytes_preserved=True,
        all_original_stages_execute=True, executed_stage_indices=[row['index'] for row in plan['stages']],
        commit_rows=plan['commit_rows'], workspace_slots=plan['workspace_slots'], assignment=plan['assignment'],
        external_ports=dict(inputs=[dict(id=row.id, path=list(row.path), kind=row.kind,
            shape=plan['input_shapes'].get(row.id), required=True) for row in region.inputs],
            providers=[dict(id=row.id, module=row.module, symbol=row.symbol, required=True) for row in region.providers],
            initialization='ALL_ABI_PORTS_MUST_BE_EXPLICITLY_SUPPLIED_NO_GENERATED_VALUES',
            missing_dependencies='REJECTED_OR_SUPPLIED_BY_UNCHANGED_ORIGINAL_PREFIX',
            external_storage_contract='ORIGINAL_DISTINCT_ARRAYS_AND_EXCLUSIVE_STATE_GUARDS_REQUIRED'),
        memory_lifetime='ORIGINAL_POOL_ALLOCATION_SLOT_ALIASING_COMMIT_AND_REVERSE_RAII_RELEASE_UNCHANGED',
        original_table_retained_for_abi_only=True, original_artifact_identity_claimed=False,
        original_loader_source_replay_must_reject=True, native_run_entry_exit_clocked=False,
        observation_owner='SAME_NATIVE_CALLING_THREAD_TLS_NONREENTRANT',
        read_requires_successful_normal_return=True, inter_call_serialization_enforced=False,
        execution_claim='ORIGINAL_SOURCE_NORMAL_SUCCESS_PATH_NOT_MACHINE_INSTRUCTION_COUNTS',
        per_group_prefix_suffix_are_relative_to_this_group_clock=True,
        timed_scope='ONLY_SELECTED_ORIGINAL_STAGE_BODIES_WITH_CLOCK_AND_LIVE_OPERAND_CONSTRAINTS',
        excluded_from_group_clock=['ALLOCATION', 'INPUT_FINITE_SCANS', 'INITIAL_CALLER_AFFINITY_FENV',
            'FINAL_RESTORATION', 'COMMIT', 'RAII_POOL_RELEASE', 'PYTHON_FFI_WRAPPER'],
        retained_residuals=['CLOCK_CALLS', 'TLS_STORES', 'MEMORY_AND_SCALAR_OPERAND_COMPILER_CONSTRAINTS',
            'EARLIER_CUTS_CAN_CHANGE_LATER_GROUP_CONTEXT', 'INSTRUMENTATION_CAN_CHANGE_OPTIMIZATION_AND_REGISTER_PRESSURE'],
        numerical_input_regime='CALLER_DECLARED_NOT_INITIALIZED_OR_OBSERVED_BY_RENDERER',
        machine_code_identity_claimed=False, component_additivity_proven=False,
        source_F4_admitted=False, compiled=False, native_loaded=False, target_program_executed=False,
        performance_measured=False, timing_qualified=False, deployment_authorized=False)
    read_artifact(view['source'], json_document=False)
    return source, json.loads(json.dumps(provenance, sort_keys=True, allow_nan=False))


__all__ = ['SCHEMA', 'READ_SYMBOL', 'IDENTITY_SYMBOL', 'normalize_stage_groups', 'render_native_stage_group']

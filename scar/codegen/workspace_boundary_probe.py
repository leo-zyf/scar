"""Render canonical workspace control cuts, omitting all target DAG/commits.

Only source generation occurs here. The sample consumer must retain skipped
work, canonical input regime, compiler/layout and clock perturbation residuals.
"""
from hashlib import sha256
from pathlib import Path

from scar.analysis.workspace_source import workspace_serial_source_contract
from scar.codegen.boundary_cuts import _fragment, _pin, _serial_lines
from scar.ir.execution_recipe import read_artifact

_HERE = Path(__file__).resolve().parent
CUT_LAYOUT = dict(prefix=['native_enter', 'outer_preflight_done', 'pool_allocate_done'],
    iteration=['step_enter', 'bind_done', 'restore_begin', 'restore_done', 'step_returned', 'progress_done'],
    suffix=['loop_done', 'before_pool_release', 'after_pool_release'])
PHASE_INTERVALS = (
    dict(id='outer_preflight', scope='ROOT', segment='prefix', start=0, end=1),
    dict(id='whole_pool_allocate', scope='ROOT', segment='prefix', start=1, end=2),
    dict(id='step_preflight_context', scope='LOOP', segment='iteration', start=0, end=1),
    dict(id='step_validate_restore', scope='LOOP', segment='iteration', start=2, end=3),
    dict(id='step_cleanup_return_progress', scope='LOOP', segment='iteration', start=3, end=5),
    dict(id='whole_pool_release', scope='ROOT', segment='suffix', start=1, end=2),
)
SYMBOL = 'scar_workspace_boundary_probe_v1'


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _hash(value):
    return sha256(value.encode() if type(value) is str else value).hexdigest()


def render_workspace_boundary_probe(native_metadata, step_plan, *, native_reference, cpp_reference):
    """Return (source, provenance); unchanged ABI records, nine C arguments."""
    contract = workspace_serial_source_contract(native_metadata, step_plan,
        native_reference=native_reference, cpp_reference=cpp_reference)
    count = native_metadata['trip_count']
    _need(type(count) is int and 0 < count <= (2**64-7)//6,
          'canonical workspace boundary observer requires one positive exact bounded count')
    raw = read_artifact(cpp_reference, json_document=False)
    text = raw.decode(); mapping = native_metadata['source_mapping']
    source_span = mapping['original_run_source_range']
    original_run = text[source_span['char_start']:source_span['char_end_exclusive']]
    _need(_hash(original_run) == mapping['original_run_sha256'], 'original run source range differs')
    helper_prefix = text[:source_span['char_start']]
    _need(helper_prefix.count('namespace {') == 1 and 'int32_t run(' not in helper_prefix,
          'one generic helper prefix without target mathematical body required')
    helper_prefix = helper_prefix.replace('namespace {', '#include <time.h>\n\nnamespace {', 1)
    signature_end = original_run.index(') {')+3
    preflight = original_run[signature_end:mapping['pool_block_char_start']]
    pool = original_run[mapping['pool_block_char_start']:mapping['pool_block_char_end_exclusive']]
    _need(_hash(preflight) == mapping['preflight_sha256'] and _hash(pool) == mapping['pool_block_sha256'],
          'original workspace pool or ABI preflight source differs')
    marker = '__attribute__((noinline, noipa))\nint32_t run_with_workspace('
    end = '\n}\n\nextern "C" SCAR_EXPORT_V1 const char *scar_workspace_plan_sha256_v1'
    _need(text.count(marker) == text.count(end) == 1, 'one retained non-inlined workspace clone required')
    clone_start = text.index(marker); clone_end = text.index(end, clone_start)
    clone = text[clone_start:clone_end]
    _need(_hash(clone) == mapping['cloned_run_sha256'], 'actual workspace clone source changed')
    renderer = _HERE/'stateful.py'
    bind, check, serial_source = _serial_lines(renderer, renderer.read_text())
    bind_end = clone.index(bind)+len(bind)
    clone_header_end = clone.index(') {')+3
    header = clone[:clone_header_end].replace('run_with_workspace(', 'workspace_boundary_step(', 1)
    header = header[:-3]+', uint64_t *step_cuts) {'
    prefix = clone[clone_header_end:bind_end]
    restore_fragment, restore_source = _fragment(_HERE/'stateful_v1.cpp.in',
        (_HERE/'stateful_v1.cpp.in').read_text(),
        '    numeric_guard |= static_cast<uint32_t>(std::fetestexcept(floating_flags) != 0);', '@COMMITS@')
    restore_start = clone.index(check, bind_end)
    restore_body_start = clone.index(restore_fragment, restore_start)
    _need(not clone[restore_start+len(check):restore_body_start].strip(),
          'unexpected work between original bound check and restore checks')
    restore_end = restore_body_start+len(restore_fragment)
    restore = clone[restore_start:restore_end]
    # The only body source retained is preflight/context and restore/status.
    # The entire contiguous DAG span and every final commit store are skipped.
    skipped_dag = clone[bind_end:restore_start]
    skipped_commit = clone[restore_end:]
    _need(skipped_commit.endswith('    return 0;\n}') and len(skipped_dag.strip()) > 0,
          'explicit skipped target-DAG and commit/return spans required')
    adapter_marker = 'extern "C" SCAR_EXPORT_V1 int32_t scar_workspace_repeat_stateful_v1('
    _need(text.count(adapter_marker) == 1, 'one actual outer workspace adapter required')
    outer = text[text.index(adapter_marker):]
    guard_start = outer.index('    if (!table || table->abi_version != 1 ||')
    guard_end = outer.index('    // Original pure ABI/shape/alias/provider/payload preflight before malloc.')
    table_guard = outer[guard_start:guard_end]
    _need(_hash(table_guard) == mapping['parent_table_check_sha256'], 'original parent table guard changed')
    _need(outer.count(preflight.rstrip()) == 1 and outer.count(pool.rstrip()) == 1,
          'outer preflight/allocation source must be the original copied blocks')
    _need(outer.count('    for (uint64_t step = 0; step < repetitions; ++step) {') == 1
          and outer.count('        *completed = step + 1;') == 1,
          'original shared loop and committed progress source changed')
    ids = native_metadata['base_ids']
    clock = '''
bool workspace_record_cut(uint64_t *out, uint64_t index) noexcept {
    struct timespec stamp;
    if (::clock_gettime(CLOCK_MONOTONIC, &stamp) != 0 || stamp.tv_sec < 0 ||
        stamp.tv_nsec < 0 || stamp.tv_nsec >= 1000000000) return false;
    const __int128 value = __int128(stamp.tv_sec) * 1000000000 + stamp.tv_nsec;
    if (value > std::numeric_limits<uint64_t>::max()) return false;
    out[index] = static_cast<uint64_t>(value);
    return true;
}
int32_t unused_parent(const scar_array_v1 *, uint64_t, const scar_scalar_v1 *, uint64_t,
                      const scar_provider_v1 *, uint64_t) { return 1; }
'''
    # Base metadata does not duplicate parameter kinds; original run guards are
    # already concretely specialized. Recover their integer ABI constants only.
    import re
    match = re.search(r'if \(array_count != ([0-9]+) \|\| scalar_count != ([0-9]+) \|\|\s*provider_count != ([0-9]+)', preflight)
    _need(match is not None, 'literal original ABI counts required')
    array_count, scalar_count, provider_count = (int(v) for v in match.groups())
    _need(array_count == len(step_plan['input_shapes']), 'original array ABI count differs from the source plan')
    dummy = ('const scar_stateful_table_v1 canonical_table = {1, sizeof(scar_stateful_table_v1), '
        +f'{array_count}, {scalar_count}, {provider_count}, &unused_parent, '
        +'"'+ids['region_sha256']+'", "'+ids['plan_sha256']+'", "'+ids['contract_sha256']+'"};\n'
        +'const scar_stateful_table_v1 *volatile canonical_parent = &canonical_table;\n')
    step = (header+'\n    if (!workspace_record_cut(step_cuts, 0)) return 3;\n'+prefix
        +'\n    if (!workspace_record_cut(step_cuts, 1)) return 3;\n'
        +'    if (!workspace_record_cut(step_cuts, 2)) return 3;\n'+restore
        +'    if (!workspace_record_cut(step_cuts, 3)) return 3;\n    return 0;\n}\n')
    cut_count = 6*count+6
    outer_probe = f'''
}}
extern "C" SCAR_EXPORT_V1 int32_t {SYMBOL}(
    const scar_array_v1 *arrays, uint64_t array_count, const scar_scalar_v1 *scalars, uint64_t scalar_count,
    const scar_provider_v1 *providers, uint64_t provider_count, uint64_t repetitions,
    uint64_t *cuts, uint64_t cut_count) {{
    if (!cuts || cut_count != {cut_count}ULL || repetitions != {count}ULL) return 1;
    for (uint64_t i = 0; i < cut_count; ++i) cuts[i] = 0;
    const scar_stateful_table_v1 *table = canonical_parent;
    uint64_t completed_value = 0; uint64_t *completed = &completed_value;
    if (!workspace_record_cut(cuts, 0)) return 3;
{table_guard}{preflight}
    if (!workspace_record_cut(cuts, 1)) return 3;
    {{
{pool}
    if (!workspace_record_cut(cuts, 2)) return 3;
    for (uint64_t step = 0; step < repetitions; ++step) {{
        uint64_t *step_cuts = cuts + 3 + 6 * step;
        const int32_t status = workspace_boundary_step(arrays, array_count, scalars, scalar_count,
            providers, provider_count, slots, step_cuts);
        if (!workspace_record_cut(step_cuts, 4)) return 3;
        if (status) return status;
        *completed = step + 1;
        if (!workspace_record_cut(step_cuts, 5)) return 3;
    }}
    if (!workspace_record_cut(cuts, 3 + 6 * repetitions)) return 3;
    if (!workspace_record_cut(cuts, 4 + 6 * repetitions)) return 3;
    }}
    if (!workspace_record_cut(cuts, 5 + 6 * repetitions)) return 3;
    return *completed == repetitions ? 0 : 1;
}}
'''
    source = helper_prefix+clock+dummy+step+outer_probe
    _need('@' not in source and 'run_with_workspace(' not in source
          and 'scar_workspace_repeat_stateful_v1(' not in source,
          'target run/adapter or unresolved template escaped into the canonical observer')
    provenance = dict(schema='scar.workspace-boundary-probe-source', schema_version=1,
        profile='CANONICAL_SERIAL_WORKSPACE_BOUNDARY_COMPONENTS_NOT_TARGET_DAG',
        native_reference=native_reference, cpp_reference=cpp_reference, source_contract=contract,
        source_pins=[_pin(Path(__file__)), _pin(_HERE/'stateful_abi_v1.h'), _pin(_HERE/'scar_abi_v1.h')],
        fragments=dict(generic_helpers_sha256=_hash(text[:source_span['char_start']]),
            preflight_sha256=_hash(preflight), pool_sha256=_hash(pool), prefix_sha256=_hash(prefix),
            restore_sha256=_hash(restore), table_guard_sha256=_hash(table_guard),
            restore_template=restore_source, serial_renderer=serial_source),
        skipped_source_spans=[dict(kind='TARGET_DAG', char_start=clone_start+bind_end,
            char_end_exclusive=clone_start+restore_start, sha256=_hash(skipped_dag)),
            dict(kind='COMMIT_AND_ORIGINAL_RETURN', char_start=clone_start+restore_end,
                char_end_exclusive=clone_end, sha256=_hash(skipped_commit))],
        generated_source_sha256=_hash(source), symbol=SYMBOL, argument_count=9,
        cut_layout=CUT_LAYOUT, phase_intervals=list(PHASE_INTERVALS), cut_count=cut_count,
        trip_count=count, prefix_offset=0, iteration_offset=3, iteration_stride=6, suffix_offset=3+6*count,
        abi_counts=dict(arrays=array_count, scalars=scalar_count, providers=provider_count),
        workspace_slot_capacities_bytes=[row['capacity_bytes'] for row in step_plan['workspace_slots']],
        memory={key: step_plan[key] for key in ('external_array_bytes', 'workspace_payload_bytes',
            'total_region_array_payload_bytes', 'memory_budget_bytes', 'budget_scope')},
        original_compiler=native_metadata['compiler'],
        clock=dict(api='clock_gettime', clock_id='CLOCK_MONOTONIC', unit='ns'),
        noinline_noipa_step_boundary_preserved=True, pool_release_order='CPP_STD_ARRAY_REVERSE_DESTRUCTION',
        dummy_parent_table_never_dispatched=True, providers_never_called=True,
        skipped_work=['TARGET_DAG', 'COMMIT', 'PROVIDER_CALLS'],
        retained_residuals=['SKIPPED_DAG_AND_COMMIT_FP_CACHE_STATE', 'OPAQUE_PROVIDER_AFFINITY_EFFECTS',
            'CANONICAL_INPUT_DATA_REGIME', 'ALLOCATOR_STATE_AND_CACHES', 'CLOCK_AND_OUTPUT_STORES',
            'LOOP_AND_CONTROL_GAPS', 'DUMMY_PARENT_POINTER_AND_PROBE_ABI',
            'CUT_BUFFER_INITIALIZATION_BEFORE_NATIVE_ENTER', 'STATUS_RETURN_AFTER_FINAL_CUT'],
        instrumentation_subtracted=False, target_program_executed=False,
        native_compiled=False, performance_measured=False)
    return source, provenance


__all__ = ['render_workspace_boundary_probe', 'CUT_LAYOUT', 'PHASE_INTERVALS', 'SYMBOL']

"""Reviewed serial source effects for one admitted workspace step template.

The parent reader owns full preparation/native admission. This small contract
rechecks retained bytes and exact emitter fragments without running a target,
opening a native library, or inferring placement backwards from a final guard.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.record_codec import encode
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.workspace-serial-source-contract'
_ROOT = Path(__file__).resolve().parents[1]
_FRAGMENTS = (
    ('stage_emitter', 'stateful.py', '    def scalar_value(key, statements):', "    program='\\n'.join",
     '283aed31ce7347299696c8f043ee535289810626e9175576d89fc2f7342783f6'),
    ('scalar_helpers', 'stateful_v1.cpp.in', 'bool finite(double value) {', '\nint32_t run(',
     '67224c397ffeb2a4d593ab8499ded1158fb4dfc66527fa80e5e7b4fa376b3b30'),
    ('capture_and_finite', 'stateful_v1.cpp.in', '    ThreadAffinity caller_affinity;', '@SCALAR_DECLARATIONS@',
     'e06d21aca5d031d22f33a14a2f7d9d1d7dcc1150f737c6db1264e99ac34b585a'),
    ('restore_before_commit', 'stateful_v1.cpp.in',
     '    numeric_guard |= static_cast<uint32_t>(std::fetestexcept(floating_flags) != 0);', '@COMMITS@',
     '2f8c1f947a3cf9d30256751b1619d0c992384c7ec657f88dc384b9cef7bc737d'),
    ('successful_bind', 'loop_v1.cpp.in', '    bool bind(int cpu) noexcept {',
     '    bool restore() noexcept {\n        if (!changed_)',
     'eee798a72b6f50233cb3e240f50683526ecffdc39789727a855c3bf7e4f3eed9'),
)


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _hash(raw):
    return sha256(raw).hexdigest()


def workspace_serial_source_contract(native_metadata, step_plan, *, native_reference, cpp_reference):
    """Return conditional source rules, not observations or runtime permission."""
    from scar.codegen import stateful as backend
    from scar.codegen.boundary_cuts import _fragment, _serial_lines
    native_reference, cpp_reference = encode(native_reference), encode(cpp_reference)
    _need(native_reference['schema'] == 'scar.native-stateful-workspace-repeat'
          and cpp_reference['schema'] == 'scar.cpp-source'
          and semantic_sha256(read_artifact(native_reference)) == semantic_sha256(native_metadata),
          'original admitted workspace native manifest required')
    _need(native_metadata['cpu_threads'] == step_plan['cpu_threads'] == 1
          and type(step_plan['cpu_threads']) is int
          and native_metadata['cpu_ids'] == list(step_plan['resource_binding']['cpu_ids'])
          and len(native_metadata['cpu_ids']) == 1,
          'only the original serial one-calling-thread workspace profile is supported; OpenMP is unresolved')
    _need(cpp_reference['sha256'] == native_metadata['sha256']['source']
          and Path(cpp_reference['uri']).resolve()
              == (Path(native_reference['uri']).parent/native_metadata['files']['source']).resolve(),
          'retained C++ reference differs from the admitted workspace source member')
    raw = read_artifact(cpp_reference, json_document=False); text = raw.decode('utf-8')
    base_reference = native_metadata['base_native_reference']; base = read_artifact(base_reference)
    _need(base['implementation_sha256'] == backend.implementation_sha256()
          and base['plan_sha256'] == semantic_sha256(step_plan)
          and native_metadata['base_ids'] == {key: base[key] for key in
              ('build_id', 'region_sha256', 'plan_sha256', 'contract_sha256')},
          'original base implementation/step plan/source identities differ')
    mapping = native_metadata['source_mapping']
    _need(all(mapping.get(key) is True for key in ('per_step_preflight_retained',
        'per_step_teams_and_restoration_retained', 'per_step_commits_retained',
        'original_run_and_getter_copied_verbatim', 'reinstating_pool_and_original_signature_restores_original_run'))
        and mapping.get('mathematical_body_changed') is False,
        'source-lifetime clone must retain original guards, context, mathematical body and commits')
    fragments = {}
    for name, filename, start, end, digest in _FRAGMENTS:
        path = _ROOT/'codegen'/filename
        _, receipt = _fragment(path, path.read_text(), start, end)
        _need(receipt['extracted_sha256'] == digest, 'reviewed serial source effect fragment changed: '+name)
        fragments[name] = receipt
    workspace_template = _ROOT/'codegen/stateful_workspace_v1.cpp.in'
    _need(_hash(workspace_template.read_bytes()) == '022b5a7013b1365972a8c13118578c6bdb6b5650e5b0dda0185e417d8c0cbb11',
          'reviewed workspace control wrapper source changed')
    renderer = _ROOT/'codegen/stateful.py'
    bind, check, serial = _serial_lines(renderer, renderer.read_text())
    fragments['serial_renderer'] = serial
    separator = '\n\n// Workspace-lifetime adapter only.'
    _need(text.count(separator) == 1, 'one preserved original source prefix required')
    original = text.split(separator, 1)[0]
    _need(_hash(original.encode()) == mapping['original_full_source_sha256'],
          'preserved original source prefix differs')
    location = mapping['original_run_source_range']
    old_run = original[location['char_start']:location['char_end_exclusive']]
    _need(_hash(old_run.encode()) == mapping['original_run_sha256'], 'original run source range differs')
    marker = '__attribute__((noinline, noipa))\nint32_t run_with_workspace('
    end = '\n}\n\nextern "C" SCAR_EXPORT_V1 const char *scar_workspace_plan_sha256_v1'
    _need(text.count(marker) == text.count(end) == 1, 'one original workspace clone source range required')
    clone_start = text.index(marker); clone_end = text.index(end, clone_start)
    clone = text[clone_start:clone_end]
    _need(_hash(clone.encode()) == mapping['cloned_run_sha256'], 'actual clone differs from its admitted source mapping')
    old_signature_end = old_run.index(') {')+3
    clone_signature_end = clone.index(') {')+3
    old_body_without_pool = (old_run[old_signature_end:mapping['pool_block_char_start']]
        +old_run[mapping['pool_block_char_end_exclusive']:])
    _need(clone[clone_signature_end:] == old_body_without_pool
          and _hash(old_body_without_pool.encode()) == mapping['original_body_without_pool_sha256'],
          'workspace per-step body differs beyond the original pool allocation removal')
    _need(clone.count(bind) == clone.count(check) == 1, 'original serial bind/check cutpoints differ')
    bind_start = clone.index(bind); bind_end = bind_start+len(bind)
    restore_start = clone.index(check, bind_end)
    _need(bind_end < restore_start, 'serial stage body is not between successful bind and restore')
    def actual_span(start, stop):
        a, b = clone_start+start, clone_start+stop
        return dict(artifact=cpp_reference, char_start=a, char_end_exclusive=b,
            line_start=text.count('\n', 0, a)+1, line_end=text.count('\n', 0, max(a, b-1))+1,
            sha256=_hash(text[a:b].encode()))
    fragments['actual_context_through_bind'] = actual_span(0, bind_end)
    fragments['actual_shared_stage_body'] = actual_span(bind_end, restore_start)
    fragments['actual_restore_and_commit_tail'] = actual_span(restore_start, len(clone))
    closed = [dict(kind=kind, capability=capability, stage_kind=stage,
        whole_thread_units=1, affinity_preserving=True)
        for kind, capability, stage in (
            ('SNAPSHOT', 'CPU_HOST_SNAPSHOT', 'SNAPSHOT_ROOT'),
            ('COMPUTE', 'CPU_SCALAR_DAG', 'COMPUTE'),
            ('COMPUTE', 'CPU_SCALAR_SETUP', 'SCALAR'),
            ('STATE_WRITE', 'CPU_STATE_WRITE', 'WRITE_SHADOW'))]
    files = ('analysis/workspace_source.py', 'codegen/stateful.py', 'codegen/stateful_v1.cpp.in',
             'codegen/loop_v1.cpp.in', 'codegen/stateful_workspace.py', 'codegen/stateful_workspace_v1.cpp.in')
    result = dict(schema=SCHEMA, schema_version=1, cpu_ids=list(native_metadata['cpu_ids']),
        resource_binding=deepcopy(step_plan['resource_binding']),
        target=deepcopy(step_plan['resource_context']['target_snapshot']),
        step_plan_sha256=semantic_sha256(step_plan),
        base_implementation_sha256=base['implementation_sha256'],
        native_reference=native_reference, cpp_reference=cpp_reference,
        enter_phase_kind='ORIGINAL_PER_STEP_PREFLIGHT_AND_CONTEXT', enter_fact_at='COMPLETION',
        restore_phase_kind='ORIGINAL_PER_STEP_VALIDATE_AND_RESTORE', restore_kill_at='START',
        enter_success_means_selected_bind_complete=True, original_stage_work_before_enter_complete=False,
        original_body_between_bind_and_restore=True, context_whole_thread_units=None,
        closed_stages=closed, opaque_stage_kinds=['PROVIDER'], opaque_capabilities=['CPU_NUMPY_UFUNC_PROVIDER'],
        opaque_transition='KILL_SELECTED_AFFINITY_KNOWLEDGE_UNTIL_A_NEW_SOURCE_BIND',
        unknown_stage_transition='KILL_SELECTED_AFFINITY_KNOWLEDGE',
        provider_whole_thread_units=None, provider_caller_participation_units=1,
        commit_whole_thread_units=None, commit_caller_participation_units=1,
        commit_exact_selected_cpu=False, commit_after_original_caller_restore=True,
        commit_reason='STD_MEMCPY_WHOLE_SCOPE_HELPER_THREAD_EFFECTS_NOT_REVIEWED',
        root_or_mixed_phase_whole_thread_units=None,
        source_pins={name: artifact_reference(_ROOT/name, 'scar.python-or-template-source') for name in files},
        fragments=fragments,
        source_refs=[dict(artifact=native_reference, locator='#/source_mapping'),
            dict(artifact=cpp_reference, locator='#'),
            dict(artifact=base_reference, locator='#/implementation_sha256')],
        qualification='NORMAL_COMPLETE_ADMITTED_SERIAL_CALLING_THREAD_SOURCE_NOT_OBSERVATION',
        conditions=['ORIGINAL_VALID_STABLE_INPUT_AND_EXCLUSIVE_STATE_CONTRACT',
            'NO_FOREIGN_CALLING_TID_AFFINITY_MUTATION_WITHIN_CLOSED_SEGMENT',
            'SELECTED_CPU_ONLINE_AND_EFFECTIVE_CPUSET_STABLE_WITHIN_CLOSED_SEGMENT'],
        backward_from_final_bound_guard=False, actual_cpu_observation=False,
        whole_scope_resource_demand_complete=False, costs_inferred=False, native_execution_authorized=False)
    read_artifact(native_reference); read_artifact(cpp_reference, json_document=False)
    return result


__all__ = ['SCHEMA', 'workspace_serial_source_contract']

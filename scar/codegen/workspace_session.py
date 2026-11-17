"""Append a single-owner explicit-step session to the admitted residency host.

Sessions retain private slots, never Python values or caller-loop authority.
Source effects and external-memory invalidation remain the producer/runtime
contract. Every admitted step executes the original transactional helper once.
"""
from hashlib import sha256
from pathlib import Path

from scar.ir.workspace_session import SESSION_PROTOCOL
from scar.optimization.resource_solver import semantic_sha256
from .workspace_shadow_residency import render_shadow_residency


PLAN_SCHEMA = 'scar.workspace-session-plan'
EXECUTION_SCHEMA = 'scar.workspace-session-execution-plan'
MAPPING_SCHEMA = 'scar.workspace-session-source-mapping'
_HERE = Path(__file__).resolve().parent


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _hash(raw):
    return sha256(raw).hexdigest()


def _same(left, right):
    return semantic_sha256(left) == semantic_sha256(right)


_SESSION_SOURCE = r'''

// Explicit dynamic session control. The entire residency source above,
// including the original ABI table and fixed-count adapter, is unchanged.
#include <atomic>
#include <thread>
#include <utility>

namespace {
struct ScarWorkspaceSession {
    // Declared first: metadata is destroyed before the reverse array-element
    // destruction of the original private unique_ptr payload slots.
    std::array<std::unique_ptr<double[]>, @SLOT_COUNT@> slots;
    std::array<scar_array_v1, @ARRAY_COUNT@> arrays{};
    std::array<scar_provider_v1, @PROVIDER_COUNT@> providers{};
    bool have_bindings = false;
    bool dirty = true;
    const std::thread::id owner = std::this_thread::get_id();
    std::atomic_flag busy = ATOMIC_FLAG_INIT;
};

class ScarSessionLease {
    ScarWorkspaceSession *session_ = nullptr;
public:
    explicit ScarSessionLease(ScarWorkspaceSession *session) noexcept {
        if (session && session->owner == std::this_thread::get_id() &&
            !session->busy.test_and_set(std::memory_order_acquire)) session_ = session;
    }
    ~ScarSessionLease() noexcept {
        if (session_) session_->busy.clear(std::memory_order_release);
    }
    explicit operator bool() const noexcept { return session_ != nullptr; }
    ScarSessionLease(const ScarSessionLease &) = delete;
    ScarSessionLease &operator=(const ScarSessionLease &) = delete;
};

bool scar_session_same_array(const scar_array_v1 &a, const scar_array_v1 &b) noexcept {
    return a.data == b.data && a.rank == b.rank &&
           a.shape[0] == b.shape[0] && a.shape[1] == b.shape[1] &&
           a.strides[0] == b.strides[0] && a.strides[1] == b.strides[1] &&
           a.lower_bound == b.lower_bound && a.upper_bound == b.upper_bound;
}

bool scar_session_same_bindings(const ScarWorkspaceSession &session,
    const scar_array_v1 *arrays, uint64_t array_count,
    const scar_provider_v1 *providers, uint64_t provider_count) noexcept {
    if (!session.have_bindings || array_count != session.arrays.size() ||
        provider_count != session.providers.size() || (array_count && !arrays) ||
        (provider_count && !providers)) return false;
    for (size_t i = 0; i < session.arrays.size(); ++i)
        if (!scar_session_same_array(session.arrays[i], arrays[i])) return false;
    for (size_t i = 0; i < session.providers.size(); ++i) {
        const auto &a = session.providers[i]; const auto &b = providers[i];
        if (a.loop != b.loop || a.context != b.context || a.auxdata != b.auxdata) return false;
    }
    return true;
}
}

extern "C" SCAR_EXPORT_V1 const char *scar_workspace_session_plan_sha256_v1() noexcept {
    return "@SESSION_SHA@";
}

extern "C" SCAR_EXPORT_V1 int32_t scar_workspace_session_create_v1(
    const scar_stateful_table_v1 *table, const scar_array_v1 *arrays,
    uint64_t array_count, const scar_scalar_v1 *scalars, uint64_t scalar_count,
    const scar_provider_v1 *providers, uint64_t provider_count, void **output) noexcept {
    if (!output) return 1;
    *output = nullptr;
    try {
@TABLE_GUARD@
@PREFLIGHT@
@ALLOCATION@
        std::unique_ptr<ScarWorkspaceSession> session(new (std::nothrow) ScarWorkspaceSession);
        if (!session) return 5;
        // Only unique_ptr metadata moves: no payload copy or second new[].
        session->slots = std::move(slots);
        *output = session.release();
        return 0;
    } catch (...) { return 5; }
}

extern "C" SCAR_EXPORT_V1 int32_t scar_workspace_session_step_v1(
    void *opaque, const scar_stateful_table_v1 *table, const scar_array_v1 *arrays,
    uint64_t array_count, const scar_scalar_v1 *scalars, uint64_t scalar_count,
    const scar_provider_v1 *providers, uint64_t provider_count,
    uint64_t *committed, uint64_t *used_residency) noexcept {
    if (committed) *committed = 0;
    if (used_residency) *used_residency = 0;
    auto *session = static_cast<ScarWorkspaceSession *>(opaque);
    if (!session) return 1;
    ScarSessionLease lease(session);
    // Wrong-owner/busy calls violate the explicit handle protocol. They do
    // not read payload/bindings or mutate non-atomic session state.
    if (!lease) return 9;
    const bool previously_clean = !session->dirty;
    session->dirty = true;
    if (!committed || !used_residency || committed == used_residency) return 1;
    try {
@TABLE_GUARD@
        const bool first_iteration = !previously_clean ||
            !scar_session_same_bindings(*session, arrays, array_count, providers, provider_count);
        const int32_t status = run_with_workspace(arrays, array_count, scalars,
            scalar_count, providers, provider_count, session->slots, first_iteration);
        if (status) return status;
        for (size_t i = 0; i < session->arrays.size(); ++i) session->arrays[i] = arrays[i];
        for (size_t i = 0; i < session->providers.size(); ++i) session->providers[i] = providers[i];
        session->have_bindings = true;
        session->dirty = false;
        *committed = 1;
        *used_residency = first_iteration ? 0 : 1;
        return 0;
    } catch (...) { session->dirty = true; return 1; }
}

extern "C" SCAR_EXPORT_V1 int32_t scar_workspace_session_invalidate_v1(void *opaque) noexcept {
    auto *session = static_cast<ScarWorkspaceSession *>(opaque);
    if (!session) return 1;
    ScarSessionLease lease(session);
    if (!lease) return 9;
    session->dirty = true;
    session->have_bindings = false;
    return 0;
}

extern "C" SCAR_EXPORT_V1 int32_t scar_workspace_session_destroy_v1(void **opaque) noexcept {
    if (!opaque) return 1;
    auto *session = static_cast<ScarWorkspaceSession *>(*opaque);
    if (!session) return 0;
    if (session->owner != std::this_thread::get_id() ||
        session->busy.test_and_set(std::memory_order_acquire)) return 9;
    *opaque = nullptr;
    // No lease destructor may access the handle after deletion.
    delete session;
    return 0;
}
'''


def render_workspace_session(original_source, metadata, residency_plan, session_plan):
    """Append only session control; full semantic/lease admission stays upstream."""
    _need(type(session_plan) is dict and session_plan.get('schema') == PLAN_SCHEMA
          and type(session_plan.get('schema_version')) is int and session_plan['schema_version'] == 1
          and session_plan.get('status') == 'PROVEN_CONDITIONAL',
          'an upstream-qualified conditional session plan is required')
    _need(session_plan['parent_native_manifest'] == residency_plan['native_blueprint']['manifest']
          and _same(session_plan['native_blueprint'], residency_plan['native_blueprint'])
          and session_plan['residency_plan_sha256'] == semantic_sha256(residency_plan)
          and type(session_plan['geometry']) is dict and bool(session_plan['geometry'])
          and type(session_plan['protocol']) is dict
          and all(session_plan['protocol'].get(key) == value for key, value in SESSION_PROTOCOL.items()),
          'session source, residency geometry or explicit dirty protocol differs')
    residency_source, residency_mapping = render_shadow_residency(original_source, metadata, residency_plan)
    text = original_source.decode('utf-8')
    ranges = metadata['source_mapping']['original_run_source_range']
    original_run = text[ranges['char_start']:ranges['char_end_exclusive']]
    _need(_hash(original_run.encode()) == metadata['source_mapping']['original_run_sha256'],
          'one exact original ABI step body required')
    signature_end = original_run.index(') {')+3
    declaration = '    std::array<std::unique_ptr<double[]>, '
    pool_start = original_run.index(declaration)
    pool_end = original_run.index('    ThreadAffinity caller_affinity;', pool_start)
    preflight, allocation = original_run[signature_end:pool_start], original_run[pool_start:pool_end]
    _need(_hash(preflight.encode()) == metadata['source_mapping']['preflight_sha256']
          and _hash(allocation.encode()) == metadata['source_mapping']['pool_block_sha256'],
          'original no-math preflight and allocation block differ')
    from scar.compiler.stateful_model import read_stateful
    import json
    directory = Path(session_plan['parent_native_manifest']['uri']).parent
    region = read_stateful(directory/metadata['files']['region'])
    step = json.loads((directory/metadata['files']['step_plan']).read_bytes())
    slot_count = len(step['workspace_slots'])
    array_count = sum(row.kind == 'array' for row in region.inputs)
    provider_count = len(region.providers)
    _need(declaration+str(slot_count)+'> slots;' in allocation,
          'session slot-array type differs from the source allocation')
    repeat = (_HERE/'stateful_repeat_v1.cpp.in').read_text()
    begin = '    if (!table || table->abi_version != 1 ||'
    end = '    for (uint64_t step = 0; step < repetitions; ++step) {'
    _need(repeat.count(begin) == repeat.count(end) == 1, 'one original parent-table guard required')
    guard = repeat[repeat.index(begin):repeat.index(end)]
    for key in ('region_sha256', 'plan_sha256', 'contract_sha256'):
        guard = guard.replace('@'+key.upper()+'@', metadata['base_ids'][key])
    _need('@' not in guard and _hash(guard.encode()) == metadata['source_mapping']['parent_table_check_sha256'],
          'session parent-table guard is not the retained original ABI check')
    execution_plan = dict(schema=EXECUTION_SCHEMA, schema_version=1,
        parent_execution_plan_sha256=metadata['execution_plan_sha256'],
        residency_plan_sha256=semantic_sha256(residency_plan), session_plan_sha256=semantic_sha256(session_plan))
    execution_sha = semantic_sha256(execution_plan)
    replacements = dict(SLOT_COUNT=str(slot_count), ARRAY_COUNT=str(array_count), PROVIDER_COUNT=str(provider_count),
        SESSION_SHA=execution_sha, TABLE_GUARD=guard.rstrip(), PREFLIGHT=preflight.rstrip(), ALLOCATION=allocation.rstrip())
    appended = _SESSION_SOURCE
    for name, value in replacements.items():
        _need(appended.count('@'+name+'@') == (2 if name == 'TABLE_GUARD' else 1),
              'session controller template marker count differs: '+name)
        appended = appended.replace('@'+name+'@', value)
    _need('@' not in appended, 'unresolved session controller template')
    source = residency_source+appended.encode()
    _need(source[:-len(appended.encode())] == residency_source,
          'removing session control must restore every residency-source byte')
    mapping = dict(schema=MAPPING_SCHEMA, schema_version=1, original_source_sha256=_hash(original_source),
        residency_source_sha256=_hash(residency_source), source_sha256=_hash(source),
        appended_source_sha256=_hash(appended.encode()), residency_mapping=residency_mapping,
        execution_plan=execution_plan, execution_plan_sha256=execution_sha,
        parent_execution_plan_sha256=metadata['execution_plan_sha256'],
        original_preflight_sha256=_hash(preflight.encode()), original_allocation_sha256=_hash(allocation.encode()),
        original_parent_table_guard_sha256=_hash(guard.encode()),
        source_pins={name:_hash((_HERE/name).read_bytes()) for name in
            ('workspace_session.py', 'workspace_shadow_residency.py', 'stateful_repeat_v1.cpp.in',
             'stateful_workspace.py', 'stateful_workspace_v1.cpp.in')},
        protocol_source=dict(path='scar/ir/workspace_session.py',
            sha256=_hash((_HERE.parent/'ir/workspace_session.py').read_bytes())),
        abi=dict(version=1, create='scar_workspace_session_create_v1', step='scar_workspace_session_step_v1',
            invalidate='scar_workspace_session_invalidate_v1', destroy='scar_workspace_session_destroy_v1',
            getter='scar_workspace_session_plan_sha256_v1', session_guard_status=9,
            opaque_handle='VALID_OWNER_PROCESS_HANDLE_REQUIRED', step_committed='UINT64_ZERO_OR_ONE',
            step_used_residency='UINT64_ONE_ONLY_AFTER_SUCCESSFUL_REUSE'),
        protocol=dict(SESSION_PROTOCOL), source_inverse_to_residency_byte_equal=True,
        original_run_table_fixed_repeat_and_residency_getter_unchanged=True,
        helper_math_checks_and_commits_unchanged=True, helper_called_once_per_step=True,
        step_scalars_read_fresh=True, create_has_no_math_or_payload_reads=True,
        create_original_preflight_before_payload_allocation=True, original_allocation_block_verbatim=True,
        local_to_session_transfer='UNIQUE_PTR_METADATA_MOVE_NO_PAYLOAD_COPY',
        session_metadata_allocation='ONE_ADDITIONAL_NOTHROW_NEW_NOT_ARRAY_PAYLOAD',
        destroy_slot_order='CPP_STD_ARRAY_REVERSE_ELEMENT_DESTRUCTION',
        zero_target_calls='CALLER_LAZY_PROTOCOL_CREATES_NO_SESSION',
        dynamic_call_count_fixed_N_authority=False, automatic_caller_replacement_authorized=False,
        semantic_session_admission_required_by_caller=True, complete_session_plan_replayed_here=False,
        trial_only=True, default_backend='NO_OP', deployment_authorized=False,
        numerical_backend_added=False, native_compiled=False, target_program_executed=False, performance_measured=False)
    return source, mapping


__all__ = ['render_workspace_session', 'SESSION_PROTOCOL', 'PLAN_SCHEMA', 'EXECUTION_SCHEMA', 'MAPPING_SCHEMA']

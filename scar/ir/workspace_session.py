"""Versioned dirty/commit protocol for a single-owner resident workspace.

This native primitive does not grant authority to replace an original caller.
Source effects and actual/formal binding must authorize invalidation events.
"""
from types import MappingProxyType

SESSION_PROTOCOL = MappingProxyType({
    'dynamic_call_count':'EXPLICIT_STEPS_NOT_FIXED_N_CALLER_REPLACEMENT',
    'single_owner_thread':True,
    'invalid_owner_or_busy':'REJECT_NO_PAYLOAD_OR_NONATOMIC_STATE_ACCESS',
    'create_original_preflight_before_allocation':True,
    'scalar_fresh_each_step':True,
    'array_provider_descriptor_change_forces_full':True,
    'dirty_on_admitted_failure':True,
    'commit_one_only_after_success':True,
    'unknown_effects_require_invalidate':True,
    'current_kernel_only_fallback':True,
    'owner_idle_destroy_reverse_release':True,
})

SESSION_SYMBOLS = MappingProxyType({
    'create':'scar_workspace_session_create_v1',
    'step':'scar_workspace_session_step_v1',
    'invalidate':'scar_workspace_session_invalidate_v1',
    'destroy':'scar_workspace_session_destroy_v1',
    'plan':'scar_workspace_session_plan_sha256_v1',
})

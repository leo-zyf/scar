"""Metadata-only cost admission for one exact automatic-runner invocation.

This module and its cost replay dependency use only the standard library. A
positive decision permits considering a load; the lightweight loader and source
installer still own admission, callable, input and live execution guards.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import re

from scar.optimization.scope_accounting import _digest, _profile, plan_deployment, verify_deployment_plan


_IDS = {'build_id', 'loop_sha256', 'strategy_sha256', 'execution_plan_sha256', 'contract_sha256'}
_ENTRY = {'kind', 'entry_path', 'entry_sha256', 'argv', 'runner_implementation_sha256'}
_ERRORS = (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError)
MEMORY_SCOPE = 'REGION_INCREMENTAL_FRESH_FINAL_PAYLOAD_EXCLUDES_EXTERNAL_INPUTS_AND_PROCESS_RSS'
STATEFUL_MEMORY_SCOPE = 'ALL_REGION_ARRAY_PAYLOAD_EXTERNAL_STATE_PLUS_FIXED_WORKSPACE'
_STATEFUL_IDS = {'build_id', 'region_sha256', 'plan_sha256', 'contract_sha256'}


def _hash(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def deployment_gate(report, *, bundle_ids, admission_sha256, entry_scope, reuse_count,
                    available_cpu_ids, memory_budget_bytes, profile=None,
                    required_cpu_ids=None, required_memory_bytes=None, memory_scope=None):
    """Replay cost evidence and current resources before loading any native code.

    ``bundle_ids`` are the profile's identities from runtime.json.native.ids; the
    admission hash is the caller's explicit producer pin. This function compares
    metadata identities, not the authenticity or bytes of a native library.
    Entry argv is exact, including output paths and all flags. No legacy/manual
    installation scope is promoted to an automatic-runner observation.
    Positive admission additionally requires the actual validated runtime
    contract's CPU order and payload scope, not profile self-declarations.
    Loop contracts count the fresh final payload; stateful contracts count
    external region arrays plus the fixed private workspace.
    """
    frontier, reasons = [], []
    def issue(field, reason): frontier.append(dict(field=field, reason=reason))
    value, source, current = None, None, None
    try:
        value, source = _profile(report)
        value = verify_deployment_plan(value)
    except _ERRORS as error:
        issue('deployment_plan', 'original cost replay failed: ' + str(error)); value = None
    stateful = isinstance(bundle_ids, dict) and set(bundle_ids) == _STATEFUL_IDS
    expected_scope = STATEFUL_MEMORY_SCOPE if stateful else MEMORY_SCOPE
    if (not isinstance(bundle_ids, dict) or set(bundle_ids) not in (_IDS, _STATEFUL_IDS)
            or not all(_hash(v) for v in bundle_ids.values())):
        issue('bundle_ids', 'complete exact native identity hashes required')
    if not _hash(admission_sha256): issue('admission_sha256', 'explicit producer admission SHA256 required')
    actual_cpus = list(required_cpu_ids) if isinstance(required_cpu_ids, (list, tuple)) else None
    if (not actual_cpus or any(type(cpu) is not int or cpu < 0 for cpu in actual_cpus)
            or len(set(actual_cpus)) != len(actual_cpus)):
        issue('required_cpu_ids', 'actual runtime contract CPU allocation and order required'); actual_cpus = None
    if type(required_memory_bytes) is not int or required_memory_bytes < 0:
        issue('required_memory_bytes', 'actual validated runtime payload bytes required')
    if memory_scope != expected_scope:
        issue('memory_scope', 'explicit payload scope matching the native identity profile required')
    if not isinstance(entry_scope, dict) or set(entry_scope) != _ENTRY:
        issue('entry_scope', 'complete automatic-runner entry and exact argv scope required')
    else:
        if entry_scope['kind'] != 'AUTOMATIC_RUNNER': issue('entry_scope.kind', 'manual installation costs cannot qualify an automatic runner')
        if not _hash(entry_scope['entry_sha256']) or not _hash(entry_scope['runner_implementation_sha256']):
            issue('entry_scope', 'entry and runner implementation SHA256 required')
        if (not isinstance(entry_scope['argv'], list) or
                any(type(arg) is not str for arg in entry_scope['argv'])):
            issue('entry_scope.argv', 'exact argv string list required; no flag normalization is inferred')
        try:
            path = Path(entry_scope['entry_path'])
            if type(entry_scope['entry_path']) is not str or not path.is_absolute():
                raise ValueError('absolute entry source path required')
            if sha256(path.read_bytes()).hexdigest() != entry_scope['entry_sha256']:
                raise ValueError('current entry bytes differ from the declared entry SHA256')
        except _ERRORS as error: issue('entry_scope.entry_path', str(error))
    context = value['context'] if value else {}
    if value:
        if context.get('boundary') != 'FRESH_PROCESS':
            issue('context.boundary', 'this automatic-runner gate requires an original FRESH_PROCESS scope')
        if context.get('launch_policy') != 'COST_GATED_DEPLOYMENT':
            issue('context.launch_policy', 'original costs must describe COST_GATED_DEPLOYMENT, not a measurement bypass')
        if actual_cpus is not None and context.get('cpu_ids') != actual_cpus:
            issue('context.cpu_ids', 'cost CPU allocation differs from the actual runtime contract')
        if type(required_memory_bytes) is int and required_memory_bytes >= 0 and context.get('required_memory_bytes') != required_memory_bytes:
            issue('context.required_memory_bytes', 'cost payload differs from the actual runtime array.final_bytes')
        if memory_scope == expected_scope and context.get('memory_scope') != memory_scope:
            issue('context.memory_scope', 'cost memory scope differs from the actual runtime payload contract')
        bundle = context.get('runtime_bundle')
        expected_bundle = dict(ids=bundle_ids, admission_sha256=admission_sha256)
        if not isinstance(bundle, dict) or set(bundle) != {'ids', 'admission_sha256'}:
            issue('context.runtime_bundle', 'original cost evidence lacks complete runtime bundle identities and admission pin')
        else:
            try:
                if _digest(bundle) != _digest(expected_bundle):
                    issue('context.runtime_bundle', 'original cost bundle identities/admission differ from the candidate')
            except _ERRORS as error: issue('context.runtime_bundle', str(error))
        original_entry = context.get('entry_scope')
        if not isinstance(original_entry, dict) or set(original_entry) != _ENTRY:
            issue('context.entry_scope', 'original evidence lacks an automatic-runner entry/argv/implementation scope')
        else:
            try:
                if _digest(original_entry) != _digest(entry_scope):
                    issue('context.entry_scope', 'original entry, argv or runner implementation scope differs')
            except _ERRORS as error: issue('context.entry_scope', str(error))
        if type(reuse_count) is not int or reuse_count != value['request']['reuse_count']:
            issue('reuse_count', 'actual reuse count must equal the cost plan invocation contract')
        if value['decision'] != 'LOAD_NATIVE_CANDIDATE' or value['selected'] is not True:
            reasons.append('original_deployment_cost_plan_declined')
        selected_profile = value['profile']
        if profile is not None:
            try:
                selected_profile, _ = _profile(profile)
                if _digest(selected_profile) != value['profile_sha256']:
                    raise ValueError('supplied profile differs from the replayed deployment plan')
            except _ERRORS as error: issue('profile', str(error)); selected_profile = value['profile']
        try:
            current = plan_deployment(selected_profile, context=context, reuse_count=reuse_count,
                available_cpu_ids=available_cpu_ids, memory_budget_bytes=memory_budget_bytes,
                minimum_gain=value['request']['minimum_gain'])
            frontier.extend(dict(field='current.' + row['field'], reason=row['reason']) for row in current['frontier'])
            reasons.extend(current['reasons'])
            if current['decision'] != 'LOAD_NATIVE_CANDIDATE': reasons.append('current_cost_or_resource_contract_declined')
        except _ERRORS as error: issue('current_resources', str(error))
    accepted = value is not None and current is not None and not frontier and not reasons
    return dict(schema='scar.deployment-gate', schema_version=1,
        decision='LOAD_NATIVE_CANDIDATE' if accepted else 'NO_OP', selected=accepted,
        status='CANDIDATE_UNDER_EXACT_SCOPE' if accepted else 'DECLINED',
        deployment_plan_sha256=None if value is None else _digest(value), deployment_plan_source=source,
        cost_profile_sha256=None if value is None else value['profile_sha256'],
        required_launch_policy='COST_GATED_DEPLOYMENT', source_launch_policy=context.get('launch_policy'),
        bundle_ids=deepcopy(bundle_ids), admission_sha256=admission_sha256, entry_scope=deepcopy(entry_scope),
        reuse_count=reuse_count,
        current_resources=None if current is None else dict(
            available_cpu_ids=current['request']['available_cpu_ids'], memory_budget_bytes=memory_budget_bytes,
            required_cpu_ids=actual_cpus, required_memory_bytes=required_memory_bytes, memory_scope=memory_scope,
            requirements_source='CALLER_VALIDATED_PINNED_RUNTIME_CONTRACT',
            budget_excludes=([] if stateful else ['EXTERNAL_INPUTS']) +
                ['ALLOCATOR_METADATA_AND_ALIGNMENT_OVERHEAD', 'THREAD_SCRATCH', 'PROCESS_RSS']),
        mode=None if value is None else value['mode'],
        cost_bases={} if value is None else deepcopy(value['cost_bases']),
        whole_scope_cost_observed=False if value is None else value['whole_scope_cost_observed'],
        model_assumptions=[] if value is None else deepcopy(value['model_assumptions']),
        external_assumptions=[] if value is None else deepcopy(value['external_assumptions']),
        runtime_guards=[] if value is None else deepcopy(value['runtime_guards']),
        frontier=frontier, reasons=list(dict.fromkeys(reasons)),
        selection_before_heavy_load=dict(metadata_only=True, requires_heavy_load=False,
            candidate_may_be_loaded=accepted, execution_contract_must_be_revalidated=True),
        bundle_bytes_and_admission_not_authenticated_here=True,
        native_library_loaded=False, target_executed=False, deployment_authorized=False)


def verify_deployment_gate(report, result, **actual_context):
    """Replay against caller-provided actual identities and resource conditions."""
    value, _ = _profile(result)
    expected = deployment_gate(report, **actual_context)
    if _digest(value) != _digest(expected):
        raise ValueError('deployment gate differs from exact current scope/resource replay')
    return value


__all__ = ['MEMORY_SCOPE', 'deployment_gate', 'verify_deployment_gate']

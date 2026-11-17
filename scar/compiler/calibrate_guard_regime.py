"""Paired trial guard configurations over one retained canonical scope probe.

Only the existing compiled empty-DAG probe is loaded.  No target module/DAG is
executed and no compiler is invoked.  Per-pair differences describe this
canonical configuration experiment; they are not intrinsic component costs or
an attribution of any earlier whole-method prediction error.
"""
import argparse
import ast
from copy import deepcopy
import ctypes
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
import types

from scar.compiler import calibrate_scope_boundary as scope
from scar.compiler.calibrate_stateful_boundary import _json, _now, _ordinary_node
from scar.compiler.stateful_model import stateful_from_dict
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.stateful_contract import validate_contract
from scar.trace.code_snapshot import code_key


SCHEMA = 'scar.stateful-guard-regime-calibration'
_RESULT = 'scar.stateful-guard-regime-result'
_REGIMES = ('trial_false', 'trial_true')
_RUNTIME_PATH = ['scar.runtime.stateful_guard.guarded_stateful_function.call',
    'scar.compiler.stateful_runtime._ArtifactAdapter.run', 'scar.codegen.stateful.StatefulArtifact.run']
_CONDITIONS = [
    'ONE_RETAINED_CANONICAL_EMPTY_DAG_LIBRARY_AND_ORIGINAL_STATEFUL_TABLE_ABI',
    'SAME_ARGUMENT_OBJECTS_PROVIDER_BINDING_ADAPTER_BODY_AND_UNMODIFIED_SELECTION_FOR_BOTH_REGIMES',
    'TWO_PERSISTENT_WRAPPERS_FROM_THE_SAME_FACTORY_ONLY_TRIAL_DIFFERS',
    'ALTERNATING_FALSE_TRUE_AND_TRUE_FALSE_PAIRS_WITH_IDENTICAL_CLOCK_AND_CUTS',
    'NATIVE_SERIAL_CPU_BIND_AND_RESTORE_RETAINED_NO_COLLECTOR_CPU_PIN_OR_ENVIRONMENT_CHANGE',
    'CLOCK_READ_AND_CUT_STORAGE_OVERHEAD_RETAINED_NOT_SUBTRACTED',
    'PAIRED_CONFIGURATION_DIFFERENCE_NOT_A_WHOLE_TARGET_OR_COMPONENT_COST_PROFILE',
]


def _need(value, message):
    if not value:
        raise ValueError(message)


def _reference(value, schema):
    if isinstance(value, (str, Path)):
        value = artifact_reference(Path(value).resolve(), schema)
    read_artifact(value)
    _need(value['schema'] == schema, 'exact retained artifact schema required')
    return deepcopy(value)


def _receipt_reference(receipt, schema):
    result = artifact_reference(receipt['path'], schema)
    _need((result['sha256'], result['size_bytes']) == (receipt['sha256'], receipt['bytes']),
        'retained canonical component byte identity differs')
    return result


def _identity(fn):
    path = Path(inspect.getsourcefile(fn)).resolve()
    return dict(source=artifact_reference(path, 'scar.python-source'), module=fn.__module__,
        qualname=fn.__qualname__, code_qualname=fn.__code__.co_qualname, code_key=code_key(fn.__code__))


def _binding(contract, retained_source):
    """Reinstantiate the retained synthetic body without rewriting its file.

    The old runtime contract/selection stays byte-identical.  Only ordinary
    receiver/input objects are fresh.  Every retained function is a literal
    raising canonical placeholder, and no placeholder body is called.
    """
    import numpy as np
    from scar.runtime.stateful_guard import _function
    raw = read_artifact(retained_source, json_document=False)
    tree = ast.parse(raw, filename=retained_source['uri'])
    _need(tree.body and all(isinstance(fn, ast.FunctionDef) and not fn.decorator_list
        and not fn.args.defaults and not any(fn.args.kw_defaults)
        and len(fn.body) == 1 and isinstance(fn.body[0], ast.Raise)
        and fn.body[0].cause is None and isinstance(fn.body[0].exc, ast.Call)
        and isinstance(fn.body[0].exc.func, ast.Name) and fn.body[0].exc.func.id == 'RuntimeError'
        and not fn.body[0].exc.keywords and len(fn.body[0].exc.args) == 1
        and isinstance(fn.body[0].exc.args[0], ast.Constant)
        and type(fn.body[0].exc.args[0].value) is str for fn in tree.body),
        'only retained literal-raising canonical function definitions may be instantiated')
    _need(all(not fn.returns and all(not arg.annotation for arg in
        [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs])
        and not fn.args.vararg and not fn.args.kwarg for fn in tree.body),
        'canonical definitions must have no executable annotations or variadic parameters')
    namespace = dict(__name__='scar_canonical_boundary_probe')
    exec(compile(raw, retained_source['uri'], 'exec'), namespace)
    original = namespace[contract['source']['qualname']]
    _function(contract['source'], original)
    _need(tuple(inspect.signature(original).parameters) == tuple(contract['parameters']),
        'retained canonical function parameters differ')
    counter = [0]
    bound = {name: _ordinary_node(counter) for name in contract['parameters']}
    arrays, scalar_values = {}, {}
    for row in contract['inputs']:
        if row['kind'] == 'array':
            value = np.full(row['shape'], 1.25, dtype=np.float64)
            value.flags.writeable = row['written']
            arrays[row['id']] = value
        else:
            value = 0.75
            scalar_values[row['id']] = value
        parts = row['path']
        if len(parts) == 1:
            bound[parts[0]] = value
        else:
            current = bound[parts[0]]
            for name in parts[1:-1]:
                if name not in vars(current):
                    setattr(current, name, _ordinary_node(counter))
                current = vars(current)[name]
            _need(parts[-1] not in vars(current), 'canonical input paths conflict')
            setattr(current, parts[-1], value)

    def bind_lookup(parts, value):
        mapping = bound if parts[0] in bound else namespace
        if len(parts) == 1:
            _need(parts[0] not in mapping, 'canonical lookup conflicts with an input')
            mapping[parts[0]] = value
            return
        if parts[0] not in mapping:
            mapping[parts[0]] = types.ModuleType('CanonicalGlobals')
        current = mapping[parts[0]]
        for name in parts[1:-1]:
            _need(type(current) is types.ModuleType, 'only ordinary intermediate module lookup is supported')
            if name not in vars(current):
                setattr(current, name, types.ModuleType('CanonicalGlobals'))
            current = vars(current)[name]
        if type(current) is types.ModuleType:
            _need(parts[-1] not in vars(current), 'canonical global lookup conflicts')
            setattr(current, parts[-1], value)
        else:
            _need(parts[-1] not in vars(current) and parts[-1] not in vars(type(current)),
                'canonical staticmethod lookup conflicts')
            setattr(type(current), parts[-1], staticmethod(value))

    for row in contract['callee_guards']:
        _need(row['source']['path'] == retained_source['uri'] and row['source']['sha256'] == retained_source['sha256'],
            'canonical callee must belong to the retained placeholder source')
        fn = namespace[row['source']['qualname']]
        _function(row['source'], fn)
        bind_lookup(row['path'], fn)
    for row in contract['providers']:
        _need(row['module'] == 'numpy' and row['symbol'] == 'power', 'only the original NumPy power binding is admitted')
        bind_lookup(row['lookup_path'], np.power)
    return original, tuple(bound[name] for name in contract['parameters']), arrays, scalar_values


def _input_digests(arrays):
    return {key: sha256(memoryview(value).cast('B')).hexdigest() for key, value in sorted(arrays.items())}


def _summary(values):
    return dict(samples_ns=values, median_ns=statistics.median(values), min_ns=min(values), max_ns=max(values),
        stdev_ns=statistics.stdev(values))


def guard_regime_statistics(document):
    """Replay paired cut algebra; subtract within a pair, never scope medians."""
    _need((document.get('schema'), document.get('schema_version')) == (SCHEMA, 1), 'versioned guard-regime record required')
    _need(document.get('cuts') == list(scope.CUTS) and document.get('phases') == list(scope.PHASES)
        and document.get('clock') == dict(api='clock_gettime', clock_id='CLOCK_MONOTONIC', unit='ns')
        and document.get('conditions') == _CONDITIONS, 'original scope cuts/units and applicability conditions required')
    pairs, warmup = document.get('pairs'), document.get('warmup_pairs')
    _need(type(pairs) is int and 3 <= pairs <= 1000 and type(warmup) is int and 1 <= warmup <= 1000,
        'bounded warm paired canonical calls required')
    measured = document.get('samples')
    warm = document.get('warmup_samples')
    _need(type(measured) is list and len(measured) == pairs and type(warm) is list and len(warm) == warmup,
        'complete measured and warmup pair vectors required')
    columns = {key: [[] for _ in range(len(scope.PHASES)+1)] for key in _REGIMES}
    mask = document['caller_affinity_before']
    for phase, rows in (('warmup', warm), ('measurement', measured)):
        for index, pair in enumerate(rows):
            order = list(_REGIMES if index % 2 == 0 else reversed(_REGIMES))
            _need(pair.get('id') == phase + ':' + str(index) and pair.get('order') == order
                and set(pair.get('calls', {})) == set(_REGIMES), 'complete alternating canonical pair identity/order required')
            for key in _REGIMES:
                call = pair['calls'][key]
                stamps = call.get('timestamps_ns')
                _need(call.get('trial') is (key == 'trial_true') and type(stamps) is list
                    and len(stamps) == len(scope.CUTS) and all(type(n) is int and n > 0 for n in stamps)
                    and all(left <= right for left, right in zip(stamps, stamps[1:]))
                    and call.get('caller_affinity_before') == mask == call.get('caller_affinity_after')
                    and call.get('native_call_increment') == 1 and call.get('fallback_call_increment') == 0,
                    'explicit trial, monotone original cuts, restored mask and successful native call required')
                deltas = [right-left for left, right in zip(stamps, stamps[1:])]
                _need(sum(deltas) == stamps[-1]-stamps[0], 'exclusive cuts must partition each individual call')
                if phase == 'measurement':
                    for column, value in zip(columns[key], [*deltas, stamps[-1]-stamps[0]]):
                        column.append(value)
    result = []
    for index, name in enumerate([row['id'] for row in scope.PHASES] + ['whole_canonical_call']):
        false, true = columns['trial_false'][index], columns['trial_true'][index]
        result.append(dict(phase=name, unit='ns', trial_false=_summary(false), trial_true=_summary(true),
            paired_false_minus_true=_summary([left-right for left, right in zip(false, true)]),
            interpretation='PAIRED_CANONICAL_CONFIGURATION_DIFFERENCE_NOT_INTRINSIC_COMPONENT_COST'))
    return result


def calibrate_guard_regime(scope_observations, out, *, domain, candidate_id, pairs=7, warmup_pairs=2):
    """Run one small paired experiment, reusing the original compiled library."""
    from scar.codegen.stateful import StatefulArtifact, _Table, _providers
    from scar.compiler.stateful_runtime import _ArtifactAdapter
    from scar.runtime.compiled_deployment import thread_environment
    from scar.runtime.stateful_guard import guarded_stateful_function
    _need(type(pairs) is int and 3 <= pairs <= 1000 and type(warmup_pairs) is int and 1 <= warmup_pairs <= 1000,
        'bounded warm paired canonical calls required')
    _need(sys.gettrace() is None and sys.getprofile() is None and threading.active_count() == 1,
        'canonical pairing requires one Python thread and no trace/profile hook')
    _need(not any(sys.monitoring.get_events(tool) for tool in range(6)) if hasattr(sys, 'monitoring') else True,
        'canonical pairing requires no active monitoring events')
    scope_ref = _reference(scope_observations, scope.SCHEMA)
    domain_ref = _reference(domain, 'scar.stateful-cpu-domain')
    old = scope.read_scope_boundary_calibration(scope_ref, domain=domain_ref['uri'], candidate_id=candidate_id)
    _need(old['whole_target_call_measured'] is False and old['target_variants_compiled'] is False,
        'a retained canonical non-target scope probe is required')
    contract_ref = _receipt_reference(old['canonical_runtime_contract'], 'scar.stateful-runtime-contract')
    contract = validate_contract(read_artifact(contract_ref))
    source_ref = _receipt_reference(old['canonical_structure']['canonical_source'], 'scar.python-source')
    _need(contract['source']['path'] == source_ref['uri'] and contract['source']['sha256'] == source_ref['sha256'],
        'original canonical contract and retained function source must agree')
    library_ref = _receipt_reference(old['probe']['library'], 'scar.native-library')
    probe_source_ref = _receipt_reference(old['probe']['source'], 'scar.cpp-source')
    region_ref = old['domain_source']['region']; plan_ref = old['domain_source']['stage_plan']
    region = stateful_from_dict(read_artifact(region_ref)); plan = read_artifact(plan_ref)
    caller = sorted(os.sched_getaffinity(0)); environment = thread_environment()
    _need(caller == old['observed_caller_affinity_before']
        and environment == old['observed_native_thread_environment'] and set(contract['cpu']['ids']) <= set(caller),
        'caller mask and native thread environment must match the retained canonical context')
    output = Path(out).resolve(); output.mkdir(parents=True, exist_ok=False)
    started, observed_from = time.monotonic(), _now()
    original, args, arrays, scalar_values = _binding(contract, source_ref)
    runtime, providers, owners, power = _providers(region)
    _need(runtime == contract['provider_runtime'], 'canonical provider binary/ABI differs')
    library = ctypes.CDLL(library_ref['uri'])
    getter = library.scar_get_stateful_table_v1; getter.argtypes = []; getter.restype = ctypes.POINTER(_Table)
    table = getter()
    _need(table and table.contents.abi_version == 1 and table.contents.struct_size == ctypes.sizeof(_Table)
        and table.contents.run, 'retained stateful table ABI differs')
    _need((table.contents.array_count, table.contents.scalar_count, table.contents.provider_count) ==
        (len(arrays), len(scalar_values), len(contract['providers'])), 'retained canonical ABI input counts differ')
    for key in ('region_sha256', 'plan_sha256', 'contract_sha256'):
        _need(getattr(table.contents, key) == contract['native']['ids'][key].encode(), 'retained canonical table identity differs')
    artifact = StatefulArtifact(region, plan, Path(library_ref['uri']).parent, dict(contract['native']['ids']),
        library, table, providers, owners, power)
    adapter = object.__new__(_ArtifactAdapter)
    adapter.original, adapter.contract, adapter.report = artifact, contract, artifact.report
    wrappers, configurations = {}, {}
    for key, trial in (('trial_false', False), ('trial_true', True)):
        wrapper = guarded_stateful_function(original, adapter, owner=type(args[0]),
            exclusive_inputs=True, selection=contract['selection'], trial=trial)
        closed = dict(zip(wrapper.__code__.co_freevars, (cell.cell_contents for cell in wrapper.__closure__)))
        _need(closed['trial'] is trial and closed['artifact'] is adapter and closed['original'] is original
            and closed['selection'] is contract['selection'] and closed['exclusive_inputs'] is True,
            'actual persistent wrapper closure differs from explicit paired configuration')
        wrappers[key] = wrapper
        configurations[key] = dict(trial=trial, observed_wrapper_closure_trial=closed['trial'],
            wrapper=_identity(wrapper), same_original_adapter_and_selection_object_verified=True)
    _need(wrappers['trial_false'].__code__ is wrappers['trial_true'].__code__, 'both regimes must use the identical wrapper code object')
    reader = library.scar_boundary_cut_read_v1
    reader.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64]; reader.restype = ctypes.c_int32
    native_cuts = (ctypes.c_uint64*7)()
    before_inputs = _input_digests(arrays)
    before_contract = _json(contract)
    loads_before = list(os.getloadavg())
    emergency_restore = False

    def take_sample(key):
        wrapper = wrappers[key]
        mask_before = sorted(os.sched_getaffinity(0))
        _need(mask_before == caller, 'caller mask changed before canonical call')
        stats = deepcopy(wrapper.scar_stats)
        enter = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        result = wrapper(*args)
        returned = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        mask_after = sorted(os.sched_getaffinity(0))
        _need(result is None and reader(native_cuts, 7) == 0 and mask_after == caller,
            'canonical native call/cut collection or affinity restoration failed')
        return dict(trial=configurations[key]['trial'], timestamps_ns=[enter, *native_cuts, returned],
            caller_affinity_before=mask_before, caller_affinity_after=mask_after,
            native_call_increment=wrapper.scar_stats['native_calls']-stats['native_calls'],
            fallback_call_increment=wrapper.scar_stats['fallback_calls']-stats['fallback_calls'])

    warm, measured = [], []
    try:
        for phase, count, destination in (('warmup', warmup_pairs, warm), ('measurement', pairs, measured)):
            for index in range(count):
                order = list(_REGIMES if index % 2 == 0 else reversed(_REGIMES))
                calls = {key: take_sample(key) for key in order}
                destination.append(dict(id=phase+':'+str(index), order=order, calls=calls))
    finally:
        after = sorted(os.sched_getaffinity(0))
        if after != caller:
            # Restore the saved complete mask after a failing probe; never pin
            # the collector to the probe's selected core for sampling.
            os.sched_setaffinity(0, caller)
            emergency_restore = True
    after_inputs = _input_digests(arrays)
    _need(after_inputs == before_inputs and _json(contract) == before_contract,
        'shared canonical input bytes or contract/selection changed')
    _need(after == caller and not emergency_restore and thread_environment() == environment,
        'successful canonical sampling must preserve the original mask and thread environment without collector intervention')
    for wrapper in wrappers.values():
        _need(wrapper.scar_stats['native_calls'] == pairs+warmup_pairs and wrapper.scar_stats['fallback_calls'] == 0,
            'both regimes must traverse native for every warmup/measured call')
    pins = dict(collector=_identity(calibrate_guard_regime), guard_factory=_identity(guarded_stateful_function),
        compiler_adapter_run=_identity(_ArtifactAdapter.run), artifact_run=_identity(StatefulArtifact.run),
        canonical_original=_identity(original))
    document = dict(schema=SCHEMA, schema_version=1, domain_id=old['domain_id'], candidate_id=candidate_id,
        source=dict(scope_observations=scope_ref, domain=domain_ref, canonical_contract=contract_ref,
            canonical_source=source_ref, library=library_ref, probe_source=probe_source_ref,
            region=region_ref, stage_plan=plan_ref), resource_context=deepcopy(old['resource_context']),
        function_identities=pins, guard_regimes=configurations, runtime_path=list(_RUNTIME_PATH),
        shared_path=dict(same_wrapper_code_object=True, same_argument_objects=True, same_native_table=True,
            canonical_body='RETAINED_SOURCE_DERIVED_EMPTY_DAG_SCOPE_PROBE',
            input_shapes={key:list(value.shape) for key,value in arrays.items()}, scalar_values=scalar_values,
            input_sha256_before=before_inputs, input_sha256_after=after_inputs,
            contract_and_selection_unchanged=True, selection_evidence=contract_ref,
            selection_is_measured_target_cost_profile=False, selection_checks_bypassed_for_false=False),
        cuts=list(scope.CUTS), phases=deepcopy(list(scope.PHASES)),
        clock=dict(api='clock_gettime', clock_id='CLOCK_MONOTONIC', unit='ns'),
        pairs=pairs, warmup_pairs=warmup_pairs, samples=measured, warmup_samples=warm,
        runtime_stats={key:deepcopy(value.scar_stats) for key,value in wrappers.items()},
        caller_affinity_before=caller, caller_affinity_after=after, caller_mask_preserved=True,
        collector_affinity_changed=False, collector_emergency_mask_restoration_used=emergency_restore,
        internal_native_cpu_ids=contract['cpu']['ids'], native_thread_environment=environment,
        thread_environment_changed=False, conditions=list(_CONDITIONS),
        retained_scope_missing_terms=deepcopy(old['missing_terms']),
        observed_from=observed_from, observed_until=_now(), seconds=time.monotonic()-started,
        host_load=dict(before=loads_before, after=list(os.getloadavg()), isolated=False),
        source_function_identity_not_basename=True, input_content_reads_inside_timed_span_by_collector=0,
        exclusive_partition_verified=True, clock_overhead_subtracted=False,
        old_holdout_used_for_fit_or_residual_attribution=False, residual_attribution_ns=None,
        component_profile_qualified=False, whole_target_call_measured=False,
        target_program_executed=False, native_compiled=False, existing_canonical_library_loaded=True,
        model_or_policy_updated=False, physical_resource_qualification='SEPARATE_NOT_EVALUATED',
        deployment_authorized=False)
    document['paired_statistics'] = guard_regime_statistics(document)
    path = output/'observations.json'; path.write_bytes(_json(document))
    observations_ref = artifact_reference(path, SCHEMA)
    read_guard_regime_calibration(observations_ref, domain=domain_ref, candidate_id=candidate_id)
    result = dict(schema=_RESULT, schema_version=1, observations=observations_ref,
        paired_statistics=document['paired_statistics'], pairs=pairs, warmup_pairs=warmup_pairs,
        caller_mask_preserved=True, component_profile_qualified=False, whole_target_call_measured=False,
        target_program_executed=False, native_compiled=False, deployment_authorized=False)
    (output/'result.json').write_bytes(_json(result))
    return result


def read_guard_regime_calibration(reference, *, domain, candidate_id):
    """Cold metadata/source/cut replay; no library is loaded or invoked here."""
    from scar.codegen.stateful import StatefulArtifact
    from scar.compiler.stateful_runtime import _ArtifactAdapter
    from scar.runtime.stateful_guard import guarded_stateful_function
    reference = _reference(reference, SCHEMA)
    document = read_artifact(reference)
    domain_ref = _reference(domain, 'scar.stateful-cpu-domain')
    source = document['source']
    _need(domain_ref == source['domain'], 'paired domain artifact differs')
    old = scope.read_scope_boundary_calibration(source['scope_observations'], domain=domain_ref['uri'], candidate_id=candidate_id)
    _need(document['domain_id'] == old['domain_id'] and document['candidate_id'] == candidate_id
        and document['resource_context'] == old['resource_context'] and document['runtime_path'] == _RUNTIME_PATH
        and document['retained_scope_missing_terms'] == old['missing_terms'], 'paired scope/resource/source context differs')
    for ref in source.values():
        read_artifact(ref, json_document=False)
    _need(source['library'] == _receipt_reference(old['probe']['library'], 'scar.native-library')
        and source['probe_source'] == _receipt_reference(old['probe']['source'], 'scar.cpp-source')
        and source['canonical_contract'] == _receipt_reference(old['canonical_runtime_contract'], 'scar.stateful-runtime-contract')
        and source['canonical_source'] == _receipt_reference(old['canonical_structure']['canonical_source'], 'scar.python-source')
        and source['region'] == old['domain_source']['region'] and source['stage_plan'] == old['domain_source']['stage_plan'],
        'paired experiment must retain exactly the old canonical source/body/library/contract')
    contract = validate_contract(read_artifact(source['canonical_contract']))
    _need(set(document['guard_regimes']) == set(_REGIMES), 'both explicit trial regimes required')
    for key in _REGIMES:
        regime = document['guard_regimes'][key]
        _need(regime['trial'] is (key == 'trial_true') and regime['observed_wrapper_closure_trial'] is regime['trial']
            and regime['same_original_adapter_and_selection_object_verified'] is True,
            'actual passed and closure trial values must agree')
    _need(document['guard_regimes']['trial_false']['wrapper'] == document['guard_regimes']['trial_true']['wrapper'],
        'wrapper source/function identity must be the same for both regimes')
    for identity in document['function_identities'].values():
        read_artifact(identity['source'], json_document=False)
    for name, fn in (('collector', calibrate_guard_regime), ('guard_factory', guarded_stateful_function),
        ('compiler_adapter_run', _ArtifactAdapter.run), ('artifact_run', StatefulArtifact.run)):
        _need(document['function_identities'][name] == _identity(fn),
            'paired collector/runtime function identity differs: ' + name)
    nested = [value for value in guarded_stateful_function.__code__.co_consts
        if type(value) is types.CodeType and value.co_name == 'call']
    _need(len(nested) == 1, 'one original returned guard code object required')
    expected_wrapper = dict(source=document['function_identities']['guard_factory']['source'],
        module='scar_canonical_boundary_probe', qualname=contract['source']['qualname'],
        code_qualname=nested[0].co_qualname, code_key=code_key(nested[0]))
    _need(document['guard_regimes']['trial_false']['wrapper'] == expected_wrapper,
        'paired wrapper identity differs from the retained guard code and canonical original')
    _need(document['function_identities']['canonical_original'] == dict(source=source['canonical_source'],
        module='scar_canonical_boundary_probe', qualname=contract['source']['qualname'],
        code_qualname=contract['source']['qualname'], code_key=contract['source']['code_key']),
        'canonical original function identity differs from its unchanged contract')
    shared = document['shared_path']
    _need(shared['input_shapes'] == {row['id']: row['shape'] for row in contract['inputs'] if row['kind'] == 'array'}
        and shared['input_sha256_before'] == shared['input_sha256_after']
        and shared['selection_evidence'] == source['canonical_contract']
        and shared['selection_is_measured_target_cost_profile'] is False
        and shared['selection_checks_bypassed_for_false'] is False
        and all(shared[key] is True for key in ('same_wrapper_code_object', 'same_argument_objects',
            'same_native_table', 'contract_and_selection_unchanged')), 'paired common-path evidence differs')
    _need(document['paired_statistics'] == guard_regime_statistics(document), 'paired statistics differ from raw per-call cuts')
    expected = document['pairs']+document['warmup_pairs']
    _need(all(document['runtime_stats'][key] == dict(native_calls=expected, fallback_calls=0, reasons={}) for key in _REGIMES),
        'all canonical warmup/measured calls must have used native successfully')
    _need(document['caller_affinity_before'] == document['caller_affinity_after'] == old['observed_caller_affinity_before']
        and document['internal_native_cpu_ids'] == contract['cpu']['ids']
        and document['native_thread_environment'] == old['observed_native_thread_environment'],
        'paired saved/restored caller mask, internal CPU binding or thread context differs')
    for key in ('caller_mask_preserved', 'exclusive_partition_verified', 'existing_canonical_library_loaded'):
        _need(document.get(key) is True, 'paired evidence requires ' + key)
    for key in ('collector_affinity_changed', 'collector_emergency_mask_restoration_used', 'thread_environment_changed',
        'clock_overhead_subtracted', 'old_holdout_used_for_fit_or_residual_attribution', 'component_profile_qualified',
        'whole_target_call_measured', 'target_program_executed', 'native_compiled', 'model_or_policy_updated', 'deployment_authorized'):
        _need(document.get(key) is False, 'canonical pairing cannot claim ' + key)
    _need(document['residual_attribution_ns'] is None, 'canonical configuration comparison cannot attribute old residuals')
    read_artifact(reference)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope-observations')
    parser.add_argument('--domain', required=True)
    parser.add_argument('--candidate', required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument('--out')
    destination.add_argument('--check')
    parser.add_argument('--pairs', type=int, default=7)
    parser.add_argument('--warmup-pairs', type=int, default=2)
    args = parser.parse_args(argv)
    if args.check:
        document = read_guard_regime_calibration(args.check, domain=args.domain, candidate_id=args.candidate)
        result = dict(metadata_replayed=True, paired_statistics=document['paired_statistics'],
            native_loaded=False, target_program_executed=False)
    else:
        _need(args.scope_observations is not None, '--scope-observations required for paired sampling')
        result = calibrate_guard_regime(args.scope_observations, args.out, domain=args.domain,
            candidate_id=args.candidate, pairs=args.pairs, warmup_pairs=args.warmup_pairs)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

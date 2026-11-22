"""Source-pinned entry template for the compiler StatefulArtifact facade.

The protocol describes a call path, not F4 admission, a measured selection, or
cost qualification. No target module, generated ELF or NumPy loop is loaded.
"""
import ast
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import CodeType, SimpleNamespace
import sys

from scar.ir.execution_recipe import artifact_reference, read_artifact, _reference
from scar.optimization.resource_solver import semantic_sha256
from scar.trace.code_snapshot import code_key

SCHEMA = 'scar.stateful-entry-protocol'
FAMILY = 'COMPILER_STATEFUL_ARTIFACT_ADAPTER_GUARD_V1'
TIMER = dict(clock='perf_counter_ns', span='SELECTED_CALLABLE_DISPATCH_THROUGH_RETURN',
    includes=['PYTHON_CALL_DISPATCH','IMPLEMENTATION_GUARDS','ABI_PACKING',
        'IMPLEMENTATION_ALLOCATIONS','NATIVE_WORK','RETURN'],
    excludes=['CALLER_ATTRIBUTE_LOOKUP','ALREADY_EVALUATED_ARGUMENT_EXPRESSIONS',
        'INPUT_METADATA_OBSERVATION','POST_ENTRY_STATE_AND_FILE_COMPARISONS',
        'ORIGINAL_CALLER_CONTROL_AND_IO','IMPORT_LOAD_INSTALL_ONE_TIME_ADMISSION'])


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _source(source):
    reference = artifact_reference(source.path, 'scar.python-source')
    _need(reference['sha256'] == source.sha256, 'original stateful source bytes changed')
    return dict(artifact=reference, qualname=source.qualname, code_key=source.code_key,
        line=source.line, column=source.column)


def _owner(source):
    """Select the installer's owner branch without importing original code."""
    identity = _source(source); parts = source.qualname.split('.')
    _need(parts and all(part.isidentifier() for part in parts), 'closures and nonordinary source paths are unsupported')
    body = ast.parse(read_artifact(identity['artifact'], json_document=False), filename=source.path).body
    classes = []
    for part in parts[:-1]:
        matches = [node for node in body if isinstance(node, ast.ClassDef) and node.name == part]
        _need(len(matches) == 1, 'source owner must be a unique ordinary lexical class')
        cls = matches[0]
        _need(not cls.decorator_list and not any(k.arg == 'metaclass' for k in cls.keywords),
            'decorated classes and explicit metaclasses require another entry protocol')
        classes.append(cls); body = cls.body
    matches = [node for node in body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == parts[-1]]
    _need(len(matches) == 1 and type(matches[0]) is ast.FunctionDef, 'one ordinary synchronous source function required')
    node = matches[0]; decorators = node.decorator_list
    _need(not decorators or (len(decorators) == 1 and isinstance(decorators[0], ast.Name)
        and decorators[0].id == 'staticmethod' and classes), 'classmethods and other function decorators are unsupported')
    first = min([node.lineno, *(item.lineno for item in decorators)])
    _need(first <= source.line <= node.lineno, 'source callable location differs from its lexical declaration')
    instance = bool(classes and not decorators)
    _need(not instance or node.args.posonlyargs or node.args.args, 'instance method requires an explicit receiver parameter')
    return dict(kind='INSTANCE_METHOD' if instance else 'STATICMETHOD' if classes else 'MODULE_FUNCTION',
        policy='EXACT_RECEIVER_CLASS_PER_INSTALLATION' if instance else 'NO_RECEIVER_CLASS_CHECK',
        factory_owner_argument='PER_INSTALLATION_ORDINARY_SOURCE_OWNER_CLASS' if instance else None,
        symbolic_source_owner=None if not classes else dict(artifact=identity['artifact'], qualname='.'.join(parts[:-1])),
        physical_class_identity_serialized=False, cross_installation_class_identity_asserted=False,
        original_descriptor_resolution_required_at_installation=True,
        ordinary_receiver_protocol_still_runtime_guarded=True)


def _implementations():
    from scar.codegen import stateful, cpp
    from scar.compiler import stateful_runtime as stateful_runtime, stateful_runtime as export_stateful, stateful_model
    from scar.runtime import stateful_guard, stateful_contract
    from scar.trace import code_snapshot as code_snapshot
    from scar.optimization.selection import select_stateful_measured
    from scar.optimization.stateful_schedule import verify_stateful_plan
    functions = dict(compiler_guard_factory=stateful_runtime.guarded_stateful_function,
        compiler_install_generator=stateful_runtime.install_stateful_regions.__wrapped__,
        shared_install_generator=stateful_guard.install_stateful_regions.__wrapped__,
        shared_guard_factory=stateful_guard.guarded_stateful_function,
        adapter_init=stateful_runtime._ArtifactAdapter.__init__, adapter_run=stateful_runtime._ArtifactAdapter.run,
        artifact_run=stateful.StatefulArtifact.run, artifact_report=stateful.StatefulArtifact.report.fget,
        artifact_load_factory=stateful.load, contract_projection=export_stateful.project_stateful_contract,
        contract_validation=stateful_contract.validate_contract, region_validation=stateful_model.validate_stateful_region,
        descriptor=cpp._descriptor, code_identity=code_snapshot.code_key,
        measured_selection=select_stateful_measured, selection_receipt_replay=export_stateful._replay_selection,
        selection_plan_replay=verify_stateful_plan)
    for name in ('_function','_ordinary','_field','_input','_lookup','_function_state','_same_function_state'):
        functions['shared_guard'+name] = getattr(stateful_guard, name)
    codes = {name:value.__code__ for name,value in functions.items()}
    for name in ('call','fallback'):
        matches = [value for value in functions['shared_guard_factory'].__code__.co_consts
            if type(value) is CodeType and value.co_name == name]
        _need(len(matches) == 1, 'shared guard closure code identity is ambiguous')
        codes['shared_guard_'+name] = matches[0]
    refs = {}; result = {}
    for name, code in codes.items():
        path = str(Path(code.co_filename).resolve())
        if path not in refs:
            refs[path] = artifact_reference(path, 'scar.implementation-source')
        result[name] = dict(source=refs[path], qualname=code.co_qualname,
            code_key=code_key(code), first_line=code.co_firstlineno)
    refs[str(Path(__file__).resolve())] = artifact_reference(__file__, 'scar.implementation-source')
    return result, refs


def derive_stateful_entry_protocol(native_manifest, *, trial, exclusive_inputs, selection=None):
    """Bind the compiler artifact -> adapter -> shared guarded call template.

    ``selection`` is the exact dictionary supplied to the original factory.
    A non-trial native path requires retained real measured-selector evidence;
    no selection is manufactured. Original F4 and live admission remain separate.
    """
    from scar.codegen import stateful
    from scar.compiler.stateful_model import stateful_from_dict, stateful_digest
    from scar.compiler.stateful_runtime import project_stateful_contract, _replay_selection
    from scar.compiler.invariant_loop_runtime import _receipt
    _need(type(trial) is bool and type(exclusive_inputs) is bool, 'exact trial and exclusive_inputs booleans required')
    _need(selection is None or type(selection) is dict, 'selection must be the original optional factory dictionary')
    _need(trial or selection is not None, 'non-trial protocol requires a real retained measured selection')
    reference = (_reference(native_manifest) if type(native_manifest) is dict else
        artifact_reference(Path(native_manifest).resolve(), stateful.SCHEMA))
    manifest = read_artifact(reference)
    _need(reference['schema'] == stateful.SCHEMA and manifest.get('schema') == stateful.SCHEMA
        and type(manifest.get('schema_version')) is int and manifest['schema_version'] == 1
        and type(manifest.get('stateful_abi_version')) is int and manifest['stateful_abi_version'] == 1,
        'compiler StatefulArtifact manifest/ABI1 required; compact runtime is a different protocol')
    schemas = dict(region='scar.stateful-array-region', plan='scar.stateful-stage-plan', source='scar.cpp-source',
        library='scar.native-library', **{'scar_abi_v1.h':'scar.cpp-header','stateful_abi_v1.h':'scar.cpp-header'})
    _need(set(manifest['files']) == set(manifest['sha256']) == set(schemas)
        and len(set(manifest['files'].values())) == len(schemas)
        and all(type(name) is str and Path(name).name == name for name in manifest['files'].values()),
        'complete local compiler artifact member set required')
    directory = Path(reference['uri']).parent; members = {}
    for key, name in manifest['files'].items():
        members[key] = artifact_reference(directory/name, schemas[key])
        _need(members[key]['sha256'] == manifest['sha256'][key], 'native protocol member bytes changed: '+key)
    region = stateful_from_dict(read_artifact(members['region'])); plan = read_artifact(members['plan'])
    core = dict(region_sha256=stateful_digest(region), plan_sha256=semantic_sha256(plan), provider_runtime=manifest['provider_runtime'])
    _need(all(manifest[key] == value for key,value in core.items())
        and semantic_sha256(core) == manifest['contract_sha256']
        and plan['region_sha256'] == core['region_sha256'] and plan['region_id'] == region.id,
        'native body/plan/embedded contract identity differs')
    _need(stateful.implementation_sha256() == manifest['implementation_sha256'], 'original compiler backend implementation changed')
    compiler = manifest['compiler']
    build = dict(source=members['source']['sha256'],
        headers={name:members[name]['sha256'] for name in ('stateful_abi_v1.h','scar_abi_v1.h')},
        compiler=compiler['argv'], identity=compiler['identity'], options=compiler['options'],
        implementation_sha256=manifest['implementation_sha256'], contract=core)
    _need(semantic_sha256(build) == manifest['build_id'], 'native build identity differs from retained compiler/source/header bytes')
    source = _source(region.source); owner = _owner(region.source)
    callee_sources = [_source(row.source) for row in region.callees]
    artifact = SimpleNamespace(region=region, plan=plan, report=manifest)
    selected = None if selection is None else _json(selection); evidence = []
    if not trial:
        def retain(path, role, **expected):
            row = _receipt(path, role, **expected); evidence.append(row); return row
        _replay_selection(artifact, selected, retain)
    contract = project_stateful_contract(artifact, selected, require_selection=not trial)
    implementations, sources = _implementations()
    result = dict(schema=SCHEMA, schema_version=1, family=FAMILY,
        configuration=dict(trial=trial, exclusive_inputs=exclusive_inputs, selection=selected,
            selection_semantic_sha256=None if selected is None else semantic_sha256(selected)),
        selection_evidence=evidence, owner=owner,
        factory_paths=['COMPILER_INSTALL_TO_SHARED_INSTALL_TO_SHARED_GUARD_FACTORY',
            'COMPILER_GUARD_FACTORY_WITH_SAME_PER_INSTALLATION_OWNER_POLICY',
            'PREPARED_ARTIFACT_ADAPTER_TO_SHARED_GUARD_WITH_FAILURE_ONLY_CALLBACK'],
        shared_call_code_key=implementations['shared_guard_call']['code_key'],
        artifact_type='scar.codegen.stateful.StatefulArtifact', adapter_type='scar.compiler.stateful_runtime._ArtifactAdapter',
        compact_artifact_protocol_supported=False, native_manifest=reference, native_members=members,
        native_identity={key:manifest[key] for key in ('build_id','region_sha256','plan_sha256','contract_sha256','implementation_sha256')},
        abi=dict(stateful_abi_version=1, operator_descriptor_abi_version=1, native_argument_count=6,
            array_count=sum(row.kind == 'array' for row in region.inputs),
            scalar_count=sum(row.kind == 'scalar' for row in region.inputs), provider_count=len(region.providers)),
        original_source=source, callee_sources=callee_sources,
        projected_contract_sha256=semantic_sha256(contract), parameter_names=list(region.parameter_names),
        inputs=contract['inputs'], callee_guards=contract['callee_guards'], providers=contract['providers'],
        guards=contract['guards'], external_assumptions=contract['external_assumptions'],
        cpu=contract['cpu'], memory=contract['memory'], provider_runtime=contract['provider_runtime'],
        cpu_binding_scope='NATIVE_PRIVATE_STAGES_NOT_WHOLE_METHOD_CALLER_ASSIGNMENT',
        caller_data_allocator_and_physical_context='SEPARATELY_FROZEN_CAPTURE_CONTEXT_NOT_ESTABLISHED_BY_PROTOCOL',
        timer=deepcopy(TIMER), implementation_functions=implementations, implementation_sources=sources,
        guard_lifetime='ONE_PERSISTENT_WRAPPER_AFTER_CALLABLE_ADMISSION',
        lazy_callee_binding='FIRST_RECEIVER_DEPENDENT_CALLEE_BINDING_OCCURS_INSIDE_CALL_UNLESS_ALREADY_WARM',
        first_binding_and_warm_cache_costs_assumed_equal=False,
        native_success_branch_possible_under_runtime_guards=exclusive_inputs,
        scope='NATIVE_SUCCESS_PATH_DISPATCH_THROUGH_RETURN', normal_success_only=True,
        required_sample_guard_delta=dict(native_calls=1, fallback_calls=0),
        fallback_cost_ns=None, failure_callback_identity_part_of_success_cost=False,
        failure_callback_policy='FATAL_CANONICAL_CALLBACK_ALLOWED_ONLY_ON_UNMEASURED_FAILURE_BRANCH',
        original_fallback_required_for_delivery=True,
        fallback_policy='ORIGINAL_METHOD_ONCE_ON_DECLINED_GUARDS_OR_PRECOMMIT_NATIVE_FAILURE',
        python=dict(executable=sys.executable, version=sys.version),
        source_F4_admitted=False, live_guards_discharged=False, performance_measured=False,
        native_loaded=False, target_executed=False, compiled=False, cost_qualified=False,
        selection_replay_is_not_new_deployment_authorization=True, deployment_authorized=False)
    result['id'] = 'stateful-entry-protocol:'+semantic_sha256(result)
    return _json(result)


def verify_stateful_entry_protocol(report, native_manifest, *, trial, exclusive_inputs, selection=None):
    expected = derive_stateful_entry_protocol(native_manifest, trial=trial, exclusive_inputs=exclusive_inputs, selection=selection)
    _need(_json(report) == expected, 'stateful entry protocol differs from exact source/configuration/native replay')
    return expected


__all__ = ['FAMILY','TIMER','derive_stateful_entry_protocol','verify_stateful_entry_protocol']

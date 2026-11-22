"""Invariant loop runtime."""

from functools import wraps
from dataclasses import asdict
import builtins
import inspect
import sys
import threading
import json
import platform
import struct
from pathlib import Path
from hashlib import sha256
from types import FunctionType
from scar.compiler.pipeline_runtime import _state, _same_state
from scar.compiler.runtime import SourceMismatch
from scar.trace.code_snapshot import code_key
from scar.optimization.resource_solver import semantic_sha256


# Original-callable admission around dependency-derived invariant placement.
_RANGE=range
_AUTHORITY='ORIGINAL_F4_COMPOSITE_FUSION_AND_FUNCTION_GLOBAL_ROUTE_REPLAY'


def _require_source_movement_binding(artifact,schema):
    """Construction check for evidence already consumed by the native loader.

    This does not replay F4 or turn runtime guards into movement authority.
    An unbound conditional micro artifact remains available through run().
    """
    plan=artifact.plan;reference=plan['loop_interface'].get('source_motion')
    binding=plan.get('execution_movement_binding')
    if (type(reference)is not dict or reference.get('schema')!=schema
            or type(binding)is not dict or binding.get('schema')!='scar.temporal-source-movement-binding'
            or type(binding.get('schema_version'))is not int or binding['schema_version']!=1
            or binding.get('source_motion')!=reference
            or binding.get('source_body_sha256')!=plan['body_sha256']):
        raise SourceMismatch('original caller requires consumed source movement evidence')


def _guarded_loop_call(original, artifact, proof, *, exclusive_inputs, trial, bindings_match,
                       sources_match=None):
    composite = sources_match is not None
    shadow = FunctionType(original.__code__, original.__globals__, argdefs=original.__defaults__)
    shadow.__kwdefaults__ = original.__kwdefaults__
    signature = inspect.signature(shadow, follow_wrapped=False)
    if composite and tuple(signature.parameters) != proof.parameters:
        raise SourceMismatch('original outer parameter order differs from composite proof')
    stats = dict(native_calls=0, fallback_calls=0, reasons={})
    if composite:
        stats['alias_returns'] = 0
    parameter_map = dict(proof.parameter_map)
    selection_reason = 'no_validated_composite_cost_selection' if composite else 'no_validated_cost_selection'
    binding_reason = 'composite_global_callable_or_range_changed' if composite else 'loop_leaf_or_range_binding_changed'
    composite_interface = artifact.plan['loop_interface'] if composite else None

    def fallback(reason, args, kwargs):
        stats['fallback_calls'] += 1
        stats['reasons'][reason] = stats['reasons'].get(reason, 0) + 1
        return original(*args, **kwargs)

    @wraps(original)
    def run(*args, **kwargs):
        if not trial:
            return fallback(selection_reason, args, kwargs)
        if not exclusive_inputs or threading.active_count() != 1 or sys.gettrace() is not None or sys.getprofile() is not None:
            return fallback('exclusive_synchronous_inputs_required', args, kwargs)
        monitoring = getattr(sys, 'monitoring', None)
        if monitoring is not None:
            slots = _RANGE(6) if composite else range(6)
            if any(monitoring.get_tool(index) is not None for index in slots):
                return fallback('monitoring_observer_active', args, kwargs)
        if not bindings_match():
            return fallback(binding_reason, args, kwargs)
        if sources_match is not None and not sources_match():
            return fallback('composite_source_bytes_changed', args, kwargs)
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError:
            return fallback('original_argument_binding', args, kwargs)
        bound.apply_defaults()
        count = bound.arguments[proof.trip_count_parameter]
        if type(count) is not int or not -(2**63) <= count < 2**63:
            return fallback('trip_count_outside_int64', args, kwargs)
        inputs = {name: bound.arguments[outer] for name, outer in parameter_map.items()}
        interface = composite_interface if composite else artifact.plan['loop_interface']
        specialization = interface.get('execution_specialization')
        if specialization is not None:
            mismatch = 'outside_sourced_execution_specialization'
            if count != specialization['trip_count']:
                return fallback(mismatch, args, kwargs)
            import numpy as np

            if any(type(value) is not np.ndarray for value in inputs.values()):
                return fallback('outside_exact_array_input_contract', args, kwargs)
            shapes = specialization['input_shapes']
            if any(list(inputs[name].shape) != shape for name, shape in shapes.items()):
                return fallback(mismatch, args, kwargs)
            if composite or specialization.get('input_layout') == 'C_CONTIGUOUS':
                arrays = (inputs[name] for name in shapes) if composite else inputs.values()
                if any(not value.flags.c_contiguous for value in arrays):
                    return fallback(mismatch, args, kwargs)
        failed = False

        def original_once():
            nonlocal failed
            failed = True
            return fallback('native_precommit_declined', args, kwargs)

        result = artifact.run(inputs, count, original_loop=original_once)
        if not failed:
            stats['alias_returns' if composite and count <= 0 else 'native_calls'] += 1
        return result

    run.scar_stats = stats
    run.scar_artifact = artifact
    if composite:
        run.scar_composite_loop_proof = proof
    else:
        run.scar_loop_proof = proof
    return run


def guarded_invariant_loop(original,artifact,proof,*,exclusive_inputs=False,trial=False):
    """Admit a source-derived loop for validation; deployment cost is separate.

    No import-time or user-provided node list selects hoisted expressions.
    Trial mode permits correctness/A-B validation, not automatic deployment.
    """
    _require_source_movement_binding(artifact,'scar.source-loop-motion')
    if type(original) is not FunctionType or original.__closure__ or code_key(original.__code__)!=proof.source.code_key:
        raise SourceMismatch('original ordinary loop callable differs from F4 interface')
    leaf=original.__globals__.get(proof.leaf_global_name)
    if type(leaf) is not FunctionType or leaf.__closure__ or code_key(leaf.__code__)!=proof.body.source.code_key:
        raise SourceMismatch('original loop leaf differs from F4 interface')
    for source in (proof.source,proof.body.source):
        if sha256(Path(source.path).read_bytes()).hexdigest()!=source.sha256:
            raise SourceMismatch('loop source bytes changed')
    if artifact.plan['body_sha256']!=semantic_sha256(asdict(proof.body)) or artifact.plan['carry_parameter']!=proof.carry_parameter:
        raise SourceMismatch('LICM placement refers to a different original body or carry')
    if semantic_sha256(artifact.plan['loop_interface'].get('original_f4_proof'))!=semantic_sha256(asdict(proof)):
        raise SourceMismatch('LICM caller mapping differs from original F4 proof')
    snapshot = _state(original)
    leaf_state = _state(leaf)

    def bindings_match():
        namespace = original.__globals__
        return (_same_state(original, snapshot) and namespace.get(proof.leaf_global_name) is leaf
                and _same_state(leaf, leaf_state)
                and (namespace['range'] if 'range' in namespace else original.__builtins__.get('range')) is _RANGE)

    return _guarded_loop_call(original, artifact, proof, exclusive_inputs=exclusive_inputs, trial=trial,
        bindings_match=bindings_match)


def _loop_interface(proof,*,execution_specialization=None,source_motion=None):
    """One interface for preview and lowering; source admission stays separate."""
    from scar.compiler.pipeline_loop_frontend import CompositeInvariantLoopProof

    if type(proof) is not CompositeInvariantLoopProof:
        raise ValueError('separate original composite proof required')
    interface=dict(proof_kind=proof.profile,original_composite_f4_proof=asdict(proof),
        carry_parameter=proof.carry_parameter,
        readonly_parameters=[name for name in proof.body.parameters if name!=proof.carry_parameter],
        root_callee_source=asdict(proof.root_callee_source),
        function_global_guards=[asdict(guard) for guard in proof.nested_callee_guards],
        authority=_AUTHORITY,runtime_route_admission_required=True,
        synthetic_body_is_original_callable=False)
    if execution_specialization is not None:
        spec=execution_specialization
        if (type(spec) is not dict or set(spec)!={'input_shapes','trip_count','input_layout'}
                or spec['input_layout']!='C_CONTIGUOUS' or type(spec['trip_count']) is not int
                or not -(2**63)<=spec['trip_count']<2**63 or type(spec['input_shapes']) is not dict
                or set(spec['input_shapes'])!=set(proof.body.parameters)
                or any(type(shape) is not list or len(shape)!=2 or any(type(n) is not int or n<=0 for n in shape)
                    for shape in spec['input_shapes'].values())):
            raise ValueError('exact composite shape/count/contiguous-layout specialization required')
        interface['execution_specialization']=dict(input_shapes={name:list(shape) for name,shape in spec['input_shapes'].items()},
            trip_count=spec['trip_count'],input_layout=spec['input_layout'])
    if source_motion is not None:
        from scar.ir.execution_recipe import _reference
        reference=_reference(source_motion)
        if reference['schema']!='scar.source-composite-loop-motion':
            raise ValueError('separate composite source movement reference required')
        interface['source_motion']=reference
    return interface


def _composite_function(function,source):
    if (type(function) is not FunctionType or function.__closure__
            or type(function.__globals__) is not dict or type(function.__builtins__) is not dict
            or function.__kwdefaults__ is not None and type(function.__kwdefaults__) is not dict
            or code_key(function.__code__)!=source.code_key):
        raise SourceMismatch('actual ordinary function differs from original composite CodeID')
    return function


def _composite_route(original,guard):
    if (guard.lookup_semantics!='FUNCTION_GLOBALS_AT_EACH_HOP' or not guard.global_path
            or guard.global_path[-1]!=guard.lookup_name):
        raise SourceMismatch('composite guard has no original function-global route')
    owner=original
    for name in guard.global_path[:-1]:
        if type(owner) is not FunctionType or owner.__closure__ or type(owner.__globals__) is not dict:
            raise SourceMismatch('composite route owner is not an ordinary function')
        owner=owner.__globals__.get(name)
    _composite_function(owner,guard.owner_source)
    target=owner.__globals__.get(guard.lookup_name)
    _composite_function(target,guard.target_source)
    return owner,target


def guarded_composite_invariant_loop(original,artifact,proof,*,exclusive_inputs=False,trial=False):
    """Guard every original global CALL hop before one native loop invocation.

    A changed binding/code/default/source falls back to the original entire
    loop exactly once. Only exact FunctionType.__globals__ dictionaries are
    traversed; no custom attribute lookup is performed on application objects.
    """
    from scar.compiler.pipeline_loop_frontend import CompositeInvariantLoopProof

    _require_source_movement_binding(artifact,'scar.source-composite-loop-motion')
    if type(proof) is not CompositeInvariantLoopProof:
        raise SourceMismatch('separate original composite loop proof required')
    _composite_function(original,proof.source)
    root=original.__globals__.get(proof.root_global_name)
    _composite_function(root,proof.root_callee_source)  # Not the synthetic body's source.
    interface=artifact.plan['loop_interface']
    if (interface.get('authority')!=_AUTHORITY or interface.get('proof_kind')!=proof.profile
            or interface.get('runtime_route_admission_required') is not True
            or interface.get('synthetic_body_is_original_callable') is not False
            or semantic_sha256(interface.get('original_composite_f4_proof'))!=semantic_sha256(asdict(proof))
            or artifact.plan['body_sha256']!=semantic_sha256(asdict(proof.body))
            or artifact.plan['carry_parameter']!=proof.carry_parameter):
        raise SourceMismatch('native body/composite route interface differs from original F4 proof')
    sources={}
    for source in (proof.source,proof.root_callee_source,
            *(guard.owner_source for guard in proof.nested_callee_guards),
            *(guard.target_source for guard in proof.nested_callee_guards)):
        if source.path in sources and sources[source.path]!=source.sha256:
            raise SourceMismatch('composite source path has conflicting versions')
        sources[source.path]=source.sha256
    def sources_match():
        try:return all(sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in sources.items())
        except OSError:return False
    if not sources_match():raise SourceMismatch('original composite source bytes changed')
    routes=[]
    for guard in proof.nested_callee_guards:
        owner,target=_composite_route(original,guard)
        routes.append((guard,owner,target,_state(owner),_state(target)))
    if not routes or not any(guard.global_path==(proof.root_global_name,)
            and target is root for guard,_,target,_,_ in routes):
        raise SourceMismatch('original composite root CALL guard missing')
    original_state=_state(original);root_state=_state(root)

    def bindings_match():
        try:
            _composite_function(original,proof.source);_composite_function(root,proof.root_callee_source)
        except SourceMismatch:return False
        if (not _same_state(original,original_state) or original.__globals__.get(proof.root_global_name) is not root
                or not _same_state(root,root_state)):
            return False
        for guard,owner,target,owner_state,target_state in routes:
            try:actual_owner,actual_target=_composite_route(original,guard)
            except SourceMismatch:return False
            if (actual_owner is not owner or actual_target is not target
                    or not _same_state(owner,owner_state) or not _same_state(target,target_state)):
                return False
        namespace=original.__globals__
        current_range=namespace['range'] if 'range' in namespace else original.__builtins__.get('range')
        return current_range is _RANGE
    return _guarded_loop_call(original, artifact, proof, exclusive_inputs=exclusive_inputs, trial=trial,
        bindings_match=bindings_match, sources_match=sources_match)


# Build-time admission and small runtime projection of an existing loop ELF.
#
# The complete loader owns graph/strategy/profile/source/ISA replay. Exported
# hashes bind bytes to an explicitly trusted producer receipt, not a signature.
def _json_bytes(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def _receipt(path,role,*,expected_sha256=None,expected_bytes=None):
    path=Path(path).absolute()
    if path.is_symlink() or not path.is_file():raise ValueError('original regular export evidence required: '+str(path))
    before=path.stat();digest=sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    after=path.stat();stamp=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if stamp(before)!=stamp(after):raise ValueError('export evidence changed while reading: '+str(path))
    result=dict(role=role,path=str(path),sha256=digest.hexdigest(),bytes=after.st_size)
    if ((expected_sha256 is not None and result['sha256']!=expected_sha256)
            or expected_bytes is not None and result['bytes']!=expected_bytes):
        raise ValueError('export evidence differs from complete admission: '+str(path))
    return result


def _read_json(path):
    if Path(path).stat().st_size>16*1024*1024:raise ValueError('export JSON byte budget exceeded')
    def pairs(rows):
        value={}
        for key,item in rows:
            if key in value:raise ValueError('duplicate export JSON field')
            value[key]=item
        return value
    def invalid(value):raise ValueError('nonfinite export JSON value: '+value)
    return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,parse_constant=invalid)


def _source(value):
    return dict(path=value.path,sha256=value.sha256,code_key=value.code_key,qualname=value.qualname)


def _profile_artifacts(value):
    """Retain original fact artifacts; no cost or graph semantics are inferred."""
    pending=[value];seen=0
    while pending:
        row=pending.pop();seen+=1
        if seen>100000:raise ValueError('export evidence traversal budget exceeded')
        if type(row)is dict:
            if {'uri','sha256','size_bytes','schema'}<=set(row):
                if type(row['uri'])is str and Path(row['uri']).is_absolute():yield row
            pending.extend(reversed(tuple(row.values())))
        elif type(row)is list or type(row)is tuple:pending.extend(reversed(row))


def export_loop(artifact_path,out,*,selection_path):
    """Replay full admission, then copy one ELF and project its runtime guards.

    The returned expected_admission_sha256 must be trusted separately by the
    caller. Runtime consumers never need the paths in admission.evidence.
    """
    from scar.codegen.loop import load,_contract
    from scar.runtime.loop_contract import (SCHEMA,ADMISSION_SCHEMA,TRUST_POLICY,IDENTITIES,
        validate_contract,validate_admission,canonical_sha256)

    output=Path(out).absolute()
    if output.resolve()!=output or output.exists():raise ValueError('fresh canonical runtime bundle directory required')
    original=Path(artifact_path).resolve();directory=original if original.is_dir() else original.parent
    if output.is_relative_to(directory):raise ValueError('runtime export must not modify the original artifact directory')
    manifest=_receipt(directory/'artifact.json','original-native-artifact')
    artifact=load(artifact_path)  # Complete original admission, including ELF table identities.
    report=artifact.report;loop=artifact.loop;plan=artifact.execution_plan;strategy=artifact.strategy;layout=artifact.layout
    if artifact.directory.resolve()!=directory:raise ValueError('full loader returned another artifact directory')
    if (sys.platform!='linux' or platform.machine().lower() not in ('x86_64','amd64')
            or struct.calcsize('P')!=8 or sys.byteorder!='little'):
        raise ValueError('runtime export supports the admitted Linux x86_64 little-endian contract')
    evidence=[];by_path={}
    def retain(path,role,**expected):
        row=_receipt(path,role,**expected)
        old=by_path.get(row['path'])
        if old is not None:
            if (old['sha256'],old['bytes'])!=(row['sha256'],row['bytes']):
                raise ValueError('one export evidence path has conflicting content versions')
            return old
        evidence.append(row);by_path[row['path']]=row
        if len(evidence)>4096:raise ValueError('runtime export evidence receipt budget exceeded')
        return row
    retain(directory/'artifact.json',manifest['role'],expected_sha256=manifest['sha256'],expected_bytes=manifest['bytes'])
    for key,name in report['files'].items():
        retain(directory/name,'original-native-'+key,expected_sha256=report['sha256'][key])
    selection_file=Path(selection_path).absolute();selection_receipt=retain(selection_file,'selection:#')
    document=_read_json(selection_file)
    if type(document)is dict and 'selection' in document:
        selection=document['selection'];pointer='#/selection'
    else:selection=document;pointer='#'
    selection_receipt['role']='selection:'+pointer
    assumptions=list(strategy['external_assumptions'])
    if report['external_assumptions']!=assumptions:
        raise ValueError('native metadata and admitted strategy assumptions differ')
    guards=[];paths={}
    def guard(path,source):
        projected=_source(source);key=tuple(path)
        if key in paths:
            if paths[key]!=projected:raise ValueError('one runtime globals path has different code/source guards')
            return
        paths[key]=projected;guards.append(dict(environment_path=list(path),source=projected))
    body_path=tuple(loop.body_callee.lookup_path);guard(body_path,loop.body_callee.source)
    for call in loop.body.calls:guard((*body_path,*call.callee.lookup_path),call.callee.source)
    for source in (loop.source,loop.body.source,*(call.callee.source for call in loop.body.calls)):
        retain(source.path,'original-application-source',expected_sha256=source.sha256)
    # Sidecar receipts retain complete IR/profile provenance on the build side;
    # only their byte identities cross the runtime admission boundary.
    binding=plan.resource_binding
    for reference in _profile_artifacts(asdict(binding)):
        retain(reference['uri'],'original-target-fact',expected_sha256=reference['sha256'],expected_bytes=reference['size_bytes'])
    if layout is not None:
        profile=layout['hardware_profile'];profile_receipt=layout['hardware_profile_receipt']
        if profile_receipt is not None:
            retain(profile_receipt['path'],'original-hardware-profile',expected_sha256=profile_receipt['sha256'],expected_bytes=profile_receipt['bytes'])
            profile=_read_json(profile_receipt['path'])
        for reference in _profile_artifacts(profile):
            retain(reference['uri'],'original-hardware-profile-fact',expected_sha256=reference['sha256'],expected_bytes=reference['size_bytes'])
    library=report['files']['library']
    if library in ('runtime.json','admission.json','export-report.json'):raise ValueError('native library basename collides with runtime bundle metadata')
    library_bytes=(directory/library).read_bytes();library_sha256=sha256(library_bytes).hexdigest()
    if library_sha256!=report['sha256']['library']:raise ValueError('native library changed after full admission')
    ids={key:report[key] for key in sorted(IDENTITIES)}
    native_contract=_contract(loop,strategy,plan,layout)
    if canonical_sha256(native_contract)!=ids['contract_sha256']:raise ValueError('projected native contract differs from full admission')
    runtime=validate_contract(dict(schema=SCHEMA,schema_version=1,profile='cpu-f64-rank2-loop-v1',
        native=dict(library=library,library_sha256=library_sha256,loop_abi_version=report['loop_abi_version'],ids=ids,contract=native_contract),
        source=_source(loop.source),parameters=list(loop.parameters),state_parameter=loop.state_parameter,trip_count_parameter=loop.trip_count_parameter,
        callee_guards=guards,array=dict(shape=list(report['state_shape']),dtype='float64',device='cpu',layout='C',alignment_bytes=8,final_bytes=report['final_bytes']),
        trip_count=report['trip_count'],memory_budget_bytes=report['memory_budget_bytes'],
        cpu=dict(ids=list(report['cpu_ids']),threads=report['cpu_threads'],architecture='x86_64',os='linux',pointer_bits=64,byteorder='little',
            required_isa='SSE2_X86_64' if layout is not None else 'BASELINE_X86_64'),
        external_assumptions=assumptions,floating_environment_policy=report['floating_environment_policy'],selection=selection))
    runtime_bytes=_json_bytes(runtime)
    admission=validate_admission(dict(schema=ADMISSION_SCHEMA,schema_version=1,runtime_contract_sha256=sha256(runtime_bytes).hexdigest(),
        library_sha256=library_sha256,native_ids=dict(ids),producer=dict(name='SCAR',version='0.1.0',exporter_sha256=sha256(Path(__file__).read_bytes()).hexdigest()),
        evidence=evidence,trust_policy=TRUST_POLICY))
    admission_bytes=_json_bytes(admission)
    for row in evidence:_receipt(row['path'],row['role'],expected_sha256=row['sha256'],expected_bytes=row['bytes'])
    output.mkdir(parents=True)
    (output/library).write_bytes(library_bytes);(output/'runtime.json').write_bytes(runtime_bytes);(output/'admission.json').write_bytes(admission_bytes)
    result=dict(schema='scar.loop-runtime-export',schema_version=1,bundle=str(output),runtime_contract=str(output/'runtime.json'),
        admission=str(output/'admission.json'),library=str(output/library),expected_admission_sha256=sha256(admission_bytes).hexdigest(),
        runtime_contract_sha256=sha256(runtime_bytes).hexdigest(),library_sha256=library_sha256,build_id=ids['build_id'],
        admission_trust_policy=TRUST_POLICY,full_original_loader_admission=True,evidence_receipts=len(evidence),
        native_recompiled=False,target_executed=False,IR_or_profile_copied=False)
    (output/'export-report.json').write_bytes(_json_bytes(result))
    return result


__all__ = [
    'export_loop',
    'guarded_invariant_loop',
]

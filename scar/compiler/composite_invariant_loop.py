"""F4 composite proof -> existing C++20 temporal host -> guarded whole loop.

This build-side facade deliberately keeps composite admission separate from
InvariantLoopProof/v1. It qualifies candidate validation only: no deployment
cost, whole-program C++ coverage or physical optimum is established here.
"""
from dataclasses import asdict
from functools import wraps
from hashlib import sha256
import inspect
from pathlib import Path
import sys
import threading
from types import FunctionType

from scar.compiler.pipeline_loop_frontend import (CompositeInvariantLoopProof,
    validate_composite_invariant_loop_proof)
from scar.compiler.pipeline_runtime import _state,_same_state
from scar.compiler.runtime import SourceMismatch
from scar.optimization.resource_solver import semantic_sha256
from scar.trace.code_snapshot import code_key

_RANGE=range
_AUTHORITY='ORIGINAL_F4_COMPOSITE_FUSION_AND_FUNCTION_GLOBAL_ROUTE_REPLAY'


def build_composite_invariant_loop(proof,out,*,catalog,memory_budget_bytes,resource_binding=None,
        execution_specialization=None):
    """Replay source/fusion/routes, then lower the existing derived Expr DAG."""
    from scar.compiler.temporal_invariant_loop import build_temporal_invariant_loop
    proof=validate_composite_invariant_loop_proof(proof,catalog=catalog)
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
    return build_temporal_invariant_loop(proof.body,out,carry_parameter=proof.carry_parameter,
        loop_interface=interface,memory_budget_bytes=memory_budget_bytes,resource_binding=resource_binding)


def _function(function,source):
    if (type(function) is not FunctionType or function.__closure__
            or type(function.__globals__) is not dict or type(function.__builtins__) is not dict
            or function.__kwdefaults__ is not None and type(function.__kwdefaults__) is not dict
            or code_key(function.__code__)!=source.code_key):
        raise SourceMismatch('actual ordinary function differs from original composite CodeID')
    return function


def _route(original,guard):
    if (guard.lookup_semantics!='FUNCTION_GLOBALS_AT_EACH_HOP' or not guard.global_path
            or guard.global_path[-1]!=guard.lookup_name):
        raise SourceMismatch('composite guard has no original function-global route')
    owner=original
    for name in guard.global_path[:-1]:
        if type(owner) is not FunctionType or owner.__closure__ or type(owner.__globals__) is not dict:
            raise SourceMismatch('composite route owner is not an ordinary function')
        owner=owner.__globals__.get(name)
    _function(owner,guard.owner_source)
    target=owner.__globals__.get(guard.lookup_name)
    _function(target,guard.target_source)
    return owner,target


def guarded_composite_invariant_loop(original,artifact,proof,*,exclusive_inputs=False,trial=False):
    """Guard every original global CALL hop before one native loop invocation.

    A changed binding/code/default/source falls back to the original entire
    loop exactly once. Only exact FunctionType.__globals__ dictionaries are
    traversed; no custom attribute lookup is performed on application objects.
    """
    if type(proof) is not CompositeInvariantLoopProof:
        raise SourceMismatch('separate original composite loop proof required')
    _function(original,proof.source)
    root=original.__globals__.get(proof.root_global_name)
    _function(root,proof.root_callee_source)  # Not the synthetic body's source.
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
        owner,target=_route(original,guard)
        routes.append((guard,owner,target,_state(owner),_state(target)))
    if not routes or not any(guard.global_path==(proof.root_global_name,)
            and target is root for guard,_,target,_,_ in routes):
        raise SourceMismatch('original composite root CALL guard missing')
    original_state=_state(original);root_state=_state(root)
    signature_source=FunctionType(original.__code__,original.__globals__,argdefs=original.__defaults__)
    signature_source.__kwdefaults__=original.__kwdefaults__
    signature=inspect.signature(signature_source,follow_wrapped=False)
    if tuple(signature.parameters)!=proof.parameters:
        raise SourceMismatch('original outer parameter order differs from composite proof')
    stats=dict(native_calls=0,fallback_calls=0,alias_returns=0,reasons={})
    parameter_map=dict(proof.parameter_map)
    def fallback(reason,args,kwargs):
        stats['fallback_calls']+=1;stats['reasons'][reason]=stats['reasons'].get(reason,0)+1
        return original(*args,**kwargs)
    def bindings_match():
        try:
            _function(original,proof.source);_function(root,proof.root_callee_source)
        except SourceMismatch:return False
        if (not _same_state(original,original_state) or original.__globals__.get(proof.root_global_name) is not root
                or not _same_state(root,root_state)):
            return False
        for guard,owner,target,owner_state,target_state in routes:
            try:actual_owner,actual_target=_route(original,guard)
            except SourceMismatch:return False
            if (actual_owner is not owner or actual_target is not target
                    or not _same_state(owner,owner_state) or not _same_state(target,target_state)):
                return False
        namespace=original.__globals__
        current_range=namespace['range'] if 'range' in namespace else original.__builtins__.get('range')
        return current_range is _RANGE
    @wraps(original)
    def call(*args,**kwargs):
        if not trial:return fallback('no_validated_composite_cost_selection',args,kwargs)
        if not exclusive_inputs or threading.active_count()!=1 or sys.gettrace() is not None or sys.getprofile() is not None:
            return fallback('exclusive_synchronous_inputs_required',args,kwargs)
        monitoring=getattr(sys,'monitoring',None)
        if monitoring is not None and any(monitoring.get_tool(i) is not None for i in _RANGE(6)):
            return fallback('monitoring_observer_active',args,kwargs)
        if not bindings_match():return fallback('composite_global_callable_or_range_changed',args,kwargs)
        if not sources_match():return fallback('composite_source_bytes_changed',args,kwargs)
        try:bound=signature.bind(*args,**kwargs)
        except TypeError:return fallback('original_argument_binding',args,kwargs)
        bound.apply_defaults();count=bound.arguments[proof.trip_count_parameter]
        if type(count) is not int or not -(2**63)<=count<2**63:
            return fallback('trip_count_outside_int64',args,kwargs)
        inputs={name:bound.arguments[outer] for name,outer in parameter_map.items()}
        specialization=interface.get('execution_specialization')
        if specialization is not None:
            if count!=specialization['trip_count']:
                return fallback('outside_sourced_execution_specialization',args,kwargs)
            import numpy as np
            if any(type(value) is not np.ndarray for value in inputs.values()):
                return fallback('outside_exact_array_input_contract',args,kwargs)
            if any(list(inputs[name].shape)!=shape or not inputs[name].flags.c_contiguous
                    for name,shape in specialization['input_shapes'].items()):
                return fallback('outside_sourced_execution_specialization',args,kwargs)
        failed=[False]
        def original_once():
            failed[0]=True;return fallback('native_precommit_declined',args,kwargs)
        result=artifact.run(inputs,count,original_loop=original_once)
        if not failed[0]:
            if count>0:stats['native_calls']+=1
            else:stats['alias_returns']+=1
        return result
    call.scar_stats=stats;call.scar_artifact=artifact;call.scar_composite_loop_proof=proof
    return call


__all__=['build_composite_invariant_loop','guarded_composite_invariant_loop']

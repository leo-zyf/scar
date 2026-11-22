"""Live Python boundary guards for an admitted loop; no compiler dependencies."""
from contextlib import contextmanager
from functools import wraps
from hashlib import sha256
import inspect
import os
from pathlib import Path
import sys
import threading
from types import FunctionType, ModuleType

from scar.trace.code_snapshot import code_key
from .loop_contract import validate_contract
from .native_loop import GuardFailure

_RANGE = range


class SourceMismatch(ValueError):
    pass


def _state(function):
    return (function.__code__,function.__defaults__,
            tuple((name,id(value)) for name,value in (function.__kwdefaults__ or {}).items()))


def _same_state(function,state):
    return (function.__code__ is state[0] and function.__defaults__ is state[1]
            and tuple((name,id(value)) for name,value in (function.__kwdefaults__ or {}).items())==state[2])


def _range(function):
    namespace=function.__globals__
    if 'range' in namespace:return namespace['range']
    # A function retains its own builtins dict. Replacing globals['__builtins__']
    # after creation does not change Python's actual LOAD_GLOBAL fallback.
    namespace=function.__builtins__
    return namespace.get('range') if type(namespace) is dict else None


def guarded_loop_function(original,artifact,*,exclusive_inputs=False):
    contract=validate_contract(artifact.contract)
    source=contract['source']
    if (type(original) is not FunctionType or original.__closure__
            or type(original.__globals__) is not dict or code_key(original.__code__)!=source['code_key']):
        raise SourceMismatch('original loop code/environment differs from the admitted boundary')
    if _range(original) is not _RANGE:raise SourceMismatch('original loop does not resolve builtin range')
    versions={source['path']:source['sha256']};functions={():original};guards=[]
    for guard in sorted(contract['callee_guards'],key=lambda row:len(row['environment_path'])):
        path=tuple(guard['environment_path']);parent=functions[path[:-1]]
        namespace=parent.__globals__;function=namespace.get(path[-1]);expected=guard['source']
        if (type(function) is not FunctionType or function.__closure__
                or type(function.__globals__) is not dict or code_key(function.__code__)!=expected['code_key']):
            raise SourceMismatch('nested function-globals binding differs from the admitted code')
        if versions.setdefault(expected['path'],expected['sha256'])!=expected['sha256']:
            raise SourceMismatch('nested source versions disagree')
        functions[path]=function;guards.append((namespace,path[-1],function,_state(function)))
    for path,digest in versions.items():
        if sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise SourceMismatch('original application source differs from the admitted version')
    entry_state=_state(original)
    signature_source=FunctionType(original.__code__,original.__globals__,argdefs=original.__defaults__)
    signature_source.__kwdefaults__=original.__kwdefaults__
    signature=inspect.signature(signature_source,follow_wrapped=False)
    shape=tuple(contract['array']['shape']);count=contract['trip_count'];cpus=set(contract['cpu']['ids'])
    stats=dict(native_calls=0,fallback_calls=0,reasons={})

    def fallback(reason,args,kwargs):
        stats['fallback_calls']+=1;stats['reasons'][reason]=stats['reasons'].get(reason,0)+1
        return original(*args,**kwargs)

    @wraps(original)
    def run(*args,**kwargs):
        if (not exclusive_inputs or threading.active_count()!=1
                or sys.getprofile() is not None or sys.gettrace() is not None):
            return fallback('synchronous_exclusive_input_contract',args,kwargs)
        monitoring = getattr(sys, 'monitoring', None)
        if monitoring is not None and any(monitoring.get_tool(tool) is not None for tool in _RANGE(6)):
            return fallback('monitoring_observer_active', args, kwargs)
        if not hasattr(os,'sched_getaffinity') or not cpus<=os.sched_getaffinity(0):
            return fallback('planned_cpu_allocation_unavailable',args,kwargs)
        if not _same_state(original,entry_state) or _range(original) is not _RANGE:
            return fallback('loop_code_defaults_or_range_changed',args,kwargs)
        if any(namespace.get(name) is not function or not _same_state(function,state)
               for namespace,name,function,state in guards):
            return fallback('actual_nested_callee_changed',args,kwargs)
        try:bound=signature.bind(*args,**kwargs)
        except TypeError:return fallback('call_signature',args,kwargs)
        bound.apply_defaults();steps=bound.arguments[contract['trip_count_parameter']]
        if type(steps) is not int or steps!=count or steps<=0:
            return fallback('trip_count_outside_exact_specialization',args,kwargs)
        value=bound.arguments[contract['state_parameter']]
        import numpy as np
        if (type(value) is not np.ndarray or value.dtype!=np.dtype('float64') or value.shape!=shape
                or not value.flags.c_contiguous or not value.flags.aligned):
            return fallback('outside_modeled_shape_dtype_or_layout',args,kwargs)
        try:result=artifact.run(value,steps)
        except GuardFailure as error:return fallback(str(error),args,kwargs)
        stats['native_calls']+=1
        return result

    run.scar_stats=stats;run.scar_artifact=artifact
    return run


@contextmanager
def install_loop(module,artifact,*,exclusive_inputs=False):
    name=artifact.contract['source']['qualname']
    if type(module) is not ModuleType or not name.isidentifier():
        raise SourceMismatch('runtime installation requires a direct module function')
    original=vars(module)[name]
    wrapped=guarded_loop_function(original,artifact,exclusive_inputs=exclusive_inputs)
    setattr(module,name,wrapped)
    try:yield wrapped
    finally:setattr(module,name,original)


__all__=['SourceMismatch','guarded_loop_function','install_loop']

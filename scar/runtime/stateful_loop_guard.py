"""Live original-loop admission and progress-aware Python suffix recovery."""
from functools import wraps
import inspect
import math
import sys
import threading
from types import FunctionType

import numpy as np

from .native_stateful_repeat import NativeStatefulRepeat,RepeatGuardFailure,validate_contract
from .stateful_guard import _function,_function_state,_same_function_state,_input,_lookup

_RANGE=range
_NDARRAY=np.ndarray;_FLOAT64=np.dtype('float64').type
_GETTRACE=sys.gettrace;_GETPROFILE=sys.getprofile
_MONITOR=getattr(sys,'monitoring',None)
_MONITOR_APIS=None if _MONITOR is None else(_MONITOR.get_tool,_MONITOR.get_events,_MONITOR.get_local_events)


def _plain(function,source):
    _function(source,function)
    if (type(function.__globals__)is not dict or type(function.__builtins__)is not dict
            or function.__defaults__ is not None or function.__kwdefaults__ is not None
            or '__signature__'in vars(function)or'__wrapped__'in vars(function)):
        raise ValueError('ordinary no-default caller/step function required')


def _name(function,name):
    if name in function.__globals__:return function.__globals__[name]
    if name in function.__builtins__:return function.__builtins__[name]
    raise NameError("name '"+name+"' is not defined",name=name)


def _observers(codes):
    if _GETTRACE()is not None or _GETPROFILE()is not None:return True
    if _MONITOR_APIS is not None:
        get_tool,get_events,get_local=_MONITOR_APIS
        for tool in _RANGE(6):
            if get_tool(tool)is not None or get_events(tool)or any(get_local(tool,code)for code in codes):return True
    return False


def guarded_stateful_loop(original,artifact,*,exclusive_inputs=False,trial=False):
    """Run one source-proven homogeneous loop or its exact original semantics.

    A native failure with committed progress k resumes only original steps
    k..N. Every suffix CALL resolves the current global/builtin callee using
    the retained source positional/keyword layout, not its new signature.
    """
    if type(artifact)is not NativeStatefulRepeat:raise TypeError('compact stateful-repeat trial artifact required')
    contract=validate_contract(artifact.loop_contract)
    return _guarded_stateful_loop_core(original,artifact,contract,
        exclusive_inputs=exclusive_inputs,trial=trial)


def _guarded_stateful_loop_core(original,artifact,contract,*,exclusive_inputs=False,trial=False):
    """Shared live source/argument/progress guards after profile admission.

    Public factories own their distinct artifact type and contract validation.
    This core does not admit either profile or manufacture a runtime contract.
    """
    step_contract=contract['step_contract']
    if type(original)is not FunctionType:raise ValueError('ordinary original caller function required')
    stats=dict(native_calls=0,fallback_calls=0,reasons={},native_step_calls=0,
        fallback_step_calls=0,original_loop_calls=0,partial_native_prefixes=0)
    initial_error=None;callees={}
    try:
        _plain(original,contract['source']);outer_state=_function_state(original)
        signature=inspect.signature(FunctionType(original.__code__,original.__globals__),follow_wrapped=False)
        if list(signature.parameters)!=contract['parameters']:raise ValueError('original caller parameters differ')
        step=_name(original,contract['step_global_name']);_plain(step,contract['step_source'])
        step_state=_function_state(step)
        step_signature=inspect.signature(FunctionType(step.__code__,step.__globals__),follow_wrapped=False)
        if list(step_signature.parameters)!=step_contract['parameters']:raise ValueError('original step parameters differ')
    except (OSError,ValueError,TypeError,KeyError,NameError):initial_error='original_loop_step_source_or_abi_not_admitted'

    def reason(name):stats['reasons'][name]=stats['reasons'].get(name,0)+1
    def original_loop(name,args,kwargs):
        stats['fallback_calls']+=1;stats['original_loop_calls']+=1;reason(name)
        return original(*args,**kwargs)

    @wraps(original,assigned=('__module__','__name__','__qualname__','__doc__'))
    def call(*args,**kwargs):
        if trial is not True:return original_loop('NO_OP_unmeasured_loop_trial_required',args,kwargs)
        if exclusive_inputs is not True or threading.active_count()!=1:
            return original_loop('exclusive_synchronous_state_contract_required',args,kwargs)
        if initial_error:return original_loop(initial_error,args,kwargs)
        try:
            bound=signature.bind(*args,**kwargs);bound.apply_defaults()
            count=bound.arguments[contract['trip_count_parameter']]
        except (TypeError,KeyError):return original_loop('original_loop_argument_binding',args,kwargs)
        # Preserve range(bool/index/negative/zero) and avoid touching unused
        # receiver fields or providers on the source zero-trip path.
        if type(count)is not int or count<=0 or count!=contract['trip_count']:
            return original_loop('outside_positive_exact_loop_count',args,kwargs)
        try:
            _plain(original,contract['source']);_plain(step,contract['step_source'])
            if (not _same_function_state(original,outer_state)or not _same_function_state(step,step_state)
                    or _name(original,contract['step_global_name'])is not step
                    or type(_RANGE)is not type or _RANGE.__module__!='builtins'or _RANGE.__name__!='range'
                    or _name(original,'range')is not _RANGE):
                raise ValueError('original caller/global step/default/range changed')
            if (_NDARRAY.__name__!='ndarray'or _NDARRAY.__module__!='numpy'
                    or not _NDARRAY.__flags__&256 or np.ndarray is not _NDARRAY or np.float64 is not _FLOAT64):
                raise ValueError('canonical ndarray/scalar runtime aliases changed')
            layout=contract['call_arguments']
            positional=[bound.arguments[name]for name in layout['positional_parameters']]
            keywords={name:bound.arguments[outer]for name,outer in layout['keyword_parameters']}
            step_bound=step_signature.bind(*positional,**keywords);step_bound.apply_defaults()
            if any(step_bound.arguments[name]is not bound.arguments[outer]
                   for name,outer in contract['parameter_map'].items()):
                raise ValueError('original step parameter routing differs')
            codes=[original.__code__,step.__code__]
            for guard in step_contract['callee_guards']:
                path=tuple(guard['path']);fn=_lookup(path,step_bound.arguments,step.__globals__)
                _plain(fn,guard['source'])
                if fn.__globals__ is not step.__globals__:raise ValueError('same-source nested callee globals differ')
                saved=callees.get(path)
                if saved is None:saved=(fn,_function_state(fn));callees[path]=saved
                if fn is not saved[0]or not _same_function_state(fn,saved[1]):raise ValueError('nested stateful callee changed')
                codes.append(fn.__code__)
            if _observers(codes):raise ValueError('source metadata observer active')
            for provider in step_contract['providers']:
                if _lookup(provider['lookup_path'],step_bound.arguments,step.__globals__)is not artifact.base._power:
                    raise ValueError('original native provider changed')
            arrays={};scalars={}
            for spec in step_contract['inputs']:
                value=_input(spec['path'],step_bound.arguments)
                if spec['kind']=='array':
                    if(type(value)is not _NDARRAY or value.dtype!=np.dtype('float64')or not value.dtype.isnative
                            or value.ndim!=2 or not value.flags.c_contiguous or not value.flags.aligned
                            or spec['written']and not value.flags.writeable):
                        raise ValueError('original state array type/layout/writeability differs')
                    arrays[spec['id']]=value
                else:
                    if(type(value)not in(int,float,_FLOAT64)or type(value)is int and abs(value)>2**53
                            or not math.isfinite(value)):
                        raise ValueError('finite exact original scalar required')
                    scalars[spec['id']]=value if type(value)is int else float(value)
        except (OSError,ValueError,TypeError,KeyError,AttributeError,StopIteration,NameError):
            return original_loop('live_original_loop_step_or_input_guard_declined',args,kwargs)
        try:result=artifact.run(arrays,scalars,count)
        except RepeatGuardFailure as error:
            completed=error.completed
            if completed==0:return original_loop('native_repeat_declined_before_first_commit',args,kwargs)
            stats['native_step_calls']+=completed;stats['partial_native_prefixes']+=1
            stats['fallback_calls']+=1;reason('native_repeat_partial_prefix_resume_original_suffix')
            for _ in _RANGE(completed,count):
                stats['fallback_step_calls']+=1
                _name(original,contract['step_global_name'])(*positional,**keywords)
            return None
        stats['native_calls']+=1;stats['native_step_calls']+=result['completed']
        return None
    call.scar_stats=stats;call.scar_artifact=artifact
    return call


__all__=['guarded_stateful_loop']

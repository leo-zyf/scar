"""Shared live source/receiver/provider guards for stateful native delivery."""
from contextlib import contextmanager
from functools import wraps
from hashlib import sha256
import inspect
import math
from pathlib import Path
import threading
import types

from scar.trace.code_snapshot import code_key
from .native_stateful import GuardFailure
from .stateful_contract import validate_contract


class SourceMismatch(ValueError):
    pass


def _function(source,value):
    def get(name): return source[name] if type(source) is dict else getattr(source,name)
    if (type(value) is not types.FunctionType or value.__closure__
            or code_key(value.__code__)!=get('code_key')
            or sha256(Path(get('path')).read_bytes()).hexdigest()!=get('sha256')):
        raise SourceMismatch('stateful source/code/closure differs from discovered boundary')
    return value


def _ordinary(value):
    cls=type(value)
    if type(cls) is not type: raise ValueError('custom receiver metaclass')
    for base in cls.__mro__:
        if base is object: continue
        if any(name in vars(base) for name in ('__getattribute__','__getattr__','__setattr__')):
            raise ValueError('custom receiver attribute protocol')
    dictionary=next((vars(base)['__dict__'] for base in cls.__mro__ if '__dict__' in vars(base)),None)
    if (type(dictionary) is not types.GetSetDescriptorType or dictionary.__name__!='__dict__'
            or dictionary.__objclass__ not in cls.__mro__):
        raise ValueError('receiver requires the ordinary instance dictionary descriptor')
    fields=object.__getattribute__(value,'__dict__')
    if type(fields) is not dict: raise ValueError('receiver requires ordinary instance fields')
    return fields


def _field(value,name):
    fields=_ordinary(value)
    for base in type(value).__mro__:
        if name in vars(base):
            descriptor=vars(base)[name]; descriptor_type=type(descriptor)
            if type(descriptor_type) is not type: raise ValueError('state descriptor has a custom metaclass')
            if any('__set__' in vars(parent) or '__delete__' in vars(parent) for parent in descriptor_type.__mro__):
                raise ValueError('state input has a data descriptor')
            break
    return fields[name]


def _input(path,bound):
    value=bound[path[0]]
    for name in path[1:]: value=_field(value,name)
    return value


def _lookup(path,bound,globals):
    value=bound[path[0]] if path[0] in bound else globals[path[0]]
    for name in path[1:]:
        if type(value) is types.ModuleType: value=vars(value)[name]
        elif type(value) is type:
            descriptor=vars(value)[name]
            if type(descriptor) is not staticmethod: raise ValueError('callee requires exact staticmethod')
            value=descriptor.__func__
        else:
            fields=_ordinary(value)
            if name in fields: raise ValueError('instance shadows a discovered callee')
            descriptor=next(vars(base)[name] for base in type(value).__mro__ if name in vars(base))
            if type(descriptor) is not staticmethod: raise ValueError('callee requires exact staticmethod')
            value=descriptor.__func__
    return value


def _function_state(fn):
    return fn.__code__,fn.__defaults__,tuple((key,id(value)) for key,value in (fn.__kwdefaults__ or {}).items())


def _same_function_state(fn,state):
    return (fn.__code__ is state[0] and fn.__defaults__ is state[1]
            and tuple((key,id(value)) for key,value in (fn.__kwdefaults__ or {}).items())==state[2])


def guarded_stateful_function(original,artifact,*,owner=None,exclusive_inputs=False,selection=None,trial=False):
    import numpy as np
    contract=validate_contract(artifact.contract,require_selection=False)
    _function(contract['source'],original)
    if '__signature__' in vars(original) or '__wrapped__' in vars(original):
        raise SourceMismatch('custom function signature metadata is outside the discovered call ABI')
    signature=inspect.signature(original); state=_function_state(original); identity=artifact.report
    selection=contract['selection'] if selection is None else selection
    stats=dict(native_calls=0,fallback_calls=0,reasons={}); callee_states={}
    for callee in contract['callee_guards']:
        path=tuple(callee['path'])
        if path[0] not in signature.parameters:
            fn=_function(callee['source'],_lookup(path,{},original.__globals__))
            callee_states[path]=(fn,_function_state(fn))
    provider_targets={}
    for provider in contract['providers']:
        if provider['module']!='numpy' or provider['symbol']!='power' or type(np.power) is not np.ufunc:
            raise SourceMismatch('stateful provider has no admitted runtime binding')
        provider_targets[provider['id']]=np.power

    def fallback(reason,args,kwargs):
        stats['fallback_calls']+=1; stats['reasons'][reason]=stats['reasons'].get(reason,0)+1
        return original(*args,**kwargs)

    @wraps(original)
    def call(*args,**kwargs):
        if not exclusive_inputs or threading.active_count()!=1:
            return fallback('exclusive_synchronous_state_contract',args,kwargs)
        if not _same_function_state(original,state) or '__signature__' in vars(original) or '__wrapped__' in vars(original):
            return fallback('function_state_changed',args,kwargs)
        try:
            bound=signature.bind(*args,**kwargs); bound.apply_defaults()
            if owner is not None and type(bound.arguments[contract['parameters'][0]]) is not owner:
                raise ValueError('receiver class differs')
            for callee in contract['callee_guards']:
                path=tuple(callee['path']); fn=_lookup(path,bound.arguments,original.__globals__)
                current=callee_states.get(path)
                if current is None:
                    fn=_function(callee['source'],fn); current=(fn,_function_state(fn)); callee_states[path]=current
                if fn is not current[0] or not _same_function_state(fn,current[1]): raise ValueError('nested callee changed')
            for provider in contract['providers']:
                if _lookup(provider['lookup_path'],bound.arguments,original.__globals__) is not provider_targets[provider['id']]:
                    raise ValueError('native provider changed')
            arrays,scalars={},{}
            for spec in contract['inputs']:
                value=_input(spec['path'],bound.arguments)
                if spec['kind']=='array':
                    if (type(value) is not np.ndarray or value.dtype!=np.float64 or value.ndim!=2
                            or not value.flags.c_contiguous or spec['written'] and not value.flags.writeable):
                        raise ValueError('array type/layout differs')
                    arrays[spec['id']]=value
                else:
                    if (type(value) not in (int,float,np.float64) or type(value) is int and abs(value)>2**53
                            or not math.isfinite(value)): raise ValueError('finite exact scalar required')
                    scalars[spec['id']]=value if type(value) is int else float(value)
            values=list(arrays.values())
            for index,value in enumerate(values):
                if any(np.may_share_memory(value,other) for other in values[:index]): raise ValueError('external array roots may alias')
        except (TypeError,ValueError,KeyError,AttributeError,StopIteration) as error:
            return fallback(str(error),args,kwargs)
        if not trial:
            pinned=selection.get('artifact',{}) if type(selection) is dict else {}
            if (not selection or selection.get('decision')!='TRANSFORM_UNDER_CONTRACT'
                    or any(pinned.get(key)!=identity.get(key) or pinned.get(key) is None for key in ('build_id','plan_sha256'))):
                return fallback('no_measured_selection_for_stateful_plan',args,kwargs)
            shapes=selection.get('input_shapes',{})
            if type(shapes) is not dict or any(list(value.shape)!=shapes.get(name) for name,value in arrays.items()):
                return fallback('outside_measured_state_shapes',args,kwargs)
        try: result=artifact.run(arrays,scalars)
        except GuardFailure as error: return fallback('native_precommit_guard:'+str(error),args,kwargs)
        stats['native_calls']+=1
        return result
    call.scar_stats,call.scar_artifact=stats,artifact
    return call


@contextmanager
def install_stateful_regions(module,artifacts,*,exclusive_inputs=False,selection=None,trial=False):
    changes,wrappers=[],[]
    try:
        for artifact in artifacts:
            parts=artifact.contract['source']['qualname'].split('.'); owner=module
            for part in parts[:-1]:
                if part=='<locals>' or type(owner) not in (types.ModuleType,type): raise SourceMismatch('stateful boundary has no ordinary module/class path')
                owner=vars(owner)[part]
            if type(owner) not in (types.ModuleType,type): raise SourceMismatch('custom boundary owner')
            name=parts[-1]; descriptor=vars(owner)[name]
            if type(descriptor) is staticmethod: original,wrap,receiver_owner=descriptor.__func__,staticmethod,None
            elif type(descriptor) is types.FunctionType: original,wrap,receiver_owner=descriptor,lambda value:value,owner if type(owner) is type else None
            else: raise SourceMismatch('stateful boundary requires an ordinary function descriptor')
            wrapper=guarded_stateful_function(original,artifact,owner=receiver_owner,exclusive_inputs=exclusive_inputs,selection=selection,trial=trial)
            installed=wrap(wrapper); setattr(owner,name,installed); changes.append((owner,name,descriptor,installed)); wrappers.append(wrapper)
        yield wrappers
    finally:
        for owner,name,descriptor,installed in reversed(changes):
            if vars(owner).get(name) is installed: setattr(owner,name,descriptor)


__all__=['SourceMismatch','guarded_stateful_function','install_stateful_regions']

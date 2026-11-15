"""Observe one original stateful call without graphs or native execution.

All source, callable, receiver, provider and input checks run outside the
clock. The sole timed operation is ORIGINAL(*args, **kwargs) through return
or raise. This is a guarded wrapper-dispatch cost, not an entry benchmark.
"""
import argparse
from functools import wraps
from hashlib import sha256
import inspect
import json
import math
import os
from pathlib import Path
import runpy
import sys
import threading
import time
from types import FunctionType,ModuleType


_BASIS='OBSERVED_ORIGINAL_CALL_WITH_WRAPPER_DISPATCH_COST_MODEL'
_RANGE=range


def _json(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def _digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _receipt(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return dict(path=str(path),sha256=sha256(raw).hexdigest(),bytes=len(raw))


def _resolve(module,qualname):
    """Traverse exact module/class dictionaries, never application getattr."""
    parts=qualname.split('.')
    if any(not part.isidentifier()for part in parts):raise ValueError('ordinary source qualname required')
    owner=module
    for part in parts[:-1]:
        if type(owner)not in(ModuleType,type):raise ValueError('custom boundary owner')
        owner=vars(owner)[part]
    if type(owner)not in(ModuleType,type):raise ValueError('custom boundary owner')
    descriptor=vars(owner)[parts[-1]]
    if type(descriptor)is staticmethod:return owner,parts[-1],descriptor,descriptor.__func__,True
    if type(descriptor)is FunctionType:return owner,parts[-1],descriptor,descriptor,False
    raise ValueError('ordinary function or exact staticmethod descriptor required')


def worker(spec_path,sample_path):
    from scar.runtime.entry_runner import _file,_script_state
    from scar.runtime.stateful_guard import (_function,_ordinary,_input,_lookup,
        _function_state,_same_function_state)
    from scar.runtime.compiled_deployment import entry_context,thread_environment

    spec=json.loads(Path(spec_path).read_bytes());sample_path=Path(sample_path)
    if (spec.get('schema'),spec.get('schema_version'))!=('scar.original-stateful-observation-design',1):
        raise ValueError('versioned frozen original stateful observation design required')
    if sample_path.exists():raise ValueError('fresh original stateful sample path required')
    entry=Path(spec['entry']).resolve();argv=list(spec['argv']);observer_pid=os.getpid()
    observed=dict(schema='scar.original-stateful-call-sample',schema_version=1,
        phase=spec['phase'],index=spec['index'],worker_nonce=spec['worker_nonce'],
        frozen_design=spec['frozen_design'],process_id=observer_pid,basis=_BASIS,
        status='NOT_OBSERVED',recorded_calls=0,duration_ns=None,native_calls=0,
        function_internals_instrumented=False,global_trace_or_profile_installed=False,
        tensor_values_recorded=False,scalar_values_recorded=False,target_workload_executed=True,
        target_variants_executed=False,target_variants_ranked=False,total_calls_collected=False,
        clean_whole_entry_benchmark=False,input_content_identity_proven=False,
        clock_includes_wrapper_dispatch=True,scope='ONE_ORIGINAL_CALL_DISPATCH_THROUGH_RETURN_OR_RAISE')
    captured=[False];initial_reasons=[]

    def context():return entry_context(entry,argv)
    def receipt_reasons():
        reasons=[]
        try:
            if _digest(dict(os.environ))!=spec['environment_sha256']:reasons.append('FROZEN_EFFECTIVE_ENVIRONMENT_CHANGED')
            if context()!=spec['entry_context']:reasons.append('FROZEN_ENTRY_OR_THREAD_CONTEXT_CHANGED')
            for source in spec['sources']:
                if _receipt(source['path'])['sha256']!=source['sha256']:
                    reasons.append('ORIGINAL_SOURCE_VERSION_CHANGED');break
            for receipt in spec['tool_receipts']:
                if _receipt(receipt['path'])!=receipt:reasons.append('OBSERVER_OR_RUNTIME_IMPLEMENTATION_CHANGED');break
            design=spec['frozen_design'];raw=Path(design['uri']).read_bytes()
            if (len(raw)!=design['size_bytes']or sha256(raw).hexdigest()!=design['sha256']
                    or any(spec.get(key)!=value for key,value in json.loads(raw).items())):
                reasons.append('FROZEN_DESIGN_RECEIPT_OR_FIELDS_CHANGED')
        except (OSError,ValueError,TypeError,KeyError):reasons.append('FROZEN_CONTEXT_RECEIPT_UNAVAILABLE')
        return reasons

    def observer_reasons():
        reasons=[]
        if sys.gettrace()is not None or sys.getprofile()is not None:reasons.append('EXISTING_GLOBAL_OBSERVER')
        monitoring=getattr(sys,'monitoring',None)
        if monitoring is not None and any(monitoring.get_tool(i)is not None for i in _RANGE(6)):
            reasons.append('EXISTING_MONITORING_OBSERVER')
        if threading.active_count()!=1:reasons.append('EXCLUSIVE_SYNCHRONOUS_STATE_NOT_ESTABLISHED')
        return reasons

    initial_reasons.extend(receipt_reasons());initial_reasons.extend(observer_reasons())
    # Freeze canonical providers before application imports can replace them.
    import numpy as np
    canonical_power=np.power;numpy_version=np.__version__
    ndarray_type=np.ndarray;float64_type=np.float64;ufunc_type=np.ufunc
    float64_dtype=np.dtype('float64');may_share_memory=np.may_share_memory
    provider_runtime=spec.get('provider_runtime')
    def provider_runtime_reasons():
        reasons=[]
        if np.__version__!=numpy_version:reasons.append('NUMPY_VERSION_CHANGED')
        if provider_runtime is not None:
            try:
                if numpy_version!=provider_runtime['numpy_version']:reasons.append('NUMPY_VERSION_OUTSIDE_FROZEN_CONTEXT')
                binary=provider_runtime['loaded_binary']
                loaded=any(type(module)is ModuleType and _file(module)==Path(binary['path']).resolve()
                    for name,module in tuple(sys.modules.items())if name.startswith('numpy.'))
                if not loaded or _receipt(binary['path'])!=binary:reasons.append('NUMPY_PROVIDER_BINARY_CHANGED')
            except (OSError,ValueError,TypeError,KeyError):reasons.append('NUMPY_PROVIDER_RUNTIME_UNAVAILABLE')
        return reasons
    initial_reasons.extend(provider_runtime_reasons())

    class Scope:
        source=Path(spec['source']['path']).resolve()
        def __init__(self):self.frontier=[];self.seen=set();self.bindings=[];self.active=True

        def attach(self,module,name,*,natural):
            if not self.active or id(module)in self.seen:return
            self.seen.add(id(module))
            if _file(module)!=self.source:return
            if not natural:
                self.frontier.append('ORIGINAL_SOURCE_NOT_A_FRESH_NATURAL_IMPORT');return
            try:
                owner,attribute,descriptor,original,static=_resolve(module,spec['source']['qualname'])
                _function(spec['source'],original)
                if (type(original.__globals__)is not dict or type(original.__builtins__)is not dict
                        or '__signature__'in vars(original)or'__wrapped__'in vars(original)):
                    raise ValueError('ordinary original function ABI required')
                shadow=FunctionType(original.__code__,original.__globals__,argdefs=original.__defaults__)
                shadow.__kwdefaults__=original.__kwdefaults__
                signature=inspect.signature(shadow,follow_wrapped=False)
                if tuple(signature.parameters)!=tuple(spec['parameters']):raise ValueError('original parameter order differs')
                state=_function_state(original);callee_states=[]
                for callee in spec['callee_guards']:
                    if callee['kind']!='STATICMETHOD':raise ValueError('unadmitted callee descriptor')
                    _,_,_,target,is_static=_resolve(module,callee['source']['qualname'])
                    if not is_static:raise ValueError('callee is not an exact original staticmethod')
                    _function(callee['source'],target)
                    callee_states.append((callee,target,_function_state(target)))
                receiver_owner=owner if type(owner)is type and not static else None
            except (OSError,ValueError,TypeError,KeyError,AttributeError)as error:
                self.frontier.append('ORIGINAL_BOUNDARY_ADMISSION_REJECTED:'+type(error).__name__);return

            installed=[None]
            def guards(args,kwargs):
                reasons=[*initial_reasons,*receipt_reasons(),*observer_reasons(),*provider_runtime_reasons()]
                metadata=dict(numpy_version=numpy_version,arrays={},scalars={},callee_guards_checked=0,
                    provider_guards_checked=0,body_internals_instrumented=False,native_calls=0)
                try:
                    _function(spec['source'],original)
                    if (not _same_function_state(original,state)or'__signature__'in vars(original)
                            or'__wrapped__'in vars(original)or vars(owner).get(attribute)is not installed[0]):
                        raise ValueError('original function state or descriptor changed')
                    bound=signature.bind(*args,**kwargs);bound.apply_defaults()
                    if receiver_owner is not None:
                        receiver=bound.arguments[spec['parameters'][0]]
                        if type(receiver)is not receiver_owner:raise ValueError('receiver class differs')
                        _ordinary(receiver)
                    for callee,target,target_state in callee_states:
                        current=_lookup(callee['path'],bound.arguments,original.__globals__)
                        _function(callee['source'],current)
                        if current is not target or not _same_function_state(current,target_state):
                            raise ValueError('nested source callee identity/default state changed')
                        metadata['callee_guards_checked']+=1
                    for provider in spec['providers']:
                        if (provider['module']!='numpy'or provider['symbol']!='power'
                                or type(canonical_power)is not ufunc_type or np.power is not canonical_power
                                or _lookup(provider['lookup_path'],bound.arguments,original.__globals__)is not canonical_power):
                            raise ValueError('canonical source provider identity changed')
                        metadata['provider_guards_checked']+=1
                    arrays=[]
                    for row in spec['inputs']:
                        value=_input(row['path'],bound.arguments)
                        if row['kind']=='array':
                            if type(value)is not ndarray_type:
                                metadata['arrays'][row['id']]=dict(exact_ndarray=False)
                                raise ValueError('input is not an exact ndarray')
                            metadata['arrays'][row['id']]=dict(exact_ndarray=True,shape=list(value.shape),
                                dtype=value.dtype.str,rank=value.ndim,strides=list(value.strides),nbytes=value.nbytes,
                                c_contiguous=bool(value.flags.c_contiguous),aligned=bool(value.flags.aligned),
                                writeable=bool(value.flags.writeable))
                            if (value.dtype!=float64_dtype or not value.dtype.isnative or value.ndim!=2
                                    or not value.flags.c_contiguous or not value.flags.aligned
                                    or list(value.shape)!=row['shape']or value.nbytes!=row['nbytes']
                                    or row['written']and not value.flags.writeable):
                                raise ValueError('array metadata outside exact cost context')
                            if any(may_share_memory(value,other)for other in arrays):
                                raise ValueError('external array roots may alias')
                            arrays.append(value)
                        elif row['kind']=='scalar':
                            valid=(type(value)in(int,float,float64_type)
                                and not(type(value)is int and abs(value)>2**53)and math.isfinite(value))
                            metadata['scalars'][row['id']]=dict(exact_supported_type=type(value)in(int,float,float64_type),
                                finite_exact_range=bool(valid))
                            if not valid:raise ValueError('scalar outside finite exact range')
                        else:raise ValueError('unknown admitted state input kind')
                except (OSError,ValueError,TypeError,KeyError,AttributeError,StopIteration)as error:
                    reasons.append('ORIGINAL_CALL_GUARD_REJECTED:'+str(error))
                return tuple(dict.fromkeys(reasons)),metadata

            def checked_guards(args,kwargs):
                try:return guards(args,kwargs)
                except Exception as error:
                    return ('OBSERVER_GUARD_UNAVAILABLE:'+type(error).__name__,),{}

            @wraps(original)
            def wrapped(*args,**kwargs):
                if not self.active or os.getpid()!=observer_pid or captured[0]:return original(*args,**kwargs)
                captured[0]=True;observed['recorded_calls']=1
                reasons,metadata=checked_guards(args,kwargs);observed['metadata']=metadata
                if reasons:
                    observed.update(status='OUTSIDE_FROZEN_CONTEXT',reason=list(reasons))
                    return original(*args,**kwargs)
                start=time.perf_counter_ns()
                try:result=original(*args,**kwargs)
                except BaseException as error:
                    end=time.perf_counter_ns()
                    observed.update(status='ORIGINAL_RAISED',duration_ns=end-start,exception_type=type(error).__name__)
                    raise
                else:
                    end=time.perf_counter_ns()
                    observed.update(status='OBSERVED_ORIGINAL_RETURN',duration_ns=end-start)
                    return result
                finally:
                    # State array contents may change. Recheck only the same
                    # source/callable/provider/context and metadata contract.
                    after,post_metadata=checked_guards(args,kwargs)
                    observed['post_call_metadata']=post_metadata
                    if after:observed.update(status='OUTSIDE_FROZEN_CONTEXT',reason=list(after),duration_ns=None)

            installed[0]=staticmethod(wrapped)if static else wrapped
            setattr(owner,attribute,installed[0])
            self.bindings.append((owner,attribute,descriptor,installed[0],wrapped,original))

        def finish(self):
            self.active=False
            replacements={id(row[4]):(row[4],row[5])for row in self.bindings}
            for module in tuple(sys.modules.values()):
                if type(module)is not ModuleType:continue
                namespace=vars(module)
                for name,value in tuple(namespace.items()):
                    pair=replacements.get(id(value))
                    if pair is not None and value is pair[0]:namespace[name]=pair[1]
            for owner,name,descriptor,installed,_,_ in reversed(self.bindings):
                if vars(owner).get(name)is installed:setattr(owner,name,descriptor)

    scope=Scope()
    try:
        with _script_state(entry,argv,scope):
            try:runpy.run_path(str(entry),run_name='__main__')
            except SystemExit as error:
                code=error.code
                if code is not None and not(isinstance(code,int)and int.__int__(code)==0):raise
    finally:
        final_reasons=tuple(dict.fromkeys((*initial_reasons,*receipt_reasons(),*observer_reasons(),*provider_runtime_reasons())))
        if final_reasons:observed.update(status='OUTSIDE_FROZEN_CONTEXT',reason=list(final_reasons),duration_ns=None)
        try:actual=context()
        except (OSError,ValueError,TypeError):
            actual=None;observed.update(status='OUTSIDE_FROZEN_CONTEXT',reason=['FINAL_ENTRY_CONTEXT_UNAVAILABLE'],duration_ns=None)
        observed.update(entry_context=actual,frontier=scope.frontier,
            observed_caller_affinity_cpu_ids=sorted(os.sched_getaffinity(0)),
            observed_native_thread_environment=thread_environment(),actual_full_thread_environment=thread_environment(),
            restored_bindings=True,observer=_receipt(__file__))
        with sample_path.open('xb')as stream:stream.write(_json(observed))
    return observed


_worker=worker


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True);parser.add_argument('--sample',required=True)
    args=parser.parse_args(argv);worker(args.spec,args.sample);return 0


if __name__=='__main__':raise SystemExit(main())

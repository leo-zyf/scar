"""Explicit, guarded Python boundary around a compiled C++ region.

The C++ host owns region iteration. The caller retains whole-program effects.
No memoized outputs; fallback calls the original function exactly once. Use
under the declared synchronous CPU/no-foreign-writer contract only.
"""
from contextlib import contextmanager
from functools import wraps
from hashlib import sha256
import inspect
import os
from pathlib import Path
import threading
import types
from scar.trace.code_snapshot import code_key
from scar.codegen import GuardFailure

class SourceMismatch(ValueError):pass

def guarded_function(original,artifact,*,exclusive_inputs=False,selection=None,trial=False,observer=None):
    source=artifact.plan.region.source
    if sha256(Path(source.path).read_bytes()).hexdigest()!=source.sha256 or code_key(original.__code__)!=source.code_key:
        raise SourceMismatch('source/code version differs from compiled region')
    binding=artifact.plan.resource_binding
    if binding is not None:
        from scar.optimization.hardware import validate_cpu_placement
        validate_cpu_placement(binding)
    code=original.__code__;defaults=original.__defaults__
    kwdefaults={k:id(v) for k,v in (original.__kwdefaults__ or {}).items()}
    signature=inspect.signature(original);stats=dict(native_calls=0,fallback_calls=0,reasons={})
    # Artifact identity is frozen at installation. Copying the full metadata
    # document on every invocation is unnecessary guard overhead.
    artifact_identity=artifact.report
    plan_sha256=artifact_identity['plan_sha256']
    observer_active=observer is not None
    def fallback(reason,args,kwargs):
        stats['fallback_calls']+=1;stats['reasons'][reason]=stats['reasons'].get(reason,0)+1
        return original(*args,**kwargs)
    @wraps(original)
    def call(*args,**kwargs):
        nonlocal observer_active
        if not exclusive_inputs or threading.active_count()!=1:
            return fallback('input_ownership_or_thread_contract',args,kwargs)
        if binding is not None and (not hasattr(os,'sched_getaffinity')
                or not set(binding.cpu_ids)<=os.sched_getaffinity(0)):
            return fallback('planned_cpu_allocation_unavailable',args,kwargs)
        if (original.__code__ is not code or original.__defaults__ is not defaults
                or {k:id(v) for k,v in (original.__kwdefaults__ or {}).items()}!=kwdefaults):
            return fallback('function_state_changed',args,kwargs)
        try:bound=signature.bind(*args,**kwargs)
        except TypeError:return fallback('call_signature',args,kwargs)
        bound.apply_defaults()
        values=[bound.arguments[p] for p in artifact.plan.region.parameters]
        if 'EXACT_NUMPY_NDARRAY_COPY_METHOD_DESCRIPTOR' in artifact.plan.region.assumptions:
            from scar.runtime.array_copy_guard import guard_copy_inputs,CopyGuardFailure
            try:guard_copy_inputs(values)
            except CopyGuardFailure as error:return fallback(str(error),args,kwargs)
        if observer_active and observer(source,artifact.plan.region.parameters,values) is False:
            observer_active=False
        if not trial:
            if (not selection or selection.get('decision')!='TRANSFORM_UNDER_CONTRACT'
                    or selection.get('artifact',{}).get('plan_sha256')!=plan_sha256
                    or selection.get('artifact',{}).get('build_id')!=artifact_identity['build_id']
                    or selection.get('artifact',{}).get('resource_schedule_sha256')!=artifact_identity.get('resource_schedule_sha256')):
                return fallback('no_measured_selection_for_plan',args,kwargs)
            shape=tuple(selection.get('shape') or ())
            input_shapes=selection.get('input_shapes')
            import numpy as np
            if input_shapes is not None and type(input_shapes)is not dict:
                return fallback('outside_measured_shape_or_layout',args,kwargs)
            for parameter,value in zip(artifact.plan.region.parameters,values):
                expected_shape=tuple(input_shapes.get(parameter,())) if input_shapes is not None else shape
                if (type(value)is not np.ndarray or value.dtype!=np.float64
                        or value.shape!=expected_shape or not value.flags.c_contiguous):
                    return fallback('outside_measured_shape_or_layout',args,kwargs)
        try:result=artifact.run(*values)
        except GuardFailure as error:return fallback(type(error).__name__+':'+str(error),args,kwargs)
        stats['native_calls']+=1
        return result
    call.scar_stats=stats
    call.scar_artifact=artifact
    return call

@contextmanager
def install_regions(module,artifacts,*,exclusive_inputs=False,selection=None,trial=False,observer=None):
    """Explicit workload boundary; selection uses versioned discovered regions."""
    changes=[];wrappers=[]
    try:
        for artifact in artifacts:
            parts=artifact.plan.region.source.qualname.split('.')
            owner=module
            for part in parts[:-1]:
                if part=='<locals>':raise SourceMismatch('nested local functions require an explicit captured binding')
                if type(owner) not in (types.ModuleType,type):raise SourceMismatch('custom lookup protocols require an explicit boundary')
                owner=vars(owner)[part]
            if type(owner) not in (types.ModuleType,type):raise SourceMismatch('custom metaclass effects are not admitted')
            name=parts[-1];descriptor=vars(owner)[name]
            if type(descriptor) is staticmethod:original=descriptor.__func__;wrap=staticmethod
            elif inspect.isfunction(descriptor) and owner is module:original=descriptor;wrap=lambda fn:fn
            else:raise SourceMismatch('only module functions and exact staticmethod descriptors supported')
            wrapped=guarded_function(original,artifact,exclusive_inputs=exclusive_inputs,selection=selection,trial=trial,observer=observer)
            changes.append((owner,name,descriptor));setattr(owner,name,wrap(wrapped));wrappers.append(wrapped)
        yield wrappers
    finally:
        for owner,name,descriptor in reversed(changes):setattr(owner,name,descriptor)

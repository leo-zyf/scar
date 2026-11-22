"""Original-callable admission around dependency-derived invariant placement."""
from functools import wraps
from dataclasses import asdict
import builtins
import inspect
from pathlib import Path
from hashlib import sha256
import sys
import threading
from types import FunctionType

from scar.compiler.pipeline_runtime import _state, _same_state
from scar.compiler.runtime import SourceMismatch
from scar.trace.code_snapshot import code_key
from scar.optimization.resource_solver import semantic_sha256

_RANGE=range


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
    parameter_map=dict(proof.parameter_map)
    snapshot=_state(original);leaf_state=_state(leaf)
    signature_source=FunctionType(original.__code__,original.__globals__,argdefs=original.__defaults__)
    signature_source.__kwdefaults__=original.__kwdefaults__
    signature=inspect.signature(signature_source,follow_wrapped=False)
    stats=dict(native_calls=0,fallback_calls=0,reasons={})
    def fallback(reason,args,kwargs):
        stats['fallback_calls']+=1;stats['reasons'][reason]=stats['reasons'].get(reason,0)+1
        return original(*args,**kwargs)
    def exact_range():
        namespace=original.__globals__
        return namespace['range'] if 'range' in namespace else original.__builtins__.get('range')
    @wraps(original)
    def run(*args,**kwargs):
        if not trial:return fallback('no_validated_cost_selection',args,kwargs)
        if not exclusive_inputs or threading.active_count()!=1 or sys.gettrace() is not None or sys.getprofile() is not None:
            return fallback('exclusive_synchronous_inputs_required',args,kwargs)
        monitoring=getattr(sys,'monitoring',None)
        if monitoring is not None and any(monitoring.get_tool(i) is not None for i in range(6)):
            return fallback('monitoring_observer_active',args,kwargs)
        if (not _same_state(original,snapshot) or original.__globals__.get(proof.leaf_global_name) is not leaf
                or not _same_state(leaf,leaf_state) or exact_range() is not _RANGE):
            return fallback('loop_leaf_or_range_binding_changed',args,kwargs)
        try:bound=signature.bind(*args,**kwargs)
        except TypeError:return fallback('original_argument_binding',args,kwargs)
        bound.apply_defaults();count=bound.arguments[proof.trip_count_parameter]
        if type(count) is not int or not -(2**63)<=count<2**63:
            return fallback('trip_count_outside_int64',args,kwargs)
        inputs={leaf_name:bound.arguments[outer] for leaf_name,outer in parameter_map.items()}
        specialization=artifact.plan['loop_interface'].get('execution_specialization')
        if specialization is not None:
            if count!=specialization['trip_count']:
                return fallback('outside_sourced_execution_specialization',args,kwargs)
            import numpy as np
            if any(type(value) is not np.ndarray for value in inputs.values()):
                return fallback('outside_exact_array_input_contract',args,kwargs)
            if any(list(inputs[name].shape)!=shape for name,shape in specialization['input_shapes'].items()) or (
                    specialization.get('input_layout')=='C_CONTIGUOUS' and any(not value.flags.c_contiguous for value in inputs.values())):
                return fallback('outside_sourced_execution_specialization',args,kwargs)
        failed=[False]
        def original_once():
            failed[0]=True
            return fallback('native_precommit_declined',args,kwargs)
        result=artifact.run(inputs,count,original_loop=original_once)
        if not failed[0]:stats['native_calls']+=1
        return result
    run.scar_stats=stats;run.scar_artifact=artifact;run.scar_loop_proof=proof
    return run

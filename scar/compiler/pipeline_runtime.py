"""Guard the original Python boundary of a native, pure array pipeline."""
from contextlib import contextmanager
from functools import wraps
from hashlib import sha256
import inspect
import math
import os
from pathlib import Path
import sys
import threading
from types import FunctionType, ModuleType

from scar.codegen import GuardFailure
from scar.trace.code_snapshot import code_key
from .pipeline_model import validate_pipeline
from .runtime import SourceMismatch


def _state(function):
    return (function.__code__, function.__defaults__,
            tuple((key, id(value)) for key, value in (function.__kwdefaults__ or {}).items()))


def _same_state(function, state):
    return (function.__code__ is state[0] and function.__defaults__ is state[1]
            and tuple((key, id(value)) for key, value in (function.__kwdefaults__ or {}).items()) == state[2])


def guarded_pipeline_function(original, artifact, *, exclusive_inputs=False, selection=None, trial=False):
    pipeline = validate_pipeline(artifact.pipeline)
    if type(original) is not FunctionType or original.__closure__:
        raise SourceMismatch('pipeline boundary requires an ordinary function without closure state')
    if code_key(original.__code__) != pipeline.source.code_key:
        raise SourceMismatch('pipeline entry code differs from original graph')
    expected_sources = {pipeline.source.path: pipeline.source.sha256}
    guards = []
    names = {}
    for call in pipeline.calls:
        guard = call.callee
        name = guard.lookup_path[0]
        function = original.__globals__.get(name)
        if (type(function) is not FunctionType or function.__closure__
                or code_key(function.__code__) != guard.source.code_key):
            raise SourceMismatch('actual global callee differs from admitted graph: ' + name)
        if name in names and names[name] is not function:
            raise SourceMismatch('one global lookup selects different pipeline callees')
        names[name] = function
        if guard.source.path in expected_sources and expected_sources[guard.source.path] != guard.source.sha256:
            raise SourceMismatch('pipeline source versions disagree')
        expected_sources[guard.source.path] = guard.source.sha256
    for path, digest in expected_sources.items():
        if sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise SourceMismatch('pipeline source file differs from graph')
    guards = [(name, function, _state(function)) for name, function in names.items()]
    state = _state(original)
    signature = inspect.signature(original)
    identity = artifact.report
    modeled_shapes = artifact.memory_plan['input_shapes']
    cpus = {cpu for operator in identity['operators'] for cpu in (operator.get('cpu_ids') or ())}
    stats = dict(native_calls=0, fallback_calls=0, reasons={})

    def fallback(reason, args, kwargs):
        stats['fallback_calls'] += 1
        stats['reasons'][reason] = stats['reasons'].get(reason, 0) + 1
        return original(*args, **kwargs)

    @wraps(original)
    def run(*args, **kwargs):
        if (not exclusive_inputs or threading.active_count() != 1
                or sys.getprofile() is not None or sys.gettrace() is not None):
            return fallback('synchronous_exclusive_input_contract', args, kwargs)
        if cpus and (not hasattr(os, 'sched_getaffinity') or not cpus <= os.sched_getaffinity(0)):
            return fallback('planned_cpu_allocation_unavailable', args, kwargs)
        if not _same_state(original, state):
            return fallback('pipeline_function_state_changed', args, kwargs)
        for name, function, expected in guards:
            if original.__globals__.get(name) is not function or not _same_state(function, expected):
                return fallback('actual_callee_changed', args, kwargs)
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError:
            return fallback('call_signature', args, kwargs)
        bound.apply_defaults()
        values = [bound.arguments[name] for name in pipeline.parameters]
        if any('EXACT_NUMPY_NDARRAY_COPY_METHOD_DESCRIPTOR' in call.plan.region.assumptions
               for call in pipeline.calls):
            from scar.runtime.array_copy_guard import guard_copy_inputs,CopyGuardFailure
            try:guard_copy_inputs(values)
            except CopyGuardFailure as error:return fallback(str(error),args,kwargs)
        if not trial:
            if (not selection or selection.get('decision') != 'TRANSFORM_UNDER_CONTRACT'
                    or any(selection.get('artifact', {}).get(key) != identity[key]
                           for key in ('build_id', 'pipeline_sha256', 'memory_plan_sha256'))):
                return fallback('no_measured_pipeline_selection', args, kwargs)
        import numpy as np
        if any(type(value) is not np.ndarray or value.dtype != np.float64
               or value.shape != tuple(modeled_shapes[name]) or not value.flags.c_contiguous
               for name, value in zip(pipeline.parameters, values)):
            return fallback('outside_modeled_shape_dtype_or_layout', args, kwargs)
        try:
            output = artifact.run(*values)
        except GuardFailure as error:
            return fallback(str(error), args, kwargs)
        stats['native_calls'] += 1
        return output

    run.scar_stats = stats
    run.scar_artifact = artifact
    return run


@contextmanager
def install_pipeline(module, artifact, *, exclusive_inputs=False, selection=None, trial=False):
    """Replace one declared module function, restoring the exact old binding."""
    if type(module) is not ModuleType or '.' in artifact.pipeline.source.qualname:
        raise SourceMismatch('pipeline installation currently supports direct module functions')
    name = artifact.pipeline.source.qualname
    original = vars(module)[name]
    wrapped = guarded_pipeline_function(original, artifact, exclusive_inputs=exclusive_inputs,
                                         selection=selection, trial=trial)
    setattr(module, name, wrapped)
    try:
        yield wrapped
    finally:
        setattr(module, name, original)


def select_pipeline_measured(artifact, baseline_seconds, native_seconds, *, validated,
                             native_fallbacks=0, minimum_gain=0.05):
    """Apply the measured NO_OP policy to this exact pipeline contract."""
    from scar.optimization.selection import _samples, _summary
    validate_pipeline(artifact.pipeline)
    reasons = []
    baseline, error = _samples(baseline_seconds)
    if error: reasons.append('baseline_' + error)
    native, error = _samples(native_seconds)
    if error: reasons.append('native_' + error)
    if validated is not True: reasons.append('correctness_not_validated')
    if type(native_fallbacks) is not int or native_fallbacks != 0:
        reasons.append('native_fallbacks_or_invalid_count')
    if any(call.plan.proof.get('legal') is not True or call.plan.proof.get('lowering_eligible') is not True
           for call in artifact.pipeline.calls):
        reasons.append('unsupported_pipeline_leaf_contract')
    threshold_ok = (type(minimum_gain) in (int, float) and math.isfinite(minimum_gain)
                    and 0 <= minimum_gain < 1)
    if not threshold_ok: reasons.append('invalid_minimum_gain')
    a, b = _summary(baseline), _summary(native)
    gain = 1 - b['median'] / a['median'] if baseline and native else None
    speedup = a['median'] / b['median'] if baseline and native else None
    if gain is not None and (not math.isfinite(gain) or not math.isfinite(speedup)):
        reasons.append('nonfinite_derived_statistics'); gain = speedup = None
    if threshold_ok and gain is not None and (gain <= 0 or
            gain < minimum_gain and not math.isclose(gain, minimum_gain, rel_tol=1e-12, abs_tol=0)):
        reasons.append('measured_gain_below_required_threshold')
    return dict(status='MEASURED', decision='NO_OP' if reasons else 'TRANSFORM_UNDER_CONTRACT',
        selected=not reasons, reasons=reasons or ['validated_native_median_meets_gain_threshold'],
        baseline=a, native=b, speedup=speedup, gain_fraction=gain, minimum_gain=minimum_gain,
        external_assumptions=['SYNCHRONOUS_CPU_EXECUTION', 'NO_FOREIGN_CONCURRENT_INPUT_WRITERS'],
        scope='EXACT_VALIDATED_PIPELINE_SOURCE_CALLEES_SHAPES_LAYOUT_AND_MEMORY_PLAN')

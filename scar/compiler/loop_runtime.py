"""Original nested callable guards around a compiled, shared array loop."""
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
from .loop_model import validate_loop
from .pipeline_runtime import _state, _same_state
from .runtime import SourceMismatch

_RANGE = range


def _range(function):
    namespace = function.__globals__
    if 'range' in namespace:
        return namespace['range']
    standard = function.__builtins__
    return standard.get('range') if type(standard) is dict else None


def guarded_loop_function(original, artifact, *, exclusive_inputs=False, selection=None, trial=False):
    loop = validate_loop(artifact.loop)
    if (type(original) is not FunctionType or original.__closure__
            or code_key(original.__code__) != loop.source.code_key):
        raise SourceMismatch('loop requires its original ordinary Python function')
    if _range(original) is not _RANGE:
        raise SourceMismatch('loop range lookup is not builtin range')
    source_versions = {loop.source.path: loop.source.sha256}
    guards = []

    def require(namespace, guard):
        name = guard.lookup_path[0]
        function = namespace.get(name)
        if (type(function) is not FunctionType or function.__closure__
                or code_key(function.__code__) != guard.source.code_key):
            raise SourceMismatch('actual nested loop callable differs from source graph: ' + name)
        if source_versions.setdefault(guard.source.path, guard.source.sha256) != guard.source.sha256:
            raise SourceMismatch('nested loop source versions disagree')
        guards.append((namespace, name, function, _state(function)))
        return function

    body = require(original.__globals__, loop.body_callee)
    seen = set()
    for call in loop.body.calls:
        key = call.callee.lookup_path, call.callee.source.code_key
        if key not in seen:
            require(body.__globals__, call.callee)
            seen.add(key)
    for path, digest in source_versions.items():
        if sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise SourceMismatch('loop source file differs from admitted graph')
    state = _state(original)
    # Function metadata may override inspect.signature without changing Python
    # argument binding. Derive the boundary from the guarded code/defaults.
    signature_source = FunctionType(original.__code__, original.__globals__,
                                    argdefs=original.__defaults__)
    signature_source.__kwdefaults__ = original.__kwdefaults__
    signature = inspect.signature(signature_source, follow_wrapped=False)
    report = artifact.report
    strategy = artifact.strategy
    memory = strategy['selected_memory_plan']
    shape = tuple(memory['input_shapes'][loop.state_parameter])
    steps = strategy['trip_count']
    placement = getattr(artifact, 'execution_plan', None)
    cpus = set(placement.resource_binding.cpu_ids) if placement and placement.resource_binding else set()
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
        if not _same_state(original, state) or _range(original) is not _RANGE:
            return fallback('loop_code_or_range_changed', args, kwargs)
        if any(namespace.get(name) is not function or not _same_state(function, expected)
               for namespace, name, function, expected in guards):
            return fallback('actual_nested_callee_changed', args, kwargs)
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError:
            return fallback('call_signature', args, kwargs)
        bound.apply_defaults()
        count = bound.arguments[loop.trip_count_parameter]
        if type(count) is not int or count != steps or count <= 0:
            return fallback('trip_count_outside_exact_specialization', args, kwargs)
        if not trial:
            if (not selection or selection.get('decision') != 'TRANSFORM_UNDER_CONTRACT'
                    or any(selection.get('artifact', {}).get(key) != report[key]
                           for key in ('build_id', 'loop_sha256', 'strategy_sha256'))):
                return fallback('no_measured_loop_selection', args, kwargs)
        import numpy as np
        value = bound.arguments[loop.state_parameter]
        if (type(value) is not np.ndarray or value.dtype != np.float64 or value.shape != shape
                or not value.flags.c_contiguous or not value.flags.aligned):
            return fallback('outside_modeled_shape_dtype_or_layout', args, kwargs)
        try:
            result = artifact.run(value, count)
        except GuardFailure as error:
            return fallback(str(error), args, kwargs)
        stats['native_calls'] += 1
        return result

    run.scar_stats = stats
    run.scar_artifact = artifact
    return run


@contextmanager
def install_loop(module, artifact, *, exclusive_inputs=False, selection=None, trial=False):
    if type(module) is not ModuleType or '.' in artifact.loop.source.qualname:
        raise SourceMismatch('loop installation supports direct module functions')
    name = artifact.loop.source.qualname
    original = vars(module)[name]
    wrapped = guarded_loop_function(original, artifact, exclusive_inputs=exclusive_inputs,
                                    selection=selection, trial=trial)
    setattr(module, name, wrapped)
    try:
        yield wrapped
    finally:
        setattr(module, name, original)


def select_loop_measured(artifact, baseline_seconds, native_seconds, *, validated,
                         native_fallbacks=0, minimum_gain=0.05):
    from copy import deepcopy
    from scar.optimization.selection import _samples, _summary
    validate_loop(artifact.loop)
    reasons = []
    baseline, error = _samples(baseline_seconds)
    if error: reasons.append('baseline_' + error)
    native, error = _samples(native_seconds)
    if error: reasons.append('native_' + error)
    if validated is not True: reasons.append('correctness_not_validated')
    if type(native_fallbacks) is not int or native_fallbacks != 0:
        reasons.append('native_fallbacks_or_invalid_count')
    valid_threshold = (type(minimum_gain) in (int, float) and math.isfinite(minimum_gain)
                       and 0 <= minimum_gain < 1)
    if not valid_threshold: reasons.append('invalid_minimum_gain')
    a, b = _summary(baseline), _summary(native)
    gain = 1 - b['median'] / a['median'] if baseline and native else None
    speedup = a['median'] / b['median'] if baseline and native else None
    if gain is not None and (not math.isfinite(gain) or not math.isfinite(speedup)):
        reasons.append('nonfinite_statistics'); gain = speedup = None
    if valid_threshold and gain is not None and gain < minimum_gain:
        reasons.append('measured_gain_below_required_threshold')
    return dict(schema='scar.loop-measured-selection', schema_version=1,
        status='MEASURED', decision='NO_OP' if reasons else 'TRANSFORM_UNDER_CONTRACT',
        selected=not reasons, reasons=reasons, baseline=a, native=b, speedup=speedup,
        gain_fraction=gain, minimum_gain=minimum_gain,
        artifact={key:artifact.report[key] for key in ('build_id','loop_sha256','strategy_sha256')},
        scope='EXACT_LOOP_SOURCE_NESTED_CALLEES_ARRAY_LAYOUT_SHAPE_AND_TRIP_COUNT',
        external_assumptions=deepcopy(artifact.strategy['external_assumptions']))

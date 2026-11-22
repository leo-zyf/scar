"""Pipeline runtime."""

from contextlib import contextmanager
from functools import wraps
from hashlib import sha256
import inspect
import math
import os
import sys
import threading
import json
from pathlib import Path
from types import FunctionType, ModuleType
from scar.codegen import GuardFailure
from scar.optimization.paired_entry_selection import DEFAULT_MINIMUM_GAIN
from scar.trace.code_snapshot import code_key
from scar.compiler.pipeline_frontend import validate_pipeline
from scar.compiler.runtime import SourceMismatch


# Guard the original Python boundary of a native, pure array pipeline.
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
            from scar.runtime.array_copy_guard import guard_copy_inputs, CopyGuardFailure
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
                             native_fallbacks=0, minimum_gain=DEFAULT_MINIMUM_GAIN, strict_minimum_gain=True):
    """Apply the measured NO_OP policy to this exact pipeline contract."""
    from scar.optimization.selection import _samples, _strict_measured_gain_met, _summary
    if type(strict_minimum_gain) is not bool:
        raise ValueError('minimum gain comparator must be a boolean')
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
    if threshold_ok and gain is not None:
        profitable = (_strict_measured_gain_met(baseline, native, minimum_gain) if strict_minimum_gain else
                      gain > 0 and (gain >= minimum_gain or math.isclose(gain, minimum_gain,
                                                                        rel_tol=1e-12, abs_tol=0)))
        if not profitable:
            reasons.append('measured_gain_below_required_threshold')
    result = dict(status='MEASURED', decision='NO_OP' if reasons else 'TRANSFORM_UNDER_CONTRACT',
        selected=not reasons, reasons=reasons or ['validated_native_median_meets_gain_threshold'],
        baseline=a, native=b, speedup=speedup, gain_fraction=gain, minimum_gain=minimum_gain,
        external_assumptions=['SYNCHRONOUS_CPU_EXECUTION', 'NO_FOREIGN_CONCURRENT_INPUT_WRITERS'],
        scope='EXACT_VALIDATED_PIPELINE_SOURCE_CALLEES_SHAPES_LAYOUT_AND_MEMORY_PLAN')
    if strict_minimum_gain:
        result['strict_minimum_gain'] = True
    return result


# Load and bind the existing full-fusion delivery under the original caller guards.
def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _admit_artifact(artifact, blueprint):
    from scar.codegen import cpp, fused_pipeline
    from scar.compiler.model import document as plan_document
    from scar.compiler.pipeline_frontend import document, pipeline_digest
    _need(type(artifact) is fused_pipeline.FusedPipelineArtifact,
        'actual existing fused pipeline artifact required')
    _need(_copy(document(artifact.pipeline)) == blueprint['pipeline']
        and _copy(artifact._strategy) == blueprint['strategy']
        and artifact.memory_plan == blueprint['selected_memory_plan'],
        'native artifact source, fusion strategy or memory specialization differs')
    operator = artifact._operator
    backend = blueprint['backend']
    report = operator.report
    _need(_copy(plan_document(operator.plan)) == blueprint['optimized_plan']
        and operator.plan.resource_binding is None and operator.plan.cpu_threads == 1,
        'actual native body differs from the unbound fused blueprint')
    _need(report['runtime_implementation_sha256'] == backend['operator_runtime_implementation_sha256']
        and report['compiler'] == backend['compiler']['command']
        and report['compiler_options'] == backend['compiler']['options']
        and report['sha256']['source'] == backend['generated_source_sha256']
        and report['sha256']['abi_header'] == backend['abi_header_sha256']
        and sha256(operator.source_path.read_bytes()).hexdigest() == backend['generated_source_sha256']
        and sha256(operator.abi_header_path.read_bytes()).hexdigest() == backend['abi_header_sha256'],
        'actual native compiler, guards, C++ body or ABI header differs')
    binding = fused_pipeline._operator_receipt(operator)
    expected = fused_pipeline._manifest(artifact.pipeline, blueprint['strategy'],
        artifact.memory_plan, operator, binding, backend['implementation_sha256'], artifact.directory)
    identity = artifact.report
    # report adds convenience paths to the on-disk manifest.
    for name in ('artifact_directory', 'metadata_path', 'source_path', 'library_path'):
        identity.pop(name, None)
    _need(identity == expected and identity['pipeline_sha256'] == pipeline_digest(artifact.pipeline)
        and identity['strategy_sha256'] == blueprint['strategy_sha256']
        and identity['memory_plan_sha256'] == blueprint['memory_plan_sha256'],
        'actual fused artifact identity differs from the selected blueprint')
    _need(report['plan_sha256'] == cpp._plan_digest(operator.plan), 'native math plan identity differs')
    return artifact


def build_fusion_blueprint(pipeline, blueprint, out):
    """Compile the replayed existing operator and fused adapter; never execute it."""
    from scar.codegen import cpp, fused_pipeline
    from scar.compiler.model import plan_from_dict
    from scar.optimization.macro_pass import replay_fusion_blueprint
    value = replay_fusion_blueprint(pipeline, blueprint)
    directory = Path(out).resolve()
    _need(not directory.exists(), 'fresh fusion delivery output directory required')
    directory.mkdir(parents=True)
    operator = cpp.build(plan_from_dict(value['optimized_plan']), directory/'operator')
    artifact = fused_pipeline.build(pipeline, directory/'artifact',
        strategy=value['strategy'], operator=operator)
    _admit_artifact(artifact, value)
    (directory/'blueprint.json').write_text(json.dumps(value, sort_keys=True,
        indent=2, allow_nan=False)+'\n')
    return artifact


def select_fusion_callable(original, pass_report, *, artifact=None, trial=False,
        exclusive_inputs=False, region_document=None, blueprint=None):
    """Keep Python unchanged, or explicitly trial the selected guarded artifact."""
    if type(pass_report) is dict and pass_report.get('decision') == 'KEEP':
        return original
    _need(type(pass_report) is dict and pass_report.get('decision') == 'REWRITE',
        'replayed fusion pass decision required')
    _need(trial is True, 'native fusion delivery requires an explicit trial')
    _need(region_document is not None and blueprint is not None and artifact is not None,
        'original region, fusion blueprint and actual artifact required for trial delivery')
    from scar.optimization.macro_pass import replay_fusion_pass
    admitted, _ = replay_fusion_pass(pass_report, region_document, blueprint)
    return _bind_fusion_entry(original, artifact, admitted.blueprint,
        exclusive_inputs=exclusive_inputs)


def _guarded_fusion_entry(original, artifact, blueprint, *, exclusive_inputs):
    """Shared entry for an admitted source producer or explicit calibration.

    This private factory grants no cost or source permission. Production uses
    select_fusion_callable above; calibration supplies generated canonical
    arithmetic programs and never installs them into the target workload.
    Keeping the factory shared makes the measured live boundary identical.
    """
    from scar.optimization.macro_pass import replay_fusion_blueprint
    verified = replay_fusion_blueprint(artifact.pipeline, blueprint)
    return _bind_fusion_entry(original, artifact, verified, exclusive_inputs=exclusive_inputs)


def _bind_fusion_entry(original, artifact, blueprint, *, exclusive_inputs):
    _admit_artifact(artifact, blueprint)
    guarded = guarded_pipeline_function(original, artifact,
        exclusive_inputs=exclusive_inputs, trial=True)
    allowed = frozenset(blueprint['execution']['allowed_cpu_ids'])

    @wraps(original)
    def run(*args, **kwargs):
        try:
            applicable = hasattr(os, 'sched_getaffinity') and os.sched_getaffinity(0) == allowed
        except OSError:
            applicable = False
        if not applicable:
            stats = guarded.scar_stats
            stats['fallback_calls'] += 1
            reason = 'caller_cpu_domain_differs_from_cost_context'
            stats['reasons'][reason] = stats['reasons'].get(reason, 0)+1
            return original(*args, **kwargs)
        return guarded(*args, **kwargs)

    run.scar_stats = guarded.scar_stats
    run.scar_artifact = artifact
    return run


__all__ = [
    'build_fusion_blueprint',
    'select_fusion_callable',
    'guarded_pipeline_function',
    'install_pipeline',
    'select_pipeline_measured',
]

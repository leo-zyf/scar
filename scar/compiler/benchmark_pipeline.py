"""Clean paired validation of a native pipeline and its original Python entry."""
import argparse
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np

from .benchmark import load_source, resolve, distribution, bitwise_equal
from .pipeline_runtime import guarded_pipeline_function, select_pipeline_measured


def benchmark_pipeline(artifact_path, out, *, input_metadata_path=None, warmup=3, repetitions=9):
    from scar.codegen.pipeline_loader import load
    if sys.getprofile() is not None or sys.gettrace() is not None:
        raise ValueError('clean benchmark requires no Python tracing or profile hooks')
    if warmup < 1 or repetitions < 3:
        raise ValueError('positive warmup and at least three paired repetitions required')
    out = Path(out).absolute()
    if out.exists():
        raise ValueError('fresh pipeline benchmark report required')
    artifact = load(artifact_path)
    pipeline = artifact.pipeline
    shapes = artifact.memory_plan['input_shapes']
    shape_evidence = None
    if input_metadata_path is not None:
        from .input_metadata import shapes_from_metadata
        observed, shape_evidence = shapes_from_metadata(input_metadata_path, pipeline.source, pipeline.parameters)
        if observed != shapes:
            raise ValueError('benchmark input observations differ from native memory plan')
    module = load_source(pipeline.source.path)
    original = resolve(module, pipeline.source.qualname)
    wrapped = guarded_pipeline_function(original, artifact, exclusive_inputs=True, trial=True)
    signature = inspect.signature(original)

    def invoke(function, arrays):
        positional, keywords = [], {}
        for name, value in zip(pipeline.parameters, arrays, strict=True):
            if signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY:
                keywords[name] = value
            else:
                positional.append(value)
        return function(*positional, **keywords)

    rng = np.random.default_rng(42)
    arrays = [rng.normal(0, 0.1, size=shapes[name]) for name in pipeline.parameters]
    snapshots = [value.copy() for value in arrays]
    expected, actual = invoke(original, arrays), invoke(wrapped, arrays)
    if not bitwise_equal(expected, actual):
        raise AssertionError('pipeline result differs bitwise')
    if any(not bitwise_equal(a, b) for a, b in zip(arrays, snapshots)):
        raise AssertionError('pipeline mutated input')
    if any(np.shares_memory(actual, value) for value in arrays):
        raise AssertionError('pipeline output aliases input')
    actual[:] = 123
    if not bitwise_equal(invoke(original, arrays), invoke(wrapped, arrays)):
        raise AssertionError('pipeline exposes reusable workspace to the caller')
    arrays[0][:] += 0.001
    if not bitwise_equal(invoke(original, arrays), invoke(wrapped, arrays)):
        raise AssertionError('pipeline reused stale input contents')
    before = wrapped.scar_stats['fallback_calls']
    low = [value.astype(np.float32) for value in arrays]
    np.testing.assert_array_equal(invoke(original, low), invoke(wrapped, low))
    if wrapped.scar_stats['fallback_calls'] != before + 1:
        raise AssertionError('dtype mismatch must fall back once')
    bad = [value.copy() for value in arrays]
    bad[0][:] = np.nan
    before = wrapped.scar_stats['fallback_calls']
    with warnings.catch_warnings(record=True):
        np.testing.assert_array_equal(invoke(original, bad), invoke(wrapped, bad))
    if wrapped.scar_stats['fallback_calls'] != before + 1:
        raise AssertionError('native partial-result guard must fall back once')
    # This is a generic binding guard test, selected from the graph's first
    # callee reference rather than from a workload/function name.
    name = pipeline.calls[0].callee.lookup_path[0]
    saved = original.__globals__[name]
    counter = [0]

    def changed(*args, **kwargs):
        counter[0] += 1
        return saved(*args, **kwargs) + 0.001

    original.__globals__[name] = changed
    try:
        expected = invoke(original, arrays)
        expected_callee_calls = counter[0]
        counter[0] = 0
        before = wrapped.scar_stats['fallback_calls']
        actual = invoke(wrapped, arrays)
        if (not bitwise_equal(expected, actual) or counter[0] != expected_callee_calls
                or wrapped.scar_stats['fallback_calls'] != before + 1):
            raise AssertionError('changed callee must execute current original pipeline once')
    finally:
        original.__globals__[name] = saved
    native_before = wrapped.scar_stats['native_calls']
    fallback_before = wrapped.scar_stats['fallback_calls']
    for _ in range(warmup):
        invoke(original, arrays)
        invoke(wrapped, arrays)
    times = [[], []]
    for index in range(repetitions):
        for which in ((0, 1) if index % 2 == 0 else (1, 0)):
            start = time.perf_counter_ns()
            result = invoke(original if which == 0 else wrapped, arrays)
            times[which].append((time.perf_counter_ns() - start) / 1e9)
            del result
    fallbacks = wrapped.scar_stats['fallback_calls'] - fallback_before
    native = wrapped.scar_stats['native_calls'] - native_before
    if fallbacks or native != warmup + repetitions:
        raise AssertionError('clean pipeline benchmark did not execute every native call')
    selection = select_pipeline_measured(artifact, times[0], times[1], validated=True, native_fallbacks=fallbacks)
    report = dict(schema='scar.pipeline-benchmark', schema_version=1,
        scope='GUARDED_SYNCHRONOUS_CPU_PIPELINE_ONLY', artifact=artifact.report,
        input_shapes=shapes, input_shape_evidence=shape_evidence,
        validation=dict(bitwise_output=True, input_unchanged=True, fresh_final=True,
            mutable_input_invalidation=True, caller_output_mutation_safe=True,
            dtype_fallback_once=True, partial_native_failure_fallback_once=True, changed_callee_fallback_once=True),
        instrumented=False, warmup=warmup, repetitions=repetitions, paired_order='AB_BA_ALTERNATING',
        baseline_seconds=distribution(times[0]), optimized_seconds=distribution(times[1]),
        speedup=selection['speedup'], selection=selection, decision=selection['decision'],
        stats=wrapped.scar_stats, native_timed_calls=repetitions, timed_fallbacks=0,
        whole_program_validated=False, whole_program_compiled=False,
        environment=dict(python=sys.version, numpy=np.__version__, platform=platform.platform(),
            cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            torch_cuda_used=False))
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact')
    parser.add_argument('--out', required=True)
    parser.add_argument('--inputs')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repetitions', type=int, default=9)
    args = parser.parse_args()
    report = benchmark_pipeline(args.artifact, args.out, input_metadata_path=args.inputs,
                                warmup=args.warmup, repetitions=args.repetitions)
    print(json.dumps({key: report[key] for key in ('decision', 'speedup', 'baseline_seconds', 'optimized_seconds')}))


if __name__ == '__main__':
    main()

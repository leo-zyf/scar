"""Probe three graph-derived CPU paths at one warm caller boundary."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns

from scar.ir.execution_recipe import artifact_reference, read_artifact


OBSERVATIONS = 'scar.pipeline-path-observations'


def _pointwise_edges(pipeline, blueprint):
    strategy = blueprint['strategy']
    certificate = strategy['fusion_certificate']
    shapes = {row['value_id']: row['original_shape'] for row in strategy['fusion_shape_replay']}
    output_shape = blueprint['selected_memory_plan']['output_shape']
    indices = defaultdict(list)
    for call in pipeline.calls:
        for value in call.inputs:
            indices[value].append(call.id)
    if any(shape != output_shape for shape in shapes.values()):
        raise ValueError('this calibration requires one common pointwise iteration domain')
    edges = []
    for call, mapping in zip(pipeline.calls, certificate['calls'], strict=True):
        if call.output == pipeline.output:
            continue
        users = indices[call.output]
        if not users:
            raise ValueError('unconsumed intermediate lacks a qualified materialization rewrite')
        edges.append(dict(value=call.output, producer=call.id, consumers=list(dict.fromkeys(users)),
            last_source_use=users[-1], index_map='SAME_ELEMENT_IN_COMMON_DOMAIN',
            merged_expression=mapping['output_expr'], shape=shapes[call.output],
            publication='INTERNAL_ONLY_UNDER_SOURCE_CLOSURE_AND_LIVE_GUARDS'))
    return edges


def _direct(pipeline, memory, output):
    from scar.codegen.cpp import build as build_leaf, _plan_digest
    from scar.codegen.pipeline import build as build_pipeline
    artifacts = {}
    operators = {}
    for call in pipeline.calls:
        key = _plan_digest(call.plan)
        if key not in artifacts:
            if call.plan.cpu_threads != 1 or call.plan.resource_binding is not None:
                raise ValueError('direct path requires original unbound single-thread leaf plans')
            artifacts[key] = build_leaf(call.plan, output/'leaves'/str(len(artifacts)))
        operators[call.id] = artifacts[key]
    return build_pipeline(pipeline, output/'host', memory_plan=memory, operators=operators)


@contextmanager
def _clock(instance, name):
    saved = getattr(instance, name)
    samples = []

    def recorded(*args, **kwargs):
        start = perf_counter_ns()
        result = saved(*args, **kwargs)
        samples.append(perf_counter_ns() - start)
        return result

    setattr(instance, name, recorded)
    try:
        yield samples
    finally:
        setattr(instance, name, saved)


def _breakdown(guarded, artifact, ffi_owner, inputs):
    with _clock(artifact, 'run') as adapter, _clock(ffi_owner, '_run') as ffi:
        start = perf_counter_ns()
        guarded(*inputs)
        total = perf_counter_ns() - start
    if len(adapter) != 1 or len(ffi) != 1:
        raise ValueError('diagnostic sample did not reach the exact native path once')
    # Nested clocks cover the same call. The residual includes instrumentation
    # and represents the rest of the enclosing boundary, not a pure syscall cost.
    return dict(protected_elapsed_ns=total, adapter_elapsed_ns=adapter[0], ffi_elapsed_ns=ffi[0],
        outside_adapter_ns=total-adapter[0], adapter_outside_ffi_ns=adapter[0]-ffi[0])


def _decide(samples, minimum_gain):
    cost = {name: dict(min_ns=min(rows), median_ns=median(rows), max_ns=max(rows))
            for name, rows in samples.items()}
    best = min(cost, key=lambda name: cost[name]['median_ns'])
    upper = cost[best]['max_ns']
    if best == 'original':
        return cost, 'NO_OP', 'ORIGINAL_HAS_LOWEST_OBSERVED_MEDIAN'
    if (upper >= min(row['min_ns'] for name, row in cost.items() if name != best)
            or upper > (1-minimum_gain)*cost['original']['min_ns']):
        return cost, 'NO_OP', 'SAMPLED_RANGES_OR_GAIN_THRESHOLD_UNRESOLVED'
    return cost, best, 'MATCHED_WARM_SCOPE_PROBE_PASSES_RANGE_AND_GAIN_GATE'


def _cost_records(cost):
    return {role: dict(value=cost[name]['median_ns']/1e9, unit='seconds', basis='OBSERVED',
                       reuse_count=1, measurement_scope='DIRECT_COMPLETE_SCOPE_TIMING')
            for role, name in (('original_total', 'original'), ('native_total', 'fused'))}


def check_observations(path):
    """Reopen the original graph, recompute demand and replay the sampled gate."""
    from scar.compiler.macro_pipeline import check

    observations = read_artifact(artifact_reference(path, OBSERVATIONS))
    saved, pipeline, _, blueprint, report = check(observations['source_preparation'])
    demand = read_artifact(saved['files']['candidate_demand'])
    samples = observations['clean_samples_ns']
    repetitions = observations['repetitions']
    if (set(samples) != {'original', 'direct', 'fused'} or type(repetitions) is not int or repetitions < 3
            or any(len(rows) != repetitions or any(type(ns) is not int or ns <= 0 for ns in rows)
                   for rows in samples.values())):
        raise ValueError('matched positive samples required for each scoped implementation')
    cost, selected, reason = _decide(samples, saved['minimum_gain'])
    expected = dict(context=report['cost_context'], materialization_edges=_pointwise_edges(pipeline, blueprint),
        source_demand=demand['source_baseline']['features'], fused_demand=demand['implementation_lowering_demand']['features'],
        sampled_ranges=cost, selected=selected, selection_reason=reason, cost_records=_cost_records(cost))
    if any(observations[key] != value for key, value in expected.items()):
        raise ValueError('observations differ from source, successor demand or sampled cost gate replay')
    return dict(source_and_cost_replayed=True, selected=selected, sampled_ranges=cost)


def calibrate(preparation, out, *, warmup=2, repetitions=7):
    from scar.compiler.calibrate_pipeline_corpus import _module, _write
    from scar.compiler.macro_pipeline import check
    from scar.compiler.pipeline_fusion_delivery import build_fusion_blueprint, _guarded_fusion_entry
    from scar.compiler.pipeline_runtime import guarded_pipeline_function
    import numpy as np

    if warmup < 1 or repetitions < 3 or sys.gettrace() is not None or sys.getprofile() is not None:
        raise ValueError('untraced warm scope with at least one warmup and three samples required')
    saved, pipeline, _, blueprint, report = check(preparation)
    if not report['legality']['qualified'] or saved['files']['candidate_demand'] is None:
        raise ValueError('source-qualified pointwise successor required before cost probes')
    if sorted(os.sched_getaffinity(0)) != saved['available_cpu_ids']:
        raise ValueError('live CPU domain differs from the structural preparation')
    edges = _pointwise_edges(pipeline, blueprint)
    demand = read_artifact(saved['files']['candidate_demand'])
    output = Path(out).resolve()
    output.mkdir(parents=True)
    fused = build_fusion_blueprint(pipeline, blueprint, output/'fused')
    direct = _direct(pipeline, blueprint['original_memory_plan'], output/'direct')
    original = getattr(_module(Path(pipeline.source.path), 'scar_pipeline_path_probe'), pipeline.source.qualname)
    functions = dict(original=original,
        direct=guarded_pipeline_function(original, direct, exclusive_inputs=True, trial=True),
        fused=_guarded_fusion_entry(original, fused, blueprint, exclusive_inputs=True))
    rng = np.random.default_rng(83)
    inputs = tuple(rng.uniform(.5, 1., size=blueprint['input_shapes'][name]) for name in pipeline.parameters)
    for array in inputs:
        array.flags.writeable = False
    expected = original(*inputs).tobytes()
    samples = {name: [] for name in functions}
    for index in range(warmup+repetitions):
        ordered = list(functions)
        offset = index % len(ordered)
        for name in ordered[offset:]+ordered[:offset]:
            start = perf_counter_ns()
            result = functions[name](*inputs)
            elapsed = perf_counter_ns() - start
            if result.tobytes() != expected:
                raise ValueError('probe output differs from original operation order')
            del result
            if index >= warmup:
                samples[name].append(elapsed)
    if any(fn.scar_stats['native_calls'] != warmup+repetitions or fn.scar_stats['fallback_calls']
           for fn in (functions['direct'], functions['fused'])):
        raise ValueError('clean calibration did not execute every native probe')
    diagnostic = {name: [_breakdown(functions[name], artifact, owner, inputs) for _ in range(3)]
        for name, artifact, owner in (('direct', direct, direct), ('fused', fused, fused._operator))}
    cost, selected, reason = _decide(samples, saved['minimum_gain'])
    context = report['cost_context']
    records = _cost_records(cost)
    receipt = _write(output/'observations.json', dict(schema=OBSERVATIONS, schema_version=1,
        context=context, source_preparation=str(Path(preparation).resolve()), materialization_edges=edges,
        source_demand=demand['source_baseline']['features'], fused_demand=demand['implementation_lowering_demand']['features'],
        native_plan_sha256=fused.report['operators'][0]['plan_sha256'], protected_scope=report['cost_context']['cost_scope'],
        clean_samples_ns=samples, sampled_ranges=cost, warmup=warmup, repetitions=repetitions,
        diagnostic_nested_clocks=diagnostic, cost_records=records, selected=selected, selection_reason=reason,
        direct_host_ffi_contains_leaf_compute=True, fused_ffi_includes_native_guards=True,
        sampling_bounds_are_not_guarantees=True, workload_entry_executed=False, deployment_authorized=False))
    profile = None
    if selected == 'fused':
        profile = _write(output/'profile.json', dict(schema='scar.deployment-cost-profile', schema_version=1,
            mode='COMPLETE_SCOPE', context=context, observed_reuse_count=1,
            context_evidence=[dict(artifact=receipt, locator='#/context')],
            costs={name: {**{key: row[key] for key in ('value','unit','basis')},
                'evidence':[dict(artifact=receipt, locator='#/cost_records/'+name)]} for name, row in records.items()}))
    result = dict(schema='scar.pipeline-path-calibration', schema_version=1, observations=receipt,
        empirical_selection=selected, fusion_profile=profile, reason=reason,
        multi_cpu_lowering='UNKNOWN_FOR_REWRITTEN_REGION', default_backend='NO_OP', deployment_authorized=False)
    _write(output/'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--preparation')
    source.add_argument('--check')
    parser.add_argument('--out')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repetitions', type=int, default=7)
    args = parser.parse_args(argv)
    if args.check:
        if args.out:
            parser.error('--check reopens the retained observations without publishing new files')
        result = check_observations(args.check)
    else:
        if not args.out:
            parser.error('--out is required for new calibration')
        result = calibrate(args.preparation, args.out, warmup=args.warmup, repetitions=args.repetitions)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

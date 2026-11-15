"""Replay independent fused-model residuals without fitting or selecting code.

Grouped residuals describe the retained canonical measurements. They do not
establish causal hardware explanations, target latency bounds, or confidence
intervals. Once inspected here, these holdouts are development evidence.
"""
from collections import defaultdict
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics

from scar.codegen.cpu_fused_cost import fused_features
from scar.ir.execution_recipe import artifact_reference, read_artifact


def _identity(row):
    p = row.get('parameters', row)
    return row['primitive'], row['geometry'], p['rows'], p['cols']


def _ref(artifact, index):
    return dict(artifact=artifact, locator=f'#/observations/{index}')


def _summary(rows):
    errors = [r['relative_error'] for r in rows]
    return dict(count=len(rows), maximum_relative_error=max(errors),
        median_relative_error=statistics.median(errors),
        mean_signed_relative_error=statistics.mean(r['signed_relative_error'] for r in rows),
        maximum_absolute_error_ns=max(abs(r['residual_ns']) for r in rows),
        overpredictions=sum(r['residual_ns'] > 0 for r in rows),
        underpredictions=sum(r['residual_ns'] < 0 for r in rows),
        predictions_outside_observed_repeat_range=sum(not r['prediction_inside_repeat_range'] for r in rows),
        outside_repeat_range_is_not_a_statistical_significance_test=True)


def _dimensions(row, capacity):
    p = row['parameters']; operations = p['arithmetic']
    footprint = row['geometry_properties']['backing_bytes'] + p['rows'] * p['cols'] * 8
    return dict(primitive=row['primitive'], geometry=row['geometry'], family=row['family'],
        arithmetic=','.join(f'{op}:{count}' for op, count in sorted(operations.items()) if count),
        dependency_depth=p['dependent_operation_depth'], backing_storage_groups=p['backing_storage_groups'],
        array_loads=p['array_loads'], shape=[p['rows'], p['cols']],
        cache_extent=('WITHIN_REPORTED_LLC' if footprint <= capacity else 'EXCEEDS_REPORTED_LLC'),
        misaligned_load_rows='PRESENT' if row['features']['misaligned_loads'] else 'ABSENT')


def analyze_fused_profile(model, artifact, *, top_n=10):
    """Pure diagnostics of a retained v2 profile; no coefficients are refitted."""
    if (model.get('schema'), model.get('schema_version')) != ('scar.cpu-fused-cost-model', 2):
        raise ValueError('retained independent fused model v2 required')
    if type(top_n) is not int or top_n < 1:
        raise ValueError('positive diagnostic example limit required')
    features = model['features']; capacity = model['cache_geometry']['reported_capacity_bytes']
    observations = model['observations']; lookup = {}
    for index, row in enumerate(observations):
        identity = _identity(row)
        if identity in lookup: raise ValueError('ambiguous canonical observation identity')
        actual, geometry = fused_features(row['parameters'], capacity)
        samples = row['samples_ns']
        if (actual != row['features'] or geometry != row['geometry_properties']
                or set(features) != set(actual) or not samples
                or any(type(n) not in (int, float) or not math.isfinite(n) or n < 0 for n in samples)
                or statistics.median(samples) != row['median_ns']):
            raise ValueError('canonical observation feature or timing replay differs')
        lookup[identity] = index, row
    models = []
    for name, fitted in sorted(model['local_models'].items()):
        coefficients = fitted['coefficients']
        if (set(coefficients) != set(features)
                or any(type(n) not in (int, float) or not math.isfinite(n) or n < 0 for n in coefficients.values())):
            raise ValueError('finite nonnegative original coefficients required')
        residuals = []
        for held in fitted['holdout']:
            index, row = lookup[_identity(held)]
            if not row['holdout']: raise ValueError('validation row is not an independent producer holdout')
            prediction = sum(row['features'][key] * coefficients[key] for key in features)
            observed = row['median_ns']; residual = prediction - observed
            relative = abs(residual) / max(observed, 1)
            if (held['observed_ns'] != observed
                    or not math.isclose(prediction, held['predicted_ns'], rel_tol=1e-12, abs_tol=1e-7)
                    or not math.isclose(relative, held['relative_error'], rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError('retained holdout prediction differs from original coefficient replay')
            samples = row['samples_ns']; lo = min(samples); hi = max(samples)
            residuals.append(dict(observation=_ref(artifact, index), dimensions=_dimensions(row, capacity),
                observed_median_ns=observed, predicted_ns=prediction, residual_ns=residual,
                signed_relative_error=residual / max(observed, 1), relative_error=relative,
                observed_repeat_min_ns=lo, observed_repeat_max_ns=hi,
                observed_repeat_stdev_ns=statistics.stdev(samples) if len(samples) > 1 else None,
                prediction_inside_repeat_range=lo <= prediction <= hi,
                feature_contributions_ns={key: row['features'][key] * coefficients[key] for key in features}))
        if not residuals: raise ValueError('local model has no retained holdout rows')
        limit = fitted['precision_policy']['maximum_holdout_relative_error']
        measured_max = max(r['relative_error'] for r in residuals)
        if (not math.isclose(measured_max, fitted['maximum_holdout_relative_error'], rel_tol=1e-12)
                or fitted['admitted'] != (measured_max <= limit)):
            raise ValueError('original precision admission differs from holdout replay')
        groups = []
        for dimension in residuals[0]['dimensions']:
            buckets = defaultdict(list)
            for row in residuals:
                buckets[json.dumps(row['dimensions'][dimension], sort_keys=True)].append(row)
            for value, rows in buckets.items():
                groups.append(dict(dimension=dimension, value=json.loads(value), **_summary(rows)))
        models.append(dict(model=name, original_admitted=fitted['admitted'], precision_policy=fitted['precision_policy'],
            original_training_residual=fitted['residual'], original_rank=fitted['rank'],
            original_active_parameters=fitted['active_parameters'], summary=_summary(residuals),
            grouped_residuals=sorted(groups, key=lambda r: (-r['maximum_relative_error'], r['dimension'], str(r['value']))),
            largest_residuals=sorted(residuals, key=lambda r: -r['relative_error'])[:top_n],
            zero_coefficients=[key for key in features if coefficients[key] == 0],
            zero_coefficients_are_not_zero_physical_operation_costs=True))

    # Exact-feature collisions are a separate observational check; differences
    # may also come from run noise/order. They are not causal attribution.
    buckets = defaultdict(list)
    for index, row in enumerate(observations):
        buckets[tuple(row['features'][key] for key in features)].append((index, row))
    collisions = []
    for rows in buckets.values():
        if len(rows) < 2: continue
        low = min(rows, key=lambda pair: pair[1]['median_ns'])
        high = max(rows, key=lambda pair: pair[1]['median_ns'])
        collisions.append(dict(observation_count=len(rows),
            minimum=_ref(artifact, low[0]), maximum=_ref(artifact, high[0]),
            minimum_median_ns=low[1]['median_ns'], maximum_median_ns=high[1]['median_ns'],
            relative_median_spread=(high[1]['median_ns'] - low[1]['median_ns']) / max(low[1]['median_ns'], 1),
            repeat_ranges_disjoint=max(low[1]['samples_ns']) < min(high[1]['samples_ns'])))
    return dict(schema='scar.fused-cost-diagnostics', schema_version=1, source_profile=artifact,
        status='DIAGNOSTIC_ONLY', evidence_status='OBSERVED_AND_REPLAYED',
        observation_count=len(observations), local_models=models,
        feature_collisions=dict(bucket_count=len(collisions),
            largest=sorted(collisions, key=lambda row: -row['relative_median_spread'])[:top_n],
            causal_feature_insufficiency_proven=False),
        inferred=[
            dict(claim='FULL_RANK_DOES_NOT_ESTABLISH_PREDICTIVE_ACCURACY',
                basis='Replay retained training residuals and independent holdout errors.'),
            dict(claim='ADDITIVE_COUNTS_AND_MAX_DEPTH_DO_NOT_ENCODE_COMPLETE_DAG_OR_COMPILER_PATH',
                basis='Feature vector contains aggregate op counts/depth/layout; no dependency-width, '
                    'per-function vector path, alias-versioning branch, or spill facts.'),
            dict(claim='OBSERVED_REPEAT_RANGE_IS_NOT_A_TARGET_LATENCY_OR_CONFIDENCE_BOUND',
                basis='Finite canonical repeats do not bound future contention, cache residency or target programs.')],
        proposed_calibration=dict(evidence_status='PROPOSED_NOT_RUN',
            purpose='Separate arithmetic throughput, critical-path latency and memory geometry before refitting.',
            minimum_diagnostic_ablation=dict(
                purpose='Distinguish model-form and compiler-path failures; not a complete new cost calibration.',
                dag_topologies=['SERIAL_CHAIN', 'BALANCED_FANIN', 'TWO_INDEPENDENT_CHAINS'],
                identical_operation_multiset=dict(add=4, sub=2, mul=2), array_loads=4,
                all_expression_nodes_consumed=True, backing_storage_groups=[1, 2],
                column_alignment=['ALIGNED_EVEN_STRIDE', 'SHIFTED_ODD_STRIDE'],
                shapes=[[256, 384], [512, 640]], configuration_count=24,
                repetitions=7, warmup=2, measured_stage_executions=168, total_stage_executions=216,
                native_compilations=1, native_kernel_bodies=3,
                worker_context='EXACT_ORIGINAL_RETAINED_RESOURCE_CONTEXT',
                output_is_selection_authorizing=False,
                fresh_validation='After model/features freeze, reserve untouched DAGs and shapes; '
                    'these 24 diagnostic configurations cannot serve as independent admission holdouts.'),
            fixed_design=[
                'Cross balanced, serial and parallel-chain DAGs with identical operation multisets and load counts.',
                'Cross operation budgets 4/8/12, load counts 2/4/8, backing groups 1/2 and aligned/shifted layouts.',
                'Cross even/odd row lengths and working sets below/near/above reported cache extents.',
                'Retain per-canonical-function vectorization/alias-versioning evidence; do not use target DAGs.',
                'Interleave repeated samples with a fixed seed and record host load; do not clear shared caches.',
                'Freeze features/model/admission before fresh holdouts; all inspected current holdouts become development data.'],
            candidate_model='Compare throughput/critical-path bottleneck composition and explicit compiler-path regimes; '
                'do not assume arithmetic latency and throughput costs add.',
            existing_precision_policy_changed=False, target_variants_required=False),
        coefficients_refitted=False, new_measurements_collected=False,
        target_workload_executed=False, target_variants_ranked=False,
        selectable=False, exportable_as_complete_cost=False,
        empirical_error_is_theoretical_bound=False, deployment_authorized=False)


def diagnose_fused_cost_profile(profile, out, *, top_n=10):
    """Write a fresh evidence report from retained samples, never target runs."""
    profile = Path(profile).absolute(); out = Path(out).absolute()
    reference = artifact_reference(profile, 'scar.cpu-fused-cost-model')
    model = read_artifact(reference)
    result = analyze_fused_profile(model, reference, top_n=top_n)
    if artifact_reference(profile, reference['schema']) != reference:
        raise ValueError('source calibration changed during diagnostic replay')
    if out.exists(): raise ValueError('fresh diagnostics output required')
    implementation = Path(__file__).resolve(); raw = implementation.read_bytes()
    result['diagnostic_implementation'] = dict(path=str(implementation), sha256=sha256(raw).hexdigest(), bytes=len(raw))
    out.mkdir(parents=True)
    (out / 'diagnostics.json').write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, help='retained independent fused model profile JSON')
    parser.add_argument('--out', required=True, help='fresh diagnostics directory')
    parser.add_argument('--top-n', type=int, default=10)
    args = parser.parse_args(argv)
    result = diagnose_fused_cost_profile(args.profile, args.out, top_n=args.top_n)
    print(json.dumps(dict(report=str(Path(args.out).absolute() / 'diagnostics.json'),
        observation_count=result['observation_count'],
        models=[dict(name=row['model'], original_admitted=row['original_admitted'],
            **row['summary']) for row in result['local_models']],
        selectable=False, exportable_as_complete_cost=False), sort_keys=True))
    return 0


if __name__ == '__main__': raise SystemExit(main())

"""Compose a guarded entry cost forecast from original demand and hot-call evidence."""

import argparse
from fractions import Fraction
import json
from math import ceil, floor, prod
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.aten_boundary_trials import ATEN_GAIN_POLICY, _passes_gain
from scar.optimization.paired_entry_selection import _metrics
from scar.runtime.aten_python_trial import _read_manifest


_DEMAND_SCHEMA = 'scar.aten-python-entry-run'
_CALL_SCHEMA = 'scar.aten-python-trial-validation'
_GEOMETRY = 'VALUE_INDEPENDENT_GEOMETRY'
_MAX_NS = (1 << 63) - 1
_DIGEST = re.compile(r'[0-9a-f]{64}')


def _read_pinned(path, digest, schema):
    ref = artifact_reference(Path(path).resolve(strict=True), schema)
    if ref['sha256'] != digest:
        raise ValueError(f'{schema} differs from its declared artifact digest')
    record = read_artifact(ref)
    if type(record.get('schema_version')) is not int or record['schema_version'] != 1:
        raise ValueError(f'{schema} requires schema version 1')
    return record, ref


def _integer(value, field, *, minimum=0):
    if type(value) is not int or not minimum <= value <= _MAX_NS:
        raise ValueError(f'{field} must be a bounded integer of at least {minimum}')
    return value


def _startup_ns(overhead_ms):
    if type(overhead_ms) not in (int, float):
        raise ValueError('a declared nonnegative startup overhead in milliseconds is required')
    try:
        duration = Fraction(str(overhead_ms)) * 1_000_000
    except ValueError as error:
        raise ValueError('declared startup overhead must be finite') from error
    if not 0 <= duration <= _MAX_NS:
        raise ValueError('declared startup overhead must be in the nonnegative nanosecond range')
    return ceil(duration)


def _observed_entry(record, manifest, digest):
    if (record.get('mode') != 'OBSERVE_ORIGINAL' or record.get('trial_sha256') != digest
            or record.get('cpu_ids') != sorted(manifest['cpus']) or record.get('affinity_restored') is not True
            or record.get('original_binding_restored') is not True
            or _integer(record.get('native_calls'), 'observed native calls') != 0
            or _integer(record.get('fallback_calls'), 'observed fallback calls') != 0):
        raise ValueError('demand evidence must observe the unchanged original entry in this trial context')
    entry = record.get('entry_function')
    if type(entry) is not str or not entry.isidentifier() or entry == manifest['source_proof']['function']:
        raise ValueError('demand evidence must identify the enclosing application entry')
    args = record.get('entry_args')
    invocation = record.get('invocation_sha256')
    if not (type(args) is list and all(type(arg) is str for arg in args)):
        if type(invocation) is not str or _DIGEST.fullmatch(invocation) is None:
            raise ValueError('demand evidence must identify the observed entry arguments')
        args = None
    request = dict(entry_function=entry, entry_args=args)
    if type(invocation) is str and _DIGEST.fullmatch(invocation) is not None:
        request['invocation_sha256'] = invocation
    demand = record.get('demand')
    if type(demand) is not dict or demand.get('comparison_scope') != 'OBSERVED_ORIGINAL_TRIAL_CALLABLE':
        raise ValueError('demand evidence must time the original trial callable')
    request['process_id'] = _integer(record.get('process_id'), 'observed process ID', minimum=1)
    entry_ns = _integer(record.get('entry_ns'), 'original entry time', minimum=1)
    setup_ns = _integer(record.get('setup_ns'), 'original setup time')
    teardown_ns = _integer(record.get('teardown_ns'), 'original teardown time')
    calls = _integer(demand.get('calls'), 'original call count', minimum=1)
    total = _integer(demand.get('total_callable_ns'), 'original total callable time', minimum=1)
    minimum = _integer(demand.get('min_callable_ns'), 'original minimum callable time', minimum=1)
    maximum = _integer(demand.get('max_callable_ns'), 'original maximum callable time', minimum=1)
    if (total > entry_ns or minimum > maximum
            or not ((calls - 1) * minimum + maximum <= total <= (calls - 1) * maximum + minimum)):
        raise ValueError('original callable timings do not fit the observed entry and call range')
    grad_modes = demand.get('grad_mode_calls')
    if grad_modes is not None and (type(grad_modes) is not dict or set(grad_modes) != {'enabled', 'disabled'}
            or sum(_integer(count, 'observed grad mode call count') for count in grad_modes.values()) != calls):
        raise ValueError('observed grad mode counts differ from the original call count')
    return request, dict(entry_ns=entry_ns, setup_ns=setup_ns, teardown_ns=teardown_ns,
                         calls=calls, total_callable_ns=total, min_callable_ns=minimum, max_callable_ns=maximum,
                         grad_mode_calls=grad_modes)


def _hot_calls(record, manifest, digest):
    if (record.get('trial_sha256') != digest or record.get('cpu_ids') != manifest['cpus']
            or record.get('scope') != 'WARM_FULL_PYTHON_FUNCTION_WITH_GUARDS'
            or record.get('input_guard_kind') != _GEOMETRY or record.get('changed_value_native_verified') is not True
            or record.get('bitwise_original_results') is not True or record.get('inputs_unchanged') is not True
            or record.get('original_restored') is not True
            or _integer(record.get('reference_input_bytes_loaded'), 'loaded reference input bytes') != 0):
        raise ValueError('call evidence must validate the exact geometry-guarded trial on changed values')
    recipe_inputs = manifest['recipe']['inputs']
    receipts = record.get('validated_inputs')
    if (type(receipts) is not list or len(receipts) != len(recipe_inputs)
            or any(type(row) is not dict or type(row.get('path')) is not str or not Path(row['path']).is_absolute()
                   or type(row.get('sha256')) is not str or _DIGEST.fullmatch(row['sha256']) is None
                   or type(row.get('size_bytes')) is not int or row['size_bytes'] != 4 * prod(spec['shape'])
                   for row, spec in zip(receipts, recipe_inputs, strict=True))):
        raise ValueError('held-out call inputs must match the retained recipe slot geometries')
    if [row['sha256'] for row in receipts] == [row['sha256'] for row in manifest['inputs']]:
        raise ValueError('entry forecasts require hot-call evidence from an input set held out of calibration')

    warmups = _integer(record.get('warmup_pairs'), 'hot-call warmup pairs', minimum=1)
    count = _integer(record.get('measured_pairs'), 'hot-call measured pairs', minimum=3)
    samples, counters = record.get('samples'), record.get('counters')
    if type(samples) is not list or len(samples) != count or count % 2 != 1:
        raise ValueError('hot-call evidence must have an odd number of recorded paired measurements')
    ambient = record.get('ambient_grad_enabled', False)
    if type(ambient) is not bool:
        raise ValueError('hot-call evidence must declare its ambient grad mode')
    negative_probes = {'CALL_ABI': 1, 'TENSOR_ABI' if ambient else 'RUNTIME_CONTEXT': 1}
    if any(len(spec['shape']) >= 2 and min(spec['shape'][-2:]) > 1 for spec in recipe_inputs):
        negative_probes['TENSOR_ABI'] = negative_probes.get('TENSOR_ABI', 0) + 1
    if (type(counters) is not dict
            or _integer(counters.get('native_calls'), 'guarded native count') != warmups + count + 1
            or _integer(counters.get('fallback_calls'), 'guarded fallback count') != sum(negative_probes.values())
            or counters.get('fallback_reasons') != negative_probes):
        raise ValueError('guarded timings must all run native, with fallbacks confined to probe cases')
    original, guarded = [], []
    for i, pair in enumerate(samples):
        if (type(pair) is not dict or type(pair.get('pair')) is not int or pair['pair'] != i
                or pair.get('order') != ('AB' if (i + warmups) % 2 == 0 else 'BA')):
            raise ValueError('invalid hot-call pair index or alternating order')
        original.append(_integer(pair.get('original_ns'), 'hot original callable time', minimum=1))
        guarded.append(_integer(pair.get('guarded_ns'), 'hot guarded callable time', minimum=1))
    metrics = {name: _metrics(times) for name, times in zip(('ORIGINAL', 'GUARDED_NATIVE'), (original, guarded))}
    reported = record.get('metrics')
    if type(reported) is not dict or any(
            type(reported.get(name)) is not dict or any(reported[name].get(key) != value for key, value in fields.items())
            for name, fields in metrics.items()):
        raise ValueError('reported hot-call metrics differ from the paired samples')
    return dict(warmup_pairs=warmups, measured_pairs=count, validated_inputs=receipts, ambient_grad_enabled=ambient,
                original_median_ns=metrics['ORIGINAL']['median_ns'],
                guarded_median_ns=metrics['GUARDED_NATIVE']['median_ns'])


def _estimate(observed, hot, startup_ns, *, gain_policy=ATEN_GAIN_POLICY):
    calls = observed['calls']
    original_total = observed['setup_ns'] + observed['entry_ns'] + observed['teardown_ns']
    noncall = original_total - observed['total_callable_ns']
    guarded_total = noncall + calls * hot['guarded_median_ns'] + startup_ns
    origin_per_call = Fraction(observed['total_callable_ns'], calls)
    saving_per_call = origin_per_call - hot['guarded_median_ns']
    estimate = dict(original_observed_ns=original_total, guarded_modeled_ns=guarded_total, original_noncall_ns=noncall,
                original_observed_average_callable_ns=float(origin_per_call),
                incremental_guarded_startup_ns=startup_ns,
                modeled_speedup=original_total / guarded_total,
                modeled_gain_fraction=(original_total - guarded_total) / original_total,
                break_even_minimum_calls_for_positive_gain=(floor(startup_ns / saving_per_call) + 1
                                                           if saving_per_call > 0 else None))
    legacy = gain_policy is None
    qualifies = (guarded_total * 20 <= original_total * 19 if legacy
                 else _passes_gain(original_total, guarded_total, gain_policy))
    threshold = Fraction(1, 20) if legacy else Fraction(str(gain_policy['minimum_gain']))
    qualified_saving_per_call = origin_per_call * (1 - threshold) - hot['guarded_median_ns']
    startup_limit = original_total * (1 - threshold) - noncall - calls * hot['guarded_median_ns']
    startup_limit = floor(startup_limit) if legacy else ceil(startup_limit) - 1
    minimum_calls = None
    if qualified_saving_per_call > 0:
        calls_at_threshold = (startup_ns + noncall * threshold) / qualified_saving_per_call
        minimum_calls = max(1, ceil(calls_at_threshold) if legacy else floor(calls_at_threshold) + 1)
    startup_limit = startup_limit if startup_limit >= 0 else None
    if legacy:
        return dict(estimate, meets_five_percent_modeled_gain=qualifies,
                    minimum_calls_for_five_percent_gain=minimum_calls,
                    maximum_startup_ns_for_five_percent_gain=startup_limit)
    return dict(estimate, gain_policy=dict(gain_policy), meets_minimum_modeled_gain=qualifies,
                minimum_calls_for_modeled_gain=minimum_calls, maximum_startup_ns_for_modeled_gain=startup_limit)


def forecast_aten_python_entry(trial, *, trial_sha256, demand, demand_sha256, call_evidence,
                               call_evidence_sha256, startup_overhead_ms, out):
    """Materialize the cost model without executing either entry variant."""
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    startup_ns = _startup_ns(startup_overhead_ms)
    trial = Path(trial).resolve(strict=True)
    manifest = _read_manifest(trial, trial_sha256)
    if manifest.get('input_guard', {}).get('kind') != _GEOMETRY:
        raise ValueError('an explicit value-independent geometry trial is required')
    demand_record, demand_ref = _read_pinned(demand, demand_sha256, _DEMAND_SCHEMA)
    hot_record, hot_ref = _read_pinned(call_evidence, call_evidence_sha256, _CALL_SCHEMA)
    invocation, observed = _observed_entry(demand_record, manifest, trial_sha256)
    hot = _hot_calls(hot_record, manifest, trial_sha256)
    grad_modes = observed['grad_mode_calls']
    if grad_modes is not None and grad_modes['enabled' if hot['ambient_grad_enabled'] else 'disabled'] != observed['calls']:
        raise ValueError('observed original calls and hot calibration have different grad modes')
    estimate = _estimate(observed, hot, startup_ns)
    recommendation = 'CANDIDATE_ENTRY_GAIN_FORECAST' if estimate['meets_minimum_modeled_gain'] else 'KEEP_ORIGINAL'
    report = dict(schema='scar.aten-python-entry-forecast', schema_version=1,
        scope='MODELED_IN_PROCESS_ENTRY_WITH_OBSERVED_ORIGINAL_DEMAND',
        trial=str(trial / 'trial.json'), trial_sha256=trial_sha256, cpu_ids=manifest['cpus'],
        entry_function=invocation['entry_function'], entry_args=invocation['entry_args'], demand_calls=observed['calls'],
        estimated_original_ns=estimate['original_observed_ns'], estimated_guarded_ns=estimate['guarded_modeled_ns'],
        invocation=invocation, original_demand=demand_ref, call_evidence=hot_ref,
        observed_original=observed, hot_call_evidence=hot, estimate=estimate,
        modeled_speedup=estimate['modeled_speedup'], recommendation=recommendation,
        assumptions=['UNCHANGED_OBSERVED_NONCALL_ENTRY_COST', 'ALL_OBSERVED_CALLS_USE_THE_GEOMETRY_GUARD',
                     'HELD_OUT_HOT_MEDIAN_REPRESENTS_EACH_ADMITTED_CALL',
                     'INCREMENTAL_GUARDED_STARTUP_IS_AN_EXPLICIT_SCENARIO',
                     'BREAK_EVEN_ESTIMATES_HOLD_NONCALL_COST_AND_PER_CALL_RATES_FIXED'],
        includes_interpreter_process_startup=False, timing_instrumentation_overhead_subtracted=False,
        point_forecast_has_runtime_bound=False, automatic_deployment_authorized=False,
        application_timing_verified=False, source_replacement_authorized=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-aten-entry-forecast-', dir=out.parent) as temporary:
        stage = Path(temporary)
        (stage / 'forecast.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
        stage.rename(out)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar forecast-aten-python-entry', description=__doc__)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--trial-sha256', required=True)
    parser.add_argument('--demand', required=True, help='pinned original entry observation JSON')
    parser.add_argument('--demand-sha256', required=True)
    parser.add_argument('--call-evidence', required=True, help='pinned held-out hot-call validation JSON')
    parser.add_argument('--call-evidence-sha256', required=True)
    parser.add_argument('--startup-overhead-ms', type=float, required=True,
                        help='incremental startup overhead of the guarded path in the declared scenario')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    report = forecast_aten_python_entry(**vars(args))
    print(json.dumps(dict(recommendation=report['recommendation'], modeled_speedup=report['modeled_speedup'],
                          report=str(Path(args.out) / 'forecast.json'))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

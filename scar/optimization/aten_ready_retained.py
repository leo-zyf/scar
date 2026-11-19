"""Price one lane-local Tensor replica retained across standalone calls."""

from statistics import median_low

from scar.optimization.aten_native_team import _bounded_duration
from scar.optimization.aten_ready_dag import _prediction
from scar.optimization.aten_ready_replica import _CALIBRATION_FIELDS, price_ready_replica


_CALIBRATION_RENAMES = {'replicated_samples_ns': 'retained_samples_ns', 'copy_samples_ns': 'check_samples_ns'}
_RETAINED_FIELDS = {
    *(_CALIBRATION_RENAMES.get(field, field) for field in _CALIBRATION_FIELDS),
    'cold_samples_ns',
}
_GAIN_PERCENT = 5


def price_ready_retained(recipe, schedule, model, candidate, raw, warmup):
    """Compare the matched schedule with a content-checked retained mapping.

    Warm calls use modeled critical paths. Cold calls use measured region
    wall time, including initialization. Amortization assumes one cold call
    followed by uninterrupted hits. Actual workload reuse is unknown.
    """
    if type(warmup) is not int or not 3 <= warmup <= 100:
        raise ValueError('ready retained calibration requires three to one hundred warmups')
    if type(raw) is not dict or raw.keys() != _RETAINED_FIELDS:
        raise ValueError('ready retained calibration requires matched direct, warm retained and cold observations')
    cold = raw['cold_samples_ns']
    if type(cold) is not list or len(cold) != warmup or not all(map(_bounded_duration, cold)):
        raise ValueError('ready retained calibration requires one bounded cold call in each calibration pair')

    # The same nominated mapping, page evidence and lane-local predecessor
    # apply. On a warm retained call the predecessor is an exact content check.
    report = price_ready_replica(recipe, schedule, model, candidate, {
        field: raw[_CALIBRATION_RENAMES.get(field, field)] for field in _CALIBRATION_FIELDS
    }, warmup)
    report.update(status='KEEP_DIRECT_READY', check_ns=report.pop('copy_ns'),
                  scope='CONCURRENT_PROVIDER_SAMPLE_CRITICAL_PATH_WITH_LANE_LOCAL_EXACT_CONTENT_CHECK',
                  cold_first_call=dict(_prediction(median_low(cold), cold),
                                       scope='MEASURED_INITIAL_RETAINED_REGION_WALL_TIME_WITHOUT_OPERATION_PROBES'),
                  break_even_total_calls=None, break_even_gain_percent=_GAIN_PERCENT,
                  break_even_basis='ONE_COLD_CALL_THEN_ALL_HITS', reuse_rate='UNKNOWN',
                  resident_mapping_count=1)
    if report['reason']:
        report['reason'] = report['reason'].replace('READY_REPLICA_', 'READY_RETAINED_', 1)
    if not report['predictions']:
        return report

    predictions = report['predictions']
    predictions['RETAINED'] = predictions.pop('REPLICATED')
    direct_ns = predictions['DIRECT']['estimate_ns']
    hot_ns = predictions['RETAINED']['estimate_ns']
    headroom = (100 - _GAIN_PERCENT) * direct_ns - 100 * hot_ns
    if headroom < 0:
        return dict(report, reason='READY_RETAINED_PREDICTED_GAIN_BELOW_THRESHOLD')

    # Solve C + (N-1)H <= N * D * (1-gain) for the first N with a hit.
    overhead = 100 * (report['cold_first_call']['estimate_ns'] - hot_ns)
    if headroom == 0 and overhead > 0:
        return dict(report, reason='READY_RETAINED_NO_FINITE_BREAK_EVEN')
    calls = max(2, (overhead + headroom - 1) // headroom) if headroom else 2
    return dict(report, status='PLAN_READY_RETAINED', reason=None, break_even_total_calls=calls)

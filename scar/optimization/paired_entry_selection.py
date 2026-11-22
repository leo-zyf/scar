"""Compare one incumbent and one challenger after profile-specific admission."""

from fractions import Fraction
from numbers import Real
import re


SCHEMA = 'scar.paired-entry-selection'
DEFAULT_MINIMUM_GAIN = 0.01
_RUN_FIELDS = {'pair', 'mode', 'wall_ns', 'returncode', 'stdout_sha256', 'stderr_sha256'}
_DIGEST = re.compile('[0-9a-f]{64}')


def _median_twice(ordered):
    return ordered[(len(ordered) - 1) // 2] + ordered[len(ordered) // 2]


def strict_entry_gain_met(original_ns, candidate_ns, minimum_gain):
    """Compare integer elapsed samples at an exact decimal gain boundary."""
    return (_median_twice(sorted(candidate_ns))
            < (1 - Fraction(str(minimum_gain))) * _median_twice(sorted(original_ns)))


def _metrics(samples):
    ordered = sorted(samples)
    median_twice = _median_twice(ordered)
    median_ns = median_twice // 2 if median_twice % 2 == 0 else median_twice / 2
    return dict(samples_ns=samples, min_ns=ordered[0], max_ns=ordered[-1],
                median_ns=median_ns, median_ns_twice=median_twice)


def compare_entry_pairs(runs, *, incumbent, challenger, warmup_pairs, measurement_pairs,
                        minimum_gain=DEFAULT_MINIMUM_GAIN, strict_minimum_gain=True):
    """Reduce matched fresh-process timing rows supplied by an evidence adapter.

    The adapter is responsible for matching context, treatment identity and raw
    receipts before projection. Bounds are observed sample extrema. The result
    can send a candidate for review without granting execution authority.
    """
    modes = incumbent, challenger
    if (any(type(mode) is not str or not mode for mode in modes) or incumbent == challenger
            or type(warmup_pairs) is not int or warmup_pairs < 0
            or type(measurement_pairs) is not int or measurement_pairs < 3):
        raise ValueError('two distinct modes and at least three measured pairs are required')
    if type(strict_minimum_gain) is not bool:
        raise ValueError('minimum gain comparator must be a boolean')
    if isinstance(minimum_gain, bool) or not isinstance(minimum_gain, Real):
        raise ValueError('minimum gain must be a finite fraction in [0, 1)')
    try:
        minimum = Fraction(str(minimum_gain))
    except ValueError as error:
        raise ValueError('minimum gain must be a finite fraction in [0, 1)') from error
    if not 0 <= minimum < 1:
        raise ValueError('minimum gain must be a finite fraction in [0, 1)')

    count = warmup_pairs + measurement_pairs
    if type(runs) is not list or len(runs) != 2 * count:
        raise ValueError('exactly one run per mode and pair is required')
    timings, outputs = {}, set()
    for row in runs:
        if (type(row) is not dict or set(row) != _RUN_FIELDS
                or type(row['pair']) is not int or not 0 <= row['pair'] < count
                or type(row['mode']) is not str or row['mode'] not in modes
                or type(row['wall_ns']) is not int or row['wall_ns'] <= 0
                or type(row['returncode']) is not int or row['returncode'] != 0
                or any(type(row[field]) is not str or _DIGEST.fullmatch(row[field]) is None
                       for field in ('stdout_sha256', 'stderr_sha256'))):
            raise ValueError('invalid paired full-entry timing row')
        key = row['pair'], row['mode']
        if key in timings:
            raise ValueError('duplicate run for the same mode and pair')
        timings[key] = row['wall_ns']
        outputs.add((row['stdout_sha256'], row['stderr_sha256']))
    if len(outputs) != 1:
        raise ValueError('paired runs changed stdout or stderr')

    metrics = {mode: _metrics([timings[pair, mode] for pair in range(warmup_pairs, count)]) for mode in modes}
    baseline, candidate = metrics[incumbent], metrics[challenger]
    median_gain = Fraction(baseline['median_ns_twice'] - candidate['median_ns_twice'], baseline['median_ns_twice'])
    separated = candidate['max_ns'] < baseline['min_ns']
    reasons = []
    if median_gain <= 0 or median_gain < minimum or (strict_minimum_gain and median_gain == minimum):
        reasons.append('MEASURED_MEDIAN_GAIN_BELOW_THRESHOLD')
    if not separated:
        reasons.append('CHALLENGER_MAX_NS_NOT_BELOW_INCUMBENT_MIN_NS')
    policy = dict(minimum_gain=float(minimum), strict_observed_separation=True)
    if strict_minimum_gain:
        policy['strict_minimum_gain'] = True
    return dict(schema=SCHEMA, schema_version=1, incumbent=incumbent, challenger=challenger,
                warmup_pairs=warmup_pairs, measurement_pairs=measurement_pairs, metrics=metrics,
                paired_improvement_ns=[old - new for old, new in zip(baseline['samples_ns'], candidate['samples_ns'])],
                gain_fraction=float(median_gain), observed_envelopes_separated=separated,
                policy=policy,
                verdict='KEEP_INCUMBENT' if reasons else 'ADVANCE_TO_RELEASE_REVIEW',
                reasons=reasons, deployment_authorized=False)


def replay_entry_pairs(runs, archived, *, incumbent, challenger, warmup_pairs, measurement_pairs):
    """Recompute a paired decision using the exact policy encoded in its receipt."""
    if type(archived) is not dict or type(archived.get('policy')) is not dict:
        raise ValueError('archived paired selection policy required')
    policy = archived['policy']
    replayed = compare_entry_pairs(runs, incumbent=incumbent, challenger=challenger,
                                  warmup_pairs=warmup_pairs, measurement_pairs=measurement_pairs,
                                  minimum_gain=policy.get('minimum_gain'),
                                  strict_minimum_gain=policy.get('strict_minimum_gain', False))
    if replayed != archived:
        raise ValueError('archived paired comparison did not replay')
    return replayed

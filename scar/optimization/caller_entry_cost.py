"""Conditional whole-process accounting from paired, nested observation windows."""
from fractions import Fraction
from math import floor
from statistics import median

from scar.optimization.paired_entry_selection import DEFAULT_MINIMUM_GAIN


def _interval(values):
    return min(values), median(values), max(values)


def entry_phase_sensitivity(pairs, *, observed_calls, requested_calls, minimum_gain=DEFAULT_MINIMUM_GAIN):
    """Compare a fixed per-entry remainder and a proportional repeated window.

    The window is timed inside the exact matching process sample on each side.
    Its remainder covers all other work, including startup and output. The
    caller must separately establish whether the regime transfers to a new
    invocation. Sample ranges are empirical, and this model cannot deploy code.
    """
    if (type(observed_calls) is not int or observed_calls < 1 or type(requested_calls) is not int
            or requested_calls < 1 or not 0 <= minimum_gain < 1 or len(pairs) < 3):
        raise ValueError('positive observed/requested counts and at least three paired samples required')
    outside, inside = {}, {}
    sides = ('original', 'candidate')
    for side in sides:
        window = [pair[side]['repeated_window_ns'] for pair in pairs]
        wall = [pair[side]['wall_ns'] for pair in pairs]
        if any(type(w) is not int or type(t) is not int or not 0 <= w <= t or t == 0
               for w, t in zip(window, wall, strict=True)):
            raise ValueError('each repeated window must fit its own complete-process duration')
        outside[side] = _interval([t-w for w, t in zip(window, wall, strict=True)])
        inside[side] = _interval([Fraction(ns, observed_calls) for ns in window])
    baseline, candidate = sides
    total = {side: [s + requested_calls*c for s, c in zip(outside[side], inside[side], strict=True)]
             for side in sides}
    gain = Fraction(str(minimum_gain))
    # Worst empirical combination.  The count gate uses a strict time gain,
    # even at a zero configured fractional threshold.
    fixed = outside[candidate][2] - (1-gain)*outside[baseline][0]
    marginal = (1-gain)*inside[baseline][0] - inside[candidate][2]
    crossing = (1 if fixed - marginal < 0 else
                floor(fixed/marginal)+1 if marginal > 0 else None)
    paired_outside_delta = [(pair[candidate]['wall_ns']-pair[candidate]['repeated_window_ns'])
        - (pair[baseline]['wall_ns']-pair[baseline]['repeated_window_ns']) for pair in pairs]
    favorable = total[candidate][2] < (1-gain)*total[baseline][0]
    return dict(observed_count=observed_calls, requested_count=requested_calls,
        outside_window_ns={side:list(values) for side, values in outside.items()},
        repeated_ns_per_call={side:[float(value) for value in values] for side, values in inside.items()},
        candidate_extra_outside_window_ns=list(_interval(paired_outside_delta)),
        complete_process_ns={side:[float(value) for value in values] for side, values in total.items()},
        minimum_gain=minimum_gain,
        conditional_first_count_passing_sample_envelope=crossing,
        decision='REVIEW_EXACT_ENTRY' if favorable else 'NO_OP',
        assumption='FIXED_REMAINDER_AND_STATIONARY_PER_CALL_COST_IN_THE_SAME_REGIME',
        cost_basis='OBSERVED_PHASE_SENSITIVITY_NOT_PROSPECTIVE_CALIBRATION',
        ranges_are_physical_bounds=False, deployment_authorized=False)

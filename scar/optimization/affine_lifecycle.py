"""Extrapolate a serial owner lifecycle under an explicit affine service model."""

from fractions import Fraction
from itertools import combinations

from scar.ir.execution_recipe import _canonical


SCHEMA = 'scar.affine-owner-lifecycle-cost'
WINDOWS = ('before_recurrence', 'recurrence', 'after_recurrence')


def _fraction(value):
    if value is None:
        return None
    return dict(numerator=value.numerator, denominator=value.denominator)


def _median(samples):
    ordered = sorted(samples)
    size = len(ordered)
    return Fraction(ordered[(size - 1) // 2] + ordered[size // 2], 2)


def _summary(samples):
    return dict(sample_count=len(samples), min=min(samples), median=_fraction(_median(samples)), max=max(samples))


def _estimate(low, point, high):
    return dict(lower=_fraction(low), point=_fraction(point), upper=_fraction(high))


def _recurrence_vertices(counts, samples):
    """Enumerate the vertices of the bounded two-coefficient calibration domain."""
    envelopes = [(count, min(values), max(values)) for count, values in zip(counts, samples, strict=True)]
    lines = [(1, 0, 0), (0, 1, 0)]
    for count, minimum, maximum in envelopes:
        lines.extend(((1, count, minimum), (1, count, maximum)))
    vertices = set()
    for (a, b, c), (d, e, f) in combinations(lines, 2):
        determinant = a * e - b * d
        if determinant == 0:
            continue
        intercept = Fraction(c * e - b * f, determinant)
        step = Fraction(a * f - c * d, determinant)
        if intercept >= 0 and step >= 0 and all(
                low <= intercept + count * step <= high for count, low, high in envelopes):
            vertices.add((intercept, step))
    return sorted(vertices)


def fit_affine_lifecycle(training_by_count, *, predict_steps):
    """Fit the three measured intervals of one independently bound launch context.

    Training maps two distinct positive step counts to three lists of positive
    integer nanosecond samples. The caller pins context, markers and ordering.
    Sample envelopes constrain the conditional model, with no physical or
    statistical coverage guarantee for a different execution.
    """
    if (type(training_by_count) is not dict or len(training_by_count) != 2
            or any(type(count) is not int or count <= 0 for count in training_by_count)
            or type(predict_steps) is not int or predict_steps <= 0):
        raise ValueError('exactly two distinct positive training counts and one positive prediction count required')
    for windows in training_by_count.values():
        if (type(windows) is not dict or set(windows) != set(WINDOWS)
                or any(type(samples) not in (list, tuple) or len(samples) < 2
                       or any(type(value) is not int or value <= 0 for value in samples)
                       for samples in windows.values())):
            raise ValueError('all three phase windows require at least two positive integer nanosecond samples')

    counts = sorted(training_by_count)
    low_count, high_count = counts
    training = [dict(steps=count, windows={window: _summary(training_by_count[count][window]) for window in WINDOWS})
                for count in counts]
    samples = [training_by_count[count]['recurrence'] for count in counts]
    first, second = map(_median, samples)
    step = (second - first) / (high_count - low_count)
    intercept = first - low_count * step
    point_coefficients = intercept >= 0 and step >= 0
    vertices = _recurrence_vertices(counts, samples)

    prediction = {}
    windows = {}
    for name in ('before_recurrence', 'after_recurrence'):
        samples = [sample for count in counts for sample in training_by_count[count][name]]
        windows[name] = min(samples), _median(samples), max(samples)
        prediction[name] = _estimate(*windows[name])
    if vertices:
        service = [base + predict_steps * slope for base, slope in vertices]
        windows['recurrence'] = (min(service), intercept + predict_steps * step if point_coefficients else None, max(service))
        prediction['recurrence'] = _estimate(*windows['recurrence'])
        prediction['total'] = _estimate(*(sum(values) if None not in values else None
                                          for values in zip(*windows.values(), strict=True)))
    else:
        prediction['recurrence'] = prediction['total'] = _estimate(None, None, None)

    return dict(schema=SCHEMA, schema_version=1,
        status='CONDITIONAL_ESTIMATE' if vertices else 'AFFINE_CALIBRATION_INFEASIBLE',
        training_steps=counts, training=training, predict_steps=predict_steps,
        prediction_relation=('EXTRAPOLATION_ABOVE' if predict_steps > high_count else
                             'EXTRAPOLATION_BELOW' if predict_steps < low_count else 'WITHIN_CALIBRATION_RANGE'),
        recurrence_model=dict(form='INTERCEPT_NS_PLUS_STEPS_TIMES_STEP_NS', nonnegative_coefficients=True,
            point_coefficients=(dict(intercept_ns=_fraction(intercept), step_ns=_fraction(step)) if point_coefficients else None),
            point_status='NONNEGATIVE_MEDIAN_FIT' if point_coefficients else 'MEDIAN_FIT_HAS_NEGATIVE_COEFFICIENT',
            feasible_vertices=[dict(intercept_ns=_fraction(base), step_ns=_fraction(slope)) for base, slope in vertices]),
        prediction_ns=prediction, total_interval_method='SUM_OF_SERIAL_EMPIRICAL_WINDOW_EXTREMA_NO_INDEPENDENCE_ASSUMED',
        point_method='SUM_OF_POOLED_WING_MEDIANS_AND_AFFINE_RECURRENCE_MEDIAN_FIT',
        assumptions=['SAME_OWNER_RESOURCES_INPUTS_AND_MARKER_ORDER', 'WING_SAMPLE_ENVELOPES_APPLY_AT_PREDICTION_COUNT',
                     'RECURRENCE_FOLLOWS_ONE_NONNEGATIVE_AFFINE_SERVICE_MODEL'],
        basis='CONDITIONAL_EMPIRICAL_COMPONENT_MODEL', physical_time_bound=False,
        statistical_coverage_guaranteed=False, whole_entry_legality_proven=False, deployment_authorized=False)


def verify_affine_lifecycle(result, training_by_count, *, predict_steps):
    """Recompute the exact arithmetic and compare its canonical representation."""
    expected = fit_affine_lifecycle(training_by_count, predict_steps=predict_steps)
    if _canonical(result) != _canonical(expected):
        raise ValueError('affine lifecycle model differs from exact calibration replay')
    return expected

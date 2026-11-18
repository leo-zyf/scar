"""Historical raw floats expose arithmetic-order drift in exact validation."""
from scar.compiler.stateful_holdout import _prediction_evaluation


def test_prediction_evaluation_exactly_matches_historical_capture_and_still_rejects():
    original,native,predicted=1967329,1727909.25,919245
    assert (original-native)/original!=1-native/original
    policy=dict(max_relative_prediction_error=0.2,minimum_observed_gain=0.05)
    # Literal five-field observation from matched-entry-cost-v1. This test
    # neither rewrites that capture nor gives it authority under new pins.
    recorded=dict(predicted_native_method_ns=919245,
        relative_prediction_error=0.4680015747354787,
        observed_method_gain=0.12169787056460815,
        prediction_error_threshold_passed=False,observed_gain_threshold_passed=True)
    replay=_prediction_evaluation(predicted,original,native,policy)
    assert replay==recorded
    assert not replay['prediction_error_threshold_passed']
    assert replay['relative_prediction_error']>policy['max_relative_prediction_error']

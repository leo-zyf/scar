"""Measured implementation selection within the guarded array-region contract."""

import math
from numbers import Real
import statistics

from scar.compiler.model import ExecutionPlan
from scar.compiler.validation import validate_plan


def _samples(values):
    if isinstance(values, (str, bytes, dict)):
        return (), "invalid_measurements"
    try:
        samples = tuple(values)
    except TypeError:
        return (), "no_measurements" if values is None else "invalid_measurements"
    if not samples:
        return (), "no_measurements"
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in samples):
        return (), "invalid_measurements"
    try:
        samples = tuple(float(v) for v in samples)
    except (ValueError, TypeError, OverflowError):
        return (), "invalid_measurements"
    if any(not math.isfinite(v) or v <= 0 for v in samples):
        return (), "invalid_measurements"
    return samples, None


def _summary(samples):
    return {"count": len(samples), "units": "seconds",
            "median": statistics.median(samples) if samples else None,
            "min": min(samples) if samples else None,
            "max": max(samples) if samples else None,
            "stdev": (statistics.stdev(samples) if len(samples) > 1 else
                      0.0 if samples else None)}


def select_measured(plan, baseline_seconds, native_seconds, *, validated,
                    native_fallbacks=0, minimum_gain=0.05) -> dict:
    """Select only a validated, faster implementation under its explicit contract.

    Timing samples must be positive finite seconds. The gain is the reduction in
    median elapsed time relative to the baseline, not a predicted speedup. A
    result never mutates the plan, proves global purity, or broadens its source
    identity, runtime guards or external assumptions.
    """
    baseline, baseline_error = _samples(baseline_seconds)
    native, native_error = _samples(native_seconds)
    reasons = []
    for label, error in (("baseline", baseline_error), ("native", native_error)):
        if error:
            reasons.append(f"{label}_{error}")
    proof = plan.proof if isinstance(plan, ExecutionPlan) and isinstance(plan.proof, dict) else {}
    legality = proof.get("legality", {})
    assumptions = (legality.get("external_assumptions", ())
                   if isinstance(legality, dict) else ())
    if not isinstance(assumptions, (list, tuple)) or any(not isinstance(a, str) for a in assumptions):
        assumptions = ()
        reasons.append("invalid_external_assumptions")
    guards = proof.get("runtime_guards", ())
    if not isinstance(guards, (list, tuple)) or any(not isinstance(g, str) for g in guards):
        guards = ()
        reasons.append("invalid_runtime_guards")
    if not isinstance(plan, ExecutionPlan) or proof.get("supported") is not True:
        reasons.append("unknown_or_unsupported_profile")
    elif proof.get("legal") is not True or proof.get("lowering_eligible") is not True:
        reasons.append("plan_not_eligible_for_lowering")
    else:
        try:
            validate_plan(plan)
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
            reasons.append("invalid_execution_plan")
    required = {"NO_FOREIGN_CONCURRENT_INPUT_WRITERS", "SYNCHRONOUS_CPU_EXECUTION"}
    if not required.issubset(assumptions):
        reasons.append("missing_external_contract_assumptions")
    return _finish_decision(baseline, native, reasons, assumptions, guards,
                            validated, native_fallbacks, minimum_gain)


def _finish_decision(baseline, native, reasons, assumptions, guards,
                     validated, native_fallbacks, minimum_gain):
    """Cost policy shared by separately admitted semantic profiles."""
    if validated is not True:
        reasons.append("correctness_not_validated")
    if type(native_fallbacks) is not int or native_fallbacks < 0:
        reasons.append("invalid_fallback_count")
    elif native_fallbacks:
        reasons.append("native_fallbacks_observed")
    valid_threshold = (not isinstance(minimum_gain, bool) and isinstance(minimum_gain, Real)
                       and math.isfinite(minimum_gain) and 0 <= minimum_gain < 1)
    if not valid_threshold:
        reasons.append("invalid_minimum_gain")
    baseline_stats, native_stats = _summary(baseline), _summary(native)
    speedup = (baseline_stats["median"] / native_stats["median"]
               if baseline and native else None)
    gain = (1.0 - native_stats["median"] / baseline_stats["median"]
            if baseline and native else None)
    if speedup is not None and (not math.isfinite(speedup) or not math.isfinite(gain)):
        reasons.append("nonfinite_derived_statistics")
        speedup, gain = None, None
    if valid_threshold and gain is not None and (gain <= 0 or
            (gain < minimum_gain and not math.isclose(gain, minimum_gain,
                                                      rel_tol=1e-12, abs_tol=0.0))):
        reasons.append("measured_gain_below_required_threshold")
    selected = not reasons
    return {
        "status": "MEASURED",
        "decision": "TRANSFORM_UNDER_CONTRACT" if selected else "NO_OP",
        "selected": selected,
        "baseline": baseline_stats, "native": native_stats,
        "speedup": speedup, "gain_fraction": gain,
        "minimum_gain": float(minimum_gain) if valid_threshold else None,
        "reasons": reasons if reasons else ["validated_native_median_meets_gain_threshold"],
        "external_assumptions": list(assumptions),
        "runtime_guards": list(guards),
        "scope": "Selection applies only to the validated workload and explicit profile "
                 "contract; it does not prove global purity or new provenance.",
    }


def select_stateful_measured(artifact, baseline_seconds, native_seconds, *, validated,
                             native_fallbacks=0, minimum_gain=0.05):
    """Admit one measured stateful implementation, retaining original execution.

    Legality is checked through the ordered stateful plan, independently of the
    pure-array profile. No comparison expands source, shape or provider scope.
    """
    from scar.compiler.stateful_model import validate_stateful_region
    from scar.optimization.stateful_schedule import verify_stateful_plan

    baseline, baseline_error = _samples(baseline_seconds)
    native, native_error = _samples(native_seconds)
    reasons = [label + '_' + error for label, error in
               (('baseline', baseline_error), ('native', native_error)) if error]
    region, plan, report = artifact.region, artifact.plan, artifact.report
    try:
        validate_stateful_region(region)
        verify_stateful_plan(region, plan)
        if (plan.get('frontier') or plan.get('frontiers') or plan.get('budget_satisfied') is not True
                or plan.get('status') != 'READY' or plan.get('allocation_search_complete') is not True):
            reasons.append('stateful_plan_not_executable_within_budget')
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        reasons.append('invalid_stateful_execution_plan')
    assumptions = plan.get('external_assumptions', ())
    if (not isinstance(assumptions, (list, tuple))
            or not {'SYNCHRONOUS_CPU_EXECUTION', 'NO_FOREIGN_CONCURRENT_STATE_READERS_OR_WRITERS',
                    'FP_STATUS_FLAGS_NOT_USED_AS_PROGRAM_OUTPUT'}.issubset(assumptions)):
        assumptions = ()
        reasons.append('missing_stateful_external_contract_assumptions')
    result = _finish_decision(baseline, native, reasons, assumptions, region.guards,
                             validated, native_fallbacks, minimum_gain)
    result.update(profile=region.profile,
        artifact={key: report.get(key) for key in ('build_id', 'plan_sha256')},
        input_shapes=plan.get('input_shapes', {}),
        automatic_runner_cost_context_qualified=False,
        fresh_process_scope_measured=False)
    return result

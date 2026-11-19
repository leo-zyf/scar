"""Compare closed service sample envelopes for native placement treatments."""


def _envelope(row):
    interval = row.get('service_interval') or {}
    low, mid, high = (interval.get(name) for name in ('lower_ns', 'estimate_ns', 'upper_ns'))
    if (row['status'] == 'VERIFIED_SCHEDULE' and interval.get('status') == 'BOUNDED'
            and all(type(value) is int for value in (low, mid, high)) and 0 <= low <= mid <= high):
        return low, high
    return 0, None


def _comparison_blocker(ordinal, upper, other, lower):
    reason = 'SAMPLE_ENVELOPES_TOUCH' if upper == lower else 'SAMPLE_ENVELOPES_OVERLAP'
    return dict(candidate_ordinal=ordinal, other_ordinal=other, candidate_upper_ns=upper,
                other_lower_ns=lower, reason=reason)


def choose_native_device_interval(evaluated, baseline_ordinal=0, *, enumeration_complete=True):
    """Nominate a GPU only when its worst sample-model cost beats all rivals."""
    if type(baseline_ordinal) is not int or baseline_ordinal < 0 or type(enumeration_complete) is not bool:
        raise ValueError('expected a nonnegative baseline ordinal and an explicit enumeration status')
    bounds, seen, selected = {}, set(), None
    for row in evaluated:
        ordinal = row['ordinal']
        if type(ordinal) is not int or ordinal < 0 or ordinal in seen:
            raise ValueError('placement ordinals must be unique nonnegative integers')
        seen.add(ordinal)
        if row['status'] != 'PLACEMENT_CAPACITY_EXCEEDED':
            bounds[ordinal] = _envelope(row)
        if ordinal == baseline_ordinal and row['status'] == 'VERIFIED_SCHEDULE':
            selected = baseline_ordinal

    incumbent = bounds.get(baseline_ordinal, (0, None))
    blockers = [dict(ordinal=ordinal, lower_ns=low, upper_ns=high, reason='UNBOUNDED_SERVICE_INTERVAL')
                for ordinal, (low, high) in bounds.items() if high is None]
    if baseline_ordinal not in bounds:
        blockers.append(dict(ordinal=baseline_ordinal, reason='CPU_BASELINE_NOT_AVAILABLE'))
    if not enumeration_complete:
        blockers.append(dict(reason='PLACEMENT_BUDGET_EXHAUSTED'))

    if incumbent[1] is None:
        decision = 'CPU_INCUMBENT_UNRESOLVED'
    elif not enumeration_complete:
        decision = 'PLACEMENT_BUDGET_EXHAUSTED'
    elif blockers:
        decision = 'PLACEMENT_DOMAIN_UNRESOLVED'
    else:
        alternatives = [(ordinal, interval) for ordinal, interval in bounds.items() if ordinal != baseline_ordinal]
        if all(incumbent[1] <= interval[0] for _, interval in alternatives):
            decision = 'CPU_INTERVAL_DOMINANCE'
        else:
            # The least upper endpoint is the only possible strict winner.
            best, best_interval = min(alternatives, key=lambda row: (row[1][1], row[0]))
            blockers = [_comparison_blocker(best, best_interval[1], ordinal, interval[0])
                        for ordinal, interval in bounds.items() if ordinal != best and best_interval[1] >= interval[0]]
            if blockers:
                decision = 'INTERVALS_OVERLAP'
            else:
                decision, selected = 'STRICT_INTERVAL_GAIN', best

    return dict(schema='scar.native-device-interval-choice', schema_version=1, decision=decision,
                baseline_ordinal=baseline_ordinal, selected_ordinal=selected, blockers=blockers,
                scope='OBSERVED_SERVICE_SAMPLE_ENVELOPES_ONLY', statistical_confidence_claimed=False,
                physical_optimality_proven=False, deployment_authorized=False)

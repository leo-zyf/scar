"""Reduce matched full-entry timings for competing guarded ATen schedules."""

import re
from statistics import variance

from scar.optimization.paired_entry_selection import _metrics


_ORIGINAL = 'ORIGINAL'
_LABEL = re.compile(r'[A-Z][A-Z0-9_]*')
_CHANNELS = ('stdout', 'stderr', 'output_tree', 'entry_return', 'grad_enabled_at_entry', 'grad_enabled_at_return')
_BOUNDARIES = ('process_ns', 'entry_ns')


def entry_round_orders(candidates, *, warmup_rounds, measured_rounds):
    """Assign reverse order to consecutive measured rounds for each pair of treatments."""
    if (type(candidates) not in (list, tuple) or not 1 <= len(candidates) <= 2
            or any(type(label) is not str or _LABEL.fullmatch(label) is None or label == _ORIGINAL
                   for label in candidates) or len(set(candidates)) != len(candidates)):
        raise ValueError('one or two distinct guarded candidate names are required')
    if (type(warmup_rounds) is not int or not 0 <= warmup_rounds <= 100
            or type(measured_rounds) is not int or not 4 <= measured_rounds <= 500 or measured_rounds % 2):
        raise ValueError('candidate comparison requires 0–100 warmups and 4–500 even measured rounds')
    labels = [_ORIGINAL, *candidates]

    def rotate(index):
        start = (index // 2) % len(labels)
        order = labels[start:] + labels[:start]
        return order[::-1] if index % 2 else order

    return [rotate(i) for count in (warmup_rounds, measured_rounds) for i in range(count)]


def _check_rounds(rounds, orders, warmup_rounds):
    if type(rounds) is not list or len(rounds) != len(orders):
        raise ValueError('one full-entry result is required for each planned round')
    processes = set()
    for index, (row, order) in enumerate(zip(rounds, orders, strict=True)):
        if (type(row) is not dict or type(row.get('round')) is not int or row['round'] != index
                or row.get('warmup') is not (index < warmup_rounds) or row.get('order') != order
                or type(row.get('runs')) is not dict or set(row['runs']) != set(order)):
            raise ValueError('round results must follow the exact matched entry experiment order')
        for label, run in row['runs'].items():
            mode = _ORIGINAL if label == _ORIGINAL else 'GUARDED'
            if (type(run) is not dict or not set(_CHANNELS) <= run.keys() or run.get('mode') != mode
                    or any(type(run.get(field)) is not int or run[field] <= 0
                           for field in ('process_id', *_BOUNDARIES))
                    or any(type(run.get(field)) is not int or run[field] < 0
                           for field in ('native_calls', 'fallback_calls'))
                    or run['entry_ns'] > run['process_ns'] or run['process_id'] in processes
                    or type(run.get('exit_status')) is not int or run['exit_status'] != 0
                    or any(type(run[field]) is not bool
                           for field in ('grad_enabled_at_entry', 'grad_enabled_at_return'))):
                raise ValueError('each entry result requires a distinct successful fresh process')
            if label == _ORIGINAL and (run['native_calls'] or run['fallback_calls']):
                raise ValueError('the original entry result must have no guarded dispatch')
            processes.add(run['process_id'])


def _coverage(timed, candidate, expected):
    runs = [row['runs'][candidate] for row in timed]
    native = [run['native_calls'] for run in runs]
    fallback = sum(run['fallback_calls'] for run in runs)
    positive_constant = native[0] > 0 and all(count == native[0] for count in native)
    return dict(native_calls_per_process=native, fallback_calls=fallback, expected_native_calls=expected,
                coverage_requirement_met=positive_constant and fallback == 0
                and (expected is None or native[0] == expected))


def _compare(timed, metrics, reference, candidate):
    deltas = {'REFERENCE_FIRST': [], 'CANDIDATE_FIRST': []}
    differences = []
    for row in timed:
        runs = row['runs']
        direction = 'REFERENCE_FIRST' if row['order'].index(reference) < row['order'].index(candidate) else 'CANDIDATE_FIRST'
        delta = runs[reference]['process_ns'] - runs[candidate]['process_ns']
        deltas[direction].append(delta)
        differences.append(delta)
    directional = {name: _metrics(values) for name, values in deltas.items()}
    reference_median = metrics[reference]['process_ns']['median_ns_twice']
    candidate_median = metrics[candidate]['process_ns']['median_ns_twice']
    faster = sum(delta > 0 for delta in differences)
    checks = dict(median_gain_at_least_five_percent=20 * candidate_median <= 19 * reference_median,
                  at_least_three_quarters_rounds_faster=4 * faster >= 3 * len(timed),
                  positive_median_gain_in_each_order=all(group['median_ns_twice'] > 0 for group in directional.values()))
    return dict(boundary='FRESH_PROCESS_WALL_TIME', checks=checks, profitable=all(checks.values()),
                faster_rounds=faster, paired_improvement_ns=differences,
                median_delta_ns_by_order={direction: group['median_ns'] for direction, group in directional.items()},
                speedup=reference_median / candidate_median)


def select_aten_entry_candidate(rounds, candidates, *, warmup_rounds, measured_rounds, expected_native_calls):
    """Advance a verified challenger only when it beats the original and the incumbent."""
    orders = entry_round_orders(candidates, warmup_rounds=warmup_rounds, measured_rounds=measured_rounds)
    if expected_native_calls is not None and (type(expected_native_calls) is not int or expected_native_calls < 1):
        raise ValueError('the expected native call count must be a positive integer')
    _check_rounds(rounds, orders, warmup_rounds)
    timed = rounds[warmup_rounds:]
    labels = [_ORIGINAL, *candidates]
    metrics = {}
    for label in labels:
        metrics[label] = {}
        for boundary in _BOUNDARIES:
            samples = [row['runs'][label][boundary] for row in timed]
            metrics[label][boundary] = dict(_metrics(samples), variance_ns2=variance(samples))
    coverage = {candidate: _coverage(timed, candidate, expected_native_calls) for candidate in candidates}
    equivalence = {}
    for candidate in candidates:
        matches = {
            field: all(row['runs'][candidate][field] == row['runs'][_ORIGINAL][field] for row in rounds)
            for field in _CHANNELS
        }
        equivalence[candidate] = dict(matches, equivalent=all(matches.values()))
    comparisons = {candidate: {reference: _compare(timed, metrics, reference, candidate)
                               for reference in labels if reference != candidate}
                   for candidate in candidates}

    selected = _ORIGINAL
    decisions = {}
    for candidate in candidates:
        reasons = []
        if not coverage[candidate]['coverage_requirement_met']:
            reasons.append('INCOMPLETE_NATIVE_CALL_COVERAGE')
        if not equivalence[candidate]['equivalent']:
            reasons.append('APPLICATION_OUTPUT_MISMATCH')
        if not comparisons[candidate][_ORIGINAL]['profitable']:
            reasons.append('INSUFFICIENT_GAIN_OVER_ORIGINAL')
        if selected != _ORIGINAL:
            if coverage[candidate]['native_calls_per_process'] != coverage[selected]['native_calls_per_process']:
                reasons.append('NATIVE_CALL_COVERAGE_DIFFERS_FROM_INCUMBENT')
            if not comparisons[candidate][selected]['profitable']:
                reasons.append('INSUFFICIENT_GAIN_OVER_INCUMBENT')
        decisions[candidate] = dict(incumbent=selected, promoted=not reasons, reasons=reasons)
        if not reasons:
            selected = candidate

    speedups = {boundary: metrics[_ORIGINAL][boundary]['median_ns_twice'] / metrics[selected][boundary]['median_ns_twice']
                for boundary in _BOUNDARIES}
    return dict(selected=selected, metrics=metrics, comparisons=comparisons, coverage=coverage,
                equivalence=equivalence, decisions=decisions, speedups=speedups,
                recommendation='KEEP_ORIGINAL' if selected == _ORIGINAL else 'CANDIDATE_ENTRY_GAIN_OBSERVED',
                automatic_deployment_authorized=False)

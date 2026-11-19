"""Bind measured owner spans to the admitted source order without pricing a native plan."""
from __future__ import annotations

import json

from scar.ir.execution_recipe import read_artifact
from scar.runtime.sparse_ordered_scope_observe import CLOCKS, COST, PUBLICATION, cost_boundaries


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_source_segments(design, observation):
    """Check process identity and necessary bounds for compact owner summaries."""
    witness = observation['segmented_cost']
    phases = witness['phases']
    order = [*design['ordered_children'], PUBLICATION]
    _require(witness['schema'] == COST and witness['schema_version'] == 1
        and observation['entry'] == design['entry'] and observation['argv'] == design['argv']
        and observation['local_event_counts'] == design['expected_counts']
        and all(witness[key] == observation[key] for key in ('process_id', 'thread_id', 'affinity'))
        and witness['scope'] == 'INSTRUMENTED_ORIGINAL_OWNER_ONLY'
        and witness['clock_sources'] == dict(wall_ns='PERF_COUNTER_NS', thread_cpu_ns='THREAD_TIME_NS')
        and witness['boundaries'] == cost_boundaries(design)
        and witness['phase_order'] == order and set(phases) == set(order)
        and witness['clean_measurement'] is False and witness['occurrence_intervals_saved'] is False
        and set(witness['envelope']) == set(CLOCKS), 'owner segment scope or exact observation sites differ')
    count = design['count']
    _require(all(type(row['count']) is int and row['count'] == count
        and set(row) == {'count', *CLOCKS} for row in phases.values()),
        'each retained owner phase requires the same number of normal occurrences')

    measured = {}
    for axis in CLOCKS:
        first, last = witness['envelope'][axis]
        _require(type(first) is int and type(last) is int and 0 <= first < last, 'invalid owner clock envelope')
        parts = [phases[name][axis] for name in order]
        for part in parts:
            start, stop, total, lower, upper = (part[key] for key in
                ('first_ns', 'last_ns', 'sum_ns', 'min_ns', 'max_ns'))
            _require(all(type(value) is int for value in (start, stop, total, lower, upper))
                and first <= start <= stop <= last and 0 <= lower <= upper
                and (count-1)*lower + upper <= total <= (count-1)*upper + lower and total <= stop-start,
                'observed phase aggregate exceeds occurrence or owner clock bounds')
        _require(all(a['first_ns'] <= b['first_ns'] and a['last_ns'] <= b['last_ns']
            for a,b in zip(parts, parts[1:])), 'phase clock extrema violate the original ordered loop')
        total = sum(part['sum_ns'] for part in parts)
        _require(total <= last-first, 'disjoint owner phase totals cannot exceed the loop envelope')
        measured[axis] = dict(envelope_ns=last-first, selected_disjoint_ns=total,
            unattributed_complement_ns=last-first-total)
    return witness, measured


def _f4_bind(graph, boundary, process_id, thread_id):
    found = []
    for index, row in enumerate(graph['observed_targets']):
        attempt = row['observation']
        if (row['entry_scope_reachable'] and attempt['process_id'] == process_id
                and attempt['thread_id'] == thread_id and attempt['caller_site'] == boundary['caller']
                and attempt['target_code'] == boundary['code']
                and ('callable' not in boundary or
                     json.loads(attempt['metadata_json']) == boundary['callable'])):
            found.append((index, row))
    _require(len(found) == 1, 'timed scope or phase is missing its exact original F4 operation')
    index, row = found[0]
    return dict(observed_call=f'#/observed_targets/{index}', operation=row['operation'],
        result_port=row['observation']['source_call'], boundary=boundary)


def project_source_cost(design, observation, observation_ref, seed, ordered_scope, scope_ref, resources):
    """Project same-run source service evidence onto the source loop's ordered edges."""
    witness, measured = validate_source_segments(design, observation)
    _require('process_id' in observation and read_artifact(observation_ref) == observation,
        'source cost must name exact captured observation bytes')
    frames = read_artifact(seed['provenance']['frames'])
    _require(frames['process_id'] == observation['process_id']
        and frames['thread_id'] == observation['thread_id'] and frames['cpu_ids'] == observation['affinity'],
        'timed owner scope and F4 frames require the same process, thread and CPU mask')
    graph_ref = seed['entry_calls']
    graph = read_artifact(graph_ref)
    boundaries = witness['boundaries']
    order = witness['phase_order']
    observation_site = observation['process_id'], observation['thread_id']
    nodes = {name: {**_f4_bind(graph, boundaries['phases'][name], *observation_site),
        'measurement': f'#/segmented_cost/phases/{name}', 'observed': witness['phases'][name]} for name in order}
    edges = [dict(source=left, target=right, relation='SAME_ITERATION_ORDER')
        for left,right in zip(order, order[1:])]
    edges.append(dict(source=order[-1], target=order[0], relation='NEXT_ITERATION_ORDERED_PUBLICATION'))
    affinity = observation['affinity']
    placement = resources['placement']
    _require(design['count'] == resources['case']['call_count']
        and ordered_scope['loop']['exact_count_under_source_conditions']['normal_scope_completion'] is True,
        'source owner measurement must belong to the admitted loop and fixed count')
    return dict(schema='scar.ordered-source-owner-cost-projection', schema_version=1,
        observation=observation_ref, graph=graph_ref, ordered_scope=scope_ref,
        loop={**_f4_bind(graph, boundaries['scope'], *observation_site), 'count': design['count'],
            'control': '#/loop/original_operation', 'ordered_effect': '#/dependences/ordered_publication'},
        phase_order=order, nodes=nodes, ordered_edges=edges, measured=measured,
        lane=dict(process_id=observation['process_id'], thread_id=observation['thread_id'],
            affinity=affinity, single_allowed_cpu_id=affinity[0] if len(affinity) == 1 else None,
            matches_native_owner_affinity=affinity == [placement['owner_cpu_id']],
            native_core_topology_ref=placement['hardware']),
        cost_policy=dict(source_clock_scope='OBSERVED_INSTRUMENTED_OWNER_SPANS',
            phase_includes_descendants=True, measured_sibling_spans_disjoint=True,
            occurrence_order_checked_by_collector=True, clean_source_time_ns=None,
            unassigned_time_kind='UNKNOWN', native_worker_service_ns=None, native_transfer_service_ns=None,
            native_retirement_wait_ns=None, native_startup_close_ns=None,
            native_duration_bounds_ns=None, source_time_transfer=False, width_selection_basis='LIVE_CAPACITY_ONLY'))

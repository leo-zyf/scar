"""Validate instrumented ready-host cuts; report per-thread observations only."""
import re
import statistics

from scar.codegen.pipeline_host_probe import CALLER_SLOTS, LANE_SLOTS, CALL_SLOTS
from scar.ir.program.providers import ProviderArtifact

SCHEMA = 'scar.pipeline-host-sample-observations'
_POLICY = 'FIXED_CPU_LANE_HEAD_ONLY_WAIT_TRUE_PREDECESSORS_NO_TIMESTAMP_WAIT'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _cuts(value, names):
    _need(type(value) is dict and set(value) == set(names), 'complete original cut names required')
    for row in value.values():
        _need(type(row) is dict and set(row) == {'ns', 'valid'} and row['valid'] is True
              and type(row['ns']) is int and row['ns'] >= 0,
              'every successful source cut needs a valid nonnegative integer timestamp')
    return {key:value[key]['ns'] for key in names}


def _sequence(actor, events):
    _need(all(a[1] <= b[1] for a, b in zip(events, events[1:])),
          'source order differs within one thread: '+actor)
    rows = []
    for (left, begin), (right, end) in zip(events, events[1:]):
        a, b = left.rsplit(':', 1)[-1], right.rsplit(':', 1)[-1]
        kind = 'SAME_THREAD_FRAGMENT_OBSERVATION'
        if (a, b) == ('ABI_BEGIN', 'ABI_END'):
            kind = 'ORIGINAL_LEAF_ABI_OBSERVATION_NOT_HOST_CHARGE'
        elif (a, b) in (('JOIN_BEGIN', 'JOIN_END'), ('PUBLISH_LOCK_BEGIN', 'PUBLISH_LOCK_ACQUIRED')):
            kind = 'WAIT_OBSERVATION'
        elif a == 'PUBLISH_PRE_UNLOCK' or (a, b) == ('CLAIM_END', 'EXECUTE_BEGIN'):
            kind = 'UNLOCK_BRACKET_ORDER_OBSERVATION'
        elif a == 'EMPLACE_BEGIN':
            kind = 'THREAD_SUBMISSION_ELAPSED_NOT_FIXED_CPU_SERVICE'
        elif actor != 'caller' and b == 'CLAIM_BEGIN':
            kind = 'WAIT_AND_UNRESOLVED_DISPATCH_PREDICATE'
        elif actor != 'caller' and b == 'LAMBDA_EXIT':
            kind = 'WAIT_AND_WORKER_EXIT_OBSERVATION'
        elif (a, b) == ('EXECUTE_END', 'PUBLISH_LOCK_BEGIN'):
            kind = 'INSTRUMENTATION_STATUS_BOOKKEEPING_RESIDUAL'
        elif (a, b) in (('POOL_ALLOC_BEGIN', 'POOL_ALLOC_END'), ('POOL_DESTRUCT_BEGIN', 'POOL_DESTRUCT_END')):
            kind = 'WHOLE_POOL_GROUP_ELAPSED_OBSERVATION'
        rows.append(dict(id=actor+'/'+left+'->'+right, actor=actor, begin=left, end=right,
            begin_ns=begin, end_ns=end, duration_ns=end-begin, kind=kind,
            automatically_admitted_clean_cost=False))
    _need(sum(row['duration_ns'] for row in rows) == events[-1][1]-events[0][1],
          'adjacent per-thread cuts must exactly partition that recorded thread span')
    return rows


def validate_host_trace(trace, *, call_predecessors, lane_call_order, expected_cpu_ids):
    """Check a source-admitted fixed lane-head certificate and successful trace.

    The caller qualifies probe/source/certificate bytes. In particular the
    supplied lane order must be the strict lane-head policy, not the list of
    assignments of an executor that may skip an unready queue head.
    """
    _need((trace.get('schema'), trace.get('schema_version'), trace.get('clock'))
          == ('scar.pipeline-host-probe-trace', 1, 'CLOCK_MONOTONIC')
          and trace.get('normal_complete') is True and trace.get('cost_admissible') is True
          and trace.get('target_region_executed') is True
          and trace.get('full_target_program_executed') is False
          and trace.get('instrumentation_cost_subtracted') is False,
          'successful instrumented regional trace with its actual execution scope required')
    cpus = list(expected_cpu_ids)
    _need(cpus and all(type(c) is int and c >= 0 for c in cpus) and len(set(cpus)) == len(cpus)
          and type(lane_call_order) is dict and set(lane_call_order) == {str(c) for c in cpus},
          'explicit unique ordered CPU lanes and exact strict call-order coverage required')
    _need(all(type(order) in (list, tuple) and all(type(name) is str and name for name in order)
              for order in lane_call_order.values()), 'strict lane order must contain original call ID sequences')
    lanes = trace['lanes']; calls = trace['calls']; caller = _cuts(trace['caller'], CALLER_SLOTS)
    _need(type(lanes) is list and [row['cpu_id'] for row in lanes] == cpus,
          'trace lane identities/order differ from the original certificate')
    _need(type(calls) is list and calls and all(type(row['call_id']) is str and row['call_id']
          and type(row['status']) is int and row['status'] == 0 for row in calls),
          'every original call must have a successful recorded status')
    by_call = {row['call_id']:_cuts(row['cuts'], CALL_SLOTS) for row in calls}
    order = [name for cpu in cpus for name in lane_call_order[str(cpu)]]
    _need(len(by_call) == len(calls) == len(order) and set(order) == set(by_call)
          and type(call_predecessors) is dict and set(call_predecessors) == set(by_call),
          'all original calls/predecessors must be retained exactly once')
    by_lane = {row['cpu_id']:_cuts(row['cuts'], LANE_SLOTS) for row in lanes}
    caller_events = [(key, caller[key]) for key in CALLER_SLOTS[:9]]
    for cpu in cpus:
        caller_events.extend((str(cpu)+':'+k, by_lane[cpu][k]) for k in ('EMPLACE_BEGIN', 'EMPLACE_END'))
    caller_events.append(('LAUNCH_LOOP_END', caller['LAUNCH_LOOP_END']))
    for cpu in cpus:
        caller_events.extend((str(cpu)+':'+k, by_lane[cpu][k]) for k in ('JOIN_BEGIN', 'JOIN_END'))
    caller_events.extend((key, caller[key]) for key in CALLER_SLOTS[10:])
    intervals = _sequence('caller', caller_events); launches = []; joins = []
    for cpu in cpus:
        lane = by_lane[cpu]
        _need(lane['EMPLACE_BEGIN'] <= lane['LAMBDA_ENTRY'] <= lane['LAMBDA_EXIT'] <= lane['JOIN_END'],
              'worker entry/exit must follow launch begin and precede its completed join')
        events = [('LAMBDA_ENTRY', lane['LAMBDA_ENTRY'])]
        for name in lane_call_order[str(cpu)]:
            events.extend((name+':'+key, by_call[name][key]) for key in CALL_SLOTS)
        events.append(('LAMBDA_EXIT', lane['LAMBDA_EXIT']))
        intervals.extend(_sequence('worker:'+str(cpu), events))
        launches.append(dict(cpu_id=cpu, launch_delay_ns=lane['LAMBDA_ENTRY']-lane['EMPLACE_BEGIN'],
            kind='ASYNCHRONOUS_START_ORDER_OBSERVATION_NOT_CPU_SERVICE',
            emplace_end_before_worker_entry_required=False))
        joins.append(dict(cpu_id=cpu, elapsed_ns=lane['JOIN_END']-lane['JOIN_BEGIN'],
            after_exit_tail_ns=lane['JOIN_END']-max(lane['JOIN_BEGIN'], lane['LAMBDA_EXIT']),
            kind='JOIN_WAIT_AND_AFTER_EXIT_TAIL_OBSERVATION_NOT_FIXED_CPU_SERVICE'))
    for name, predecessors in call_predecessors.items():
        _need(type(predecessors) in (list, tuple) and len(set(predecessors)) == len(predecessors)
              and set(predecessors) <= set(by_call)-{name}, 'original predecessor domain differs')
        _need(all(by_call[p]['PUBLISH_PRE_UNLOCK'] <= by_call[name]['CLAIM_BEGIN'] for p in predecessors),
              'successor claim precedes the source publication/release lower bracket')
    critical = sorted(span for row in by_call.values() for span in (
        (row['CLAIM_BEGIN'], row['CLAIM_END']), (row['PUBLISH_LOCK_ACQUIRED'], row['PUBLISH_PRE_UNLOCK'])))
    _need(all(a[1] <= b[0] for a, b in zip(critical, critical[1:])),
          'observed mutex-held claim/publication interiors overlap')
    brackets = [dict(call_id=name, release_timestamp_ns=None,
        lower_ns=row['PUBLISH_PRE_UNLOCK'], upper_ns=row['PUBLISH_POST_UNLOCK']) for name, row in by_call.items()]
    return dict(schema='scar.pipeline-host-trace-validation', schema_version=1,
        calls=len(calls), lanes=len(lanes), intervals=intervals, publication_brackets=brackets,
        launch_observations=launches, join_observations=joins,
        instrumented_native_region_elapsed_ns=caller['OUTER_RETURN']-caller['OUTER_ENTRY'],
        per_thread_partitions_exact=True, expected_dispatch_policy=_POLICY,
        publication_check='SOURCE_MUTEX_BRACKET_COMPATIBILITY_NOT_AN_OBSERVED_ATOMIC_RELEASE',
        successor_may_claim_before_predecessor_post_unlock_stamp=True,
        lane_labels_are_not_observed_phase_cpu_locations=True,
        cross_thread_interval_sum_is_whole_latency=False, clean_performance_cost_admitted=False)


def summarize_host_samples(document):
    """Retain scope/residuals while summarizing successful post-warmup traces."""
    _need((document.get('schema'), document.get('schema_version')) == (SCHEMA, 1), 'versioned host observations required')
    probe = ProviderArtifact(**document['probe'])
    _need(probe.schema == 'scar.pipeline-host-probe' and type(document.get('source_refs')) in (list, tuple)
          and document['source_refs'] and all(type(row) is dict for row in document['source_refs']),
          'original probe and source references required')
    repetitions = document['repetitions']; warmup = document['warmup']; samples = document['samples']
    expected = document['expected_output_sha256']
    _need(type(repetitions) is int and 3 <= repetitions <= 10000 and type(warmup) is int and warmup >= 0
          and type(samples) is list and len(samples) == repetitions
          and type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected),
          'all post-warmup samples and exact expected output hash required')
    groups = {}; seen = set(); native_elapsed = []; order_samples = []
    for sample in samples:
        identity = sample['id']
        _need(type(identity) is str and identity and identity not in seen and type(sample['status']) is int
              and sample['status'] == 0 and sample['output_sha256'] == expected,
              'each successful sample needs its unique identity and unchanged expected output')
        seen.add(identity)
        summary = validate_host_trace(sample['trace'], call_predecessors=document['call_predecessors'],
            lane_call_order=document['lane_call_order'], expected_cpu_ids=document['expected_cpu_ids'])
        native_elapsed.append(summary['instrumented_native_region_elapsed_ns'])
        order_samples.append(dict(sample_id=identity, launch=summary['launch_observations'],
            joins=summary['join_observations'], publication_brackets=summary['publication_brackets']))
        keys = {row['id'] for row in summary['intervals']}
        _need(not groups or set(groups) == keys, 'per-thread source intervals changed across samples')
        for row in summary['intervals']:
            record = groups.setdefault(row['id'], {key:row[key] for key in ('id', 'actor', 'begin', 'end', 'kind')})
            record.setdefault('samples_ns', []).append(row['duration_ns'])
    phases = []
    for row in groups.values():
        values = row['samples_ns']
        phases.append(dict(**row, median_ns=statistics.median(values), min_ns=min(values), max_ns=max(values),
            stdev_ns=statistics.stdev(values), automatically_admitted_clean_cost=False))
    return dict(schema='scar.pipeline-host-sample-summary', schema_version=1,
        probe=document['probe'], source_refs=document['source_refs'], samples=repetitions, warmup=warmup,
        phases=phases, instrumented_native_region_elapsed_samples_ns=native_elapsed,
        order_wait_observations=order_samples, worker_lane_labels_are_not_observed_phase_cpu_locations=True,
        pool_allocation_group_count=1, pool_release_group_count=1, per_slot_costs_added=False,
        conditions=list(document.get('conditions', ()))+['PER_THREAD_EXCLUSIVE_INTERVALS_NOT_A_SERIAL_GLOBAL_SUM',
            'ORIGINAL_ABI_SERVICE_IS_NOT_CHARGED_AGAIN_AS_HOST_OVERHEAD',
            'ACTUAL_INSTRUMENTED_REGION_NOT_CANONICAL_NOOP_OR_FULL_PROGRAM'],
        residuals=list(document.get('residuals', ()))+['CLOCK_AND_TRACE_STORES_AND_INSTRUMENTATION_COMPILER_PERTURBATION',
            'INPUT_CACHE_ALLOCATOR_AND_CONCURRENT_SCHEDULING_REGIME_NOT_EXTRAPOLATED'],
        target_region_executed=True, full_target_program_executed=False,
        physical_scope_cost_ns=None, cross_thread_intervals_summed=False,
        clean_performance_cost_admitted=False, execution_authorized=False, deployment_authorized=False)


__all__ = ['SCHEMA', 'validate_host_trace', 'summarize_host_samples']

"""Check a native diagnostic run against a replayed resource-event certificate."""
from scar.optimization.pipeline_execution_refinement import RESOURCE_EVENTS_FAMILY


_TERMINAL = {
    'ALLOCATE': {'FINAL_STORAGE_ADOPTED', 'WORKSPACE_ALLOCATED'},
    'COMPUTE': {'COMPUTE_SUCCESS'},
    'SUCCESS_FENCE': {'SUCCESS_FENCE'},
    'RELEASE': {'WORKSPACE_RELEASED'},
    'EXPORT': {'FINAL_PUBLISHED'},
}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def verify_resource_event_observations(certificate, observations):
    """Verify order, actual CPU, storage retirement and publication on success.

    The caller supplies the certificate replayed by the compiled artifact
    loader. Durations are diagnostic and play no role in this check.
    """
    _need(certificate['execution_family'] == RESOURCE_EVENTS_FAMILY, 'resource-event certificate required')
    events = {row['id']: row for row in certificate['resource_events']}
    publication = certificate['publication_task']
    completed, started, value_ready, acquired, retired = {}, {}, {}, {}, {}
    lanes = {cpu: [] for cpu in certificate['cpu_ids']}
    for sequence, row in enumerate(observations):
        kind, event = row['kind'], row['event_id']
        _need(row['sequence'] == sequence and event in events and row['status'] == 0,
              'invalid or failed resource-event observation')
        plan = events[event]
        if kind in _TERMINAL[plan['kind']]:
            _need(event not in completed and all(dep in completed for dep in plan['depends_on']),
                  'resource event completed out of dependency order')
            if plan['kind'] == 'COMPUTE':
                _need(event in started and row['cpu_id'] == plan['cpu_id'], 'compute completion differs from CPU binding')
            elif plan['kind'] in ('ALLOCATE', 'RELEASE'):
                _need(row['slot_id'] == plan['slot_id'] and
                      (plan['slot_id'] == 'final:output') == (kind == 'FINAL_STORAGE_ADOPTED'),
                      'physical storage event differs from its certified slot')
            completed[event] = sequence
        elif kind == 'COMPUTE_START':
            cpu = plan.get('cpu_id')
            _need(event not in started and cpu in lanes and cpu == row['cpu_id']
                  and row['call_id'] == plan['call_id'] and all(dep in completed for dep in plan['depends_on']),
                  'compute started outside certified CPU or predecessors')
            started[event] = sequence
            lanes[cpu].append(plan['call_id'])
        else:
            table = {'VALUE_READY': value_ready, 'SLOT_ACQUIRED': acquired, 'SLOT_RETIRED': retired}.get(kind)
            _need(table is not None and plan['kind'] == 'COMPUTE' and row['value_id'] not in table,
                  'unsupported or duplicate slot-version observation')
            table[row['value_id']] = row

    _need(completed.keys() == events.keys() and observations[-1]['event_id'] == publication
          and observations[-1]['kind'] == 'FINAL_PUBLISHED',
          'incomplete resource events or missing final publication')
    _need({str(cpu): calls for cpu, calls in lanes.items()} == certificate['lane_call_order'],
          'actual CPU lane order differs from the certificate')
    expected_owned, expected_workspace = set(), set()
    for slot in certificate['slot_lifetimes']:
        if slot['allocation'] is None:
            continue
        previous_retirement = completed[slot['allocation']]
        for version in slot['versions']:
            value, producer = version['logical_value'], version['producer']
            expected_owned.add(value)
            _need(value in value_ready, 'missing native value readiness')
            publish = value_ready[value]
            consumer_entries = [started[name] if name in started else completed[name]
                                for name in version['consumers']]
            _need(publish['event_id'] == producer and slot['slot_id'] == publish['slot_id']
                  and completed[producer] < publish['sequence'] < min([completed[publication], *consumer_entries]),
                  'native value made ready out of causal order')
            if slot['release'] is None:
                _need(previous_retirement < started[producer] and observations[-1]['slot_id'] == slot['slot_id']
                      and observations[-1]['value_id'] == value, 'published final value or storage differs')
                continue
            expected_workspace.add(value)
            _need(value in acquired and value in retired and retired[value]['slot_id'] == slot['slot_id'],
                  'missing native workspace slot acquisition or retirement')
            reserve = acquired[value]
            _need(reserve['event_id'] == producer and reserve['slot_id'] == slot['slot_id'] and
                  previous_retirement < reserve['sequence'] < started[producer],
                  'native workspace slot acquired before its prior version retired')
            previous_retirement = retired[value]['sequence']
            last_use = max((completed[name] for name in version['consumers']), default=publish['sequence'])
            _need(max(publish['sequence'], last_use) <= previous_retirement < completed[slot['release']],
                  'native slot retired before its last reader or after physical release')
    _need(expected_owned == value_ready.keys() and expected_workspace == acquired.keys() == retired.keys(),
          'unexpected slot-version observations')
    return dict(event_count=len(events), observation_count=len(observations), lane_call_order=lanes,
                owned_slot_versions=len(expected_owned), retired_slot_versions=len(expected_workspace),
                publication_task=publication)

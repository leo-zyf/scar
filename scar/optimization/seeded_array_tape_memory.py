"""Prove branch-dependent returned-value routing in a private ndarray tape."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from scar.analysis import ndarray_tape_projection as projection
from scar.analysis.ndarray_tape_schedule import SCHEMA as SCHEDULE, TAPE, lower_tape
from scar.codegen import native_numpy_vector_normalize as normalize
from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.seeded-array-tape-memory'
_ITEM = 16


def _need(valid, reason):
    if not valid:
        raise ValueError('seeded ndarray tape memory: ' + reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _access_index(schedule):
    accesses = defaultdict(list)
    for index, row in enumerate(schedule['operations']):
        for root in set(row['accesses']['reads']) | set(row['accesses']['writes']):
            accesses[root].append(index)
    layout = schedule['storage']
    _need(set(accesses) == set(layout['lifetimes']) == set(layout['offset_bytes']) and
          all(list(layout['lifetimes'][root]) == [uses[0], uses[-1]] for root, uses in accesses.items()),
          'arena liveness changed from its ordered read/write schedule')
    return accesses


def _root(tape, schedule, view, extent):
    identity = view['storage']
    root = tape['storages'][identity]
    start = schedule['storage']['offset_bytes'][identity]
    shape, strides = view['shape'], view['strides_bytes']
    _need(view['dtype'] == '<c16' and view['offset_bytes'] == 0 and len(shape) == len(strides) and
          type(start) is int and 0 <= start <= schedule['storage']['arena_bytes'] - extent and
          type(root['address_mod_64']) is int and root['address_mod_64'] in range(0, 64, 8) and
          view['address_mod_64'] == root['address_mod_64'] == start % 64 and root['size_bytes'] == extent,
          'routing needs an exact whole root in its checked private arena placement')
    product = _ITEM
    for n, stride in zip(reversed(shape), reversed(strides), strict=True):
        _need(type(n) is int and n > 0 and stride == product, 'routing needs a dense C-order root or view')
        product *= n
    _need(product == extent, 'view does not cover all bytes of the selected private root')
    return start


def _disjoint(left, right, extent):
    return abs(left - right) >= extent


def _signature(operation, reads, writes):
    return tuple(operation['accesses']['reads']) == reads and tuple(operation['accesses']['writes']) == writes


def _normalization_source(tape, normalization):
    if normalization is None:
        return None
    _need(normalization['schema'] == normalize.SCHEMA and normalization['schema_version'] == 1 and
          normalization['abi'] == normalize._ABI and normalization['whole_entry'] is False,
          'normalizer must preserve its exact private selection and floating-point ABI')
    witness = read_artifact(normalization['witness'])
    provider = normalization['provider']['numpy']
    observation = read_artifact(tape['observation'])
    endpoint = observation['endpoints']['result']
    _need(endpoint['source'] == witness['source'] and endpoint['code_id'] == witness['code_id'] and
          all(provider[key] == tape['numpy_provider'][key] for key in ('uri', 'sha256', 'size_bytes')) and
          witness['numpy_extension'] == provider,
          'normalization provider is not the original result-selection operation')
    chosen_ordinals = {event['original_ordinal'] for event in tape['events'] if event['kind'] == 'NORMALIZE'}
    original = [event for row in observation['rounds'] for event in row['events']
                if event['kind'] == 'NORMALIZE' and event['ordinal'] in chosen_ordinals]
    _need(len(original) == len(chosen_ordinals) and all(
        event['source']['code_id'] == endpoint['code_id'] and
        event['strategy'] == 'ORIGINAL_NORM_DIVIDE_AND_VDOT_COMPARISON' for event in original),
        'a compiled selection has no original normalization transition')
    return dict(source=endpoint['source'], code_id=endpoint['code_id'], numpy=provider,
                normalizer_witness=normalization['witness'], provider_binary=normalization['binary'],
                abi=normalization['abi']['symbol'])


def _carry(tape, schedule, source_round, lowered_round, chosen, extent):
    events, ops = tape['events'], schedule['operations']
    first = lowered_round[0]
    operation = ops[first]
    _need(operation['kind'] == 'COPY' and len(operation['groups']) == 1,
          'next state must first flow into one fresh whole-root copy')
    event_index = operation['event_index']
    _need(source_round['begin'] <= event_index < source_round['end'], 'copy is outside its original round')
    for event in events[source_round['begin']:event_index]:
        _need(event['kind'] == 'VIEW' and event['input'] == chosen and event['output']['storage'] == chosen['storage'],
              'next state has an unrelated use or escapes before its copy')
        _root(tape, schedule, event['output'], extent)
        chosen = event['output']
    event = events[event_index]
    target = event['output']
    _need(event['kind'] == 'COPY' and event['input'] == chosen and
          operation['source'] == chosen['storage'] and operation['target'] == target['storage'] and
          _signature(operation, (chosen['storage'],), (target['storage'],)) and
          operation['groups'][0] == dict(read=0, write=0, count=extent // _ITEM, increments=[_ITEM, _ITEM]),
          'compiled copy does not represent the same complete logical returned-value edge')
    return first, target['storage'], _root(tape, schedule, target, extent)


def derive(tape, schedule, *, normalization=None):
    """Derive legal routes from compiler-checked tape/schedule dictionaries.

    Consumers with stored static offsets can use uniform canonical repair. A
    pointer loan is optional where the alternate root stays disjoint at the
    only following copy. CLI project/check independently replay the inputs.
    """
    _need(tape['schema'] == TAPE and tape['schema_version'] == 1 and not tape['frontiers'] and
          not tape['deploy'] and not tape['captures_evolving_state_payloads'] and
          schedule['schema'] == SCHEDULE and schedule['schema_version'] == 1 and
          schedule['shadow_state_is_private'] is True and schedule['preserved_source_effect_order'] is True and
          type(schedule['storage']['arena_bytes']) is int and schedule['storage']['arena_bytes'] > 0 and
          schedule['storage']['arena_bytes'] % 64 == 0, 'requires an exact bounded private, ordered arena')
    source = _normalization_source(tape, normalization)
    operations, events = schedule['operations'], tape['events']
    ranges, original = schedule['rounds'], tape['rounds']
    _need(len(ranges) == len(original) == len(schedule['checkpoints']) and ranges,
          'one original return binding per normal round is required')
    accesses = _access_index(schedule)
    rows = []
    end, event_end = 0, 0
    for index, (source_round, bounds) in enumerate(zip(original, ranges, strict=True)):
        start, stop = bounds
        begin, finish = source_round['begin'], source_round['end']
        _need(start == end and stop >= start + 2 and stop <= len(operations) and
              begin == event_end and finish >= begin + 2 and finish <= len(events) and source_round['index'] == index,
              'original normal round ranges are not contiguous')
        end, event_end = stop, finish
        normal_index, checkpoint_index = stop - 2, stop - 1
        normal, checkpoint = operations[normal_index:stop]
        event, original_checkpoint = events[finish - 2:finish]
        _need(normal['kind'] == event['kind'] == 'NORMALIZE' and checkpoint['kind'] == original_checkpoint['kind'] == 'CHECKPOINT'
              and normal['event_index'] == finish - 2 and checkpoint['event_index'] == finish - 1 and
              type(normal['selected']) is bool and type(normal['elements']) is int and normal['elements'] > 0 and
              normal['selected'] is event['selected'] and normal['source'] == event['input'] and normal['target'] == event['normalized'] and
              checkpoint['checkpoint'] == index and checkpoint['source'] == original_checkpoint['input'],
              'selected return and its checkpoint are not the exact original ordered operations')
        raw, scratch = normal['source'], normal['target']
        raw_id, scratch_id = raw['storage'], scratch['storage']
        extent = normal['elements'] * _ITEM
        _need(raw_id != scratch_id and raw['shape'] == scratch['shape'] == [normal['elements']] and
              _signature(normal, (raw_id,), (scratch_id,)), 'normalizer raw and scratch must bind distinct flat versions')
        raw_offset, scratch_offset = (_root(tape, schedule, view, extent) for view in (raw, scratch))
        _need(_disjoint(raw_offset, scratch_offset, extent), 'normalization roots overlap at selection')
        observed = int(normal['selected'])
        chosen, other = (scratch, raw) if observed else (raw, scratch)
        canonical, alternate = (scratch_offset, raw_offset) if observed else (raw_offset, scratch_offset)
        chosen_id, other_id = chosen['storage'], other['storage']
        _need(checkpoint['source'] == chosen and checkpoint['bytes'] == extent and
              _signature(checkpoint, (chosen_id,), ()) and
              schedule['checkpoints'][index] == dict(sha256=original_checkpoint['sha256'], size_bytes=extent),
              'original return checkpoint differs from its selected root and has another effect')
        consumers = [checkpoint_index]
        carry_op = carry_id = carry_offset = None
        overlap = 'NO_FOLLOWUP_COPY'
        if index + 1 < len(ranges):
            next_round = ranges[index + 1]
            _need(next_round[0] == stop, 'operations intervene in the selected-root loan')
            carry_op, carry_id, carry_offset = _carry(tape, schedule, original[index + 1], next_round, chosen, extent)
            _need(carry_id not in (raw_id, scratch_id) and accesses[carry_id][0] == carry_op and
                  _disjoint(canonical, carry_offset, extent), 'original fresh carry motion conflicts with a live version')
            consumers.append(carry_op)
            overlap = ('DISJOINT' if _disjoint(alternate, carry_offset, extent) else
                       'EXACT_SAME_SPAN' if alternate == carry_offset else 'PARTIAL_OVERLAP')
        _need([use for use in accesses[chosen_id] if use > normal_index] == consumers and
              not any(use > normal_index for use in accesses[other_id]) and accesses[scratch_id][0] == normal_index,
              'raw/scratch has another live consumer or an earlier scratch value')
        rebind = overlap in ('DISJOINT', 'NO_FOLLOWUP_COPY')
        rows.append(dict(round=index, normalize_operation=normal_index, normalize_event=finish - 2,
                         checkpoint_operation=checkpoint_index, carry_operation=carry_op, bytes=extent,
                         expected_status=observed, alternate_status=1 - observed,
                         raw_root=raw_id, scratch_root=scratch_id, canonical_root=chosen_id, alternate_root=other_id,
                         raw_offset=raw_offset, scratch_offset=scratch_offset, canonical_offset=canonical,
                         carry_root=carry_id, carry_offset=carry_offset, alternate_mode='REBIND' if rebind else 'REPAIR_CANONICAL',
                         alternate_followup_overlap=overlap, exact_copy_transfer_available=overlap == 'EXACT_SAME_SPAN',
                         repair_canonical_legal=True, canonical_repair_bytes=extent, repair_bytes=0 if rebind else extent,
                         selected_consumers=consumers, alternate_original_last_use=accesses[other_id][-1]))
    _need(end == len(operations) and event_end == len(events) and
          sum(row['kind'] == 'NORMALIZE' for row in operations) == sum(event['kind'] == 'NORMALIZE' for event in events) == len(rows)
          and sum(row['kind'] == 'CHECKPOINT' for row in operations) == len(rows),
          'operations or unmodeled selections escape the certified rounds')
    copy_bound = sum(row['canonical_repair_bytes'] for row in rows)
    return dict(schema=SCHEMA, schema_version=1, rows=rows, normalization_source=source,
                arena_bytes=schedule['storage']['arena_bytes'], extra_arena_bytes=0,
                normal_statuses=[0, 1], decline_statuses=[-1, 2],
                decline_policy='NO_SELECTED_STATE_REPAIR_OR_PUBLICATION_ON_PROVIDER_DECLINE',
                uniform_repair_copy_bound_bytes=copy_bound, uniform_repair_traffic_bound_bytes=2 * copy_bound,
                scope='EXACT_PRIVATE_NORMALIZATION_RESULT_HANDOFF', deployment_authorized=False)


def _checked(tape_ref, schedule_ref, provider_ref):
    tape = projection.check(read_artifact(tape_ref))
    schedule = read_artifact(schedule_ref)
    _need(schedule == json.loads(json.dumps(lower_tape(tape))), 'stored arena differs from its replayed tape')
    normal = normalize.check(provider_ref['uri'])
    _need(normal == read_artifact(provider_ref), 'normalizer changed during cold replay')
    return dict(derive(tape, schedule, normalization=normal), tape=tape_ref, schedule=schedule_ref, normalization=provider_ref)


def project(tape_path, schedule_path, normalization_path):
    """Rebuild source binding, storage layout and normalizer before proving routes."""
    return _checked(_ref(tape_path, TAPE), _ref(schedule_path, SCHEDULE), _ref(normalization_path, normalize.SCHEMA))


def check(path):
    stored = read_artifact(_ref(path, SCHEMA))
    _need(stored == _checked(stored['tape'], stored['schedule'], stored['normalization']),
          'stored route decision differs from original source and arena replay')
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('tape', 'schedule', 'normalization', 'out', 'check'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    if args.check and not any((args.tape, args.schedule, args.normalization, args.out)):
        result = check(args.check)
    elif not args.check and all((args.tape, args.schedule, args.normalization, args.out)):
        result = project(args.tape, args.schedule, args.normalization)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --check RECEIPT or --tape TAPE --schedule SCHEDULE --normalization PROVIDER --out REPORT')
    print(json.dumps(dict(arena_bytes=result['arena_bytes'], routes=len(result['rows']),
                          uniform_repair_bound_bytes=result['uniform_repair_copy_bound_bytes'])))


if __name__ == '__main__':
    main()

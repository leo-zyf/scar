"""Prove storage ownership on the seed-driven array loop backedge."""

import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import seeded_array_tape_memory as finite

SCHEMA = 'scar.seeded-array-loop-memory'
_EVENT_VARIANTS = {'original_ordinal', 'address_mod_64', 'sha256'}
_OP_VARIANTS = {'event_index', 'checkpoint', 'address_mod_64'}
_ROOT_KEYS = {'storage', 'root', 'source', 'target', 'reads', 'writes'}


def _need(condition, detail):
    if not condition:
        raise ValueError('seeded array loop memory: ' + detail)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _template_rows(events, operations, event_start):
    """Use one consistent alpha-renaming across source events and lowered roots."""
    names = {}

    def rename(row, omitted, key=None):
        if type(row) is dict:
            return {name: rename(value, omitted, name) for name, value in sorted(row.items()) if name not in omitted}
        if isinstance(row, (list, tuple)):
            return [rename(value, omitted, key) for value in row]
        if type(row) is str and key in _ROOT_KEYS:
            return names.setdefault(row, len(names))
        return row

    source = [rename(event, _EVENT_VARIANTS) for event in events]
    lowered = [dict(rename(row, _OP_VARIANTS), relative_event=row['event_index'] - event_start)
               for row in operations]
    return source, lowered


def _recurrence(tape, schedule, memory):
    """Check that the two observed steady bodies differ only in instance identities."""
    def template(index):
        bounds = tape['rounds'][index]
        begin, end = schedule['rounds'][index]
        return _template_rows(tape['events'][bounds['begin']:bounds['end']],
                              schedule['operations'][begin:end], bounds['begin'])

    steady = template(1)
    _need(all(template(index) == steady for index in range(2, len(memory['rows']))),
          'later original rounds differ in control, logical geometry, coefficients or operator order')


def _live_before(lifetimes, boundary):
    return {root for root, (first, last) in lifetimes.items() if first < boundary <= last}


def _choices(row, layout):
    return [dict(status=status, root=row[name + '_root'], offset=row[name + '_offset'], bytes=row['bytes'],
                 address_mod_64=layout['offset_bytes'][row[name + '_root']] % 64)
            for status, name in ((0, 'raw'), (1, 'scratch'))]


def _phase(index, schedule, memory, phase, incoming):
    row = memory['rows'][index]
    begin, end = schedule['rounds'][index]
    layout = schedule['storage']
    lifetimes = layout['lifetimes']
    exit_root = row['canonical_root']
    _need(_live_before(lifetimes, begin) == incoming and _live_before(lifetimes, end) == {exit_root},
          'there is an independent live value across the periodic phase boundary')
    _need(end == row['checkpoint_operation'] + 1 and row['checkpoint_operation'] == row['normalize_operation'] + 1,
          'an operation intervenes between selection, checkpoint and handoff')
    choices = _choices(row, layout)
    _need(all(choice['offset'] == layout['offset_bytes'][choice['root']] and
              0 <= choice['offset'] <= layout['arena_bytes'] - row['bytes'] and
              choice['address_mod_64'] in range(0, 64, 8) and
              lifetimes[choice['root']][0] >= begin for choice in choices),
          'a selected result is not private to the producing phase or leaves its arena')
    return dict(kind=phase, round=index, operations=[begin, end], normalize_operation=row['normalize_operation'],
                checkpoint_operation=row['checkpoint_operation'], vector_bytes=row['bytes'], choices=choices,
                witnessed_live_in=sorted(incoming), witnessed_live_out=[exit_root],
                no_independent_consumer_after_selection=True)


def _handoff(schedule, first, steady, memory):
    layout = schedule['storage']
    operations = schedule['operations']
    former = memory['rows'][0]
    start, _ = steady['operations']
    step = operations[start]
    src, dst = former['canonical_offset'], former['carry_offset']
    target, size = former['carry_root'], first['vector_bytes']
    _need(size == steady['vector_bytes'] and 0 < size <= layout['arena_bytes'] and
          former['carry_operation'] == start and step['kind'] == 'COPY' and
          tuple(step['accesses']['reads']) == (former['canonical_root'],) and
          tuple(step['accesses']['writes']) == (target,) and layout['lifetimes'][target][0] == start and
          layout['lifetimes'][former['canonical_root']][-1] == start and
          0 <= dst <= layout['arena_bytes'] - size and layout['offset_bytes'][target] == dst and
          step['groups'] == [dict(count=size // finite._ITEM, read=0, write=0, increments=[finite._ITEM, finite._ITEM])],
          'steady entry does not immediately consume the exact complete prior return')
    routes = []
    for phase in (first, steady):
        for choice in phase['choices']:
            source = choice['offset']
            same = source == dst
            _need(same or finite._disjoint(source, dst, size), 'partial overlap needs another scoped transition')
            routes.append(dict(predecessor=phase['kind'], status=choice['status'], source_root=choice['root'],
                               source_offset=source, destination_offset=dst,
                               mode='TRANSFER_SAME_SPAN' if same else 'MEMCPY', bytes=size,
                               physical_copy_bytes=0 if same else size, independent_live_versions=[]))
    return dict(operation=start, static_source_root=former['canonical_root'], static_source_offset=src,
                destination_root=target, destination_offset=dst, bytes=size, routes=routes,
                before_any_other_steady_operation=True,
                requires_selected_pointer_for_checkpoint_and_terminal=True)


def derive(tape, schedule, memory):
    """Derive pointer-based carry from caller-checked tape, schedule and memory dictionaries."""
    original = finite.derive(tape, schedule)
    original['normalization_source'] = memory['normalization_source']
    _need(all(memory.get(key) == value for key, value in original.items()) and
          memory['normalization_source'] is not None and len(memory['rows']) >= 3,
          'requires the checked exact normalizer and at least two original steady iterations')
    _recurrence(tape, schedule, memory)
    first = _phase(0, schedule, memory, 'FIRST', set())
    steady = _phase(1, schedule, memory, 'STEADY', {memory['rows'][0]['canonical_root']})
    carry = _handoff(schedule, first, steady, memory)
    layout = schedule['storage']
    return dict(schema=SCHEMA, schema_version=1, first=first, steady=steady, carry_into_steady=carry,
                arena_bytes=layout['arena_bytes'], extra_arena_bytes=0, persistent_carry_bytes=0,
                canonical_repair_bytes_per_round=0, maximum_carry_copy_bytes_per_transition=carry['bytes'],
                maximum_carry_traffic_bytes_per_transition=2 * carry['bytes'],
                all_normalizer_statuses=[0, 1], decline_statuses=[-1, 2],
                declined_state_is_published=False, repeated_template_witnesses=len(memory['rows']) - 1,
                original_round_control_is_caller_obligation=True,
                runtime_order=['RUN_FIRST_ONCE', 'CHECKPOINT_SELECTED_BYTES', 'COPY_OR_TRANSFER_TO_STEADY',
                               'RUN_STEADY_WHEN_REQUIRED', 'FINISH_FROM_LAST_SELECTED_BYTES'],
                scope='BOUNDED_PRIVATE_FIRST_STEADY_STORAGE_HANDOFF', deployment_authorized=False)


def project(memory_path):
    """Cold-replay the prior source/arena proof before admitting a periodic placement."""
    reference = _ref(memory_path, finite.SCHEMA)
    memory = finite.check(reference['uri'])
    _need(memory == read_artifact(reference), 'selected original memory proof changed during replay')
    return dict(derive(read_artifact(memory['tape']), read_artifact(memory['schedule']), memory), memory=reference)


def check(path):
    recorded = read_artifact(_ref(path, SCHEMA))
    _need(recorded == project(recorded['memory']['uri']), 'periodic placement differs on source/arena replay')
    return recorded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('memory', 'out', 'check'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    if args.check and not (args.memory or args.out):
        result = check(args.check)
    elif args.memory and args.out and not args.check:
        result = project(args.memory)
        with args.out.open('x', encoding='utf-8') as out:
            json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
            out.write('\n')
    else:
        parser.error('provide --check RECEIPT or --memory FINITE_MEMORY_PROOF --out RECEIPT')
    print(json.dumps(dict(arena_bytes=result['arena_bytes'], extra_arena_bytes=result['extra_arena_bytes'],
                          routes=[row['mode'] for row in result['carry_into_steady']['routes']])))


if __name__ == '__main__':
    main()

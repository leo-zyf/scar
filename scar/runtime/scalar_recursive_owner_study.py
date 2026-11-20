"""Keep the admitted scalar driver world live between original generators."""

import argparse
import ctypes as ffi
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from threading import get_ident

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime import scalar_recursive_driver_study as bridge


SCHEMA = 'scar.scalar-recursive-owner-source-study'
PLAN = 'scar.scalar-recursive-owner-plan'
_RECEIVER_ERRORS = (ValueError, TypeError, KeyError, AttributeError, OverflowError)


class _Input:
    def __init__(self, stream, expected):
        self.original = stream
        self.remaining = iter(expected)

    def __getattr__(self, name):
        return getattr(self.original, name)

    def readline(self, *args):
        line = self.original.readline(*args)
        expected = next(self.remaining, '')
        bridge._require(line == expected, 'source input left the isolated original replay')
        return line


@dataclass
class _World:
    handle: ffi.c_void_p
    owner: object
    worker: int
    codec: bridge._Codec
    holder: bridge._Moves
    moves: dict
    scores: dict


class _MovePort(bridge._Moves):
    def get(self, position, default=None):
        self.offer.count('idle_lookups', int(self.offer.resident is not None))
        return super().get(position, default)


class _Owner(bridge._Offer):
    def __init__(self, driver, ui, recursive, sorted_provider, sorted_plan, guard, native, binary, revised, *,
                 observe, pairs):
        super().__init__(driver, ui, recursive, sorted_provider, sorted_plan, guard, native, binary, revised,
                         observe=observe)
        self.resident = None
        self.pairs = pairs
        self.accepting = True
        self.rebinds = 0
        self._dso = ffi.CDLL(str(binary))
        self.rebind = self._dso.scar_recursive_driver_rebind_v2
        self.rebind.argtypes = (ffi.c_void_p, ffi.c_char_p, ffi.c_size_t)
        self.rebind.restype = ffi.c_int

    def _seed(self, frame, owner, history, resident):
        index = self.completed
        bridge._require(self.accepting and frame.f_code is self.revised and not self.live
                        and index < len(self.driver['sessions']), 'no original owner continuation remains')
        if resident is None:
            bridge._require(index == 0, 'a continuation cannot start a new native world')
            return self._guard(frame, owner, history)

        expected = self.driver['sessions'][index]
        prior = self.driver['sessions'][index - 1]
        storage = vars(owner)
        namespace = frame.f_globals
        bridge._require((index - 1, index) in self.pairs and owner is resident.owner and get_ident() == resident.worker
                        and not bridge._observer_active() and self._ui_mode(frame, owner, history, expected)
                        and storage.get(self.move_map) is resident.holder and type(storage.get(self.score_map)) is dict
                        and storage[self.score_map] is not resident.scores and not storage[self.score_map],
                        'original owner or its ordered move and fresh score bindings changed')
        bridge._require(type(history) is list and len(history) == expected['history_input_size']
                        and [bridge._pod(value) for value in history] == expected['history_input']
                        and storage[self.root] is history[-1]
                        and type(storage[self.history]) is set and storage[self.counter] == 0
                        and prior['final']['ordered_map_digests'][self.move_map] == expected['initial_move_digest'],
                        'next generator does not bind the observed history and carried MOVE version')

        codec = resident.codec
        bridge._require(namespace is codec.namespace and self.recursion_owner['qualname'].rsplit('.', 1)[-1] not in storage,
                        'source module or instance dispatch changed during ownership')
        config = [codec._scalar(namespace[name], 'int') for name in self.config]
        label = expected['table_version']
        table = namespace[self.driver['initialization']['table']['symbol']]
        bridge._require(dict(zip(self.config, config, strict=True)) == expected['native_configuration']
                        and bridge._namespace_exact(table)
                        and all(type(row) is tuple and all(bridge._int64(item) for item in row) for row in table.values())
                        and table == self.expected[label], 'the next source configuration or scoring version changed')
        seed_state = dict(storage)
        seed_state[self.move_map] = {}
        seed = (b'SCAR_RECURSIVE_INPUT_V1\n' + label.encode('ascii') + b'\n'
                + codec.position_bytes(storage[self.root]) + bridge._encoded((0, 0, True)).encode('ascii')
                + codec.pack_world(seed_state, self.states) + bridge._encoded(config).encode('ascii'))
        return namespace, codec, storage, seed, self.table_slots.index(label)

    def __call__(self, owner, history):
        frame = sys._getframe(1)
        resident = self.resident
        started = self.tick()
        try:
            namespace, codec, storage, seed, slot = self._seed(frame, owner, history, resident)
            soft = bridge._clock_value(storage[self.soft])
            deadline = bridge._clock_value(storage[self.deadline])
            stop = namespace[self.exception]
        except _RECEIVER_ERRORS as error:
            self._fallback(str(error))
            return None
        if resident is None:
            handle = ffi.c_void_p()
            status = self.library.create(seed, len(seed), ffi.byref(handle))
            if status:
                self._fallback(f'DRIVER_CREATE_STATUS_{status}')
                return None
            try:
                holder = _MovePort(self, handle, codec)
                world = _World(handle, owner, get_ident(), codec, holder,
                               storage[self.move_map], storage[self.score_map])
            except BaseException:
                self.library.close(handle)
                raise
            self.count('native_worlds_created')
        else:
            world = resident
            status = self.rebind(world.handle, seed, len(seed))
            if status:
                self._fallback(f'DRIVER_REBIND_STATUS_{status}')
                return None
            # The original prelude already stored a fresh score dictionary.
            world.scores = storage[self.score_map]
            self.resident = None
            self.rebinds += 1
            self.count('native_rebinds')
        self.count('native_sessions')
        self.elapsed('guard_and_begin', started)
        self.live = 1
        return self._run_session(world, storage, stop, slot, soft, deadline)

    def _fallback(self, reason):
        self.accepting = False
        self.count('fallback')
        if self.observe:
            self.frontiers[reason] += 1
        self.close()

    def _record_finish(self, world, trace):
        values = self._snapshot(world.handle, world.codec)
        trace.update(final_nodes=values[self.counter], final_move_entries=len(values[self.move_map]),
                     final_score_entries=len(values[self.score_map]), ordered_map_digests={
                         field: bridge.ordered_map_digest(values[field]) for field in (self.move_map, self.score_map)})

    def _run_session(self, world, storage, stop, slot, soft, deadline):
        keep = False
        valid = False
        trace = None
        failure = []
        advances = 0
        try:
            row = self.library.Row()
            expected = self.driver['sessions'][self.completed]
            if self.observe:
                trace = dict(table_version=self.table_slots[slot], yields=[], outcome='consumer_close',
                             initial_move_entries=expected['initial']['move_table']['size'],
                             initial_score_entries=len(world.scores))
                self.sessions.append(trace)

            def live_deadline(context):
                try:
                    self.count('deadline_reads')
                    return bridge._clock_value(storage[self.deadline])
                except _RECEIVER_ERRORS as error:
                    failure.append(error)
                    return 0.0

            reader = bridge._READER(live_deadline)
            bridge._require(self.library.deadline_reader(world.handle, reader, None) == 0,
                            'native provider declined the original deadline')
            storage[self.move_map] = world.holder
            while True:
                valid = False
                started = self.tick()
                status = self.library.advance(world.handle, soft, deadline, ffi.byref(row))
                self.elapsed('native_advance', started)
                self.count('native_advances')
                bridge._require(not failure, 'the original deadline departed from the admitted domain')
                if status == bridge._DONE:
                    valid = True
                    if trace is not None:
                        trace['outcome'] = 'source_exhausted'
                    break
                if status == bridge._STOP:
                    valid = True
                    if trace is not None:
                        trace['outcome'] = 'source_stop'
                    raise stop
                bridge._require(status == 0 and row.table_slot == slot and row.has_move in (0, 1),
                                'native driver left its admitted continuation')
                advances += 1
                move = world.codec.move_record(row.move) if row.has_move else None
                storage[self.counter] = row.nodes
                self.count('native_yields')
                if trace is not None:
                    trace['yields'].append(dict(value=[row.depth, row.threshold, row.score,
                                                       list(move) if move is not None else None], nodes=row.nodes))
                valid = True
                yield row.depth, row.threshold, row.score, move
                soft = bridge._clock_value(storage[self.soft])
                deadline = bridge._clock_value(storage[self.deadline])
        except GeneratorExit:
            if valid:
                bridge._require(storage.get(self.move_map) is world.holder
                                and storage.get(self.score_map) is world.scores,
                                'the live source lost its owned state ports')
                null_reader = bridge._READER()
                bridge._require(self.library.deadline_reader(world.handle, null_reader, None) == 0,
                                'the native owner did not release the source deadline callback')
                if trace is not None:
                    self._record_finish(world, trace)
                last = expected['yields'][-1]
                complete = (advances == len(expected['yields']) and row.nodes == last['nodes']
                            and [row.depth, row.threshold, row.score] == last['value'][:3])
                if complete:
                    keep = (self.completed, self.completed + 1) in self.pairs
                    self.completed += 1
                else:
                    self.accepting = False
                self.count('native_closed')
                if keep:
                    self.resident = world
            raise
        finally:
            self.live = 0
            if not keep:
                self._release(world, publish=valid)

    def _release(self, world, *, publish):
        storage = vars(world.owner)
        started = self.tick()
        try:
            if publish:
                bridge._require(storage.get(self.move_map) is world.holder, 'lost native move lookup ownership')
                sampled_before = self.statistics['snapshot_bytes']
                values = self._snapshot(world.handle, world.codec)
                self.count('published_snapshot_bytes', self.statistics['snapshot_bytes'] - sampled_before)
                world.moves.clear()
                world.moves.update(values[self.move_map])
                if storage.get(self.score_map) is world.scores:
                    world.scores.clear()
                    world.scores.update(values[self.score_map])
                    storage[self.counter] = values[self.counter]
                self.count('terminal_flushes')
                self.elapsed('terminal_flush', started)
        finally:
            if storage.get(self.move_map) is world.holder:
                storage[self.move_map] = world.moves
            self.library.close(world.handle)
            self.count('native_worlds_released')

    def close(self):
        if self.resident is not None:
            world, self.resident = self.resident, None
            self._release(world, publish=True)


def run(args):
    plan_ref, plan = bridge._pinned(args.plan, PLAN, args.plan_sha256)
    scope = plan['scope']
    decision = plan['decision']
    bridge._require(plan['schema_version'] == 1 and decision['runtime_experiment_authorized'] is True
                    and decision['deployment_authorized'] is False
                    and scope['mode'] == 'ISOLATED_FIXED_ADJACENT_SOURCE_GENERATORS'
                    and plan['backend_contract']['abi'] == 'scar_recursive_driver_rebind_v2',
                    'continuous ownership requires a qualified isolated plan')
    driver_ref, ui_ref = scope['source_driver'], scope['ui_contract']
    driver, ui = read_artifact(driver_ref), read_artifact(ui_ref)
    entry = Path(driver['source']['uri']).resolve(strict=True)
    expected_input = read_artifact(scope['stdin'], json_document=False).decode('ascii').splitlines(keepends=True)
    bridge._require(ui['source_driver'] == driver_ref and scope['source'] == driver['source']
                    and scope['owner_code'] == driver['driver']['code']
                    and expected_input and all(line.endswith('\n') for line in expected_input)
                    and sys.implementation.name == 'cpython' and getattr(sys, '_is_gil_enabled', lambda: True)()
                    and not bridge._observer_active(), 'isolated ownership prerequisites changed')
    bridge._require(ui == bridge.project_ui(ui['ui_source']['uri'], driver_ref['uri'],
                                             ui['request_port'], ui['limit_port'], ui['constraints']),
                    'original consumer no longer satisfies the admitted driver')
    bridge._require(plan['state']['move']['source_field'] == driver['yield']['move_lookup']['field']
                    and plan['state']['score']['source_field'] == driver['initialization']['score_table']['field'],
                    'ownership plan state ports differ from the original driver')

    library_ref, native = bridge._pinned(args.library, bridge.LIBRARY, args.library_sha256)
    base = Path(library_ref['uri']).parent
    binary = (base / native['library']['uri']).resolve(strict=True)
    bridge._require(binary.is_relative_to(base) and native['schema_version'] == 2,
                    'expected a local version 2 native owner')
    read_artifact({**native['library'], 'uri': str(binary)}, json_document=False)
    recursive = read_artifact(native['native'])
    sorted_provider = read_artifact(read_artifact(recursive['fusion'])['sorted'])
    sorted_plan = read_artifact(sorted_provider['region'])
    region = read_artifact(driver['recursive_region'])
    guard = read_artifact(read_artifact(region['original_fusion_flow'])['source_guard'])
    bridge._require(native['code'] == driver['driver']['code'] and recursive['region'] == driver['recursive_region']
                    and guard['plan'] == sorted_provider['region'] and native['world'] == recursive['world']
                    and native['world_order'] == list(recursive['world'])
                    and native['configuration_ports'] == recursive['configuration_ports']
                    and native['table_slots'] == sorted_provider['abi']['table_slots']
                    and native['abi']['move'] == sorted_provider['abi']['record']
                    and native['world'].get(native['clock_fields']['deadline'], {}).get('type') == 'float'
                    and native['clock_fields']['soft'] == driver['time_and_exit']['soft']['deadline_field'],
                    'native owner belongs to a different original graph')

    raw, recipe, revised = bridge._seal(driver)
    pairs = {(row['before_session'], row['after_session']) for row in plan['adjacent_sessions']}
    bridge._require(len(driver['sessions']) > 1
                    and pairs == {(index, index + 1) for index in range(len(driver['sessions']) - 1)},
                    'the observed original searches do not form an uninterrupted owner chain')
    offer = _Owner(driver, ui, recursive, sorted_provider, sorted_plan, guard, native, binary, revised,
                   observe=args.observe, pairs=pairs)
    original_input = sys.stdin
    checked_input = _Input(original_input, expected_input)
    try:
        sys.stdin = checked_input
        execution = bridge._run_main(entry, [], dict(compile_recipe=recipe,
            original_code_key=driver['driver']['code']['key']), raw, revised, offer)
        bridge._require(next(checked_input.remaining, None) is None,
                        'source entry exited before the closed protocol finished')
    finally:
        sys.stdin = original_input
        offer.close()
    bridge._require(offer.live == 0, 'the native owner outlived the original source and executor')
    return dict(schema=SCHEMA, schema_version=1, plan=plan_ref, library=library_ref, execution=execution,
                bridge=artifact_reference(Path(__file__).resolve(), 'scar.python-source'), python_version=sys.version,
                completion=dict(qualified=offer.accepting and offer.completed == len(driver['sessions'])
                                and offer.rebinds == len(pairs),
                                closed_sessions=offer.completed, rebinds=offer.rebinds),
                statistics=dict(offer.statistics) if args.observe else None,
                phase_ns=dict(offer.timings) if args.observe else None,
                sessions=offer.sessions if args.observe else None,
                frontiers=dict(offer.frontiers) if args.observe else None, admitted_for_deployment=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--library-sha256', required=True)
    parser.add_argument('--observe', action='store_true')
    args = parser.parse_args(argv)
    result = run(args)
    with args.report.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()

"""Bounded source-bound observation of ndarray operations and allocation views."""

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import struct
import sys
from types import CodeType, ModuleType
import weakref

from scar.analysis.ndarray_view_contract import NdarrayView, ndarray_view_contract
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.trace.code_snapshot import code_key, site
from scar.trace.numpy_partition_entry_validation import SCHEMA as TRAJECTORY_SCHEMA
from scar.trace.returned_state_recurrence import _codes


SCHEMA = 'scar.ndarray-program-tape'
_EVENTS = {'PY_START', 'PY_RETURN', 'INSTRUCTION'}


@dataclass(frozen=True, slots=True)
class Probe:
    """One selected code-object event with a diagnostic operand projector."""

    code: CodeType
    event: str
    kind: str
    project: object
    offset: int | None = None
    boundary: str | None = None
    outside: bool = False


class _Generations:
    def __init__(self):
        self.current = {}
        self.total = 0

    def key(self, value, fingerprint=None):
        address = id(value)
        previous = self.current.get(address)
        if previous is not None and previous[0]() is value and previous[2] == fingerprint:
            return previous[1]
        generation = self.total
        self.total += 1
        self.current[address] = weakref.ref(value), generation, fingerprint
        return generation


def _pointer(array):
    return array.__array_interface__['data'][0]


def scalar(value):
    """Encode a live primitive at its observed width without decimal conversion."""
    kind = type(value)
    if kind is int:
        return dict(kind='EXACT_INT', value=value)
    if kind is bool:
        return dict(kind='EXACT_BOOL', value=value)
    if kind is float:
        return dict(kind='BITS', dtype='<f8', bytes_hex=struct.pack('<d', value).hex())
    if kind is complex:
        return dict(kind='BITS', dtype='<c16', bytes_hex=struct.pack('<dd', value.real, value.imag).hex())
    numpy = sys.modules.get('numpy')
    if type(numpy) is ModuleType and isinstance(value, numpy.generic):
        dtype = value.dtype
        if dtype.type is kind and dtype.kind in 'biufc' and 0 < dtype.itemsize <= 32:
            return dict(kind='BITS', dtype=dtype.str, bytes_hex=value.tobytes().hex())
    raise ValueError('exact fixed-width numerical scalar required')


def selector(index):
    if index is Ellipsis:
        return dict(kind='ELLIPSIS')
    if type(index) is int:
        return dict(kind='INDEX', index=index)
    if type(index) is slice and all(part is None or type(part) is int
                                    for part in (index.start, index.stop, index.step)):
        return dict(kind='SLICE', start=index.start, stop=index.stop, step=index.step)
    if type(index) is tuple:
        return dict(kind='TUPLE', items=[selector(item) for item in index])
    if index is None:
        return dict(kind='NEW_AXIS')
    raise ValueError('unsupported array selector; basic exact indexing required')


def selected_layout(layout, index):
    """Derive an indexed view from exact geometry without indexing app data."""
    items = list(index if type(index) is tuple else (index,))
    if sum(value is Ellipsis for value in items) > 1:
        raise ValueError('duplicate ellipsis in selector')
    shape, strides = layout['shape'], layout['strides_bytes']
    missing = len(shape) - sum(value is not Ellipsis and value is not None for value in items)
    if missing < 0:
        raise ValueError('selector rank exceeds array rank')
    if any(value is Ellipsis for value in items):
        axis = next(i for i, value in enumerate(items) if value is Ellipsis)
        items[axis:axis + 1] = [slice(None)] * missing
    else:
        items.extend([slice(None)] * missing)
    shape_out, strides_out, displacement = [], [], 0
    dimensions = iter(zip(shape, strides, strict=True))
    for item in items:
        if item is None:
            shape_out.append(1)
            strides_out.append(0)
            continue
        extent, stride = next(dimensions)
        if type(item) is int and -extent <= item < extent:
            displacement += (item % extent) * stride
        elif type(item) is slice:
            selector(item)
            start, stop, step = item.indices(extent)
            size = len(range(start, stop, step))
            if size:
                displacement += start * stride
            shape_out.append(size)
            strides_out.append(stride * step)
        else:
            raise ValueError('invalid basic array selector')
    return dict(layout, object=None, offset_bytes=layout['offset_bytes'] + displacement,
                address_mod_64=(layout['address_mod_64'] + displacement) % 64,
                shape=shape_out, strides_bytes=strides_out,
                derivation='OBSERVED_BASIC_SELECTOR', scalar_index=not shape_out and any(type(item) is int for item in items))


class _Tape:
    def __init__(self, probes, limit):
        self.monitor = sys.monitoring
        self.events = self.monitor.events
        self.tool = next((idx for idx in range(6) if self.monitor.get_tool(idx) is None), None)
        if self.tool is None:
            raise RuntimeError('no free CPython monitor slot')
        self.limit = limit
        self.probes, self.sources, self.masks = {}, {}, {}
        self.allocations, self.objects = _Generations(), _Generations()
        self.storages, self.rows, self.frontiers, self.rounds = {}, [], [], []
        self.round = None
        self.completion = False
        self.cpu_ids = sorted(os.sched_getaffinity(0))
        for probe in probes:
            if (type(probe) is not Probe or type(probe.code) is not CodeType or probe.event not in _EVENTS
                    or probe.boundary not in (None, 'BEGIN', 'END')
                    or (probe.event == 'INSTRUCTION') != (probe.offset is not None)):
                raise ValueError('invalid ndarray tape probe')
            key = code_key(probe.code)
            if key not in self.sources:
                source = artifact_reference(Path(probe.code.co_filename).resolve(strict=True), 'scar.python-source')
                self.sources[key] = dict(source=source, qualname=probe.code.co_qualname)
            self.probes.setdefault((id(probe.code), probe.event, probe.offset), []).append(probe)
            self.masks[probe.code] = self.masks.get(probe.code, 0) | getattr(self.events, probe.event)

    def layout(self, value):
        view = ndarray_view_contract(value)
        if type(view) is not NdarrayView:
            raise ValueError(view.reason)
        allocation_base = _pointer(view.owner)
        storage = self.allocations.key(view.owner, (allocation_base, view.allocation_nbytes))
        residue = allocation_base % 64
        known = dict(size_bytes=view.allocation_nbytes, address_mod_64=residue)
        if storage in self.storages and self.storages[storage] != known:
            raise ValueError('observed owning allocation moved')
        self.storages[storage] = known
        return dict(storage=storage, object=self.objects.key(view.value), dtype=view.dtype,
                    shape=list(view.shape), strides_bytes=list(view.byte_strides),
                    offset_bytes=view.byte_offset, address_mod_64=(residue + view.byte_offset) % 64,
                    writeable=view.writeable)

    def coefficients(self, array, *, max_bytes=64):
        """Retain small explicitly named coefficients, never implicit state data."""
        view = ndarray_view_contract(array)
        if type(view) is not NdarrayView or array.nbytes > max_bytes or max_bytes > 256:
            raise ValueError('coefficient byte budget exceeded')
        return dict(dtype=view.dtype, shape=list(view.shape), order='C', bytes_hex=array.tobytes(order='C').hex())

    def _dispatch(self, event, code, offset, value=None):
        selected = self.probes.get((id(code), event, offset if event == 'INSTRUCTION' else None))
        if not selected:
            return
        frame = sys._getframe(2)
        for probe in selected:
            if self.round is None and not probe.outside and probe.boundary != 'BEGIN':
                continue
            if len(self.rows) >= self.limit:
                if not self.frontiers or self.frontiers[-1].get('reason') != 'TAPE_EVENT_LIMIT':
                    self.frontiers.append(dict(reason='TAPE_EVENT_LIMIT'))
                return
            coordinate = site(code, offset)
            coordinate['code_id'] = coordinate.pop('code')
            try:
                payload = probe.project(self, frame, value)
                if payload is None:
                    continue
                if type(payload) is not dict:
                    raise ValueError('tape projector must return a dict')
                if probe.boundary == 'BEGIN':
                    if self.round is not None:
                        raise ValueError('overlapping observed operation rounds')
                    self.round = len(self.rounds)
                    self.rounds.append(dict(index=self.round, begin_ordinal=len(self.rows)))
                row = dict(kind=probe.kind, event=event, source=coordinate, ordinal=len(self.rows), round=self.round, **payload)
                self.rows.append(row)
                if probe.boundary == 'END':
                    self.rounds[self.round]['end_ordinal'] = row['ordinal']
                    self.round = None
            except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                reason = str(error) if type(error) is ValueError else type(error).__name__
                self.frontiers.append(dict(kind=probe.kind, source=coordinate, reason=reason,
                                           at_ordinal=len(self.rows), round=self.round))

    def start(self, code, offset):
        self._dispatch('PY_START', code, offset)

    def finish(self, code, offset, value):
        self._dispatch('PY_RETURN', code, offset, value)

    def instruction(self, code, offset):
        self._dispatch('INSTRUCTION', code, offset)

    def report(self, trajectory, *, providers, configuration):
        actual = read_artifact(trajectory)
        if (not self.completion or self.round is not None or trajectory['schema'] != TRAJECTORY_SCHEMA
                or actual['capture_is_diagnostic'] is not True or actual['performance_measured'] is not False
                or actual['cpu_ids'] != self.cpu_ids or len(actual['records']) != len(self.rounds)):
            raise ValueError('tape scope differs from the normal round trajectory')
        read_artifact(configuration, json_document=False)
        for material in self.sources.values():
            read_artifact(material['source'], json_document=False)
        endpoints = actual['endpoints']
        rounds = []
        for span, state in zip(self.rounds, actual['records'], strict=True):
            begin, end = span['begin_ordinal'], span['end_ordinal']
            if (span['index'] != state['index']
                    or self.rows[begin]['source']['code_id'] != endpoints['call']['code_id']
                    or self.rows[end]['source']['code_id'] != endpoints['result']['code_id']):
                raise ValueError('tape round order differs from the trajectory')
            rounds.append(dict(**span, input=state['input'], output=state['output'],
                               events=self.rows[begin:end + 1]))
        value = dict(schema=SCHEMA, schema_version=1, trajectory=trajectory, configuration=configuration,
                     entry=endpoints['caller_source'], endpoints=endpoints, providers=providers,
                     sources=self.sources, storages=self.storages, rounds=rounds,
                     cpu_ids=self.cpu_ids, entry_events=[row for row in self.rows if row['round'] is None],
                     counts=dict(sorted(Counter(row['kind'] for row in self.rows).items())),
                     frontiers=self.frontiers, capture_is_diagnostic=True, performance_measured=False,
                     completeness='SELECTED_OPERATION_SITES_ONLY', target_semantics_proven=False,
                     deployment_authorized=False)
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


@contextmanager
def observe_tape(probes, *, max_events=20_000):
    """Monitor only explicit code-object sites and retain a bounded tape."""
    if type(max_events) is not int or not 1 <= max_events <= 1_000_000:
        raise ValueError('invalid ndarray tape event budget')
    tape = _Tape(probes, max_events)
    tape.monitor.use_tool_id(tape.tool, 'scar-ndarray-program-tape')
    try:
        tape.monitor.register_callback(tape.tool, tape.events.PY_START, tape.start)
        tape.monitor.register_callback(tape.tool, tape.events.PY_RETURN, tape.finish)
        tape.monitor.register_callback(tape.tool, tape.events.INSTRUCTION, tape.instruction)
        for code, mask in tape.masks.items():
            tape.monitor.set_local_events(tape.tool, code, mask)
        yield tape
        tape.completion = True
    finally:
        tape.monitor.clear_tool_id(tape.tool)
        tape.monitor.free_tool_id(tape.tool)


def _verify_layouts(value, storage):
    if type(value) is list:
        for element in value:
            _verify_layouts(element, storage)
    elif type(value) is dict:
        if 'storage' in value:
            span = storage[str(value['storage'])]
            shape, strides, dtype, offset = (value[key] for key in ('shape', 'strides_bytes', 'dtype', 'offset_bytes'))
            if (len(shape) != len(strides) or any(type(n) is not int or n < 0 for n in shape)
                    or any(type(n) is not int for n in strides) or type(offset) is not int
                    or not dtype[2:].isdecimal()
                    or (span['address_mod_64'] + offset) % 64 != value['address_mod_64']):
                raise ValueError('tape allocation-relative layout differs')
            steps = [(size - 1) * stride for size, stride in zip(shape, strides, strict=True)]
            lower = offset + sum(min(0, step) for step in steps) if all(shape) else offset
            upper = offset + sum(max(0, step) for step in steps) + int(dtype[2:]) if all(shape) else offset
            if not 0 <= lower <= upper <= span['size_bytes']:
                raise ValueError('tape view reaches outside its allocation')
        for child in value.values():
            _verify_layouts(child, storage)


def check(path):
    """Validate retained bytecode, geometry and trajectory references cold."""
    tape = read_artifact(artifact_reference(path, SCHEMA))
    if (tape['schema_version'] != 1 or tape['capture_is_diagnostic'] is not True
            or tape['performance_measured'] is not False or tape['deployment_authorized'] is not False
            or tape['target_semantics_proven'] is not False or tape['completeness'] != 'SELECTED_OPERATION_SITES_ONLY'):
        raise ValueError('tape qualification differs')
    trajectory = read_artifact(tape['trajectory'])
    read_artifact(tape['configuration'], json_document=False)
    read_artifact(tape['entry'], json_document=False)
    for value in tape['providers'].values():
        if type(value) is dict and set(value) == {'uri', 'size_bytes', 'sha256', 'schema'}:
            read_artifact(value, json_document=False)
    if (tape['trajectory']['schema'] != TRAJECTORY_SCHEMA or tape['endpoints'] != trajectory['endpoints']
            or tape['entry'] != tape['endpoints']['caller_source'] or tape['cpu_ids'] != trajectory['cpu_ids']):
        raise ValueError('tape source or trajectory scope differs')

    groups, codes = {}, {}
    for key, material in tape['sources'].items():
        reference = material['source']
        stored, required = groups.setdefault(reference['uri'], (reference, set()))
        if stored != reference:
            raise ValueError('tape references conflicting source bytes')
        required.add(key)
    for reference, required in groups.values():
        codes.update(_codes(reference, required))
    for span in tape['storages'].values():
        if (type(span['size_bytes']) is not int or span['size_bytes'] < 0
                or type(span['address_mod_64']) is not int or not 0 <= span['address_mod_64'] < 64):
            raise ValueError('tape allocation extent differs')

    rows = list(tape['entry_events'])
    for span, observed in zip(tape['rounds'], trajectory['records'], strict=True):
        events = span['events']
        if (span['index'] != observed['index'] or not events
                or span['begin_ordinal'] != events[0]['ordinal'] or span['end_ordinal'] != events[-1]['ordinal']
                or any(row['round'] != span['index'] for row in events)
                or events[0]['source']['code_id'] != tape['endpoints']['call']['code_id']
                or events[-1]['source']['code_id'] != tape['endpoints']['result']['code_id']
                or span['input'] != observed['input'] or span['output'] != observed['output']):
            raise ValueError('tape round disagrees with its trajectory')
        rows.extend(events)
    rows.sort(key=lambda row: row['ordinal'])
    if [row['ordinal'] for row in rows] != list(range(len(rows))) or dict(Counter(row['kind'] for row in rows)) != tape['counts']:
        raise ValueError('tape event sequence or counts differ')
    for row in rows:
        coordinate = row['source']
        key = coordinate['code_id']
        exact = site(codes[key], coordinate['offset'])
        exact['code_id'] = exact.pop('code')
        if coordinate != exact:
            raise ValueError('tape source event coordinate differs')
        _verify_layouts(row, tape['storages'])
    return tape

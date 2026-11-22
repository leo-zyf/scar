"""Lower an exact private complex128 array tape to checked stock-loop groups."""

from collections import Counter
from math import prod
import re

import numpy as np

from scar.analysis.numpy_partition_iteration import _steps
from scar.optimization.array_live_regions import pack_roots


SCHEMA = 'scar.complex-array-tape-schedule'
TAPE = 'scar.complex-array-tape'
_DTYPE = np.dtype('<c16')
_ITEM = 16
_MAX = 64 << 20
_ID = re.compile(r'[\w.-]{1,80}', re.ASCII)
_DIGEST = re.compile(r'[0-9a-f]{64}')


def _need(valid, reason):
    if not valid:
        raise ValueError('array tape: ' + reason)


def _integer(value, *, maximum=_MAX, minimum=0):
    return type(value) is int and minimum <= value <= maximum


def _view(storage, shape, strides=None, offset=0):
    if strides is None:
        strides = [0] * len(shape)
        step = _ITEM
        for axis in range(len(shape) - 1, -1, -1):
            strides[axis] = step
            step *= shape[axis]
    return dict(storage=storage, shape=list(shape), strides_bytes=list(strides), offset_bytes=offset)


class _Lower:
    def __init__(self, tape):
        self.tape = tape
        self.roots = {}
        self.initialized = {}
        self.dummy = {}
        self.ops = []
        self.constants = []
        self.constant_index = {}
        self.checkpoints = []
        self.signatures = dict(MULTIPLY=set(), ADD=set())
        count = 0
        for name, row in tape['storages'].items():
            count += row['size_bytes']
            _need(count <= _MAX, 'exact tape storage metadata exceeds compilation budget')
            self.root(name, row['size_bytes'], row['address_mod_64'])

    def root(self, name, size, residue=0):
        _need(type(name) is str and _ID.fullmatch(name) and name not in self.roots
              and _integer(size, minimum=_ITEM) and size % _ITEM == 0
              and type(residue) is int and residue in range(0, 64, 8), 'invalid private root')
        self.roots[name] = dict(size_bytes=size, address_mod_64=residue)
        self.initialized[name] = np.zeros(size // _ITEM, dtype=bool)
        self.dummy[name] = bytearray(size)

    def layout(self, view):
        key, shape, strides, offset = (view[k] for k in ('storage', 'shape', 'strides_bytes', 'offset_bytes'))
        _need(key in self.roots and view.get('dtype', '<c16') == '<c16'
              and type(shape) is list and type(strides) is list and len(shape) == len(strides) <= 24
              and all(_integer(n, minimum=1) for n in shape)
              and all(_integer(s, minimum=_ITEM) and s % _ITEM == 0 for s in strides)
              and _integer(offset) and offset % _ITEM == 0, 'invalid complex128 geometry')
        extent = _ITEM
        for stride, n in sorted(zip(strides, shape, strict=True)):
            if n > 1:
                _need(stride >= extent, 'array view may contain overlapping elements')
                extent += (n - 1) * stride
        _need(offset + extent <= self.roots[key]['size_bytes'], 'view escapes its private root')
        return key, shape, strides, offset

    def view(self, value, *, read=False, write=False):
        key, shape, strides, offset = self.layout(value)
        mask = np.ndarray(shape, dtype=bool, buffer=self.initialized[key], offset=offset // _ITEM,
                          strides=tuple(stride // _ITEM for stride in strides))
        if read:
            _need(mask.all(), 'original value has a read before every byte was initialized: ' + key)
        if write:
            mask[...] = True
        return np.ndarray(shape, dtype=_DTYPE, buffer=self.dummy[key], offset=offset, strides=strides)

    def _record(self, kind, read, write, **arguments):
        self.ops.append(dict(kind=kind, accesses=dict(reads=read, writes=write), **arguments))

    def arithmetic(self, kind, left, right, out):
        inputs = (left, right)
        read = tuple(view['storage'] for view in inputs)
        aid = out['storage']
        for view in inputs:
            if view['storage'] == aid:
                _need(self.layout(view) == self.layout(out), 'partial arithmetic alias is unresolved')
        values = [self.view(view, read=True) for view in inputs]
        values.append(self.view(out, write=True))
        keys = (*read, aid)
        addresses = {key: np.ndarray((), dtype=_DTYPE, buffer=self.dummy[key]).ctypes.data for key in keys}
        group = _steps(np, kind, keys, values, addresses)
        strides = tuple(row['stride'] for row in group['operands'])
        self.signatures[kind].add(strides)
        self._record(kind, read, (aid,), group=group)

    def copy(self, source, target):
        left, right = self.view(source, read=True), self.view(target)
        inputs, outputs = source['storage'], target['storage']
        if inputs == outputs:
            _need(self.layout(source) == self.layout(target), 'partial copy alias is unresolved')
            return
        _need(left.size == right.size, 'copy requires equal logical element counts')
        lbase, rbase = left.ctypes.data - source['offset_bytes'], right.ctypes.data - target['offset_bytes']
        groups = []
        if left.shape == right.shape:
            with np.nditer([left, right], order='K', flags=['external_loop'],
                           op_flags=[['readonly'], ['readwrite', 'no_broadcast']]) as iterator:
                for before, after in iterator:
                    groups.append(dict(read=before.ctypes.data-lbase, write=after.ctypes.data-rbase,
                                       count=before.size, increments=[before.strides[0], after.strides[0]]))
        else:
            _need(right.flags.c_contiguous, 'shape-changing copy needs original C-order destination')
            offset = target['offset_bytes']
            with np.nditer(left, order='C', flags=['external_loop']) as iterator:
                for chunk in iterator:
                    groups.append(dict(read=chunk.ctypes.data-lbase, write=offset, count=chunk.size,
                                       increments=[chunk.strides[0], _ITEM]))
                    offset += chunk.nbytes
        self.view(target, write=True)
        self._record('COPY', (inputs,), (outputs,), source=inputs, target=outputs, groups=groups)

    def basis(self, value, index):
        key, shape, _, offset = self.layout(value)
        array = self.view(value)
        _need(array.flags.c_contiguous and _integer(index, maximum=array.size-1),
              'basis producer needs a known valid logical C-contiguous index')
        self.view(value, write=True)
        self._record('INITIAL_BASIS', (), (key,), target=key,
                     offset=offset, index=index, elements=prod(shape))

    def literal(self, name, raw, *, shape):
        _need(type(raw) is str and len(raw) <= 128 and len(raw) == 2 * _ITEM * prod(shape)
              and re.fullmatch(r'[0-9a-f]+', raw), 'only small complex coefficients may be retained')
        if raw not in self.constant_index:
            self.constant_index[raw] = len(self.constants)
            self.constants.append(raw)
        self.root(name, len(raw) // 2)
        value = _view(name, shape)
        self.view(value, write=True)
        self._record('LOAD_LITERAL', (), (name,), target=name,
                     literal=self.constant_index[raw], size=len(raw)//2)
        return value

    def outer(self, left, right, out):
        arrays = tuple(self.view(view) for view in (left, right, out))
        _need(all(array.flags.c_contiguous for array in arrays)
              and arrays[0].size * arrays[1].size == arrays[2].size, 'outer requires stock C-order roots')
        a, b = arrays[0].size, arrays[1].size
        flat = lambda v, shape: _view(v['storage'], shape, offset=v['offset_bytes'])
        self.arithmetic('MULTIPLY', flat(left, [a, 1]), flat(right, [1, b]), flat(out, [a, b]))

    @staticmethod
    def _slice(view, index):
        return _view(view['storage'], view['shape'][:-1], view['strides_bytes'][:-1],
                     view['offset_bytes'] + index * view['strides_bytes'][-1])

    def partition(self, target, matrix, out, identifier):
        self.layout(target)
        _need(target['shape'] == out['shape'] and target['shape'][-1:] == [2]
              and matrix['shape'] == [2, 2] and matrix['strides_bytes'] == [32, 16]
              and len({target['storage'], matrix['storage'], out['storage']}) == 3,
              'partition requires exact disjoint two-slice geometry')
        self.copy(target, out)
        for i in range(2):
            result = self._slice(out, i)
            coef = _view(matrix['storage'], [], offset=matrix['offset_bytes'] + (i * 2 + i) * _ITEM)
            self.arithmetic('MULTIPLY', result, coef, result)
            peer = 1 - i
            source = self._slice(target, peer)
            array = self.view(source, read=True)
            packed = np.empty_like(array, order='K')
            temp_id = f'part-tmp-{identifier}-{i}'
            self.root(temp_id, packed.nbytes)
            temporary = _view(temp_id, list(packed.shape), list(packed.strides))
            coef = _view(matrix['storage'], [], offset=matrix['offset_bytes'] + (i * 2 + peer) * _ITEM)
            self.arithmetic('MULTIPLY', source, coef, temporary)
            self.arithmetic('ADD', result, temporary, result)

    def normalize(self, source, target, selected):
        left, right = self.view(source, read=True), self.view(target)
        _need(all(array.flags.c_contiguous for array in (left, right))
              and left.size == right.size and source['storage'] != target['storage']
              and type(selected) is bool, 'normalization needs two disjoint contiguous arrays and observed choice')
        self.view(target, write=True)
        self._record('NORMALIZE', (source['storage'],),
                     (target['storage'],), source=source, target=target, elements=left.size, selected=selected)

    def checkpoint(self, source, expected):
        value = self.view(source, read=True)
        _need(value.flags.c_contiguous and type(expected) is str and _DIGEST.fullmatch(expected),
              'validation needs a full contiguous state digest')
        self._record('CHECKPOINT', (source['storage'],), (), source=source,
                     bytes=value.nbytes, checkpoint=len(self.checkpoints))
        self.checkpoints.append(dict(sha256=expected, size_bytes=value.nbytes))

    def lower(self, event, index):
        kind = event['kind']
        if kind == 'INITIAL_BASIS':
            self.basis(event['output'], event['index'])
        elif kind == 'COPY':
            self.copy(event['input'], event['output'])
        elif kind == 'VIEW':
            before, after = event['input'], event['output']
            self.view(before)
            self.view(after)
            _need(before['storage'] == after['storage'], 'view operation cannot introduce new storage')
        elif kind == 'OUTER':
            self.outer(event['left'], event['right'], event['output'])
        elif kind == 'PARTITION':
            coefficient = self.literal(f'matrix-{index}', event['coefficients'], shape=[2, 2])
            self.partition(event['target'], coefficient, event['out'], index)
        elif kind == 'PHASE_MULTIPLY':
            coefficient = self.literal(f'factor-{index}', event['coefficient'], shape=[])
            self.arithmetic('MULTIPLY', event['view'], coefficient, event['view'])
        elif kind == 'NORMALIZE':
            self.normalize(event['input'], event['normalized'], event['selected'])
        elif kind == 'CHECKPOINT':
            self.checkpoint(event['input'], event['sha256'])
        else:
            raise ValueError('array tape: missing lowering for ' + kind)


def lower_tape(tape):
    """Replay normal guards and assign storage before C++ emission."""
    _need(tape.get('schema') == TAPE and tape.get('schema_version') == 1
          and not tape['frontiers'] and 0 < len(tape['events']) <= 8192,
          'complete bounded exact operation tape required')
    lower = _Lower(tape)
    starts = []
    for index, event in enumerate(tape['events']):
        begin = len(lower.ops)
        starts.append(begin)
        lower.lower(event, index)
        for op in lower.ops[begin:]:
            op['event_index'] = index
    _need(bool(lower.checkpoints), 'state comparison points required')
    starts.append(len(lower.ops))
    cursor = 0
    rounds = []
    for index, row in enumerate(tape['rounds']):
        start, end = row['begin'], row['end']
        _need(row['index'] == index and start == cursor and start < end < len(starts)
              and lower.ops[starts[end] - 1]['kind'] == 'CHECKPOINT',
              'compiled round does not have its original contiguous operation span')
        rounds.append([starts[start], starts[end]])
        cursor = end
    _need(cursor == len(tape['events']), 'source events escape their compiled round scope')
    accesses = [(op['accesses']['reads'], op['accesses']['writes']) for op in lower.ops]
    used = {key for reads, writes in accesses for key in (*reads, *writes)}
    storage_plan = pack_roots({k: v for k, v in lower.roots.items() if k in used}, accesses)
    return dict(schema=SCHEMA, schema_version=1, operations=lower.ops, storage=storage_plan, rounds=rounds,
                literals=lower.constants, checkpoints=lower.checkpoints,
                source_operation_counts=dict(Counter(row['kind'] for row in tape['events'])),
                signatures={key: [list(row) for row in sorted(value)] for key, value in lower.signatures.items()},
                shadow_state_is_private=True, preserved_source_effect_order=True)

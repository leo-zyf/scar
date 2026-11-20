"""Retain bounded exact operands and ordinary returns of a NumPy partition call."""

import argparse
from collections import Counter
import ctypes
from hashlib import sha256
import inspect
from itertools import product
import json
import os
from pathlib import Path
import runpy
import sys
from types import FunctionType, ModuleType

from scar.analysis.ndarray_view_contract import NdarrayView, ndarray_view_contract
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.trace.code_snapshot import code_key
from scar.trace.numpy_ufunc_calls import CORE_MODULES


SCHEMA = 'scar.numpy-partition-witness'
ROLES = ('target', 'matrix', 'out')
PAIRS = (('target', 'out'), ('target', 'matrix'), ('matrix', 'out'))
_SUSPENDED = inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR
_MAX_GEOMETRY_ROWS = 512


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _provider(numpy):
    core = next((sys.modules.get(name) for name in CORE_MODULES
                 if type(sys.modules.get(name)) is ModuleType
                 and vars(sys.modules[name]).get('ndarray') is numpy.ndarray), None)
    if core is None:
        raise ValueError('loaded NumPy core provider required')
    ufuncs = {}
    for name in ('multiply', 'add'):
        value = vars(numpy).get(name)
        if type(value) is not numpy.ufunc or vars(core).get(name) is not value:
            raise ValueError('NumPy core ufunc binding differs: ' + name)
        ufuncs[name] = dict(name=value.__name__, nin=value.nin, nout=value.nout,
                            signature=value.signature, types=list(value.types))
    return dict(version=numpy.__version__, module=_ref(numpy.__file__, 'scar.python-source'),
                extension=_ref(core.__file__, 'scar.native-provider-library'),
                extension_module=core.__name__, ufuncs=ufuncs, errstate=numpy.geterr())


def _selectors(value, extent):
    if type(value) not in (tuple, list) or len(value) != extent:
        raise ValueError('expected a complete basic last-axis partition')
    if any(type(item) is not tuple or len(item) != 2 or item[0] is not Ellipsis
           or type(item[1]) is not int or not -extent <= item[1] < extent for item in value):
        raise ValueError('unsupported basic last-axis index')
    indices = [item[1] % extent for item in value]
    if len(set(indices)) != extent:
        raise ValueError('partition indices overlap')
    return [dict(kind='ELLIPSIS_LAST_AXIS_INDEX', index=item[1], normalized_index=index)
            for item, index in zip(value, indices, strict=True)]


def _view_groups(views):
    groups, assigned = {}, {}
    for name, view in views.items():
        identity = next((key for key, (owner, _, _) in groups.items() if owner is view.owner), None)
        start, end = view.byte_bounds
        if identity is None:
            identity = f'storage-{len(groups)}'
        else:
            _, old_start, old_end = groups[identity]
            start, end = min(start, old_start), max(end, old_end)
        groups[identity] = view.owner, start, end
        assigned[name] = identity
    return groups, assigned


def _pointer(array):
    return array.__array_interface__['data'][0]


def _geometry(view, identity, start):
    lower, upper = view.byte_bounds
    return dict(storage_id=identity, dtype=view.dtype, shape=list(view.shape),
                strides_bytes=list(view.byte_strides), offset_bytes=view.byte_offset - start,
                byte_bounds=[lower - start, upper - start], address_mod_64=_pointer(view.value) % 64,
                writeable=view.writeable)


def _aliases(views):
    results = {}
    for left, right in PAIRS:
        a, b = views[left], views[right]
        a_base, b_base = _pointer(a.owner), _pointer(b.owner)
        disjoint = (a_base + a.byte_bounds[1] <= b_base + b.byte_bounds[0]
                    or b_base + b.byte_bounds[1] <= a_base + a.byte_bounds[0])
        results[f'{left}_{right}'] = dict(same_object=a.value is b.value,
                    same_storage=a.same_storage(b), byte_envelopes_disjoint=disjoint)
    return results


class _Observer:
    def __init__(self, module, symbol, ports, limit, max_bytes):
        self.module, self.symbol, self.ports = module, symbol, ports
        self.limit, self.max_bytes = limit, max_bytes
        self.monitor = sys.monitoring
        self.events = self.monitor.events
        self.tool = next((i for i in range(6) if self.monitor.get_tool(i) is None), None)
        if self.tool is None:
            raise RuntimeError('no free CPython monitoring slot')
        self.code = self.focus = self.numpy = self.provider = self.function = None
        self.calls = self.returns = 0
        self.uncatalogued_geometry_calls = 0
        self.groups, self.frontiers = Counter(), Counter()
        self.pending, self.samples, self.discovery = {}, [], {}
        self.discovery_limit = limit - 2
        self.rounding = ctypes.CDLL(None).fegetround
        self.rounding.restype = ctypes.c_int
        self.rounding.argtypes = ()

    def _claim(self, code, frame):
        if code.co_qualname != self.symbol or frame.f_globals.get('__name__') != self.module:
            return False
        module = sys.modules.get(self.module)
        if type(module) is not ModuleType or vars(module) is not frame.f_globals:
            return False
        fn = vars(module).get(self.symbol)
        if type(fn) is not FunctionType or fn.__code__ is not code or code.co_flags & _SUSPENDED:
            return False
        self.numpy = sys.modules['numpy']
        self.provider = _provider(self.numpy)
        self.code, self.function = code, fn
        self.focus = dict(module=self.module, symbol=self.symbol, code_id=code_key(code),
                          qualname=code.co_qualname, first_line=code.co_firstlineno,
                          source=_ref(code.co_filename, 'scar.python-source'), parameters=self.ports)
        self.monitor.set_local_events(self.tool, code, self.events.PY_START | self.events.PY_RETURN)
        self.monitor.set_events(self.tool, 0)
        return True

    def start(self, code, offset):
        if self.code is None and code.co_qualname != self.symbol:
            return
        frame = sys._getframe(1)
        if self.code is None and not self._claim(code, frame):
            return
        if code is not self.code or frame.f_code is not code:
            return
        self.calls += 1
        local = frame.f_locals
        values = {name: local.get(self.ports[name]) for name in ROLES}
        if any(type(value) is not self.numpy.ndarray for value in values.values()):
            self.frontiers['EXACT_NDARRAY_OPERANDS_REQUIRED'] += 1
            return
        signature = tuple((value.dtype.str, value.shape, value.strides) for value in values.values())
        if signature in self.groups or len(self.groups) < _MAX_GEOMETRY_ROWS:
            self.groups[signature] += 1
        else:
            self.uncatalogued_geometry_calls += 1
        if len(self.samples) + len(self.pending) >= self.limit:
            return
        discovered = self.discovery.get(signature)
        if discovered is None and len(self.discovery) >= self.discovery_limit:
            return
        if discovered is not None and (len(self.discovery) < 2 or discovered['holdout']):
            return
        try:
            self._sample(frame, values, signature, discovered, local.get(self.ports['slices']))
        except ValueError as error:
            self.frontiers[str(error)] += 1

    def _sample(self, frame, values, signature, discovered, selectors):
        views = {name: ndarray_view_contract(value) for name, value in values.items()}
        for view in views.values():
            if type(view) is not NdarrayView:
                raise ValueError(view.reason)
            if view.dtype != '<c16' or not view.shape or not view.value.size:
                raise ValueError('NONEMPTY_NATIVE_COMPLEX128_OPERANDS_REQUIRED')
        target, matrix, out = (views[name] for name in ROLES)
        extent = target.shape[-1]
        if target.shape != out.shape or matrix.shape != (extent, extent):
            raise ValueError('PARTITION_GEOMETRY_DIFFERS')
        slices = _selectors(selectors, extent)
        groups, ids = _view_groups(views)
        if 2 * sum(end - start for _, start, end in groups.values()) + matrix.value.nbytes > self.max_bytes:
            raise ValueError('SAMPLE_BYTE_BUDGET_EXCEEDED')
        reads = {ids['target'], ids['matrix']}
        before = {key: ctypes.string_at(_pointer(owner) + start, end - start)
                  for key, (owner, start, end) in groups.items() if key in reads}
        packed = matrix.value.tobytes(order='C')
        fingerprint = (sha256(before[ids['target']]).digest(), sha256(packed).digest())
        if discovered is not None and fingerprint == discovered['fingerprint']:
            return
        caller = frame.f_back
        record = dict(ordinal=self.calls, case='DISCOVERY' if discovered is None else 'HOLDOUT',
                      slices=slices, numpy_errstate=self.numpy.geterr(), rounding_mode_raw=self.rounding(),
                      caller=dict(code_id=code_key(caller.f_code), file=caller.f_code.co_filename,
                                  offset=caller.f_lasti, line=caller.f_lineno),
                      aliases=_aliases(views),
                      storages={key: dict(size_bytes=end - start, allocation_nbytes=owner.nbytes,
                                         span_start_in_allocation_bytes=start,
                                         address_mod_64=(_pointer(owner) + start) % 64)
                                for key, (owner, start, end) in groups.items()},
                      before_omissions={key: 'OUT_ONLY_INITIALIZATION_UNPROVEN' for key in groups if key not in reads})
        for name, view in views.items():
            record[name] = _geometry(view, ids[name], groups[ids[name]][1])
        # A pending record holds its actual frame until a normal return or
        # collection cleanup. An unwound frame's address cannot be recycled to
        # complete an unrelated record.
        self.pending[id(frame)] = (frame, record, groups, views, ids, before, packed, signature, fingerprint)

    def finish(self, code, offset, result):
        self.returns += 1
        frame = sys._getframe(1)
        pending = self.pending.pop(id(frame), None)
        if pending is None:
            return
        original, record, groups, views, ids, before, packed, signature, fingerprint = pending
        if original is not frame:
            return
        if result is not views['out'].value:
            self.frontiers['NORMAL_RESULT_NOT_IDENTICAL_TO_OUT'] += 1
            return
        post = {}
        for name, view in views.items():
            actual = ndarray_view_contract(view.value)
            if type(actual) is not NdarrayView or not actual.same_storage(view) or actual.geometry() != view.geometry():
                self.frontiers['NORMAL_RETURN_GEOMETRY_OR_STORAGE_CHANGED'] += 1
                return
            post[name] = _geometry(actual, ids[name], groups[ids[name]][1])
        after = {key: ctypes.string_at(_pointer(owner) + start, end - start)
                 for key, (owner, start, end) in groups.items()}
        unchanged = all(before[ids[name]] == after[ids[name]] for name in ('target', 'matrix'))
        record.update(normal_return=True, returned_alias='out', post=post, read_storage_bytes_unchanged=unchanged)
        self.samples.append((record, before, after, packed))
        if signature in self.discovery:
            self.discovery[signature]['holdout'] = True
        else:
            self.discovery[signature] = dict(fingerprint=fingerprint, holdout=False)

    def run(self, entry, argv):
        old_argv, old_path = sys.argv, sys.path[:]
        sys.argv, sys.path[0] = [str(entry), *argv], str(entry.parent)
        self.monitor.use_tool_id(self.tool, 'scar-numpy-partition-witness')
        try:
            self.monitor.register_callback(self.tool, self.events.PY_START, self.start)
            self.monitor.register_callback(self.tool, self.events.PY_RETURN, self.finish)
            self.monitor.set_events(self.tool, self.events.PY_START)
            runpy.run_path(str(entry), run_name='__main__')
        finally:
            self.monitor.clear_tool_id(self.tool)
            self.monitor.free_tool_id(self.tool)
            self.pending.clear()
            sys.argv, sys.path[:] = old_argv, old_path


def capture(out, entry, argv, *, module, symbol, ports, samples=6, max_bytes=2_000_000):
    """Observe the supplied original entry; keep no persistent tensor IDs."""
    if not 4 <= samples <= 8 or not 1 <= max_bytes <= 16_000_000:
        raise ValueError('expected a 4–8 sample limit and positive bounded byte budget')
    target = Path(out).absolute()
    if target.exists():
        raise FileExistsError(target)
    entry = Path(entry).resolve(strict=True)
    entry_ref = _ref(entry, 'scar.python-source')
    observer = _Observer(module, symbol, dict(ports), samples, max_bytes)
    observer.run(entry, argv)
    if (observer.focus is None or _ref(entry, 'scar.python-source') != entry_ref
            or vars(sys.modules[module]).get(symbol) is not observer.function
            or observer.function.__code__ is not observer.code
            or _ref(observer.code.co_filename, 'scar.python-source') != observer.focus['source']):
        raise ValueError('original source or callable binding differs')
    if len(observer.samples) < 4 or len(observer.discovery) < 2 or not any(
            record['case'] == 'HOLDOUT' for record, *_ in observer.samples):
        raise ValueError('insufficient independent geometry and holdout operands')
    target.mkdir(parents=True)

    def publish(path, data, schema):
        with path.open('xb') as stream:
            stream.write(data)
        return _ref(path, schema)

    records = []
    for record, before, after, packed in sorted(observer.samples, key=lambda row: row[0]['ordinal']):
        name = f'call-{record["ordinal"]:06d}'
        directory = target / name
        directory.mkdir()
        record['id'] = name
        for label, mapping in [('storage_before', before), ('storage_after', after)]:
            record[label] = {key: publish(directory / f'{label}-{key}.bin', data, 'scar.ndarray-storage-bytes')
                             for key, data in mapping.items()}
        record['matrix_packed'] = publish(directory / 'matrix.c128', packed, 'scar.packed-c128-array')
        records.append(record)
    report = dict(schema=SCHEMA, schema_version=1, entry=entry_ref, argv=[str(entry), *argv],
                  focus=observer.focus, numpy=observer.provider, cpu_ids=sorted(os.sched_getaffinity(0)),
                  counts=dict(calls=observer.calls, normal_returns=observer.returns,
                              without_normal_return=observer.calls - observer.returns, retained=len(records),
                              discovery_geometries=len(observer.discovery), frontiers=dict(observer.frontiers),
                              uncatalogued_geometry_calls=observer.uncatalogued_geometry_calls),
                  geometries=[dict(roles={name: dict(dtype=dtype, shape=shape, strides_bytes=strides)
                                         for name, (dtype, shape, strides) in zip(ROLES, signature, strict=True)},
                                   calls=count) for signature, count in observer.groups.items()],
                  samples=records, target_executed=True, capture_is_diagnostic=True,
                  performance_measured=False, deployment_authorized=False)
    with (target / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return report


def _check_geometry(view, storage):
    shape, strides = view['shape'], view['strides_bytes']
    offset, size = view['offset_bytes'], storage['size_bytes']
    if (view['dtype'] != '<c16' or not shape or len(shape) != len(strides)
            or any(type(n) is not int or n <= 0 for n in shape)
            or any(type(n) is not int for n in strides) or type(offset) is not int
            or type(view['writeable']) is not bool):
        raise ValueError('partition view geometry differs')
    steps = [(n - 1) * stride for n, stride in zip(shape, strides, strict=True)]
    lower = offset + sum(min(0, step) for step in steps)
    upper = offset + sum(max(0, step) for step in steps) + 16
    if (not 0 <= lower < upper <= size or view['byte_bounds'] != [lower, upper]
            or view['address_mod_64'] != (offset + storage['address_mod_64']) % 64):
        raise ValueError('partition view storage-relative bounds or alignment differs')


def _read_bytes(reference, schema):
    if reference['schema'] != schema:
        raise ValueError('partition witness artifact type differs')
    return read_artifact(reference, json_document=False)


def _check_sample(record):
    if (record['normal_return'] is not True or record['returned_alias'] != 'out'
            or record['case'] not in ('DISCOVERY', 'HOLDOUT')
            or record['id'] != f'call-{record["ordinal"]:06d}'):
        raise ValueError('partition normal return witness differs')
    storages = record['storages']
    views = {name: record[name] for name in ROLES}
    ids = {name: view['storage_id'] for name, view in views.items()}
    if set(storages) != set(ids.values()) or views != record['post']:
        raise ValueError('partition view and return storage mapping differs')
    for storage in storages.values():
        size, start = storage['size_bytes'], storage['span_start_in_allocation_bytes']
        capacity, alignment = storage['allocation_nbytes'], storage['address_mod_64']
        if (any(type(value) is not int for value in (size, start, capacity, alignment))
                or not 0 <= start < start + size <= capacity or not 0 <= alignment < 64):
            raise ValueError('partition storage geometry differs')
    for name, view in views.items():
        _check_geometry(view, storages[ids[name]])
    target, matrix, out = (views[name] for name in ROLES)
    extent = target['shape'][-1]
    if target['shape'] != out['shape'] or matrix['shape'] != [extent, extent]:
        raise ValueError('partition operand shapes differ')
    slices = [(Ellipsis, row['index']) for row in record['slices']]
    if _selectors(slices, extent) != record['slices']:
        raise ValueError('partition selector encoding differs')
    reads = {ids['target'], ids['matrix']}
    before_refs, after_refs = record['storage_before'], record['storage_after']
    if (set(before_refs) != reads or set(after_refs) != set(storages)
            or record['before_omissions'] != {key: 'OUT_ONLY_INITIALIZATION_UNPROVEN' for key in storages.keys() - reads}):
        raise ValueError('partition operand byte evidence differs')
    before, after = ({key: _read_bytes(ref, 'scar.ndarray-storage-bytes') for key, ref in refs.items()}
                     for refs in (before_refs, after_refs))
    if (any(len(data) != storages[key]['size_bytes'] for snapshot in (before, after)
            for key, data in snapshot.items())
            or record['read_storage_bytes_unchanged'] is not all(before[key] == after[key] for key in reads)):
        raise ValueError('partition captured storage lengths or preservation differs')
    if set(record['aliases']) != {f'{a}_{b}' for a, b in PAIRS}:
        raise ValueError('partition alias pair coverage differs')
    for a, b in PAIRS:
        alias = record['aliases'][f'{a}_{b}']
        if (set(alias) != {'same_object', 'same_storage', 'byte_envelopes_disjoint'}
                or any(type(value) is not bool for value in alias.values())
                or alias['same_storage'] is not (ids[a] == ids[b])
                or (alias['same_object'] and views[a] != views[b])):
            raise ValueError('partition recorded alias differs')
        if alias['same_storage']:
            a0, a1 = views[a]['byte_bounds']
            b0, b1 = views[b]['byte_bounds']
            if alias['byte_envelopes_disjoint'] is not (a1 <= b0 or b1 <= a0):
                raise ValueError('partition shared storage overlap differs')
    matrix_bytes = before[ids['matrix']]
    packed = _read_bytes(record['matrix_packed'], 'scar.packed-c128-array')
    if len(packed) != extent * extent * 16:
        raise ValueError('partition packed matrix shape differs')
    strides = matrix['strides_bytes']
    for index, (row, col) in enumerate(product(range(extent), repeat=2)):
        offset = matrix['offset_bytes'] + row * strides[0] + col * strides[1]
        if matrix_bytes[offset:offset + 16] != packed[index * 16:index * 16 + 16]:
            raise ValueError('partition matrix packed values differ')
    signature = tuple((view['dtype'], tuple(view['shape']), tuple(view['strides_bytes'])) for view in views.values())
    return signature, (before_refs[ids['target']]['sha256'], record['matrix_packed']['sha256'])


def check(path):
    """Verify retained artifacts and geometry without executing target code."""
    report = json.loads(Path(path).read_bytes())
    if (report['schema'] != SCHEMA or report['schema_version'] != 1
            or report['target_executed'] is not True or report['capture_is_diagnostic'] is not True
            or report['deployment_authorized'] is not False or report['performance_measured'] is not False
            or not report['focus']['code_id'].startswith('pycode-v2:')):
        raise ValueError('partition witness identity differs')
    for reference in (report['entry'], report['focus']['source'], report['numpy']['module']):
        _read_bytes(reference, 'scar.python-source')
    _read_bytes(report['numpy']['extension'], 'scar.native-provider-library')
    samples, counts = report['samples'], report['counts']
    ordinals = [row['ordinal'] for row in samples]
    if (not 4 <= len(samples) <= 8 or counts['retained'] != len(samples)
            or ordinals != sorted(set(ordinals)) or not 1 <= ordinals[0] <= ordinals[-1] <= counts['calls']
            or counts['calls'] != counts['normal_returns'] + counts['without_normal_return']
            or len(samples) > counts['normal_returns']):
        raise ValueError('partition call counts differ')
    discovery, holdouts = {}, set()
    for row in samples:
        signature, fingerprint = _check_sample(row)
        if row['case'] == 'DISCOVERY':
            if signature in discovery:
                raise ValueError('repeated partition discovery geometry')
            discovery[signature] = fingerprint
        else:
            if signature not in discovery or signature in holdouts or fingerprint == discovery[signature]:
                raise ValueError('partition holdout lacks a distinct input')
            holdouts.add(signature)
    if not holdouts or len(discovery) < 2 or len(discovery) != counts['discovery_geometries']:
        raise ValueError('partition discovery or holdout coverage differs')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--module', required=True)
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--max-bytes', type=int, default=2_000_000)
    for role in (*ROLES, 'slices'):
        parser.add_argument(f'--{role}-parameter', default=role)
    parser.add_argument('target', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    target = args.target[1:] if args.target[:1] == ['--'] else args.target
    if not target:
        parser.error('provide --out DIRECTORY --module MODULE --symbol FUNCTION -- SCRIPT [ARGS]')
    ports = {role: getattr(args, f'{role}_parameter') for role in (*ROLES, 'slices')}
    report = capture(args.out, target[0], target[1:], module=args.module, symbol=args.symbol,
                     ports=ports, samples=args.samples, max_bytes=args.max_bytes)
    print(json.dumps(report['counts'], sort_keys=True))


if __name__ == '__main__':
    main()

"""Run an isolated CPU trial at one checked eager sort boundary."""

import argparse
import builtins
from collections import Counter, namedtuple
from _collections import _tuplegetter
from contextlib import nullcontext
import ctypes as ffi
from importlib import import_module
import json
from pathlib import Path
from struct import Struct
import sys
from threading import Lock
from time import perf_counter_ns
from types import FunctionType, GeneratorType

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.scalar_score_ir import _int64, _require
from scar.runtime.csr_region_entry import _observer_active
from scar.runtime.entry_runner import run_original_script
from scar.runtime.scalar_sorted_observe import _seal
from scar.runtime.scalar_table_contract import _run_main
from scar.trace.code_snapshot import describe_code


SCHEMA = 'scar.scalar-sorted-source-study'
_MISSING = object()
_SORTED = builtins.sorted
_ABS = builtins.abs
_ENUMERATE = builtins.enumerate
_MAX_ROWS = 65536
_MAX_STAMP = (1 << 64) - 1


def _namespace_exact(value):
    return type(value) is dict and all(type(name) is str for name in value)


def _same_rows(root, anchors):
    original, rows = anchors
    return (root is original and _namespace_exact(root) and len(root) == len(rows)
            and all(root.get(name) is value for name, value in rows))


def _freeze_rows(root, expected):
    if not _namespace_exact(root) or len(root) != len(expected):
        return None
    if any(type(row) is not tuple or any(type(value) is not int for value in row)
           for row in root.values()) or root != expected:
        return None
    return root, tuple(root.items())


def _unstarted(generator):
    return (type(generator) is GeneratorType and not generator.gi_running and not generator.gi_suspended
            and generator.gi_frame is not None)


class _TypeBindings:
    """Remember the exact lookup path for each admitted descriptor."""
    def __init__(self):
        self.slots = []
        self.hierarchies = []

    def watch(self, cls):
        hierarchy = cls.__mro__
        _require(type(cls) is type and all(type(base) is type for base in hierarchy),
                 'study requires standard class lookup')
        self.hierarchies.append((cls, hierarchy))
        return [vars(base) for base in hierarchy]

    def attr(self, hierarchy, name):
        for namespace in hierarchy:
            value = namespace.get(name, _MISSING)
            self.slots.append((namespace, name, value))
            if value is not _MISSING:
                return value
        return _MISSING

    def valid(self):
        return (all(cls.__mro__ is hierarchy for cls, hierarchy in self.hierarchies) and
                all(scope.get(name, _MISSING) is value for scope, name, value in self.slots))


class _Offer:
    def __init__(self, binary, artifact, plan, transition, score_globals, *, epoch, observe, shadow):
        from scar.codegen.scalar_sorted_library import pod_types

        self.plan = plan
        self.reverse = plan['descending']
        self.observe = observe
        self.shadow = shadow
        self.counts = Counter()
        self.versions = Counter()
        self.thresholds = Counter()
        self.phase_ns = Counter()
        self.lock = Lock()
        self.binding = None
        self.table_binding = None
        self.epoch = epoch
        self.stamp_slot = None
        self.source_name = plan['factory']['parameters'][0]
        self.scorer_name = plan['score']['qualname'].rsplit('.', 1)[-1]
        self.factory_name = plan['factory']['qualname'].rsplit('.', 1)[-1]
        self.fields = plan['state_field_order']
        self.types = plan['state_fields']
        self.expected_imports = {name:getattr(import_module(module), export)
                                 for name, (module, export) in plan['builtin_calls'].items()}
        _require(all(name not in score_globals or score_globals[name] == value
                     for name, value in plan['scalar_globals'].items()), 'factory/scorer constants disagree')
        self.constants = {**plan['scalar_globals'], **score_globals}
        self.maps = {name:{key:tuple(row) for key, row in mapping.items()}
                     for name, mapping in plan['integer_rows'].items()}
        self.live_maps = {}
        self.expected_new = namedtuple('_SortedStudyRecord', plan['record_fields']).__new__.__code__
        abi = artifact['abi']
        self.snapshots = [
            {name:tuple(row) for name, row in transition['table_versions'][label]['rows'].items()}
            for label in abi['table_slots']
        ]
        self.labels = abi['table_slots']
        self.input_layout = abi['input']
        self.output_layout = abi['record']
        self.Input, self.Record = pod_types(abi)
        self.record_unpack = Struct(f"={self.output_layout['scalars']}q{self.output_layout['lengths']}I"
                                    f"{self.output_layout['strings']}s")
        self.record_stride = ffi.sizeof(self.Record)
        self.length_base = self.output_layout['scalars']
        _require(self.record_unpack.size <= self.record_stride
                 and self.Record.lengths.offset == 8 * self.length_base
                 and self.Record.strings.offset == self.Record.lengths.offset + 4 * self.output_layout['lengths'],
                 'record layout differs from its native C ABI')
        native = ffi.PyDLL(str(binary))
        self.call_native = native.scar_scalar_sorted_v1
        self.call_native.argtypes = (ffi.c_uint32, ffi.POINTER(self.Input), ffi.c_int64, ffi.c_int64,
                                     ffi.POINTER(self.Record), ffi.c_size_t, ffi.POINTER(ffi.c_size_t))
        self.call_native.restype = ffi.c_int
        self.inputs = self.Input()
        self.scalars = self.inputs.scalars
        self.lengths = self.inputs.lengths
        self.string_base = ffi.addressof(self.inputs) + self.Input.strings.offset
        self.written = ffi.c_size_t()
        self.outputs = (self.Record * 128)()
        self.output_pointer = ffi.byref(self.written)
        self.inputs_pointer = ffi.byref(self.inputs)
        self.input_ports = []
        by_port = {}
        for index, name in enumerate(self.fields):
            if self.types[name] == 'bool_pair':
                for bit in range(2):
                    by_port[f'field_{name}__{bit}'] = index, bit
            else:
                by_port['field_'+name] = index, None
        for port in self.input_layout['ports']:
            index, bit = by_port[port['name']]
            offset = self.string_base + port.get('offset', 0)
            self.input_ports.append((port['type'], index, bit, port['slot'], offset, port.get('capacity', 0)))
        _require([dict(name=row['name'],type=row['type']) for row in self.input_layout['ports']] == plan['inputs']
                 and [row['type'] for row in self.output_layout['ports']] == ['int', *plan['record_types']],
                 'study input or record layout differs from the source contract')

    def _fallback(self, target, iterator, reverse, reason):
        if self.observe:
            self.counts[reason] += 1
        return target(iterator, reverse=reverse)

    def _bind(self, receiver, factory_frame, scope_frame, source_code, scope_code):
        actor = type(receiver)
        namespace = scope_frame.f_globals
        builtin_scope = scope_frame.f_builtins
        try:
            _require(type(actor) is type and issubclass(actor, tuple) and
                     _namespace_exact(namespace) and _namespace_exact(builtin_scope),
                     'source receiver and globals need exact tuple and dictionary types')
            cls = namespace.get(self.plan['constructor'])
            _require(type(cls) is type and issubclass(cls, tuple), 'output constructor must be an ordinary tuple type')
            bindings = _TypeBindings()
            actor_mro = bindings.watch(actor)
            constructor_mro = bindings.watch(cls)
            attr = bindings.attr
            _require(attr(actor_mro, '__getattribute__') is tuple.__getattribute__
                     and attr(actor_mro, '__getattr__') is _MISSING
                     and attr(constructor_mro, '__getattribute__') is tuple.__getattribute__
                     and attr(constructor_mro, '__init__') is tuple.__init__
                     and attr(constructor_mro, '__del__') is _MISSING
                     and all(attr(constructor_mro, name) is getattr(tuple, name)
                             for name in ('__eq__','__lt__','__gt__','__iter__','__getitem__')),
                     'exact record comparison, construction and reads must use tuple behavior')
            for i, name in enumerate(self.fields):
                member = attr(actor_mro, name)
                _require(type(member) is _tuplegetter and member.__reduce__()[1][0] == i,
                         'source field is not the original tuple slot')
            fields = attr(constructor_mro, '_fields')
            declared_new = attr(constructor_mro, '__new__')
            new = declared_new.__func__ if type(declared_new) is staticmethod else declared_new
            _require(type(fields) is tuple and fields == tuple(self.plan['record_fields'])
                     and type(new) is FunctionType and new.__code__ == self.expected_new
                     and new.__defaults__ is None and new.__kwdefaults__ is None
                     and new.__globals__.get('_tuple_new') is tuple.__new__,
                     'record construction differs from exact namedtuple')
            factory = attr(actor_mro, self.factory_name)
            scorer = attr(actor_mro, self.scorer_name)
            _require(type(factory) is FunctionType and type(scorer) is FunctionType
                     and factory.__code__ is source_code
                     and describe_code(source_code) == self.plan['factory']
                     and describe_code(scope_code) == self.plan['record_source']
                     and describe_code(scorer.__code__) == self.plan['score']
                     and factory_frame.f_globals is namespace and factory_frame.f_builtins is builtin_scope
                     and factory.__globals__ is namespace and scorer.__globals__ is namespace
                     and factory.__builtins__ is builtin_scope and scorer.__builtins__ is builtin_scope
                     and scorer.__closure__ is None and factory.__closure__ is None
                     and scorer.__defaults__ is None and factory.__defaults__ is None
                     and scorer.__kwdefaults__ is None and factory.__kwdefaults__ is None,
                     'original factory or scorer dispatch differs')
        except ValueError:
            return False
        if self.epoch is not None:
            if any(self.epoch.watch_dict(root) != 0 for root in (namespace, builtin_scope, new.__globals__)):
                return False
            for _, hierarchy in bindings.hierarchies:
                if any(self.epoch.watch_type(base) != 0 for base in hierarchy):
                    return False
        self.binding = (actor, cls, namespace, builtin_scope, source_code, scope_code,
                        bindings, factory, scorer, scorer.__code__, new, new.__code__)
        return True

    def _version(self, namespace):
        root = namespace.get(self.plan['table'])
        binding = self.table_binding
        if binding is not None and _same_rows(root, binding[1]):
            return binding[0]
        for slot, expected in enumerate(self.snapshots):
            rows = _freeze_rows(root, expected)
            if rows is not None:
                self.table_binding = slot, rows
                return slot
        return None

    def _functions(self, binding):
        _, _, _, _, source_code, _, bindings, factory, scorer, score_code, new, new_code = binding
        return (factory.__code__ is source_code and scorer.__code__ is score_code and new.__code__ is new_code
                and factory.__defaults__ is None and scorer.__defaults__ is None and new.__defaults__ is None
                and factory.__kwdefaults__ is None and scorer.__kwdefaults__ is None and new.__kwdefaults__ is None
                and all(cls.__mro__ is original for cls, original in bindings.hierarchies))

    def _context(self, binding):
        _, cls, namespace, builtin_scope, _, _, bindings, _, _, _, new, _ = binding
        if (not self._functions(binding) or not bindings.valid()
                or not _namespace_exact(namespace) or not _namespace_exact(builtin_scope)
                or namespace.get(self.plan['constructor']) is not cls
                or new.__globals__.get('_tuple_new') is not tuple.__new__
                or namespace.get('enumerate', builtin_scope.get('enumerate')) is not _ENUMERATE
                or namespace.get('abs', builtin_scope.get('abs')) is not _ABS
                or any(namespace.get(name) is not value for name, value in self.expected_imports.items())
                or any(type(namespace.get(name)) is not int or namespace[name] != value
                       for name, value in self.constants.items())):
            return False
        for name, expected in self.maps.items():
            root = namespace.get(name)
            anchor = self.live_maps.get(name)
            if anchor is None or not _same_rows(root, anchor):
                anchor = _freeze_rows(root, expected)
                if anchor is None:
                    return False
                self.live_maps[name] = anchor
        return True

    def _epoch_slot(self, namespace):
        stamp = self.epoch.value()
        if stamp == _MAX_STAMP or not self._functions(self.binding):
            return None, None
        cached = self.stamp_slot
        if cached is not None and cached[0] == stamp:
            return cached
        if not _namespace_exact(namespace):
            return None, None
        roots = tuple(namespace.get(name) for name in (*self.maps, self.plan['table']))
        if any(type(root) is not dict or self.epoch.watch_dict(root) != 0 for root in roots):
            return None, None
        if not self._context(self.binding):
            return None, None
        slot = self._version(namespace)
        if slot is None or self.epoch.value() != stamp:
            return None, None
        self.stamp_slot = stamp, slot
        if self.observe:
            self.counts['validated_mutation_epochs'] += 1
        return self.stamp_slot

    def _pack(self, receiver):
        for kind, index, bit, slot, address, capacity in self.input_ports:
            value = tuple.__getitem__(receiver, index)
            if bit is not None:
                if type(value) is not tuple or len(value) != 2:
                    return False
                value = value[bit]
            if kind == 'int' and _int64(value):
                self.scalars[slot] = value
            elif kind == 'bool' and type(value) is bool:
                self.scalars[slot] = value
            elif kind == 'str' and type(value) is str and value.isascii() and '\0' not in value and len(value) <= capacity:
                self.lengths[slot] = len(value)
                if value:
                    ffi.memmove(address, value.encode('ascii'), len(value))
            else:
                return False
        return True

    def _decode(self, count, cls):
        def value(fields, port):
            slot = port['slot']
            if port['type'] == 'int':
                return fields[slot]
            length = fields[self.length_base + slot]
            offset = port['offset']
            _require(length <= port['capacity'], 'native record outgrew its scalar layout')
            return fields[-1][offset:offset + length].decode('ascii')

        rows = []
        for i in range(count):
            fields = self.record_unpack.unpack_from(self.outputs, i * self.record_stride)
            entries = [value(fields, port) for port in self.output_layout['ports']]
            rows.append((entries[0], tuple.__new__(cls, entries[1:])))
        return rows

    def __call__(self, target, iterator, *, reverse):
        if not self.lock.acquire(blocking=False):
            return self._fallback(target, iterator, reverse, 'overlapping_offer')
        try:
            return self._evaluate(target, iterator, reverse)
        finally:
            self.lock.release()

    def _evaluate(self, target, iterator, reverse, *, publish=None):
        started = perf_counter_ns() if self.observe else 0
        if target is not _SORTED or reverse is not self.reverse or not _unstarted(iterator) or _observer_active():
            return self._fallback(target, iterator, reverse, 'original_sort_or_observer')
        frame = iterator.gi_frame
        locals_ = frame.f_locals
        generator = locals_.get('.0')
        receiver = locals_.get(self.plan['receiver'], _MISSING)
        if (not _unstarted(generator) or receiver is _MISSING
                or generator.gi_frame.f_locals.get(self.source_name) is not receiver
                or generator.gi_frame.f_globals is not frame.f_globals):
            return self._fallback(target, iterator, reverse, 'original_factory_iterator')
        if self.binding is None and not self._bind(receiver, generator.gi_frame, frame, generator.gi_code, iterator.gi_code):
            return self._fallback(target, iterator, reverse, 'initial_bindings')
        actor, cls, scope, builtins_scope, body, expression, *_ = self.binding
        if (type(receiver) is not actor or tuple.__len__(receiver) != len(self.fields)
                or frame.f_globals is not scope or frame.f_builtins is not builtins_scope
                or generator.gi_code is not body or iterator.gi_code is not expression):
            return self._fallback(target, iterator, reverse, 'changed_bindings')
        if self.epoch is None:
            stamp = None
            slot = self._version(scope) if self._context(self.binding) else None
        else:
            stamp, slot = self._epoch_slot(scope)
        if slot is None:
            return self._fallback(target, iterator, reverse, 'unsupported_source_context')
        predicate = self.plan['predicate']
        threshold = scope.get(predicate['threshold'], builtins_scope.get(predicate['threshold']))
        capture = locals_.get(predicate['nonzero_capture']) if predicate['nonzero_capture'] else 0
        if not _int64(threshold) or not _int64(capture):
            return self._fallback(target, iterator, reverse, 'unsupported_predicate')
        guarded = perf_counter_ns() if self.observe else 0
        if not self._pack(receiver):
            return self._fallback(target, iterator, reverse, 'unsupported_state')
        packed = perf_counter_ns() if self.observe else 0
        status = self.call_native(slot, self.inputs_pointer, threshold, capture, self.outputs,
                                  len(self.outputs), self.output_pointer)
        if status == 1 and len(self.outputs) < self.written.value <= _MAX_ROWS:
            self.outputs = (self.Record * self.written.value)()
            status = self.call_native(slot, self.inputs_pointer, threshold, capture, self.outputs,
                                      len(self.outputs), self.output_pointer)
        completed = perf_counter_ns() if self.observe else 0
        if status != 0 or self.written.value > len(self.outputs):
            return self._fallback(target, iterator, reverse, 'native_domain')
        rows = self._decode(self.written.value, cls)
        if self.epoch is None:
            unchanged = self._version(scope) == slot and self._context(self.binding)
        else:
            unchanged = self.epoch.value() == stamp and self._functions(self.binding)
        if not unchanged or scope.get(predicate['threshold'], builtins_scope.get(predicate['threshold'])) is not threshold:
            return self._fallback(target, iterator, reverse, 'changed_during_offer')
        if self.shadow:
            original = target(iterator, reverse=reverse)
            _require(rows == original, 'the original sort differs from its native source projection')
            self.counts['shadow_verified'] += 1
        if self.observe:
            self.phase_ns['guard_and_version'] += guarded - started
            self.phase_ns['input_pack'] += packed - guarded
            self.phase_ns['ffi_and_native'] += completed - packed
            self.phase_ns['decode_and_recheck'] += perf_counter_ns() - completed
            self.counts['native'] += 1
            self.versions[self.labels[slot]] += 1
            self.thresholds[str(threshold)] += 1
        return rows if publish is None else publish(rows, receiver, slot, stamp)


def run_native(entry, argv, manifest, digest, guard_path, guard_digest, *, epoch_path=None,
               epoch_digest=None, observe=False, shadow=False):
    from scar.analysis.scalar_sorted_source_guard import SCHEMA as GUARD
    from scar.codegen.scalar_sorted_library import SCHEMA as LIBRARY

    _require(sys.implementation.name == 'cpython' and getattr(sys, '_is_gil_enabled', lambda: True)()
             and not _observer_active(), 'isolated study requires CPython with a GIL and inactive observers')
    reference = artifact_reference(Path(manifest).resolve(strict=True), LIBRARY)
    _require(reference['sha256'] == digest, 'sorted native receipt differs from the admitted study')
    artifact = read_artifact(reference)
    package = Path(reference['uri']).parent
    library = artifact['library']
    binary = (package / library['uri']).resolve(strict=True)
    _require(binary.is_relative_to(package), 'native library must be an exact package member')
    read_artifact({**library,'uri':str(binary)}, json_document=False)
    plan = read_artifact(artifact['region'])
    guard_ref = artifact_reference(Path(guard_path).resolve(strict=True), GUARD)
    _require(guard_ref['sha256'] == guard_digest, 'source guard receipt differs from the admitted study')
    guard = read_artifact(guard_ref)
    _require(str(entry) == plan['source']['uri'] and guard['source'] == plan['source']
             and guard['plan'] == artifact['region'] and guard['creation']['sort_call'] == plan['sorted_site']
             and guard['qualifications']['complete_effect_closure'] is False
             and artifact['abi']['table_slots'] == guard['bindings']['score_table']['observed_versions'],
             'C++ region and guard must reference the same original source, call and table versions')
    transition = read_artifact(plan['transition'])
    scorer = read_artifact(plan['scorer'])
    evidence = read_artifact(scorer['evidence'])
    recipe = read_artifact(evidence['proof'])['cases'][evidence['proof_case']]['compile_recipe']
    raw = read_artifact(plan['source'], json_document=False)
    leaf = _seal(plan, raw, recipe, snapshot_inputs=False)
    score_globals = read_artifact(evidence['capture'])['globals']
    _require({**plan['scalar_globals'], **score_globals} == guard['bindings']['immutable_i64'],
             'live scalar context does not cover all original guard reads')
    epoch_ref, scope = None, nullcontext()
    if epoch_path is not None:
        from scar.runtime.sorted_study_epoch import load

        epoch_ref, scope = load(epoch_path, epoch_digest)
    with scope as epoch:
        offer = _Offer(binary, artifact, plan, transition, score_globals,
                       epoch=epoch, observe=observe or shadow, shadow=shadow)
        execution = _run_main(entry, argv, dict(compile_recipe=recipe, original_code_key=plan['sorted_owner']['key']),
                              raw, leaf, offer)
    return dict(schema=SCHEMA, schema_version=1, source=plan['source'], library=reference, guard=guard_ref,
                epoch_helper=epoch_ref,
                original_site=plan['sorted_site'], execution=execution,
                statistics=dict(offer.counts) if offer.observe else None,
                table_calls=dict(offer.versions) if offer.observe else None,
                threshold_calls=dict(offer.thresholds) if offer.observe else None,
                phase_ns=dict(offer.phase_ns) if offer.observe and not shadow else None,
                shadow=shadow, admitted_for_deployment=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path)
    parser.add_argument('--sha256')
    parser.add_argument('--source-guard', type=Path)
    parser.add_argument('--guard-sha256')
    parser.add_argument('--epoch-helper', type=Path)
    parser.add_argument('--epoch-sha256')
    parser.add_argument('--isolated-study', action='store_true')
    parser.add_argument('--observe', action='store_true')
    parser.add_argument('--shadow', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    native_inputs = (args.library, args.sha256, args.source_guard, args.guard_sha256)
    if not command or (args.isolated_study and not all(native_inputs)) or (not args.isolated_study and any(native_inputs)):
        parser.error('provide an original script, or a pinned C++ library and source guard with --isolated-study')
    if bool(args.epoch_helper) != bool(args.epoch_sha256) or (args.epoch_helper and not args.isolated_study):
        parser.error('provide both epoch helper and SHA-256 in the isolated study')
    if (args.observe or args.shadow) and not args.isolated_study:
        parser.error('--observe and --shadow require the isolated study')
    entry = Path(command[0]).resolve(strict=True)
    if args.library:
        report = run_native(entry, command[1:], *native_inputs, epoch_path=args.epoch_helper,
                            epoch_digest=args.epoch_sha256, observe=args.observe, shadow=args.shadow)
    else:
        report = dict(schema=SCHEMA, schema_version=1, library=None, admitted_for_deployment=False,
                      execution=run_original_script(entry, command[1:]))
    with args.report.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()

"""Exercise a shared native ranking and lazy transition at the original two sites."""

import argparse
import ast
from collections import Counter, namedtuple
import ctypes as ffi
import json
from pathlib import Path
from struct import Struct
import sys
from threading import get_ident
from time import perf_counter_ns
from types import FunctionType, GeneratorType, MethodType

from scar.analysis import scalar_score_reuse_evidence as evidence
from scar.codegen.scalar_transition_resident import resident_types
from scar.compiler.scalar_table_region import _definition, code_tree
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.scalar_score_ir import _int64, _require
from scar.runtime.csr_region_entry import _observer_active
from scar.runtime.scalar_sorted_observe import _seal
from scar.runtime.scalar_sorted_study import _MISSING, _Offer, _TypeBindings
from scar.runtime.scalar_table_contract import OFFER, compile_original, replace_leaf
from scar.runtime.scalar_table_entry import _run_main
from scar.runtime.sorted_study_epoch import load as load_epoch
from scar.trace.code_snapshot import describe_code, site


SCHEMA = 'scar.scalar-sorted-transition-source-study'


def _seal_sites(raw, recipe, sorted_plan, transition_evidence):
    """Splice two source calls without changing their receiver evaluation order."""
    sorted_code = _seal(sorted_plan, raw, recipe, snapshot_inputs=False)
    original_root = compile_original(raw, recipe)
    caller = transition_evidence['caller']['code']
    transition_site = transition_evidence['original_calls']['producer']['site']
    original = evidence._one((code for code in code_tree(original_root) if describe_code(code) == caller),
                             'one original transition owner is required')
    _require(site(original, transition_site['offset']) == transition_site and transition_site['opname'] == 'CALL',
             'transition must keep its exact original caller site')
    tree = ast.parse(raw, filename=recipe['filename'])
    function = _definition(tree, original.co_qualname)
    target = evidence._one((node for node in ast.walk(function) if type(node) is ast.Call and
                            evidence._span(node) == transition_site['position']), 'transition source call is ambiguous')
    _require(type(target.func) is ast.Attribute and len(target.args) == 1 and not target.keywords,
             'transition call must have its original bound callable and one input')
    loop = evidence._one((node for node in ast.walk(function) if type(node) is ast.For
                         and any(child is target for child in ast.walk(node))),
                         'one original consuming loop is required')
    fields = evidence._tuple_names(loop.target)
    _require(fields is not None and evidence._name(target.args[0]) == fields[1],
             'transition no longer consumes the current unpacked move')
    target.args = [target.func, *target.args, ast.Name(id=fields[0], ctx=ast.Load())]
    target.func = ast.Name(id=OFFER, ctx=ast.Load())
    revised = compile(ast.fix_missing_locations(tree), recipe['filename'], 'exec',
                      dont_inherit=True, optimize=recipe['optimize'])
    chosen = evidence._one((code for code in code_tree(revised) if code.co_qualname == original.co_qualname
                           and code.co_firstlineno == original.co_firstlineno), 'transition owner was lost')
    abi = ('co_flags', 'co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_varnames', 'co_cellvars', 'co_freevars')
    _require(all(getattr(chosen, name) == getattr(original, name) for name in abi),
             'fused region changed source function parameters or closure')
    result = replace_leaf(chosen, sorted_plan['sorted_owner']['key'], sorted_code)
    return caller['key'], result, sorted_code


class _Lease:
    __slots__ = ('receiver', 'slot', 'stamp', 'row', 'used', 'state', 'source')

    def __init__(self, receiver, slot, stamp):
        self.receiver = receiver
        self.slot = slot
        self.stamp = stamp
        self.row = None
        self.used = False
        self.state = None
        self.source = None


def _read_values(layout, structure):
    # The C ABI puts scalars before string sizes and payload bytes.
    count, sizes, strings = (layout[key] for key in ('scalars','lengths','strings'))
    wire = Struct(f'={count}q{sizes}I{strings}s')
    _require(structure.lengths.offset == 8 * count and structure.strings.offset == 8 * count + 4 * sizes
             and ffi.sizeof(structure) >= wire.size, 'transition state ABI disagrees with host representation')
    return wire, {
        port['name']: (port['type'], port['slot'] if port['type'] != 'str' else count + port['slot'],
                       port.get('offset', 0), port.get('capacity', 0))
        for port in layout['ports']
    }


class _Pods:
    def __init__(self, library, transition_abi, Record, fields, state_fields):
        self.abi = transition_abi
        _, self.State, self.Token = resident_types(transition_abi)
        self.call = library.scar_scalar_lazy_transition_v1
        self.call.argtypes = (ffi.c_uint32, ffi.POINTER(self.State), ffi.POINTER(Record),
                              ffi.POINTER(self.Token), ffi.POINTER(self.State))
        self.call.restype = ffi.c_int
        self.layout_state = transition_abi['state']
        self.layout_input = transition_abi['input']
        self.record_ports = transition_abi['move']['ports']
        self.state_writes = self._writes(self.layout_state, fields, state_fields)
        self.input_writes = self._writes(self.layout_input, fields, state_fields)
        self.move_writes = self._writes(self.layout_input, fields, state_fields, move=True)
        self.output = self.State()
        self.move = Record()
        self.output_wire, outputs = _read_values(self.layout_state, self.State)
        self.result_fields = []
        for name in fields:
            base = 'field_' + name
            names = [base+'__0',base+'__1'] if state_fields[name]=='bool_pair' else [base]
            self.result_fields.append([outputs[key] for key in names])

    @staticmethod
    def _writes(layout, fields, state_fields, *, move=False):
        slots = {}
        for index, name in enumerate(fields):
            base = 'field_' + name
            if state_fields[name] == 'bool_pair':
                for bit in range(2):
                    slots[f'{base}__{bit}'] = index, bit
            else:
                slots[base] = index, None
        return [(port, *((int(port['name'].removeprefix('arg_')), None) if move else slots[port['name']]))
                for port in layout['ports'] if port['name'].startswith('arg_') == move]

    @staticmethod
    def _store(target, port, value):
        kind, slot = port['type'], port['slot']
        if kind=='int' and _int64(value) or kind=='bool' and type(value) is bool:
            target.scalars[slot] = value
            return True
        if (kind=='str' and type(value) is str and value.isascii()
                and '\0' not in value and len(value) <= port['capacity']):
            target.lengths[slot] = len(value)
            if value:
                ffi.memmove(ffi.addressof(target) + type(target).strings.offset + port['offset'],
                            value.encode('ascii'), len(value))
            return True
        return False

    def _populate(self, output, values, writes):
        for port, index, bit in writes:
            leaf = tuple.__getitem__(values, index)
            if bit is not None:
                if type(leaf) is not tuple or len(leaf) != 2:
                    return False
                leaf = leaf[bit]
            if not self._store(output, port, leaf):
                return False
        return True

    def _record(self, move, score):
        for index, port in enumerate(self.record_ports):
            leaf = score if index == 0 else tuple.__getitem__(move, index - 1)
            if not self._store(self.move, port, leaf):
                return False
        return True

    def invoke(self, lease, move, cls):
        score = lease.row[0]
        if lease.state is None:
            state, token = self.State(), self.Token()
            if not self._populate(state, lease.receiver, self.state_writes) or not self._populate(
                    token.source_inputs, lease.receiver, self.input_writes):
                return None
            token.receipt_tag = self.abi['receipt_tag']
            token.table_slot = lease.slot
            lease.state, lease.source = state, token
        token = lease.source
        if not self._record(move, score) or not self._populate(token.source_inputs, move, self.move_writes):
            return None
        token.delta = score
        status = self.call(lease.slot, ffi.byref(lease.state), ffi.byref(self.move),
                           ffi.byref(token), ffi.byref(self.output))
        if status not in (0, 1):
            return None
        slots = self.output_wire.unpack_from(self.output)
        values = []
        for ports in self.result_fields:
            parts = []
            for kind, offset, start, capacity in ports:
                value = slots[offset]
                if kind == 'str':
                    _require(value <= capacity, 'native string outgrew its admitted state shape')
                    value = slots[-1][start:start + value].decode('ascii')
                elif kind == 'bool':
                    _require(value in (0, 1), 'native boolean result was not exact')
                    value = bool(value)
                parts.append(value)
            values.append(tuple(parts) if len(parts) != 1 else parts[0])
        return status, tuple.__new__(cls, values)


class _FusedOffer(_Offer):
    def __init__(self, binary, artifact, plan, transition, transition_evidence, constructor, score_globals,
                 owner_code, sort_code, *, epoch, observe, shadow):
        super().__init__(binary, {'abi':artifact['abi']['sort']}, plan, transition, score_globals,
                         epoch=epoch, observe=observe, shadow=shadow)
        self.pods = _Pods(ffi.PyDLL(str(binary)), artifact['abi']['transition'], self.Record, self.fields, self.types)
        self.owner_code = owner_code
        self.sort_code = sort_code
        self.leases = {}
        self.transition_counts = Counter()
        self.transition_ns = Counter()
        self.state_bindings = None
        self.state_constructor = constructor
        self.transition_roles = {role:transition_evidence[role]['code'] for role in ('callee','rotation')}
        self.expected_state_new = namedtuple('_StateStudy', self.fields).__new__.__code__

    def _carry(self, key, rows, receiver, slot, stamp):
        lease = _Lease(receiver, slot, stamp)
        self.leases[key] = lease
        try:
            for row in rows:
                lease.row = row
                lease.used = False
                yield row
        finally:
            if self.leases.get(key) is lease:
                del self.leases[key]

    def _publish(self, key, rows, receiver, slot, stamp):
        if not rows or key in self.leases:
            return rows
        return self._carry(key, rows, receiver, slot, stamp)

    def _sort(self, owner, target, iterator, reverse):
        parent = owner.f_back
        if (owner.f_code is not self.sort_code or parent is None or parent.f_code is not self.owner_code
                or type(getattr(owner,'f_generator',None)) is not GeneratorType):
            return super().__call__(target, iterator, reverse=reverse)
        key = id(parent), get_ident()
        if not self.lock.acquire(blocking=False):
            return self._fallback(target, iterator, reverse, 'overlapping_offer')
        try:
            return self._evaluate(target, iterator, reverse,
                                  publish=lambda rows, receiver, slot, stamp:
                                      self._publish(key, rows, receiver, slot, stamp))
        finally:
            self.lock.release()

    def _bind_state(self, target, cls, scope):
        guard = _TypeBindings()
        mro = guard.watch(cls)
        try:
            _require(scope.get(self.state_constructor) is cls and
                     guard.attr(mro,'__init__') is tuple.__init__ and guard.attr(mro,'__del__') is _MISSING,
                     'state construction is not a closed tuple constructor')
            # Tuple comparison is exercised by shadow mode without dispatching custom code.
            _require(guard.attr(mro,'__eq__') is tuple.__eq__, 'state has a custom value comparison')
            constructor = guard.attr(mro, '__new__')
            new = constructor.__func__ if type(constructor) is staticmethod else constructor
            _require(type(new) is FunctionType and new.__code__ == self.expected_state_new
                     and new.__defaults__ is None and new.__kwdefaults__ is None
                     and new.__globals__.get('_tuple_new') is tuple.__new__,
                     'state no longer has the original immutable field layout')
            functions = []
            for role, code in self.transition_roles.items():
                name = code['qualname'].rsplit('.',1)[-1]
                method = guard.attr(mro,name)
                _require(type(method) is FunctionType and method.__globals__ is scope and method.__closure__ is None
                         and describe_code(method.__code__) == code and method.__kwdefaults__ is None
                         and (method.__defaults__ is None if role == 'callee' else method.__defaults__ == (False,))
                         and (role != 'callee' or method is target.__func__), 'transition or rotation dispatch changed')
                functions.append((method, method.__code__, method.__defaults__))
        except ValueError:
            return False
        if self.epoch.watch_dict(new.__globals__) != 0:
            return False
        self.state_bindings = guard, new, new.__code__, functions
        return True

    def _state_context(self, target, cls, scope, stamp):
        bindings = self.state_bindings
        if bindings is None:
            if not self._bind_state(target, cls, scope):
                return False
            bindings = self.state_bindings
        guard, new, code, functions = bindings
        return (self.epoch.value()==stamp and guard.valid() and target.__func__ is functions[0][0]
                and scope.get(self.state_constructor) is cls and new.__code__ is code
                and new.__defaults__ is None and new.__kwdefaults__ is None
                and new.__globals__.get('_tuple_new') is tuple.__new__
                and all(fn.__code__ is body and fn.__defaults__ is defaults and fn.__kwdefaults__ is None
                        for fn, body, defaults in functions))

    def _move_fallback(self, target, argument, reason):
        if self.observe:
            self.transition_counts[reason] += 1
        return target(argument)

    def _transition(self, owner, target, argument, score):
        started = perf_counter_ns() if self.observe else 0
        lease = self.leases.get((id(owner), get_ident()))
        if lease is None or lease.row is None or owner.f_code is not self.owner_code or lease.used:
            return self._move_fallback(target, argument, 'no_ordered_producer')
        if (type(target) is not MethodType or target.__self__ is not lease.receiver
                or lease.row[1] is not argument or lease.row[0] is not score):
            return self._move_fallback(target, argument, 'producer_identity_changed')
        lease.used = True
        if not self.lock.acquire(blocking=False):
            return self._move_fallback(target, argument, 'overlapping_offer')
        try:
            result = self._consume(target, argument, lease, started)
        finally:
            self.lock.release()
        return result

    def _consume(self, target, argument, lease, started):
        actor, _, namespace, *_ = self.binding
        if (_observer_active() or not self._functions(self.binding)
                or not self._state_context(target, actor, namespace, lease.stamp)):
            return self._move_fallback(target, argument, 'producer_epoch_or_binding_changed')
        guarded = perf_counter_ns() if self.observe else 0
        result = self.pods.invoke(lease, argument, actor)
        if result is None:
            return self._move_fallback(target, argument, 'native_domain')
        status, value = result
        if not self._functions(self.binding) or not self._state_context(target, actor, namespace, lease.stamp):
            return self._move_fallback(target, argument, 'producer_epoch_or_binding_changed')
        if self.shadow:
            original = target(argument)
            _require(type(original) is actor and value == original, 'lazy transition differs from original source result')
            self.transition_counts['shadow_verified'] += 1
        if self.observe:
            self.transition_counts['transported' if status == 1 else 'native_recompute'] += 1
            self.transition_ns['guard'] += guarded - started
            self.transition_ns['pod_native_decode_and_recheck'] += perf_counter_ns() - guarded
        return value

    def __call__(self, target, argument, score=_MISSING, *, reverse=_MISSING):
        owner = sys._getframe(1)
        if reverse is _MISSING:
            return self._transition(owner, target, argument, score)
        return self._sort(owner, target, argument, reverse)


def _pinned(path, schema, digest):
    reference = artifact_reference(Path(path).resolve(strict=True), schema)
    _require(reference['sha256'] == digest, 'source trial input differs from its pinned receipt')
    return reference, read_artifact(reference)


def run(entry, argv, options):
    from scar.analysis.scalar_sorted_transition_provenance import SCHEMA as FLOW
    from scar.codegen.scalar_sorted_transition_library import SCHEMA as LIBRARY

    _require(sys.implementation.name == 'cpython' and getattr(sys,'_is_gil_enabled',lambda:True)()
             and not _observer_active(), 'fusion study needs CPython with a GIL and inactive observers')
    library_ref, package = _pinned(options.library, LIBRARY, options.sha256)
    flow_ref, flow = _pinned(options.provenance, FLOW, options.provenance_sha256)
    epoch_ref, epoch = load_epoch(options.epoch_helper, options.epoch_sha256)
    guard = read_artifact(flow['source_guard'])
    plan = read_artifact(guard['plan'])
    _require(str(entry) == plan['source']['uri'] == flow['source']['uri']
             and plan['source'] == flow['source'] == guard['source'], 'fused study must launch the original source')
    transport = read_artifact(flow['transport'])
    transition = read_artifact(transport['transition_plan'])
    sorted_package = read_artifact(package['sorted'])
    resident_package = read_artifact(package['resident'])
    _require(guard['plan']==sorted_package['region'] and flow['transport']==resident_package['transport']
             and transport['transition_plan']==resident_package['plan']
             and transition['source']==flow['source'], 'fusion library does not match the exact source and transport evidence')
    root = Path(library_ref['uri']).parent
    binary = (root/package['library']['uri']).resolve(strict=True)
    _require(binary.is_relative_to(root), 'fused native library must remain a local package member')
    read_artifact({**package['library'],'uri':str(binary)}, json_document=False)
    transition_evidence = read_artifact(transition['evidence'])
    capture = read_artifact(transition_evidence['capture'])
    selected = read_artifact(capture['selection'])
    scorer = read_artifact(plan['scorer'])
    score_evidence = read_artifact(scorer['evidence'])
    score_capture = read_artifact(score_evidence['capture'])
    recipe = read_artifact(score_evidence['proof'])['cases'][score_evidence['proof_case']]['compile_recipe']
    raw = read_artifact(plan['source'], json_document=False)
    original_key, owner_code, sort_code = _seal_sites(raw, recipe, plan, transition_evidence)
    _require({**plan['scalar_globals'], **score_capture['globals']} == guard['bindings']['immutable_i64'],
             'fused sort source bindings differ from guard closure')
    with epoch:
        offer = _FusedOffer(binary, package, plan, transition, transition_evidence, selected['constructor']['name'],
                            capture['globals'], owner_code, sort_code, epoch=epoch,
                            observe=options.observe or options.shadow, shadow=options.shadow)
        execution = _run_main(entry,argv,dict(compile_recipe=recipe,original_code_key=original_key),raw,owner_code,offer)
        _require(not offer.leases, 'a sorted result lease outlived its source scope')
    return dict(schema=SCHEMA, schema_version=1, source=plan['source'], library=library_ref,
                source_guard=flow['source_guard'], provenance=flow_ref, epoch_helper=epoch_ref,
                execution=execution, shadow=options.shadow, admitted_for_deployment=False,
                sort_statistics=dict(offer.counts) if offer.observe else None,
                sort_phase_ns=dict(offer.phase_ns) if offer.observe and not options.shadow else None,
                table_calls=dict(offer.versions) if offer.observe else None,
                transition_statistics=dict(offer.transition_counts) if offer.observe else None,
                transition_phase_ns=dict(offer.transition_ns) if offer.observe and not options.shadow else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--provenance-sha256', required=True)
    parser.add_argument('--epoch-helper', type=Path, required=True)
    parser.add_argument('--epoch-sha256', required=True)
    parser.add_argument('--isolated-study', action='store_true', required=True)
    parser.add_argument('--observe', action='store_true')
    parser.add_argument('--shadow', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('provide the original source script')
    report = run(Path(command[0]).resolve(strict=True), command[1:], args)
    with args.report.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()

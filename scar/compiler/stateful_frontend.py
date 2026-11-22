"""Discover ordered array-state updates from original F4 plans and UnitFacts.

No source AST decides expressions or bindings. Retained F2 supplies scalar
selectors/coordinates; compile-only F0 recovery supplies complete CodeIDs.
"""
from collections import defaultdict
from dataclasses import dataclass, replace
from hashlib import sha256
import math
from pathlib import Path

from scar.analysis.program_view import ProgramView
from scar.analysis.source_body_interfaces import BodyInterfaceResolver
from scar.analysis.source_references import source_port
from scar.compiler.graph_frontend import (_GraphLowerer, _Array, _Scalar, _Slice,
    _WORLD, _ENV, _OPS, _origin, _reject, _metadata, _source_identity, GraphRegionRejected)
from scar.compiler.model import ArrayRegion, Expr, SourceRef
from scar.compiler.stateful_model import (StatefulRegion, StatefulInput, StatefulAction,
    StatefulCallee, StatefulProvider, validate_stateful_region, REQUIRED_GUARDS, POWER2_SEMANTICS)
from scar.ir.program import source_productions as P, normal_graph as F, normal_graph as G
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.normal_graph import Body
from scar.ir.record_codec import encode
from scar.trace.code_snapshot import code_key


@dataclass(frozen=True)
class _Path:
    path: tuple[str, ...]
    origin: dict


@dataclass(frozen=True)
class _View:
    base: _Path
    bounds: tuple
    origin: dict


@dataclass(frozen=True)
class _Value:
    node: str
    kind: str


@dataclass(frozen=True)
class _Module:
    module: str
    path: tuple[str, ...]


@dataclass(frozen=True)
class _Provider:
    module: str
    symbol: str
    path: tuple[str, ...]


@dataclass(frozen=True)
class _Method:
    path: tuple[str, ...]
    function: object
    region: ArrayRegion


def _reachable(facts, start):
    edges = defaultdict(list)
    for left, right in facts.aliases: edges[left].append(right)
    seen, pending = set(), [start]
    while pending:
        value = pending.pop()
        if value not in seen:
            seen.add(value); pending.extend(edges[value])
    return seen


def _declarations(original, resolver):
    """Bind imports and class members through original reference transports."""
    facts = original.facts
    imports, members = defaultdict(list), defaultdict(list)
    indices = {source_port(facts, i).port: i for i in range(len(facts.ports))}
    for imported in facts.imports:
        reachable = _reachable(facts, imported.value)
        for name in facts.names:
            if name.write and name.namespace.kind is F.NamespaceKind.MODULE and name.value in reachable:
                imports[name.namespace.name].append(imported.requested_module or imported.module)
    for function in facts.functions:
        starts = [function.value]
        context = resolver.plan(facts.unit, function.template)
        for step in context.plan.steps:
            if type(step.instruction) is P.ApplyDecorator:
                index = indices.get(context.result(step).port)
                if index is not None: starts.append(index)
        reachable = set().union(*(_reachable(facts, value) for value in starts))
        for name in facts.names:
            if name.write and name.namespace.kind is F.NamespaceKind.CLASS and name.value in reachable:
                members[function.template].append((name.namespace.owner, name.namespace.name))
    return imports, members


class _StateLowerer(_GraphLowerer):
    def __init__(self, original, function, resolver, syntax, imports, members, leaves,
                 *, array_paths=()):
        super().__init__(original, function, resolver, syntax=syntax)
        self.imports, self.members, self.leaves = imports, members, leaves
        self.array_paths = frozenset(array_paths)
        self.actions, self.inputs, self.epochs, self.callees, self.providers = [], {}, {}, [], []
        self.call_facts = {source_port(self.facts, row.result).port: row for row in self.facts.calls}
        self.attributes = {source_port(self.facts, row.value).port: row for row in self.facts.attributes
                           if row.action is F.AttributeAction.GET}
        owners = {owner for owner, _ in members.get(function.template, ())}
        self.class_owner = next(iter(owners)) if len(owners) == 1 else None

    def emit_value(self, op, origin, inputs=(), attrs=None):
        node = Expr('n' + str(len(self.nodes)), op, tuple(inputs), {} if attrs is None else attrs)
        self.nodes.append(node); self.by_id[node.id] = node
        self.origins.append(dict(origin, expression=node.id))
        kind = ('scalar' if op == 'const' else node.attrs['kind'] if op == 'input' else
                'array' if op in ('slice', 'power2') else
                'array' if any(self.kind(i) == 'array' for i in node.inputs) else 'scalar')
        if op not in ('input', 'const', 'slice'):
            self.actions.append(StatefulAction('a' + str(len(self.actions)), 'evaluate', node.id,
                world_before=origin.get('world_before'), world_after=origin.get('world_after'), origins=(origin,)))
        return _Value(node.id, kind)

    def kind(self, identity):
        node = self.by_id[identity]
        if node.op == 'input': return node.attrs['kind']
        if node.op == 'const': return 'scalar'
        if node.op in ('slice', 'power2'): return 'array'
        return 'array' if any(self.kind(i) == 'array' for i in node.inputs) else 'scalar'

    def input(self, value, kind):
        path = value.path
        row = self.inputs.get(path)
        if row is None:
            row = StatefulInput('input:' + str(len(self.inputs)), path, kind, (value.origin,))
            self.inputs[path] = row
            if kind == 'array': self.epochs[row.id] = 0
        if row.kind != kind:
            _reject('STATE_INPUT_KIND_CONFLICT', 'one input path cannot be both an array and scalar', value.origin)
        return row

    def numeric(self, value, *, array=False):
        if isinstance(value, _Value):
            if array and value.kind != 'array': _reject('STATE_ARRAY_OPERAND', 'array operand required')
            return value
        if isinstance(value, _Scalar):
            numeric = value.value
            if (type(numeric) not in (int, float) or not math.isfinite(numeric)
                    or type(numeric) is int and abs(numeric) > 2**53):
                _reject('STATE_NUMERIC_CONSTANT', 'finite exactly representable numeric operand required', value.origin)
            return self.emit_value('const', value.origin, attrs={'value': numeric})
        if isinstance(value, _View):
            base = self.numeric(value.base, array=True)
            return self.emit_value('slice', value.origin, (base.node,),
                dict(bounds=[list(axis) for axis in value.bounds], semantics='ALIAS_DESCRIPTOR'))
        if isinstance(value, _Path):
            row = self.input(value, 'array' if array or value.path in self.array_paths
                or value.path in self.inputs and self.inputs[value.path].kind == 'array' else 'scalar')
            epoch = self.epochs[row.id] if row.kind == 'array' else None
            return self.emit_value('input', dict(value.origin, storage_epoch=epoch),
                attrs=dict(name=row.id, kind=row.kind, epoch=epoch))
        _reject('STATE_NONNUMERIC_OPERAND', 'operand lacks an admitted array/scalar input or immutable computation')

    def binding(self, context, step, args):
        action = step.instruction.action
        if action not in (P.BindingActionKind.READ, P.BindingActionKind.WRITE):
            _reject('STATE_BINDING_DELETE', 'local deletion is outside this profile')
        write = action is P.BindingActionKind.WRITE
        port = context.value(step.arguments[0]).port if write else context.result(step).port
        namespaces = tuple(dict.fromkeys(self.names.get((port, write), ())))
        if namespaces and all(ns.kind is F.NamespaceKind.LEXICAL for ns in namespaces):
            if write:
                for namespace in namespaces: self.locals[namespace] = args[0]
                return (_WORLD,)
            if len(namespaces) != 1 or namespaces[0] not in self.locals:
                _reject('STATE_UNBOUND_LOCAL', 'local read lacks an original parameter/assignment binding')
            return (self.locals[namespaces[0]], _WORLD)
        if (write or not namespaces or any(ns.kind not in (F.NamespaceKind.MODULE, F.NamespaceKind.BUILTINS) for ns in namespaces)
                or len({ns.name for ns in namespaces}) != 1):
            _reject('STATE_EXTERNAL_BINDING', 'only original local bindings and admitted module imports are supported')
        name = namespaces[0].name
        modules = set(self.imports.get(name, ()))
        if len(modules) != 1 or not any(ns.kind is F.NamespaceKind.MODULE for ns in namespaces):
            _reject('STATE_UNKNOWN_GLOBAL', 'global binding has no unique original import declaration')
        return (_Module(next(iter(modules)), (name,)), _WORLD)

    def _write(self, target, value, operation, origin):
        if isinstance(target, _Path):
            base, bounds = target, None
        elif isinstance(target, _View):
            base, bounds = target.base, target.bounds
        else:
            _reject('STATE_WRITE_TARGET', 'write must address an original input array or fixed slice alias', origin)
        row = self.input(base, 'array'); rhs = self.numeric(value)
        before = self.epochs[row.id]
        self.actions.append(StatefulAction('a' + str(len(self.actions)), 'write', rhs.node, row.id,
            bounds, before, before + 1, operation, origin.get('world_before'), origin.get('world_after'), (origin,)))
        self.epochs[row.id] = before + 1
        return target

    def inplace(self, context, step, args, origin):
        operation = self.operations.get(context.result(step).port)
        if (operation is None or operation.action.value != 'INPLACE_BINARY'
                or operation.selector.value not in ('Add', 'Sub', 'Mult', 'Div') or len(args) != 3
                or tuple(source_port(self.facts, i).port for i in operation.operands) !=
                tuple(context.value(v).port for v in step.arguments[:-1])):
            _reject('STATE_INPLACE_OPERATOR', 'only original float64 inplace add/sub/mul/div supported', origin)
        return (self._write(args[0], args[1], _OPS[operation.selector.value], origin), _WORLD)

    def action(self, context, step, args, origin):
        ins = step.instruction
        if ins.action is P.LanguageActionKind.ATTRIBUTE_GET:
            fact = self.attributes.get(context.result(step).port)
            if fact is None or len(args) != 2 or source_port(self.facts, fact.receiver).port != context.value(step.arguments[0]).port:
                _reject('STATE_ATTRIBUTE_FACT', 'attribute read differs from original receiver/field ports', origin)
            receiver, name = args[0], fact.name
            if isinstance(receiver, _Module):
                return (_Provider(receiver.module, name, (*receiver.path, name)), _WORLD)
            if isinstance(receiver, _Path):
                if len(receiver.path) == 1 and self.class_owner is not None:
                    targets = [(fn, region) for fn, region in self.leaves.values()
                               if (self.class_owner, name) in self.members.get(fn.template, ())]
                    if len(targets) == 1:
                        fn, region = targets[0]
                        return (_Method((*receiver.path, name), fn, region), _WORLD)
                return (_Path((*receiver.path, name), origin), _WORLD)
            _reject('STATE_ATTRIBUTE_RECEIVER', 'attribute receiver is not an original parameter/field or module reference', origin)
        if ins.action is P.LanguageActionKind.ITEM_GET:
            if (len(args) != 3 or not isinstance(args[0], _Path) or type(args[1]) is not tuple
                    or len(args[1]) != 2 or any(not isinstance(v, _Slice) for v in args[1])):
                _reject('STATE_FIXED_SLICE', 'array view requires two original fixed unit-step slice selectors', origin)
            self.input(args[0], 'array')
            return (_View(args[0], tuple(axis.bounds for axis in args[1]), origin), _WORLD)
        if ins.action is P.LanguageActionKind.ITEM_SET:
            if (len(args) != 4 or not isinstance(args[0], _Path) or type(args[1]) is not tuple
                    or len(args[1]) != 2 or any(not isinstance(v, _Slice) for v in args[1])):
                _reject('STATE_FIXED_WRITE_SLICE', 'array write requires two original fixed slice selectors', origin)
            self._write(_View(args[0], tuple(axis.bounds for axis in args[1]), origin), args[2], 'assign', origin)
            return (_WORLD,)
        if ins.action in (P.LanguageActionKind.BINARY, P.LanguageActionKind.UNARY):
            operation = self.operations.get(context.result(step).port)
            if (operation is None or operation.action.value != ins.action.value or operation.selector.value not in _OPS
                    or tuple(source_port(self.facts, i).port for i in operation.operands) !=
                    tuple(context.value(v).port for v in step.arguments[:-1])):
                _reject('STATE_OPERATOR', 'unsupported original scalar/array operator', origin)
            values = args[:-1]; selector = operation.selector.value
            if selector == 'USub' and len(values) == 1 and isinstance(values[0], _Scalar):
                if type(values[0].value) not in (int, float):
                    _reject('STATE_UNARY_CONSTANT', 'unary minus requires a numeric constant', origin)
                return (_Scalar(-values[0].value, origin), _WORLD)
            operands = tuple(self.numeric(value).node for value in values)
            return (self.emit_value(_OPS[selector], origin, operands), _WORLD)
        if ins.action is P.LanguageActionKind.CALL:
            return (self.call(context, step, args, origin), _WORLD)
        _reject('STATE_UNSUPPORTED_ACTION', ins.action.value + ' has no closed stateful-array semantics', origin)

    def call(self, context, step, args, origin):
        fact = self.call_facts.get(context.result(step).port)
        if (fact is None or fact.positional_pack is not None or fact.keyword_pack is not None or fact.keywords):
            _reject('STATE_CALL_LAYOUT', 'only explicit positional calls with original argument ports supported', origin)
        expected = (source_port(self.facts, fact.callee).port,
            *(source_port(self.facts, i).port for i in fact.positional), source_port(self.facts, fact.world).port)
        if tuple(context.value(value).port for value in step.arguments) != expected:
            _reject('STATE_CALL_ABI', 'CALL operands differ from original UnitFacts', origin)
        normal = next(c for c in step.completions if c.kind is C.FALLTHROUGH)
        if context.result(step, len(normal.types) - 1).port != source_port(self.facts, fact.result_world).port:
            _reject('STATE_CALL_ABI', 'CALL post-world differs from original UnitFacts', origin)
        callee, values = args[0], args[1:-1]
        if isinstance(callee, _Provider):
            if (callee.module != 'numpy' or callee.symbol != 'power' or len(values) != 2
                    or not isinstance(values[1], _Scalar) or type(values[1].value) is not int or values[1].value != 2):
                _reject('STATE_PROVIDER_CALL', 'only independently guarded numpy.power(array, exact int 2) provider supported', origin)
            value = self.numeric(values[0], array=True)
            if value.kind != 'array': _reject('STATE_PROVIDER_ARRAY', 'power provider requires an array', origin)
            identity = 'numpy.power.float64.integer-two.v1'
            provider = StatefulProvider(identity, 'numpy', 'power', callee.path,
                POWER2_SEMANTICS, (origin,))
            existing = next((i for i, item in enumerate(self.providers) if item.id == identity), None)
            if existing is None: self.providers.append(provider)
            else:
                old = self.providers[existing]
                if old.lookup_path != provider.lookup_path:
                    _reject('STATE_PROVIDER_LOOKUP_ALTERNATIVES', 'one provider ID cannot select multiple global lookup paths', origin)
                self.providers[existing] = replace(old, origins=(*old.origins, origin))
            return self.emit_value('power2', origin, (value.node,), dict(provider=identity))
        if not isinstance(callee, _Method) or len(values) != len(callee.region.parameters):
            _reject('STATE_DYNAMIC_CALL', 'callee is not a uniquely bound same-source bare staticmethod array leaf', origin)
        guard = StatefulCallee(callee.region.source, callee.path, origins=(origin,))
        existing = next((i for i, item in enumerate(self.callees) if item.path == guard.path), None)
        if existing is None: self.callees.append(guard)
        else:
            old = self.callees[existing]
            if old.source != guard.source: _reject('STATE_CALLEE_IDENTITY_CONFLICT', 'one method guard path has multiple source identities', origin)
            self.callees[existing] = replace(old, origins=(*old.origins, origin))
        mapping = {}
        actuals = dict(zip(callee.region.parameters, values, strict=True))
        original_origins = defaultdict(list)
        for item in callee.region.source.origins:
            if item.get('expression'): original_origins[item['expression']].append(item)
        for node in callee.region.nodes:
            if node.op == 'input':
                mapping[node.id] = self.numeric(actuals[node.attrs['name']], array=True)
                if mapping[node.id].kind != 'array':
                    _reject('STATE_CALLEE_ARRAY_ARGUMENT', 'admitted staticmethod requires exact array arguments', origin)
                continue
            call_origin = dict((original_origins[node.id] or [origin])[0], callsite=origin)
            call_origin.update(world_before=origin.get('world_before'), world_after=origin.get('world_after'),
                               world_scope='CALL_BOUNDARY_SHARED_PURE_BODY_PROJECTION')
            operands = tuple(mapping[value].node for value in node.inputs)
            attrs = dict(node.attrs)
            if node.op == 'slice': attrs['semantics'] = 'ALIAS_DESCRIPTOR'
            mapping[node.id] = self.emit_value(node.op, call_origin, operands, attrs)
        return mapping[callee.region.output]

    def run(self, context, region_id, parameters):
        identity = context.key, region_id
        if identity in self.active: _reject('STATE_RECURSIVE_REGION', 'recursive control is unsupported')
        if context.owner(region_id).owner != self.function.body_scope.owner:
            _reject('STATE_CODE_SCOPE_CROSSING', 'body action crosses the original function owner')
        region = context.regions[region_id]
        if len(parameters) != len(region.parameters): _reject('STATE_REGION_ARITY', 'original region parameters differ')
        self.active.add(identity)
        values = {P.ProductionParameter(region_id, i): value for i, value in enumerate(parameters)}; evaluated = {}
        try:
            for step_id in region.steps:
                step = context.steps[step_id]; ins = step.instruction
                origin = _origin(context, region_id, step, arguments=[encode(context.value(v)) for v in step.arguments])
                try:
                    before, after = context.worlds(step)
                    origin.update(world_before=encode(before), world_after=encode(after))
                except (ValueError, StopIteration): pass
                self.origins.append(origin); args = tuple(values[value] for value in step.arguments)
                if type(ins) is P.TransferCompletion:
                    if ins.kind not in (C.FALLTHROUGH, C.RETURN):
                        _reject('STATE_COMPLETION', 'only normal None return/fallthrough supported', origin)
                    return ins.kind, args
                if type(ins) is P.EvaluateChild:
                    link = self.source.material.interfaces['links'][context.key, step.id]
                    child = self.resolver.plan(self.source.unit, link.callee)
                    kind, result = self.run(child, child.plan.root, args)
                    if kind is not C.FALLTHROUGH: return kind, result
                    if result: evaluated[ins.field] = result[0]
                elif type(ins) is P.ApplyRegion:
                    kind, result = self.run(context, ins.invocation.region, tuple(values[v] for v in ins.invocation.arguments))
                    if kind is not C.FALLTHROUGH: return kind, result
                elif type(ins) is P.HandleCompletions:
                    # Ordinary implicit None return is an original protected
                    # suite plus a real completion handler, never skipped.
                    # Admit only its exact two-step literal/return handler;
                    # user handlers and cleanup effects remain unsupported.
                    if (len(ins.routes) != 1 or ins.routes[0].kind is not C.FALLTHROUGH
                            or ins.routes[0].target is not None or ins.routes[0].captures):
                        _reject('STATE_COMPLETION_HANDLER', 'only original normal-fallthrough implicit None handler supported', origin)
                    route = ins.routes[0]; handler = context.regions[route.handler]
                    steps = tuple(context.steps[key] for key in handler.steps)
                    if (handler.kind is not P.ProductionRegionKind.HANDLER or len(handler.parameters) != 1
                            or len(steps) != 2 or type(steps[0].instruction) is not P.DefineLiteral
                            or steps[0].instruction.value.to_python() is not None
                            or type(steps[1].instruction) is not P.TransferCompletion
                            or steps[1].instruction.kind is not C.RETURN or steps[1].instruction.target is not None):
                        _reject('STATE_COMPLETION_HANDLER_EFFECT', 'handler is not the original literal None then return/world adapter', origin)
                    kind, result = self.run(context, ins.protected.region,
                        tuple(values[v] for v in ins.protected.arguments))
                    if kind is C.FALLTHROUGH:
                        kind, result = self.run(context, route.handler, result)
                    if kind is not C.FALLTHROUGH: return kind, result
                elif type(ins) is P.AccessBinding: result = self.binding(context, step, args)
                elif type(ins) is P.ConstructValue: result = self.construct(context, step, args, evaluated, origin)
                elif type(ins) is P.DefineLiteral:
                    if args not in ((), (_WORLD,)):
                        _reject('STATE_LITERAL_WORLD', 'literal adapter has unknown operands', origin)
                    result = (_Scalar(ins.value.to_python(), origin), *args)
                elif type(ins) is P.ApplyLanguageAction: result = self.action(context, step, args, origin)
                elif type(ins) is P.ApplyInplaceOperator: result = self.inplace(context, step, args, origin)
                elif type(ins) is P.EnterDeferredBody: result = (_WORLD,)
                elif type(ins) is P.ApplyPythonProtocol and ins.protocol.value == 'FRAME_RELEASE_TEMPORARY': result = (_WORLD,)
                else: _reject('STATE_UNSUPPORTED_INSTRUCTION', type(ins).__name__ + ' has open control/call/state semantics', origin)
                normal = next((c for c in step.completions if c.kind is C.FALLTHROUGH), None)
                if normal is None or len(result) != len(normal.types):
                    _reject('STATE_RESULT_ARITY', 'original fallthrough ABI differs', origin)
                for i, value in enumerate(result): values[P.ProductionResult(step.id, i)] = value
            _reject('STATE_MISSING_TRANSFER', 'region has no original completion transfer')
        finally: self.active.remove(identity)

    def lower(self):
        fn = self.function
        if (fn.body_scope is None or fn.environment.kind is not P.DeferredBodyKind.FUNCTION
                or fn.environment.captures or fn.environment.owned_cells or not fn.parameter_names
                or any(p.kind.value in ('VAR_POSITIONAL', 'VAR_KEYWORD') for p in fn.environment.parameters)):
            _reject('STATE_ENVIRONMENT', 'ordinary fixed-signature function without captures required')
        context = self.resolver.plan(self.source.unit, fn.template)
        if any(type(step.instruction) is P.ApplyDecorator for step in context.plan.steps):
            _reject('STATE_ENTRY_DECORATOR', 'stateful entry cannot have decorators')
        arguments = [_ENV]
        for i, name in enumerate(fn.parameter_names):
            formal = source_port(self.facts, fn.formals[i + 1]); origin = _origin(context, fn.body, value=encode(formal))
            value = _Path((name,), origin); arguments.append(value)
            namespaces = self.names.get((formal.port, True), ())
            if len(namespaces) != 1 or namespaces[0].kind is not F.NamespaceKind.LEXICAL:
                _reject('STATE_PARAMETER_BINDING', 'formal lacks a unique original lexical binding')
            self.locals[namespaces[0]] = value
        kind, result = self.run(context, fn.body, (*arguments, _WORLD))
        if kind is not C.RETURN or len(result) != 2 or not isinstance(result[0], _Scalar) or result[0].value is not None:
            _reject('STATE_RETURN', 'stateful function must return None without escaping values')
        if not any(action.kind == 'write' for action in self.actions): _reject('STATE_NO_WRITES', 'no admitted array write')
        return tuple(self.inputs.values()), tuple(self.nodes), tuple(self.actions), tuple(self.callees), tuple(self.providers), tuple(self.origins)


class _KindConstraintLowerer(_StateLowerer):
    """Collect F4 array requirements; every temporary expression is discarded.

    An unconstrained numeric path is a soft scalar *candidate*, not a scalar
    fact. A later array read/write/provider can therefore constrain it before
    the strict second interpretation starts. Unknown control/calls retain the
    ordinary rejections. There is no retry/fixed-point loop or source parsing.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.required_array_paths = set()

    def input(self, value, kind):
        if kind == 'array':
            self.required_array_paths.add(value.path)
            old = self.inputs.get(value.path)
            if old is not None and old.kind == 'scalar':
                self.inputs[value.path] = replace(old, kind='array')
                self.epochs[old.id] = 0
                # Keep collection traversal coherent when it subsequently
                # inspects the kind of an earlier numeric expression. These
                # provisional nodes never escape this pass or its validator.
                for index, node in enumerate(self.nodes):
                    if node.op == 'input' and node.attrs['name'] == old.id:
                        updated = replace(node, attrs=dict(name=old.id, kind='array', epoch=0))
                        self.nodes[index] = updated
                        self.by_id[node.id] = updated
        return super().input(value, kind)

    def numeric(self, value, *, array=False):
        if isinstance(value, _Value):
            node = self.by_id[value.node]
            if array and node.op == 'input' and node.attrs['kind'] == 'scalar':
                row = next(row for row in self.inputs.values() if row.id == node.attrs['name'])
                self.input(_Path(row.path, row.origins[0]), 'array')
            kind = self.kind(value.node)
            if array and kind != 'array':
                # An array-only composite operand may have an array path
                # established later by a write. Do not guess which operand
                # is array; the final strict pass must resolve or reject it.
                return _Value(value.node, 'array')
            return _Value(value.node, kind)
        return super().numeric(value, array=array)


def discover_stateful_regions(catalog, source=None):
    view = catalog if isinstance(catalog, ProgramView) else ProgramView(catalog)
    units = tuple(view.catalog.fact_store.rows) if source is None else tuple(sorted(view.catalog.units_for_source(str(Path(source).absolute())) or ()))
    if source is not None and not units: return [], [dict(path=str(Path(source).absolute()), code='SOURCE_NOT_IN_CATALOG')]
    result, rejected, resolver = [], [], BodyInterfaceResolver(view)
    for unit in units:
        original = view.unit(unit); functions = original.facts.functions
        resolver.prepare_many(Body(unit, fn.template, fn.body) for fn in functions)
        model, syntax, codes, recipe, digest = _source_identity(view.catalog, unit)
        metadata, leaves = {}, {}
        for ordinal, function in enumerate(functions):
            name, span, lines = _metadata(syntax, function.template.occurrence)
            matches = [code for code in codes if code.co_name == name and code.co_firstlineno in lines
                and code.co_varnames[:code.co_argcount + code.co_kwonlyargcount] == function.parameter_names]
            if len(matches) != 1: continue
            code = matches[0]; identity = dict(kind='source_identity', method='COMPILE_ONLY_RETAINED_F0',
                semantic_authority='ORIGINAL_F4_PRODUCTION_PLANS_AND_UNIT_FACTS', unit=unit,
                occurrence=function.template.occurrence, recipe=recipe, code_key=code_key(code), span=span)
            source_ref = SourceRef(model['path'], digest, code.co_qualname, span['line'], span['column'], code_key(code), (identity,))
            metadata[function.template] = source_ref
            context = resolver.plan(unit, function.template)
            if not any(type(step.instruction) is P.ApplyDecorator for step in context.plan.steps): continue
            try:
                nodes, output, origins = _GraphLowerer(original, function, resolver, syntax=syntax).lower()
                leaves[function.template] = (function, ArrayRegion('shared:' + str(function.template.occurrence),
                    replace(source_ref, origins=(*source_ref.origins, *origins)), function.parameter_names, nodes, output))
            except GraphRegionRejected: pass
        imports, members = _declarations(original, resolver)
        for ordinal, function in enumerate(functions):
            reference = metadata.get(function.template)
            if reference is None: continue
            try:
                arguments = (original, function, resolver, syntax, imports, members, leaves)
                constraints = _KindConstraintLowerer(*arguments)
                constraints.lower()
                inputs, nodes, actions, callees, providers, origins = _StateLowerer(
                    *arguments, array_paths=constraints.required_array_paths).lower()
                identifier = 'stateful-region:' + sha256((unit + '\0' + function.body.wire + '\0' + reference.code_key).encode()).hexdigest()[:24]
                region = StatefulRegion(identifier, replace(reference, origins=(*reference.origins, *origins)), function.parameter_names,
                    inputs, nodes, actions, callees, providers,
                    REQUIRED_GUARDS)
                result.append(validate_stateful_region(region))
            except (GraphRegionRejected, ValueError) as error:
                rejected.append(dict(unit=unit, function=ordinal, path=reference.path, qualname=reference.qualname,
                    code=error.code if isinstance(error, GraphRegionRejected) else 'STATE_INVALID_REGION',
                    detail=str(error), origin=getattr(error, 'origin', None)))
    return result, rejected


__all__ = ['discover_stateful_regions']

"""Lower original F4 scalar controller structure with explicit effect calls."""

import argparse
from collections import deque
import json
from pathlib import Path

from scar.analysis.bounded_entry_calls import _admit_entry_calls
from scar.analysis.build import BuildSession
from scar.analysis.replay_transaction import replay_transaction
from scar.analysis.session_caller_effects import _fact_indexes
from scar.analysis.session_scalar_flow import derive_scalar_flow
from scar.analysis.source_address_paths import AddressResolver
from scar.analysis.source_body_interfaces import BodyInterfaceResolver
from scar.analysis.source_native_update import SCHEMA as UPDATE, check as check_update
from scar.analysis.source_reference_slice import ReferenceSlicer, source_port
from scar.compiler.graph_frontend import _origin
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program import normal_links as F, normal_targets as T, source_productions as P
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.source_address_paths import AddressRootKind, FieldSelector
from scar.ir.program.structured import PythonProtocolKind as K
from scar.ir.record_codec import decode, encode


SCHEMA = 'scar.source-native-controller'
_WORLD = object()
_ENV = object()


def _only(rows, detail):
    items = tuple(rows)
    if len(items) != 1:
        raise ValueError(detail)
    return items[0]


def _ref(identity):
    return dict(kind='VALUE', id=identity)


def _literal(value):
    return dict(kind='LITERAL', literal=encode(value))


def _slot(namespace):
    return f'lexical:{namespace.owner}:{namespace.name}'


class _Controller:
    def __init__(self, view, source, ordinal, function, incoming, loop, update):
        self.source, self.facts = source, source.facts
        self.function, self.incoming, self.loop = function, incoming, loop
        self.owner = function.body_scope.owner
        self.resolver = BodyInterfaceResolver(view)
        self.slicer = ReferenceSlicer(view.catalog)
        self.addresses = AddressResolver(view.catalog.fact_store, self.facts, self.owner)
        self.update = update
        self.callee_unit = view.unit(incoming.body.unit)
        self.callee = _only((fn for fn in self.callee_unit.facts.functions
                             if (fn.template, fn.body) == (incoming.body.template, incoming.body.region)),
                            'controller update must target one original function')
        _, self.world_facts, self.name_facts = _fact_indexes(source, function, self.slicer)
        self.query = _only((i for i, row in enumerate(self.facts.calls)
                            if source_port(self.facts, row.result) == incoming.result),
                           'selected outer CALL must match one original F4 result')
        self.scalar = derive_scalar_flow(view, unit=source.unit, function_ordinal=ordinal,
                                         known_call_returns={self.query:'NUMBER'}, resolver=self.resolver)
        self.scalar_ops = {T.GlobalPort(source.unit, decode(T.Port, self.scalar['ports'][row['entry']]['port'])):row
                           for row in self.scalar['operations']}
        self.constants = {source_port(self.facts, row.value): (i, row.literal)
                          for i, row in enumerate(self.facts.constants)}
        self.slots = {decode(F.Namespace, row['namespace']):row['kinds'] for row in self.scalar['lexical_joins']}
        self.params = {}
        self.parameter_ports = {}
        self.effects = []
        self.active = set()
        self.generated = 0
        self.selected_calls = 0
        self.loops = 0

    def _native_arguments(self, fact, args):
        explicit = [(source_port(self.facts, value), dict(role='POSITIONAL', ordinal=i, value=args[1+i]))
                    for i, value in enumerate(fact.positional)]
        keyword_offset = 1 + len(fact.positional)
        explicit.extend((source_port(self.facts, value),
                         dict(role='KEYWORD', name=name, ordinal=i, value=args[keyword_offset+i]))
                        for i, (name, value) in enumerate(fact.keywords))
        _, _, aliases, *_ = self.slicer.unit(self.source.unit)
        ancestors, pending = {fact.callee}, deque((fact.callee,))
        while pending:
            for source, _ in aliases[pending.popleft()]:
                if source not in ancestors:
                    ancestors.add(source)
                    pending.append(source)

        rows = []
        for i, (name, index) in enumerate(zip(self.callee.parameter_names, self.callee.formals[1:-1], strict=True)):
            formal = source_port(self.callee_unit.facts, index)
            route = _only((row for row in self.incoming.parameters
                           if row.kind is T.RouteKind.ARGUMENT and row.formal == formal),
                          'native update parameter needs one original formal route')
            actual, = route.actuals
            known = tuple(binding for port, binding in explicit if port == actual)
            if known:
                binding, = known
            else:
                attr, row = _only(((j, row) for j, row in enumerate(self.facts.attributes)
                                  if row.action is F.AttributeAction.GET and row.value in ancestors
                                  and source_port(self.facts, row.receiver) == actual),
                                 'native update receiver must be the original attribute receiver')
                _, ordinal = self.slicer.local(actual)
                path = _only(self.addresses.paths(ordinal), 'native update receiver needs one original address')
                if (path.frontier is not None or path.root_kind is not AddressRootKind.FUNCTION_PARAMETER
                        or path.root not in self.parameter_ports
                        or any(type(item) is not FieldSelector for item in path.selectors)):
                    raise ValueError('native update requires a guarded original controller receiver path')
                parent = self.parameter_ports[path.root]
                binding = dict(role='ATTRIBUTE_RECEIVER', source_path=[parent['name'], *(p.name for p in path.selectors)],
                    slot=parent['slot'], source_receiver=path,
                    attribute_fact=self.slicer.fact(self.facts, 'attributes', attr),
                    requires_original_descriptor=True, attribute=row.name)
            observed = []
            for activation in self.update['parent']['observations']:
                parameter = activation['formal_references'][i]
                if parameter['name'] != name:
                    raise ValueError('native update actual frame formal differs from source')
                observed.append(dict(invocation=activation['invocation'], entry=activation['entry'],
                                     reference=parameter['ref'], monitored_arg0_matches=
                                     parameter['ref'] == activation['monitoring_arg0']))
            rows.append(dict(parameter=name, formal=formal, actual=actual, binding=binding, observations=observed))
        return rows

    def _emit(self, block, origin, kind, **fields):
        block.append(dict(kind=kind, origin=origin, **fields))

    def _value(self, block, origin, scalar, **expr):
        self.generated += 1
        identity = 'value:' + str(self.generated)
        kinds = scalar['output_kinds'][0] if scalar is not None and scalar['output_kinds'] else ['UNKNOWN']
        self._emit(block, origin, 'EVAL', id=identity, expression=expr, value_kinds=kinds)
        return _ref(identity)

    def _facts(self, context, step, table):
        worlds = context.worlds(step)
        return _only(((i, row, evidence) for name, i, row, evidence in self.world_facts.get(worlds, ())
                      if name == table), 'controller step lacks one original ' + table + ' fact')

    def _binding(self, context, step, args, block, origin, scalar):
        ins = step.instruction
        write = ins.action is P.BindingActionKind.WRITE
        if ins.action not in (P.BindingActionKind.READ, P.BindingActionKind.WRITE):
            raise ValueError('controller local deletion needs a separate protocol')
        endpoint = context.value(step.arguments[0]) if write else context.result(step)
        choices = tuple((row.namespace, ref) for _, _, row, ref in self.name_facts.get(endpoint, ())
                        if row.write is write)
        namespaces = tuple(dict.fromkeys(namespace for namespace, _ in choices))
        if not namespaces:
            raise ValueError('controller binding has no original namespace fact')
        facts = tuple(dict.fromkeys(ref for _, ref in choices))
        if all(row.kind is F.NamespaceKind.LEXICAL for row in namespaces):
            namespace, = namespaces
            self.slots.setdefault(namespace, ['UNKNOWN'])
            if write:
                self._emit(block, origin, 'STORE_SLOT', slot=_slot(namespace), value=args[0], source_facts=facts)
                return (_WORLD,)
            return self._value(block, origin, scalar, kind='SLOT', slot=_slot(namespace), source_facts=facts), _WORLD
        if write:
            raise ValueError('controller namespace write is outside its ordinary local profile')
        return (self._value(block, origin, scalar, kind='NAMESPACE', namespace=namespaces,
                            source_facts=facts, requires_original_lookup=True), _WORLD)

    def _operator(self, context, step, args, block, origin, scalar):
        _, fact, evidence = self._facts(context, step, 'operations')
        if (source_port(self.facts, fact.result) != context.result(step)
                or tuple(source_port(self.facts, i) for i in fact.operands)
                != tuple(context.value(value) for value in step.arguments[:-1])):
            raise ValueError('controller arithmetic differs from the original operand layout')
        value = self._value(block, origin, scalar, kind='OPERATOR', operator=fact.selector.value,
            operands=args[:-1], inplace=type(step.instruction) is P.ApplyInplaceOperator,
            source_fact=evidence, operand_kinds=scalar['input_kinds'],
            semantics='EXACT_PYTHON_BUILTIN_OPERAND_PROTOCOL')
        return value, _WORLD

    def _call(self, context, step, args, block, origin, scalar):
        query, fact, evidence = self._facts(context, step, 'calls')
        original = (fact.callee, *fact.positional, *(value for _, value in fact.keywords))
        if (fact.positional_pack is not None or fact.keyword_pack is not None
                or tuple(source_port(self.facts, value) for value in original)
                != tuple(context.value(value) for value in step.arguments[:-1])):
            raise ValueError('controller CALL needs its original direct argument and keyword order')
        site = context.result(step)
        selected = query == self.query
        if selected and site != self.incoming.result:
            raise ValueError('controller selected CALL differs from the original update result')
        self.generated += 1
        identity = 'value:' + str(self.generated)
        count = len(fact.positional) + 1
        fields = dict(role='NATIVE_UPDATE' if selected else 'ORIGINAL_EFFECT', site=site,
            source_fact=evidence, query_ordinal=query, callee=args[0], positional=args[1:count],
            keywords=[dict(name=name, value=value) for (name, _), value in zip(fact.keywords, args[count:-1], strict=True)],
            result=identity, result_kinds=scalar['output_kinds'][0])
        if selected:
            self.selected_calls += 1
            fields['formal_routes'] = self._native_arguments(fact, args)
        else:
            effect = 'effect:' + str(len(self.effects))
            fields['effect'] = effect
            self.effects.append(dict(id=effect, site=site, source_fact=evidence, query_ordinal=query,
                origin=origin, positional_count=len(fact.positional), keyword_names=[name for name, _ in fact.keywords],
                result=identity,
                required_boundary='PRESERVE_ORIGINAL_LOOKUP_ARGUMENTS_CALL_RESULT_AND_EXTERNAL_EFFECTS',
                effects_closed=False))
        self._emit(block, origin, 'CALL', **fields)
        return _ref(identity), _WORLD

    def _language(self, context, step, args, block, origin, scalar):
        action = step.instruction.action
        if action is P.LanguageActionKind.CALL:
            return self._call(context, step, args, block, origin, scalar)
        if action in (P.LanguageActionKind.BINARY, P.LanguageActionKind.UNARY, P.LanguageActionKind.COMPARE):
            return self._operator(context, step, args, block, origin, scalar)
        if action in (P.LanguageActionKind.ATTRIBUTE_GET, P.LanguageActionKind.ATTRIBUTE_SET):
            _, fact, evidence = self._facts(context, step, 'attributes')
            if source_port(self.facts, fact.receiver) != context.value(step.arguments[0]):
                raise ValueError('controller field receiver differs from its original operand')
            if action is P.LanguageActionKind.ATTRIBUTE_SET:
                if fact.action is not F.AttributeAction.SET or source_port(self.facts, fact.value) != context.value(step.arguments[1]):
                    raise ValueError('controller field write differs from its original payload')
                self._emit(block, origin, 'STORE_FIELD', receiver=args[0], field=fact.name, value=args[1],
                           source_fact=evidence, requires_ordinary_field_protocol=True)
                return (_WORLD,)
            if fact.action is not F.AttributeAction.GET or source_port(self.facts, fact.value) != context.result(step):
                raise ValueError('controller field read differs from its original result')
            return (self._value(block, origin, scalar, kind='FIELD', receiver=args[0], field=fact.name,
                                source_fact=evidence, requires_original_lookup=True), _WORLD)
        raise ValueError('unsupported controller language action: ' + action.value)

    def _protocol(self, step, args, block, origin, scalar):
        protocol = step.instruction.protocol
        if protocol is K.FRAME_RELEASE_TEMPORARY:
            self._emit(block, origin, 'RELEASE', value=args[0], original_release_required=bool(scalar['frontiers']),
                       source_frontiers=scalar['frontiers'])
            return (_WORLD,)
        if protocol is K.TRUTH:
            return self._value(block, origin, scalar, kind='TRUTH', value=args[0]), _WORLD
        if protocol in (K.FSTRING_FORMAT, K.BUILD_STRING, K.FSTRING_CONVERT):
            kind = {K.FSTRING_FORMAT:'FORMAT', K.BUILD_STRING:'STRING_JOIN', K.FSTRING_CONVERT:'STRING_CONVERT'}[protocol]
            return (self._value(block, origin, scalar, kind=kind, operands=args[:-1],
                                bindings=encode(step.instruction.bindings, record_tags=P.PRODUCTION_RECORD_TAGS)), _WORLD)
        raise ValueError('unsupported controller Python protocol: ' + protocol.value)

    def _invoke(self, context, call, values, block):
        return self._region(context, call.region, tuple(values[value] for value in call.arguments), block)

    def _region(self, context, region_id, args, block):
        identity = context.key, region_id
        if identity in self.active or len(self.active) >= 128 or context.owner(region_id).owner != self.owner:
            raise ValueError('controller requires bounded same-owner original F4 expansion')
        region = context.regions[region_id]
        if len(args) != len(region.parameters):
            raise ValueError('controller original region input arity differs')
        self.active.add(identity)
        values = {P.ProductionParameter(region_id, i):value for i, value in enumerate(args)}
        try:
            for step_id in region.steps:
                step = context.steps[step_id]
                ins = step.instruction
                args = tuple(values[value] for value in step.arguments)
                origin = _origin(context, region_id, step)
                scalar = self.scalar_ops.get(context.site(step))
                if type(ins) is P.TransferCompletion:
                    return ins.kind, args
                if type(ins) is P.EvaluateChild:
                    selected = self.source.material.interfaces['links'][context.key, step.id]
                    child = self.resolver.plan(context.unit, selected.callee)
                    kind, result = self._region(child, child.plan.root, args, block)
                    if kind is not C.FALLTHROUGH:
                        return kind, result
                elif type(ins) is P.ApplyRegion:
                    kind, result = self._invoke(context, ins.invocation, values, block)
                    if kind is not C.FALLTHROUGH:
                        return kind, result
                elif type(ins) is P.HandleCompletions:
                    if any(route.kind is not C.FALLTHROUGH for route in ins.routes):
                        raise ValueError('controller exception or nonlocal completion handler remains open')
                    kind, result = self._invoke(context, ins.protected, values, block)
                    handlers = tuple(route for route in ins.routes if route.kind is kind and route.target is None)
                    if handlers:
                        handler, = handlers
                        captures = tuple(values[value] for value in handler.captures)
                        kind, result = self._region(context, handler.handler, (*result, *captures), block)
                    if kind is not C.FALLTHROUGH:
                        return kind, result
                elif type(ins) is P.BranchControl:
                    then, otherwise = [], []
                    left = self._invoke(context, ins.on_true, values, then)
                    right = self._invoke(context, ins.on_false, values, otherwise)
                    if left != (C.FALLTHROUGH, (_WORLD,)) or right != left:
                        raise ValueError('controller branches need original world-only fallthrough')
                    self._emit(block, origin, 'IF', test=args[0], then=then, otherwise=otherwise)
                    result = (_WORLD,)
                elif type(ins) is P.LoopControl:
                    result = self._while(context, step, args, block, origin)
                elif type(ins) is P.EnterDeferredBody:
                    result = (_WORLD,)
                elif type(ins) is P.AccessBinding:
                    result = self._binding(context, step, args, block, origin, scalar)
                elif type(ins) is P.DefineLiteral:
                    if args not in ((), (_WORLD,)):
                        raise ValueError('controller literal must preserve only its original optional world')
                    result = (_literal(ins.value), *args)
                elif type(ins) is P.ConstructValue and ins.construction is P.ConstructionKind.LITERAL:
                    ordinal, literal = self.constants[context.result(step)]
                    result = (dict(**_literal(literal), source_fact=self.slicer.fact(self.facts, 'constants', ordinal)), _WORLD)
                elif type(ins) is P.ApplyInplaceOperator:
                    result = self._operator(context, step, args, block, origin, scalar)
                elif type(ins) is P.ApplyLanguageAction:
                    result = self._language(context, step, args, block, origin, scalar)
                elif type(ins) is P.ApplyPythonProtocol:
                    result = self._protocol(step, args, block, origin, scalar)
                else:
                    raise ValueError('unsupported original controller instruction: ' + type(ins).__name__)
                normal, = (completion for completion in step.completions if completion.kind is C.FALLTHROUGH)
                if len(result) != len(normal.types):
                    raise ValueError(f'controller {type(ins).__name__} at {step.id.wire} returns '
                                     f'{len(result)} ports, expected {len(normal.types)}')
                values.update((P.ProductionResult(step.id, i), value) for i, value in enumerate(result))
            # The F4 normal graph forwards the last step's normal result to
            # the containing region's matching completion even with no
            # TransferCompletion step, e.g. a LoopControl tail region.
            if region.steps and normal in region.completions:
                return C.FALLTHROUGH, result
            raise ValueError('original controller region has no matching completion')
        finally:
            self.active.remove(identity)

    def _while(self, context, step, args, block, origin):
        ins = step.instruction
        if (context.site(step) != self.loop or type(ins.condition) is not P.ProductionRegionID
                or args != (_WORLD,) or ins.invariant_count):
            raise ValueError('controller needs the admitted original pre-test loop with a lexical carry')
        self.loops += 1
        condition, body, otherwise = [], [], []
        kind, result = self._region(context, ins.condition, args, condition)
        if kind is not C.FALLTHROUGH or len(result) != 2 or result[-1] is not _WORLD:
            raise ValueError('controller loop predicate differs from the original truth/world ABI')
        if self._region(context, ins.body, args, body) != (C.FALLTHROUGH, args):
            raise ValueError('controller loop requires normal body carry')
        if ins.otherwise is not None and self._region(context, ins.otherwise, args, otherwise) != (C.FALLTHROUGH, args):
            raise ValueError('controller loop otherwise requires normal world carry')
        self._emit(block, origin, 'WHILE', condition=condition, test=result[0], body=body,
                   otherwise=otherwise, header=self.loop)
        return (_WORLD,)

    def lower(self):
        ctx = self.resolver.plan(self.source.unit, self.function.template)
        params = []
        for i, (name, formal) in enumerate(zip(self.function.parameter_names, self.function.formals[1:-1], strict=True)):
            port = source_port(self.facts, formal)
            namespace = _only((row.namespace for _, _, row, _ in self.name_facts[port]
                               if row.write and row.namespace.kind is F.NamespaceKind.LEXICAL),
                              'controller parameter needs one original lexical slot')
            self.params[namespace] = name
            self.slots.setdefault(namespace, ['UNKNOWN'])
            declared = self.function.environment.parameters[i]
            param = dict(name=name, slot=_slot(namespace), formal=port, value='parameter:' + str(i),
                         parameter_kind=declared.kind, default_source=declared.default_source)
            params.append(param)
            self.parameter_ports[port] = param
        statements = []
        kind, result = self._region(ctx, self.function.body, (_ENV, *(_ref(row['value']) for row in params), _WORLD), statements)
        if (kind is not C.RETURN or len(result) != 2 or result[-1] is not _WORLD
                or self.selected_calls != 1 or self.loops != 1):
            raise ValueError('controller needs one original selected-call loop and a normal return')
        statements.append(dict(kind='RETURN', value=result[0], origin=_origin(ctx, self.function.body)))
        for effect in self.effects:
            uses = _value_uses(statements, effect['result'])
            effect['result_uses'] = uses
            effect['result_policy'] = 'ORIGINAL_RELEASE_ONLY' if uses == ['RELEASE'] else 'ORIGINAL_VALUE_REQUIRED'
        slots = [dict(id=_slot(namespace), namespace=namespace, kinds=kinds, parameter=self.params.get(namespace))
                 for namespace, kinds in self.slots.items()]
        self.source.check()
        return params, slots, statements


def _value_uses(statements, identity):
    result = []
    def visit(node, owner=None):
        if type(node) is dict:
            kind = node.get('kind')
            if kind == 'VALUE':
                if node['id'] == identity:
                    result.append(owner)
                return
            if 'origin' in node:
                owner = kind
            for value in node.values():
                visit(value, owner)
        elif type(node) in (tuple, list):
            for value in node:
                visit(value, owner)
    visit(statements)
    return result


@replay_transaction()
def project(update_source):
    reference = artifact_reference(Path(update_source).resolve(strict=True), UPDATE)
    update = check_update(reference['uri'])
    if read_artifact(reference) != update:
        raise ValueError('original update changed during controller replay')
    entry = read_artifact(update['entry_calls'])
    ordered = read_artifact(update['ordered_graph'])
    namespace, index = update['parent']['interface'].rsplit('/', 1)
    if namespace != '#/interfaces' or str(int(index)) != index or int(index) < 0:
        raise ValueError('controller requires the original outer CALL interface')
    row = entry['interfaces'][int(index)]
    incoming = decode(T.CallLink, row['interface']['link'])
    caller = decode(F.CodeScopeRef, row['caller_scope'])
    loop = decode(T.GlobalPort, ordered['original_loop_header'])
    with BuildSession(Path(entry['program']['uri']).parent) as session:
        with _admit_entry_calls(session, entry):
            source = session.view.unit(caller.unit)
            ordinal, function = _only(((i, fn) for i, fn in enumerate(source.facts.functions)
                                      if fn.body_scope is not None and fn.body_scope.owner == caller),
                                     'controller requires the original outer code body')
            lowerer = _Controller(session.view, source, ordinal, function, incoming, loop, update)
            parameters, slots, program = lowerer.lower()
    return encode(dict(schema=SCHEMA, schema_version=1, update_source=reference, source=lowerer.scalar['source'],
        source_function=lowerer.scalar['source_function'], body=lowerer.scalar['body'],
        original_f4=lowerer.scalar['original_f4'], original_loop_header=loop, update_call=incoming.result,
        parameters=parameters, slots=slots, program=program, effects=lowerer.effects,
        scalar_obligations=dict(ledger=lowerer.scalar['id'], guards=lowerer.scalar['guards'],
            port_bindings=[dict(index=i, port=row['port']) for i, row in enumerate(lowerer.scalar['ports'])
                           if any(i == guard.get('port') for guard in lowerer.scalar['guards'])]),
        qualification=['ORIGINAL_ARGUMENTS_AND_LEXICAL_ACTIVATION',
            'EXACT_BUILTIN_SCALAR_OPERANDS_WITH_ORIGINAL_ORDERED_PROTOCOL',
            'ORDINARY_FIELD_BINDING_AND_OLD_REFERENCE_RELEASE',
            'SAME_SELECTED_UPDATE_BODY_AND_DATATYPE_GUARDS_EACH_OCCURRENCE',
            'ORIGINAL_EXTERNAL_LOOKUPS_CALLS_RETURNS_AND_RELEASES_AT_SELECTED_SITES'],
        remaining=[*update['remaining'], 'EXTERNAL_EFFECT_PROVIDER_AND_ORDINARY_SCALAR_RUNTIME_GUARDS',
                   'ORIGINAL_EXCEPTION_AND_FINALIZER_PROTOCOL'],
        effect_closure_proven=False, transformation_authorized=False))


def check(path):
    result = read_artifact(artifact_reference(Path(path).resolve(strict=True), SCHEMA))
    if result != project(result['update_source']['uri']):
        raise ValueError('controller differs from original admitted F4 source and update CALL')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--update-source', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args(argv)
    if args.check and not (args.update_source or args.out):
        result = check(args.check)
    elif args.update_source and args.out and not args.check:
        result = project(args.update_source)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --check or --update-source with --out')
    print(json.dumps(dict(schema=SCHEMA, parameters=len(result['parameters']), opaque_effects=len(result['effects']))))


if __name__ == '__main__':
    main()

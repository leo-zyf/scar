"""Guarded read reuse for repeated scalar table slots in one expression."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from operator import add, mul, sub
from pathlib import Path
import sys
from types import SimpleNamespace

from scar.analysis.normal_graph import normal_edges
from scar.analysis.program_view import ProgramView
from scar.analysis.source_content_forwarding import _Refuse, _normal_route
from scar.analysis.source_references import ReferenceSlicer, source_port
from scar.compiler.graph_frontend import _metadata, _source_identity
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program import normal_graph as G, normal_graph as F, normal_graph as T, source_productions as P
from scar.ir.record_codec import encode
from scar.trace.code_snapshot import code_key


SCHEMA = 'scar.scalar-table-cse'
_OPS = {'Add': add, 'Sub': sub, 'Mult': mul}
_MAX_KEYS = 64
_MISSING = object()
_QUALIFICATION = (
    'ORIGINAL_F4_ACTIVATION_AND_NAME_BINDINGS_CURRENT',
    'ORIGINAL_REPEATED_BINDING_READS_ARE_CALLBACK_FREE_LOCAL_OR_MODULE_READS',
    'RESOLVED_TABLE_IS_EXACT_DICT_WITH_ONLY_EXACT_STRING_KEYS',
    'REPEATED_KEY_IS_EXACT_STRING_AND_SELECTED_ROW_IS_EXACT_TUPLE',
    'INDICES_ARE_EXACT_INTS_AND_SELECTED_OPERANDS_ARE_EXACT_INTS',
    'NO_REENTRANT_CONCURRENT_OR_ASYNC_CHANGE_TO_BINDINGS_OR_TABLE_DURING_EXPRESSION',
    'NO_TRACE_PROFILE_DEBUGGER_OR_FRAME_REFLECTION_ON_CHANGED_EXPRESSION',
    'ORIGINAL_LOOKUPS_FALLBACK_ON_UNSUPPORTED_TYPES_OR_MISSING_ITEMS',
)


def _require(condition, detail):
    if not condition:
        raise ValueError(detail)


class _Facts:
    def __init__(self, catalog, source):
        self.catalog = catalog
        self.unit = source.facts
        self.ports = [source_port(self.unit, i) for i in range(len(self.unit.ports))]
        self.slicer = ReferenceSlicer(SimpleNamespace(fact_store=catalog.fact_store))
        self.predecessors = defaultdict(list)
        for ordinal, (before, after) in enumerate(self.unit.aliases):
            self.predecessors[after].append((before, ordinal))
        self.items = {row.value:(ordinal, row) for ordinal, row in enumerate(self.unit.items) if not row.write}
        self.names = defaultdict(list)
        for ordinal, row in enumerate(self.unit.names):
            if not row.write:
                self.names[row.value].append((ordinal, row))
        self.constants = {row.value:(ordinal, row) for ordinal, row in enumerate(self.unit.constants)}
        scopes = {fn.body_scope.owner:fn for fn in self.unit.functions if fn.body_scope is not None}
        self.functions = {i:scopes[group.owner] for group in self.unit.scope_groups if group.owner in scopes
                          for table in group.tables if table.table is F.FactTable.OPERATIONS
                          for i in range(table.start, table.stop)}

    def origin(self, index, table):
        """Follow only single-parent original value transport edges."""
        seen = set()
        trail = []
        while index not in table:
            following = self.predecessors[index]
            if index in seen or len(following) != 1:
                return None
            seen.add(index)
            index, ordinal = following[0]
            trail.append(ordinal)
        return index, table[index], tuple(trail)

    def reference(self, table, ordinal):
        return encode(self.slicer.fact(self.unit, table, ordinal))

    def evidence(self, item, table):
        position, (ordinal, _), trail = item
        return dict(fact=self.reference(table, ordinal), result=encode(self.ports[position]),
            value_transport=[self.reference('aliases', edge) for edge in trail])


def _syntax_fields(syntax, occurrence):
    row = syntax.nodes[occurrence]
    names = next(schema.fields for schema in syntax.profile.ast_schema if schema.name == row.constructor)
    return dict(zip(names, row.fields, strict=True))


def _node(syntax, occurrence, field):
    value = _syntax_fields(syntax, occurrence)[field]
    return value.reference if value.kind == 'NODE' else None


def _name(syntax, occurrence):
    if syntax.nodes[occurrence].constructor != 'Name':
        return None
    return _syntax_fields(syntax, occurrence)['id'].scalar.to_python()


def _shape(syntax, root):
    """Use retained token spans to delimit the exact four-subscript expression."""
    if syntax.nodes[root].constructor != 'BinOp':
        return None
    left, right = (_node(syntax, root, side) for side in ('left', 'right'))
    if any(item is None or syntax.nodes[item].constructor != 'Subscript' for item in (left, right)):
        return None
    first, second = (_node(syntax, item, 'value') for item in (left, right))
    if any(item is None or syntax.nodes[item].constructor != 'Subscript' for item in (first, second)):
        return None
    bases = [_node(syntax, item, 'value') for item in (first, second)]
    keys = [_node(syntax, item, 'slice') for item in (first, second)]
    indices = [_node(syntax, item, 'slice') for item in (left, right)]
    if any(node is None for node in (*bases, *keys, *indices)):
        return None
    if (_name(syntax, bases[0]) is None or _name(syntax, bases[0]) != _name(syntax, bases[1])
            or _name(syntax, keys[0]) is None or _name(syntax, keys[0]) != _name(syntax, keys[1])
            or any(syntax.nodes[node].constructor not in ('Name', 'Constant') for node in indices)):
        return None
    return (first, left, second, right), bases, keys, indices


def _attach_path(source, binary, expected):
    """Close the static effects of the expression's normal-success control window."""
    plans = source.material.closure(binary.port.template)
    root = plans[binary.port.template].root
    edges = tuple(edge for plan in plans.values() for edge in normal_edges(plan, source.material.interfaces))
    flow = SimpleNamespace(edges=edges, body=T.Body(source.unit, binary.port.template, root), source_function=binary)
    instructions = {(key, step.id):step.instruction for key, plan in plans.items() for step in plan.steps}
    ordered, effects = [], []
    originals = set(expected)
    try:
        for port, _, _ in _normal_route(flow):
            if port.kind is not G.PortKind.ENTER:
                continue
            identity = port.template, port.owner
            instruction = instructions.get(identity)
            if instruction is None:
                if type(port.owner) is P.ProductionStepID:
                    return None
                continue
            if type(instruction) in (P.EvaluateChild, P.TransferCompletion):
                continue
            if identity == (binary.port.template, binary.port.owner):
                if (type(instruction) is P.ApplyLanguageAction and instruction.action is P.LanguageActionKind.BINARY
                        and ordered == list(expected)):
                    return effects
                return None
            if type(instruction) is P.AccessBinding and instruction.action is P.BindingActionKind.READ:
                effect = 'ORIGINAL_BINDING_READ'
            elif type(instruction) is P.ConstructValue and instruction.construction is P.ConstructionKind.LITERAL:
                effect = 'ORIGINAL_LITERAL'
            elif (identity in originals and type(instruction) is P.ApplyLanguageAction
                    and instruction.action is P.LanguageActionKind.ITEM_GET):
                ordered.append(identity)
                effect = 'ORIGINAL_ITEM_READ_REQUIRES_EXACT_BUILTIN'
            else:
                return None
            effects.append(dict(point=encode(T.GlobalPort(source.unit, port)), effect=effect))
    except _Refuse:
        return None
    return None


def _case(source, syntax, facts, ordinal, binary, shape):
    item_nodes, bases, keys, indices = shape
    outs = [facts.origin(value, facts.items) for value in binary.operands]
    if any(value is None for value in outs):
        return None
    inner = [facts.origin(item[1][1].receiver, facts.items) for item in outs]
    if any(value is None for value in inner):
        return None
    items = (inner[0], outs[0], inner[1], outs[1])
    result_ports = [facts.ports[item[0]] for item in items]
    if [port.port.template.occurrence for port in result_ports] != list(item_nodes):
        return None
    lookups = [item[1][1] for item in inner]
    outer = [item[1][1] for item in outs]
    named = dict()
    for role, operands, occurrences in (
            ('table', [row.receiver for row in lookups], bases),
            ('key', [row.key for row in lookups], keys),
            ('index', [row.key for row in outer], indices)):
        evidence = []
        namespaces = []
        for value, occurrence in zip(operands, occurrences, strict=True):
            if type(value) is not int:
                return None
            if syntax.nodes[occurrence].constructor == 'Constant':
                item = facts.origin(value, facts.constants)
                if item is None or type(item[1][1].literal.to_python()) is not int:
                    return None
                evidence.append(facts.evidence(item, 'constants'))
                continue
            item = facts.origin(value, facts.names)
            if item is None:
                return None
            initial, names, trail = item
            if facts.ports[initial].port.template.occurrence != occurrence:
                return None
            # Global reads can also retain a builtin-miss alternative. That
            # alternative remains a guard obligation, never a resolved table.
            admissible = [(i, name) for i,name in names if name.namespace.kind in
                          (F.NamespaceKind.LEXICAL, F.NamespaceKind.MODULE)]
            if len(admissible) != 1:
                return None
            i, name = admissible[0]
            if name.namespace.name != _name(syntax, occurrence):
                return None
            namespaces.append(name.namespace)
            evidence.append(dict(facts.evidence((initial, (i, name), trail), 'names'), namespace=encode(name.namespace),
                original_read_alternatives=[facts.reference('names', i) for i, _ in names]))
        if role != 'index' and (len(namespaces) != 2 or namespaces[0] != namespaces[1]):
            return None
        named[role] = evidence
    result = facts.ports[binary.result]
    identity = [(port.port.template,port.port.owner) for port in result_ports]
    if len(set(identity)) != 4:
        return None
    effects = _attach_path(source, result, identity)
    if effects is None:
        return None
    reads = [dict(facts.evidence(item, 'items'), role=role,
        world_before=encode(facts.ports[item[1][1].world]), world_after=encode(facts.ports[item[1][1].result_world]))
        for item, role in zip(items, ('FIRST_ROW', 'LEFT_ELEMENT', 'REPEATED_ROW', 'RIGHT_ELEMENT'), strict=True)]
    return dict(operation=binary.selector.value, result=encode(result), operation_fact=facts.reference('operations', ordinal),
        reads=reads, bindings=named, normal_window_effects=effects,
        original_normal_read_order=[row['result'] for row in reads],
        rewrite=dict(replace=reads[2]['result'], reuse=reads[0]['result'], preserve_first_lookup_order=True,
            preserve_both_index_reads=True, same_row_storage_required=True),
        obligations=list(_QUALIFICATION), disposition='CONDITIONAL_REWRITE', automatic_admission=False)


def discover(catalog, source=None):
    """Discover read reuse from original F4 item, scope, data and control records."""
    view = catalog if isinstance(catalog, ProgramView) else ProgramView(catalog)
    catalog = view.catalog
    units = catalog.fact_store.rows if source is None else catalog.units_for_source(source)
    matches = []
    for unit in sorted(units):
        source_view = view.unit(unit)
        model, syntax, codes, recipe, raw_digest = _source_identity(catalog, unit)
        facts = _Facts(catalog, source_view)
        reference = artifact_reference(model['path'], 'scar.python-source')
        _require(reference['sha256'] == raw_digest, 'live source file differs from retained F0')
        for ordinal, binary in enumerate(facts.unit.operations):
            if (ordinal not in facts.functions or binary.action is not F.LanguageOperationKind.BINARY
                    or binary.selector.value not in _OPS or len(binary.operands) != 2):
                continue
            occurrence = facts.ports[binary.result].port.template.occurrence
            shape = _shape(syntax, occurrence)
            if shape is None:
                continue
            function = facts.functions[ordinal]
            name, _, lines = _metadata(syntax, function.template.occurrence)
            compiled = [code for code in codes if code.co_name == name and code.co_firstlineno in lines
                and code.co_varnames[:code.co_argcount + code.co_kwonlyargcount] == function.parameter_names]
            if len(compiled) != 1:
                continue
            case = _case(source_view, syntax, facts, ordinal, binary, shape)
            if case is not None:
                matches.append(dict(case, source=reference, code_key=code_key(compiled[0]), qualname=compiled[0].co_qualname,
                    compile_recipe=recipe, occurrence=occurrence, span=_metadata(syntax, occurrence)[1],
                    repeated_span=_metadata(syntax, shape[0][2])[1], first_span=_metadata(syntax, shape[0][0])[1]))
    return dict(schema=SCHEMA, schema_version=1,
        catalog=artifact_reference(catalog.path/'result.json', catalog.report['schema']),
        source=None if source is None else str(Path(source).absolute()), cases=matches,
        provider_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        target_executed=False, effects_proven_without_guards=False, transformation_authorized=False)


def check(reference):
    """Independently re-derive source receipts and exact replacement edges."""
    result = read_artifact(reference)
    _require(reference['schema'] == SCHEMA, 'expected a guarded scalar table proof artifact')
    read_artifact(result['catalog'])
    expected = discover(Path(result['catalog']['uri']).parent, result['source'])
    _require(expected == result, 'guarded scalar table proof differs from F4 replay')
    return result


def _observable():
    monitoring = getattr(sys, 'monitoring', None)
    return (sys.gettrace() is not None or sys.getprofile() is not None
        or monitoring is not None and any(monitoring.get_tool(i) is not None for i in range(6)))


def try_reuse(operation, table, key, left, right, *, exclusive=False):
    """Evaluate a witnessed expression from already resolved values, or defer.

    The caller still owns the original name bindings and the no-writer window.
    Returning None requests the caller's untouched original expression. This
    helper copies no values and retains no result or table across evaluations.
    """
    if (exclusive is not True or sys.implementation.name != 'cpython' or _observable()
            or not getattr(sys, '_is_gil_enabled', lambda: True)() or type(operation) is not str
            or type(table) is not dict or len(table) > _MAX_KEYS or type(key) is not str
            or type(left) is not int or type(right) is not int):
        return None
    apply = _OPS.get(operation)
    if apply is None or any(type(item) is not str for item in table):
        return None
    row = table.get(key, _MISSING)
    if type(row) is not tuple or not (-len(row) <= left < len(row) and -len(row) <= right < len(row)):
        return None
    a, b = row[left], row[right]
    if type(a) is not int or type(b) is not int:
        return None
    return apply(a,b)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--source')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    result = discover(args.catalog, args.source)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(cases=len(result['cases']), receipt=artifact_reference(out, SCHEMA))))


if __name__ == '__main__':
    main()

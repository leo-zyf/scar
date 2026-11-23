"""Compile-only CPython material and composable Python source regions."""

import ast
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from hashlib import sha256
from io import BytesIO
import math
from pathlib import Path
import symtable
import sys
import tokenize

from scar.contract import (
    Edge,
    Graph,
    GraphProgram,
    Operation,
    Region,
    artifact,
    read_artifact,
    write_json,
)

from .python_ir import project_elementwise_ir, project_scalar_exports

_FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)
_SCOPE = (*_FUNCTION, ast.ClassDef, ast.Lambda)
_SCOPE_EFFECTS = ("STATE_READ", "MAY_RAISE", "UNKNOWN")
_TERMINATORS = {"RETURN", "RAISE", "BREAK", "CONTINUE", "UNREACHABLE"}


def _span(node):
    if not hasattr(node, "lineno"):
        return None
    return [node.lineno, node.col_offset, node.end_lineno, node.end_col_offset]


def _symbols(table):
    role = table.get_type()
    return {
        "name": table.get_name(),
        "scope": getattr(role, "value", role),
        "line": table.get_lineno(),
        "optimized_locals": table.is_optimized(),
        "bindings": [
            {
                "name": symbol.get_name(),
                "roles": [
                    flag
                    for flag in (
                        "parameter",
                        "local",
                        "global",
                        "nonlocal",
                        "free",
                        "imported",
                        "assigned",
                        "referenced",
                    )
                    if getattr(symbol, f"is_{flag}")()
                ],
            }
            for symbol in table.get_symbols()
        ],
        "children": [_symbols(child) for child in table.get_children()],
    }


def _stores(nodes):
    names, pending = set(), list(nodes)
    while pending:
        node = pending.pop()
        if isinstance(node, _SCOPE):
            if hasattr(node, "name"):
                names.add(node.name)
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        pending.extend(ast.iter_child_nodes(node))
    return names


def _parameters(args):
    return (
        *args.posonlyargs,
        *args.args,
        *([args.vararg] if args.vararg else []),
        *args.kwonlyargs,
        *([args.kwarg] if args.kwarg else []),
    )


class _Graph:
    def __init__(self, unit, digest):
        self.unit = unit
        identity = sha256((unit + "\0" + digest).encode()).hexdigest()
        self.root = f"python:{identity}:module"
        self.regions = {self.root: ("MODULE", unit)}
        self.children = defaultdict(list)
        self.owners = defaultdict(list)
        self.operations = {}
        self.producers = {}
        self.edges = []
        self.calls = []
        self.definitions = defaultdict(list)

    def region(self, parent, kind, name):
        identity = f"{parent}:r{len(self.children[parent])}"
        self.regions[identity] = kind, name
        self.children[parent].append(identity)
        return identity

    def emit(
        self,
        owner,
        kind,
        node,
        inputs=(),
        *,
        name=None,
        result=True,
        effects=("UNKNOWN",),
        attributes=None,
        ordered=True,
    ):
        siblings = self.owners[owner]
        identity = f"{owner}:op{len(siblings)}"
        value = f"{identity}:result" if result else None
        operation = Operation(
            identity,
            owner,
            kind,
            name or type(node).__name__,
            tuple(item for item in inputs if item is not None),
            (value,) if result else (),
            effects,
            {"unit": self.unit, "source_span": _span(node), **(attributes or {})},
        )
        if (
            ordered
            and siblings
            and self.operations[siblings[-1]].kind not in _TERMINATORS
        ):
            self.edges.append(Edge(siblings[-1], identity, "CONTROL"))
        for item in operation.inputs:
            if item in self.producers:
                self.edges.append(Edge(self.producers[item], identity, "DATA", item))
        if result:
            self.producers[value] = identity
        self.operations[identity] = operation
        siblings.append(identity)
        return value

    def last(self, region):
        return self.owners[region][-1] if self.owners[region] else region

    def entry(self, region):
        return self.owners[region][0] if self.owners[region] else region

    def _expression(self, node, region, bindings):
        match node:
            case ast.Constant(value=value):
                primitive = (
                    type(value) in {type(None), bool, int, str}
                    or type(value) is float
                    and math.isfinite(value)
                )
                attributes = {"dtype": type(value).__name__}
                if primitive:
                    attributes["value"] = value
                else:
                    attributes["representation"] = repr(value)
                return self.emit(
                    region,
                    "CONSTANT",
                    node,
                    effects=("PURE",) if primitive else ("UNKNOWN",),
                    attributes=attributes,
                )
            case ast.Name(id=name):
                source = bindings.get(name)
                return self.emit(
                    region,
                    "READ",
                    node,
                    (source,),
                    name=name,
                    effects=_SCOPE_EFFECTS,
                    attributes={
                        "binding": "REGION_VALUE" if source else "RUNTIME_SCOPE"
                    },
                )
            case ast.BinOp(left=left, op=op, right=right):
                operands = [
                    self._expression(value, region, bindings) for value in (left, right)
                ]
                return self.emit(
                    region,
                    "ARITHMETIC",
                    node,
                    operands,
                    name=type(op).__name__,
                    effects=_SCOPE_EFFECTS,
                )
            case ast.Call(func=func, args=args, keywords=keywords):
                callable_ = self._expression(func, region, bindings)
                operands = [
                    self._expression(value, region, bindings)
                    for value in (*args, *(keyword.value for keyword in keywords))
                ]
                value = self.emit(
                    region,
                    "CALL",
                    node,
                    (callable_, *operands),
                    attributes={
                        "dispatch": "RUNTIME",
                        "callee_syntax": ast.unparse(func),
                        "keywords": [keyword.arg for keyword in keywords],
                    },
                )
                if isinstance(func, ast.Name) and func.id not in bindings:
                    self.calls.append((self.producers[value], func.id))
                return value
            case ast.Attribute(value=base, attr=name):
                return self.emit(
                    region,
                    "READ",
                    node,
                    (self._expression(base, region, bindings),),
                    name=name,
                    effects=_SCOPE_EFFECTS,
                    attributes={"binding": "ATTRIBUTE_PROTOCOL"},
                )
            case ast.Subscript(value=base, slice=index):
                operands = [
                    self._expression(value, region, bindings) for value in (base, index)
                ]
                return self.emit(
                    region,
                    "READ",
                    node,
                    operands,
                    effects=_SCOPE_EFFECTS,
                    attributes={"binding": "INDEX_PROTOCOL"},
                )
            case _:
                # Keep the full nested syntax for dynamic binding, protocol and
                # short-circuit expressions outside the structured graph profile.
                return self.emit(
                    region, "SYNTAX", node, attributes={"ast": ast.dump(node)}
                )

    def _bind(self, node, value, region, bindings):
        if isinstance(node, ast.Name):
            bindings[node.id] = self.emit(
                region, "ASSIGN", node, (value,), name=node.id, effects=("STATE_WRITE",)
            )
        elif isinstance(node, (ast.Tuple, ast.List)):
            iterable = self.emit(
                region,
                "UNPACK",
                node,
                (value,),
                effects=_SCOPE_EFFECTS,
                attributes={"elements": len(node.elts)},
            )
            for index, item in enumerate(node.elts):
                element = self.emit(
                    region, "ELEMENT", item, (iterable,), attributes={"index": index}
                )
                self._bind(item, element, region, bindings)
        else:
            address = self._target_address(node, region, bindings)
            self.emit(
                region,
                "ASSIGN",
                node,
                (address, value),
                result=False,
                effects=("STATE_WRITE", "UNKNOWN"),
            )

    def _target_address(self, node, region, bindings):
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            children = (
                (node.value, node.slice)
                if isinstance(node, ast.Subscript)
                else (node.value,)
            )
            inputs = [self._expression(child, region, bindings) for child in children]
            return self.emit(
                region,
                "TARGET_ADDRESS",
                node,
                inputs,
                attributes={"attribute": getattr(node, "attr", None)},
            )
        return self.emit(region, "SYNTAX", node, attributes={"ast": ast.dump(node)})

    def _merge(self, owner, site, bindings, alternatives, incoming_control):
        names = sorted(set().union(bindings, *alternatives))
        self.emit(
            owner, "MERGE", site, result=False, effects=("CONTROL",), ordered=False
        )
        merge = self.last(owner)
        self.edges.extend(Edge(source, merge, "CONTROL") for source in incoming_control)
        for name in names:
            choices = tuple(state.get(name) for state in alternatives)
            if all(choice == choices[0] for choice in choices):
                if choices[0] is not None:
                    bindings[name] = choices[0]
            else:
                bindings[name] = self.emit(
                    owner,
                    "PHI",
                    site,
                    choices,
                    name=name,
                    effects=("CONTROL",),
                    attributes={"may_be_unbound": None in choices},
                )

    def _loop(self, node, owner, bindings):
        initial = dict(bindings)
        for_loop = isinstance(node, (ast.For, ast.AsyncFor))
        source = self._expression(node.iter, owner, bindings) if for_loop else None
        self.emit(
            owner,
            "LOOP_ENTER",
            node,
            (source,),
            result=False,
            effects=("CONTROL", "UNKNOWN"),
        )
        entry = self.last(owner)
        region = self.region(owner, "LOOP", type(node).__name__)
        active = dict(bindings)
        assigned = _stores([*node.body, *([node.target] if for_loop else [])])
        carried = {
            name: self.emit(
                region,
                "PHI",
                node,
                (bindings.get(name),),
                name=name,
                effects=("CONTROL",),
                attributes={
                    "loop_carried": True,
                    "may_be_unbound": name not in bindings,
                },
            )
            for name in sorted(assigned)
        }
        active.update(carried)
        condition = None if for_loop else self._expression(node.test, region, active)
        iteration = self.emit(
            region,
            "ITERATION",
            node,
            (source, condition),
            effects=("CONTROL", "UNKNOWN"),
        )
        header = self.producers[iteration]
        self.edges.append(Edge(entry, self.entry(region), "CONTROL"))
        if for_loop:
            self._bind(node.target, iteration, region, active)
        self._body(node.body, region, active)
        latch = self.last(region)
        if self.operations[latch].kind not in {"RETURN", "RAISE", "BREAK"}:
            self.edges.append(Edge(latch, self.entry(region), "CONTROL_BACKEDGE"))
        for name, value in carried.items():
            if (end := active.get(name)) is None or end == value:
                continue
            identity = self.producers[value]
            op = self.operations[identity]
            self.operations[identity] = replace(op, inputs=(*op.inputs, end))
            self.edges.append(Edge(self.producers[end], identity, "DATA_BACKEDGE", end))
        self._merge(owner, node, bindings, (initial, active), (header, latch))
        if node.orelse:
            tail = self.region(owner, "BLOCK", "loop else")
            self._body(node.orelse, tail, dict(bindings))
            self.edges.append(Edge(header, self.entry(tail), "CONTROL"))
            for name in _stores(node.orelse):
                bindings.pop(name, None)

    def _nested_syntax(self, node, owner, bindings):
        self.emit(
            owner,
            "SYNTAX",
            node,
            result=False,
            attributes={"ast_kind": type(node).__name__},
        )
        operation = self.last(owner)
        for field, value in ast.iter_fields(node):
            items = value if isinstance(value, list) else [value]
            statements = [item for item in items if isinstance(item, ast.stmt)]
            bodies = [
                item.body
                for item in items
                if isinstance(item, (ast.ExceptHandler, ast.match_case))
            ]
            for body in [statements, *bodies]:
                if body:
                    child = self.region(owner, "BLOCK", field)
                    self._body(body, child, dict(bindings))
                    self.edges.append(Edge(operation, child, "SYNTAX_CONTAINS", field))

    def _body(self, nodes, region, bindings):
        for node in nodes:
            match node:
                case ast.FunctionDef() | ast.AsyncFunctionDef():
                    supplied = (
                        *node.decorator_list,
                        *node.args.defaults,
                        *(
                            argument
                            for argument in node.args.kw_defaults
                            if argument is not None
                        ),
                    )
                    values = [
                        self._expression(value, region, bindings) for value in supplied
                    ]
                    declaration = self.emit(
                        region,
                        "DEFINE",
                        node,
                        values,
                        name=node.name,
                        effects=("STATE_WRITE", "UNKNOWN"),
                    )
                    bindings[node.name] = declaration
                    function = self.region(region, "FUNCTION", node.name)
                    self.edges.append(
                        Edge(self.producers[declaration], function, "DECLARATION")
                    )
                    if region == self.root:
                        self.definitions[node.name].append(function)
                    parameters = {
                        argument.arg: self.emit(
                            function,
                            "PARAMETER",
                            argument,
                            name=argument.arg,
                            effects=("FRAME_INPUT",),
                        )
                        for argument in _parameters(node.args)
                    }
                    self._body(node.body, function, parameters)
                case ast.ClassDef():
                    values = [
                        self._expression(value, region, bindings)
                        for value in (
                            *node.decorator_list,
                            *node.bases,
                            *(keyword.value for keyword in node.keywords),
                        )
                    ]
                    declaration = self.emit(
                        region,
                        "DEFINE",
                        node,
                        values,
                        name=node.name,
                        effects=("STATE_WRITE", "UNKNOWN"),
                    )
                    bindings[node.name] = declaration
                    scope = self.region(region, "CLASS", node.name)
                    self.edges.append(
                        Edge(self.producers[declaration], scope, "DECLARATION")
                    )
                    self._body(node.body, scope, {})
                case ast.Assign(targets=targets, value=expr):
                    value = self._expression(expr, region, bindings)
                    for target in targets:
                        self._bind(target, value, region, bindings)
                case ast.AnnAssign(target=target, value=expr) if expr is not None:
                    value = self._expression(expr, region, bindings)
                    self._bind(target, value, region, bindings)
                    self.emit(region, "ANNOTATE", node.annotation, result=False)
                case ast.AugAssign(target=target, op=op, value=expr):
                    address = None
                    if isinstance(target, ast.Name):
                        current = self._expression(target, region, bindings)
                    else:
                        address = self._target_address(target, region, bindings)
                        current = self.emit(
                            region, "READ", target, (address,), effects=_SCOPE_EFFECTS
                        )
                    operand = self._expression(expr, region, bindings)
                    value = self.emit(
                        region,
                        "ARITHMETIC",
                        node,
                        (current, operand),
                        name=type(op).__name__,
                        effects=(*_SCOPE_EFFECTS, "STATE_WRITE"),
                        attributes={"inplace_protocol": True},
                    )
                    if address is None:
                        self._bind(target, value, region, bindings)
                    else:
                        self.emit(
                            region,
                            "ASSIGN",
                            target,
                            (address, value),
                            result=False,
                            effects=("STATE_WRITE", "UNKNOWN"),
                        )
                case ast.Expr(value=expr):
                    self._expression(expr, region, bindings)
                case ast.Return(value=value):
                    result = (
                        self._expression(value, region, bindings)
                        if value is not None
                        else None
                    )
                    self.emit(
                        region,
                        "RETURN",
                        node,
                        (result,),
                        result=False,
                        effects=("CONTROL",),
                    )
                case ast.Raise() | ast.Break() | ast.Continue():
                    self.emit(
                        region,
                        type(node).__name__.upper(),
                        node,
                        result=False,
                        attributes={"ast": ast.dump(node)},
                    )
                case ast.For() | ast.AsyncFor() | ast.While():
                    self._loop(node, region, bindings)
                case ast.If(test=test, body=body, orelse=otherwise):
                    test_value = self._expression(test, region, bindings)
                    self.emit(
                        region,
                        "BRANCH",
                        node,
                        (test_value,),
                        result=False,
                        effects=("CONTROL", "UNKNOWN"),
                    )
                    branch = self.last(region)
                    states, exits = [], []
                    for label, statements in (("true", body), ("false", otherwise)):
                        child = self.region(region, "BLOCK", label)
                        state = dict(bindings)
                        self._body(statements, child, state)
                        last = self.last(child)
                        if (
                            last == child
                            or self.operations[last].kind not in _TERMINATORS
                        ):
                            states.append(state)
                            exits.append(last)
                        self.edges.append(
                            Edge(branch, self.entry(child), "CONTROL", label)
                        )
                    if exits:
                        self._merge(region, node, bindings, states, exits)
                    else:
                        self.emit(
                            region,
                            "UNREACHABLE",
                            node,
                            result=False,
                            effects=("CONTROL",),
                            ordered=False,
                        )
                case ast.Import() | ast.ImportFrom():
                    self.emit(
                        region,
                        "IMPORT",
                        node,
                        result=False,
                        effects=("IO", "STATE_WRITE", "UNKNOWN"),
                        attributes={"ast": ast.dump(node)},
                    )
                case _:
                    self._nested_syntax(node, region, bindings)
                    for name in _stores([node]):
                        bindings.pop(name, None)

    def project(self, tree):
        self._body(tree.body, self.root, {})
        for operation, name in self.calls:
            targets = self.definitions[name]
            if len(targets) == 1:
                self.edges.append(Edge(operation, targets[0], "CALL_CANDIDATE", name))
        regions = tuple(
            Region(
                identity,
                kind,
                name,
                tuple(self.owners[identity]),
                tuple(self.children[identity]),
            )
            for identity, (kind, name) in self.regions.items()
        )
        return (
            regions,
            tuple(self.operations.values()),
            tuple(dict.fromkeys(self.edges)),
        )


def build_python(path, out, *, target=None, manifest=None, declarations=None):
    if target is not None:
        if target.get("schema_version") != 1:
            raise ValueError("Python manifest requires scar.python-entry v1")
        root = (path.parent / target.get("source_root", ".")).resolve(strict=True)
        name = target.get("source")
        if not isinstance(name, str) or Path(name).is_absolute():
            raise ValueError(
                "Python manifest source must be a path relative to source_root"
            )
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or path.suffix != ".py":
            raise ValueError(
                "Python manifest source must be a Python file within source_root"
            )
        target.update(
            language="python",
            source_root=str(root),
            entry=target.get("entry"),
            units=[
                {
                    "path": path.relative_to(root).as_posix(),
                    "language": "python",
                    "flags": [],
                }
            ],
        )
    reference = artifact(path)
    raw = read_artifact(reference)
    root = path.parent if target is None else Path(target["source_root"])
    unit = path.relative_to(root).as_posix()
    encoding, _ = tokenize.detect_encoding(BytesIO(raw).readline)
    text = raw.decode(encoding)
    tokens = [
        {
            "kind": tokenize.tok_name.get(token.exact_type, str(token.exact_type)),
            "start": list(token.start),
            "end": list(token.end),
        }
        for token in tokenize.tokenize(BytesIO(raw).readline)
    ]
    tree = ast.parse(text, filename=str(path), type_comments=True)
    compile(tree, str(path), "exec", dont_inherit=True)
    bindings = _symbols(symtable.symtable(text, str(path), "exec"))
    materials = write_json(
        out / "material.json",
        {
            "schema": "scar.cpython-source-material",
            "schema_version": 1,
            "source": asdict(reference),
            "encoding": encoding,
            "interpreter": sys.implementation.name,
            "version": list(sys.version_info),
            "tokens": tokens,
            "ast": ast.dump(tree, include_attributes=True),
            "symbols": bindings,
        },
    )
    if target is None:
        target = {
            "language": "python",
            "source_root": str(root),
            "entry": None,
            "units": [{"path": unit, "language": "python", "flags": []}],
        }
    lock = {"files": {unit: {"sha256": reference.sha256, "size_bytes": reference.size}}}
    if "source_lock" in target and target["source_lock"] != lock:
        raise ValueError(f"Python source differs from manifest lock: {unit}")
    target["source_lock"] = lock
    regions, operations, edges = _Graph(unit, reference.sha256).project(tree)
    evidence = {
        "input": asdict(manifest or reference),
        "locked_sources": {unit: asdict(reference)},
        "source_material": asdict(materials),
        "coverage": {
            "source_resolution": "CPYTHON_AST_AND_SYMBOL_TABLE",
            "runtime": "NOT_CAPTURED",
            "unexpanded_protocols": dict(
                Counter(row.name for row in operations if row.kind == "SYNTAX")
            ),
        },
    }
    portable = project_elementwise_ir(text, target.get("entry"), tree=tree)
    if portable:
        target["entry"] = portable["entry"]
    functions = ()
    if "scalar_abi" in target:
        functions = project_scalar_exports(
            tree, target["scalar_abi"], declarations or {}
        )
        module = regions[0]
        members = {
            region.name: region.id
            for region in regions
            if region.id in module.children and region.kind == "FUNCTION"
        }
        for function in functions:
            function["region"] = members[function["source_name"]]
    read_artifact(reference)
    if manifest:
        read_artifact(manifest)
    program = GraphProgram(scalars=tuple(functions), elementwise=portable)
    return Graph(target, regions, operations, edges, (), evidence, program)

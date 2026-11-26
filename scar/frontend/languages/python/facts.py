"""F1: discover local imports and capture syntax, lexical scopes and source bindings."""

import ast
import builtins
from bisect import bisect_right
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, replace
import dis
import inspect
from functools import cache
from itertools import pairwise
from pathlib import Path
import symtable
import sys
from types import CodeType
import unicodedata

from scar.contract import digest, json_bytes
from scar.source_contracts import python_binary_operator, python_unary_operator
from .environment import code_digest
from .imports import (
    Imports,
    numeric_wire_value,
    relative_module as _relative,
    source_member,
)


@dataclass(frozen=True)
class LexicalScope:
    name: str
    line: int
    local: frozenset
    declared_global: frozenset
    free: frozenset
    cell: frozenset
    forwarded: frozenset
    children: dict

    @classmethod
    def capture(cls, table):
        symbols = table.get_symbols()
        children = [cls.capture(c) for c in table.get_children()]
        local = frozenset(s.get_name() for s in symbols if s.is_local())
        nested_free = frozenset(
            n for child in children for n in child.free | child.forwarded
        )
        is_class = table.get_type() == "class"
        return cls(
            table.get_name(),
            table.get_lineno(),
            local,
            frozenset(s.get_name() for s in symbols if s.is_declared_global()),
            frozenset(s.get_name() for s in symbols if s.is_free() or s.is_nonlocal()),
            nested_free & ({"__class__"} if is_class else local),
            nested_free - {"__class__"} if is_class else frozenset(),
            {(c.name, c.line): c for c in children},
        )


@dataclass(frozen=True)
class ParsedUnit:
    path: Path
    root: Path
    abi: dict
    tree: ast.Module | None
    scopes: LexicalScope | None
    builtins: frozenset
    libraries: frozenset
    exports: dict
    source_hash: str
    diagnostics: tuple = ()
    providers: dict = field(default_factory=dict)
    runtime_facts: dict = field(default_factory=dict)
    source_text: str = ""
    text_profiles: dict = field(default_factory=dict)
    tuple_sites: dict = field(default_factory=dict)
    source_key: str = ""
    source_origins: dict = field(default_factory=dict)
    import_value_uses: frozenset = frozenset()
    mapping_builders: dict = field(default_factory=dict)
    constant_stores: dict = field(default_factory=dict)
    module_docstring: dict = field(default_factory=dict)
    source_code: dict = field(default_factory=dict)
    frame_exit: dict = field(default_factory=dict)
    name_loads: dict = field(default_factory=dict)
    import_stores: dict = field(default_factory=dict)
    statement_expressions: dict = field(default_factory=dict)
    definition_sites: dict = field(default_factory=dict)
    class_sites: dict = field(default_factory=dict)
    class_body_sites: dict = field(default_factory=dict)
    function_expression_sites: dict = field(default_factory=dict)
    exception_handlers: dict = field(default_factory=dict)

    @property
    def source_digests(self):
        return {self.source_key: self.source_hash}

    compiler_id = f"{sys.implementation.name} {sys.version.split()[0]}"


def tuple_position(node):
    return node.lineno, node.end_lineno, node.col_offset, node.end_col_offset


def _constant_syntax(node):
    """Project literal syntax onto the same ordered operands as the neutral graph."""
    if isinstance(node, ast.Constant) and type(node.value) in (
        str,
        int,
        float,
        bool,
        type(None),
    ):
        return ["literal", type(node.value).__name__, numeric_wire_value(node.value)]
    if isinstance(node, (ast.UnaryOp, ast.BinOp)):
        unary = isinstance(node, ast.UnaryOp)
        children = (node.operand,) if unary else (node.left, node.right)
        parts = [_constant_syntax(child) for child in children]
        if all(part is not None for part in parts):
            return [
                "unary" if unary else "binary",
                type(node.op).__name__.lower(),
                *parts,
            ]


def _constant_store_rhs(value):
    if isinstance(value, ast.Constant):
        return type(value.value) is str
    return _constant_syntax(value) is not None


def _constant_store_sites(
    program, instructions, assignments, handler_targets, docstring
):
    """Identify a single source LOAD_CONST feeding the precise module store."""
    by_target = {
        tuple_position(node.targets[0]): (node, node.targets[0].id)
        for node in assignments
    }
    document = {}
    if docstring is not None:
        by_target[tuple_position(docstring.value)] = (docstring, "__doc__")
    doc_write = False
    # Single-target assignments have disjoint RHS ranges. Count every positioned
    # instruction in each folded RHS, including instructions before its last load.
    folded = sorted(
        (v.lineno, v.col_offset, v.end_lineno, v.end_col_offset)
        for a in assignments
        if not isinstance(v := a.value, ast.Constant)
    )
    starts = [(line, col) for line, col, _, _ in folded]
    counts = Counter()
    sites = {}
    load = None
    for instruction in instructions:
        pos = instruction.positions
        if folded and None not in pos:
            start = pos.lineno, pos.col_offset
            ix = bisect_right(starts, start) - 1
            if ix >= 0 and start < folded[ix][2:]:
                counts[folded[ix]] += 1
        opcode = getattr(instruction, "baseopname", instruction.opname)
        if instruction.is_jump_target or instruction.offset in handler_targets:
            load = None
        if opcode == "EXTENDED_ARG":
            continue
        target = by_target.get(pos)
        if (
            opcode in {"STORE_NAME", "STORE_GLOBAL"}
            and docstring is not None
            and instruction.argval == "__doc__"
        ):
            # Missing column information can make an explicit later write on
            # the same line indistinguishable from the implicit one.
            doc_write |= pos.lineno is None or (
                docstring.lineno <= pos.lineno <= docstring.end_lineno
                and (None in pos or pos == tuple_position(docstring.value))
            )
        if target is not None and opcode in {"STORE_NAME", "STORE_GLOBAL"}:
            node, name = target
            span = tuple_position(node)
            # A finally suite may give one source target several bytecode stores.
            # A match at one copy cannot serve as a receipt for all copies.
            if span in sites:
                sites[span] = None
            elif (
                load is not None
                and load.positions == tuple_position(node.value)
                and type(value := program.co_consts[load.arg])
                in (str, int, float, bool, type(None))
                and (
                    not isinstance(node.value, ast.Constant)
                    or type(value) is str
                    and (node is docstring or value == node.value.value)
                )
                and program.co_names[instruction.arg] == name
            ):
                sites[span] = dict(
                    semantics_version=1,
                    load=dict(
                        opcode="LOAD_CONST",
                        index=load.arg,
                        offset=load.offset,
                        position=list(load.positions),
                        type=type(value).__name__,
                        value_sha256=digest(json_bytes(numeric_wire_value(value))),
                    ),
                    store=dict(
                        opcode=opcode,
                        index=instruction.arg,
                        name=name,
                        offset=instruction.offset,
                        position=list(instruction.positions),
                    ),
                )
                if node is docstring:
                    document["value"] = value
            else:
                sites[span] = None
        load = instruction if opcode == "LOAD_CONST" else None
    for a in assignments:
        rhs = a.value
        if (
            not isinstance(rhs, ast.Constant)
            and counts[(rhs.lineno, rhs.col_offset, rhs.end_lineno, rhs.end_col_offset)]
            != 1
        ):
            sites.pop(tuple_position(a), None)
    stores = {span: receipt for span, receipt in sites.items() if receipt is not None}
    if docstring is not None and tuple_position(docstring) not in stores:
        document = (
            {"omitted": True} if sys.flags.optimize == 2 and not doc_write else {}
        )
    return stores, document


_FAST_READS = {"LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_BORROW"}


def _name_load_sites(program, instructions, names):
    """Pair a module-frame read with its unique lexical name and instruction."""
    sites = {}
    for ins in instructions:
        opcode = getattr(ins, "baseopname", ins.opname)
        span = ins.positions
        if (
            opcode not in _FAST_READS and opcode not in {"LOAD_NAME", "LOAD_GLOBAL"}
        ) or span not in names:
            continue
        # Since CPython 3.11 the low LOAD_GLOBAL argument bit is a call flag.
        index = ins.arg >> 1 if opcode == "LOAD_GLOBAL" else ins.arg
        table = program.co_varnames if opcode in _FAST_READS else program.co_names
        if (
            span in sites
            or not 0 <= index < len(table)
            or table[index] != ins.argval
            or ins.argval != names[span]
        ):
            sites[span] = None
        else:
            sites[span] = dict(
                semantics_version=1,
                opcode=opcode,
                name=ins.argval,
                index=index,
                oparg=ins.arg,
                offset=ins.offset,
                position=list(span),
            )
    return {span: receipt for span, receipt in sites.items() if receipt is not None}


def _import_layout(node):
    """Expected stack consumers derived from exactly one source import statement."""
    if isinstance(node, ast.ImportFrom):
        steps = []
        for alias in node.names:
            steps.extend(
                [("IMPORT_FROM", alias.name), ("store", alias.asname or alias.name)]
            )
        steps.append(("POP_TOP", None))
        return node.level, tuple(a.name for a in node.names), node.module or "", steps
    alias = node.names[0]
    members = alias.name.split(".")[1:] if alias.asname else []
    steps = []
    for member in members:
        if steps:
            steps.extend([("SWAP", 2), ("POP_TOP", None)])
        steps.append(("IMPORT_FROM", member))
    steps.append(("store", alias.asname or alias.name.partition(".")[0]))
    if members:
        steps.append(("POP_TOP", None))
    return 0, None, alias.name, steps


def _import_operand(ins):
    row = dict(opcode=ins.opname, offset=ins.offset)
    if ins.opname in {"IMPORT_NAME", "IMPORT_FROM", "STORE_NAME", "STORE_GLOBAL"}:
        row.update(index=ins.arg, name=ins.argval)
    elif ins.opname == "SWAP":
        row["oparg"] = ins.arg
    return row


def _import_store_sites(program, instructions, imports, handlers):
    """Capture one absolute import or named fromlist, through its final consumer."""
    regions = defaultdict(list)
    ordinal, incoming = 0, False
    for ins in instructions:
        incoming |= ins.is_jump_target or ins.offset in handlers
        if ins.opname == "EXTENDED_ARG":
            continue
        if ins.positions in imports:
            regions[ins.positions].append((ordinal, ins, incoming))
        ordinal += 1
        incoming = False
    sites = {}
    for span, region in regions.items():
        requests = [
            i for i, (_, ins, _) in enumerate(region) if ins.opname == "IMPORT_NAME"
        ]
        if len(requests) != 1 or requests[0] < 2:
            continue
        depth, members, module, steps = _import_layout(imports[span])
        size = 3 + len(steps)
        start = requests[0] - 2
        rows = region[start : start + size]
        if (
            len(rows) != size
            or rows[-1][0] - rows[0][0] != size - 1
            or any(incoming for _, _, incoming in rows[1:])
        ):
            continue
        level, fromlist, request, *tail = (ins for _, ins, _ in rows)
        if (
            fromlist.opname != "LOAD_CONST"
            or program.co_consts[fromlist.arg] != members
            or program.co_names[request.arg] != module
            or any(
                (
                    ins.opname not in {"STORE_NAME", "STORE_GLOBAL"}
                    if opcode == "store"
                    else ins.opname != opcode
                )
                or ins.argval != arg
                for ins, (opcode, arg) in zip(tail, steps, strict=True)
            )
        ):
            continue
        if (
            level.opname == "LOAD_CONST"
            and type(program.co_consts[level.arg]) is int
            and program.co_consts[level.arg] == depth
        ):
            source_level = {"index": level.arg}
        elif level.opname == "LOAD_SMALL_INT" and level.arg == depth:
            source_level = {"oparg": level.arg}
        else:
            continue
        proof = dict(
            semantics_version=1,
            position=list(span),
            level=dict(
                opcode=level.opname, offset=level.offset, value=depth, **source_level
            ),
            fromlist=dict(
                opcode="LOAD_CONST",
                index=fromlist.arg,
                offset=fromlist.offset,
                value=list(members) if members is not None else None,
            ),
            import_name=_import_operand(request),
        )
        result = [_import_operand(ins) for ins in tail]
        if result[-1]["opcode"] == "POP_TOP":
            proof["release"] = result.pop()
        if members is not None:
            proof["bindings"] = [
                dict(read=read, store=store)
                for read, store in zip(result[::2], result[1::2], strict=True)
            ]
        else:
            proof["store"] = result.pop()
            if result:
                proof["projection"] = result
        sites[span] = proof
    return sites


_STATEMENT_END = {
    ast.Constant: {"NOP"},
    ast.Call: {"POP_TOP"},
    ast.Raise: {"RAISE_VARARGS"},
    ast.Return: {"RETURN_VALUE", "RETURN_CONST"},
    ast.Assign: {"STORE_NAME", "STORE_GLOBAL", "STORE_FAST", "STORE_SUBSCR"},
    ast.Delete: {"DELETE_NAME", "DELETE_GLOBAL"},
    ast.If: {"POP_JUMP_IF_TRUE", "POP_JUMP_IF_FALSE"},
    ast.For: {"GET_ITER"},
}


def _pool_literal_items(value, elements):
    """Bind the ordered scalar operands to the objects in a compiler tuple."""
    if type(value) is not tuple or len(value) != len(elements):
        return None
    if not all(
        isinstance(node, ast.Constant)
        and type(item) in (str, int, float, bool, type(None))
        and type(node.value) is type(item)
        and node.value == item
        for node, item in zip(elements, value)
    ):
        return None
    tokens = [
        ["literal", type(item).__name__, numeric_wire_value(item)] for item in value
    ]
    return dict(
        type="tuple", item_count=len(value), items_sha256=digest(json_bytes(tokens))
    )


def _pool_display_items(value, parts):
    for display in parts:
        if not isinstance(display, (ast.List, ast.Tuple)) or len(value) != len(
            display.elts
        ):
            continue
        if items := _pool_literal_items(value, display.elts):
            return items
        syntax = [_constant_syntax(node) for node in display.elts]
        if None in syntax or any(
            type(v) not in (str, int, float, bool, type(None)) for v in value
        ):
            continue
        values = [
            dict(kind=type(v).__name__, value=numeric_wire_value(v)) for v in value
        ]
        tokens = [["literal", v["kind"], v["value"]] for v in values]
        if any(
            expr[0] == "literal" and json_bytes(expr) != json_bytes(token)
            for expr, token in zip(syntax, tokens)
        ):
            continue
        return dict(
            type="tuple",
            item_count=len(value),
            folded_items=values,
            items_sha256=digest(json_bytes(tokens)),
            item_sources=[digest(json_bytes(expr)) for expr in syntax],
        )
    return None


def _slice_key(value, node):
    """Describe an actual pool constant only when every slice component matches the syntax."""
    if type(value) is slice and isinstance(node, ast.Slice):
        kind, actual, syntax = (
            "slice",
            (value.start, value.stop, value.step),
            (node.lower, node.upper, node.step),
        )
    elif (
        type(value) is tuple
        and isinstance(node, ast.Tuple)
        and len(value) == len(node.elts)
    ):
        kind, actual, syntax = "tuple", value, node.elts
    elif (
        node is None
        and value is None
        or isinstance(node, ast.Constant)
        and type(node.value) is type(value)
        and node.value == value
    ):
        return dict(kind=type(value).__name__, value=numeric_wire_value(value))
    else:
        return None
    items = [_slice_key(item, part) for item, part in zip(actual, syntax)]
    if all(items):
        return dict(
            kind=kind,
            **(
                dict(zip(("lower", "upper", "step"), items))
                if kind == "slice"
                else {"items": items}
            ),
        )


def _expression_instruction(program, ins, parts, keywords):
    opcode = getattr(ins, "baseopname", ins.opname)
    arg, span = ins.arg, ins.positions
    row = dict(opcode=opcode, offset=ins.offset, position=list(span))
    if arg is not None:
        row["oparg"] = arg
    if opcode in _FAST_READS or opcode in {
        "STORE_FAST",
        "LOAD_NAME",
        "LOAD_GLOBAL",
        "LOAD_ATTR",
        "STORE_NAME",
        "STORE_GLOBAL",
        "DELETE_NAME",
        "DELETE_GLOBAL",
    }:
        index = arg >> 1 if opcode in {"LOAD_GLOBAL", "LOAD_ATTR"} else arg
        table = (
            program.co_varnames
            if opcode in _FAST_READS or opcode == "STORE_FAST"
            else program.co_names
        )
        if not 0 <= index < len(table) or table[index] != ins.argval:
            return None
        name = table[index]
        kind, field = (
            (ast.Attribute, "attr") if opcode == "LOAD_ATTR" else (ast.Name, "id")
        )
        ctx = (
            ast.Store
            if opcode.startswith("STORE_")
            else ast.Del if opcode.startswith("DELETE_") else ast.Load
        )
        if not any(
            isinstance(n, kind) and isinstance(n.ctx, ctx) and getattr(n, field) == name
            for n in parts
        ):
            return None
        row.update(index=index, name=name)
    elif opcode in {"LOAD_CONST", "LOAD_SMALL_INT", "KW_NAMES", "RETURN_CONST"}:
        value = arg if opcode == "LOAD_SMALL_INT" else program.co_consts[arg]
        if opcode != "LOAD_SMALL_INT":
            row["index"] = arg
        if opcode == "LOAD_CONST" and (
            type(value) is slice
            or type(value) is tuple
            and any(type(item) is slice for item in value)
        ):
            key = next((key for n in parts if (key := _slice_key(value, n))), None)
            if key is None:
                return None
            row.update(type=type(value).__name__, constant_key=key)
        elif type(value) is tuple:
            if value in keywords:
                row.update(type="tuple", names=list(value))
            elif opcode == "LOAD_CONST" and (
                items := _pool_display_items(value, parts)
            ):
                row.update(items)
            else:
                return None
        elif (
            opcode != "KW_NAMES"
            and type(value) in (str, int, float, bool, type(None))
            and (
                syntax := next(
                    (
                        form
                        for part in parts
                        if isinstance(part, (ast.UnaryOp, ast.BinOp))
                        and (form := _constant_syntax(part)) is not None
                    ),
                    None,
                )
            )
        ):
            wire = numeric_wire_value(value)
            row.update(
                type=type(value).__name__,
                folded_value=wire,
                rhs_sha256=digest(json_bytes(syntax)),
                value_sha256=digest(json_bytes(wire)),
            )
        elif opcode != "KW_NAMES" and any(
            isinstance(n, ast.Constant)
            and type(n.value) is type(value)
            and n.value == value
            or value is None
            and isinstance(n, ast.Slice)
            and (n.lower is None or n.upper is None)
            or value is None
            and isinstance(n, ast.Return)
            and n.value is None
            for n in parts
        ):
            row.update(
                type=type(value).__name__,
                value_sha256=digest(json_bytes(numeric_wire_value(value))),
            )
        else:
            return None
    elif (
        opcode in {"BINARY_SUBSCR", "STORE_SUBSCR"}
        or opcode == "BINARY_OP"
        and ins.argrepr == "[]"
    ):
        ctx = ast.Store if opcode == "STORE_SUBSCR" else ast.Load
        if not any(
            isinstance(n, ast.Subscript) and isinstance(n.ctx, ctx) for n in parts
        ):
            return None
        if opcode == "BINARY_OP":
            row["operator"] = "subscript"
    elif opcode == "BUILD_SLICE":
        if not any(
            isinstance(n, ast.Slice) and arg == (3 if n.step is not None else 2)
            for n in parts
        ):
            return None
    elif unary := python_unary_operator(opcode, arg):
        if (unary == "uadd" and ins.argrepr != "INTRINSIC_UNARY_POSITIVE") or not any(
            isinstance(n, ast.UnaryOp) and type(n.op).__name__.lower() == unary
            for n in parts
        ):
            return None
        row["operator"] = unary
    elif opcode == "COMPARE_OP":
        names = {
            "<": "lt",
            "<=": "lte",
            "==": "eq",
            "!=": "noteq",
            ">": "gt",
            ">=": "gte",
        }
        op = names.get(ins.argval)
        if not any(
            isinstance(n, ast.Compare)
            and len(n.ops) == 1
            and type(n.ops[0]).__name__.lower() == op
            for n in parts
        ):
            return None
        row.update(
            operator=op, force_bool=sys.version_info >= (3, 13) and bool(arg & 16)
        )
    elif opcode == "BINARY_OP":
        binary = python_binary_operator(arg)
        if (
            not binary
            or ins.argrepr != binary[1]
            or not any(
                isinstance(n, ast.BinOp) and type(n.op).__name__.lower() == binary[0]
                for n in parts
            )
        ):
            return None
        row["operator"] = binary[0]
    elif opcode in {"CALL", "CALL_KW"}:
        if not any(
            isinstance(n, ast.Call) and arg == len(n.args) + len(n.keywords)
            for n in parts
        ):
            return None
    elif opcode in {
        "FORMAT_VALUE",
        "FORMAT_SIMPLE",
        "FORMAT_WITH_SPEC",
        "CONVERT_VALUE",
    }:
        if not any(isinstance(n, ast.FormattedValue) for n in parts):
            return None
    elif opcode == "BUILD_STRING":
        if not any(isinstance(n, ast.JoinedStr) for n in parts):
            return None
    elif opcode in {
        "BUILD_LIST",
        "BUILD_TUPLE",
        "LIST_APPEND",
        "LIST_EXTEND",
        "CALL_INTRINSIC_1",
    }:
        display = next((n for n in parts if isinstance(n, (ast.List, ast.Tuple))), None)
        if display is None:
            return None
        if opcode == "BUILD_LIST" and not (
            arg == 0 or isinstance(display, ast.List) and arg == len(display.elts)
        ):
            return None
        if opcode == "BUILD_TUPLE" and not (
            isinstance(display, ast.Tuple) and arg == len(display.elts)
        ):
            return None
        if opcode in {"LIST_APPEND", "LIST_EXTEND"} and arg != 1:
            return None
        if opcode == "CALL_INTRINSIC_1":
            if (
                not isinstance(display, ast.Tuple)
                or ins.argrepr != "INTRINSIC_LIST_TO_TUPLE"
            ):
                return None
            row["intrinsic"] = ins.argrepr
    elif opcode in {"POP_JUMP_IF_TRUE", "POP_JUMP_IF_FALSE", "TO_BOOL"}:
        if not any(isinstance(n, ast.If) for n in parts):
            return None
        if opcode != "TO_BOOL":
            row["target"] = ins.argval
    elif opcode == "NOP":
        if not any(isinstance(n, ast.Constant) for n in parts):
            return None
    elif opcode == "RETURN_VALUE":
        pass
    elif opcode == "GET_ITER":
        if not any(isinstance(n, ast.For) for n in parts):
            return None
    elif opcode == "RAISE_VARARGS":
        if not any(
            isinstance(n, ast.Raise)
            and arg == int(n.exc is not None) + int(n.cause is not None)
            for n in parts
        ):
            return None
    elif opcode not in {"PUSH_NULL", "POP_TOP"}:
        return None
    return row


# The final value producer must belong to the complete source operand. Its
# inputs, code slots and stack effect are checked by the expression capture.
_CONDITION_PRODUCERS = {
    ast.Name: _FAST_READS | {"LOAD_GLOBAL"},
    ast.Attribute: {"LOAD_ATTR"},
    ast.Subscript: {"BINARY_SUBSCR", "BINARY_OP"},
    ast.BinOp: {"BINARY_OP"},
    ast.Call: {"CALL", "CALL_KW"},
    ast.Compare: {"COMPARE_OP"},
    ast.List: {"BUILD_LIST", "LIST_APPEND", "LIST_EXTEND"},
    ast.Tuple: {"BUILD_TUPLE", "CALL_INTRINSIC_1", "LOAD_CONST"},
    ast.JoinedStr: {"BUILD_STRING"},
}


def _condition_operand(test, *, function_frame=False):
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        test = test.operand
    if isinstance(test, ast.Compare) and len(test.ops) != 1:
        return None
    if (
        isinstance(test, ast.Call)
        or function_frame
        and type(test) in _CONDITION_PRODUCERS
    ):
        return test


def _within_position(spot, span):
    return (
        None not in spot
        and (span[0], span[2]) <= (spot.lineno, spot.col_offset)
        and (spot.end_lineno, spot.end_col_offset) <= (span[1], span[3])
    )


def _suite_entry(ins, statement, kind, preceding=()):
    span = tuple_position(statement)
    if (
        ins.opname == "EXTENDED_ARG"
        or not _within_position(ins.positions, span)
        or any(_within_position(other.positions, span) for other in preceding)
    ):
        return None
    return dict(
        kind=kind,
        offset=ins.offset,
        opcode=ins.opname,
        position=list(ins.positions),
        statement_position=list(span),
    )


def _implicit_return(instructions, index, span):
    target = instructions[index]
    if (
        target.opname not in {"RETURN_CONST", "LOAD_CONST"}
        or target.argval is not None
        or target.positions != span
    ):
        return None
    ret = target
    if target.opname == "LOAD_CONST":
        if index + 1 >= len(instructions):
            return None
        ret = instructions[index + 1]
        if ret.opname != "RETURN_VALUE" or ret.is_jump_target:
            return None
    return dict(
        kind="implicit_return",
        opcode=target.opname,
        offset=target.offset,
        position=list(span),
        statement_position=None,
        constant_index=target.arg,
        return_offset=ret.offset,
    )


def _loop_backedge(instructions, index, loop, operand_position):
    if not loop:
        return None
    parent, iteration = loop
    start = index
    while instructions[index].opname == "EXTENDED_ARG":
        index += 1
    back = instructions[index]
    backs = iteration.get("backedges") or [iteration["backedge"]]
    if (
        back.positions != operand_position
        or not any(row["offset"] == back.offset for row in backs)
        or any(
            i.positions != operand_position or i.is_jump_target
            for i in instructions[start + 1 : index + 1]
        )
    ):
        return None
    first = instructions[start]
    return dict(
        kind="loop_backedge",
        statement_position=list(tuple_position(parent)),
        offset=first.offset,
        opcode=first.opname,
        position=list(first.positions),
        prefixes=[_positioned_instruction(i) for i in instructions[start:index]],
        backedge_offset=back.offset,
        target=back.argval,
    )


def _suite_exits(instructions, start, stop, suite, destination, handlers):
    """Check every normal transfer out of this suite against one successor."""
    if start >= stop:
        return None
    first, limit = instructions[start].offset, instructions[stop].offset
    result = []
    for i in range(start, stop):
        ins = instructions[i]
        if (
            ins.offset in handlers
            or not any(_within_position(ins.positions, span) for span in suite)
            or ins.opname
            in {
                "RETURN_VALUE",
                "RETURN_CONST",
                "RAISE_VARARGS",
                "RERAISE",
                "YIELD_VALUE",
            }
        ):
            return None
        if ins.opcode in _JUMP_OPCODES:
            if ins.opname not in {
                "JUMP_FORWARD",
                "JUMP_BACKWARD",
                "POP_JUMP_IF_TRUE",
                "POP_JUMP_IF_FALSE",
            }:
                return None
            if not first <= ins.argval < limit:
                if ins.argval != destination:
                    return None
                result.append(dict(_positioned_instruction(ins), target=destination))
            elif ins.argval <= ins.offset:
                return None
            if ins.opname in {"JUMP_FORWARD", "JUMP_BACKWARD"}:
                continue
        if i + 1 == stop:
            if limit != destination:
                return None
            result.append(dict(_positioned_instruction(ins), fallthrough=destination))
    return result or None


def _branch_receipt(
    node, after, instructions, end, offsets, rows, *, function_frame, loop, handlers
):
    """Map the truth jump's two physical successors to their source suites."""
    operand = _condition_operand(node.test, function_frame=function_frame)
    negated = operand is not node.test
    pos = tuple_position(operand)
    branch = instructions[end]
    if branch.opname not in {"POP_JUMP_IF_TRUE", "POP_JUMP_IF_FALSE"}:
        return None
    taken_then = (branch.opname == "POP_JUMP_IF_TRUE") != negated
    if taken_then and not function_frame:
        return None
    # 3.12 truth-tests the object in the jump. Later bytecode must produce an
    # exact bool first. A separate UNARY_NOT is not interchangeable with this lowering.
    truth = int(sys.version_info >= (3, 13) and not isinstance(operand, ast.Compare))
    opcode = _CONDITION_PRODUCERS[type(operand)]
    if (
        len(rows) < 2 + truth
        or rows[-2 - truth]["opcode"] not in opcode
        or isinstance(operand, ast.Compare)
        and rows[-2]["force_bool"] != (sys.version_info >= (3, 13))
        or truth
        and rows[-2]["opcode"] != "TO_BOOL"
        or any(row["position"] != list(pos) for row in rows[-2 - truth :])
    ):
        return None
    fall = end + 1
    if fall >= len(instructions):
        return None
    marker = None
    if instructions[fall].opname == "NOT_TAKEN":
        ins = instructions[fall]
        if ins.positions != pos or ins.is_jump_target:
            return None
        marker = _positioned_instruction(ins)
        fall += 1
    taken = offsets[branch.argval]
    if not fall < taken:
        return None
    then, other = (taken, fall) if taken_then else (fall, taken)

    def destination(index, statement, kind):
        return _suite_entry(
            instructions[index], statement, kind, instructions[fall:index]
        )

    yes = destination(then, node.body[0], "suite")
    if node.orelse:
        no = destination(other, node.orelse[0], "suite")
    elif after is not None:
        no = destination(other, after, "continuation")
    elif loop:
        no = _loop_backedge(instructions, other, loop, pos)
    else:
        no = _implicit_return(instructions, other, pos)
    if not yes or not no:
        return None
    edges = dict(
        condition_position=list(tuple_position(node.test)),
        operand_position=list(pos),
        negated=negated,
        fallthrough_marker=marker,
        then=yes,
        **{"else": no},
    )
    if function_frame:
        edges.update(
            taken="then" if taken_then else "else",
            fallthrough="else" if taken_then else "then",
        )
    if not (function_frame and node.orelse and then < other):
        return edges

    # A branch may have several normal exits, for example the alternatives of
    # an elif that each advance the same enclosing iterator.
    if loop:
        parent, iteration = loop
        head = (iteration.get("next_prefix") or [iteration["next"]])[0]
        join = {k: head[k] for k in ("opcode", "offset", "position")}
        join.update(kind="loop_header", statement_position=list(tuple_position(parent)))
        stop = offsets[iteration["cleanup"][0]["offset"]]
    elif after is not None:
        jump = instructions[other - 1]
        if jump.opname != "JUMP_FORWARD" or jump.argval not in offsets:
            return edges
        stop = offsets[jump.argval]
        join = destination(stop, after, "continuation")
    else:
        return edges
    if not join or stop <= other:
        return edges
    true_exits = _suite_exits(
        instructions,
        then,
        other,
        [tuple_position(n) for n in node.body],
        join["offset"],
        handlers,
    )
    false_exits = _suite_exits(
        instructions,
        other,
        stop,
        [tuple_position(n) for n in node.orelse],
        join["offset"],
        handlers,
    )
    if true_exits and false_exits:
        edges["join"] = join
        if (
            len(true_exits) == len(false_exits) == 1
            and true_exits[0]["opcode"] == "JUMP_FORWARD"
            and "fallthrough" in false_exits[0]
            and _within_position(
                instructions[offsets[true_exits[0]["offset"]]].positions,
                tuple_position(node.body[-1]),
            )
            and _within_position(
                instructions[offsets[false_exits[0]["offset"]]].positions,
                tuple_position(node.orelse[-1]),
            )
        ):
            edges.update(then_exit=true_exits[0], else_exit=false_exits[0])
        else:
            edges["exits"] = {"then": true_exits, "else": false_exits}
    return edges


def _iteration_region(
    node, instructions, end, offsets, jumps, handlers, nested, *, function_frame
):
    if not isinstance(node.target, ast.Name) or any(
        isinstance(part, (ast.Break, ast.Continue))
        for statement in (*node.body, *node.orelse)
        for part in ast.walk(statement)
    ):
        return None
    if function_frame and any(
        isinstance(part, (ast.Return, ast.Yield, ast.YieldFrom))
        for part in scope_nodes(node.body)
    ):
        return None
    acquire = instructions[end]
    if end + 3 >= len(instructions):
        return None
    header = end + 1
    advance = header
    while (
        advance < len(instructions) and instructions[advance].opname == "EXTENDED_ARG"
    ):
        advance += 1
    if advance + 2 >= len(instructions):
        return None
    prefix = instructions[header:advance]
    if prefix and not function_frame:
        return None
    step, store = instructions[advance : advance + 2]
    if (
        acquire.positions not in {tuple_position(node), tuple_position(node.iter)}
        or step.opname != "FOR_ITER"
        or step.positions != acquire.positions
        or store.opname
        not in ({"STORE_FAST"} if function_frame else {"STORE_NAME", "STORE_GLOBAL"})
        or store.positions != tuple_position(node.target)
        or store.argval != node.target.id
    ):
        return None
    exit_index = offsets.get(step.argval)
    if exit_index is None or exit_index < advance + 3:
        return None
    head = instructions[header].offset
    body_code = instructions[advance + 2 : exit_index]
    backs = [
        ins for ins in body_code if ins.opname == "JUMP_BACKWARD" and ins.argval == head
    ]
    back = instructions[exit_index - 1]
    cleanup = instructions[exit_index]
    if (
        back.opname != "JUMP_BACKWARD"
        or back.argval != head
        or cleanup.opname != "END_FOR"
        or cleanup.positions != acquire.positions
        or jumps[head] != [ins.offset for ins in backs]
        or jumps[cleanup.offset] != [step.offset]
        or not _suite_entry(back, node.body[-1], "suite")
    ):
        return None
    stop = exit_index + 1
    if stop < len(instructions) and instructions[stop].opname == "POP_ITER":
        stop += 1
    if stop >= len(instructions):
        return None
    clean = instructions[exit_index:stop]
    if (
        any(i.positions != acquire.positions for i in (*prefix, *clean))
        or any(
            i.offset in handlers
            or i.is_jump_target
            and i.offset not in {head, cleanup.offset}
            and not (
                function_frame
                and body_code[0].offset <= i.offset < cleanup.offset
                and all(
                    body_code[0].offset <= pred < cleanup.offset
                    for pred in jumps[i.offset]
                )
            )
            for i in instructions[header:stop]
        )
        or any(
            i.opcode in _JUMP_OPCODES
            and not (
                function_frame
                and (
                    i.opname == "JUMP_BACKWARD"
                    and i.argval == head
                    or i.opname
                    in {"POP_JUMP_IF_TRUE", "POP_JUMP_IF_FALSE", "JUMP_FORWARD"}
                    and i.offset < i.argval < cleanup.offset
                    or i.offset in nested
                    and body_code[0].offset <= nested[i.offset][0]
                    and nested[i.offset][1] < cleanup.offset
                )
            )
            for i in body_code[:-1]
        )
    ):
        return None
    body = _suite_entry(body_code[0], node.body[0], "suite")
    if not body:
        return None
    return (
        dict(
            next=dict(_positioned_instruction(step), target=step.argval),
            store=_positioned_instruction(store),
            body=body,
            **(
                {"next_prefix": [_positioned_instruction(i) for i in prefix]}
                if prefix
                else {}
            ),
            **(
                {"backedge": dict(_positioned_instruction(back), target=head)}
                if len(backs) == 1
                else {
                    "backedges": [
                        dict(_positioned_instruction(i), target=head) for i in backs
                    ]
                }
            ),
            cleanup=[_positioned_instruction(i) for i in clean],
        ),
        stop,
    )


_JUMP_OPCODES = frozenset(
    dis.hasjump if hasattr(dis, "hasjump") else dis.hasjabs + dis.hasjrel
)


def _expression_parts(expr, *, allow_indexed):
    parts, keywords = defaultdict(list), defaultdict(list)
    object_ports = False
    for node in ast.walk(expr):
        if not allow_indexed and isinstance(
            node, (ast.Subscript, ast.Slice, ast.Compare)
        ):
            return None
        if isinstance(
            node,
            (
                ast.Load,
                ast.Store,
                ast.Del,
                ast.keyword,
                ast.operator,
                ast.unaryop,
                ast.cmpop,
            ),
        ):
            continue
        if not isinstance(
            node,
            (
                ast.Call,
                ast.BinOp,
                ast.UnaryOp,
                ast.Compare,
                ast.Subscript,
                ast.Slice,
                ast.Raise,
                ast.Return,
                ast.List,
                ast.Tuple,
                ast.Assign,
                ast.Delete,
                ast.Name,
                ast.Attribute,
                ast.Constant,
                ast.JoinedStr,
                ast.FormattedValue,
            ),
        ):
            return None
        if isinstance(node, (ast.List, ast.Tuple)) and any(
            isinstance(elt, (ast.List, ast.Tuple, ast.Starred)) for elt in node.elts
        ):
            return None
        if isinstance(node, ast.Constant) and type(node.value) not in (
            int,
            str,
            bool,
            float,
            type(None),
        ):
            return None
        pos = tuple_position(node)
        parts[pos].append(node)
        if isinstance(node, ast.Call) and node.keywords:
            keys = tuple(k.arg for k in node.keywords)
            if None in keys:
                return None
            keywords[pos].append(keys)
            keywords[tuple_position(node.func)].append(keys)
        object_ports |= isinstance(
            node, (ast.Attribute, ast.Constant, ast.JoinedStr, ast.List, ast.Tuple)
        )
    return parts, keywords, object_ports


def _statement_expression(
    program, instructions, root, end, handlers, *, function_frame=False
):
    """Capture source operands and the terminal operation consuming them."""
    expr = root
    if isinstance(root, ast.If):
        expr = _condition_operand(root.test, function_frame=function_frame)
    elif isinstance(root, ast.For):
        expr = root.iter
    span = tuple_position(expr)
    discarded_call = isinstance(root, ast.Call)
    allow_indexed = function_frame or (
        isinstance(root, ast.Assign) and isinstance(root.targets[0], ast.Subscript)
    )
    operands = _expression_parts(expr, allow_indexed=allow_indexed)
    if operands is None:
        return None
    parts, keywords, object_ports = operands
    if discarded_call and not (function_frame or object_ports or keywords):
        return None

    if isinstance(root, (ast.For, ast.If)):
        parts[instructions[end].positions].append(root)
    start = end
    while start > 0:
        previous = instructions[start - 1].positions
        if None in previous or not (
            (span[0], span[2]) <= (previous.lineno, previous.col_offset)
            and (previous.end_lineno, previous.end_col_offset) <= (span[1], span[3])
        ):
            break
        start -= 1
    rows, depth = [], 0
    for index in range(start, end + 1):
        ins = instructions[index]
        if (
            ins.positions not in parts
            or index > start
            and (ins.is_jump_target or ins.offset in handlers)
        ):
            return None
        if ins.opname in _STATEMENT_END[ast.If] or ins.opname == "TO_BOOL":
            if depth != 1 or index != end - int(ins.opname == "TO_BOOL"):
                return None
        row = _expression_instruction(
            program, ins, parts[ins.positions], keywords.get(ins.positions, ())
        )
        if row is None:
            return None
        if (
            isinstance(root, ast.Return)
            and index == end
            and ins.opname == "RETURN_VALUE"
        ):
            # The return consumes the sole source result. CPython 3.14 reports
            # its frame-exit stack effect as zero.
            if depth != 1:
                return None
            depth = 0
        else:
            depth += dis.stack_effect(ins.opcode, ins.arg)
        if depth < 0:
            return None
        rows.append(row)
    if isinstance(root, ast.Constant) and [row["opcode"] for row in rows] != ["NOP"]:
        return None
    if isinstance(root, ast.Delete) and [row["position"] for row in rows] != [
        list(tuple_position(target)) for target in root.targets
    ]:
        return None
    if depth != int(isinstance(root, ast.For)) or (
        discarded_call
        and (
            len(rows) < 2
            or rows[-2]["opcode"] not in {"CALL", "CALL_KW"}
            or rows[-2]["position"] != list(span)
        )
    ):
        return None
    proof = dict(
        semantics_version=1, position=list(tuple_position(root)), instructions=rows
    )
    if rows[-1]["opcode"] in {"STORE_SUBSCR", "STORE_SLICE"}:
        if end + 1 >= len(instructions):
            return None
        following = instructions[end + 1]
        caches = list(range(rows[-1]["offset"] + 2, following.offset, 2))
        if any(
            program.co_code[i : i + 2] != bytes((dis.opmap["CACHE"], 0)) for i in caches
        ):
            return None
        proof["normal_successor"] = dict(
            offset=following.offset, opcode=following.opname, cache_offsets=caches
        )
    return proof


def _statement_expression_sites(
    program,
    instructions,
    expressions,
    handlers,
    stores,
    following,
    *,
    function_frame=False,
    inherited=None,
):
    # Multiple compiled copies of a finally suite have the same source span.
    ends = defaultdict(list)
    offsets, jumps = {}, defaultdict(list)
    for index, ins in enumerate(instructions):
        offsets[ins.offset] = index
        if ins.opcode in _JUMP_OPCODES:
            jumps[ins.argval].append(ins.offset)

        root = expressions.get(ins.positions)
        if root is not None and ins.opname in _STATEMENT_END[type(root)]:
            ends[root].append(index)
    # Inner headers and backedges can be checked before their parent. The
    # source successor after exhaustion is then bound in forward order.
    loops, nested_transfers = {}, {}
    for root, matches in reversed(ends.items()):
        if len(matches) != 1 or not isinstance(root, ast.For):
            continue
        region = _iteration_region(
            root,
            instructions,
            matches[0],
            offsets,
            jumps,
            handlers,
            nested_transfers,
            function_frame=function_frame,
        )
        if region:
            ports, stop = loops[root] = region
            interval = (instructions[matches[0]].offset, ports["cleanup"][-1]["offset"])
            for row in [
                ports["next"],
                *(ports.get("backedges") or [ports["backedge"]]),
            ]:
                nested_transfers[row["offset"]] = interval
    sites = {}
    for root, matches in ends.items():
        span = tuple_position(root)
        if len(matches) == 1 and span not in stores:
            receipt = _statement_expression(
                program,
                instructions,
                root,
                matches[0],
                handlers,
                function_frame=function_frame,
            )
            if receipt is not None:
                after = following.get(root)
                loop = None
                if inherited and root in inherited:
                    kind, owner = inherited[root]
                    if kind == "continuation":
                        after = owner
                    else:
                        iteration = sites.get(tuple_position(owner), {}).get(
                            "iteration"
                        )
                        if iteration:
                            loop = (owner, iteration)
                if isinstance(root, ast.If):
                    edges = _branch_receipt(
                        root,
                        after,
                        instructions,
                        matches[0],
                        offsets,
                        receipt["instructions"],
                        function_frame=function_frame,
                        loop=loop,
                        handlers=handlers,
                    )
                    if edges is None:
                        continue
                    receipt["branch"] = edges
                if isinstance(root, ast.For):
                    if root not in loops:
                        continue
                    iteration, stop = loops[root]
                    if root.orelse:
                        exit_ = _suite_entry(
                            instructions[stop], root.orelse[0], "suite"
                        )
                    elif after is not None:
                        exit_ = _suite_entry(instructions[stop], after, "continuation")
                    elif loop:
                        exit_ = _loop_backedge(
                            instructions, stop, loop, instructions[matches[0]].positions
                        )
                    else:
                        exit_ = _implicit_return(
                            instructions, stop, instructions[matches[0]].positions
                        )
                    if not exit_:
                        continue
                    receipt["iteration"] = dict(iteration, exhausted=exit_)
                sites[span] = receipt
    return sites


def _positioned_instruction(ins):
    op = ins.opname
    row = dict(
        opcode=op,
        offset=ins.offset,
        position=list(ins.positions) if ins.positions.lineno is not None else None,
    )
    if ins.arg is not None:
        row["oparg"] = ins.arg
    if op in {
        "LOAD_NAME",
        "LOAD_GLOBAL",
        "STORE_NAME",
        "STORE_GLOBAL",
        "STORE_FAST",
        "DELETE_NAME",
        "DELETE_GLOBAL",
    }:
        row.update(
            name=ins.argval, index=ins.arg >> 1 if op == "LOAD_GLOBAL" else ins.arg
        )
    elif op == "LOAD_CONST":
        row.update(index=ins.arg, value=ins.argval)
    elif op == "POP_JUMP_IF_FALSE":
        row["target"] = ins.argval
    return row


def _exception_handler_sites(instructions, entries, tries, loads):
    """Join a module try's source clauses to its actual match, cleanup and unwind edges."""
    by_offset = {ins.offset: i for i, ins in enumerate(instructions)}
    starts = [e.start for e in entries]

    def covering(offset):
        i = bisect_right(starts, offset) - 1
        return entries[i] if i >= 0 and offset < entries[i].end else None

    def within(position, node):
        return (
            None not in position
            and (node.lineno, node.col_offset) <= (position.lineno, position.col_offset)
            and (position.end_lineno, position.end_col_offset)
            <= (node.end_lineno, node.end_col_offset)
        )

    def alias_cleanup(sequence, binding):
        return (
            len(sequence) == 3
            and sequence[0].opname == "LOAD_CONST"
            and sequence[0].argval is None
            and sequence[1].opname == binding.opname
            and sequence[1].arg == binding.arg
            and sequence[2].opname == binding.opname.replace("STORE_", "DELETE_")
            and sequence[2].arg == binding.arg
        )

    def arm(handler, index, first, mismatch, entry_op, primary, unwind):
        # A break or continue can restore the handled exception before its actual
        # jump. Such an exit is not an ordinary fallthrough, even if the clearing
        # sequence is byte-for-byte identical.
        if not all(
            isinstance(
                stmt,
                (
                    ast.Assign,
                    ast.AnnAssign,
                    ast.AugAssign,
                    ast.Expr,
                    ast.Delete,
                    ast.Raise,
                    ast.Pass,
                    ast.Import,
                    ast.ImportFrom,
                ),
            )
            for stmt in handler.body
        ):
            return None
        span = tuple_position(handler)
        target = unwind[0].offset
        load, check, branch = instructions[first : first + 3]
        header = instructions[first + 3 : mismatch]
        if (
            not header
            or check.opname != "CHECK_EXC_MATCH"
            or branch.opname != "POP_JUMP_IF_FALSE"
            or branch.argval != instructions[mismatch].offset
            or check.positions != span
            or branch.positions != span
        ):
            return None
        count = 2 if header[0].opname == "NOT_TAKEN" else 1
        matched = header[:count]
        binding = matched[-1]
        if (
            len(matched) != count
            or any(row.positions != span for row in matched)
            or (
                handler.name
                and (
                    binding.opname not in {"STORE_NAME", "STORE_GLOBAL"}
                    or binding.argval != handler.name
                )
            )
            or (not handler.name and binding.opname != "POP_TOP")
        ):
            return None
        table = set(primary)

        def targets(ins, dest):
            entry = covering(ins.offset)
            if entry and entry.target == dest and entry.depth == 1 and entry.lasti:
                table.add(entry)
                return True
            return False

        if not all(targets(i, target) for i in (load, check, branch, binding)):
            return None
        remaining = header[count:]
        end = next(
            (
                i
                for i, row in enumerate(remaining)
                if row.opname in {"POP_EXCEPT", "RAISE_VARARGS"}
            ),
            None,
        )
        if end is None:
            return None
        finish = remaining[end]
        raises = finish.opname == "RAISE_VARARGS"
        body = remaining[: end + int(raises)]
        # Several compiled control paths can share the same source span. Only
        # follow a straight-line suite with an unambiguous exit and cleanup.
        if any(
            row.opcode in _JUMP_OPCODES
            or row.is_jump_target
            or row.opname in {"RETURN_VALUE", "RETURN_CONST", "RERAISE", "EXTENDED_ARG"}
            or not any(within(row.positions, stmt) for stmt in handler.body)
            for row in body
        ):
            return None
        normal = [] if raises else remaining[end : end + (4 if handler.name else 1)]
        if handler.name and normal and not alias_cleanup(normal[1:], binding):
            return None
        exceptional = []
        if handler.name:
            exceptional = header[-4:]
            if (
                not alias_cleanup(exceptional[:-1], binding)
                or exceptional[-1].opname != "RERAISE"
                or exceptional[-1].arg != 1
            ):
                return None
        body_target = exceptional[0].offset if exceptional else target
        if not all(targets(row, body_target) for row in body):
            return None
        if not all(targets(row, target) for row in exceptional):
            return None
        for row in normal:
            entry = covering(row.offset)
            if entry:
                table.add(entry)
        return dict(
            semantics_version=1,
            position=list(span),
            arm_index=index,
            entry=_positioned_instruction(entry_op),
            load=loads[tuple_position(handler.type)],
            check=_positioned_instruction(check),
            branch=_positioned_instruction(branch),
            matched=list(map(_positioned_instruction, matched)),
            body_offset=remaining[0].offset,
            body_raise=_positioned_instruction(finish) if raises else None,
            mismatch=_positioned_instruction(instructions[mismatch]),
            normal_exit=list(map(_positioned_instruction, normal)),
            exceptional_cleanup=list(map(_positioned_instruction, exceptional)),
            unwind=list(map(_positioned_instruction, unwind)),
            exception_table=[dict(e._asdict()) for e in sorted(table)],
        )

    sites = {}
    for node in tries:
        lookups = [
            loads.get(tuple_position(h.type)) if isinstance(h.type, ast.Name) else None
            for h in node.handlers
        ]
        if not lookups or not all(lookups):
            continue
        indexes = [by_offset[row["offset"]] for row in lookups]
        first = indexes[0]
        if first == 0:
            continue
        entry_op = instructions[first - 1]
        primary = [e for e in entries if e.target == entry_op.offset]
        if (
            entry_op.opname != "PUSH_EXC_INFO"
            or not primary
            or any(e.depth != 0 or e.lasti for e in primary)
            or any(
                not any(
                    within(instructions[by_offset[e.start]].positions, stmt)
                    for stmt in node.body
                )
                for e in primary
            )
        ):
            continue
        header_entry = covering(entry_op.offset)
        if not header_entry or not header_entry.lasti or header_entry.depth != 1:
            continue
        start = by_offset[header_entry.target]
        unwind = instructions[start : start + 3]
        if [row.opname for row in unwind] != ["COPY", "POP_EXCEPT", "RERAISE"] or (
            unwind[0].arg,
            unwind[2].arg,
        ) != (3, 1):
            continue
        last = start - 1
        if instructions[last].opname != "RERAISE" or instructions[last].arg != 0:
            continue
        mismatch_entry = covering(instructions[last].offset)
        if (
            not mismatch_entry
            or mismatch_entry.target != header_entry.target
            or mismatch_entry.depth != 1
            or not mismatch_entry.lasti
        ):
            continue
        table = [*primary, header_entry, mismatch_entry]
        arms = [
            arm(h, i, a, b, entry_op, table, unwind) if a + 3 < b else None
            for i, (h, a, b) in enumerate(
                zip(node.handlers, indexes, [*indexes[1:], last], strict=True)
            )
        ]
        if all(arms):
            for h, receipt in zip(node.handlers, arms, strict=True):
                sites[tuple_position(h)] = dict(receipt, arm_count=len(arms))
    return sites


def _code_parameters(code):
    names = code.co_varnames
    positional, keyword = code.co_argcount, code.co_argcount + code.co_kwonlyargcount
    params = [
        ["posonly" if i < code.co_posonlyargcount else "positional", name]
        for i, name in enumerate(names[:positional])
    ]
    variadic = bool(code.co_flags & inspect.CO_VARARGS)
    if variadic:
        params.append(["varargs", names[keyword]])
    params.extend(["keywordonly", name] for name in names[positional:keyword])
    if code.co_flags & inspect.CO_VARKEYWORDS:
        params.append(["kwargs", names[keyword + variadic]])
    return params


def _function_expression_sites(program, node, following, inherited):
    """Collect located expression, assignment and control regions in one child frame."""
    expressions = {}
    for stmt in scope_nodes(node.body):
        if isinstance(stmt, ast.Expr) and isinstance(
            stmt.value, (ast.Call, ast.Constant)
        ):
            expressions[tuple_position(stmt.value)] = stmt.value
        elif isinstance(stmt, ast.Return):
            expressions[tuple_position(stmt)] = stmt
            if stmt.value is not None:
                expressions[tuple_position(stmt.value)] = stmt
        elif (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], (ast.Name, ast.Subscript))
        ):
            expressions[tuple_position(stmt.targets[0])] = stmt
        elif isinstance(stmt, ast.If) and (
            operand := _condition_operand(stmt.test, function_frame=True)
        ):
            expressions[tuple_position(operand)] = stmt
        elif isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
            expressions[tuple_position(stmt)] = stmt
            expressions[tuple_position(stmt.iter)] = stmt
    if not expressions:
        return {}, {}
    instructions = list(dis.get_instructions(program))
    handlers = {entry.target for entry in dis.Bytecode(program).exception_entries}
    regions = _statement_expression_sites(
        program,
        instructions,
        expressions,
        handlers,
        {},
        following,
        function_frame=True,
        inherited=inherited,
    )
    regions = {
        span: proof
        for span, proof in regions.items()
        if proof["instructions"][-1]["opcode"]
        in {
            "NOP",
            "STORE_FAST",
            "STORE_SUBSCR",
            "POP_TOP",
            "RETURN_VALUE",
            "RETURN_CONST",
            "POP_JUMP_IF_TRUE",
            "POP_JUMP_IF_FALSE",
            "GET_ITER",
        }
    }
    names = {
        tuple(row["position"]): row["name"]
        for proof in regions.values()
        for row in proof["instructions"]
        if row["opcode"] in _FAST_READS or row["opcode"] == "LOAD_GLOBAL"
    }
    loads = _name_load_sites(program, instructions, names)
    verified = {}
    for span, proof in regions.items():
        if not all(
            loads.get(tuple(row["position"]), {}).get("offset") == row["offset"]
            for row in proof["instructions"]
            if row["opcode"] in _FAST_READS or row["opcode"] == "LOAD_GLOBAL"
        ):
            continue
        branch = proof.get("branch", {})
        successors = (
            branch.get("join"),
            branch.get("else"),
            proof.get("iteration", {}).get("exhausted"),
        )
        if any(
            edge
            and edge["kind"] in {"loop_header", "loop_backedge"}
            and tuple(edge["statement_position"]) not in verified
            for edge in successors
        ):
            continue
        verified[span] = proof
    return verified, loads


def _default_tuple_items(value, defaults):
    if flat := _pool_literal_items(value, defaults):
        return flat

    def token(actual, node):
        if type(actual) is tuple and isinstance(node, ast.Tuple):
            if len(actual) == len(node.elts):
                items = [token(v, n) for v, n in zip(actual, node.elts)]
                if all(item is not None for item in items):
                    return ["tuple", items]
            return None
        if isinstance(node, ast.Constant):
            original = node.value
        elif (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.UAdd, ast.USub))
            and isinstance(node.operand, ast.Constant)
            and type(node.operand.value) in (int, float)
            and not isinstance(numeric_wire_value(node.operand.value), dict)
        ):
            original = ast.literal_eval(node)
        else:
            return None
        if type(actual) in (str, int, float, bool, type(None)) and type(
            original
        ) is type(actual):
            value = numeric_wire_value(actual)
            if json_bytes(numeric_wire_value(original)) == json_bytes(value):
                return ["literal", type(actual).__name__, value]
        return None

    if type(value) is tuple and len(value) == len(defaults):
        items = [token(v, n) for v, n in zip(value, defaults)]
        if all(item is not None for item in items):
            return dict(
                type="tuple",
                item_count=len(items),
                items_sha256=digest(json_bytes(items)),
            )
    return None


def _definition_instruction(ins):
    row = _positioned_instruction(ins)
    row.pop("value", None)
    return row


def _default_operand(program, instructions, default, stop, loads):
    """Match one original expression producing a single defining-frame stack value."""
    start = stop
    span = tuple_position(default)
    while start and _within_position(instructions[start - 1].positions, span):
        start -= 1
    parts = _expression_parts(default, allow_indexed=True)
    if start == stop or parts is None:
        return None
    operands, keywords, _ = parts
    rows, depth = [], 0
    for ins in instructions[start:stop]:
        if ins.positions not in operands:
            return None
        row = _expression_instruction(
            program, ins, operands[ins.positions], keywords.get(ins.positions, ())
        )
        if row is None or (
            row["opcode"] in {"LOAD_NAME", "LOAD_GLOBAL"}
            and loads.get(ins.positions, {}).get("offset") != ins.offset
        ):
            return None
        depth += dis.stack_effect(ins.opcode, ins.arg)
        if depth < 0:
            return None
        rows.append(row)
    return (start, rows) if depth == 1 else None


def _definition_default_window(program, instructions, node, stop, loads):
    """Decode the positional tuple and keyword map in actual stack order."""
    keywords = [
        (a.arg, v)
        for a, v in zip(node.args.kwonlyargs, node.args.kw_defaults)
        if v is not None
    ]
    groups, ports = [], {}
    position = tuple_position(node)
    if keywords:
        stop -= 1
        if stop < 0:
            return None
        build = instructions[stop]
        names = [name for name, _ in keywords]
        if (
            build.opname not in {"BUILD_CONST_KEY_MAP", "BUILD_MAP"}
            or build.arg != len(names)
            or build.positions != position
        ):
            return None
        ports["keyword_defaults"] = dict(names=names, build_offset=build.offset)
        groups.append([_definition_instruction(build)])
        packed = build.opname == "BUILD_CONST_KEY_MAP"
        if packed:
            stop -= 1
            if stop < 0:
                return None
            key = instructions[stop]
            if (
                key.opname != "LOAD_CONST"
                or key.positions != position
                or program.co_consts[key.arg] != tuple(names)
            ):
                return None
            groups.append(
                [
                    dict(
                        _definition_instruction(key),
                        index=key.arg,
                        type="tuple",
                        names=names,
                    )
                ]
            )
        for name, expression in reversed(keywords):
            part = _default_operand(program, instructions, expression, stop, loads)
            if part is None:
                return None
            stop, rows = part
            groups.append(rows)
            if not packed:
                stop -= 1
                if stop < 0:
                    return None
                key = instructions[stop]
                if (
                    key.opname != "LOAD_CONST"
                    or key.positions != position
                    or program.co_consts[key.arg] != name
                ):
                    return None
                groups.append(
                    [
                        dict(
                            _definition_instruction(key),
                            index=key.arg,
                            type="str",
                            value_sha256=digest(json_bytes(name)),
                        )
                    ]
                )
    defaults = node.args.defaults
    if defaults:
        stop -= 1
        if stop < 0:
            return None
        tail = instructions[stop]
        if tail.positions != position:
            return None
        if tail.opname == "LOAD_CONST" and (
            items := _default_tuple_items(program.co_consts[tail.arg], defaults)
        ):
            ports["positional_defaults"] = dict(index=tail.arg, count=len(defaults))
            groups.append(
                [dict(_definition_instruction(tail), index=tail.arg, **items)]
            )
        elif tail.opname == "BUILD_TUPLE" and tail.arg == len(defaults):
            ports["positional_defaults"] = dict(
                count=len(defaults), build_offset=tail.offset
            )
            groups.append([_definition_instruction(tail)])
            for expression in reversed(defaults):
                part = _default_operand(program, instructions, expression, stop, loads)
                if part is None:
                    return None
                stop, rows = part
                groups.append(rows)
        else:
            return None
    return stop, [r for group in reversed(groups) for r in group], ports


def _definition_sites(program, instructions, definitions, handlers, loads):
    """Plain module definitions whose code object is loaded, constructed and stored."""
    found, seen, repeated = {}, set(), set()
    for i, make in enumerate(instructions):
        node = definitions.get(make.positions)
        if (
            not node
            or make.opname != "MAKE_FUNCTION"
            or not 0 < i < len(instructions) - 1
        ):
            continue
        load = instructions[i - 1]
        if not (
            load.opname == "LOAD_CONST"
            and type(load.argval) is CodeType
            and load.argval is program.co_consts[load.arg]
            and load.argval.co_name == node.name
        ):
            continue
        if make.positions in seen:
            repeated.add(make.positions)
        seen.add(make.positions)
        matched = _definition_default_window(program, instructions, node, i - 1, loads)
        if matched is None:
            continue
        start, value_rows, ports = matched
        attributes = [
            flag
            for field, flag in (("keyword_defaults", 2), ("positional_defaults", 1))
            if field in ports
        ]
        end = i + 1
        if sys.version_info[:2] == (3, 12):
            if make.arg != sum(attributes):
                continue
        elif sys.version_info[:2] in ((3, 13), (3, 14)) and make.arg is None:
            if [
                (r.opname, r.arg) for r in instructions[end : end + len(attributes)]
            ] != [("SET_FUNCTION_ATTRIBUTE", flag) for flag in attributes]:
                continue
            end += len(attributes)
        else:
            continue
        if end >= len(instructions):
            continue
        window = instructions[i - 1 : end + 1]
        store = window[-1]
        if not (
            all(op.positions == make.positions for op in window)
            and not (start and instructions[start - 1].opname == "EXTENDED_ARG")
            and store.opname in {"STORE_NAME", "STORE_GLOBAL"}
            and program.co_names[store.arg] == store.argval == node.name
            and not any(
                op.is_jump_target or op.offset in handlers
                for op in instructions[start + 1 : end + 1]
            )
        ):
            continue
        child = load.argval
        if (
            child.co_firstlineno != node.lineno
            or child.co_qualname != node.name
            or child.co_name != node.name
            or child.co_freevars
            or child.co_flags
            & (inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR)
        ):
            continue
        rows = [_definition_instruction(op) for op in window]
        rows[0]["index"] = load.arg
        rows[-1].update(index=store.arg, name=node.name)
        proof = dict(
            semantics_version=1,
            position=list(make.positions),
            instructions=value_rows + rows,
            **ports,
            child=dict(
                name=child.co_name,
                qualname=child.co_qualname,
                first_line=child.co_firstlineno,
                code_sha256=code_digest(child),
                parameters=_code_parameters(child),
                varnames=list(child.co_varnames),
            ),
        )
        found[make.positions] = proof, child
    return {k: v for k, v in found.items() if k not in repeated}


def _module_frame_exit(instructions, code_sha256):
    """Locate the unique normal return, even when exception handlers follow it."""
    returns = [
        i
        for i, row in enumerate(instructions)
        if row.opname in {"RETURN_CONST", "RETURN_VALUE"}
    ]
    if len(returns) != 1:
        return {}
    index = returns[0]
    last = instructions[index]
    if last.opname == "RETURN_CONST":
        tail = [last]
    elif last.opname == "RETURN_VALUE" and index > 0 and not last.is_jump_target:
        tail = instructions[index - 1 : index + 1]
        if tail[0].opname != "LOAD_CONST" or tail[0].offset + 2 != last.offset:
            return {}
    else:
        return {}
    constant = tail[0]
    if constant.argval is not None or constant.arg > 255:
        return {}
    rows = [_positioned_instruction(i) for i in tail]
    rows[0].pop("value", None)
    rows[0].update(
        index=constant.arg, type="NoneType", value_sha256=digest(json_bytes(None))
    )
    return dict(code_sha256=code_sha256, instructions=rows)


def _class_body_site(code, node, code_sha256):
    """Witness every opcode of a plain class suite containing a docstring or passes."""
    doc = ast.get_docstring(node, clean=False)
    passes = node.body[int(doc is not None) :]
    if any(not isinstance(stmt, ast.Pass) for stmt in passes) or code.co_exceptiontable:
        return None
    ops = list(dis.get_instructions(code))
    if not ops or ops[0].positions.lineno != code.co_firstlineno:
        return None
    head = ops[0].positions
    expected = [
        ("RESUME", 0, head),
        ("LOAD_NAME", "__name__", head),
        ("STORE_NAME", "__module__", head),
        ("LOAD_CONST", code.co_qualname, head),
        ("STORE_NAME", "__qualname__", head),
    ]
    newer = sys.version_info[:2] >= (3, 13)
    if newer:
        expected.extend(
            [
                ({"LOAD_CONST", "LOAD_SMALL_INT"}, code.co_firstlineno, head),
                ("STORE_NAME", "__firstlineno__", head),
            ]
        )
    statements = []
    if doc is not None:
        statement = node.body[0]
        pos = tuple_position(statement.value)
        statements.append(
            dict(
                kind="docstring",
                position=list(tuple_position(statement)),
                value_position=list(pos),
                source_value_sha256=digest(json_bytes(doc)),
            )
        )
        # CPython 3.13+ can clean indentation. Keep the actual pool string
        # and pair its span with the original literal without reformatting it.
        expected.extend(
            [("LOAD_CONST", str if newer else doc, pos), ("STORE_NAME", "__doc__", pos)]
        )
    statements.extend(
        dict(kind="pass", position=list(tuple_position(stmt))) for stmt in passes
    )
    expected.extend(("NOP", None, tuple_position(stmt)) for stmt in passes[:-1])
    end = tuple_position(passes[-1] if passes else node.body[0].value)
    # On a single physical line a final pass can have no instruction of its
    # own. In 3.12 an otherwise empty suite can return at its prologue span.
    if (
        doc is None
        and len(passes) == 1
        and end[0] == head.lineno
        and ops[-1].positions == head
    ):
        end = head
    if newer:
        expected.extend(
            [
                ("LOAD_CONST", (), end),
                ("STORE_NAME", "__static_attributes__", end),
                ("LOAD_CONST", None, end),
                ("RETURN_VALUE", None, end),
            ]
        )
    else:
        expected.append(("RETURN_CONST", None, end))
    if len(ops) != len(expected) or len(code.co_code) != 2 * len(ops):
        return None
    rows = []
    for index, (ins, (opcode, value, position)) in enumerate(zip(ops, expected)):
        if (
            ins.offset != 2 * index
            or ins.is_jump_target
            or None in ins.positions
            or ins.positions != position
            or (
                ins.opname not in opcode
                if isinstance(opcode, set)
                else ins.opname != opcode
            )
            or (
                type(ins.argval) is not str
                if value is str
                else type(ins.argval) is not type(value) or ins.argval != value
            )
        ):
            return None
        row = _positioned_instruction(ins)
        if ins.opname in {"LOAD_CONST", "RETURN_CONST", "LOAD_SMALL_INT"}:
            if ins.opname != "LOAD_SMALL_INT":
                if code.co_consts[ins.arg] is not ins.argval:
                    return None
                row["index"] = ins.arg
            row.update(
                type=type(ins.argval).__name__,
                value=list(ins.argval) if type(ins.argval) is tuple else ins.argval,
            )
        rows.append(row)
    return dict(
        semantics_version=1,
        code_sha256=code_sha256,
        position=list(tuple_position(node)),
        instructions=rows,
        statements=statements,
    )


def _class_sites(program, instructions, classes, handlers, loads):
    """Find the original builder, class code and publication for direct class sites."""
    sites, bodies, seen, repeated = {}, {}, set(), set()
    before_make = (
        ("PUSH_NULL", "LOAD_BUILD_CLASS", "LOAD_CONST")
        if sys.version_info[:2] == (3, 12)
        else ("LOAD_BUILD_CLASS", "PUSH_NULL", "LOAD_CONST")
    )
    if sys.version_info[:2] not in ((3, 12), (3, 13), (3, 14)):
        return {}, {}
    for index, make in enumerate(instructions):
        node = classes.get(make.positions)
        if node is None or make.opname != "MAKE_FUNCTION" or index < 3:
            continue
        if make.positions in seen:
            repeated.add(make.positions)
        seen.add(make.positions)
        end = index + 3 + len(node.bases)
        if end + 1 >= len(instructions):
            continue
        start = index - 3
        window = instructions[start : end + 1]
        if (
            tuple(ins.opname for ins in window[:3]) != before_make
            or make.arg != (0 if sys.version_info[:2] == (3, 12) else None)
            or start
            and instructions[start - 1].opname == "EXTENDED_ARG"
            or any(ins.is_jump_target or ins.offset in handlers for ins in window[1:])
        ):
            continue
        null, builder, code, _, name = window[:5]
        call, store = window[-2:]
        tail = instructions[end + 1]
        child = code.argval
        if not (
            type(child) is CodeType
            and child is program.co_consts[code.arg]
            and child.co_name == child.co_qualname == node.name
            and child.co_firstlineno == node.lineno
            and not child.co_freevars
            and not child.co_cellvars
            and name.opname == "LOAD_CONST"
            and name.argval == node.name
            and call.opname == "CALL"
            and call.arg == 2 + len(node.bases)
            and store.opname in {"STORE_NAME", "STORE_GLOBAL"}
            and store.argval == node.name
            and all(
                ins.positions == make.positions
                for ins in (null, builder, code, name, call, store)
            )
        ):
            continue
        if any(
            operand.opname not in {"LOAD_NAME", "LOAD_GLOBAL"}
            or operand.argval != base.id
            or operand.opname == "LOAD_GLOBAL"
            and operand.arg & 1
            or loads.get(tuple_position(base), {}).get("offset") != operand.offset
            for base, operand in zip(node.bases, window[5:-2])
        ):
            continue
        edges = [
            dict(
                offset=b.offset,
                opcode=b.opname,
                cache_offsets=list(range(a.offset + 2, b.offset, 2)),
            )
            for a, b in pairwise([*window, tail])
        ]
        if any(
            program.co_code[pc : pc + 2] != bytes((dis.opmap["CACHE"], 0))
            for edge in edges
            for pc in edge["cache_offsets"]
        ):
            continue
        rows = []
        for ins in window:
            if ins is code:
                rows.append(
                    dict(
                        opcode=ins.opname,
                        offset=ins.offset,
                        position=list(ins.positions),
                        oparg=ins.arg,
                        index=ins.arg,
                    )
                )
            else:
                rows.append(_positioned_instruction(ins))
        sites[make.positions] = dict(
            semantics_version=1,
            position=list(make.positions),
            instructions=rows,
            successors=edges,
            child=dict(
                name=child.co_name,
                qualname=child.co_qualname,
                first_line=child.co_firstlineno,
                code_sha256=code_digest(child),
            ),
        )
        if body := _class_body_site(
            child, node, sites[make.positions]["child"]["code_sha256"]
        ):
            bodies[make.positions] = body
    for span in repeated:
        sites.pop(span, None)
        bodies.pop(span, None)
    return sites, bodies


def _compiler_sites(tree, path, relative, source_hash):
    """Capture positioned construction and binding facts from one compilation."""
    if sys.implementation.name != "cpython" or not hasattr(CodeType, "co_positions"):
        return {}
    nodes, mappings, following, syntax = {}, {}, {}, deque([tree])
    inherited = {}
    while syntax:
        node = syntax.popleft()
        syntax.extend(ast.iter_child_nodes(node))
        for field in ("body", "orelse", "finalbody"):
            suite = getattr(node, field, None)
            if isinstance(suite, list):
                following.update(
                    (a, b)
                    for a, b in pairwise(suite)
                    if isinstance(a, (ast.If, ast.For))
                )
                if (
                    suite
                    and isinstance(suite[-1], (ast.If, ast.For))
                    and isinstance(node, (ast.If, ast.For))
                ):
                    tail = suite[-1]
                    if isinstance(node, ast.For) and field == "body":
                        inherited[tail] = ("loop_backedge", node)
                    elif node in following:
                        inherited[tail] = ("continuation", following[node])
                    elif node in inherited:
                        inherited[tail] = inherited[node]
        if isinstance(node, ast.Dict):
            mappings[tuple_position(node)] = []
        elif (
            isinstance(node, ast.Tuple)
            and isinstance(node.ctx, ast.Load)
            and not any(isinstance(e, ast.Starred) for e in node.elts)
        ):
            nodes[tuple_position(node)] = node
    assignments, names, imports, expressions, tries = [], {}, {}, {}, []
    definitions, classes = {}, {}
    for node in scope_nodes(tree.body):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], (ast.Name, ast.Subscript))
        ):
            expressions[tuple_position(node.targets[0])] = node
            if isinstance(node.targets[0], ast.Name) and _constant_store_rhs(
                node.value
            ):
                assignments.append(node)
        elif (
            isinstance(node, ast.FunctionDef)
            and not node.decorator_list
            and node.returns is None
            and not getattr(node, "type_params", ())
            and not any(
                x.annotation is not None
                for x in ast.walk(node.args)
                if isinstance(x, ast.arg)
            )
        ):
            definitions[tuple_position(node)] = node
        elif (
            isinstance(node, ast.ClassDef)
            and not node.decorator_list
            and not node.keywords
            and not getattr(node, "type_params", ())
            and all(isinstance(base, ast.Name) for base in node.bases)
        ):
            classes[tuple_position(node)] = node
        elif isinstance(node, ast.Delete) and all(
            isinstance(t, ast.Name) for t in node.targets
        ):
            expressions[tuple_position(node.targets[-1])] = node
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names[tuple_position(node)] = node.id
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            names[tuple_position(node.target)] = node.target.id
        elif (
            isinstance(node, ast.Import)
            and len(node.names) == 1
            or isinstance(node, ast.ImportFrom)
            and all(alias.name != "*" for alias in node.names)
        ):
            imports[tuple_position(node)] = node
        elif isinstance(node, ast.Try) and node.handlers:
            tries.append(node)
        elif isinstance(node, ast.Raise):
            expressions[tuple_position(node)] = node
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            expressions[tuple_position(node)] = node
            expressions[tuple_position(node.iter)] = node
        elif isinstance(node, ast.If) and (operand := _condition_operand(node.test)):
            expressions[tuple_position(operand)] = node
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            expressions[tuple_position(node.value)] = node.value
    docstring = (
        tree.body[0] if ast.get_docstring(tree, clean=False) is not None else None
    )
    optimize = sys.flags.optimize
    program = compile(tree, str(path), "exec", dont_inherit=True, optimize=optimize)
    unit = digest(json_bytes([relative, source_hash, ParsedUnit.compiler_id, optimize]))
    instructions = list(dis.get_instructions(program))
    entries = (
        dis.Bytecode(program).exception_entries
        if assignments
        or imports
        or expressions
        or tries
        or definitions
        or classes
        or docstring is not None
        else ()
    )
    handlers = {e.target for e in entries}
    loads = _name_load_sites(program, instructions, names) if names else {}
    definition_sites = (
        _definition_sites(program, instructions, definitions, handlers, loads)
        if definitions
        else {}
    )
    class_sites, class_body_sites = (
        _class_sites(program, instructions, classes, handlers, loads)
        if classes
        else ({}, {})
    )
    stores, document = (
        _constant_store_sites(program, instructions, assignments, handlers, docstring)
        if assignments or docstring is not None
        else ({}, {})
    )
    import_stores = (
        _import_store_sites(program, instructions, imports, handlers) if imports else {}
    )
    statement_expressions = (
        _statement_expression_sites(
            program, instructions, expressions, handlers, stores, following
        )
        if expressions
        else {}
    )
    exception_handlers = (
        _exception_handler_sites(instructions, entries, tries, loads)
        if tries and loads
        else {}
    )
    source_code = dict(
        kind="module",
        source_key=relative,
        source_sha256=source_hash,
        code_sha256=code_digest(program),
        compiler=ParsedUnit.compiler_id,
        optimize=optimize,
    )
    fresh, conflict = object(), object()
    origins = {}

    def observe(node, origin):
        if origins.setdefault(node, origin) is not origin:
            origins[node] = conflict

    pending = [program] if nodes or mappings else []
    while pending:
        code = pending.pop()
        pending.extend(c for c in reversed(code.co_consts) if isinstance(c, CodeType))
        for instruction in (
            instructions
            if code is program and instructions is not None
            else dis.get_instructions(code)
        ):
            opcode = getattr(instruction, "baseopname", instruction.opname)
            if instruction.positions in mappings and opcode in {
                "BUILD_MAP",
                "BUILD_CONST_KEY_MAP",
                "MAP_ADD",
                "DICT_UPDATE",
            }:
                mappings[instruction.positions].append((opcode, instruction.arg))
            node = nodes.get(instruction.positions)
            if node is None:
                continue
            size = len(node.elts)
            if opcode == "BUILD_TUPLE" and instruction.arg == size:
                observe(node, fresh if size else ())
            elif opcode in {"LOAD_CONST", "RETURN_CONST"}:
                value = code.co_consts[instruction.arg]
                if type(value) is tuple and len(value) == size:
                    observe(node, value)

    sites, slots = {}, {}
    # Parents precede children. A folded constant supplies its direct tuple
    # members, never intermediate operands of an expression folded into one member.
    for span, node in nodes.items():
        value = origins.get(node)
        if value is fresh:
            sites[span] = {"kind": "fresh"}
        elif type(value) is tuple:
            if not value:
                sites[span] = {
                    "kind": "constant_pool",
                    "pool": "cpython:empty_tuple",
                    "slot": 0,
                }
                continue
            slot = slots.setdefault(id(value), len(slots))
            sites[span] = {"kind": "constant_pool", "unit": unit, "slot": slot}
            for child, member in zip(node.elts, value):
                if (
                    isinstance(child, ast.Tuple)
                    and nodes.get(tuple_position(child)) is child
                    and type(member) is tuple
                    and len(member) == len(child.elts)
                ):
                    observe(child, member)
    return dict(
        tuple_sites=sites,
        mapping_builders=mappings,
        constant_stores=stores,
        module_docstring=document,
        source_code=source_code,
        frame_exit=_module_frame_exit(instructions, source_code["code_sha256"]),
        name_loads=loads,
        import_stores=import_stores,
        statement_expressions=statement_expressions,
        definition_sites={span: proof for span, (proof, _) in definition_sites.items()},
        class_sites=class_sites,
        class_body_sites=class_body_sites,
        function_expression_sites={
            span: _function_expression_sites(
                child, definitions[span], following, inherited
            )
            for span, (_, child) in definition_sites.items()
        },
        exception_handlers=exception_handlers,
    )


@cache
def _text_profiles():
    """Capture single-code-point printability for the selected interpreter."""
    intervals = []
    start = None
    for codepoint in range(0x110000):
        if chr(codepoint).isprintable():
            if start is not None:
                intervals.append([start, codepoint - 1])
                start = None
        elif start is None:
            start = codepoint
    if start is not None:
        intervals.append([start, 0x10FFFF])
    profile = dict(
        unicode_version=unicodedata.unidata_version,
        interpreter=ParsedUnit.compiler_id,
        nonprintable=intervals,
    )
    return {digest(json_bytes(profile)): profile}


def parse_unit(
    path,
    flags=(),
    *,
    root,
    abi=None,
    source_bytes=None,
    source_hashes=None,
    _imports=None,
    _source_key=None,
    _source_origin=None,
):
    """F1: read source once and capture both CPython syntax and lexical binding."""
    path, root = Path(path).resolve(), Path(root).resolve()
    imports = _imports or Imports(root, ParsedUnit.compiler_id)
    relative = _source_key or path.relative_to(root).as_posix()
    snapshot = imports.snapshot(path, source_bytes)
    source_hash = snapshot.sha256
    if _source_origin and _source_origin["sha256"] != source_hash:
        raise ValueError(f"Python provider and captured source disagree: {relative}")
    origins = (
        {relative: {**_source_origin, "size": len(snapshot.raw)}}
        if _source_origin
        else {}
    )
    if (
        source_hashes
        and relative in source_hashes
        and source_hashes[relative] != source_hash
    ):
        raise ValueError(f"Python source snapshot differs from input lock: {relative}")
    try:
        text, tree = snapshot.text, snapshot.tree
        scopes = LexicalScope.capture(symtable.symtable(text, str(path), "exec"))
        compiler_sites = _compiler_sites(tree, path, relative, source_hash)
    except (SyntaxError, UnicodeError, LookupError) as error:
        location = dict(
            file=relative,
            line=getattr(error, "lineno", None),
            column=getattr(error, "offset", None),
        )
        issue = dict(
            code="python.syntax",
            detail=getattr(error, "msg", str(error)),
            field="syntax",
            **location,
        )
        return ParsedUnit(
            path,
            root,
            abi or {},
            None,
            None,
            frozenset(),
            frozenset(),
            {},
            source_hash,
            (issue,),
            runtime_facts=imports.runtime,
            text_profiles=_text_profiles(),
            source_key=relative,
            source_origins=origins,
        )
    module = (abi or {}).get("module") or ".".join(
        path.relative_to(root).with_suffix("").parts
    )
    value_uses = set()
    providers = (
        imports.providers_for(tree, module, snapshot.package, value_uses=value_uses)
        if not origins
        else {}
    )
    return ParsedUnit(
        path,
        root,
        abi or {},
        tree,
        scopes,
        frozenset(vars(builtins)),
        frozenset(sys.stdlib_module_names),
        _catalog(tree, module),
        source_hash,
        providers=providers,
        runtime_facts=imports.runtime,
        source_text=text,
        text_profiles=_text_profiles(),
        source_key=relative,
        source_origins=origins,
        import_value_uses=frozenset(value_uses),
        **compiler_sites,
    )


def provider(name, libraries):
    base = name.split(".", 1)[0]
    return (
        "python_stdlib" if base in libraries or base == "builtins" else "python_package"
    )


def _function(module, qualified):
    return f"py:{module}:{qualified}"


def _literal_fact(value):
    if isinstance(numeric_wire_value(value), dict):
        return {"class": type(value).__name__}
    return {"class": type(value).__name__, "values": [value]}


def scope_nodes(statements):
    """Visit syntax evaluated in this lexical frame without entering child bodies."""
    pending = list(reversed(statements))
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            values = [*getattr(node, "decorator_list", ()), node.args]
        elif isinstance(node, ast.ClassDef):
            values = [*node.decorator_list, *node.bases, *node.keywords]
        elif isinstance(node, ast.comprehension):
            values = [
                node.iter,
                *node.ifs,
            ]  # Its iteration variable belongs to the comprehension.
        else:
            values = list(ast.iter_child_nodes(node))
        pending.extend(reversed(values))


def assigned_names(node):
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        yield node.id
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        yield node.name
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            yield alias.asname or alias.name.split(".")[0]
    elif (
        isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name
    ):
        yield node.name
    elif isinstance(node, ast.MatchMapping) and node.rest:
        yield node.rest


def _catalog(tree, module):
    """Index declarations and state slots; runtime bindings follow execution order."""
    exports = {}
    for node in scope_nodes(tree.body):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(
            node.value, ast.Constant
        ):
            v = node.value.value
            if (
                type(v) in {str, int, float, bool, type(None)}
                and not isinstance(numeric_wire_value(v), dict)
                and not (isinstance(v, str) and len(v) > 4096)
            ):
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                ):
                    if isinstance(target, ast.Name):
                        slot = exports.setdefault(
                            target.id, {"state": _function(module, target.id)}
                        )
                        sites = slot.setdefault("literal_sites", [])
                        if len(sites) < 16:
                            sites.append({"line": node.lineno, "value": v})
        for name in assigned_names(node):
            state = exports.setdefault(name, {"state": _function(module, name)})
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                state.setdefault("definitions", []).append(
                    {
                        "line": node.lineno,
                        "class": (
                            "type" if isinstance(node, ast.ClassDef) else "function"
                        ),
                        "decorated": bool(node.decorator_list),
                    }
                )
    for item in exports.values():
        definitions = item.get("definitions", ())
        for d in definitions:
            d["symbol"] = item["state"] + (
                f"@{d['line']}" if len(definitions) > 1 else ""
            )
    return exports


@dataclass(frozen=True)
class ParsedProject:
    profiles: tuple
    units: tuple[ParsedUnit, ...]
    catalog: dict
    source_digests: dict


class ProjectCapture:
    """Share file syntax within one build, keeping each script's import environment."""

    def __init__(self, root, *, buffers=None, source_hashes=None):
        self.root = Path(root).resolve()
        self.buffers = buffers or {}
        self.source_hashes = source_hashes
        self.parsed = {}
        self.resolved = {}
        self.finders = {}
        self.projects = {}
        self.snapshots = {}

    def _read(self, path, module, imports, provider):
        external = provider["status"] != "project"
        member = source_member(provider) if external else None
        if external and not member:
            raise ValueError(
                f"selected provider has no verified Python source: {module}"
            )
        relative = (
            f"@python/{provider['provider_id']}/{member}"
            if external
            else path.relative_to(self.root).as_posix()
        )
        key = (relative, module, imports)
        if key not in self.resolved:
            facts = self.parsed.get(relative)
            if facts is None:
                facts = parse_unit(
                    path,
                    root=self.root,
                    abi={"module": module},
                    source_bytes=self.buffers.get(relative),
                    source_hashes=self.source_hashes,
                    _imports=imports,
                    _source_key=relative,
                    _source_origin=(
                        dict(
                            module=module,
                            provider_id=provider["provider_id"],
                            origin=provider["origin"],
                            sha256=provider["sha256"],
                            member=member,
                            status=provider["status"],
                            loader_kind=provider.get("loader_kind", provider["loader"]),
                        )
                        if external
                        else None
                    ),
                )
                self.parsed[relative] = facts
            elif facts.path != path:
                raise ValueError(
                    f"two physical sources resolve to one Python unit: {relative}"
                )
            elif facts.tree is not None:
                # The same bytes can import a different helper in a different image.
                value_uses = set()
                facts = replace(
                    facts,
                    exports=(
                        facts.exports
                        if module == facts.abi["module"]
                        else _catalog(facts.tree, module)
                    ),
                    providers=(
                        {}
                        if external
                        else imports.providers_for(
                            facts.tree,
                            module,
                            path.name == "__init__.py",
                            value_uses=value_uses,
                        )
                    ),
                    import_value_uses=frozenset(value_uses),
                )
            self.resolved[key] = facts
        return self.resolved[key]

    def capture(self, entry, *, search_root=None) -> ParsedProject:
        entry = (self.root / entry).resolve()
        directory = Path(search_root).resolve() if search_root else entry.parent
        key = (directory, entry)
        if key not in self.projects:
            if directory not in self.finders:
                self.finders[directory] = Imports(
                    self.root,
                    ParsedUnit.compiler_id,
                    search_root=directory,
                    snapshots=self.snapshots,
                )
            self.projects[key] = _capture_project(
                entry, self.root, directory, self.finders[directory], self._read
            )
        return self.projects[key]


def _import_nodes(tree, requested):
    """Visit the initialization suite and code that a requested operation may invoke."""
    definitions = defaultdict(list)
    functions = (ast.FunctionDef, ast.AsyncFunctionDef)
    for node in scope_nodes(tree.body):
        if isinstance(node, (*functions, ast.ClassDef)):
            definitions[node.name].append(node)

    pending = [tree]
    visited = set()

    def callable_body(node):
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, functions) and member.name in {
                    "__init__",
                    "__new__",
                }:
                    pending.extend(member.body)
        else:
            pending.extend(node.body)

    for name, line in requested:
        for node in definitions.get(name, ()):
            if line is None or line == node.lineno:
                callable_body(node)
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        yield node
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            for definition in definitions.get(node.func.id, ()):
                callable_body(definition)
        if isinstance(node, (*functions, ast.Lambda)):
            # Definitions evaluate their headers. Calling the resulting function
            # is a separate event; a merely present nested import opens no file.
            pending.append(node.args)
            pending.extend(getattr(node, "decorator_list", ()))
            pending.extend(getattr(node, "type_params", ()))
            if annotation := getattr(node, "returns", None):
                pending.append(annotation)
        else:
            pending.extend(ast.iter_child_nodes(node))


def _capture_project(entry, root, directory, imports, read) -> ParsedProject:
    """Use one ordinary unit per local or requested verified Python source module."""

    def module_path(path):
        relative = path.relative_to(directory).with_suffix("")
        pieces = relative.parts[:-1] if relative.name == "__init__" else relative.parts
        return ".".join(pieces) or "__main__"

    entry_module = module_path(entry)
    catalog = {}
    modules = {}
    origins = {
        entry_module: dict(
            status="project",
            module=entry_module,
            origin=str(entry),
            file=entry.relative_to(root).as_posix(),
        )
    }
    requests = defaultdict(set)
    pending = deque([entry_module])
    scheduled = {entry_module}

    def enqueue(name, *, requester, directly_requested, body=None):
        record = imports.record(name)
        if record["status"] != "project" and (
            record.get("source_interface") or not source_member(record)
        ):
            return
        # A verified standard-library source can contribute its complete F2
        # unit on direct demand. Its own dependencies remain typed frontier
        # records unless independently requested by project/distribution code.
        if requester["status"] == "verified_stdlib" or (
            record["status"] == "verified_stdlib" and not directly_requested
        ):
            return
        changed = name not in origins
        origins.setdefault(name, record)
        if body is not None and body not in requests[name]:
            requests[name].add(body)
            changed = True
        if changed and name not in scheduled:
            pending.append(name)
            scheduled.add(name)

    while pending:
        module = pending.popleft()
        scheduled.remove(module)
        origin = origins[module]
        path = Path(origin["origin"])
        package = path.name == "__init__.py"
        facts = read(path, module, imports, origin)
        if facts.tree is not None and origin["status"] != "project":
            value_uses = set()
            facts = replace(
                facts,
                providers=imports.providers_for(
                    facts.tree,
                    module,
                    package,
                    nodes=_import_nodes(facts.tree, requests[module]),
                    value_uses=value_uses,
                ),
                import_value_uses=frozenset(value_uses),
            )
        modules[module] = facts
        catalog[module] = {
            "path": facts.source_key,
            "package": package,
            "exports": facts.exports,
            "source_provider": origin,
            "graph_backed": facts.tree is not None,
        }
        for name, record in facts.providers.items():
            enqueue(name, requester=origin, directly_requested=True)
            if record.get("source_interface") or not record.get("exports"):
                continue
            enqueue(
                name,
                requester=origin,
                directly_requested=True,
                body=("__getattr__", None),
            )
            for member, export in record["exports"].items():
                # Executing `from pkg import f` inspects the binding. The
                # implementation of f is demanded only when its value is used.
                if (name, member) not in facts.import_value_uses:
                    continue
                for candidate in export.get("candidates", ()):
                    if candidate.get("kind", "definition") == "definition":
                        enqueue(
                            candidate["module"],
                            requester=origin,
                            directly_requested=candidate["module"] in facts.providers,
                            body=(candidate["name"], candidate["line"]),
                        )

    for name, facts in modules.items():
        if origins[name]["status"] != "verified_stdlib":
            continue
        providers = dict(facts.providers)
        for target, record in facts.providers.items():
            if (
                target not in modules
                and not record.get("source_interface")
                and (source_member(record) or record["status"] == "project")
            ):
                providers[target] = dict(
                    record,
                    source_expansion=dict(
                        status="frontier", reason="stdlib_dependency", requested_by=name
                    ),
                )
        modules[name] = replace(facts, providers=providers)

    def visible_modules(name):
        # Catalog membership is image-wide. A translation profile needs only
        # names this source can resolve; repeating every export of every unit
        # in each profile makes serialization quadratic in the package graph.
        visible = {name, *modules[name].providers}
        for provider in modules[name].providers.values():
            for export in provider.get("exports", {}).values():
                visible.update(c["module"] for c in export.get("candidates", ()))
        return {
            module: catalog[module] for module in sorted(visible) if module in catalog
        }

    profiles = tuple(
        dict(
            path=spec["path"],
            language="python",
            abi=dict(
                kind="python_module",
                module=name,
                main=name == entry_module,
                project=visible_modules(name),
            ),
        )
        for name, spec in catalog.items()
    )
    return ParsedProject(
        profiles,
        tuple(replace(modules[p["abi"]["module"]], abi=p["abi"]) for p in profiles),
        catalog,
        {unit.source_key: unit.source_hash for unit in modules.values()},
    )


def capture_project(
    entry: Path, root: Path, *, buffers=None, source_hashes=None
) -> ParsedProject:
    capture = ProjectCapture(root, buffers=buffers, source_hashes=source_hashes)
    return capture.capture(entry, search_root=root)


def project_units(entry: Path, root: Path) -> list[dict]:
    return list(capture_project(entry, root).profiles)

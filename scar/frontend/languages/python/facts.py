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
from scar.source_contracts import python_binary_operator
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
    name_loads: dict = field(default_factory=dict)
    import_stores: dict = field(default_factory=dict)
    statement_expressions: dict = field(default_factory=dict)
    definition_sites: dict = field(default_factory=dict)
    exception_handlers: dict = field(default_factory=dict)

    @property
    def source_digests(self):
        return {self.source_key: self.source_hash}

    compiler_id = f"{sys.implementation.name} {sys.version.split()[0]}"


def tuple_position(node):
    return node.lineno, node.end_lineno, node.col_offset, node.end_col_offset


def _constant_store_rhs(value):
    if isinstance(value, ast.Constant):
        return type(value.value) is str
    return isinstance(value, (ast.UnaryOp, ast.BinOp)) and all(
        isinstance(node, (ast.UnaryOp, ast.BinOp, ast.unaryop, ast.operator))
        or isinstance(node, ast.Constant)
        and type(node.value) in (str, int, float, bool, type(None))
        for node in ast.walk(value)
    )


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


def _name_load_sites(program, instructions, names):
    """Pair a module-frame read with its unique lexical name and instruction."""
    sites = {}
    for ins in instructions:
        opcode = getattr(ins, "baseopname", ins.opname)
        span = ins.positions
        if opcode not in {"LOAD_NAME", "LOAD_GLOBAL"} or span not in names:
            continue
        # Since CPython 3.11 the low LOAD_GLOBAL argument bit is a call flag.
        index = ins.arg >> 1 if opcode == "LOAD_GLOBAL" else ins.arg
        if (
            span in sites
            or not 0 <= index < len(program.co_names)
            or program.co_names[index] != ins.argval
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
    ast.Call: {"POP_TOP"},
    ast.Raise: {"RAISE_VARARGS"},
    ast.Assign: {"STORE_NAME", "STORE_GLOBAL"},
    ast.Delete: {"DELETE_NAME", "DELETE_GLOBAL"},
    ast.If: {"POP_JUMP_IF_TRUE", "POP_JUMP_IF_FALSE"},
    ast.For: {"GET_ITER"},
}


def _pool_display_items(value, parts):
    """Bind the entire compiler tuple to the display's ordered scalar literals."""
    for display in parts:
        if not isinstance(display, (ast.List, ast.Tuple)) or len(display.elts) != len(
            value
        ):
            continue
        if not all(
            isinstance(node, ast.Constant)
            and type(node.value) is type(item)
            and node.value == item
            for node, item in zip(display.elts, value)
        ):
            continue
        tokens = [
            ["literal", type(item).__name__, numeric_wire_value(item)] for item in value
        ]
        return dict(
            type="tuple", item_count=len(value), items_sha256=digest(json_bytes(tokens))
        )
    return None


def _expression_instruction(program, ins, parts, keywords):
    opcode = getattr(ins, "baseopname", ins.opname)
    arg, span = ins.arg, ins.positions
    row = dict(opcode=opcode, offset=ins.offset, position=list(span))
    if arg is not None:
        row["oparg"] = arg
    if opcode in {
        "LOAD_NAME",
        "LOAD_GLOBAL",
        "LOAD_ATTR",
        "STORE_NAME",
        "STORE_GLOBAL",
        "DELETE_NAME",
        "DELETE_GLOBAL",
    }:
        index = arg >> 1 if opcode in {"LOAD_GLOBAL", "LOAD_ATTR"} else arg
        if (
            not 0 <= index < len(program.co_names)
            or program.co_names[index] != ins.argval
        ):
            return None
        name = program.co_names[index]
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
    elif opcode in {"LOAD_CONST", "LOAD_SMALL_INT", "KW_NAMES"}:
        value = arg if opcode == "LOAD_SMALL_INT" else program.co_consts[arg]
        if opcode != "LOAD_SMALL_INT":
            row["index"] = arg
        if type(value) is tuple:
            if value in keywords:
                row.update(type="tuple", names=list(value))
            elif opcode == "LOAD_CONST" and (
                items := _pool_display_items(value, parts)
            ):
                row.update(items)
            else:
                return None
        elif opcode != "KW_NAMES" and any(
            isinstance(n, ast.Constant)
            and type(n.value) is type(value)
            and n.value == value
            for n in parts
        ):
            row.update(
                type=type(value).__name__,
                value_sha256=digest(json_bytes(numeric_wire_value(value))),
            )
        else:
            return None
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
        if not any(isinstance(n, ast.Call) for n in parts):
            return None
        if opcode != "TO_BOOL":
            row["target"] = ins.argval
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


def _condition_operand(test):
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        test = test.operand
    return test if isinstance(test, ast.Call) else None


def _suite_entry(ins, statement, kind, preceding=()):
    span = tuple_position(statement)

    def inside(spot):
        return (
            None not in spot
            and (span[0], span[2]) <= (spot.lineno, spot.col_offset)
            and (spot.end_lineno, spot.end_col_offset) <= (span[1], span[3])
        )

    if (
        ins.opname == "EXTENDED_ARG"
        or not inside(ins.positions)
        or any(inside(other.positions) for other in preceding)
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


def _branch_receipt(node, after, instructions, end, offsets, rows):
    """Match the actual successors of a call's truth jump to the source suites."""
    operand = _condition_operand(node.test)
    negated = operand is not node.test
    pos = tuple_position(operand)
    branch = instructions[end]
    if branch.opname != ("POP_JUMP_IF_TRUE" if negated else "POP_JUMP_IF_FALSE"):
        return None
    # 3.12 truth-tests the object in the jump. Later bytecode must produce an
    # exact bool first. A separate UNARY_NOT is not interchangeable with this lowering.
    truth = 1 if sys.version_info >= (3, 13) else 0
    if (
        len(rows) < 2 + truth
        or rows[-2 - truth]["opcode"] not in {"CALL", "CALL_KW"}
        or truth
        and rows[-2]["opcode"] != "TO_BOOL"
        or any(row["position"] != list(pos) for row in rows[-2 - truth :])
    ):
        return None
    then = end + 1
    if then >= len(instructions):
        return None
    marker = None
    if instructions[then].opname == "NOT_TAKEN":
        ins = instructions[then]
        if ins.positions != pos or ins.is_jump_target:
            return None
        marker = dict(opcode=ins.opname, offset=ins.offset, position=list(pos))
        then += 1
    other = offsets[branch.argval]
    if not then < other:
        return None

    def destination(index, statement, kind):
        return _suite_entry(
            instructions[index], statement, kind, instructions[then:index]
        )

    yes = destination(then, node.body[0], "suite")
    if node.orelse:
        no = destination(other, node.orelse[0], "suite")
    elif after is not None:
        no = destination(other, after, "continuation")
    else:
        no = _implicit_return(instructions, other, pos)
    if yes and no:
        return dict(
            condition_position=list(tuple_position(node.test)),
            operand_position=list(pos),
            negated=negated,
            fallthrough_marker=marker,
            then=yes,
            **{"else": no},
        )


def _iteration_receipt(node, after, instructions, end, offsets, jumps, handlers):
    if not isinstance(node.target, ast.Name) or any(
        isinstance(part, (ast.Break, ast.Continue))
        for statement in (*node.body, *node.orelse)
        for part in ast.walk(statement)
    ):
        return None
    acquire = instructions[end]
    if end + 3 >= len(instructions):
        return None
    step, store = instructions[end + 1 : end + 3]
    if (
        acquire.positions not in {tuple_position(node), tuple_position(node.iter)}
        or step.opname != "FOR_ITER"
        or step.positions != acquire.positions
        or store.opname not in {"STORE_NAME", "STORE_GLOBAL"}
        or store.positions != tuple_position(node.target)
        or store.argval != node.target.id
    ):
        return None
    exit_index = offsets.get(step.argval)
    if exit_index is None or exit_index < end + 4:
        return None
    back = instructions[exit_index - 1]
    cleanup = instructions[exit_index]
    if (
        back.opname != "JUMP_BACKWARD"
        or back.argval != step.offset
        or cleanup.opname != "END_FOR"
        or cleanup.positions != acquire.positions
        or jumps[step.offset] != [back.offset]
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
        any(i.positions != acquire.positions for i in clean)
        or any(
            i.offset in handlers
            or (i.is_jump_target and i.offset not in {step.offset, cleanup.offset})
            for i in instructions[end + 1 : stop]
        )
        or any(
            i.opcode in _JUMP_OPCODES for i in instructions[end + 3 : exit_index - 1]
        )
    ):
        return None
    body = _suite_entry(instructions[end + 3], node.body[0], "suite")
    if node.orelse:
        exhausted = _suite_entry(instructions[stop], node.orelse[0], "suite")
    elif after is not None:
        exhausted = _suite_entry(instructions[stop], after, "continuation")
    else:
        exhausted = _implicit_return(instructions, stop, acquire.positions)
    if not body or not exhausted:
        return None
    return dict(
        next=dict(_positioned_instruction(step), target=step.argval),
        store=_positioned_instruction(store),
        body=body,
        backedge=dict(_positioned_instruction(back), target=back.argval),
        cleanup=[_positioned_instruction(i) for i in clean],
        exhausted=exhausted,
    )


_JUMP_OPCODES = frozenset(
    dis.hasjump if hasattr(dis, "hasjump") else dis.hasjabs + dis.hasjrel
)


def _statement_expression(program, instructions, root, end, handlers):
    """Capture source operands and the terminal operation consuming them."""
    expr = root
    if isinstance(root, ast.If):
        expr = _condition_operand(root.test)
    elif isinstance(root, ast.For):
        expr = root.iter
    span = tuple_position(expr)
    parts, keywords = defaultdict(list), defaultdict(list)
    discarded_call = isinstance(root, ast.Call)
    needs_object_ports = not discarded_call
    for node in ast.walk(expr):
        if isinstance(node, (ast.Load, ast.Store, ast.Del, ast.keyword, ast.operator)):
            continue
        if not isinstance(
            node,
            (
                ast.Call,
                ast.BinOp,
                ast.Raise,
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
        needs_object_ports |= isinstance(
            node, (ast.Attribute, ast.Constant, ast.JoinedStr, ast.List, ast.Tuple)
        )
    if not (needs_object_ports or keywords):
        return None

    if isinstance(root, ast.For):
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
        row = _expression_instruction(
            program, ins, parts[ins.positions], keywords.get(ins.positions, ())
        )
        if row is None:
            return None
        depth += dis.stack_effect(ins.opcode, ins.arg)
        if depth < 0:
            return None
        rows.append(row)
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
    return dict(
        semantics_version=1, position=list(tuple_position(root)), instructions=rows
    )


def _statement_expression_sites(
    program, instructions, expressions, handlers, stores, following
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
    sites = {}
    for root, matches in ends.items():
        span = tuple_position(root)
        if len(matches) == 1 and span not in stores:
            receipt = _statement_expression(
                program, instructions, root, matches[0], handlers
            )
            if receipt is not None:
                if isinstance(root, ast.If):
                    edges = _branch_receipt(
                        root,
                        following.get(root),
                        instructions,
                        matches[0],
                        offsets,
                        receipt["instructions"],
                    )
                    if edges is None:
                        continue
                    receipt["branch"] = edges
                if isinstance(root, ast.For):
                    iteration = _iteration_receipt(
                        root,
                        following.get(root),
                        instructions,
                        matches[0],
                        offsets,
                        jumps,
                        handlers,
                    )
                    if not iteration:
                        continue
                    receipt["iteration"] = iteration
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


def _definition_sites(program, instructions, definitions, handlers):
    """Plain module definitions whose code object is loaded, constructed and stored."""
    found, repeated = {}, set()
    for i, make in enumerate(instructions):
        if (
            make.opname != "MAKE_FUNCTION"
            or make.arg not in (None, 0)
            or not 0 < i < len(instructions) - 1
        ):
            continue
        load, store = instructions[i - 1], instructions[i + 1]
        node = definitions.get(make.positions)
        if not (
            node
            and load.opname == "LOAD_CONST"
            and type(load.argval) is CodeType
            and load.argval is program.co_consts[load.arg]
            and load.positions == make.positions == store.positions
            and store.opname in {"STORE_NAME", "STORE_GLOBAL"}
            and program.co_names[store.arg] == store.argval == node.name
            and not any(
                op.is_jump_target or op.offset in handlers for op in (make, store)
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
        if make.positions in found:
            repeated.add(make.positions)
        rows = [
            dict(
                opcode=op.opname,
                offset=op.offset,
                position=list(op.positions),
                **({"oparg": op.arg} if op.arg is not None else {}),
            )
            for op in (load, make, store)
        ]
        rows[0]["index"] = load.arg
        rows[2].update(index=store.arg, name=node.name)
        found[make.positions] = dict(
            semantics_version=1,
            position=list(make.positions),
            instructions=rows,
            child=dict(
                name=child.co_name,
                qualname=child.co_qualname,
                first_line=child.co_firstlineno,
                code_sha256=code_digest(child),
            ),
        )
    return {k: v for k, v in found.items() if k not in repeated}


def _compiler_sites(tree, path, relative, source_hash):
    """Capture positioned construction and binding facts from one compilation."""
    if sys.implementation.name != "cpython" or not hasattr(CodeType, "co_positions"):
        return {}
    nodes, mappings, following, syntax = {}, {}, {}, deque([tree])
    while syntax:
        node = syntax.popleft()
        syntax.extend(ast.iter_child_nodes(node))
        for field in ("body", "orelse", "finalbody"):
            suite = getattr(node, field, None)
            if isinstance(suite, list):
                following.update(
                    (a, b)
                    for a, b in pairwise(suite)
                    if isinstance(a, (ast.If, ast.For)) and not a.orelse
                )
        if isinstance(node, ast.Dict):
            mappings[tuple_position(node)] = []
        elif (
            isinstance(node, ast.Tuple)
            and isinstance(node.ctx, ast.Load)
            and not any(isinstance(e, ast.Starred) for e in node.elts)
        ):
            nodes[tuple_position(node)] = node
    assignments, names, imports, expressions, tries = [], {}, {}, {}, []
    definitions = {}
    for node in scope_nodes(tree.body):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            expressions[tuple_position(node.targets[0])] = node
            if _constant_store_rhs(node.value):
                assignments.append(node)
        elif (
            isinstance(node, ast.FunctionDef)
            and not node.decorator_list
            and not node.args.defaults
            and not any(node.args.kw_defaults)
            and node.returns is None
            and not getattr(node, "type_params", ())
            and not any(
                x.annotation is not None
                for x in ast.walk(node.args)
                if isinstance(x, ast.arg)
            )
        ):
            definitions[tuple_position(node)] = node
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
    if not (
        nodes
        or mappings
        or assignments
        or names
        or imports
        or expressions
        or definitions
        or docstring is not None
    ):
        return {}
    optimize = sys.flags.optimize
    program = compile(tree, str(path), "exec", dont_inherit=True, optimize=optimize)
    unit = digest(json_bytes([relative, source_hash, ParsedUnit.compiler_id, optimize]))
    instructions = (
        list(dis.get_instructions(program))
        if assignments
        or names
        or imports
        or expressions
        or definitions
        or docstring is not None
        else None
    )
    entries = (
        dis.Bytecode(program).exception_entries
        if assignments
        or imports
        or expressions
        or tries
        or definitions
        or docstring is not None
        else ()
    )
    handlers = {e.target for e in entries}
    definition_sites = (
        _definition_sites(program, instructions, definitions, handlers)
        if definitions
        else {}
    )
    stores, document = (
        _constant_store_sites(program, instructions, assignments, handlers, docstring)
        if assignments or docstring is not None
        else ({}, {})
    )
    loads = _name_load_sites(program, instructions, names) if names else {}
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
    source_code = (
        dict(
            kind="module",
            source_key=relative,
            source_sha256=source_hash,
            code_sha256=code_digest(program),
            compiler=ParsedUnit.compiler_id,
            optimize=optimize,
        )
        if stores
        or document
        or loads
        or import_stores
        or statement_expressions
        or definition_sites
        else {}
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
        name_loads=loads,
        import_stores=import_stores,
        statement_expressions=statement_expressions,
        definition_sites=definition_sites,
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

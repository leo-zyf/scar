"""F2: turn captured procedure declarations, storage and statements into neutral operation bodies."""

from dataclasses import replace
import math
import struct
from .facts import Gap, ParsedUnit, kind, ptr
from .operations import Body, decl, literal, number, var


def _argument_access(symbol, location, interoperable):
    """Prove the ordinary dummy association rule for a direct data address."""
    if (
        not symbol.byref
        or symbol.category not in {"INTEGER", "REAL", "LOGICAL", "COMPLEX", "CHARACTER"}
        or "VARIABLE" not in symbol.attrs
        or symbol.attrs
        - {
            "VARIABLE",
            "DIMENSION",
            "IMPLICIT-TYPE",
            "DUMMY",
            "DUMMY(IN)",
            "DUMMY(OUT)",
            "DUMMY(INOUT)",
        }
    ):
        # POINTER, TARGET, CODIMENSION and other storage/association modes
        # need their own proof. A new compiler attribute is conservative here.
        return None
    # F1 admitted only explicit-shape or assumed-size bounds. F2 has already
    # checked the data ABI, including any separate character length parameter.
    shape = "scalar"
    if symbol.dims:
        shape = "assumed_size" if symbol.dims[-1][1] is None else "explicit_shape"
    return {
        "rule": "exclusive_modified_bytes",
        "scope": "invocation",
        "paths": "based_on_parameter",
        "readonly_when_accessed": interoperable and symbol.intent == "in",
        "basis": {
            "kind": "source_precondition",
            "rule_id": "fortran.ordinary_dummy_argument_association",
            "evidence": {
                "location": location,
                "intrinsic_type": symbol.category,
                "kind_value": symbol.size,
                "attributes": sorted(symbol.attrs),
                "array_form": shape,
                "passing": "direct_data_address",
                "interoperable": interoperable,
            },
        },
    }


def _bind_storage(builder, statements, common_blocks, globals_):
    """Solve EQUIVALENCE offset equations and declare explicit shared storage."""
    adjacency = {}
    common = {}
    sizes = {}
    data = []

    def layout(name):
        sym = builder.fact(name)
        if sym.dtype[0] not in "if" or sym.dummy:
            raise Gap(
                "shared_storage", "unsupported storage sequence " + name, sym.location
            )
        builder.bind_dimensions(name)
        if any(
            n is None or n.get("op") != "literal" or lo.get("op") != "literal"
            for lo, n in sym.bounds
        ):
            raise Gap(
                "shared_storage", "nonconstant storage sequence " + name, sym.location
            )
        sizes[name] = sym.size * math.prod(n["value"] for _, n in sym.bounds)
        return sym

    def endpoint(node):
        name = str(node if kind(node) == "Name" else node.items[0]).lower()
        sym = layout(name)
        offset = 0
        if kind(node) != "Name":
            if kind(node) != "Part_Ref":
                raise Gap("equivalence", "unsupported storage reference", node)
            args = node.items[1].items
            if len(args) != len(sym.bounds):
                raise Gap("equivalence", "incomplete subscripts", node)
            stride = sym.size
            for axis, (lo, extent) in zip(args, sym.bounds):
                e, _ = builder.expression(axis)
                if e.get("op") != "literal":
                    raise Gap("equivalence", "nonconstant subscript", node)
                offset += (e["value"] - lo["value"]) * stride
                stride *= extent["value"]
        adjacency.setdefault(name, [])
        return name, offset

    for stmt in statements:
        if kind(stmt) == "Data_Stmt":
            data.extend(stmt.items)
            continue
        if kind(stmt) == "Common_Stmt":
            for block, objects in stmt.items[0]:
                names = [str(o).lower() for o in objects.items]
                syms = [layout(n) for n in names]
                if len({s.size for s in syms}) != 1:
                    raise Gap(
                        "common_layout",
                        "mixed alignment requires verified COMMON offsets",
                        stmt,
                    )
                symbol = common_blocks[str(block).lower() if block else None]
                if symbol in common:
                    raise Gap("common_layout", "split COMMON declarations", stmt)
                common[symbol] = names
                for n in names:
                    adjacency.setdefault(n, [])
                for a, b in zip(names, names[1:]):
                    adjacency[a].append((b, sizes[a]))
                    adjacency[b].append((a, -sizes[a]))
            continue
        for group in stmt.items[1].items:
            a, oa = endpoint(group.items[0])
            for other in group.items[1].items:
                b, ob = endpoint(other)
                adjacency[a].append((b, oa - ob))
                adjacency[b].append((a, ob - oa))
    initial = {}
    for group in data:
        objects, values = group.items
        actuals = objects.items
        names = [
            str(obj if kind(obj) == "Name" else obj.items[0]).lower() for obj in actuals
        ]
        if not any(name in adjacency for name in names):
            continue
        targets = []
        for name, obj in zip(names, actuals):
            sym = layout(name)
            if kind(obj) == "Name":
                positions = range(0, sizes[name], sym.size)
            else:
                positions = [endpoint(obj)[1]]
            targets.extend((name, pos) for pos in positions)
        operands = []
        for value in values.items:
            if kind(value) == "Data_Stmt_Value":
                times, constant = value.items
                repetition = builder.expression(times)[0]
                if repetition.get("op") != "literal":
                    raise Gap("static_initializer", "nonconstant DATA repetition")
                operands.extend([constant] * repetition["value"])
            else:
                operands.append(value)
        if len(operands) != len(targets):
            raise Gap("static_initializer", "nonconforming DATA object list", group)
        for (name, pos), operand in zip(targets, operands):
            if name not in adjacency:
                continue
            sym = builder.fact(name)
            value = builder.expression(operand)[0]
            if value["op"] != "literal":
                raise Gap(
                    "static_initializer", "unfolded DATA constant " + name, operand
                )
            fmt = {
                "i8": "b",
                "i16": "h",
                "i32": "i",
                "i64": "q",
                "f32": "f",
                "f64": "d",
            }[sym.dtype]
            native = (
                int(value["value"])
                if sym.dtype.startswith("i")
                else float(value["value"])
            )
            initial.setdefault(name, []).append(
                (pos, struct.pack(builder.target.byte_order + fmt, native))
            )
    done = set()
    declarations = []
    for first in adjacency:
        if first in done:
            continue
        offsets = {first: 0}
        queue = [first]
        while queue:
            a = queue.pop()
            for b, d in adjacency[a]:
                displacement = offsets[a] + d
                if b not in offsets:
                    offsets[b] = displacement
                    queue.append(b)
                elif offsets[b] != displacement:
                    raise Gap("equivalence", "inconsistent displacement")
        base = min(offsets.values())
        size = max(v + sizes[n] for n, v in offsets.items()) - base
        blocks = [b for b, names in common.items() if names[0] in offsets]
        if len(blocks) > 1 or (blocks and offsets[common[blocks[0]][0]] != base):
            raise Gap("common_layout", "overlapping COMMON blocks or negative base")
        shared = blocks[0] if blocks else builder.fresh("storage")
        store = decl(shared, {"kind": "array", "of": "u8", "shape": [size]})
        initialized = {}
        saved = False
        for name, absolute in offsets.items():
            sym = builder.fact(name)
            offset = absolute - base
            builder.storage[name] = {
                "base": {**var(shared), **({"scope": "global"} if blocks else {})},
                "offset": offset,
                "bytes": sym.size,
            }
            saved |= bool(sym.attrs & {"DATA", "SAVE", "IMPLICIT-SAVE"})
            if sym.value is not None and name not in initial:
                raise Gap(
                    "static_initializer",
                    "associated storage lacks explicit initializer positions for "
                    + name,
                    sym.location,
                )
            for pos, value in initial.get(name, ()):
                for at, byte in enumerate(value, offset + pos):
                    if at in initialized:
                        raise Gap(
                            "equivalence", "multiply initialized byte", sym.location
                        )
                    initialized[at] = byte
        if initialized:
            store["init"] = {
                "op": "init_list",
                "items": [
                    literal("u8", initialized.get(pos, 0)) for pos in range(size)
                ],
            }
        if blocks:
            store.update(
                linkage="common",
                common_extent="minimum" if shared == "__BLNK__" else "exact",
                align_bytes=max(builder.fact(n).size for n in offsets),
            )
            globals_.append(store)
        else:
            if saved:
                store["storage"] = "static"
            declarations.append(store)
        done.update(offsets)
    return declarations


def normalize_unit(parsed: ParsedUnit) -> dict:
    """Build structured operations with per-invocation dimension and storage bindings."""
    out = {"functions": [], "externs": [], "diagnostics": list(parsed.diagnostics)}
    if parsed.diagnostics:
        return out
    globals_ = []
    externs = {}
    records = {}
    for procedure in parsed.procedures:
        name = procedure.name
        key = procedure.symbol
        main = procedure.main
        head = procedure.header
        link = procedure.external_name
        table = {n: replace(s, bounds=[]) for n, s in procedure.symbols.items()}
        if procedure.block_data:
            if procedure.problem:
                out["diagnostics"].append(procedure.problem)
                continue
            try:
                b = Body(
                    table,
                    name,
                    None,
                    link,
                    parsed.location,
                    parsed.defaults,
                    parsed.target,
                )
                temporary = []
                if _bind_storage(
                    b, procedure.storage, procedure.common_blocks, temporary
                ):
                    raise Gap("block_data", "initializer has no linked storage", head)
                globals_.extend(temporary)
            except Gap as gap:
                out["diagnostics"].append(
                    {
                        "code": "fortran." + gap.code,
                        "detail": str(gap),
                        **parsed.location(gap.node or head),
                    }
                )
            continue
        fn = {
            "name": key,
            "params": [],
            "result": "i32" if main else "void",
            "location": parsed.location(head),
        }
        out["functions"].append(fn)
        if procedure.problem:
            out["diagnostics"].append(procedure.problem)
            fn["calls"] = list(procedure.calls)
            continue
        try:
            dummy_names = procedure.formals
            names = set(dummy_names)
            result = procedure.result
            owner = table.get(name)
            interoperable = bool(owner and "BIND(C)" in owner.attrs)
            builder = Body(
                table,
                name,
                result,
                link,
                parsed.location,
                parsed.defaults,
                parsed.target,
                interoperable=interoperable,
            )
            builder.interfaces = procedure.interfaces
            builder.formats = procedure.formats
            builder.main = main
            if result:
                fn["result"] = builder.fact(result).dtype
            hidden = []
            for arg in dummy_names:
                s = builder.fact(arg)
                dt = s.dtype
                if s.attrs & {
                    "OPTIONAL",
                    "POINTER",
                    "ALLOCATABLE",
                    "VOLATILE",
                    "ASYNCHRONOUS",
                    "PROCEDURE",
                    "COARRAY",
                }:
                    raise Gap(
                        "argument_abi",
                        f"{arg} requires {s.category} {','.join(sorted(s.attrs))}",
                        s.location or head,
                    )
                if dt == "char":
                    if s.dims or not s.byref:
                        raise Gap(
                            "character_abi",
                            "array or VALUE character dummy",
                            s.location or head,
                        )
                    hidden_length = None
                    if not interoperable:
                        hidden_length = builder.fresh("character_length")
                        hidden.append(
                            {
                                "name": hidden_length,
                                "type": builder.target.size_type,
                                "character_of": arg,
                            }
                        )
                    elif str(s.char_length) not in {"None", "1"}:
                        raise Gap(
                            "character_abi",
                            "interoperable character descriptor",
                            s.location or head,
                        )
                    builder.bind_character(arg, hidden_length)
                if bool(s.dims) != ("DIMENSION" in s.attrs):
                    raise Gap(
                        "array_declaration",
                        f"array bounds of {arg}",
                        s.location or head,
                    )
                param = {
                    "name": arg,
                    "type": (
                        ptr(dt, interoperable and s.intent == "in") if s.byref else dt
                    ),
                    "intent": s.intent,
                }
                if s.dims:
                    builder.bind_dimensions(arg)
                    param.update(
                        layout="column_major",
                        shape=[d[1] for d in s.bounds],
                        lower_bounds=[d[0] for d in s.bounds],
                        elementwise_noalias=s.restricted and s.intent != "in",
                    )
                access = _argument_access(
                    s, parsed.location(s.location or head), interoperable
                )
                if access is not None:
                    param["access_precondition"] = access
                fn["params"].append(param)
            fn["params"].extend(hidden)
            declarations = _bind_storage(
                builder, procedure.storage, procedure.common_blocks, globals_
            )
            for n, s in table.items():
                if (
                    n in names
                    or n in builder.storage
                    or "VARIABLE" not in s.attrs
                    or "PARAMETER" in s.attrs
                ):
                    continue
                if (
                    s.attrs
                    & {
                        "POINTER",
                        "TARGET",
                        "IN-COMMON",
                        "ALLOCATABLE",
                        "VOLATILE",
                        "ASYNCHRONOUS",
                    }
                    or s.category == "CHARACTER"
                    or any(a.startswith("USE-ASSOC") for a in s.attrs)
                ):
                    raise Gap(
                        "local_storage",
                        f"{n} requires {s.category} {','.join(sorted(s.attrs))}",
                        s.location or head,
                    )
                if s.dims:
                    builder.bind_dimensions(n)
                    if all(
                        (v := d[1]) is not None and v.get("op") == "literal"
                        for d in s.bounds
                    ):
                        extent = math.prod(d[1]["value"] for d in s.bounds)
                        if extent > 0:
                            declaration = decl(
                                n, {"kind": "array", "of": s.dtype, "shape": [extent]}
                            )
                            if s.attrs & {"DATA", "SAVE", "IMPLICIT-SAVE"}:
                                declaration["storage"] = "static"
                                if s.value:
                                    if not (
                                        s.value.startswith("(/")
                                        and s.value.endswith("/)")
                                    ):
                                        raise Gap(
                                            "static_initializer",
                                            "noncanonical array initializer for " + n,
                                            s.location,
                                        )
                                    items = [
                                        number(v, s.dtype)
                                        for v in s.value[2:-2].split(",")
                                    ]
                                    if len(items) != extent:
                                        raise Gap(
                                            "static_initializer",
                                            "partial DATA initialization for " + n,
                                            s.location,
                                        )
                                    declaration["init"] = {
                                        "op": "init_list",
                                        "items": items,
                                    }
                            declarations.append(declaration)
                            continue
                    declarations.append(
                        {
                            "op": "buffer",
                            "name": n,
                            "type": s.dtype,
                            "extent": builder.size(n),
                        }
                    )
                else:
                    declaration = decl(n, s.dtype)
                    if s.attrs & {"DATA", "SAVE", "IMPLICIT-SAVE"}:
                        declaration["storage"] = "static"
                        if s.value:
                            declaration["init"] = number(s.value, s.dtype)
                    declarations.append(declaration)
            if result and all(rec["name"] != result for rec in declarations):
                declarations.append(decl(result, fn["result"]))
            steps = builder.statements(procedure.statements)
            if not steps or steps[-1]["op"] != "return":
                steps.append(
                    {
                        "op": "return",
                        **(
                            {"value": var(result)}
                            if result
                            else {"value": literal("i32", 0)} if main else {}
                        ),
                    }
                )
            fn["body"] = [*builder.prologue, *declarations, *steps]
            externs.update(builder.externs)
            records.update(builder.records)
        except Gap as gap:
            out["diagnostics"].append(
                {
                    "code": "fortran." + gap.code,
                    "detail": str(gap),
                    "symbol": key,
                    **parsed.location(gap.node or head),
                }
            )
        fn["calls"] = sorted(builder.calls) if "body" in fn else list(procedure.calls)
    out.update(
        externs=list(externs.values()), records=list(records.values()), globals=globals_
    )
    return out

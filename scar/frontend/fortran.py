"""Read checked GNU Fortran types and lower the selected source form to SCAR."""

from functools import lru_cache
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import math
import re
import shutil
import struct
import subprocess
from tempfile import TemporaryDirectory
from threading import Lock

from .fortran_ops import Body, Gap, Symbol, decl, kind, literal, number, ptr, var

_SCOPE = re.compile(r"^([ \t]*)procedure name = (\w+)[ \t]*$", re.M)
_SYMBOL = re.compile(r"^[ \t]*symtree: '(\w+)'[ \t]*\|\| symbol: '[^'\n]+'[^\n]*", re.M)
_MARKER = re.compile(r'^\s*#\s*(\d+)\s+"([^"]+)"')
_TOKEN = re.compile(
    r"""'(?:[^'\n]|'')*'|"(?:[^"\n]|"")*"|![^\n]*|\+(?:[ \t]*&[ \t]*(?:![^\n]*)?\n[ \t]*&?)?[ \t]*(?P<unary>\+)"""
)
_PARSER = Lock()
_PROGRAMS = {
    "Function_Subprogram",
    "Subroutine_Subprogram",
    "Main_Program",
    "Main_Program0",
}


_IO_PROBE = """subroutine scar_runtime_profile(s,i,r,z,ok)
character(*) :: s
integer(kind=4) :: i
real(kind=8) :: r
complex(kind=8) :: z
logical(kind=4) :: ok
write(*,*) s,i,r,z,ok
write(6,'(i4)') i
write(237) i
if (ok) stop 'x'
stop 1
end
"""


@dataclass
class IOProfile:
    record: dict
    signatures: dict
    soname: str
    flags: dict
    default_unit: int | None = None


def _io_descriptor(compiler, target_flags, probe, descriptor, extra, char_fields):
    """Read the selected compiler's checked structure layout and call ABI."""
    common_fields = {"flags", "line", "unit", "filename"}
    with TemporaryDirectory(prefix="scar-io-abi-") as temporary:
        cwd = Path(temporary)
        source = cwd / "probe.f90"
        dump = cwd / "types.raw"
        source.write_text(probe)
        p = subprocess.run(
            [
                compiler,
                *target_flags,
                "-fsyntax-only",
                f"-fdump-tree-original-raw={dump}",
                str(source),
            ],
            cwd=cwd,
            capture_output=True,
        )
        if p.returncode or not dump.exists():
            raise Gap("io_abi", "GNU runtime layout probe unavailable")
        data = dump.read_text()
    nodes = {
        int(i): (tag, fields)
        for i, tag, fields in re.findall(
            r"^@(\d+)\s+(\w+)\s+(.*?)(?=^@|\Z)", data, re.M | re.S
        )
    }

    def ref(i, key):
        m = re.search(r"\b" + re.escape(key) + r"\s*:\s*@(\d+)", nodes[i][1])
        return int(m[1]) if m else None

    def name(i):
        m = re.search(r"\bstrg:\s*([^\s]+)", nodes[i][1])
        return m[1] if m else ""

    def constant(i):
        return int(re.search(r"\bint:\s*(-?\d+)", nodes[i][1])[1])

    def native_type(i):
        tag, fields = nodes[i]
        if tag == "integer_type":
            return ("u" if "sign: unsigned" in fields else "i") + re.search(
                r"prec:\s*(\d+)", fields
            )[1]
        if tag == "pointer_type":
            return ptr("void")
        if tag == "boolean_type":
            return "bool"
        if tag == "void_type":
            return "void"
        raise Gap("io_abi", "runtime parameter type " + tag)

    parameters = {}
    members = []
    flags_by_record = {}
    record_id = None
    for i, (tag, fields) in nodes.items():
        if tag == "function_decl" and (symbol := name(ref(i, "name"))).startswith(
            "_gfortran_"
        ):
            chain = ref(ref(i, "type"), "prms")
            types = []
            while chain:
                t = native_type(ref(chain, "valu"))
                if t != "void":
                    types.append(t)
                chain = ref(chain, "chan")
            if symbol.startswith("_gfortran_transfer_") and len(types) > 1:
                types[1] = ptr("void", True)
            if symbol == "_gfortran_stop_string":
                types[0] = ptr("void", True)
            parameters[symbol] = types
        if (
            tag == "record_type"
            and ref(i, "name")
            and name(ref(i, "name")) == descriptor
        ):
            record_id = i
        if tag == "field_decl":
            scope = ref(i, "scpe")
            owner = name(ref(scope, "name")) if scope and ref(scope, "name") else ""
            ident = name(ref(i, "name"))
            common = owner == "__st_parameter_common" and ident in common_fields
            own = owner == descriptor and ident in extra
            if common or own:
                typ = (
                    ptr("char", True)
                    if ident == "filename" or ident in char_fields
                    else native_type(ref(i, "type"))
                )
                members.append(
                    (constant(ref(i, "bpos")), constant(ref(i, "size")), ident, typ)
                )
            if (
                owner == descriptor
                and ident == "common"
                and constant(ref(i, "bpos")) != 0
            ):
                raise Gap("io_abi", "unexpected nested runtime prefix offset")
        if tag == "modify_expr":
            lhs, rhs = ref(i, "op 0"), ref(i, "op 1")
            if nodes[lhs][0] != "component_ref":
                continue
            field = ref(lhs, "op 1")
            ident = name(ref(field, "name"))
            storage = ref(lhs, "op 0")
            while nodes[storage][0] == "component_ref":
                storage = ref(storage, "op 0")
            if nodes[storage][0] == "var_decl" and nodes[rhs][0] == "integer_cst":
                flags_by_record.setdefault(storage, {})[ident] = constant(rhs)
    if record_id is None or {row[2] for row in members} != common_fields | extra:
        raise Gap("io_abi", "incomplete GNU output descriptor")
    size = constant(ref(record_id, "size")) // 8
    align = int(re.search(r"algn:\s*(\d+)", nodes[record_id][1])[1]) // 8
    layouts = []
    cursor = 0
    for offset, width, ident, typ in sorted(
        set((o, w, n, json.dumps(t, sort_keys=True)) for o, w, n, t in members)
    ):
        if cursor > offset or offset % 8 or width % 8:
            raise Gap("io_abi", "overlapping or unaligned runtime fields")
        if cursor < offset:
            layouts.append(
                {
                    "name": f"padding_{cursor}",
                    "type": {
                        "kind": "array",
                        "of": "u8",
                        "shape": [(offset - cursor) // 8],
                    },
                    "offset_bits": cursor,
                }
            )
        layouts.append({"name": ident, "type": json.loads(typ), "offset_bits": offset})
        cursor = offset + width
    if cursor < size * 8:
        layouts.append(
            {
                "name": "reserved",
                "type": {"kind": "array", "of": "u8", "shape": [size - cursor // 8]},
                "offset_bits": cursor,
            }
        )
    identity = sha256(
        json.dumps([size, align, layouts], sort_keys=True).encode()
    ).hexdigest()[:16]
    record = dict(
        name="io_" + identity,
        kind="struct",
        fields=layouts,
        size_bytes=size,
        align_bytes=align,
    )
    cmds = [v for v in flags_by_record.values() if "flags" in v]
    soname_path = Path(
        subprocess.check_output(
            [compiler, *target_flags, "-print-file-name=libgfortran.so"], text=True
        ).strip()
    )
    if not soname_path.is_file():
        raise Gap("io_abi", "target library libgfortran.so unavailable")
    major = re.search(r"(libgfortran\.so\.\d+)", soname_path.resolve().name)
    if major is None:
        raise Gap("io_abi", "target GNU runtime SONAME unavailable")
    return record, parameters, cmds, major[1]


@lru_cache(maxsize=16)
def _io_abi(compiler, target_flags):
    record, signatures, rows, soname = _io_descriptor(
        compiler,
        target_flags,
        _IO_PROBE,
        "__st_parameter_dt",
        {"format", "format_len"},
        {"format"},
    )
    modes = {
        (
            "format"
            if "format_len" in row
            else "unformatted" if row.get("unit") == 237 else "list"
        ): row
        for row in rows
    }
    if len(rows) != 3 or set(modes) != {"list", "format", "unformatted"}:
        raise Gap("io_abi", "ambiguous GNU data transfer flags")
    return IOProfile(
        record,
        signatures,
        soname,
        {k: row["flags"] for k, row in modes.items()},
        modes["list"]["unit"],
    )


@lru_cache(maxsize=32)
def _file_abi(compiler, target_flags, operation, controls):
    words = {
        "file",
        "form",
        "access",
        "status",
        "action",
        "position",
        "blank",
        "delim",
        "pad",
    }
    if operation not in {"open", "close"} or set(controls) - words - {"recl"}:
        raise Gap("io_control", operation + " (" + ", ".join(controls) + ")")
    actuals = [f"{k}={'s' if k in words else 'n'}" for k in controls]
    options = ", ".join(["unit=n", *actuals])
    source = f"subroutine profile(s,n)\ncharacter(*) s\ninteger(kind=4) n\n{operation}({options})\nend\n"
    strings = set(controls) & words
    extra = set(controls) | {n + "_len" for n in strings}
    if "recl" in extra:
        extra.remove("recl")
        extra.add("recl_in")
    record, signatures, rows, soname = _io_descriptor(
        compiler, target_flags, source, "__st_parameter_" + operation, extra, strings
    )
    if len(rows) != 1:
        raise Gap("io_abi", operation + " flag profile unavailable")
    return IOProfile(record, signatures, soname, {operation: rows[0]["flags"]})


def _without_builtin_plus(dump):
    # GNU emits U+ only for the intrinsic operation after resolution. Preserve
    # user-defined operator calls and quoted strings in the comparison.
    token = re.compile(r"""'(?:[^'\n]|'')*'|\(U\+ |[()]""")
    result = []
    stack = []
    cursor = 0
    for m in token.finditer(dump):
        result.append(dump[cursor : m.start()])
        value = m[0]
        cursor = m.end()
        if value == "(U+ ":
            stack.append(True)
            continue
        if value == "(":
            stack.append(False)
        if value == ")" and stack and stack.pop():
            continue
        result.append(value)
    return "".join(result) + dump[cursor:]


def _facts(dump):
    """GNU's resolved symbol tables include implicit types and folded parameters."""
    scopes = list(_SCOPE.finditer(dump))
    tables, imports, ancestors = {}, [], []
    for i, scope in enumerate(scopes):
        depth = len(scope[1].expandtabs())
        while ancestors and ancestors[-1][0] >= depth:
            ancestors.pop()
        ancestors.append((depth, scope[2].lower()))
        key = tuple(row[1] for row in ancestors)
        text = dump[
            scope.end() : scopes[i + 1].start() if i + 1 < len(scopes) else None
        ]
        text = re.split(r"^[ \t]*code:[ \t]*$", text, maxsplit=1, flags=re.M)[0]
        entries = list(_SYMBOL.finditer(text))
        table = tables[key] = {}
        for j, entry in enumerate(entries):
            fields = text[
                entry.end() : entries[j + 1].start() if j + 1 < len(entries) else None
            ]
            name = entry[1].lower()
            flags = re.search(r"^[ \t]*attributes:[ \t]*\((.*)\)[ \t]*$", fields, re.M)
            if flags is None:
                origin = re.search(r"from namespace '(\w+)'", entry[0])
                if origin:
                    imports.append((key, name, origin[1].lower()))
                continue
            typ = re.search(
                r"^[ \t]*type spec[ \t]*:[ \t]*\((\w+) (\d+)\b", fields, re.M
            )
            char_type = re.search(
                r"^[ \t]*type spec[ \t]*:[ \t]*\(CHARACTER .*? (\d+)(?: [A-Z_ ]+)?\)",
                fields,
                re.M,
            )
            value = re.search(r"^[ \t]*value:[ \t]*(.*)$", fields, re.M)
            binding = re.search(r"binding_label: '([^']+)'", entry[0])
            result = re.search(r"^[ \t]*result: (\w+)", fields, re.M)
            category = "CHARACTER" if char_type else typ[1] if typ else "UNRESOLVED"
            width = (
                int(char_type[1] if char_type else typ[2]) if char_type or typ else 0
            )
            table[name] = Symbol(
                category,
                width,
                frozenset(flags[1].split()),
                value[1].strip() if value else None,
                binding[1] if binding else None,
                result[1] if result else None,
            )
    for key, name, owner in imports:
        for ancestor in reversed(range(len(key))):
            target = (*key[:ancestor], owner)
            if target in tables and name in tables[target]:
                tables[key][name] = tables[target][name]
                break
    return tables


def _unit_name(unit):
    return (
        "main__"
        if kind(unit) == "Main_Program0"
        else str(unit.content[0].items[1]).lower()
    )


def _programs(tree, parents=()):
    name = kind(tree)
    if name in _PROGRAMS:
        yield parents, tree
        parents = (*parents, _unit_name(tree))
    elif name == "Module":
        parents = (*parents, str(tree.content[0].items[1]).lower())
    for child in getattr(tree, "content", ()):
        if kind(child) in _PROGRAMS or kind(child) in {
            "Module",
            "Module_Subprogram_Part",
            "Internal_Subprogram_Part",
        }:
            yield from _programs(child, parents)


def _shape(spec):
    if spec is None:
        return ()
    if kind(spec) == "Assumed_Size_Spec":
        prefix, lo = spec.items
        return (*_shape(prefix), (lo, None))
    if kind(spec) != "Explicit_Shape_Spec_List":
        raise Gap("array_descriptor", kind(spec), spec)
    return tuple(axis.items for axis in spec.items)


def _declarations(spec, symbols, interfaces, associations):
    permitted = {
        "Use_Stmt",
        "Implicit_Stmt",
        "Parameter_Stmt",
        "External_Stmt",
        "Intrinsic_Stmt",
        "Intent_Stmt",
        "Value_Stmt",
        "Data_Stmt",
        "Save_Stmt",
        "Comment",
    }
    for stmt in getattr(spec, "content", ()):
        tag = kind(stmt)
        entries = []
        if tag == "Type_Declaration_Stmt":
            type_spec, attrs, decls = stmt.items
            common = next(
                (
                    attr.items[1]
                    for attr in (attrs.items if attrs else ())
                    if kind(attr) == "Dimension_Attr_Spec"
                ),
                None,
            )
            for entity in decls.items:
                nm, dims, length, init = entity.items
                nm = str(nm).lower()
                if nm in symbols and symbols[nm].category == "CHARACTER":
                    selector = type_spec.items[1]
                    if length is None and selector is not None:
                        length = (
                            selector.items[0]
                            if kind(selector) == "Char_Selector"
                            else selector.items[1]
                        )
                    if kind(length) == "Char_Length":
                        length = length.items[1]
                    symbols[nm].char_length = length
                if (
                    init is not None
                    and nm in symbols
                    and "PARAMETER" not in symbols[nm].attrs
                ):
                    raise Gap("static_initializer", nm, stmt)
                entries.append((nm, dims or common))
        elif tag == "Dimension_Stmt":
            entries = [(str(nm).lower(), dims) for nm, dims in stmt.items[0]]
        elif tag in {"Equivalence_Stmt", "Common_Stmt", "Data_Stmt"}:
            associations.append(stmt)
        elif tag == "Interface_Block":
            if stmt.content[0].items[0] is not None:
                raise Gap("interface", "generic or abstract interface", stmt)
            for unit in stmt.content[1:-1]:
                if kind(unit) not in {"Function_Body", "Subroutine_Body"}:
                    raise Gap("interface", kind(unit), unit)
                header = unit.content[0]
                params = {
                    str(a).lower(): dict(value=False, intent="unspecified")
                    for a in (header.items[2].items if header.items[2] else [])
                }
                for part in unit.content[1:-1]:
                    for item in getattr(part, "content", ()):
                        if kind(item) == "Implicit_Part":
                            continue
                        if kind(item) != "Type_Declaration_Stmt":
                            raise Gap("interface_declaration", kind(item), item)
                        _, attributes, entities = item.items
                        attrs = {
                            str(a).lower()
                            for a in (attributes.items if attributes else ())
                        }
                        if any(
                            a.startswith(
                                (
                                    "optional",
                                    "pointer",
                                    "allocatable",
                                    "codimension",
                                    "asynchronous",
                                )
                            )
                            for a in attrs
                        ):
                            raise Gap(
                                "interface_abi", "descriptor or optional dummy", item
                            )
                        for entity in entities.items:
                            arg = str(entity.items[0]).lower()
                            if arg in params:
                                params[arg].update(
                                    value="value" in attrs,
                                    intent=next(
                                        (
                                            a[7:-1]
                                            for a in attrs
                                            if a.startswith("intent(")
                                        ),
                                        "unspecified",
                                    ),
                                )
                bind = (
                    "BIND" in str(header.items[3]).upper()
                    if len(header.items) > 3
                    else False
                )
                interfaces[str(header.items[1]).lower()] = {
                    "params": list(params.values()),
                    "bind_c": bind,
                }
        elif tag in {"Implicit_Part", "Declaration_Construct"}:
            _declarations(stmt, symbols, interfaces, associations)
        elif tag not in permitted:
            raise Gap("declaration", tag, stmt)
        for nm, dims in entries:
            if nm in symbols:
                sym = symbols[nm]
                sym.location = stmt
                if dims:
                    sym.dims = _shape(dims)


def _enabled(flags, positive, negative, default=False):
    enabled = default
    for flag in flags:
        if flag == positive:
            enabled = True
        elif flag == negative:
            enabled = False
    return enabled


@lru_cache(maxsize=16)
def _target_storage(compiler, flags):
    probe = subprocess.run(
        [compiler, *flags, "-cpp", "-dM", "-E", "-x", "f95", "-"],
        input=b"",
        capture_output=True,
    )
    width = re.search(rb"^#define __SIZEOF_SIZE_T__ (\d+)$", probe.stdout, re.M)
    if probe.returncode or not width or int(width[1]) not in {4, 8}:
        raise Gap("abi", "GNU hidden CHARACTER length width unavailable")
    order = re.search(
        rb"^#define __BYTE_ORDER__ __ORDER_(LITTLE|BIG)_ENDIAN__$", probe.stdout, re.M
    )
    if not order:
        raise Gap("abi", "target byte order unavailable")
    return f"u{8*int(width[1])}", ("<" if order[1] == b"LITTLE" else ">")


def _bind_storage(builder, statements, flags, globals_):
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
                symbol = (
                    _linker_name(str(block).lower(), flags) if block else "__BLNK__"
                )
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
                (pos, struct.pack(builder.byte_order() + fmt, native))
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


def _linker_name(name, flags, module=None, binding=None):
    if binding:
        return binding
    if module:
        return f"__{module}_MOD_{name}"
    if not _enabled(flags, "-funderscoring", "-fno-underscoring", True):
        return name
    second = _enabled(
        flags, "-fsecond-underscore", "-fno-second-underscore", "-ff2c" in flags
    )
    return name + ("__" if second and "_" in name else "_")


def _selected_text(source, path):
    lines = []
    locations = []
    current = str(path)
    line = 1
    for original in source.splitlines():
        marker = _MARKER.match(original)
        if marker:
            line, current = int(marker[1]), marker[2]
        else:
            lines.append(original)
            locations.append((current, line))
            line += 1
    return "\n".join(lines) + "\n", locations


def _logical_flags(flags):
    result = []
    i = 0
    while i < len(flags):
        x = flags[i]
        i += 1
        if x == "-J":
            result.extend(("-I", flags[i]))
            i += 1
        elif x.startswith("-J"):
            result.extend(("-I", x[2:]))
        elif x in ("-o", "-MF", "-MT", "-MQ"):
            i += 1
        elif x in {"-c", "-S", "-E", "-M", "-MM", "-MD", "-MMD"} or x.startswith(
            ("-fdump-", "-save-temps")
        ):
            continue
        else:
            result.append(x)
    return result


def _compile_capture(path, root, flags, tmp):
    compiler = shutil.which("gfortran")
    if compiler is None:
        raise Gap("compiler", "gfortran is required")
    flags = _logical_flags(flags)

    def capture(args):
        p = subprocess.run(
            [compiler, *flags, *args],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )
        if p.returncode:
            raise Gap("compiler", p.stderr.decode(errors="replace").strip()[:3000])
        return p.stdout.decode("utf-8", errors="replace")

    semantic_flags = ["-J", str(tmp), "-fsyntax-only", "-fdump-fortran-original"]
    dump = capture([*semantic_flags, str(path)])
    fixed = path.suffix.lower() in {".for", ".f", ".ftn", ".f77"}
    language = None
    for i, flag in enumerate(flags):
        if flag == "-x" and i + 1 < len(flags):
            language = flags[i + 1]
        elif flag.startswith("-x") and len(flag) > 2:
            language = flag[2:]
    if language not in {None, "none"}:
        fixed = language.startswith("f77")
    fixed = _enabled(flags, "-ffixed-form", "-ffree-form", fixed)
    preprocess = _enabled(
        flags,
        "-cpp",
        "-nocpp",
        path.suffix != path.suffix.lower()
        or bool(language and language.endswith("-cpp-input")),
    )
    source = (
        capture(["-E", str(path)])
        if preprocess
        else path.read_text(encoding="utf-8", errors="surrogateescape")
    )
    text, locations = _selected_text(source, path)
    width = 72 if fixed else 132
    prefix = "-ffixed-line-length-" if fixed else "-ffree-line-length-"
    for flag in flags:
        if flag.startswith(prefix):
            chosen = flag[len(prefix) :]
            width = 0 if chosen == "none" else int(chosen)
    if width:
        # Sequence numbers in fixed-form columns 73–80 are not source tokens.
        # GNU's fixed tab form uses a leading tab to start a statement or a
        # following 1–9 to indicate continuation in column six.
        selected = []
        for line in text.splitlines():
            if fixed and line.startswith("\t"):
                tail = line[1:]
                line = (
                    "     " if tail[:1] and tail[0] in "123456789" else "      "
                ) + tail
            selected.append(line.expandtabs()[:width])
        text = "\n".join(selected) + "\n"
    if _enabled(flags, "-fopenmp", "-fno-openmp") or _enabled(
        flags, "-fopenacc", "-fno-openacc"
    ):
        for line, (file, source_line) in zip(text.splitlines(), locations):
            if re.match(r"(?:\s*!|[cC*])\$", line):
                raise Gap(
                    "directive",
                    f"{file}:{source_line}: active accelerator or parallel directive",
                )
    return text, locations, fixed, dump, capture, semantic_flags


def _syntax(text, fixed, root, path, flags):
    from fparser.common.readfortran import FortranStringReader
    from fparser.common.sourceinfo import FortranFormat
    from fparser.two.parser import ParserFactory

    dirs = [str(path.parent), str(root)]
    for i, f in enumerate(flags):
        if f == "-I" and i + 1 < len(flags):
            dirs.append(str(root / flags[i + 1]))
        elif f.startswith("-I") and len(f) > 2:
            dirs.append(str(root / f[2:]))
    reader = FortranStringReader(text, include_dirs=dirs, ignore_comments=True)
    reader.set_format(FortranFormat(is_free=not fixed, is_strict=False))
    # fparser's parser factory owns a process-global scoped symbol table.
    with _PARSER:
        return ParserFactory().create(std="f2008")(reader)


def _translate_unit(path, root, flags):
    from fparser.two.utils import FortranSyntaxError, walk
    from fparser.two import Fortran2003 as F

    out = {"functions": [], "externs": [], "diagnostics": []}
    default_int = "i64" if "-fdefault-integer-8" in flags else "i32"
    default_real = "f64" if "-fdefault-real-8" in flags else "f32"
    double = (
        "f128" if default_real == "f64" and "-fdefault-double-8" not in flags else "f64"
    )
    defaults = (default_int, default_real, double)
    with TemporaryDirectory(prefix="scar-fortran-") as temporary:
        tmp = Path(temporary)
        text, locations, fixed, dump, capture, semantic_flags = _compile_capture(
            path, root, flags, tmp
        )
        try:
            tree = _syntax(text, fixed, root, path, flags)
        except FortranSyntaxError as exc:
            compatible = _TOKEN.sub(
                lambda m: m[0][:-1] + " " if m["unary"] else m[0], text
            )
            if compatible == text:
                raise Gap("syntax", str(exc)) from exc
            candidate = tmp / ("syntax.f" if fixed else "syntax.f90")
            candidate.write_text(compatible)
            checked = capture(
                [
                    *semantic_flags,
                    "-x",
                    "f77" if fixed else "f95",
                    "-nocpp",
                    "-I",
                    str(path.parent),
                    str(candidate),
                ]
            )
            if _without_builtin_plus(checked) != _without_builtin_plus(dump):
                raise Gap("syntax", str(exc)) from exc
            tree = _syntax(compatible, fixed, root, path, flags)
    tables = _facts(dump)

    def where(node):
        current = node
        while current is not None:
            item = getattr(current, "item", None)
            if getattr(item, "span", None):
                at = item.span[0] - 1
                if 0 <= at < len(locations):
                    file, line = locations[at]
                    return {"file": file, "line": line}
            children = getattr(current, "content", ())
            current = children[0] if children else None
        return {"file": str(path), "line": 1}

    externs = {}
    records = {}
    globals_ = []
    abi_flags = tuple(
        flag for flag in flags if flag.startswith(("-m", "-fpack-derived"))
    )
    for block in walk(tree, F.Block_Data):
        head = block.content[0]
        source_name = head.items[0]
        name = str(source_name).lower() if source_name else "scar_unnamed_common_init"
        try:
            if source_name:
                table = tables.get((name,))
            else:
                namespaces = [
                    s
                    for s in re.split(r"(?=^Namespace:)", dump, flags=re.M)
                    if "symtree:" in s and not _SCOPE.search(s)
                ]
                table = (
                    _facts(f"procedure name = {name}\n" + namespaces[0]).get((name,))
                    if len(namespaces) == 1
                    else None
                )
            if table is None:
                raise Gap("scope", "unresolved BLOCK DATA namespace", head)
            spec = next(
                (item for item in block.content if kind(item) == "Specification_Part"),
                None,
            )
            associations = []
            _declarations(spec, table, {}, associations)
            b = Body(
                table, name, None, lambda n: _linker_name(n, flags), where, defaults
            )
            b.byte_order = lambda: _target_storage(shutil.which("gfortran"), abi_flags)[
                1
            ]
            temporary = []
            declarations = _bind_storage(b, associations, flags, temporary)
            if declarations:
                raise Gap("block_data", "initializer has no linked storage", head)
            globals_.extend(temporary)
        except Gap as gap:
            out["diagnostics"].append(
                {
                    "code": "fortran." + gap.code,
                    "detail": str(gap),
                    **where(gap.node or head),
                }
            )
    program_units = list(_programs(tree))
    bindings = {}
    for parents, p in program_units:
        name = _unit_name(p)
        symbols = tables.get((*parents, name), {})
        owner = symbols.get(name)
        bindings[(*parents, name)] = _linker_name(
            name,
            flags,
            parents[-1] if parents else None,
            owner.binding if owner else None,
        )
    for parents, p in program_units:
        head = p.content[0]
        name = _unit_name(p)
        main = kind(p) in {"Main_Program", "Main_Program0"}
        key = "main" if main else bindings[(*parents, name)]
        fn = {
            "name": key,
            "params": [],
            "result": "i32" if main else "void",
            "location": where(head),
        }

        def link(n):
            fact = tables.get((*parents, name), {}).get(n)
            if fact and fact.binding:
                return fact.binding
            return (
                bindings.get((*parents, n))
                or bindings.get((n,))
                or _linker_name(n, flags)
            )

        out["functions"].append(fn)
        try:
            if _enabled(flags, "-ff2c", "-fno-f2c"):
                raise Gap("abi", "alternate return convention -ff2c", head)
            if len(parents) > 1 or (
                parents
                and len(tables.get(tuple(parents), {}))
                and "MODULE"
                not in tables[tuple(parents)]
                .get(parents[-1], Symbol("", 0, frozenset()))
                .attrs
            ):
                raise Gap(
                    "host_association", "internal procedure with host scope", head
                )
            table = tables.get((*parents, name))
            if table is None:
                raise Gap("scope", f"missing GNU symbol table for {name}", head)
            spec = next((n for n in p.content if kind(n) == "Specification_Part"), None)
            interfaces = {}
            associations = []
            _declarations(spec, table, interfaces, associations)
            owner = table.get(name)
            result = (
                (owner.result or name)
                if kind(p) == "Function_Subprogram" and owner
                else None
            )
            dummy_names = (
                [str(a).lower() for a in head.items[2].items]
                if not main and head.items[2]
                else []
            )
            names = set(dummy_names)

            builder = Body(table, name, result, link, where, defaults)
            builder.interfaces = interfaces
            builder.main = main
            builder.size_type = lambda: _target_storage(
                shutil.which("gfortran"), abi_flags
            )[0]
            builder.byte_order = lambda: _target_storage(
                shutil.which("gfortran"), abi_flags
            )[1]
            builder.io_profile = lambda: _io_abi(shutil.which("gfortran"), abi_flags)
            builder.file_profile = lambda operation, controls: _file_abi(
                shutil.which("gfortran"), abi_flags, operation, controls
            )
            builder.formats = {
                str(n.item.label): str(n.items[1]) for n in walk(p, F.Format_Stmt)
            }
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
                    if not owner or "BIND(C)" not in owner.attrs:
                        hidden_length = builder.fresh("character_length")
                        hidden.append(
                            {
                                "name": hidden_length,
                                "type": builder.size_type(),
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
                    "type": ptr(dt, s.intent == "in") if s.byref else dt,
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
                fn["params"].append(param)
            fn["params"].extend(hidden)
            declarations = _bind_storage(builder, associations, flags, globals_)
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
            execution = next(
                (n.content for n in p.content if kind(n) == "Execution_Part"), []
            )
            steps = builder.statements(execution)
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
                    **where(gap.node or head),
                }
            )
        if "body" in fn:
            fn["calls"] = sorted(builder.calls)
            continue
        call_names = {str(c.items[0]).lower() for c in walk(p, F.Call_Stmt)}
        for c in walk(p, (F.Function_Reference, F.Part_Ref)):
            cname = str(c.items[0]).lower()
            fact = tables.get((*parents, name), {}).get(cname)
            if (
                fact
                and "PROCEDURE" in fact.attrs
                and "INTRINSIC-PROC" not in fact.attrs
            ):
                call_names.add(cname)
        fn["calls"] = sorted({link(c) for c in call_names})
    out["externs"] = list(externs.values())
    out["records"] = list(records.values())
    out["globals"] = globals_
    return out


def translate(path: Path, flags: list[str], *, root: Path) -> dict:
    path = path.resolve()
    root = root.resolve()
    try:
        return _translate_unit(path, root, flags)
    except (ImportError, Gap, subprocess.SubprocessError, OSError) as error:
        description = str(error)
        line = re.search(r"(?:at line |:(?=\d+:))(\d+)", description, re.I)
        return {
            "functions": [],
            "externs": [],
            "diagnostics": [
                {
                    "code": "fortran." + getattr(error, "code", "toolchain"),
                    "detail": description,
                    "file": str(path),
                    "line": int(line[1]) if line else 1,
                }
            ],
        }

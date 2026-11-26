"""F1: capture GNU declarations, selected Fortran syntax and target ABI facts."""

from dataclasses import dataclass, field, replace
from functools import lru_cache
from hashlib import sha256
from itertools import chain
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
from tempfile import TemporaryDirectory
from threading import Lock
from time import time_ns

from scar.contract import digest, json_bytes
from ..gnu_numeric import numeric_option_rows


class Gap(ValueError):
    def __init__(self, code, detail, node=None):
        super().__init__(detail)
        self.code, self.node = code, node


def kind(node):
    return type(node).__name__


def ptr(dtype, readonly=False):
    return {"kind": "ptr", "to": {"kind": "const", "of": dtype} if readonly else dtype}


@dataclass
class Symbol:
    category: str
    size: int
    attrs: frozenset
    value: str | None = None
    binding: str | None = None
    result: str | None = None
    dims: tuple = ()
    location: object = None
    bounds: list = field(default_factory=list)
    char_length: object = None
    compiler_type_metadata: bool = False

    @property
    def source_variable(self):
        return (
            "VARIABLE" in self.attrs
            and "PARAMETER" not in self.attrs
            and not self.compiler_type_metadata
        )

    @property
    def dtype(self):
        category = self.category
        ordinary = {"REAL": "f", "INTEGER": "i", "LOGICAL": "i"}
        if category in ordinary and self.size in (1, 2, 4, 8):
            return ordinary[category] + str(8 * self.size)
        if category == "CHARACTER" and self.size == 1:
            return "char"
        if category == "COMPLEX" and self.size in (4, 8):
            return "c" + str(16 * self.size)
        raise Gap("type", f"{category} kind {self.size}", self.location)

    @property
    def dummy(self):
        return any(v == "DUMMY" or v.startswith("DUMMY(") for v in self.attrs)

    @property
    def intent(self):
        return next(
            (v[6:-1].lower() for v in self.attrs if v.startswith("DUMMY(")),
            "unspecified",
        )

    @property
    def byref(self):
        return self.dummy and "VALUE" not in self.attrs

    @property
    def restricted(self):
        return self.dummy and not self.attrs & {
            "POINTER",
            "TARGET",
            "ASYNCHRONOUS",
            "VOLATILE",
            "ALLOCATABLE",
        }


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
        # GNU materializes a derived type's dispatch table and default object
        # even when the only source operation is a USE declaration. Identify
        # the linked descriptors; keep user variables and their bindings intact.
        for name, symbol in table.items():
            if not name.startswith("__vtab_") or not symbol.value:
                continue
            family = name.removeprefix("__vtab_")
            type_name = "__vtype_" + family
            init_name = "__def_init_" + family
            descriptor = table.get(type_name)
            initializer = table.get(init_name)
            value = symbol.value.lower()
            if (
                descriptor is not None
                and "DERIVED" in descriptor.attrs
                and initializer is not None
                and "ARTIFICIAL" in initializer.attrs
                and value.startswith(type_name + "(")
                and re.search(r"\b" + re.escape(init_name) + r"\b", value)
            ):
                symbol.compiler_type_metadata = initializer.compiler_type_metadata = (
                    True
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


def _declarations(
    spec, symbols, interfaces, associations, *, coverage=None, unit=None, owner=()
):
    for stmt in getattr(spec, "content", ()):
        if kind(stmt) in {"Implicit_Part", "Declaration_Construct"}:
            _declarations(
                stmt,
                symbols,
                interfaces,
                associations,
                coverage=coverage,
                unit=unit,
                owner=owner,
            )
            continue
        record = {
            "kind": kind(stmt),
            "role": "specification",
            "owner": list(owner),
            "location": unit.location(stmt) if unit else {},
            "status": "modeled",
        }
        try:
            _declaration(stmt, symbols, interfaces, associations)
        except Gap as error:
            if coverage is None:
                raise
            record.update(
                status="incomplete",
                diagnostic={
                    "code": "fortran." + error.code,
                    "detail": str(error),
                    **unit.location(error.node or stmt),
                    "owner": list(owner),
                },
            )
        if coverage is not None:
            coverage.append(record)


def _declaration(stmt, symbols, interfaces, associations):
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
    if stmt is not None:
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


def _source_coverage(tree, unit, tables, procedures):
    """Account for source units and specifications without inventing executable bodies."""
    records = []
    captured = {id(p.header): p for p in procedures}
    wrappers = {
        "Program",
        "External_Subprogram",
        "Module_Subprogram_Part",
        "Internal_Subprogram_Part",
    }
    units = _PROGRAMS | {"Module", "Submodule", "Block_Data"}

    def record(node, owner, role, status="modeled"):
        row = {
            "kind": kind(node),
            "role": role,
            "owner": list(owner),
            "location": unit.location(node),
            "status": status,
        }
        records.append(row)
        return row

    def incomplete(row, code, detail, node):
        row.update(
            status="incomplete",
            diagnostic={
                "code": "fortran." + code,
                "detail": detail,
                **unit.location(node),
                "owner": row["owner"],
            },
        )

    def specifications(spec, owner, symbols, *, module=False, blocked=False):
        for stmt in getattr(spec, "content", ()):
            tag = kind(stmt)
            if tag in {"Implicit_Part", "Declaration_Construct"}:
                specifications(stmt, owner, symbols, module=module, blocked=blocked)
                continue
            row = record(stmt, owner, "specification")
            if blocked:
                incomplete(
                    row,
                    "program_unit_scope",
                    "declaration belongs to an unclosed program unit",
                    stmt,
                )
                row["covers_subtree"] = True
                continue
            if module and tag == "Access_Stmt":
                row["facts"] = {
                    "visibility": str(stmt.items[0]).lower(),
                    "names": [
                        str(n).lower() for n in getattr(stmt.items[1], "items", ())
                    ],
                }
                continue
            try:
                _declaration(stmt, symbols, {}, [])
            except Gap as error:
                incomplete(row, error.code, str(error), error.node or stmt)
                row["covers_subtree"] = True
            if module:
                names = (
                    [str(entity.items[0]).lower() for entity in stmt.items[2].items]
                    if tag == "Type_Declaration_Stmt"
                    else []
                )
                meanings = [(name, symbols[name]) for name in names if name in symbols]
                missing = [name for name in names if name not in symbols]
                if missing:
                    incomplete(
                        row,
                        "module_declaration_binding",
                        "missing GNU module symbol facts: " + ", ".join(missing),
                        stmt,
                    )
                elif tag in {"Common_Stmt", "Equivalence_Stmt", "Data_Stmt"}:
                    incomplete(
                        row,
                        "module_storage",
                        "module storage association is not modeled: " + tag,
                        stmt,
                    )
                elif any(sym.source_variable for _, sym in meanings):
                    incomplete(
                        row,
                        "module_state",
                        "persistent module storage is not modeled: "
                        + ", ".join(
                            name for name, sym in meanings if sym.source_variable
                        ),
                        stmt,
                    )
                elif meanings and row["status"] == "modeled":
                    row["facts"] = {
                        "symbols": [
                            {
                                "name": name,
                                "category": sym.category,
                                "kind_bytes": sym.size,
                                "attributes": sorted(sym.attrs),
                                "folded_value": sym.value,
                            }
                            for name, sym in meanings
                        ],
                        "basis": "selected_gnu_resolved_symbols",
                    }

    def visit(node, parents=(), blocked=False):
        tag = kind(node)
        if tag in wrappers:
            for child in getattr(node, "content", ()):
                visit(child, parents, blocked)
            return
        if tag not in units:
            if tag in {"Comment", "Contains_Stmt"}:
                return
            row = record(node, parents, "program_unit", "incomplete")
            incomplete(row, "program_unit", "unhandled program unit " + tag, node)
            row["covers_subtree"] = True
            return
        head = node.content[0]
        name = (
            str(head.items[0]).lower()
            if tag == "Block_Data" and head.items[0] is not None
            else "scar_unnamed_common_init" if tag == "Block_Data" else _unit_name(node)
        )
        owner = (*parents, name)
        row = record(head, owner, "program_unit")
        row["kind"] = tag
        procedure = captured.get(id(head))
        if procedure is not None:
            row["symbol"] = procedure.symbol
            records.extend(procedure.source_coverage)
            if procedure.problem:
                row.update(
                    status="incomplete",
                    diagnostic={**procedure.problem, "owner": list(owner)},
                )
                if not procedure.source_coverage:
                    row["covers_subtree"] = True
                    spec = next(
                        (
                            child
                            for child in node.content
                            if kind(child) == "Specification_Part"
                        ),
                        None,
                    )
                    specifications(spec, owner, {}, blocked=True)
        elif tag == "Module" and not blocked:
            row["status"] = "transparent"
            symbols = {
                name: replace(sym, bounds=[])
                for name, sym in tables.get(owner, {}).items()
            }
            spec = next(
                (
                    child
                    for child in node.content
                    if kind(child) == "Specification_Part"
                ),
                None,
            )
            before = len(records)
            specifications(spec, owner, symbols, module=True)
            state = [name for name, sym in symbols.items() if sym.source_variable]
            if state:
                incomplete(
                    row,
                    "module_state",
                    "persistent or associated module storage is not modeled: "
                    + ", ".join(sorted(state)),
                    head,
                )
            elif any(r["status"] == "incomplete" for r in records[before:]):
                row.update(
                    status="incomplete",
                    diagnostic=next(
                        r["diagnostic"]
                        for r in records[before:]
                        if r["status"] == "incomplete"
                    ),
                )
        else:
            incomplete(row, "program_unit", "unclosed program unit " + tag, head)
            row["covers_subtree"] = True
            blocked = True
            spec = next(
                (
                    child
                    for child in node.content
                    if kind(child) == "Specification_Part"
                ),
                None,
            )
            specifications(spec, owner, {}, blocked=True)
        for child in node.content:
            if kind(child) in wrappers:
                visit(child, owner, blocked)

    visit(tree)
    return {
        "semantics_version": 1,
        "language": "fortran",
        "scope": "program_units_and_specification_constructs",
        "complete": all(row["status"] != "incomplete" for row in records),
        "records": records,
    }


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


def _io_filenames(tree):
    """Recover the filename values that GNU actually places in I/O descriptors."""
    locations = {}
    for match in re.finditer(
        rb"^\s*(\w+\.\d+)\.common\.filename = (.+);\n"
        rb"\s*\1\.common\.line = (\d+);$",
        tree,
        re.M,
    ):
        name = re.fullmatch(rb'&"([^"\\]*)"\[1\]\{[^\n]*\}', match[2])
        value = os.fsdecode(name[1]) if name else None
        line = int(match[3])
        if line in locations and locations[line] != value:
            value = None
        locations[line] = value
    return locations


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


_TARGET_MODE_OPTIONS = ("-m16", "-m32", "-m64", "-mx32")


def _source_target_mode(rows, compiler, stdout_sha256):
    options = {
        option: (
            rows[option] == "[enabled]"
            if rows.get(option) in {"[enabled]", "[disabled]"}
            else None
        )
        for option in _TARGET_MODE_OPTIONS
    }
    result = {
        "semantics_version": 1,
        "compiler_target": compiler["target"],
        "options": options,
        "multilib_abi": None,
        "instruction_bits": None,
        "pointer_bits": None,
        "evidence": {
            "kind": "derived",
            "rule": "gnu_x86_multilib_abi_name",
            "compiler_frontend_sha256": compiler["frontend"]["sha256"],
            "option_query_stdout_sha256": stdout_sha256,
            "definition": "https://github.com/gcc-mirror/gcc/blob/releases/gcc-13.3.0/gcc/config/i386/i386.cc#L17398-L17406",
        },
    }
    architecture = compiler["target"].split("-", 1)[0]
    if not re.fullmatch(r"x86_64|i[3-6]86", architecture):
        result["unresolved"] = "unsupported_compiler_target_family"
    elif any(value is None for value in options.values()):
        result["unresolved"] = "target_mode_options_not_explicit"
    else:
        # GNU's x86 target hook tests ISA_64BIT, then ABI_X32. The -m32
        # query is an inverse ISA_64BIT mask, so it is also true under -m16.
        m16, m32, m64, mx32 = (options[option] for option in _TARGET_MODE_OPTIONS)
        mode = (
            ("x86_64", 64, 64)
            if m64 and not (m16 or m32 or mx32)
            else (
                ("x32", 64, 32)
                if mx32 and not (m16 or m32 or m64)
                else (
                    ("i386", 16 if m16 else 32, 32)
                    if m32 and not (m64 or mx32)
                    else None
                )
            )
        )
        if mode is None:
            result["unresolved"] = "inconsistent_target_mode_options"
        else:
            result.update(
                zip(("multilib_abi", "instruction_bits", "pointer_bits"), mode)
            )
    return result


def _target_condition(selector, target_mode):
    condition = {"selector": selector, "evaluation": "unresolved"}
    if selector not in {"x86_64", "x32", "i386"}:
        condition["reason"] = "unsupported_gnu_multilib_selector"
    elif target_mode["multilib_abi"] is None:
        condition["reason"] = target_mode["unresolved"]
    else:
        evidence = target_mode["evidence"]
        condition.update(
            evaluation="derived",
            value=selector == target_mode["multilib_abi"],
            evidence={
                "source_contract_field": "target_mode",
                **{
                    key: evidence[key]
                    for key in (
                        "rule",
                        "compiler_frontend_sha256",
                        "option_query_stdout_sha256",
                    )
                },
            },
        )
    return condition


@lru_cache(maxsize=16)
def _compiler_receipt(driver, frontend, identities):
    # File identities invalidate the cache if either GNU executable is replaced.
    def binary(path):
        content = Path(path).read_bytes()
        return {
            "path": path,
            "sha256": sha256(content).hexdigest(),
            "size": len(content),
        }

    def query(option):
        return subprocess.run(
            [driver, option], check=True, capture_output=True, text=True, timeout=30
        ).stdout.strip()

    return {
        "provider": "gnu_fortran",
        "version": query("-dumpfullversion"),
        "target": query("-dumpmachine"),
        "driver": binary(driver),
        "frontend": binary(frontend),
    }


def _dependency_inputs(text):
    text = text.replace("\\\n", "")
    if not re.match(r"scar-fortran-source(?=[:\s])", text):
        raise ValueError("unexpected GNU dependency target")
    index = len("scar-fortran-source")
    while index < len(text) and text[index] != ":":
        index += 2 if text[index] == "\\" else 1
    if index == len(text):
        raise ValueError("missing GNU dependency separator")
    words, word, index = [], [], index + 1
    while index < len(text):
        char = text[index]
        index += 1
        if char.isspace():
            if word:
                words.append("".join(word))
                word = []
        elif char == "\\" and index < len(text):
            word.append(text[index])
            index += 1
        elif char == "$" and text[index : index + 1] == "$":
            word.append("$")
            index += 1
        elif char in "$#:":
            raise ValueError("unsupported GNU dependency escaping")
        else:
            word.append(char)
    if word:
        words.append("".join(word))
    return words


def _math_interfaces(content, path, revision):
    records, unresolved = [], []
    grammar = re.compile(
        r"^\s*!gcc\$\s+builtin\s*\(\s*([a-z_]\w*)\s*\)\s+"
        r"attributes\s+simd\s*\(\s*(inbranch|notinbranch)\s*\)\s*"
        r"(?:if\s*\(\s*(['\"])([^'\"]+)\3\s*\))?\s*$",
        re.I,
    )
    for line, source in enumerate(content.decode(errors="replace").splitlines(), 1):
        if not re.match(r"^\s*!gcc\$", source, re.I):
            continue
        evidence = {"file": path, "sha256": revision, "line": line}
        match = grammar.fullmatch(source)
        if match is None:
            unresolved.append({"reason": "unsupported_builtin_directive", **evidence})
            continue
        records.append(
            {
                "semantics_version": 1,
                "interface": match[1].lower(),
                "declaration": "gnu_builtin",
                "attributes": {"simd": {"branch": match[2].lower()}},
                "target_condition": (
                    {"selector": match[4], "evaluation": "unresolved"}
                    if match[4]
                    else None
                ),
                "evidence": {"provider": "gnu_semantic_preinclude", **evidence},
                "execution": "not_observed",
            }
        )
    return records, unresolved


_OPEN_TRACE = (
    "-ff",
    "-y",
    "-xx",
    "-v",
    "-s",
    "0",
    "-e",
    "trace=%process,open,openat,openat2,close,dup2,dup3,fstat,newfstatat,getdents,getdents64,"
    "io_uring_setup,open_by_handle_at,recvmsg,pidfd_getfd",
)
_HEX_PATH = r"(?:\\x[0-9a-f]{2})+"


def _trace_path(encoded):
    return os.fsdecode(bytes.fromhex(encoded.replace("\\x", "")))


def _compiler_read_inventory(prefix, compiler, frontend):
    """Accept a directly observed driver → f951 pair and resolved read descriptors."""
    programs, inputs, issues = {}, {}, []
    witness = {
        "provider": "linux_strace_gnu_open_v1",
        "scope": "regular_files_opened_for_reading",
        "processes": {},
    }
    for trace in prefix.parent.glob(prefix.name + ".*"):
        data = trace.read_bytes()
        lines = data.decode("ascii").splitlines()
        execs = [line for line in lines if line.startswith(("execve(", "execveat("))]
        launch = (
            re.match(rf'^execve\("({_HEX_PATH})",.*\)\s+= 0$', execs[0])
            if execs
            else None
        )
        if len(execs) != 1 or not launch or lines[-1:] != ["+++ exited with 0 +++"]:
            raise ValueError("incomplete or indirect GNU process trace")
        executable = Path(_trace_path(launch[1]))
        if not executable.is_absolute():
            raise ValueError("relative compiler executable in trace")
        program = executable.resolve()
        role = (
            "frontend"
            if program == frontend
            else "driver" if program == compiler else None
        )
        if not role or role in programs:
            raise ValueError("ambiguous GNU process chain")
        pid = int(trace.suffix[1:])
        programs[role] = (pid, lines)
        witness["processes"][role] = {
            "pid": pid,
            "path": str(program),
            "trace_sha256": digest(data),
        }
    if programs.keys() != {"driver", "frontend"}:
        raise ValueError("GNU compiler trace is incomplete")
    frontend_pid = programs["frontend"][0]
    if not any(
        re.match(rf"^(?:vfork|fork|clone3?)\(.*\)\s+= {frontend_pid}$", line)
        for line in programs["driver"][1]
    ):
        raise ValueError("GNU frontend parent was not witnessed")
    identities, directories = {}, set()
    for role, (_, lines) in sorted(programs.items()):
        pending = {}
        for line in lines:
            if " = -1 " in line:
                continue
            syscall = line.partition("(")[0]
            if syscall in {"open", "openat", "openat2"}:
                descriptor = re.search(rf"\)\s+= (\d+)<({_HEX_PATH})>$", line)
                if not descriptor:
                    raise ValueError("unresolved open file descriptor")
                fd, path = descriptor[1], _trace_path(descriptor[2])
                if fd in pending:
                    issues.append(pending.pop(fd))
                if re.search(r"\bO_WRONLY\b", line):
                    continue
                if not os.path.isabs(path) or not re.search(
                    r"\bO_(?:RDONLY|RDWR)\b", line
                ):
                    raise ValueError("uncertain compiler input access mode or path")
                inputs[path] = "gnu_" + role + "_read"
                pending[fd] = path
            elif syscall == "close":
                closed = re.match(r"^close\((\d+)[<)]", line)
                if closed and closed[1] in pending:
                    issues.append(pending.pop(closed[1]))
            elif syscall in {"dup2", "dup3"} and pending:
                raise ValueError(
                    "descriptor replacement before input identity was witnessed"
                )
            elif syscall in {"fstat", "newfstatat"}:
                fd_stat = re.match(
                    rf'^(?:fstat|newfstatat)\((\d+)<({_HEX_PATH})>, (?:"", )?'
                    r"\{st_dev=makedev\((0x[0-9a-f]+|\d+), (0x[0-9a-f]+|\d+)\), "
                    r"st_ino=(\d+), st_mode=(S_IFREG|S_IFDIR)\|",
                    line,
                )
                if fd_stat and re.search(r"\)\s+= 0$", line):
                    fd, path = fd_stat[1], _trace_path(fd_stat[2])
                    if pending.get(fd) == path:
                        identity = (
                            os.makedev(int(fd_stat[3], 0), int(fd_stat[4], 0)),
                            int(fd_stat[5]),
                        )
                        if identities.setdefault(path, identity) != identity:
                            raise ValueError(
                                "compiler opened multiple identities at one path"
                            )
                        if fd_stat[6] == "S_IFDIR":
                            directories.add(path)
                        pending.pop(fd)
            elif syscall in {
                "getdents",
                "getdents64",
                "io_uring_setup",
                "open_by_handle_at",
                "recvmsg",
                "pidfd_getfd",
            }:
                raise ValueError("compiler input bypassed ordinary file opens")
        issues.extend(pending.values())
    for path in directories:
        del inputs[path]
    witness["opened_files"] = [
        {"path": path, "device_inode": identities.get(path)} for path in sorted(inputs)
    ]
    return inputs, identities, witness, sorted(set(issues))


def _semantic_inputs(capture, root, frontend):
    cwd = Path(capture["working_directory"])
    invocations = []
    for line in capture["semantic_stderr"].decode(errors="replace").splitlines():
        try:
            words = shlex.split(line)
        except ValueError:
            continue
        if (
            words
            and Path(words[0]).is_absolute()
            and Path(words[0]).resolve() == frontend
        ):
            invocations.append(words)
    inputs, unresolved, records = {}, [], []
    if len(invocations) != 1:
        unresolved.append({"reason": "unique_frontend_invocation_unavailable"})
    else:
        args = iter(invocations[0][1:])
        for arg in args:
            if arg == "-fpre-include":
                selected = next(args, "")
            elif arg.startswith("-fpre-include="):
                selected = arg.partition("=")[2]
            else:
                continue
            if selected:
                inputs[str((cwd / selected).resolve())] = "gnu_semantic_preinclude"
    source_coverage = "unresolved_without_preprocessor_dependency_inventory"
    identities, open_witness = {}, None
    if capture.get("dependencies") is not None:
        source_coverage = "unresolved_dependency_inventory"
        try:
            for selected in _dependency_inputs(capture["dependencies"]):
                inputs.setdefault(
                    str((cwd / selected).resolve()), "gnu_semantic_dependency"
                )
            source_coverage = "same_semantic_command_dependency_inventory"
        except ValueError as error:
            unresolved.append(
                {"reason": "dependency_inventory_unavailable", "detail": str(error)}
            )
    elif prefix := capture.get("open_trace"):
        source_coverage = "incomplete_frontend_read_inventory"
        try:
            files, identities, open_witness, missing = _compiler_read_inventory(
                prefix, Path(capture["compiler"]).resolve(), frontend
            )
            main_file = str((cwd / capture["semantic_command"][-1]).resolve())
            if not {main_file, *inputs}.issubset(files):
                raise ValueError(
                    "expected semantic source or preinclude open is missing"
                )
            for path in sorted(files):
                inputs.setdefault(path, files[path])
            if not missing:
                source_coverage = "same_frontend_read_inventory"
            unresolved.extend(
                {"reason": "compiler_open_identity_unobserved", "path": path}
                for path in missing
            )
        except (OSError, ValueError) as error:
            unresolved.append(
                {"reason": "compiler_open_inventory_unavailable", "detail": str(error)}
            )
    headers, sources, interfaces = {}, {}, []
    for selected, evidence in inputs.items():
        path = (root / selected).resolve()
        try:
            if not stat.S_ISREG(path.stat().st_mode):
                raise ValueError("compiler input is not a regular file")
            with path.open("rb") as stream:
                before = os.fstat(stream.fileno())
                if identities.get(selected, (before.st_dev, before.st_ino)) != (
                    before.st_dev,
                    before.st_ino,
                ):
                    raise ValueError("input differs from observed file identity")
                if (
                    max(before.st_mtime_ns, before.st_ctime_ns)
                    > capture["semantic_started_ns"]
                ):
                    raise ValueError(
                        "selected input modified during or after semantic capture"
                    )
                content = stream.read()
                after = os.fstat(stream.fileno())
            stamp = lambda value: (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )
            if not stamp(before) == stamp(after) == stamp(path.stat()):
                raise ValueError("selected input changed while fingerprinting")
        except (OSError, ValueError) as error:
            unresolved.append(
                {
                    "reason": "selected_input_snapshot_unavailable",
                    "path": str(path),
                    "detail": str(error),
                }
            )
            continue
        revision = sha256(content).hexdigest()
        receipt = {"sha256": revision, "size": len(content), "evidence": evidence}
        records.append({"path": str(path), **receipt})
        if path.is_relative_to(root):
            sources[path.relative_to(root).as_posix()] = revision
        elif evidence not in {"gnu_driver_read", "gnu_frontend_read"}:
            headers[str(path)] = receipt
        if evidence == "gnu_semantic_preinclude":
            declarations, missing = _math_interfaces(content, str(path), revision)
            interfaces.extend(declarations)
            unresolved.extend(missing)
    return {
        "frontend_argv": invocations[0] if len(invocations) == 1 else None,
        "files": records,
        "external_headers": headers,
        "source_digests": sources,
        "source_math_interfaces": interfaces,
        "open_witness": open_witness,
        "unresolved": unresolved,
        "coverage": {
            "automatic_preincludes": "selected_frontend_arguments",
            "source_includes": source_coverage,
            "snapshot": "stable_read_after_successful_semantic_capture",
        },
    }


def _numeric_contract(path, root, flags, capture, source_sha256):
    compiler = capture["compiler"]
    cwd = Path(capture["working_directory"])
    selected = _logical_flags(flags)
    frontend_query = subprocess.run(
        [compiler, *selected, "-print-prog-name=f951"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    frontend = Path(frontend_query.stdout.strip()).resolve()
    driver = str(Path(compiler).resolve())
    identities = tuple(
        (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        for stat in (Path(driver).stat(), frontend.stat())
    )
    identity = _compiler_receipt(driver, str(frontend), identities)
    inputs = _semantic_inputs(capture, root, frontend)
    # --help exits before parsing: this query observes f951 option selection.
    # The separate semantic command validated the same real source and flags.
    command = [
        compiler,
        *selected,
        "-fsyntax-only",
        "-Q",
        "--help=optimizers",
        "--help=fortran",
        "--help=target",
        capture["semantic_command"][-1],
    ]
    result = subprocess.run(command, cwd=cwd, capture_output=True, timeout=30)
    if result.returncode:
        raise Gap("numeric_policy", "GNU source-context option query failed")
    rows, policy, unresolved = numeric_option_rows(
        result, fortran=True, extra_rows=_TARGET_MODE_OPTIONS
    )
    target_mode = _source_target_mode(rows, identity, sha256(result.stdout).hexdigest())
    for interface in inputs["source_math_interfaces"]:
        if condition := interface["target_condition"]:
            interface["target_condition"] = _target_condition(
                condition["selector"], target_mode
            )
    contract = {
        "semantics_version": 1,
        "scope": "compilation_unit",
        "provider": "gnu_fortran",
        "compiler": identity,
        "source": {"path": str(path), "sha256": source_sha256},
        "source_flags": list(flags),
        "validated_command": capture["semantic_command"],
        "semantic_capture": {
            "cwd": str(cwd),
            "returncode": capture["semantic_returncode"],
            "stdout_sha256": capture["semantic_stdout_sha256"],
            "stderr_sha256": sha256(capture["semantic_stderr"]).hexdigest(),
            "dependency_sha256": capture.get("dependency_sha256"),
            "started_ns": capture["semantic_started_ns"],
            "finished_ns": capture["semantic_finished_ns"],
            "frontend_argv": inputs["frontend_argv"],
            "inputs": inputs["files"],
            "coverage": inputs["coverage"],
            "unresolved": inputs["unresolved"],
            **(
                {"open_witness": inputs["open_witness"]}
                if inputs["open_witness"]
                else {}
            ),
        },
        "source_math_interfaces": inputs["source_math_interfaces"],
        "target_mode": target_mode,
        "option_query": {
            "command": command,
            "cwd": str(root),
            "returncode": result.returncode,
            "stdout_sha256": sha256(result.stdout).hexdigest(),
            "stderr_sha256": sha256(result.stderr).hexdigest(),
            "selected_rows": rows,
            "evidence": "observed_frontend_option_selection_not_source_execution",
        },
        "policy": policy,
        "unresolved": unresolved,
        "applicability": {
            "evidence": "inferred_same_selected_frontend_and_flags_as_validated_source",
            "source_contents": "raw_file_digest",
            "include_contents": "selected_semantic_inputs_fingerprinted_with_coverage",
            "warning": "The option query exits before parsing; source acceptance and selected input evidence come from the separate semantic capture.",
        },
        "caller_observations": {"errno": "unproven", "fenv": "unproven"},
        "runtime_fp_environment": "unobserved",
    }
    return contract, inputs["external_headers"], inputs["source_digests"]


def _intrinsic_bindings(dump):
    scopes = list(_SCOPE.finditer(dump))
    result, ancestors = {}, []
    for i, scope in enumerate(scopes):
        depth = len(scope[1].expandtabs())
        while ancestors and ancestors[-1][0] >= depth:
            ancestors.pop()
        ancestors.append((depth, scope[2].lower()))
        section = dump[
            scope.end() : scopes[i + 1].start() if i + 1 < len(scopes) else None
        ]
        code = re.split(r"^[ \t]*code:[ \t]*$", section, maxsplit=1, flags=re.M)
        result[tuple(name for _, name in ancestors)] = tuple(
            sorted(
                set(
                    re.findall(
                        r"\b(__[a-zA-Z0-9_]+)\[\[", code[1] if len(code) == 2 else ""
                    )
                )
            )
        )
    return result


def _numeric_semantics_sha256(contract, temporary):
    """Identify the selected semantics independently of this successful capture."""
    capture = contract["semantic_capture"]
    if (
        capture["unresolved"]
        or not capture["frontend_argv"]
        or capture["coverage"]["source_includes"]
        not in {
            "same_semantic_command_dependency_inventory",
            "same_frontend_read_inventory",
        }
    ):
        return None

    def arguments(argv, *, frontend=False, compatibility=False):
        values = list(argv)
        for index, arg in enumerate(argv):
            if index and argv[index - 1] == "-J" and arg == str(temporary):
                values[index] = {"capture_output": "module_directory"}
            elif arg == "-fdump-tree-original=" + str(temporary / "descriptor.tree"):
                values[index] = {"capture_output": "gnu_io_descriptors"}
            elif (
                index
                and argv[index - 1] == "-MF"
                and arg == str(temporary / "source.d")
            ):
                values[index] = {"capture_output": "dependency_file"}
            elif frontend and arg.startswith("-cpp="):
                generated = Path(arg.removeprefix("-cpp="))
                if generated.parent == temporary:
                    values[index] = {"capture_output": "gnu_preprocessor_output"}
            elif (
                compatibility
                and index == len(argv) - 1
                and arg
                in {
                    str(temporary / "syntax.f"),
                    str(temporary / "syntax.f90"),
                }
            ):
                values[index] = {"capture_input": Path(arg).name}
        return values

    # Verbose diagnostics contain GCC-generated filenames. The actual inputs,
    # selected frontend/target/options, semantic dump and status remain in the key.
    selected = dict(contract)
    selected["validated_command"] = arguments(contract["validated_command"])
    selected["semantic_capture"] = {
        key: value
        for key, value in capture.items()
        if key not in {"started_ns", "finished_ns", "stderr_sha256", "open_witness"}
    }
    selected["semantic_capture"]["frontend_argv"] = arguments(
        capture["frontend_argv"], frontend=True
    )
    if compatible := contract.get("syntax_compatibility"):
        selected["syntax_compatibility"] = {
            key: value for key, value in compatible.items() if key != "stderr_sha256"
        }
        selected["syntax_compatibility"]["command"] = arguments(
            compatible["command"], compatibility=True
        )
    return digest(json_bytes(selected))


def _compile_capture(path, cwd, flags, tmp, operand):
    compiler = shutil.which("gfortran")
    if compiler is None:
        raise Gap("compiler", "gfortran is required")
    flags = _logical_flags(flags)

    def capture(args, receipt=None, *, observe_reads=False):
        command = [compiler, *flags, *args]
        started = time_ns()
        run_args = dict(
            cwd=cwd,
            env={**os.environ, "TMPDIR": str(tmp)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )
        observer = shutil.which("strace") if observe_reads else None
        trace = tmp / "semantic-opens"
        if observer:
            try:
                p = subprocess.run(
                    [observer, *_OPEN_TRACE, "-o", str(trace), *command], **run_args
                )
            except (FileNotFoundError, PermissionError):
                observer = None
            else:
                # A denied or unsupported tracer never starts the compiler.
                if p.returncode and not any(
                    f.stat().st_size for f in tmp.glob(trace.name + ".*")
                ):
                    observer = None
        if not observer:
            p = subprocess.run(command, **run_args)
        if p.returncode:
            raise Gap("compiler", p.stderr.decode(errors="replace").strip()[:3000])
        if receipt is not None:
            receipt.update(
                semantic_command=command,
                semantic_returncode=p.returncode,
                semantic_stdout_sha256=sha256(p.stdout).hexdigest(),
                semantic_stderr=p.stderr,
                semantic_started_ns=started,
                semantic_finished_ns=time_ns(),
            )
            if observer:
                receipt["open_trace"] = trace
        return p.stdout.decode("utf-8", errors="replace")

    fixed = path.suffix.lower() in {".for", ".f", ".ftn", ".f77", ".fpp"}
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
        (
            language.endswith("-cpp-input")
            if language not in {None, "none"}
            else path.suffix
            in {".F", ".FOR", ".fpp", ".FPP", ".F90", ".F95", ".F03", ".F08"}
        ),
    )
    io_dump = tmp / "descriptor.tree"
    semantic_flags = [
        "-J",
        str(tmp),
        "-v",
        "-fsyntax-only",
        "-fdump-fortran-original",
        "-fdump-tree-original=" + str(io_dump),
    ]
    dependencies = tmp / "source.d"
    dependency_flags = (
        ["-MD", "-MF", str(dependencies), "-MT", "scar-fortran-source"]
        if preprocess
        else []
    )
    receipt = {"compiler": compiler, "working_directory": str(cwd)}
    dump = capture(
        [*semantic_flags, *dependency_flags, operand],
        receipt,
        observe_reads=not preprocess,
    )
    receipt["io_filenames"] = (
        _io_filenames(io_dump.read_bytes()) if io_dump.is_file() else {}
    )
    dependency_bytes = dependencies.read_bytes() if dependencies.exists() else None
    receipt["dependencies"] = (
        dependency_bytes.decode("utf-8", errors="surrogateescape")
        if dependency_bytes is not None
        else None
    )
    receipt["dependency_sha256"] = (
        sha256(dependency_bytes).hexdigest() if dependency_bytes is not None else None
    )
    source = (
        capture(["-E", operand])
        if preprocess
        else path.read_text(encoding="utf-8", errors="surrogateescape")
    )
    text, locations = _selected_text(source, path)
    locations = [
        (str(path) if file == operand else file, line) for file, line in locations
    ]
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
    return (
        text,
        locations,
        fixed,
        dump,
        capture,
        semantic_flags,
        receipt,
    )


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


@dataclass(frozen=True)
class Unavailable:
    code: str
    detail: str


def required(value):
    if isinstance(value, Unavailable):
        raise Gap(value.code, value.detail)
    return value


def _probe(function, *args):
    try:
        return function(*args)
    except (Gap, OSError, subprocess.SubprocessError) as error:
        return Unavailable(getattr(error, "code", "toolchain"), str(error))


@dataclass(frozen=True)
class TargetABI:
    storage: tuple | Unavailable
    transfer: IOProfile | Unavailable | None
    files: dict

    @property
    def size_type(self):
        return required(self.storage)[0]

    @property
    def byte_order(self):
        return required(self.storage)[1]

    def data_transfer(self):
        if self.transfer is None:
            raise Gap("io_abi", "no data transfer in captured unit")
        return required(self.transfer)

    def file_control(self, operation, fields):
        profile = self.files.get((operation, fields))
        if profile is None:
            raise Gap(
                "io_abi", f"uncaptured {operation.upper()} control fields {fields}"
            )
        return required(profile)


@dataclass(frozen=True)
class Procedure:
    symbol: str
    name: str
    header: object
    symbols: dict
    links: dict
    common_blocks: dict
    formals: tuple
    result: str | None
    interfaces: dict
    storage: tuple
    statements: tuple
    formats: dict
    calls: tuple
    main: bool
    block_data: bool
    problem: dict | None
    intrinsic_bindings: tuple = ()
    source_coverage: tuple = ()

    def external_name(self, name):
        if name not in self.links:
            raise Gap("binding", f"uncaptured external procedure {name}", self.header)
        return self.links[name]


@dataclass(frozen=True)
class ParsedUnit:
    path: Path
    flags: tuple = ()
    defaults: tuple = ()
    locations: tuple = ()
    procedures: tuple = ()
    target: TargetABI | None = None
    diagnostics: tuple = ()
    compiler_id: str | None = None
    source_digests: dict = field(default_factory=dict)
    source_numeric_contract: dict = field(default_factory=dict)
    numeric_semantics_sha256: str | None = None
    intrinsic_bindings: dict = field(default_factory=dict)
    external_headers: dict = field(default_factory=dict)
    source_coverage: dict = field(default_factory=dict)
    io_filenames: dict = field(default_factory=dict)

    def location(self, node):
        original = node
        while node is not None:
            item = getattr(node, "item", None)
            if getattr(item, "span", None):
                at = item.span[0] - 1
                if 0 <= at < len(self.locations):
                    file, line = self.locations[at]
                    return {"file": file, "line": line}
            node = getattr(node, "parent", None)
        node = original
        while node is not None:
            children = getattr(node, "content", ())
            node = children[0] if children else None
            item = getattr(node, "item", None)
            if getattr(item, "span", None):
                at = item.span[0] - 1
                if 0 <= at < len(self.locations):
                    file, line = self.locations[at]
                    return {"file": file, "line": line}
        return {"file": str(self.path), "line": 1}


def _capture_unit(path, root, flags, cwd, operand):
    from fparser.two.utils import FortranSyntaxError, walk
    from fparser.two import Fortran2003 as F

    default_int = "i64" if "-fdefault-integer-8" in flags else "i32"
    default_real = "f64" if "-fdefault-real-8" in flags else "f32"
    double = (
        "f128" if default_real == "f64" and "-fdefault-double-8" not in flags else "f64"
    )
    defaults = (default_int, default_real, double)
    source_sha256 = sha256(path.read_bytes()).hexdigest()
    with TemporaryDirectory(prefix="scar-fortran-") as temporary:
        tmp = Path(temporary)
        text, locations, fixed, dump, capture, semantic_flags, commands = (
            _compile_capture(path, cwd, flags, tmp, operand)
        )
        numeric, external_headers, included_sources = _numeric_contract(
            path, root, flags, commands, source_sha256
        )
        try:
            tree = _syntax(text, fixed, cwd, path, flags)
        except FortranSyntaxError as exc:
            compatible = _TOKEN.sub(
                lambda m: m[0][:-1] + " " if m["unary"] else m[0], text
            )
            if compatible == text:
                raise Gap("syntax", str(exc)) from exc
            candidate = tmp / ("syntax.f" if fixed else "syntax.f90")
            candidate.write_text(compatible)
            compatibility_receipt = {}
            checked = capture(
                [
                    *semantic_flags,
                    "-x",
                    "f77" if fixed else "f95",
                    "-nocpp",
                    "-I",
                    str(path.parent),
                    str(candidate),
                ],
                compatibility_receipt,
            )
            if _without_builtin_plus(checked) != _without_builtin_plus(dump):
                raise Gap("syntax", str(exc)) from exc
            numeric["syntax_compatibility"] = {
                "command": compatibility_receipt["semantic_command"],
                "source_sha256": sha256(candidate.read_bytes()).hexdigest(),
                "stdout_sha256": compatibility_receipt["semantic_stdout_sha256"],
                "stderr_sha256": sha256(
                    compatibility_receipt["semantic_stderr"]
                ).hexdigest(),
                "evidence": "gnu_dump_equivalent_ignoring_builtin_unary_plus",
                "returncode": compatibility_receipt["semantic_returncode"],
            }
            tree = _syntax(compatible, fixed, cwd, path, flags)
    if sha256(path.read_bytes()).hexdigest() != source_sha256:
        raise Gap("source_changed", "raw source changed during semantic capture")
    tables = _facts(dump)
    relative = (
        path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
    )
    identity = numeric["compiler"]
    unit = ParsedUnit(
        path,
        tuple(flags),
        defaults,
        tuple(locations),
        compiler_id=f"GNU Fortran {identity['version']} ({identity['target']})",
        source_digests={**included_sources, relative: source_sha256},
        source_numeric_contract=numeric,
        numeric_semantics_sha256=_numeric_semantics_sha256(numeric, tmp),
        intrinsic_bindings=_intrinsic_bindings(dump),
        external_headers=external_headers,
        io_filenames=commands["io_filenames"],
    )
    programs = list(_programs(tree))
    bindings = {}
    for parents, program in programs:
        name = _unit_name(program)
        own = tables.get((*parents, name), {}).get(name)
        bindings[(*parents, name)] = _linker_name(
            name, flags, parents[-1] if parents else None, own.binding if own else None
        )
    block_data = walk(tree, F.Block_Data)
    if any(block.content[0].items[0] is None for block in block_data):
        unnamed = [
            section
            for section in re.split(r"(?=^Namespace:)", dump, flags=re.M)
            if "symtree:" in section and not _SCOPE.search(section)
        ]
        if len(unnamed) == 1:
            tables.update(
                _facts("procedure name = scar_unnamed_common_init\n" + unnamed[0])
            )
    procedures = tuple(
        _capture_procedure(unit, parents, node, tables, bindings)
        for parents, node in chain((((), block) for block in block_data), programs)
    )
    return replace(
        unit,
        procedures=procedures,
        target=_capture_target(flags, tree),
        source_coverage=_source_coverage(tree, unit, tables, procedures),
    )


def _capture_procedure(unit, parents, node, tables, bindings):
    from fparser.two import Fortran2003 as F
    from fparser.two.utils import walk

    head = node.content[0]
    block = kind(node) == "Block_Data"
    main = kind(node) in {"Main_Program", "Main_Program0"}
    if block:
        name = (
            str(head.items[0]).lower()
            if head.items[0] is not None
            else "scar_unnamed_common_init"
        )
        symbol = name
    else:
        name = _unit_name(node)
        symbol = "main" if main else bindings[(*parents, name)]
    table = tables.get((*parents, name))
    symbols = {
        name: replace(meaning, bounds=[]) for name, meaning in (table or {}).items()
    }
    owner = symbols.get(name)
    result = (
        (owner.result or name)
        if kind(node) == "Function_Subprogram" and owner
        else None
    )
    formals = (
        tuple(str(arg).lower() for arg in head.items[2].items)
        if not block and not main and head.items[2]
        else ()
    )
    interfaces = {}
    associations = []
    coverage = []
    problem = None
    try:
        if table is None:
            raise Gap("scope", f"missing GNU symbol table for {name}", head)
        if not block and _enabled(unit.flags, "-ff2c", "-fno-f2c"):
            raise Gap("abi", "alternate return convention -ff2c", head)
        if parents:
            host = tables.get(parents, {})
            host_symbol = host.get(parents[-1])
            if len(parents) > 1 or (
                host and (host_symbol is None or "MODULE" not in host_symbol.attrs)
            ):
                raise Gap(
                    "host_association", "internal procedure with host scope", head
                )
            # GNU may fold a host PARAMETER out of the procedure's own table.
            # The source AST still refers to it; local symbols take precedence.
            for inherited, meaning in host.items():
                if "PARAMETER" in meaning.attrs:
                    symbols.setdefault(inherited, replace(meaning, bounds=[]))
        spec = next(
            (child for child in node.content if kind(child) == "Specification_Part"),
            None,
        )
        _declarations(
            spec,
            symbols,
            interfaces,
            associations,
            coverage=coverage,
            unit=unit,
            owner=(*parents, name),
        )
        problem = next(
            (
                record["diagnostic"]
                for record in coverage
                if record["status"] == "incomplete"
            ),
            None,
        )
    except Gap as error:
        problem = {
            "code": "fortran." + error.code,
            "detail": str(error),
            **unit.location(error.node or head),
        }
        if not block:
            problem["symbol"] = symbol

    if not block:
        for record in coverage:
            if "diagnostic" in record:
                record["diagnostic"]["symbol"] = symbol

    calls = set()
    formats = {}
    for child in walk(
        node, (F.Call_Stmt, F.Function_Reference, F.Part_Ref, F.Format_Stmt)
    ):
        if isinstance(child, F.Format_Stmt):
            formats[str(child.item.label)] = str(child.items[1])
        else:
            target = str(child.items[0]).lower()
            meaning = symbols.get(target)
            if isinstance(child, F.Call_Stmt) or (
                meaning
                and "PROCEDURE" in meaning.attrs
                and "INTRINSIC-PROC" not in meaning.attrs
            ):
                calls.add(target)
    links = {}
    for target in symbols.keys() | calls:
        meaning = symbols.get(target)
        links[target] = (
            (meaning.binding if meaning else None)
            or bindings.get((*parents, target))
            or bindings.get((target,))
            or _linker_name(target, unit.flags)
        )
    common_blocks = {}
    for association in associations:
        if kind(association) == "Common_Stmt":
            for group, _ in association.items[0]:
                if group is not None:
                    group_name = str(group).lower()
                    common_blocks[group_name] = _linker_name(group_name, unit.flags)
                else:
                    common_blocks[None] = "__BLNK__"
    statements = next(
        (child.content for child in node.content if kind(child) == "Execution_Part"), ()
    )
    return Procedure(
        symbol=symbol,
        name=name,
        header=head,
        symbols=symbols,
        links=links,
        common_blocks=common_blocks,
        formals=formals,
        result=result,
        interfaces=interfaces,
        storage=tuple(associations),
        statements=tuple(statements),
        formats=formats,
        calls=tuple(sorted(links[target] for target in calls)),
        main=main,
        block_data=block,
        problem=problem,
        intrinsic_bindings=unit.intrinsic_bindings.get((*parents, name), ()),
        source_coverage=tuple(coverage),
    )


def _capture_target(flags, tree):
    from fparser.two import Fortran2003 as F
    from fparser.two.utils import walk

    compiler = shutil.which("gfortran")
    abi_flags = tuple(
        flag for flag in flags if flag.startswith(("-m", "-fpack-derived"))
    )
    transfer = False
    files = {}
    for statement in walk(
        tree, (F.Print_Stmt, F.Write_Stmt, F.Stop_Stmt, F.Open_Stmt, F.Close_Stmt)
    ):
        if isinstance(statement, (F.Open_Stmt, F.Close_Stmt)):
            operation, controls = statement.items
            fields = {str(field.items[0] or "unit").lower() for field in controls.items}
            key = (operation.lower(), tuple(sorted(fields - {"unit"})))
            if key not in files:
                files[key] = _probe(_file_abi, compiler, abi_flags, *key)
        else:
            transfer = True
    return TargetABI(
        storage=_probe(_target_storage, compiler, abi_flags),
        transfer=_probe(_io_abi, compiler, abi_flags) if transfer else None,
        files=files,
    )


def parse_unit(
    path, flags=(), *, root, abi=None, working_directory=None, source_operand=None
):
    """Read the selected source and capture the compiler evidence needed by F2."""
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        cwd = Path(working_directory) if working_directory is not None else root
        operand = source_operand if source_operand is not None else str(path)
        if (
            (source_operand is None) != (working_directory is None)
            or not cwd.is_absolute()
            or (cwd / operand).resolve() != path
        ):
            raise Gap(
                "source_operand",
                "compiler source operand must resolve to the locked input in its working directory",
            )
        return _capture_unit(path, root, flags, cwd, operand)
    except (ImportError, Gap, subprocess.SubprocessError, OSError) as error:
        description = str(error)
        line = re.search(r"(?:at line |:(?=\d+:))(\d+)", description, re.I)
        issue = {
            "code": "fortran." + getattr(error, "code", "toolchain"),
            "detail": description,
            "file": str(path),
            "line": int(line[1]) if line else 1,
        }
        return ParsedUnit(path, tuple(flags), diagnostics=(issue,))

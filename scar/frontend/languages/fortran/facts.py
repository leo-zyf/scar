"""F1: capture GNU declarations, selected Fortran syntax and target ABI facts."""

from dataclasses import dataclass, field, replace
from functools import lru_cache
from hashlib import sha256
from itertools import chain
import json
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
from threading import Lock


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


_NUMERIC_OPTIONS = {
    "math_errno": "-fmath-errno",
    "rounding_math": "-frounding-math",
    "trapping_math": "-ftrapping-math",
    "signed_zeros": "-fsigned-zeros",
    "finite_only": "-ffinite-math-only",
    "fp_contract": "-ffp-contract",
    "unsafe_math": "-funsafe-math-optimizations",
    "associative_math": "-fassociative-math",
    "reciprocal_math": "-freciprocal-math",
    "signaling_nans": "-fsignaling-nans",
    "protect_parens": "-fprotect-parens",
}


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


def _numeric_contract(path, root, flags, capture, source_sha256):
    compiler = capture["compiler"]
    selected = _logical_flags(flags)
    frontend_query = subprocess.run(
        [compiler, *selected, "-print-prog-name=f951"],
        cwd=root,
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
    # --help exits before parsing: this query observes f951 option selection.
    # The separate semantic command validated the same real source and flags.
    command = [
        compiler,
        *selected,
        "-fsyntax-only",
        "-Q",
        "--help=optimizers",
        "--help=fortran",
        str(path),
    ]
    result = subprocess.run(command, cwd=root, capture_output=True, timeout=30)
    if result.returncode:
        raise Gap("numeric_policy", "GNU source-context option query failed")
    rows = {}
    for line in (result.stdout + result.stderr).decode(errors="replace").splitlines():
        match = re.match(r"^\s*(-f[\w-]+)(?:=\[[^]]+\])?\s+(.*?)\s*$", line)
        if match and match[1] in _NUMERIC_OPTIONS.values():
            rows[match[1]] = match[2]
    policy, unresolved = {}, []
    for name, option in _NUMERIC_OPTIONS.items():
        chosen = rows.get(option)
        if chosen in {"[enabled]", "[disabled]"}:
            policy[name] = chosen == "[enabled]"
        elif name == "fp_contract" and chosen in {"off", "on", "fast"}:
            policy[name] = chosen
        else:
            policy[name] = None
            unresolved.append(
                {"field": name, "reason": "option_query_has_no_explicit_value"}
            )
    return {
        "semantics_version": 1,
        "scope": "compilation_unit",
        "provider": "gnu_fortran",
        "compiler": identity,
        "source": {"path": str(path), "sha256": source_sha256},
        "source_flags": list(flags),
        "validated_command": capture["semantic_command"],
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
            "include_contents": "not_fingerprinted",
            "warning": "The option query exits before parsing; it does not independently validate source or preinclude contents.",
        },
        "caller_observations": {"errno": "unproven", "fenv": "unproven"},
        "runtime_fp_environment": "unobserved",
    }


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
    return (
        text,
        locations,
        fixed,
        dump,
        capture,
        semantic_flags,
        {
            "compiler": compiler,
            "semantic_command": [compiler, *flags, *semantic_flags, str(path)],
        },
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
    intrinsic_bindings: dict = field(default_factory=dict)

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


def _capture_unit(path, root, flags):
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
            _compile_capture(path, root, flags, tmp)
        )
        numeric = _numeric_contract(path, root, flags, commands, source_sha256)
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
        source_digests={relative: source_sha256},
        source_numeric_contract=numeric,
        intrinsic_bindings=_intrinsic_bindings(dump),
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
    return replace(unit, procedures=procedures, target=_capture_target(flags, tree))


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
        spec = next(
            (child for child in node.content if kind(child) == "Specification_Part"),
            None,
        )
        _declarations(spec, symbols, interfaces, associations)
    except Gap as error:
        problem = {
            "code": "fortran." + error.code,
            "detail": str(error),
            **unit.location(error.node or head),
        }
        if not block:
            problem["symbol"] = symbol

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


def parse_unit(path, flags=(), *, root, abi=None):
    """Read the selected source and capture the compiler evidence needed by F2."""
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        return _capture_unit(path, root, flags)
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

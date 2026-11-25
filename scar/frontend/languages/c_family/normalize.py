"""F2 C/C++ normalization from a captured compiler semantic unit to SCAR operations."""

import hashlib
from pathlib import PurePath

from . import io as io_effects, objects, standard_library
from .facts import (
    ParsedUnit,
    cursor_kind as _kind,
    type_kind,
    is_attribute,
    is_expression,
    function_parameter_types,
    element_type,
)


class _Unsupported(Exception):
    def __init__(self, cursor, code, detail):
        super().__init__(detail)
        self.cursor = cursor
        self.code = code


def _object(rule, *args):
    try:
        return rule(*args)
    except objects.BindingError as error:
        raise _Unsupported(error.cursor, "object.protocol", str(error)) from error


_BINARY = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "%": "mod",
    "<<": "shl",
    ">>": "shr",
    "<": "lt",
    "<=": "le",
    ">": "gt",
    ">=": "ge",
    "==": "eq",
    "!=": "ne",
    "&": "band",
    "|": "bor",
    "^": "bxor",
    "&&": "land",
    "||": "lor",
    ",": "comma",
}
_UNARY = {
    1: "postinc",
    2: "postdec",
    3: "preinc",
    4: "predec",
    5: "addr",
    6: "deref",
    7: "pos",
    8: "neg",
    9: "bnot",
    10: "not",
}
_SIGNED = {"SCHAR", "SHORT", "INT", "LONG", "LONGLONG"}
_UNSIGNED = {"UCHAR", "USHORT", "UINT", "ULONG", "ULONGLONG"}
_PRIMITIVES = {
    "VOID": "void",
    "BOOL": "bool",
    "CHAR_S": "char",
    "CHAR_U": "char",
    "FLOAT": "f32",
    "DOUBLE": "f64",
}
_FUNCTIONS = {
    "FUNCTION_DECL",
    "CXX_METHOD",
    "CONSTRUCTOR",
    "DESTRUCTOR",
    "CONVERSION_FUNCTION",
}
_RECORDS = {"STRUCT_DECL", "CLASS_DECL", "UNION_DECL"}
# These declarations affect compile-time lookup, access or validation. Clang's
# resolved references and canonical types already contain their effect at each use.
_LOOKUP_DECLARATIONS = {
    "TYPEDEF_DECL",
    "TYPE_ALIAS_DECL",
    "TYPE_ALIAS_TEMPLATE_DECL",
    "USING_DECLARATION",
    "USING_DIRECTIVE",
    "NAMESPACE_ALIAS",
    "CXX_ACCESS_SPEC_DECL",
    "STATIC_ASSERT",
    "CONCEPT_DECL",
}
_TEMPLATES = {
    "FUNCTION_TEMPLATE",
    "CLASS_TEMPLATE",
    "CLASS_TEMPLATE_PARTIAL_SPECIALIZATION",
}
_DIAGNOSTIC_ATTRIBUTES = {"WARN_UNUSED_ATTR", "WARN_UNUSED_RESULT_ATTR"}
_LIBC_HEADERS = {
    "stdio.h",
    "stdlib.h",
    "string.h",
    "strings.h",
    "time.h",
    "ctype.h",
    "unistd.h",
    "signal.h",
    "assert.h",
    "errno.h",
    "wchar.h",
    "locale.h",
    "setjmp.h",
    "inttypes.h",
    "fcntl.h",
    "pthread.h",
    "malloc.h",
    "stdio2.h",
    "string_fortified.h",
}
_SDK_RECORDS = {
    ("struct_timeval.h", "timeval"): ("timeval", "sys/time.h"),
    ("time.h", "itimerval"): ("itimerval", "sys/time.h"),
    ("struct_rusage.h", "rusage"): ("rusage", "sys/resource.h"),
    ("sigaction.h", "sigaction"): ("sigaction", "signal.h"),
    ("select.h", "fd_set"): ("fd_set", "sys/select.h"),
    ("__sigset_t.h", "__sigset_t"): ("sigset_t", "signal.h"),
    ("struct_tm.h", "tm"): ("tm", "time.h"),
}
_SDK_FORWARD_HEADERS = {
    "stdlib-float.h": "stdlib.h",
    "stdlib-bsearch.h": "stdlib.h",
    "stdlib.h": "stdlib.h",
    "stdio2.h": "stdio.h",
    "string_fortified.h": "string.h",
}


def _provider(cursor, facts=None):
    loc = cursor.location
    if not loc.file:
        return None
    path = PurePath(loc.file.name)
    if path.name in {"mpi.h", "mpio.h"} and (
        not facts.owns(loc) if facts is not None else loc.is_in_system_header
    ):
        return "mpi"
    if not loc.is_in_system_header:
        return None
    if path.name in _LIBC_HEADERS | _SDK_FORWARD_HEADERS.keys() or (
        path.parent.name == "sys"
        and path.name in {"time.h", "select.h", "resource.h", "types.h", "stat.h"}
    ):
        return "libc"
    if path.name in {
        "math.h",
        "mathcalls.h",
        "mathcalls-helper-functions.h",
        "math-finite.h",
        "cmath",
    }:
        return "libm"
    return None


def _public_header(cursor):
    name = PurePath(cursor.location.file.name)
    if name.name in _SDK_FORWARD_HEADERS:
        return _SDK_FORWARD_HEADERS[name.name]
    if name.parent.name == "sys":
        return "sys/" + name.name
    if name.name in _LIBC_HEADERS or name.name in {"math.h", "mpi.h", "mpio.h"}:
        return name.name
    return None


def _va_list(t):
    kind = type_kind(t)
    if kind not in {"CONSTANTARRAY", "POINTER"}:
        return False
    element = t.get_array_element_type() if kind == "CONSTANTARRAY" else t.get_pointee()
    declaration = element.get_canonical().get_declaration()
    return (
        declaration.get_usr() == "c:@S@__va_list_tag" and not declaration.location.file
    )


def _va_interface():
    return {"kind": "library", "provider": "c_abi", "name": "va_list", "args": []}


def _writable_string(value, target, location):
    if target != {"kind": "ptr", "to": "char"}:
        return None
    literal = value
    while literal["op"] == "cast" and literal.get("conversion") in {
        "implicit",
        "array_decay",
        "adjusted_array_parameter",
    }:
        literal = literal["value"]
    if literal["op"] != "string":
        return None
    return {
        "op": "cast",
        "mode": "qualification",
        "conversion": "legacy_string_pointer",
        "source_type": {
            "kind": "array",
            "of": {"kind": "const", "of": "char"},
            "shape": [len(literal["value"].encode("utf-8")) + 1],
        },
        "type": target,
        "value": literal,
        "location": location,
        "write_contract": "mutating_string_literal_is_undefined",
    }


def _symbol(cursor, unit_tag):
    name = cursor.mangled_name or cursor.spelling
    if cursor.linkage.name in {"INTERNAL", "UNIQUE_EXTERNAL"}:
        return f"scar_local_{unit_tag}_{name}"
    return name


def _qualified_name(cursor):
    def part(decl):
        name = decl.spelling
        if _kind(decl) in _RECORDS and decl.type.get_num_template_arguments() >= 0:
            name += "__" + hashlib.sha256(decl.get_usr().encode()).hexdigest()[:16]
        return name

    parts = [part(cursor)] if cursor.spelling else []
    parent = cursor.semantic_parent
    while parent and _kind(parent) in _RECORDS | {"NAMESPACE"}:
        if parent.spelling:
            parts.append(part(parent))
        parent = parent.semantic_parent
    if parts and cursor.spelling:
        result = "::".join(reversed(parts))
        if parent and _kind(parent) in _FUNCTIONS:
            result += "__" + hashlib.sha256(cursor.get_usr().encode()).hexdigest()[:16]
        return result
    # Clang's USR disambiguates unnamed records/typedefs without quoting source.
    return "scar_record_" + hashlib.sha256(cursor.get_usr().encode()).hexdigest()[:20]


def _template_id(cursor):
    return "template:" + hashlib.sha256(cursor.get_usr().encode()).hexdigest()[:24]


class _RecordIndex(dict):
    def __init__(self, facts=None):
        super().__init__()
        self.facts = facts
        self.specializations = {}

    def observe(self, cursor):
        if (
            cursor.type.get_num_template_arguments() >= 0
            and cursor.location.file
            and not cursor.location.is_in_system_header
        ):
            self.specializations.setdefault(_qualified_name(cursor), cursor)


def _type(original, cursor, records):
    source = original.get_canonical()
    kind = type_kind(source)
    if kind == "CONSTANTARRAY" and _va_list(source):
        return _va_interface()
    if source.is_volatile_qualified():
        raise _Unsupported(
            cursor,
            "type.volatile",
            "volatile-qualified access requires an ordered memory operation",
        )
    library = None
    if kind in {"RECORD", "ENUM"}:
        try:
            library = standard_library.wire_type(
                source, lambda t: _type(t, cursor, records)
            )
        except standard_library.BindingError as error:
            raise _Unsupported(cursor, "library.type", str(error)) from error
    if library:
        result = library
    elif kind in _PRIMITIVES:
        result = _PRIMITIVES[kind]
    elif kind in _SIGNED | _UNSIGNED:
        result = ("i" if kind in _SIGNED else "u") + str(8 * source.get_size())
    elif kind == "LONGDOUBLE":
        result = "f80"
    elif kind == "COMPLEX" and type_kind(element_type(source)) in {"FLOAT", "DOUBLE"}:
        result = "c64" if type_kind(element_type(source)) == "FLOAT" else "c128"
    elif kind == "ENUM":
        result = _type(source.get_declaration().enum_type, cursor, records)
    elif kind in {"POINTER", "LVALUEREFERENCE"}:
        result = {
            "kind": "ptr" if kind == "POINTER" else "ref",
            "to": _type(source.get_pointee(), cursor, records),
        }
    elif kind == "CONSTANTARRAY":
        shape = []
        element = source
        const_element = False
        while type_kind(element) == "CONSTANTARRAY":
            const_element |= element.is_const_qualified()
            shape.append(element.get_array_size())
            element = element.get_array_element_type().get_canonical()
        element_type = _type(element, cursor, records)
        if const_element and not element.is_const_qualified():
            element_type = {"kind": "const", "of": element_type}
        return {"kind": "array", "of": element_type, "shape": shape}
    elif kind == "INCOMPLETEARRAY" and _kind(cursor) == "PARM_DECL":
        result = {
            "kind": "ptr",
            "to": _type(source.get_array_element_type(), cursor, records),
        }
    elif kind == "FUNCTIONPROTO":
        return {
            "kind": "function",
            "result": _type(source.get_result(), cursor, records),
            "params": [
                _type(a, cursor, records) for a in function_parameter_types(source)
            ],
            "variadic": source.is_function_variadic(),
        }
    elif kind == "RECORD":
        record = source.get_declaration()
        if isinstance(records, _RecordIndex):
            records.observe(record)
        name = _qualified_name(record)
        result = {"kind": "record", "name": name}
        opaque = {"name": name, "opaque": True}
        where = record.location
        if where.file:
            header = PurePath(where.file.name)
            if (
                where.is_in_system_header
                and header.name in {"struct_FILE.h", "FILE.h", "__FILE.h", "stdio.h"}
                and record.spelling == "_IO_FILE"
            ):
                opaque.update(provider="libc", symbol="FILE")
            if (
                where.is_in_system_header
                and header.name in {"struct_tm.h", "time.h"}
                and record.spelling == "tm"
            ):
                opaque.update(provider="libc", symbol="tm", complete=True)
            if _provider(record, records.facts) == "mpi":
                opaque.update(
                    provider="mpi",
                    symbol=(
                        "MPI_Status"
                        if record.spelling == "ompi_status_public_t"
                        else record.spelling
                    ),
                )
                if record.spelling == "ompi_status_public_t":
                    opaque.update(header="mpi.h", complete=True)
            if (
                where.is_in_system_header
                and (header.name, record.spelling) in _SDK_RECORDS
            ):
                symbol, public_header = _SDK_RECORDS[header.name, record.spelling]
                opaque.update(
                    provider="libc", symbol=symbol, header=public_header, complete=True
                )
        records.setdefault(name, opaque)
    else:
        raise _Unsupported(
            cursor, f"type.{kind.lower()}", f"unsupported ABI type {source.spelling}"
        )
    return {"kind": "const", "of": result} if source.is_const_qualified() else result


def _param_type(cursor, records):
    t = cursor.type.get_canonical()
    if _va_list(t):
        return _va_interface()
    if type_kind(t) in {"CONSTANTARRAY", "INCOMPLETEARRAY", "VARIABLEARRAY"}:
        return {"kind": "ptr", "to": _type(t.get_array_element_type(), cursor, records)}
    return _type(t, cursor, records)


def _variably_modified_alias(cursor):
    ty = cursor.underlying_typedef_type.get_canonical()
    while True:
        kind = type_kind(ty)
        if kind in {"VARIABLEARRAY", "DEPENDENTSIZEDARRAY"}:
            return True
        if kind in {"POINTER", "LVALUEREFERENCE", "RVALUEREFERENCE"}:
            ty = ty.get_pointee()
        elif kind in {"CONSTANTARRAY", "INCOMPLETEARRAY"}:
            ty = ty.get_array_element_type()
        else:
            return False


class _Unit:
    def __init__(self, facts: ParsedUnit):
        self.facts = facts
        self.unit_tag = hashlib.sha256(
            "\0".join((facts.source_name, *facts.flags)).encode()
        ).hexdigest()[:16]
        self.functions = {}
        self.externs = {}
        self.globals = {}
        self.records = _RecordIndex(facts)
        self.templates = {}
        self.enums = {}
        self.pending_functions = {}
        self.expanded_records = set()
        self.methods = {}
        self.diagnostics = list(facts.diagnostics)
        self.transparent = []

    def macro_at(self, cursor):
        declaration = self.facts.macro(cursor)
        if (
            not declaration
            or not declaration.location.file
            or not declaration.location.is_in_system_header
        ):
            return None
        header = PurePath(declaration.location.file.name).name
        if declaration.spelling == "assert" and header == "assert.h":
            return "assert"
        if declaration.spelling == "va_arg" and header in {
            "stdarg.h",
            "__stdarg_va_arg.h",
        }:
            if any(
                t.spelling == "__builtin_va_arg" for t in self.facts.tokens(declaration)
            ):
                return "va_arg"
        return None

    def assertion_file(self, cursor):
        if any(
            f.startswith(("-fmacro-prefix-map=", "-ffile-prefix-map="))
            for f in self.facts.flags
        ):
            raise _Unsupported(
                cursor,
                "literal.file_remap",
                "assertion file requires the compiler's remapped path",
            )
        return {"op": "string", "value": self.facts.presumed_filename(cursor)}

    def type_info(self, t, cursor):
        return _type(t, cursor, self.records)

    def symbol(self, cursor):
        return _symbol(cursor, self.unit_tag)

    def record_name(self, cursor):
        return _qualified_name(cursor)

    def param_info(self, cursor):
        return {"name": cursor.spelling, "type": _param_type(cursor, self.records)}

    def require(self, cursor):
        if not self.facts.owns(cursor.location):
            return
        definition = cursor.get_definition()
        if definition:
            symbol = self.symbol(definition)
            if symbol not in self.functions:
                self.pending_functions.setdefault(symbol, definition)

    def template(self, cursor):
        identity = _template_id(cursor)
        self.templates.setdefault(
            identity,
            {
                "id": identity,
                "name": cursor.spelling,
                "kind": _kind(cursor).lower(),
                "location": self.location(cursor.location),
                "parameters": [
                    {"name": p.spelling, "kind": _kind(p).lower()}
                    for p in cursor.get_children()
                    if _kind(p)
                    in {
                        "TEMPLATE_TYPE_PARAMETER",
                        "TEMPLATE_NON_TYPE_PARAMETER",
                        "TEMPLATE_TEMPLATE_PARAMETER",
                    }
                ],
            },
        )
        return identity

    def enumeration(self, cursor):
        if cursor.is_definition():
            name = self.record_name(cursor)
            self.enums[name] = {
                "name": name,
                "type": self.type_info(cursor.enum_type, cursor),
                "scoped": cursor.is_scoped_enum(),
                "location": self.location(cursor.location),
                "values": {
                    n.spelling: n.enum_value
                    for n in cursor.get_children()
                    if _kind(n) == "ENUM_CONSTANT_DECL"
                },
            }

    def expand(self):
        while self.pending_functions or self.records.specializations:
            while self.pending_functions:
                name, cursor = self.pending_functions.popitem()
                if name not in self.functions:
                    self.declaration(cursor)
            while self.records.specializations:
                name, cursor = self.records.specializations.popitem()
                if name not in self.expanded_records:
                    self.expanded_records.add(name)
                    self.declaration(cursor)
        for name, methods in self.methods.items():
            record = self.records.get(name)
            if record and not record.get("opaque"):
                declared = {m["symbol"]: m for m in record.get("methods", [])}
                declared.update(
                    {sym: info for sym, info in methods.items() if sym not in declared}
                )
                record["methods"] = list(declared.values())

    def location(self, location):
        return self.facts.location(location)

    def diagnostic(self, error, symbol=None):
        self.diagnostics.append(
            {
                **self.location(error.cursor.location),
                "code": "c_family." + error.code,
                "detail": str(error),
                **({"symbol": symbol} if symbol else {}),
            }
        )

    def mark_transparent(self, cursor, owner=None):
        self.transparent.append(
            {
                "kind": _kind(cursor),
                "role": "specification",
                "owner": [owner.spelling] if owner and owner.spelling else [],
                "location": self.location(cursor.location),
                "status": "transparent",
            }
        )

    def coverage(self):
        records = [
            {
                "kind": issue["code"],
                "role": "program_unit" if "symbol" in issue else "specification",
                "owner": [issue["symbol"]] if "symbol" in issue else [],
                "location": {
                    "file": issue["file"],
                    "line": issue["line"] or None,
                    "column": issue["column"],
                },
                "status": "incomplete",
                "diagnostic": issue,
            }
            for issue in self.diagnostics
        ]
        return {
            "semantics_version": 1,
            "language": self.facts.language,
            "scope": "project_lookup_and_attribute_metadata_and_local_semantic_diagnostics",
            "complete": not records,
            "records": [*self.transparent, *records],
        }

    def attribute_errors(self, cursor):
        if not self.facts.owns(cursor.location):
            return
        kind = _kind(cursor)
        for child in cursor.get_children():
            if not is_attribute(child):
                continue
            attr = _kind(child)
            if attr in _DIAGNOSTIC_ATTRIBUTES or attr in {
                "CXX_FINAL_ATTR",
                "CXX_OVERRIDE_ATTR",
            }:
                self.mark_transparent(child, cursor)
                continue
            if (
                kind in _RECORDS
                and cursor.is_definition()
                and attr in {"PACKED_ATTR", "ALIGNED_ATTR"}
            ):
                continue  # record() checks the actual layout.
            tokens = [t.spelling for t in self.facts.tokens(child)]
            head = tokens[: tokens.index("(")] if "(" in tokens else tokens
            if head[:2] in (["gnu", "::"], ["clang", "::"], ["_Clang", "::"]):
                head = head[2:]
            name = head[0].strip("_") if len(head) == 1 else ""
            if attr == "UNEXPOSED_ATTR" and name in {
                "deprecated",
                "unused",
                "maybe_unused",
                "nodiscard",
            }:
                self.mark_transparent(child, cursor)
                continue
            if kind in _FUNCTIONS and name in {"constructor", "destructor"}:
                yield _Unsupported(
                    child,
                    "state.lifecycle_hook",
                    f"process-start or exit hook for {self.symbol(cursor)}",
                )
            else:
                yield _Unsupported(
                    child,
                    "attribute." + attr.lower(),
                    f"{attr} {name or child.spelling} on {cursor.spelling or kind} requires an attribute rule",
                )

    def signature(self, cursor, *, definition=False):
        name = _symbol(cursor, self.unit_tag)
        params = []
        for i, arg in enumerate(cursor.get_arguments()):
            param = {
                "name": arg.spelling or f"arg{i}",
                "type": _param_type(arg, self.records),
            }
            # A function type can retain a qualification from a previous
            # prototype. Only the ParmVarDecl in this definition binds a root.
            if definition:
                qualified = arg.type.get_canonical()
                if (
                    type_kind(qualified) == "POINTER"
                    and qualified.is_restrict_qualified()
                    and type_kind(qualified.get_pointee())
                    not in {"FUNCTIONPROTO", "FUNCTIONNOPROTO"}
                ):
                    param["access_precondition"] = {
                        "rule": "exclusive_modified_bytes",
                        "scope": "invocation",
                        "paths": "based_on_parameter",
                        "readonly_when_accessed": qualified.get_pointee().is_const_qualified(),
                        "basis": {
                            "kind": "source_precondition",
                            "rule_id": "clang.definition_pointer_restrict",
                            "evidence": {
                                "location": self.location(arg.location),
                                "canonical_type": qualified.spelling,
                                "restrict_qualified": True,
                            },
                        },
                    }
            params.append(param)
        kind = _kind(cursor)
        result = (
            "void"
            if kind in {"CONSTRUCTOR", "DESTRUCTOR"}
            else _type(cursor.result_type, cursor, self.records)
        )
        if kind in {"CONSTRUCTOR", "DESTRUCTOR"} or (
            kind == "CXX_METHOD" and not cursor.is_static_method()
        ):
            owner = _type(cursor.semantic_parent.type, cursor, self.records)
            if cursor.is_const_method():
                owner = {"kind": "const", "of": owner}
            params.insert(0, {"name": "this", "type": {"kind": "ptr", "to": owner}})
        t = cursor.type.get_canonical()
        sig = {
            "name": name,
            "params": params,
            "result": result,
            "location": self.location(cursor.location),
        }
        method = _object(objects.method, cursor, self)
        if method:
            sig["method"] = method
            self.methods.setdefault(method["owner"], {})[method["symbol"]] = method
        if type_kind(t) == "FUNCTIONPROTO" and t.is_function_variadic():
            sig["variadic"] = True
        return sig

    def declare_call(self, cursor):
        name = _symbol(cursor, self.unit_tag)
        self.require(cursor)
        if name not in self.externs:
            signature = self.signature(cursor)
            provider = _provider(cursor, self.facts)
            if provider:
                signature["provider"] = provider
                header = _public_header(cursor)
                if header:
                    signature["header"] = header
            self.externs[name] = signature
        return name

    def call(self, cursor, args):
        name = self.declare_call(cursor)
        result = {
            "op": "call",
            "callee": name,
            "args": args,
            "parameter_types": [p["type"] for p in self.externs[name]["params"]],
        }
        definition = cursor.get_definition()
        projection = io_effects.project(
            self.externs[name],
            args,
            c_usr=cursor.get_usr(),
            application_definition=bool(
                definition and self.facts.owns(definition.location)
            ),
        )
        if projection is not None:
            result["artifact_effect"] = projection
        return result

    def function(self, cursor):
        name = _symbol(cursor, self.unit_tag)
        if not cursor.is_definition():
            return
        parent = cursor.semantic_parent
        if (
            parent
            and _kind(parent)
            in {"CLASS_TEMPLATE", "CLASS_TEMPLATE_PARTIAL_SPECIALIZATION"}
            and not cursor.mangled_name
        ):
            self.template(parent)
            return
        reader = _Body(self, cursor)
        try:
            declaration = self.signature(cursor, definition=True)
        except _Unsupported as error:
            declaration = {
                "name": name,
                "location": self.location(cursor.location),
                "result": None,
                "params": [
                    {"name": a.spelling, "type": None} for a in cursor.get_arguments()
                ],
                "signature_complete": False,
            }
            reader.errors.append(error)
        if _kind(cursor) == "CONVERSION_FUNCTION":
            reader.errors.append(
                _Unsupported(
                    cursor,
                    "object." + _kind(cursor).lower(),
                    f"lifetime or conversion protocol for {cursor.spelling}",
                )
            )
        children = list(cursor.get_children())
        blocks = [c for c in children if _kind(c) == "COMPOUND_STMT"]
        if len(blocks) == 1:
            for parameter in cursor.get_arguments():
                reader.local(parameter)
            declaration["body"] = reader.statements(blocks[0])
            try:
                _object(objects.prepare, declaration, cursor, reader)
            except _Unsupported as error:
                reader.errors.append(error)
            if name == "main" and (
                not declaration["body"] or declaration["body"][-1]["op"] != "return"
            ):
                declaration["body"].append(
                    {
                        "op": "return",
                        "value": {"op": "literal", "type": "i32", "value": 0},
                    }
                )
        elif (
            declaration.get("method", {}).get("defaulted")
            and not declaration["method"]["deleted"]
        ):
            declaration["body"] = []
        else:
            reader.errors.append(
                _Unsupported(
                    cursor, "function.body", "expected a compound function body"
                )
            )
        if reader.errors:
            declaration["partial_body"] = declaration.pop("body", [])
            for error in reader.errors:
                self.diagnostic(error, name)
            reader.collect_calls(cursor)
        declaration["calls"] = sorted(reader.calls)
        pattern = self.facts.template_pattern(cursor)
        if pattern:
            declaration["specializes"] = self.template(pattern)
        self.functions[name] = declaration

    def record(self, cursor):
        if not cursor.is_definition():
            return
        name = _qualified_name(cursor)
        children = list(cursor.get_children())
        fields = []
        try:
            if _kind(cursor) == "UNION_DECL" or any(
                _kind(c) in {"PACKED_ATTR", "ALIGNED_ATTR"} for c in children
            ):
                raise _Unsupported(
                    cursor, "record.layout", f"nontrivial record layout for {name}"
                )
            for field in cursor.type.get_fields():
                if not field.spelling or field.is_bitfield():
                    raise _Unsupported(
                        field,
                        "record.field_layout",
                        "record contains an anonymous or bit-packed field",
                    )
                fields.append(
                    {
                        "name": field.spelling,
                        "type": _type(field.type, field, self.records),
                        "offset_bits": field.get_field_offsetof(),
                    }
                )
                tokens = list(self.facts.tokens(field))
                # Keep aggregate/array bounds separate from the default initializer.
                following = [
                    t.spelling for t in tokens if t.offset > field.location.offset
                ]
                depth = 0
                for token in following:
                    if depth == 0 and token in {"=", "{"}:
                        operands = [c for c in field.get_children() if is_expression(c)]
                        if not operands:
                            raise _Unsupported(
                                field,
                                "record.field_initializer",
                                "field default lacks a resolved initializer",
                            )
                        reader = _Body(self)
                        fields[-1]["init"] = reader.expression(operands[-1])
                        if reader.calls:
                            fields[-1]["calls"] = sorted(reader.calls)
                        break
                    depth += int(token in {"[", "("}) - int(token in {"]", ")"})
            self.records[name] = {
                "name": name,
                "fields": fields,
                "size_bytes": cursor.type.get_size(),
                "align_bytes": cursor.type.get_align(),
                "standard_layout": True if cursor.type.is_pod() else None,
            }
            metadata = _object(objects.record, cursor, self)
            if (
                metadata["bases"]
                or any(m["virtual"] for m in metadata["methods"])
                or any(
                    isinstance(f["type"], dict) and f["type"].get("kind") == "ref"
                    for f in fields
                )
            ):
                self.records[name]["standard_layout"] = False
            self.records[name].update({k: v for k, v in metadata.items() if v})
        except _Unsupported as error:
            self.diagnostic(error, name)
        for child in children:
            kind = _kind(child)
            if is_attribute(child) or kind in {
                "TYPE_REF",
                "TEMPLATE_REF",
                "NAMESPACE_REF",
            }:
                continue
            if kind in {"FIELD_DECL", "CXX_BASE_SPECIFIER"} or (
                kind == "VAR_DECL" and self.facts.initializer(child) is None
            ):
                # Field/base layout was consumed above. A static data member
                # without an in-class initializer does not define its storage here.
                for error in self.attribute_errors(child):
                    self.diagnostic(error, name)
            else:
                self.declaration(child)

    def declaration(self, cursor):
        kind = _kind(cursor)
        if not self.facts.owns(cursor.location):
            return
        # A project declaration attribute can affect linkage or registration
        # even when the ordinary call graph does not reference this declaration.
        for error in self.attribute_errors(cursor):
            self.diagnostic(error)
        if kind in {"NAMESPACE", "LINKAGE_SPEC"}:
            for child in cursor.get_children():
                if not is_attribute(child):
                    self.declaration(child)
        elif kind in _FUNCTIONS:
            for parameter in cursor.get_arguments():
                for error in self.attribute_errors(parameter):
                    self.diagnostic(error, self.symbol(cursor))
            self.function(cursor)
        elif kind in _RECORDS:
            self.record(cursor)
        elif kind == "ENUM_DECL":
            for child in cursor.get_children():
                if _kind(child) == "ENUM_CONSTANT_DECL":
                    for error in self.attribute_errors(child):
                        self.diagnostic(error, self.record_name(cursor))
            try:
                self.enumeration(cursor)
            except _Unsupported as error:
                self.diagnostic(error, self.record_name(cursor))
        elif kind in _TEMPLATES:
            self.template(cursor)
        elif kind == "VAR_DECL":
            if not cursor.is_definition() and cursor.storage_class.name == "EXTERN":
                return  # The ABI of the referenced declaration is bound at its use.
            symbol = _symbol(cursor, self.unit_tag)
            try:
                self.globals[symbol] = _Body(self).variable(cursor, symbol)
            except _Unsupported as error:
                # The translation unit can run static initialization/destruction
                # before/after entry even when no function reads this variable.
                self.diagnostic(error)
        elif kind == "FRIEND_DECL":
            for child in cursor.get_children():
                # Type-only friendship changes access checking. A friend can
                # also introduce a function with its own externally callable body.
                if _kind(child) not in {
                    "TYPE_REF",
                    "TEMPLATE_REF",
                    "NAMESPACE_REF",
                } and not is_attribute(child):
                    self.declaration(child)
        elif kind in _LOOKUP_DECLARATIONS:
            self.mark_transparent(cursor, cursor.semantic_parent)
        else:
            self.diagnostic(
                _Unsupported(
                    cursor,
                    "declaration." + kind.lower(),
                    f"declaration {kind} requires a semantic rule",
                )
            )


class _Body:
    def __init__(self, unit, owner=None):
        self.unit = unit
        self.owner = owner
        self.names = {}
        self.taken_names = set()
        self.calls = set()
        self.errors = []

    def local(self, cursor):
        identity = cursor.hash
        if identity not in self.names:
            base = cursor.spelling or "local"
            name = base
            if name in self.taken_names:
                name = f"{base}_{cursor.location.offset}"
            self.names[identity] = name
            self.taken_names.add(name)
        return self.names[identity]

    def collect_calls(self, cursor):
        for node in cursor.walk_preorder():
            if (
                _kind(node) == "CALL_EXPR"
                and node.referenced
                and _kind(node.referenced) in _FUNCTIONS
                and not standard_library.is_standard(node.referenced)
            ):
                self.calls.add(_symbol(node.referenced, self.unit.unit_tag))

    def variable(self, cursor, name=None):
        if name is None:
            self.errors.extend(self.unit.attribute_errors(cursor))
        if type_kind(cursor.type.get_canonical()) == "VARIABLEARRAY" and name is None:
            element = cursor.type.get_canonical().get_array_element_type()
            extents = [c for c in cursor.get_children() if is_expression(c)]
            if (
                type_kind(element.get_canonical())
                in _SIGNED | _UNSIGNED | _PRIMITIVES.keys()
                and len(extents) == 1
            ):
                return {
                    "op": "runtime_array",
                    "name": self.local(cursor),
                    "element": self.unit.type_info(element, cursor),
                    "extent": self.expression(extents[0]),
                    "storage": "scoped",
                    "initialization": "uninitialized",
                }
            raise _Unsupported(
                cursor,
                "array.dynamic_extent",
                "dynamic local array needs a single extent and a trivial element",
            )
        ty = _type(cursor.type, cursor, self.unit.records)
        row = {"op": "decl", "name": name or self.local(cursor), "type": ty}
        if cursor.tls_kind.name != "NONE":
            raise _Unsupported(
                cursor,
                "state.thread_local",
                "thread-local allocation and initialization",
            )
        element = cursor.type.get_canonical()
        while type_kind(element) == "CONSTANTARRAY":
            element = element.get_array_element_type()
        if type_kind(element) == "RECORD" and not element.is_pod():
            element_type = _type(element, cursor, self.unit.records)
            if element_type.get("kind") == "const":
                element_type = element_type["of"]
            if element_type.get("kind") != "library":
                lifetime = _object(objects.lifetime, element, self)
                if not lifetime:
                    raise _Unsupported(
                        cursor,
                        "state.object_lifetime",
                        f"construction, destruction and cleanup of {element.spelling}",
                    )
                row["object_lifetime"] = lifetime
            row["lifetime"] = (
                "program" if name or cursor.storage_class.name == "STATIC" else "scope"
            )
        if cursor.storage_class.name == "STATIC" and name is None:
            row["storage"] = "static"
        initializer = self.unit.facts.initializer(cursor)
        if initializer is not None:
            if (
                ty == _va_interface()
                and _kind(initializer) == "CALL_EXPR"
                and not list(initializer.get_arguments())
            ):
                return row
            if (
                _kind(initializer) == "CALL_EXPR"
                and initializer.referenced
                and _kind(initializer.referenced) == "CONSTRUCTOR"
                and not any(a.location.file for a in initializer.get_arguments())
                and initializer.location.offset == cursor.location.offset
                and not any(
                    t.spelling in {"(", "{", "="}
                    for t in self.unit.facts.tokens(cursor)
                    if t.offset > cursor.location.offset
                )
            ):
                # A bare declaration default-initializes each element. Its
                # scoped storage and declared type reproduce this C++ operation.
                # The selected project constructor remains a reachability edge.
                if (
                    type_kind(element) == "RECORD"
                    and not element.is_pod()
                    and row.get("object_lifetime")
                ):
                    row["object_lifetime"]["constructor"] = self.unit.declare_call(
                        initializer.referenced
                    )
                    self.calls.add(row["object_lifetime"]["constructor"])
                return row
            row["init"] = self.expression(initializer)
        return row

    def reference(self, cursor):
        declaration = cursor.referenced
        kind = _kind(declaration) if declaration else ""
        public = standard_library.reference(
            declaration, lambda t: self.unit.type_info(t, cursor)
        )
        if public:
            return public
        if kind == "ENUM_CONSTANT_DECL":
            return {
                "op": "literal",
                "type": _type(cursor.type, cursor, self.unit.records),
                "value": declaration.enum_value,
            }
        if kind in _FUNCTIONS:
            name = self.unit.declare_call(declaration)
            self.calls.add(name)
            return {"op": "function_ref", "name": name}
        if kind not in {"VAR_DECL", "PARM_DECL", "BINDING_DECL"}:
            raise _Unsupported(
                cursor,
                "binding.reference",
                f"unresolved reference to {cursor.spelling} ({kind})",
            )
        if declaration.linkage.name in {"INTERNAL", "EXTERNAL", "UNIQUE_EXTERNAL"}:
            name = _symbol(declaration, self.unit.unit_tag)
            header_provider = _provider(declaration, self.unit.facts)
            if header_provider or (
                not declaration.is_definition() and not declaration.get_definition()
            ):
                external = {
                    "op": "decl",
                    "name": name,
                    "type": _type(declaration.type, declaration, self.unit.records),
                    "external": True,
                }
                if header_provider:
                    external["provider"] = header_provider
                self.unit.globals.setdefault(name, external)
            return {"op": "var", "name": name, "scope": "global"}
        return {"op": "var", "name": self.local(declaration)}

    def number(self, cursor):
        result = self.unit.facts.number(cursor)
        if result is None:
            raise _Unsupported(
                cursor,
                "literal.representation",
                f"no exact constant representation for {_kind(cursor)}",
            )
        return result

    def string(self, cursor):
        result = self.unit.facts.string(cursor)
        if result is None:
            raise _Unsupported(
                cursor,
                "literal.string_encoding",
                "string literal requires an explicit byte or wide-character representation",
            )
        return result

    def arguments(self, cursor, declaration):
        arguments = list(cursor.get_arguments())
        if not declaration:
            return [self.expression(arg) for arg in arguments]
        parameters = list(declaration.get_arguments())
        values = []
        offset = (
            1
            if (
                _kind(declaration) == "CXX_METHOD"
                and declaration.spelling.startswith("operator")
                and not declaration.is_static_method()
                and len(arguments) == len(parameters) + 1
            )
            else 0
        )
        for i, arg in enumerate(arguments):
            if (
                i == 1
                and declaration.spelling == "__assert_fail"
                and _provider(declaration, self.unit.facts) == "libc"
                and self.unit.macro_at(cursor) == "assert"
                and not list(arg.get_children())
                and _kind(arg) == "UNEXPOSED_EXPR"
            ):
                values.append(self.unit.assertion_file(cursor))
                continue
            if (
                not arg.location.file
                and _kind(arg) == "UNEXPOSED_EXPR"
                and 0 <= i - offset < len(parameters)
            ):
                parameter = parameters[i - offset]
                defaults = [n for n in parameter.get_children() if is_expression(n)]
                if len(defaults) != 1:
                    raise _Unsupported(
                        cursor,
                        "call.default_argument",
                        f"no resolved default initializer for parameter {i - offset + 1}",
                    )
                arg = defaults[0]
            value = self.expression(arg)
            if 0 <= i - offset < len(parameters) and type_kind(
                parameters[i - offset].type.get_canonical()
            ) in {
                "POINTER",
                "INCOMPLETEARRAY",
                "CONSTANTARRAY",
            }:
                value = (
                    _writable_string(
                        value,
                        _param_type(parameters[i - offset], self.unit.records),
                        self.unit.location(arg.location),
                    )
                    or value
                )
            if (
                0 <= i - offset < len(parameters)
                and type_kind(parameters[i - offset].type.get_canonical())
                == "LVALUEREFERENCE"
            ):
                value = {
                    "op": "cast",
                    "mode": "static",
                    "conversion": "reference_binding",
                    "source_type": self.unit.type_info(arg.type, arg),
                    "type": _param_type(parameters[i - offset], self.unit.records),
                    "value": value,
                }
            values.append(value)
        return values

    def expression(self, cursor):
        kind = _kind(cursor)
        children = list(cursor.get_children())
        if kind in {"UNARY_EXPR", "CXX_UNARY_EXPR"} and children:
            value = children[0]
            while (
                _kind(value) in {"PAREN_EXPR", "UNEXPOSED_EXPR"}
                and len(list(value.get_children())) == 1
            ):
                value = next(value.get_children())
            if (
                type_kind(value.type.get_canonical()) == "VARIABLEARRAY"
                and _kind(value) == "DECL_REF_EXPR"
                and value.referenced
                and next((t.spelling for t in self.unit.facts.tokens(cursor)), None)
                == "sizeof"
            ):
                return {
                    "op": "runtime_array_size",
                    "name": self.local(value.referenced),
                }
        if kind == "IMAGINARY_LITERAL" and len(children) == 1:
            return {
                "op": "literal",
                "type": self.unit.type_info(cursor.type, cursor),
                "value": [0.0, self.number(children[0])],
            }
        if kind in {
            "INTEGER_LITERAL",
            "FLOATING_LITERAL",
            "CHARACTER_LITERAL",
            "CXX_BOOL_LITERAL_EXPR",
            "UNARY_EXPR",
            "CXX_UNARY_EXPR",
        }:
            return {
                "op": "literal",
                "type": _type(cursor.type, cursor, self.unit.records),
                "value": self.number(cursor),
            }
        if kind == "STRING_LITERAL":
            return {"op": "string", "value": self.string(cursor)}
        if kind in {"CXX_NULL_PTR_LITERAL_EXPR", "GNU_NULL_EXPR"}:
            return {"op": "literal", "type": "i32", "value": 0}
        if kind == "DECL_REF_EXPR":
            return self.reference(cursor)
        if kind == "CXX_THIS_EXPR":
            return {"op": "var", "name": "this"}
        if kind == "CXX_NEW_EXPR":
            allocation = _object(objects.allocation, cursor, self)
            if allocation:
                return allocation
            pointee = cursor.type.get_pointee().get_canonical()
            tokens = [t.spelling for t in self.unit.facts.tokens(cursor)]
            if (
                type_kind(pointee) in {"RECORD", "CONSTANTARRAY"}
                or "new" not in tokens
                or tokens[tokens.index("new") + 1 :][:1] == ["("]
            ):
                raise _Unsupported(
                    cursor,
                    "allocation.construction",
                    "allocation requires an object or placement construction protocol",
                )
            operands = [c for c in children if is_expression(c)]
            out = {"op": "allocate", "type": _type(pointee, cursor, self.unit.records)}
            end_type = min(
                (i for i, t in enumerate(tokens) if t in {"(", "{"}),
                default=len(tokens),
            )
            if "[" in tokens[:end_type]:
                if not operands:
                    raise _Unsupported(
                        cursor,
                        "allocation.extent",
                        "array allocation has no captured extent",
                    )
                size = operands.pop(0)
                inner = list(size.get_children())
                if (
                    _kind(size) == "UNEXPOSED_EXPR"
                    and len(inner) == 1
                    and type_kind(inner[0].type.get_canonical()) in _SIGNED
                ):
                    size = inner[0]
                out["count"] = self.expression(size)
            if len(operands) > 1:
                raise _Unsupported(
                    cursor,
                    "allocation.arguments",
                    "allocation has additional placement or initialization arguments",
                )
            if operands:
                init = operands[0]
                out["init"] = (
                    {"op": "init_list", "items": []}
                    if _kind(init) == "UNEXPOSED_EXPR" and not list(init.get_children())
                    else self.expression(init)
                )
            return out
        if kind == "CXX_DELETE_EXPR" and len(children) == 1:
            destruction = _object(objects.deletion, cursor, self)
            if destruction:
                return destruction
            pointer = children[0].type.get_canonical()
            if (
                type_kind(pointer) != "POINTER"
                or type_kind(pointer.get_pointee().get_canonical()) == "RECORD"
            ):
                raise _Unsupported(
                    cursor,
                    "allocation.destruction",
                    "destruction requires a resolved destructor before releasing storage",
                )
            tokens = [t.spelling for t in self.unit.facts.tokens(cursor)]
            array = "delete" in tokens and tokens[tokens.index("delete") + 1 :][:2] == [
                "[",
                "]",
            ]
            return {
                "op": "release",
                "target": self.expression(children[0]),
                "array": array,
            }
        if kind in {
            "PAREN_EXPR",
            "UNEXPOSED_EXPR",
            "CSTYLE_CAST_EXPR",
            "CXX_STATIC_CAST_EXPR",
            "CXX_REINTERPRET_CAST_EXPR",
            "CXX_CONST_CAST_EXPR",
            "CXX_FUNCTIONAL_CAST_EXPR",
        }:
            operands = [c for c in children if is_expression(c)]
            if (
                kind == "UNEXPOSED_EXPR"
                and len(operands) == 1
                and self.unit.macro_at(cursor) == "va_arg"
                and _va_list(operands[0].type.get_canonical())
                and not _va_list(cursor.type.get_canonical())
            ):
                return {
                    "op": "variadic",
                    "name": "arg",
                    "list": self.expression(operands[0]),
                    "type": self.unit.type_info(cursor.type, cursor),
                }
            if (
                not operands
                and kind == "UNEXPOSED_EXPR"
                and type_kind(cursor.type.get_canonical())
                in _SIGNED | _UNSIGNED | {"FLOAT", "DOUBLE", "BOOL"}
            ):
                return {
                    "op": "literal",
                    "type": _type(cursor.type, cursor, self.unit.records),
                    "value": self.number(cursor),
                }
            if len(operands) != 1:
                raise _Unsupported(
                    cursor,
                    "expression.wrapper",
                    f"{kind} with {len(operands)} value operands",
                )
            source, target = (
                operands[0].type.get_canonical(),
                cursor.type.get_canonical(),
            )
            value = self.expression(operands[0])
            if kind == "PAREN_EXPR" or source == target:
                return value
            if (
                kind == "UNEXPOSED_EXPR"
                and type_kind(source) == type_kind(target) == "RECORD"
            ):
                if source.get_declaration() == target.get_declaration():
                    source_type = self.unit.type_info(source, cursor)
                    library = source_type.get("of", source_type)
                    if library.get("kind") == "library":
                        return {
                            "op": "library_value",
                            "provider": "cxx_standard",
                            "mode": "qualify",
                            "source_type": source_type,
                            "type": self.unit.type_info(target, cursor),
                            "value": value,
                        }
                    # Qualification changes on a record preserve its glvalue.
                    return {
                        "op": "cast",
                        "mode": "static",
                        "conversion": "qualification",
                        "source_type": self.unit.type_info(source, cursor),
                        "type": {
                            "kind": "ref",
                            "to": self.unit.type_info(target, cursor),
                        },
                        "value": value,
                    }
                if standard_library.is_standard(
                    source.get_declaration()
                ) and standard_library.is_standard(target.get_declaration()):
                    return value
                # Implicit derived-to-base subobject conversion preserves the lvalue.
                # Base copying or user-defined conversion appears as a constructor or call.
                ty = {"kind": "ref", "to": _type(target, cursor, self.unit.records)}
                return {
                    "op": "cast",
                    "mode": "static",
                    "conversion": "base_subobject",
                    "source_type": self.unit.type_info(source, cursor),
                    "type": ty,
                    "value": value,
                }
            if type_kind(target) == "POINTER" and type_kind(source) in {
                "CONSTANTARRAY",
                "INCOMPLETEARRAY",
                "VARIABLEARRAY",
                "FUNCTIONPROTO",
                "FUNCTIONNOPROTO",
            }:
                if _va_list(source):
                    return value
                target_type = self.unit.type_info(target, cursor)
                if type_kind(source) in {"INCOMPLETEARRAY", "VARIABLEARRAY"}:
                    original_type = {
                        "kind": "array",
                        "of": self.unit.type_info(
                            source.get_array_element_type(), cursor
                        ),
                        "shape": [None],
                    }
                elif type_kind(source) == "FUNCTIONNOPROTO":
                    original_type = target_type["to"]
                else:
                    original_type = self.unit.type_info(source, cursor)
                mode = (
                    "qualification"
                    if value["op"] == "string"
                    and not target.get_pointee().is_const_qualified()
                    else "static"
                )
                return {
                    "op": "cast",
                    "mode": mode,
                    "conversion": (
                        "function_decay"
                        if type_kind(source).startswith("FUNCTION")
                        else "array_decay"
                    ),
                    "source_type": original_type,
                    "type": target_type,
                    "value": value,
                }
            if (
                kind == "UNEXPOSED_EXPR"
                and type_kind(target) == "INCOMPLETEARRAY"
                and type_kind(source) == "POINTER"
            ):
                # Clang exposes the parameter's written array type on this
                # implicit argument conversion. Its adjusted ABI is a pointer.
                adjusted = {
                    "kind": "ptr",
                    "to": self.unit.type_info(target.get_array_element_type(), cursor),
                }
                return _writable_string(
                    value, adjusted, self.unit.location(cursor.location)
                ) or {
                    "op": "cast",
                    "mode": "static",
                    "conversion": "adjusted_array_parameter",
                    "source_type": self.unit.type_info(source, cursor),
                    "type": {
                        "kind": "ptr",
                        "to": self.unit.type_info(
                            target.get_array_element_type(), cursor
                        ),
                    },
                    "value": value,
                }
            if type_kind(target) == "POINTER" and (
                type_kind(source) == "NULLPTR"
                or (
                    isinstance(value, dict)
                    and value.get("op") == "literal"
                    and value.get("value") == 0
                )
            ):
                return {
                    "op": "literal",
                    "type": _type(target, cursor, self.unit.records),
                    "value": None,
                }
            mode = "static"
            if kind == "CXX_CONST_CAST_EXPR":
                mode = "qualification"
            elif kind == "CXX_REINTERPRET_CAST_EXPR":
                mode = "reinterpret"
            elif kind == "CSTYLE_CAST_EXPR" and (
                type_kind(source) == "POINTER" or type_kind(target) == "POINTER"
            ):
                mode = "explicit_pointer"
            target_type = self.unit.type_info(target, cursor)
            source_type = self.unit.type_info(source, cursor)
            if kind == "UNEXPOSED_EXPR":
                legacy = _writable_string(
                    value, target_type, self.unit.location(cursor.location)
                )
                if legacy:
                    return legacy
                if target_type == {"kind": "const", "of": source_type}:
                    # Qualification adds no value copy. A const reference bound to
                    # this expression must still observe writes through its alias.
                    return {
                        "op": "cast",
                        "mode": "static",
                        "conversion": "qualification",
                        "source_type": source_type,
                        "type": {"kind": "ref", "to": target_type},
                        "value": value,
                    }
            emitted_type = target_type
            if isinstance(target_type, dict) and target_type.get("kind") == "const":
                emitted_type = target_type["of"]
            return {
                "op": "cast",
                "mode": mode,
                "conversion": "implicit" if kind == "UNEXPOSED_EXPR" else "explicit",
                "source_type": source_type,
                "target_type": target_type,
                "type": emitted_type,
                "value": value,
            }
        if kind == "ARRAY_SUBSCRIPT_EXPR":
            return {
                "op": "index",
                "base": self.expression(children[0]),
                "indices": [self.expression(children[1])],
            }
        if (
            kind == "MEMBER_REF_EXPR"
            and len(children) <= 1
            and cursor.referenced
            and _kind(cursor.referenced) == "FIELD_DECL"
        ):
            base = (
                self.expression(children[0])
                if children
                else {"op": "var", "name": "this"}
            )
            if not children or type_kind(children[0].type.get_canonical()) == "POINTER":
                base = {"op": "unary", "operator": "deref", "value": base}
            if standard_library.public_field(cursor.referenced):
                return {
                    "op": "library_member",
                    "provider": "cxx_standard",
                    "receiver": base,
                    "name": cursor.spelling,
                }
            if (
                base.get("op") == "member"
                and base.get("record") == "sigaction"
                and base["field"] == "__sigaction_handler"
                and cursor.spelling in {"sa_handler", "sa_sigaction"}
            ):
                return {**base, "field": cursor.spelling}
            record = cursor.referenced.semantic_parent
            while record and _kind(record) in _RECORDS:
                self.unit.type_info(record.type, cursor)
                binding = self.unit.records.get(self.unit.record_name(record), {})
                if binding.get("provider") in {"libc", "mpi"}:
                    return {
                        "op": "member",
                        "base": base,
                        "field": cursor.spelling,
                        "provider": binding["provider"],
                        "record": binding["symbol"],
                    }
                if not record.is_anonymous():
                    break
                record = record.semantic_parent
            return {"op": "member", "base": base, "field": cursor.spelling}
        if kind in {"BINARY_OPERATOR", "COMPOUND_ASSIGNMENT_OPERATOR"}:
            symbol = self.unit.facts.binary_operator(cursor)
            left, right = (self.expression(c) for c in children)
            if symbol in _BINARY:
                return {
                    "op": "binary",
                    "operator": _BINARY[symbol],
                    "left": left,
                    "right": right,
                }
            if symbol == "=" or (symbol.endswith("=") and symbol[:-1] in _BINARY):
                return {
                    "op": "update",
                    "operator": "set" if symbol == "=" else _BINARY[symbol[:-1]],
                    "target": left,
                    "value": right,
                }
            raise _Unsupported(
                cursor, "expression.binary_operator", f"operator {symbol}"
            )
        if kind == "UNARY_OPERATOR":
            opcode = self.unit.facts.unary_opcode(cursor)
            if opcode == 13 and len(children) == 1:  # GCC __extension__ marker.
                return self.expression(children[0])
            operator = _UNARY.get(opcode)
            if operator and len(children) == 1:
                return {
                    "op": "unary",
                    "operator": operator,
                    "value": self.expression(children[0]),
                }
        if kind == "CALL_EXPR":
            declaration = cursor.referenced
            if (
                declaration
                and _kind(declaration) == "CXX_METHOD"
                and declaration.spelling == "operator="
                and declaration.is_default_method()
                and declaration.semantic_parent.type.is_pod()
            ):
                operands = list(cursor.get_arguments())
                if len(operands) == 2:
                    return {
                        "op": "update",
                        "operator": "set",
                        "target": self.expression(operands[0]),
                        "value": self.expression(operands[1]),
                    }
            va_builtins = {
                "c:@F@__builtin_va_start": "start",
                "c:@F@__builtin_va_end": "end",
                "c:@F@__builtin_va_copy": "copy",
            }
            variadic = va_builtins.get(declaration.get_usr()) if declaration else None
            if variadic:
                args = list(cursor.get_arguments())
                if not args or not _va_list(args[0].type.get_canonical()):
                    raise _Unsupported(
                        cursor,
                        "variadic.abi",
                        "variadic list needs a verified target ABI",
                    )
                result = {
                    "op": "variadic",
                    "name": variadic,
                    "list": self.expression(args[0]),
                }
                if len(args) > 1:
                    result["source" if variadic == "copy" else "last"] = (
                        self.expression(args[1])
                    )
                return result
            try:
                library = standard_library.bind_call(
                    cursor,
                    declaration,
                    self.expression,
                    lambda t: _type(t, cursor, self.unit.records),
                    owns=self.unit.facts.owns,
                )
            except standard_library.BindingError as error:
                raise _Unsupported(cursor, "library.call", str(error)) from error
            if library:
                return library
            method = _object(objects.invoke, cursor, self)
            if method:
                return method
            args = self.arguments(cursor, declaration)
            if declaration and _kind(declaration) == "FUNCTION_DECL":
                call = self.unit.call(declaration, args)
                self.calls.add(call["callee"])
                return call
            if declaration and _kind(declaration) == "CXX_METHOD":
                if declaration.is_virtual_method():
                    raise _Unsupported(
                        cursor,
                        "call.virtual",
                        f"virtual dispatch slot for {declaration.spelling}",
                    )
                if not self.unit.facts.owns(declaration.location):
                    raise _Unsupported(
                        cursor,
                        "call.library_method",
                        f"operator provider for {declaration.displayname}",
                    )
                if not declaration.is_static_method():
                    if (
                        children
                        and _kind(children[0]) == "MEMBER_REF_EXPR"
                        and children[0].referenced == declaration
                    ):
                        receiver = [
                            c for c in children[0].get_children() if is_expression(c)
                        ]
                    elif (
                        declaration.spelling.startswith("operator")
                        and len(args) == len(list(declaration.get_arguments())) + 1
                    ):
                        receiver = [next(cursor.get_arguments())]
                        args.pop(0)
                    else:
                        raise _Unsupported(
                            cursor,
                            "call.operator_receiver",
                            f"receiver for {declaration.displayname}",
                        )
                    this = {"op": "var", "name": "this"}
                    if receiver:
                        this = self.expression(receiver[0])
                        if type_kind(receiver[0].type.get_canonical()) != "POINTER":
                            this = {"op": "unary", "operator": "addr", "value": this}
                    args.insert(0, this)
                call = self.unit.call(declaration, args)
                self.calls.add(call["callee"])
                if (
                    declaration.is_static_method()
                    and children
                    and _kind(children[0]) == "MEMBER_REF_EXPR"
                ):
                    bases = [c for c in children[0].get_children() if is_expression(c)]
                    if bases:
                        return {
                            "op": "binary",
                            "operator": "comma",
                            "left": self.expression(bases[0]),
                            "right": call,
                        }
                return call
            if children and is_expression(children[0]):
                pointer = children[0].type.get_canonical()
                target_type = (
                    pointer.get_pointee()
                    if type_kind(pointer) == "POINTER"
                    else pointer
                )
                # A call through a field, subscript or conditional expression may
                # have a precise callable ABI even with no single declaration.
                if type_kind(target_type) == "FUNCTIONPROTO":
                    return {
                        "op": "indirect_call",
                        "target": self.expression(children[0]),
                        "args": args,
                        "signature": _type(target_type, cursor, self.unit.records),
                        "location": self.unit.location(cursor.location),
                    }
            raise _Unsupported(
                cursor,
                "call.dispatch",
                f"call requires receiver or indirect target for {cursor.spelling}",
            )
        if kind == "INIT_LIST_EXPR":
            return {"op": "init_list", "items": [self.expression(c) for c in children]}
        if kind == "CONDITIONAL_OPERATOR" and len(children) == 3:
            return {
                "op": "select",
                "condition": self.expression(children[0]),
                "then": self.expression(children[1]),
                "else": self.expression(children[2]),
            }
        raise _Unsupported(
            cursor,
            "expression." + kind.lower(),
            f"expression {kind} requires an operation rule",
        )

    def statements(self, cursor):
        body = []
        for child in (
            cursor.get_children() if _kind(cursor) == "COMPOUND_STMT" else (cursor,)
        ):
            try:
                body.extend(self.statement(child))
            except _Unsupported as error:
                self.errors.append(error)
                body.append(
                    {
                        "op": "unsupported",
                        "construct": _kind(error.cursor),
                        "diagnostic": "c_family." + error.code,
                        "location": self.unit.location(error.cursor.location),
                        "effects": "unknown",
                    }
                )
        return body

    def for_parts(self, cursor, children):
        if len(children) == 4:
            return children
        # libclang omits empty clauses; use only the punctuation boundaries to assign roles.
        separators = []
        depth = 0
        for token in self.unit.facts.tokens(cursor):
            spelling = token.spelling
            if spelling == "(":
                depth += 1
            elif spelling == ")":
                depth -= 1
                if depth == 0:
                    break
            elif spelling == ";" and depth == 1:
                separators.append(token.offset)
        if len(separators) != 2 or not children:
            raise _Unsupported(
                cursor, "control.for", "cannot bind the three clauses of this for loop"
            )
        clauses = [None, None, None]
        for child in children[:-1]:
            slot = sum(
                child.extent.start.offset > separator for separator in separators
            )
            if clauses[slot] is not None:
                raise _Unsupported(cursor, "control.for", "ambiguous for clause")
            clauses[slot] = child
        return [*clauses, children[-1]]

    def statement(self, cursor):
        rows = self._statement(cursor)
        location = self.unit.location(cursor.location)
        for row in rows:
            row.setdefault("location", location)
        return rows

    def _statement(self, cursor):
        kind = _kind(cursor)
        children = list(cursor.get_children())
        if kind == "NULL_STMT":
            return []
        if kind == "COMPOUND_STMT":
            return [{"op": "block", "body": self.statements(cursor)}]
        if kind == "DECL_STMT":
            result = []
            for child in children:
                if _kind(child) == "VAR_DECL":
                    result.append(self.variable(child))
                elif _kind(child) in _RECORDS:
                    self.unit.record(child)
                elif _kind(child) == "ENUM_DECL":
                    self.unit.enumeration(child)
                elif _kind(child) in _LOOKUP_DECLARATIONS:
                    if _kind(child) in {
                        "TYPEDEF_DECL",
                        "TYPE_ALIAS_DECL",
                    } and _variably_modified_alias(child):
                        raise _Unsupported(
                            child,
                            "declaration.dynamic_type_bound",
                            "block-scope type alias requires evaluation of a dynamic bound",
                        )
                    self.errors.extend(self.unit.attribute_errors(child))
                    self.unit.mark_transparent(child, self.owner)
                else:
                    raise _Unsupported(
                        child,
                        "declaration." + _kind(child).lower(),
                        "requires a scoped declaration rule",
                    )
            return result
        if kind == "RETURN_STMT":
            return [
                {
                    "op": "return",
                    **({"value": self.expression(children[0])} if children else {}),
                }
            ]
        if kind == "FOR_STMT":
            init, cond, step, body = self.for_parts(cursor, children)
            return [
                {
                    "op": "for",
                    "init": self.statement(init) if init else [],
                    "condition": (
                        self.expression(cond)
                        if cond
                        else {"op": "literal", "type": "bool", "value": True}
                    ),
                    "step": (
                        self.expression(step)
                        if step
                        else {"op": "literal", "type": "i32", "value": 0}
                    ),
                    "body": self.statements(body),
                }
            ]
        if (
            kind == "CXX_FOR_RANGE_STMT"
            and len(children) == 3
            and _kind(children[0]) == "VAR_DECL"
        ):
            variable, iterable, body = children
            if type_kind(iterable.type.get_canonical()) != "CONSTANTARRAY":
                raise _Unsupported(
                    cursor,
                    "control.range_protocol",
                    "user-defined begin/end require explicit calls",
                )
            return [
                {
                    "op": "each",
                    "name": self.local(variable),
                    "type": _type(variable.type, variable, self.unit.records),
                    "range": self.expression(iterable),
                    "body": self.statements(body),
                }
            ]
        if kind == "IF_STMT" and len(children) in {2, 3}:
            return [
                {
                    "op": "if",
                    "condition": self.expression(children[0]),
                    "then": self.statements(children[1]),
                    "else": self.statements(children[2]) if len(children) == 3 else [],
                }
            ]
        if kind in {"WHILE_STMT", "DO_STMT"} and len(children) == 2:
            cond, body = children if kind == "WHILE_STMT" else reversed(children)
            return [
                {
                    "op": "while" if kind == "WHILE_STMT" else "do_while",
                    "condition": self.expression(cond),
                    "body": self.statements(body),
                }
            ]
        if kind in {"BREAK_STMT", "CONTINUE_STMT"}:
            return [{"op": "break" if kind == "BREAK_STMT" else "continue"}]
        if kind == "SWITCH_STMT" and len(children) == 2:
            return [
                {
                    "op": "switch",
                    "condition": self.expression(children[0]),
                    "body": self.statements(children[1]),
                }
            ]
        if kind == "CASE_STMT" and len(children) == 2:
            return [
                {"op": "case", "value": self.expression(children[0])},
                *self.statement(children[1]),
            ]
        if kind == "DEFAULT_STMT" and len(children) == 1:
            return [{"op": "default"}, *self.statement(children[0])]
        if kind == "GOTO_STMT" and cursor.referenced:
            return [{"op": "goto", "name": cursor.referenced.spelling}]
        if kind == "LABEL_STMT" and len(children) == 1:
            return [
                {"op": "label", "name": cursor.spelling},
                *self.statement(children[0]),
            ]
        if is_expression(cursor):
            expr = self.expression(cursor)
            if expr["op"] == "update" and expr["operator"] == "set":
                return [
                    {"op": "assign", "target": expr["target"], "value": expr["value"]}
                ]
            return [{"op": "expr", "value": expr}]
        raise _Unsupported(
            cursor,
            "statement." + kind.lower(),
            f"statement {kind} requires a control/effect rule",
        )


def normalize_unit(facts: ParsedUnit) -> dict:
    """Consume the captured AST and return a neutral unit without compiler handles."""
    unit = _Unit(facts)
    for declaration in facts.declarations:
        unit.declaration(declaration)
    unit.expand()
    return {
        "functions": list(unit.functions.values()),
        "externs": [
            row for name, row in unit.externs.items() if name not in unit.functions
        ],
        "globals": list(unit.globals.values()),
        "records": list(unit.records.values()),
        "templates": list(unit.templates.values()),
        "enums": list(unit.enums.values()),
        "diagnostics": unit.diagnostics,
        **({"source_coverage": unit.coverage()} if facts.language else {}),
    }

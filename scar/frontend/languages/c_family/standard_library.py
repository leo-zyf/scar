"""Bind compiler-resolved standard library declarations to typed operator interfaces."""

from .facts import cursor_kind, type_kind, is_expression


class BindingError(ValueError):
    """A standard-library declaration needs a typed operation or target binding."""


_TYPES = {
    "basic_string",
    "allocator",
    "char_traits",
    "less",
    "pair",
    "map",
    "vector",
    "basic_ios",
    "basic_istream",
    "basic_ostream",
    "basic_ifstream",
    "basic_ofstream",
    "basic_istringstream",
    "basic_ostringstream",
    "basic_stringstream",
    "complex",
    "reverse_iterator",
    "pointer_to_unary_function",
    "unary_negate",
}
_ENUMS = {
    "_Ios_Openmode": "ios_openmode",
    "_Ios_Fmtflags": "ios_fmtflags",
    "_Ios_Seekdir": "ios_seekdir",
    "_Ios_Iostate": "ios_iostate",
}
_STREAMS = {"cin", "cout", "cerr", "clog"}
_MANIPULATORS = {
    "endl",
    "ends",
    "flush",
    "hex",
    "dec",
    "oct",
    "fixed",
    "scientific",
    "left",
    "right",
    "boolalpha",
    "showpos",
    "showpoint",
}
_FLAGS = {
    "beg",
    "cur",
    "end",
    "goodbit",
    "eofbit",
    "failbit",
    "badbit",
    "skipws",
    "unitbuf",
    "uppercase",
    "app",
    "ate",
    "binary",
    "in",
    "out",
    "trunc",
    "left",
    "right",
    "internal",
    "fixed",
    "scientific",
    "showpos",
    "showpoint",
    "dec",
    "oct",
    "hex",
    "boolalpha",
    "floatfield",
}
_METHODS = {
    "c_str",
    "data",
    "empty",
    "size",
    "length",
    "begin",
    "end",
    "rbegin",
    "rend",
    "find",
    "insert",
    "erase",
    "append",
    "substr",
    "compare",
    "assign",
    "clear",
    "resize",
    "good",
    "fail",
    "eof",
    "bad",
    "is_open",
    "open",
    "close",
    "getline",
    "read",
    "write",
    "get",
    "put",
    "flush",
    "str",
    "precision",
    "setf",
    "unsetf",
    "width",
    "flags",
    "fill",
    "push_back",
    "back",
    "front",
    "base",
    "swap",
    "real",
    "imag",
    "seekg",
    "seekp",
    "tellg",
    "tellp",
    "rfind",
    "find_first_not_of",
    "find_first_of",
    "find_last_not_of",
    "find_last_of",
}
_OPERATORS = {
    "operator<<",
    "operator>>",
    "operator==",
    "operator!=",
    "operator=",
    "operator[]",
    "operator->",
    "operator*",
    "operator+",
    "operator-",
    "operator++",
    "operator--",
    "operator<",
    "operator>",
    "operator<=",
    "operator>=",
    "operator+=",
    "operator!",
    "operator/",
    "operator|",
    "operator&",
    "operator-=",
    "operator*=",
    "operator/=",
}
_FREE = set(
    """getline setw setprecision setfill min max swap find_if not1 ptr_fun flush abs real imag conj acos acosh asin asinh
atan atan2 atanh cbrt ceil copysign cos cosh erf erfc exp exp2 expm1 fabs fdim floor fma fmax fmin
fmod frexp hypot ldexp lgamma log log10 log1p log2 modf nearbyint nextafter pow remainder rint round
scalbn sin sinh sqrt tan tanh tgamma trunc isfinite isinf isnan isnormal signbit fpclassify""".split()
)


def _namespace(cursor):
    if not cursor or not cursor.location.is_in_system_header:
        return ()
    parts = []
    parent = cursor.semantic_parent
    while parent:
        if cursor_kind(parent) == "NAMESPACE":
            parts.append(parent.spelling)
        parent = parent.semantic_parent
    return tuple(reversed(parts))


def _library_type(name, args=()):
    return {
        "kind": "library",
        "provider": "cxx_standard",
        "name": name,
        "args": list(args),
    }


def is_standard(declaration):
    return _namespace(declaration)[:1] == ("std",)


def wire_type(source, lower):
    declaration = source.get_declaration()
    if (
        _namespace(declaration) == ("__gnu_cxx",)
        and declaration.spelling == "__normal_iterator"
    ):
        pointer, owner = (
            source.get_template_argument_type(i).get_canonical() for i in range(2)
        )
        if type_kind(pointer) != "POINTER" or not is_standard(owner.get_declaration()):
            raise BindingError(
                "iterator requires a public container and element qualification"
            )
        name = (
            "const_iterator"
            if pointer.get_pointee().is_const_qualified()
            else "iterator"
        )
        return _library_type(name, [lower(owner)])
    if not is_standard(declaration):
        return None
    family = declaration.spelling
    if family in _ENUMS:
        return _library_type(_ENUMS[family])
    if family in {"_Rb_tree_iterator", "_Rb_tree_const_iterator"}:
        element = source.get_template_argument_type(0)
        if (
            element.get_declaration().spelling != "pair"
            or element.get_num_template_arguments() != 2
        ):
            raise BindingError(
                "ordered tree iterator requires a known container and value type"
            )
        key = lower(element.get_template_argument_type(0))
        if isinstance(key, dict) and key.get("kind") == "const":
            key = key["of"]
        name = "map_iterator" if family == "_Rb_tree_iterator" else "map_const_iterator"
        return _library_type(name, [key, lower(element.get_template_argument_type(1))])
    if family not in _TYPES:
        return None
    arguments = []
    for i in range(max(0, source.get_num_template_arguments())):
        argument = source.get_template_argument_type(i)
        if type_kind(argument) == "INVALID":
            raise BindingError(
                f"non-type template argument of {family} needs a structural binding"
            )
        arguments.append(lower(argument))
    return _library_type(family, arguments)


def reference(declaration, lower_type):
    if not is_standard(declaration):
        return None
    name = declaration.spelling
    parent = declaration.semantic_parent
    if name == "npos" and parent.spelling == "basic_string":
        return {
            "op": "library_ref",
            "provider": "cxx_standard",
            "name": name,
            "type": lower_type(parent.type),
        }
    if parent.spelling == "ios_base" and name in _FLAGS:
        name = "ios_" + name
    elif not (name in _STREAMS | _MANIPULATORS and cursor_kind(parent) == "NAMESPACE"):
        return None
    return {"op": "library_ref", "provider": "cxx_standard", "name": name}


def public_field(declaration):
    return (
        declaration.spelling in {"first", "second"}
        and declaration.semantic_parent.spelling == "pair"
        and _namespace(declaration)[:1] == ("std",)
    )


def _stream_family(source, owns):
    """Identify the captured public char stream specialization, including its traits."""
    source = source.get_canonical()
    if type_kind(source) == "POINTER":
        source = source.get_pointee().get_canonical()
    owner = source.get_declaration()
    if (
        owner.spelling not in {"basic_ifstream", "basic_istream"}
        or _namespace(owner) != ("std",)
        or owns(owner.location)
        or source.get_num_template_arguments() != 2
    ):
        return None
    char, traits = (
        source.get_template_argument_type(i).get_canonical() for i in (0, 1)
    )
    declaration = traits.get_declaration()
    if (
        type_kind(char) in {"CHAR_S", "CHAR_U"}
        and not char.is_const_qualified()
        and not char.is_volatile_qualified()
        and declaration.spelling == "char_traits"
        and _namespace(declaration) == ("std",)
        and not owns(declaration.location)
        and traits.get_num_template_arguments() == 1
        and traits.get_template_argument_type(0).get_canonical() == char
    ):
        return owner.spelling
    return None


def _file_receiver(cursor, owns):
    # Inherited getline sees a basic_istream subobject. Follow only Clang's
    # transparent receiver conversions, never a call's arguments or operator body.
    while cursor is not None:
        family = _stream_family(cursor.type, owns)
        if family == "basic_ifstream":
            return True
        if family != "basic_istream" or cursor_kind(cursor) not in {
            "UNEXPOSED_EXPR",
            "PAREN_EXPR",
        }:
            break
        children = [c for c in cursor.get_children() if is_expression(c)]
        cursor = children[0] if len(children) == 1 else None
    return False


def _ifstream_effect(declaration, receiver, argc, lower_type, owns):
    kind, name = cursor_kind(declaration), declaration.spelling
    constructor = kind == "CONSTRUCTOR"
    if not (
        constructor
        and name == "basic_ifstream"
        or kind == "CXX_METHOD"
        and name in {"open", "close", "getline"}
    ):
        return None
    family = "basic_istream" if name == "getline" else "basic_ifstream"
    if (
        owns(declaration.location)
        or _stream_family(declaration.semantic_parent.type, owns) != family
        or not constructor
        and not _file_receiver(receiver, owns)
    ):
        return None
    definition = declaration.get_definition()
    if definition is not None and owns(definition.location):
        return None
    params = [p.type.get_canonical() for p in declaration.get_arguments()]
    result = declaration.result_type.get_canonical()
    contract = dict(
        semantics_version=1,
        source_profile="cxx_standard_ifstream_char_v1",
        source_interface=f"std::{family}::{name}",
        handle="result" if constructor else "receiver",
        result="stream_object" if constructor else "void",
    )
    operands = {} if constructor else {"stream": ["receiver"]}
    defaults = {}
    obligations = [
        "defined_execution",
        "stream_object_identity",
        "stream_state_and_exception_mask",
    ]
    if constructor or name == "open":
        if (
            len(params) != 2
            or argc not in {1, 2}
            or type_kind(result) != "VOID"
            or type_kind(params[0]) != "POINTER"
            or lower_type(params[0])
            != {"kind": "ptr", "to": {"kind": "const", "of": "char"}}
            or type_kind(params[1]) != "ENUM"
            or lower_type(params[1]) != _library_type("ios_openmode")
        ):
            return None
        interface = "stream.open"
        operands["path"] = ["args", 0]
        if argc == 2:
            operands["mode"] = ["args", 1]
        else:
            defaults["mode"] = {"kind": "ios_openmode", "flag": "in"}
        contract.update(
            effective_mode="mode_or_in",
            filesystem_effect="depends_on_effective_mode",
            success_condition="filebuf_open_succeeds",
            failure="failbit_or_exception",
        )
        obligations.extend(
            [
                "terminated_path_and_execution_cwd",
                "effective_open_mode",
                "filebuf_open_success",
            ]
        )
    elif name == "close":
        if params or argc or type_kind(result) != "VOID":
            return None
        interface = "stream.close"
        contract.update(
            success_condition="filebuf_close_succeeds", failure="failbit_or_exception"
        )
        obligations.append("associated_filebuf_and_close_status")
    else:
        if (
            argc != len(params)
            or argc not in {2, 3}
            or type_kind(params[0]) != "POINTER"
            or lower_type(params[0]) != {"kind": "ptr", "to": "char"}
            or type_kind(params[1]) not in {"SCHAR", "SHORT", "INT", "LONG", "LONGLONG"}
            or argc == 3
            and lower_type(params[2]) != "char"
            or type_kind(result) != "LVALUEREFERENCE"
            or _stream_family(result.get_pointee(), owns) != "basic_istream"
        ):
            return None
        interface = "stream.getline_into"
        operands.update(dst=["args", 0], count=["args", 1])
        if argc == 3:
            operands["delimiter"] = ["args", 2]
        else:
            defaults["delimiter"] = {"kind": "widen_newline"}
        contract.update(
            result="receiver_reference",
            delimiter="extract_without_storing",
            count_unit="char_elements_including_terminator_space",
            termination="nul_if_count_positive",
            gcount="characters_extracted_including_consumed_delimiter",
            success_condition="input_sentry_and_extraction_state",
        )
        obligations.extend(
            [
                "active_filebuf_association",
                "locale_and_codecvt",
                "destination_capacity_and_count",
            ]
        )
    return dict(
        semantics_version=1,
        interface=interface,
        operands=operands,
        defaults=defaults,
        obligations=obligations,
        contract=contract,
    )


def bind_call(cursor, declaration, expression, lower_type, *, owns):
    """Use the resolved declaration for the interface and preserve explicit argument order."""
    if declaration is None and type_kind(cursor.type.get_canonical()) == "RECORD":
        args = list(cursor.get_arguments())
        ty = lower_type(cursor.type)
        bare = ty.get("of", ty) if isinstance(ty, dict) else {}
        if (
            bare.get("kind") == "library"
            and len(args) == 1
            and not cursor.spelling
            and args[0].type.get_canonical().get_declaration()
            == cursor.type.get_canonical().get_declaration()
        ):
            return {
                "op": "library_value",
                "provider": "cxx_standard",
                "mode": "copy",
                "source_type": lower_type(args[0].type),
                "type": ty,
                "value": expression(args[0]),
            }
    ns = _namespace(declaration)
    iterator_member = (
        ns == ("__gnu_cxx",)
        and declaration.semantic_parent.spelling == "__normal_iterator"
    )
    iterator_operator = (
        ns == ("__gnu_cxx",)
        and declaration.spelling in _OPERATORS
        and any(
            a.type.get_canonical().get_declaration().spelling == "__normal_iterator"
            for a in cursor.get_arguments()
        )
    )
    if ns[:1] != ("std",) and not iterator_member and not iterator_operator:
        return None
    kind, name = cursor_kind(declaration), declaration.spelling
    owner = declaration.semantic_parent
    if kind == "CONSTRUCTOR":
        ty = lower_type(cursor.type)
        if isinstance(ty, dict) and ty.get("kind") == "const":
            ty = ty["of"]
        if not isinstance(ty, dict) or ty.get("kind") != "library":
            raise BindingError(f"library constructor for {owner.displayname}")
        op = {"op": "library_construct", "provider": "cxx_standard", "type": ty}
    elif kind == "CXX_METHOD" and name in _METHODS | _OPERATORS:
        op = {"op": "library_call", "provider": "cxx_standard", "name": name}
    elif kind == "CONVERSION_FUNCTION" and name == "operator bool":
        op = {"op": "library_call", "provider": "cxx_standard", "name": name}
    elif kind == "FUNCTION_DECL" and name in _FREE | _OPERATORS:
        op = {"op": "library_call", "provider": "cxx_standard", "name": name}
    else:
        raise BindingError(
            f"no standard operator binding for {owner.displayname}::{name}"
        )

    # Clang materializes omitted default parameters as expressions with no source
    # location. The bound library supplies those defaults in the target call.
    arguments = [a for a in cursor.get_arguments() if a.location.file]
    receiver = None
    if kind in {"CXX_METHOD", "CONVERSION_FUNCTION"}:
        if declaration.is_static_method():
            raise BindingError(
                f"static standard-library member {name} needs an owning type"
            )
        children = list(cursor.get_children())
        if (
            children
            and cursor_kind(children[0]) == "MEMBER_REF_EXPR"
            and children[0].referenced == declaration
        ):
            bases = [c for c in children[0].get_children() if is_expression(c)]
            if len(bases) != 1:
                raise BindingError(f"member {name} has no explicit receiver")
            receiver = bases[0]
        elif arguments and name in _OPERATORS:
            # Clang includes the receiver as argument zero of an overloaded operator.
            receiver = arguments.pop(0)
        else:
            raise BindingError(f"member {name} has no resolved receiver")
        op["receiver"] = expression(receiver)
        if type_kind(receiver.type.get_canonical()) == "POINTER":
            op["receiver"] = {
                "op": "unary",
                "operator": "deref",
                "value": op["receiver"],
            }
    op["args"] = [expression(arg) for arg in arguments]
    modes = {
        "LVALUEREFERENCE": "reference",
        "RVALUEREFERENCE": "reference",
        "POINTER": "pointer",
        "CONSTANTARRAY": "pointer",
        "INCOMPLETEARRAY": "pointer",
    }
    op["parameter_modes"] = [
        modes.get(type_kind(parameter.type.get_canonical()), "value")
        for parameter in list(declaration.get_arguments())[: len(arguments)]
    ]
    op["effects"] = "unknown"
    effect = _ifstream_effect(declaration, receiver, len(arguments), lower_type, owns)
    if effect is not None:
        op["artifact_effect"] = effect
    return op

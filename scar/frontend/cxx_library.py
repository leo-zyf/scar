"""Bind compiler-resolved standard library declarations to typed operator interfaces."""


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
        if parent.kind.name == "NAMESPACE":
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
        if pointer.kind.name != "POINTER" or not is_standard(owner.get_declaration()):
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
        if argument.kind.name == "INVALID":
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
    elif not (name in _STREAMS | _MANIPULATORS and parent.kind.name == "NAMESPACE"):
        return None
    return {"op": "library_ref", "provider": "cxx_standard", "name": name}


def public_field(declaration):
    return (
        declaration.spelling in {"first", "second"}
        and declaration.semantic_parent.spelling == "pair"
        and _namespace(declaration)[:1] == ("std",)
    )


def bind_call(cursor, declaration, expression, lower_type):
    """Use the resolved declaration for the interface and preserve explicit argument order."""
    if declaration is None and cursor.type.get_canonical().kind.name == "RECORD":
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
    kind, name = declaration.kind.name, declaration.spelling
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
    if kind in {"CXX_METHOD", "CONVERSION_FUNCTION"}:
        if declaration.is_static_method():
            raise BindingError(
                f"static standard-library member {name} needs an owning type"
            )
        children = list(cursor.get_children())
        if (
            children
            and children[0].kind.name == "MEMBER_REF_EXPR"
            and children[0].referenced == declaration
        ):
            bases = [c for c in children[0].get_children() if c.kind.is_expression()]
            if len(bases) != 1:
                raise BindingError(f"member {name} has no explicit receiver")
            receiver = bases[0]
        elif arguments and name in _OPERATORS:
            # Clang includes the receiver as argument zero of an overloaded operator.
            receiver = arguments.pop(0)
        else:
            raise BindingError(f"member {name} has no resolved receiver")
        op["receiver"] = expression(receiver)
        if receiver.type.get_canonical().kind.name == "POINTER":
            op["receiver"] = {
                "op": "unary",
                "operator": "deref",
                "value": op["receiver"],
            }
    op["args"] = [expression(arg) for arg in arguments]
    op["effects"] = "unknown"
    return op

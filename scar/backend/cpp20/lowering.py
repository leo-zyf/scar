"""Lower a planned execution module to typed C++20 translation units."""

from contextlib import contextmanager
from dataclasses import replace
from graphlib import TopologicalSorter
from hashlib import sha256
import math
import re

from .libraries import Libraries
from .model import (
    Decl,
    Expr,
    Stmt,
    Type,
    Unit,
    binary,
    call,
    integer,
    name,
    named,
    template,
)
from .objects import ObjectModel

_SCALARS = {
    "void": "void",
    "unit": "std::nullptr_t",
    "text": "std::string",
    "bool": "bool",
    "char": "char",
    "i8": "std::int8_t",
    "u8": "std::uint8_t",
    "i16": "std::int16_t",
    "u16": "std::uint16_t",
    "i32": "std::int32_t",
    "u32": "std::uint32_t",
    "i64": "std::int64_t",
    "u64": "std::uint64_t",
    "f32": "float",
    "f64": "double",
    "f80": "long double",
}
_ARRAY_EFFECTS = {
    "array_create": {"MEM", "VAL"},
    "array_literal": {"MEM", "VAL"},
    "array_unary": {"MEM", "VAL"},
    "array_copy": {"MEM", "STATE", "XFER"},
    "array_view": {"STATE", "MEM"},
    "array_load": {"STATE"},
    "array_store": {"STATE"},
}
_BINARY = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "mod": "%",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "eq": "==",
    "ne": "!=",
    "land": "&&",
    "lor": "||",
    "band": "&",
    "bor": "|",
    "bxor": "^",
    "shl": "<<",
    "shr": ">>",
    "comma": ",",
}
_UPDATE = {
    op: token + "="
    for op, token in _BINARY.items()
    if op in ("add", "sub", "mul", "div", "mod", "band", "bor", "bxor", "shl", "shr")
}
_UPDATE["set"] = "="
_UNARY = {
    "neg": "-",
    "pos": "+",
    "not": "!",
    "bnot": "~",
    "addr": "&",
    "deref": "*",
    "preinc": "++",
    "predec": "--",
}
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
_FLOAT = re.compile(
    r"[+-]?(?:(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)|"
    r"(?:0[xX](?:[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?|\.[0-9a-fA-F]+)[pP][+-]?\d+))\Z"
)
_KEYWORDS = set(
    """alignas alignof and and_eq asm auto bitand bitor bool break case catch char
char8_t char16_t char32_t class compl concept const consteval constexpr constinit const_cast
continue co_await co_return co_yield decltype default delete do double dynamic_cast else enum
explicit export extern false float for friend goto if inline int long mutable namespace new
noexcept not not_eq nullptr operator or or_eq private protected public register reinterpret_cast
requires return short signed sizeof static static_assert static_cast struct switch template this
thread_local throw true try typedef typeid typename union unsigned using virtual void volatile
wchar_t while xor xor_eq""".split()
)


def _identifier(name, category):
    if not isinstance(name, str) or not name:
        raise ValueError(f"empty {category} identifier")
    if _IDENTIFIER.fullmatch(name) and not name.startswith("_") and "__" not in name:
        return f"{category}_{name}"
    label = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:48] or "symbol"
    key = sha256(name.encode()).hexdigest()[:20]
    return f"{category}x_{label}_{key}"


_UNARY_MATH = set(
    (
        "abs acos acosh asin asinh atan atanh cbrt ceil cos cosh erf erfc exp exp2 "
        "expm1 floor log log10 log1p log2 nearbyint rint round sin sinh sqrt tan tanh trunc"
    ).split()
)
_BINARY_MATH = set(
    "atan2 copysign fdim fmax fmin fmod hypot nextafter pow remainder".split()
)


def unary(operator, value, postfix=False):
    return Expr("unary", operator=operator, args=(value,), postfix=postfix)


def utf8_literal(value, target):
    if not isinstance(value, str):
        raise ValueError("text literal requires a UTF-8 string")
    return Expr(
        "construct",
        type=named(target),
        args=(Expr("string", value=value), integer(len(value.encode("utf-8")))),
        qualifier="paren",
    )


def variable(symbol, ty, init=None, **attributes):
    return Decl("variable", name=symbol, type=ty, initializer=init, **attributes)


def declaration(decl):
    return Stmt("declaration", decl=decl)


def expression_stmt(value):
    return Stmt("expression", value=value)


def cast(ty, operand, operator="static"):
    return Expr("cast", operator=operator, type=ty, args=(operand,))


def alignment(record):
    value = record.get("align_bytes")
    if value is not None and (
        type(value) is not int or value <= 0 or value & (value - 1)
    ):
        raise ValueError(f"invalid ABI alignment for {record['name']!r}")
    return value


def ndarray_rank(spec):
    if (
        not isinstance(spec, dict)
        or set(spec) != {"kind", "element", "rank"}
        or spec["kind"] != "ndarray"
        or spec["element"] != "f64"
        or type(spec["rank"]) is not int
        or spec["rank"] < 0
    ):
        raise ValueError(f"unsupported ndarray type {spec!r}")
    return spec["rank"]


class Lowerer:
    def __init__(self, module, physical):
        self.module = module
        self.regions = physical.get("regions", {})
        self.operator_backends = physical.get("operator_backends", {})
        self.libraries = Libraries()
        self.objects = ObjectModel(_identifier)
        self.receiver = None
        self.headers = self.libraries.headers
        self.used_sdk_headers = set()
        self.openmp = False
        self.map_count = 0
        self.coordinate_maps = 0
        self.buffer_count = 0
        self.layout_scope = {}
        self.axis_names = {}
        self.weight = 0
        self.functions = {}
        self.externs = {f["name"]: f for f in module.get("externs", ())}
        self.globals = {}
        for declaration_ in module.get("globals", ()):
            symbol = declaration_["name"]
            if symbol in self.globals:
                raise ValueError(f"duplicate global {symbol!r}")
            if declaration_.get("external"):
                target = self.libraries.global_value(declaration_)
            elif _IDENTIFIER.fullmatch(symbol) and symbol not in _KEYWORDS:
                target = symbol
            else:
                target = _identifier(symbol, "g")
            self.globals[symbol] = "::" + target.removeprefix("::")
        self.shadowed_globals = set()
        self.methods = {f["name"] for f in module["functions"] if f.get("method")}
        for fn in module["functions"]:
            key = fn["name"]
            if key in self.functions:
                raise ValueError(f"duplicate function {key!r}")
            self.functions[key] = (
                key
                if _IDENTIFIER.fullmatch(key) and key not in _KEYWORDS
                else _identifier(key, "fn")
            )

    @contextmanager
    def lexical_scope(self, bindings=()):
        enclosing = self.shadowed_globals
        self.shadowed_globals = enclosing.union(
            binding for binding in bindings if binding in self.globals
        )
        try:
            yield
        finally:
            self.shadowed_globals = enclosing

    def callee(self, symbol):
        if symbol in self.methods:
            raise ValueError(f"method {symbol!r} requires a receiver")
        if symbol in self.functions:
            return self.functions[symbol]
        if symbol not in self.externs:
            raise ValueError(f"missing external declaration for {symbol!r}")
        return self.libraries.function(self.externs[symbol], self.type_name)

    def type_name(self, spec):
        if isinstance(spec, str):
            if spec in ("c64", "c128"):
                return Type(
                    "template",
                    name="std::complex",
                    args=(named("float" if spec == "c64" else "double"),),
                )
            if spec not in _SCALARS:
                raise ValueError(f"unsupported scalar {spec!r}")
            return named(_SCALARS[spec])
        kind = spec["kind"]
        if kind == "ndarray":
            ndarray_rank(spec)
            if "array.hpp" not in self.used_sdk_headers:
                schedule = self.operator_backends.get("ndarray", {})
                packages = schedule.get("packages", {})
                if (
                    schedule.get("provider") != "xtensor-cpu"
                    or schedule.get("device") != "cpu"
                    or packages.get("xtensor", {}).get("version") != "0.27.1"
                    or packages.get("xtl", {}).get("version") != "0.8.0"
                ):
                    raise ValueError(
                        "ndarray requires a scheduled xtensor CPU provider"
                    )
                self.used_sdk_headers.add("array.hpp")
                self.libraries.links.add("XTensor")
            return Type(
                "template",
                name="scar::ndarray",
                args=(self.type_name(spec["element"]),),
            )
        if kind == "library":
            return self.libraries.cpp_type(spec, self.type_name)
        if kind == "record":
            return named(_identifier(spec["name"], "rec"))
        if kind == "function":
            alias = (
                "scar::variadic_function" if spec.get("variadic") else "scar::function"
            )
            return Type(
                "template",
                name=alias,
                args=(
                    self.type_name(spec["result"]),
                    *map(self.type_name, spec["params"]),
                ),
            )
        if kind == "array":
            ty = self.type_name(spec["of"])
            for size in reversed(spec["shape"]):
                if type(size) is not int or size <= 0:
                    raise ValueError(f"invalid static array extent {size!r}")
                ty = Type("template", name="scar::array", args=(ty, integer(size)))
            return ty
        alias = {
            "const": "std::add_const_t",
            "volatile": "std::add_volatile_t",
            "ptr": "std::add_pointer_t",
            "ref": "std::add_lvalue_reference_t",
        }.get(kind)
        if alias:
            return Type(
                "template",
                name=alias,
                args=(
                    self.type_name(
                        spec["to"] if kind in ("ptr", "ref") else spec["of"]
                    ),
                ),
            )
        raise ValueError(f"unsupported type constructor {kind!r}")

    def literal(self, spec, value):
        if isinstance(spec, dict):
            if spec["kind"] == "ptr" and value in (None, 0):
                return Expr("nullptr")
            if spec["kind"] == "const":
                return self.literal(spec["of"], value)
            raise ValueError(f"invalid literal of type {spec!r}")
        if spec == "unit":
            if value is not None:
                raise ValueError("unit literal requires null")
            return Expr("nullptr")
        if spec == "text":
            return utf8_literal(value, "std::string")
        if spec == "bool":
            if value not in (False, True, 0, 1):
                raise ValueError("invalid Boolean literal")
            return Expr("literal", value=bool(value))
        ty = self.type_name(spec)
        if spec in ("f32", "f64", "f80"):
            token = (
                value.hex()
                if isinstance(value, float) and math.isfinite(value)
                else str(value)
            )
            if not _FLOAT.fullmatch(token):
                special = {
                    "inf": "infinity",
                    "infinity": "infinity",
                    "nan": "quiet_NaN",
                }.get(token.lower().lstrip("+-"))
                if not special:
                    raise ValueError(f"invalid float literal {value!r}")
                result = call(
                    Expr(
                        "type_member",
                        type=Type("template", name="std::numeric_limits", args=(ty,)),
                        name=special,
                    )
                )
                return unary("-", result) if token.startswith("-") else result
            if not any(c in token for c in ".eEpP"):
                token += ".0"
            return Expr(
                "literal", value=token + {"f32": "f", "f64": "", "f80": "L"}[spec]
            )
        if spec in ("c64", "c128"):
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("invalid complex literal")
            part = "f32" if spec == "c64" else "f64"
            return Expr(
                "construct",
                type=ty,
                args=tuple(self.literal(part, v) for v in value),
                qualifier="brace",
            )
        if spec not in _SCALARS or spec == "void":
            raise ValueError(f"unsupported literal type {spec}")
        n = int(value)
        if not -(1 << 63) <= n < 1 << 64:
            raise ValueError("integer literal exceeds ABI range")
        v = (
            binary("-", unary("-", integer("9223372036854775807LL")), integer("1LL"))
            if n == -(1 << 63)
            else integer(str(n) + "ULL" if n >= 1 << 63 else n)
        )
        return cast(ty, v)

    def text_print(self, expr):
        if expr.get("type") != "unit" or expr.get("encoding") != "utf-8":
            raise ValueError("text_print requires unit result and UTF-8 encoding")
        if not {"IO", "STATE"}.issubset(expr.get("effects", ())):
            raise ValueError("text_print lacks observable output effects")
        spellings = ("boolean_true", "boolean_false", "unit")
        fmt = expr.get("scalar_format")
        argument_types = ("text",)
        if fmt is None:
            fmt = dict.fromkeys(spellings, "")
            fmt["integer_base"] = 10
        else:
            if (
                not isinstance(fmt, dict)
                or set(fmt) != {*spellings, "integer_base"}
                or type(fmt["integer_base"]) is not int
                or fmt["integer_base"] != 10
            ):
                raise ValueError("unsupported text_print scalar format")
            argument_types = ("text", "bool", "unit", "i64")
        options = {"sep": "text", "end": "text", "flush": "bool"}
        order = expr.get("keyword_order", tuple(options))
        if (
            not isinstance(order, (list, tuple))
            or any(not isinstance(key, str) or key not in options for key in order)
            or len(set(order)) != len(order)
        ):
            raise ValueError("invalid text_print keyword order")
        operands = []

        def append(role, value, types):
            if not isinstance(value, dict) or value.get("type") not in types:
                raise ValueError(f"unsupported text_print {role} type")
            operands.append(
                Expr(
                    "braced",
                    args=(
                        name("scar::text_role::" + role),
                        cast(self.type_name(value["type"]), self.expression(value)),
                    ),
                )
            )

        for value in expr["args"]:
            append("argument", value, argument_types)
        for key in (*order, *(k for k in options if k not in order)):
            append(key, expr[key], (options[key],))
        # List initialization snapshots each value in order before the call writes.
        formatting = Expr(
            "braced",
            args=(
                *(utf8_literal(fmt[k], "std::string_view") for k in spellings),
                integer(fmt["integer_base"]),
            ),
        )
        output = Expr("braced", args=(Expr("braced", args=tuple(operands)), formatting))
        return call("scar::text_print", output)

    def array_operation(self, node):
        op = node["op"]
        if (
            op not in _ARRAY_EFFECTS
            or set(node.get("effects", ())) != _ARRAY_EFFECTS[op]
        ):
            raise ValueError(f"unsupported ndarray operation or effects for {op}")

        def operands(items, ty):
            if not isinstance(items, (tuple, list)) or any(
                not isinstance(item, dict) or item.get("type") != ty for item in items
            ):
                raise ValueError(f"{op} requires a sequence of {ty} operands")
            return Expr("braced", args=tuple(map(self.expression, items)))

        def group(*parts):
            # C++ list initialization captures operands once in their declared order.
            return Expr("braced", args=parts)

        indexed = op in {"array_view", "array_load", "array_store"}
        ty = node["base"]["type"] if indexed else node["type"]
        rank = ndarray_rank(ty)
        self.type_name(ty)
        elem = self.type_name(ty["element"])
        target = template("scar::" + op, elem)
        if op in {"array_create", "array_literal"}:
            shape = operands(node["shape"], "i64")
            if len(shape.args) != rank:
                raise ValueError(f"{op} shape rank mismatch")
            layout = group(integer(rank), shape)
            if op == "array_create":
                if node.get("initialization") != "zero":
                    raise ValueError("unsupported ndarray initialization")
                return call(target, layout)
            return call(target, group(layout, operands(node["values"], ty["element"])))
        if op in {"array_copy", "array_unary"}:
            if node["value"].get("type") != ty:
                raise ValueError(f"{op} changes dtype or rank")
            args = (self.expression(node["value"]), integer(rank))
            if op == "array_unary":
                operator = node.get("operator")
                if operator not in ("neg", "pos"):
                    raise ValueError("unsupported ndarray unary operator")
                args += (Expr("literal", value=operator == "neg"),)
            return call(target, group(*args))

        indices = operands(node["indices"], "i64")
        remaining = rank - len(indices.args)
        if remaining < 0:
            raise ValueError(f"{op} has too many indices")
        location = group(self.expression(node["base"]), integer(rank), indices)
        if op == "array_view":
            if ndarray_rank(node["type"]) != remaining:
                raise ValueError("ndarray view result rank mismatch")
            return call(target, location)
        if op == "array_load":
            if remaining or node.get("type") != ty["element"]:
                raise ValueError("ndarray scalar access requires every axis")
            return call(target, location)
        rhs = node["value"]
        rhs_type = rhs.get("type")
        rhs_rank = 0 if rhs_type == ty["element"] else ndarray_rank(rhs_type)
        if node.get("broadcasting") != "trailing" or (not remaining and rhs_rank):
            raise ValueError("unsupported ndarray assignment broadcasting contract")
        return call(
            template("scar::array_store", elem, self.type_name(rhs_type)),
            group(self.expression(rhs), integer(rhs_rank), location),
        )

    def expression(self, expr):
        self.weight += 1
        if id(expr) in self.axis_names:
            return self.axis_names[id(expr)]
        op = expr["op"]
        E = self.expression
        if op in {
            "library_construct",
            "library_call",
            "library_ref",
            "library_member",
            "library_value",
            "variadic",
        }:
            return self.libraries.cpp_expression(expr, E, self.type_name)
        if op.startswith("object_") or op == "base_cast":
            return self.objects.expression(expr, self)
        if op == "literal":
            return self.literal(expr["type"], expr["value"])
        if op == "text_print":
            return self.text_print(expr)
        if op.startswith("array_") and op != "array_store":
            return self.array_operation(expr)
        if op == "string":
            return Expr("string", value=expr["value"])
        if op == "runtime_array_size":
            symbol = expr["name"]
            return binary(
                "*",
                Expr("sizeof_expr", args=(unary("*", name(_identifier(symbol, "v"))),)),
                name(_identifier(symbol, "stack_count")),
            )
        if op == "fp_traps_masked":
            return call("scar::fp_traps_masked")
        if op == "function_ref":
            return name(self.callee(expr["name"]))
        if op == "var":
            symbol, scope = expr["name"], expr.get("scope")
            if scope == "global" or (
                scope is None
                and symbol in self.globals
                and symbol not in self.shadowed_globals
            ):
                if symbol not in self.globals:
                    raise ValueError(
                        f"global reference lacks a declaration for {symbol!r}"
                    )
                return name(self.globals[symbol])
            if self.receiver is not None and symbol == self.receiver:
                return name("this")
            return name(_identifier(symbol, "v"))
        if op == "binary":
            return binary(_BINARY[expr["operator"]], E(expr["left"]), E(expr["right"]))
        if op == "unary":
            operator = expr["operator"]
            return unary(
                {"postinc": "++", "postdec": "--"}.get(operator, _UNARY.get(operator)),
                E(expr["value"]),
                operator in ("postinc", "postdec"),
            )
        if op == "update":
            return binary(
                _UPDATE[expr["operator"]], E(expr["target"]), E(expr["value"])
            )
        if op == "indirect_call":
            return call(E(expr["target"]), *map(E, expr["args"]))
        if op == "allocate":
            init = expr.get("init")
            args = (
                tuple(map(E, init["items"]))
                if init and init["op"] == "init_list"
                else (E(init),) if init else ()
            )
            return Expr(
                "new",
                type=self.type_name(expr["type"]),
                extent=E(expr["count"]) if expr.get("count") is not None else None,
                args=args,
                qualifier=(
                    ("brace" if init["op"] == "init_list" else "paren")
                    if init
                    else "default"
                ),
            )
        if op == "release":
            return Expr("delete", args=(E(expr["target"]),), array=expr["array"])
        if op == "intrinsic":
            return self.intrinsic(expr)
        if op == "load_bytes":
            return call(
                template("scar::load_bytes", self.type_name(expr["type"])),
                E(expr["base"]),
                E(expr["offset"]),
            )
        if op == "index":
            return Expr("subscript", args=(E(expr["base"]), *map(E, expr["indices"])))
        if op == "cast":
            mode = expr.get("mode", "static")
            ty = self.type_name(expr["type"])
            v = E(expr["value"])
            if mode == "explicit_pointer":
                return call(template("scar::pointer_cast", ty), v)
            target = {
                "static": "static",
                "qualification": "const",
                "reinterpret": "reinterpret",
            }.get(mode)
            if target is None:
                raise ValueError(f"unsupported cast {mode!r}")
            return cast(ty, v, target)
        if op == "call":
            symbol = expr["callee"]
            target = self.callee(symbol)
            if symbol not in self.functions:
                target = (
                    self.libraries.format_operator(self.externs[symbol], expr["args"])
                    or target
                )
            return call(target, *map(E, expr["args"]))
        if op == "init_list":
            return Expr("braced", args=tuple(map(E, expr["items"])))
        if op == "select":
            return Expr(
                "conditional",
                args=(E(expr["condition"]), E(expr["then"]), E(expr["else"])),
            )
        if op == "member":
            field = (
                self.libraries.field(expr["provider"], expr["record"], expr["field"])
                if expr.get("provider")
                else _identifier(expr["field"], "m")
            )
            return Expr("member", args=(E(expr["base"]),), name=field)
        raise ValueError(f"unsupported expression {op!r} at {expr.get('id','?')}")

    def intrinsic(self, expr):
        intrinsic = expr["name"]
        args = tuple(map(self.expression, expr["args"]))
        ty = self.type_name(expr["type"])
        n = len(args)
        if intrinsic in ("min", "max") and n >= 2:
            return call(template("std::" + intrinsic, ty), Expr("braced", args=args))
        if intrinsic in ("powi", "sign", "mod", "modulo") and n == 2:
            target = "scar::" + intrinsic
        elif intrinsic == "complex" and n in (1, 2):
            return Expr("construct", type=ty, args=args, qualifier="paren")
        elif intrinsic in ("real", "imag", "conjg") and n == 1:
            target = "std::" + {"conjg": "conj"}.get(intrinsic, intrinsic)
        else:
            spelling = {"ceiling": "ceil", "nint": "round"}.get(intrinsic, intrinsic)
            if spelling == "atan" and n == 2:
                spelling = "atan2"
            if not (
                spelling in _UNARY_MATH
                and n == 1
                or spelling in _BINARY_MATH
                and n == 2
            ):
                raise ValueError(f"unsupported intrinsic {intrinsic}/{n}")
            target = "std::" + spelling
        return cast(ty, call(target, *args))

    def variable_decl(self, stmt, symbol=None):
        storage = stmt.get("storage")
        if storage not in (None, "auto", "static", "thread_local"):
            raise ValueError(f"unsupported storage {storage!r}")
        return variable(
            symbol or _identifier(stmt["name"], "v"),
            self.type_name(stmt["type"]),
            self.expression(stmt["init"]) if stmt.get("init") is not None else None,
            alignment=alignment(stmt),
            storage=(storage,) if storage in ("static", "thread_local") else (),
        )

    def statements(self, items, function=None, bindings=()):
        with self.lexical_scope(bindings):
            return self.statements_in_scope(items, function)

    def statements_in_scope(self, items, function=None):
        tree = []
        for i, stmt in enumerate(items):
            op = stmt["op"]
            if op == "runtime_array":
                tree.extend(self.scoped_array(stmt, items[i + 1 :], function))
                break
            if op == "decl" and stmt["name"] in self.globals:
                self.shadowed_globals.add(stmt["name"])
            tree.extend(self.statement(stmt))
            if op == "buffer" and stmt["name"] in self.globals:
                self.shadowed_globals.add(stmt["name"])
        return tuple(tree)

    def scoped_array(self, stmt, suffix, function):
        if (
            stmt.get("storage") != "scoped"
            or stmt.get("initialization") != "uninitialized"
        ):
            raise ValueError("invalid scoped array contract")
        if (
            stmt.get("scope_control") != "lexical_continuation"
            or stmt.get("allocation") != "scoped_stack"
        ):
            raise ValueError(
                "scoped array requires an approved lexical lifetime and target stack assignment"
            )
        ty = self.type_name(stmt["element"])
        cnt = _identifier(stmt["name"], "stack_count")
        extent = declaration(
            variable(
                cnt,
                Type("const", element=named("std::size_t")),
                cast(named("std::size_t"), self.expression(stmt["extent"])),
            )
        )
        result = self.type_name(function["result"]) if function else named("void")
        code = self.statements(suffix, function, bindings=(stmt["name"],))
        if function and function["name"] == "main":
            code += (Stmt("return", value=integer(0)),)
        argument = Decl(
            "param",
            name=_identifier(stmt["name"], "v"),
            type=Type("const", element=Type("ptr", element=ty)),
        )
        action = Expr(
            "lambda", captures=("&",), params=(argument,), type=result, body=code
        )
        use = call(template("scar::with_stack_array", ty), name(cnt), action)
        return [
            extent,
            Stmt("return" if result != named("void") else "expression", value=use),
        ]

    def statement(self, stmt):
        self.weight += 1
        op = stmt["op"]
        E = self.expression
        if op == "array_store":
            return [expression_stmt(self.array_operation(stmt))]
        if op == "decl":
            return [declaration(self.variable_decl(stmt))]
        if op == "buffer":
            self.headers.add("memory")
            self.buffer_count += 1
            owner = f"scar_buffer_{self.buffer_count}"
            ty = self.type_name(stmt["type"])
            owner_type = Type(
                "template", name="std::unique_ptr", args=(Type("array", element=ty),)
            )
            allocation = Expr(
                "new", type=ty, extent=E(stmt["extent"]), qualifier="default"
            )
            box = Expr(
                "construct", type=owner_type, args=(allocation,), qualifier="paren"
            )
            raw = call(Expr("member", args=(name(owner),), name="get"))
            return [
                declaration(variable(owner, owner_type, box)),
                declaration(
                    variable(
                        _identifier(stmt["name"], "v"),
                        Type("const", element=Type("ptr", element=ty)),
                        raw,
                    )
                ),
            ]
        if op in ("label", "goto"):
            return [Stmt(op, name=_identifier(stmt["name"], "label"))]
        if op in ("case", "default", "return"):
            return [
                Stmt(
                    op,
                    value=E(stmt["value"]) if stmt.get("value") is not None else None,
                )
            ]
        if op in ("break", "continue"):
            return [Stmt(op)]
        if op == "assign":
            return [expression_stmt(binary("=", E(stmt["target"]), E(stmt["value"])))]
        if op == "store_bytes":
            return [
                expression_stmt(
                    call(
                        template("scar::store_bytes", self.type_name(stmt["type"])),
                        E(stmt["base"]),
                        E(stmt["offset"]),
                        E(stmt["value"]),
                    )
                )
            ]
        if op == "release":
            return [expression_stmt(E(stmt))]
        if op == "expr":
            val = stmt["value"]
            v = E(val)
            external = (
                self.externs.get(val.get("callee")) if val["op"] == "call" else None
            )
            if external and external["result"] != "void":
                self.headers.add("tuple")
                v = binary("=", name("std::ignore"), v)
            return [expression_stmt(v)]
        if op == "map":
            return [self.map_statement(stmt)]
        if op == "for":
            with self.lexical_scope():
                setup = self.statements_in_scope(stmt["init"])
                loop = Stmt(
                    "for",
                    condition=(
                        E(stmt["condition"])
                        if stmt.get("condition") is not None
                        else None
                    ),
                    step=E(stmt["step"]) if stmt.get("step") is not None else None,
                    body=self.statements(stmt["body"]),
                )
            return [Stmt("block", body=(*setup, loop))]
        if op == "each":
            return [
                Stmt(
                    "each",
                    decl=Decl(
                        "param",
                        name=_identifier(stmt["name"], "v"),
                        type=self.type_name(stmt["type"]),
                    ),
                    value=E(stmt["range"]),
                    body=self.statements(stmt["body"], bindings=(stmt["name"],)),
                )
            ]
        if op == "if":
            return [
                Stmt(
                    "if",
                    condition=E(stmt["condition"]),
                    body=self.statements(stmt["then"]),
                    else_body=self.statements(stmt.get("else", ())),
                )
            ]
        if op in ("while", "do_while", "switch", "block"):
            return [
                Stmt(
                    op,
                    condition=E(stmt["condition"]) if op != "block" else None,
                    body=self.statements(stmt["body"]),
                )
            ]
        raise ValueError(f"unsupported statement {op!r} at {stmt.get('id','?')}")

    def count_loop(self, index, extent, body, step=None):
        return Stmt(
            "for",
            init=declaration(variable(index, named("std::int64_t"), integer(0))),
            condition=binary("<", name(index), extent),
            step=step or unary("++", name(index)),
            body=body,
        )

    def map_statement(self, stmt):
        self.map_count += 1
        number = self.map_count
        limit = f"scar_extent_{number}"
        index = _identifier(stmt["index"], "v")
        size = name(limit)
        region = self.regions.get(stmt.get("site"), {})
        bands = region.get("widths", [dict(minimum_elements=0, workers=1)])
        if not bands or bands[0]["minimum_elements"] != 0:
            raise ValueError("map schedule must cover an empty domain")
        width = integer(1)
        previous = -1
        for band in bands:
            lower, workers = band["minimum_elements"], band["workers"]
            if (
                type(lower) is not int
                or lower <= previous
                or type(workers) is not int
                or workers < 1
            ):
                raise ValueError("invalid map schedule")
            width = (
                integer(workers)
                if lower == 0
                else Expr(
                    "conditional",
                    args=(binary(">=", size, integer(lower)), integer(workers), width),
                )
            )
            previous = lower
        body = self.statements(stmt["body"], bindings=(stmt["index"],))
        start = declaration(
            variable(
                limit,
                Type("const", element=named("std::int64_t")),
                self.expression(stmt["extent"]),
            )
        )
        linear = self.count_loop(index, size, body)
        selected = region.get("serial_layout", "linear")
        serial = (
            linear
            if selected == "linear"
            else self.coordinate_map(stmt, region, index, size, number, linear)
        )
        if any(b["workers"] > 1 for b in bands):
            if stmt.get("independent") is not True:
                raise ValueError(
                    "a parallel physical schedule requires an independent map contract"
                )
            self.openmp = True
            action = Expr(
                "lambda",
                captures=("&",),
                params=(Decl("param", name=index, type=named("std::int64_t")),),
                body=body,
            )
            args = (size, width, action)
            if selected != "linear":
                args += (Expr("lambda", captures=("&",), body=(serial,)),)
            serial = expression_stmt(call("scar::map", *args))
        return Stmt("block", body=(start, serial))

    def coordinate_map(self, stmt, region, index, size, number, fallback):
        layout = stmt.get("iteration_layout", {})
        if (
            region["serial_layout"] != "mixed_radix"
            or layout.get("kind") != "mixed_radix"
            or layout.get("order") != "first_axis_fastest"
            or layout.get("fallback") != "linear"
        ):
            raise ValueError("selected iteration layout has no matching IR contract")
        minimum = region["minimum_coordinate_elements"]
        if type(minimum) is not int or minimum < 1:
            raise ValueError("invalid coordinate threshold")
        counts = tuple(map(self.expression, layout["axes"]))
        if not counts or not layout["coordinates"]:
            raise ValueError("empty coordinate traversal")
        condition = binary(">=", size, integer(minimum))
        for count in counts:
            condition = binary("&&", condition, binary(">", count, integer(0)))
        for guard in layout["equalities"]:
            condition = binary(
                "&&",
                condition,
                binary("==", counts[guard["axis"]], self.expression(guard["radix"])),
            )
        symbols = (
            [f"scar_axis_{number}_{i}" for i in range(len(counts))]
            if len(counts) > 1
            else [index]
        )
        before = self.axis_names
        substitutions = {}
        for record in layout["coordinates"]:
            node = stmt["body"]
            for field in record["path"]:
                node = node[field]
            axis = record["axis"]
            if (
                type(axis) is not int
                or not 0 <= axis < len(symbols)
                or not isinstance(node, dict)
                or "op" not in node
            ):
                raise ValueError("invalid coordinate substitution")
            substitutions[id(node)] = name(symbols[axis])
        self.axis_names = substitutions
        try:
            body = self.statements(stmt["body"], bindings=(stmt["index"],))
        finally:
            self.axis_names = before
        for axis, (symbol, count) in enumerate(zip(symbols, counts)):
            inc = unary("++", name(symbol))
            if axis == 0 and len(counts) > 1:
                inc = binary(",", inc, unary("++", name(index)))
            body = (self.count_loop(symbol, count, body, inc),)
        if len(counts) > 1:
            body = (
                declaration(variable(index, named("std::int64_t"), integer(0))),
                *body,
            )
        self.coordinate_maps += 1
        return Stmt("if", condition=condition, body=body, else_body=(fallback,))

    def signature(self, function):
        params = tuple(
            Decl(
                "param",
                name=_identifier(p["name"], "v"),
                type=self.type_name(p["type"]),
            )
            for p in function["params"]
        )
        if function.get("variadic"):
            params += (Decl("param", definition="variadic"),)
        return Decl(
            "function",
            name=self.functions[function["name"]],
            type=self.type_name(function["result"]),
            params=params,
            linkage=(
                None
                if function["name"] == "main"
                or function["result"] == "text"
                or (
                    isinstance(function["result"], dict)
                    and function["result"].get("kind") == "ndarray"
                )
                else "C"
            ),
        )

    def entry_wrapper(self):
        entry = self.module["entry"]
        abi = self.module.get("entry_abi")
        symbol = self.functions.get(entry)
        if not symbol:
            raise ValueError(f"missing entry {entry!r}")
        fn = next(f for f in self.module["functions"] if f["name"] == entry)
        if entry == "main":
            if abi:
                raise ValueError("main may not also declare an entry adapter")
            return ()
        if "main" in self.functions:
            raise ValueError("entry conflicts with main")
        params = ()
        if abi is None:
            if fn["params"] or fn["result"] not in (
                "i32",
                "i16",
                "i8",
                "bool",
                "void",
                "unit",
            ):
                raise ValueError("entry requires explicit ABI")
            invoke = call(symbol)
            body = (
                Stmt(
                    "expression" if fn["result"] in ("void", "unit") else "return",
                    value=invoke,
                ),
            )
        else:
            if abi.get("kind") != "binary_f64_arrays":
                raise ValueError("unsupported executable ABI")
            shape, inputs, count = abi["shape"], abi["inputs"], abi["count"]
            if (
                not isinstance(shape, list)
                or any(type(n) is not int or n < 0 for n in shape)
                or type(inputs) is not int
                or inputs < 1
                or type(count) is not int
                or not 0 <= count < 1 << 63
            ):
                raise ValueError("invalid array ABI")
            n = math.prod(shape)
            if n * 8 * inputs >= 1 << 63:
                raise ValueError("array input exceeds stream offset range")
            read = {"kind": "ptr", "to": {"kind": "const", "of": "f64"}}
            write = {"kind": "ptr", "to": "f64"}
            if [p["type"] for p in fn["params"]] != [
                "i64",
                "i64",
                *[read] * inputs,
                write,
            ] or fn["result"] != "void":
                raise ValueError("array entry signature differs from ABI")
            params = (
                Decl("param", type=named("int"), name="argc"),
                Decl(
                    "param",
                    type=Type("ptr", element=Type("ptr", element=named("char"))),
                    name="argv",
                ),
            )
            invoke = call(
                template(
                    "scar::binary_f64_entry",
                    integer(n),
                    integer(count),
                    integer(inputs),
                ),
                name("argc"),
                name("argv"),
                name(symbol),
            )
            body = (Stmt("return", value=invoke),)
        return (
            Decl("function", name="main", type=named("int"), params=params, body=body),
        )

    def record_declarations(self):
        forwards = []
        definitions = []
        records = {r["name"]: r for r in self.module.get("records", ())}
        if len(records) != len(self.module.get("records", ())):
            raise ValueError("duplicate record")
        order = TopologicalSorter()
        for key, record in records.items():
            tag = _identifier(key, "rec")
            kind = record.get("kind", "struct")
            keyword = "struct" if kind in ("class", "struct", "record") else kind
            if keyword not in ("struct", "union") or record.get("packed"):
                raise ValueError(f"unsupported record layout for {key}")
            forward = Decl("forward", name=tag, definition=keyword)
            if record.get("opaque") or record.get("provider"):
                target = self.libraries.opaque(record)
                forwards.append(
                    Decl("alias", name=tag, type=target) if target else forward
                )
                if target:
                    definitions.extend(self.layout_checks(record))
                continue
            forwards.append(forward)
            deps = {base["name"] for base in record.get("bases", ())}
            for field in record["fields"]:
                ty = field["type"]
                while isinstance(ty, dict) and ty["kind"] in (
                    "array",
                    "const",
                    "volatile",
                ):
                    ty = ty["of"]
                if isinstance(ty, dict) and ty["kind"] == "record":
                    deps.add(ty["name"])
            if any(
                dep not in records
                or records[dep].get("opaque")
                and not records[dep].get("complete")
                for dep in deps
            ):
                raise ValueError(f"incomplete field of {key}")
            order.add(
                key, *(dep for dep in sorted(deps) if not records[dep].get("provider"))
            )
        for key in order.static_order():
            record = records[key]
            self.receiver = "this"
            fields = tuple(
                Decl(
                    "field",
                    name=_identifier(f["name"], "m"),
                    type=self.type_name(f["type"]),
                    initializer=(
                        self.expression(f["init"])
                        if f.get("init") is not None
                        else None
                    ),
                )
                for f in record["fields"]
            )
            self.receiver = None
            definitions.append(
                Decl(
                    "record",
                    name=_identifier(key, "rec"),
                    definition="union" if record.get("kind") == "union" else "struct",
                    bases=self.objects.bases(record),
                    members=fields + self.objects.declarations(record, self),
                    alignment=alignment(record),
                )
            )
            definitions.extend(self.layout_checks(record))
        return forwards, definitions

    def layout_checks(self, record):
        tag = named(_identifier(record["name"], "rec"))
        checks = []
        for attribute, form in (
            ("size_bytes", "sizeof_type"),
            ("align_bytes", "alignof"),
        ):
            if record.get(attribute) is not None:
                checks.append(
                    Decl(
                        "static_assert",
                        initializer=binary(
                            "==", Expr(form, type=tag), integer(int(record[attribute]))
                        ),
                    )
                )
        fields = [f for f in record.get("fields", ()) if "offset_bits" in f]
        if not fields:
            return checks
        required = bool(
            record.get("provider")
            or record.get("requires_byte_layout")
            or record.get("standard_layout") is True
        )
        self.layout_scope[record["name"]] = (
            "required" if required else "when_standard_layout"
        )
        if required:
            checks.append(
                Decl(
                    "static_assert",
                    initializer=template("std::is_standard_layout_v", tag),
                )
            )
        for field in fields:
            m = (
                self.libraries.field(
                    record["provider"], record["symbol"], field["name"]
                )
                if record.get("provider")
                else _identifier(field["name"], "m")
            )
            checks.append(
                Decl(
                    "static_assert",
                    initializer=call(
                        "SCAR_OFFSET_MATCH",
                        tag,
                        name(m),
                        integer(int(field["offset_bits"])),
                    ),
                )
            )
        return checks

    def units(self):
        forwards, records = self.record_declarations()
        prototypes = []
        globals_ = []
        parts = []
        for decl in self.module.get("globals", ()):
            if decl.get("external"):
                continue
            symbol = self.globals[decl["name"]].removeprefix("::")
            storage = decl.get("storage")
            if storage not in (None, "auto", "static", "thread_local"):
                raise ValueError(f"unsupported global duration {storage}")
            duration = ("thread_local",) if storage == "thread_local" else ()
            globals_.append(
                replace(
                    variable(
                        symbol,
                        self.type_name(decl["type"]),
                        alignment=alignment(decl),
                        storage=("extern", *duration),
                    ),
                    linkage="C",
                )
            )
            parts.append(
                replace(
                    self.variable_decl(
                        {**decl, "storage": duration[0] if duration else None}, symbol
                    ),
                    linkage="C",
                )
            )
        parts.extend(self.entry_wrapper())
        groups = {}
        group_weight = 0

        def flush():
            target = f"src/unit_{len(groups):04}.cpp" if groups else "src/program.cpp"
            groups[target] = Unit(
                includes=(("local", "scar/program.hpp"),), declarations=tuple(parts)
            )
            parts.clear()

        functions = self.module["functions"]
        entry = self.module["entry"]
        ordered = (
            fn
            for first in (True, False)
            for fn in functions
            if (fn["name"] == entry) == first
        )
        for fn in ordered:
            method = fn.get("method", {})
            if any(method.get(k) for k in ("deleted", "defaulted", "implicit")):
                if fn["body"] or fn.get("initializers"):
                    raise ValueError("compiler-defined method with explicit body")
                continue
            self.weight = 0
            with self.lexical_scope(param["name"] for param in fn["params"]):
                if method:
                    self.receiver = fn["params"][0]["name"]
                    decl = self.objects.method(method, self, definition=True)
                    decl = replace(
                        decl, initializers=self.objects.initializers(fn, self)
                    )
                else:
                    self.receiver = None
                    decl = self.signature(fn)
                    prototypes.append(decl)
                definition = replace(
                    decl, body=self.statements_in_scope(fn["body"], fn)
                )
            if parts and group_weight + self.weight > 4000:
                flush()
                group_weight = 0
            parts.append(definition)
            group_weight += self.weight
        if parts or not groups:
            flush()
        providers = tuple(
            item
            for interface in self.libraries.declarations.values()
            for item in interface
        )
        headers = (
            ("local", "scar/runtime.hpp"),
            *(("local", "scar/" + h) for h in sorted(self.used_sdk_headers)),
            *(("system", h) for h in sorted(self.headers)),
        )
        groups["include/scar/program.hpp"] = Unit(
            includes=headers,
            declarations=(*forwards, *providers, *prototypes, *globals_, *records),
            once=True,
        )
        return groups

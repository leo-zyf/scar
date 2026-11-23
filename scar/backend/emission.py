"""Lower typed SCAR operations and a physical schedule to an independent C++20 project."""

from graphlib import TopologicalSorter
from hashlib import sha256
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from .libraries import Libraries
from .object_model import ObjectModel
from .project import build

_SCALARS = {
    "void": "void",
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
    "c64": "std::complex<float>",
    "c128": "std::complex<double>",
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


def _alignment(decl):
    if "align_bytes" not in decl:
        return ""
    alignment = decl["align_bytes"]
    if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError(f"invalid ABI alignment for {decl['name']!r}")
    return f"alignas({alignment}) "


def _type(spec, libraries=None):
    if isinstance(spec, str):
        if spec not in _SCALARS:
            raise ValueError(f"unsupported IR scalar type {spec!r}")
        return _SCALARS[spec]
    kind = spec["kind"]
    if kind == "library" and libraries is not None:
        return libraries.cpp_type(spec, lambda t: _type(t, libraries))
    if kind == "record":
        return _identifier(spec["name"], "rec")
    if kind == "function":
        result = _type(spec["result"], libraries)
        parameters = [_type(t, libraries) for t in spec["params"]]
        alias = "variadic_function" if spec.get("variadic") else "function"
        return f"scar::{alias}<{', '.join([result, *parameters])}>"
    if kind == "array":
        result = _type(spec["of"], libraries)
        for extent in reversed(spec["shape"]):
            if type(extent) is not int or extent <= 0:
                raise ValueError(
                    f"static array dimension must be positive, got {extent!r}"
                )
            result = f"scar::array<{result}, {extent}>"
        return result
    if kind in ("const", "volatile"):
        return f"std::add_{kind}_t<{_type(spec['of'], libraries)}>"
    if kind in ("ptr", "ref"):
        trait = "add_pointer_t" if kind == "ptr" else "add_lvalue_reference_t"
        return f"std::{trait}<{_type(spec['to'], libraries)}>"
    raise ValueError(f"unsupported IR type constructor {kind!r}")


def _string(value):
    escapes = {9: r"\t", 10: r"\n", 13: r"\r", 34: r"\"", 92: r"\\"}
    parts = []
    for byte in value.encode("utf-8"):
        if byte in escapes:
            parts.append(escapes[byte])
        elif 32 <= byte < 127:
            parts.append(chr(byte))
        else:
            parts.append(f"\\{byte:03o}")
    return '"' + "".join(parts) + '"'


def _literal(expr):
    spec, value = expr["type"], expr["value"]
    if isinstance(spec, dict):
        if spec["kind"] == "ptr" and value in (None, 0):
            return "nullptr"
        if spec["kind"] == "const":
            return _literal({**expr, "type": spec["of"]})
        raise ValueError(f"invalid literal of type {spec!r}")
    if spec == "bool":
        if value not in (False, True, 0, 1):
            raise ValueError(f"invalid Boolean literal {value!r}")
        return "true" if value else "false"
    if spec in ("f32", "f64", "f80"):
        token = (
            value.hex()
            if isinstance(value, float) and math.isfinite(value)
            else str(value)
        )
        if not _FLOAT.fullmatch(token):
            special = {"inf": "infinity", "infinity": "infinity", "nan": "quiet_NaN"}
            spelling = token.lower().lstrip("+-")
            if spelling not in special:
                raise ValueError(f"invalid floating-point literal {value!r}")
            sign = "-" if token.startswith("-") else ""
            return f"{sign}std::numeric_limits<{_type(spec)}>::{special[spelling]}()"
        if not any(c in token for c in ".eEpP"):
            token += ".0"
        return token + {"f32": "f", "f64": "", "f80": "L"}[spec]
    if spec in ("c64", "c128"):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("complex constant requires real and imaginary components")
        scalar = "f32" if spec == "c64" else "f64"
        parts = [_literal({"value": part, "type": scalar}) for part in value]
        return f"{_type(spec)}{{{', '.join(parts)}}}"
    if spec not in _SCALARS or spec == "void":
        raise ValueError(f"unsupported literal type {spec!r}")
    number = int(value)
    if not -(1 << 63) <= number < (1 << 64):
        raise ValueError(f"integer literal exceeds supported width: {number}")
    if number == -(1 << 63):
        token = "(-9223372036854775807LL - 1LL)"
    else:
        token = str(number) + ("ULL" if number >= (1 << 63) else "")
    return f"static_cast<{_type(spec)}>({token})"


_UNARY_MATH = set(
    (
        "abs acos acosh asin asinh atan atanh cbrt ceil cos cosh erf erfc exp exp2 "
        "expm1 floor log log10 log1p log2 nearbyint rint round sin sinh sqrt tan tanh trunc"
    ).split()
)
_BINARY_MATH = set(
    "atan2 copysign fdim fmax fmin fmod hypot nextafter pow remainder".split()
)
_TARGET_HELPERS = {
    "stack_array": """template<class T, class F>
[[gnu::noinline]] decltype(auto) with_stack_array(std::size_t count, F action) {
    static_assert(std::is_trivially_default_constructible_v<T>
                  && std::is_trivially_destructible_v<T>);
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
        throw std::bad_array_new_length();
    const auto bytes = (count ? count : 1) * sizeof(T);
    void* memory = __builtin_alloca_with_align(bytes, alignof(T) * CHAR_BIT);
    return action(::new(memory) T[count]);
}""",
    "byte_access": """template<class T> T load_bytes(const void* base, std::ptrdiff_t offset) {
    static_assert(std::is_trivially_copyable_v<T>);
    T value;
    std::memcpy(&value, static_cast<const std::byte*>(base) + offset, sizeof(T));
    return value;
}
template<class T> void store_bytes(void* base, std::ptrdiff_t offset, T value) {
    static_assert(std::is_trivially_copyable_v<T>);
    std::memcpy(static_cast<std::byte*>(base) + offset, &value, sizeof(T));
}""",
    "pointer_cast": """template<class T, class U> T pointer_cast(U value) {
    static_assert(std::is_pointer_v<T>);
    if constexpr (requires { static_cast<T>(value); }) return static_cast<T>(value);
    else if constexpr (requires { const_cast<T>(value); }) return const_cast<T>(value);
    else if constexpr (requires { reinterpret_cast<T>(value); }) return reinterpret_cast<T>(value);
    else return static_cast<T>(const_cast<void*>(static_cast<const volatile void*>(value)));
}""",
    "powi": """template<class T, class E> T powi(T base, E exponent) {
    static_assert(std::is_integral_v<E>);
    const bool negative = exponent < 0;
    if constexpr (std::is_integral_v<T>) {
        if (negative && base != T{1} && base != static_cast<T>(-1)) return T{0};
    }
    using U = std::make_unsigned_t<E>;
    U power = static_cast<U>(exponent);
    if (negative) power = U{0} - power;
    T result = (power & U{1}) ? base : T{1};
    while ((power >>= 1) != 0) {
        base *= base;
        if (power & U{1}) result *= base;
    }
    return negative ? T{1} / result : result;
}""",
    "sign": """template<class T> T sign(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::copysign(std::abs(a), b);
    } else {
        using U = std::make_unsigned_t<T>;
        const U bits = static_cast<U>(a);
        const U magnitude = a < 0 ? U{0} - bits : bits;
        return static_cast<T>(b < 0 ? U{0} - magnitude : magnitude);
    }
}""",
    "mod": """template<class T> T mod(T a, T b) {
    if constexpr (std::is_integral_v<T>) return a % b;
    else return std::fmod(a, b);
}""",
    "modulo": """template<class T> T modulo(T a, T b) {
    T remainder;
    if constexpr (std::is_integral_v<T>) remainder = a % b;
    else {
        remainder = std::fmod(a, b);
        if (remainder == 0) return std::copysign(T{0}, b);
    }
    if (remainder != 0 && ((remainder < 0) != (b < 0))) remainder += b;
    return remainder;
}""",
}


_FP_TRAPS = """inline bool fp_traps_masked() {
#if defined(__GLIBC__) && defined(_GNU_SOURCE)
    return ::fegetexcept() == 0;
#else
    return false;
#endif
}"""

_PARALLEL_MAP = """inline bool team_available(int workers) {
    if (omp_in_parallel() || workers > omp_get_max_threads() || workers > omp_get_thread_limit()) {
        return false;
    }
    cpu_set_t allowed;
    return sched_getaffinity(0, sizeof(allowed), &allowed) == 0 && workers <= CPU_COUNT(&allowed);
}
template<class F> inline void map(std::int64_t n, int workers, F body) {
    std::fenv_t initial;
    if (workers <= 1 || workers > n || !team_available(workers) || !fp_traps_masked() ||
        std::fegetenv(&initial) != 0) {
        for (std::int64_t i = 0; i < n; ++i) body(i);
        return;
    }
    const int incoming = std::fetestexcept(FE_ALL_EXCEPT);
    int raised = 0;
    #pragma omp parallel num_threads(workers) reduction(|:raised)
    {
        std::fenv_t previous;
        std::fegetenv(&previous);
        std::fesetenv(&initial);
        #pragma omp for schedule(static)
        for (std::int64_t i = 0; i < n; ++i) body(i);
        raised |= std::fetestexcept(FE_ALL_EXCEPT);
        std::fesetenv(&previous);
    }
    raised &= ~incoming;
    if (raised != 0) {
        // Transfer only flags actually raised by a worker. feraiseexcept may also
        // set inexact, so construct the desired flag subset in a held environment.
        std::fenv_t saved;
        std::fexcept_t flags;
        std::feholdexcept(&saved);
        std::feraiseexcept(raised);
        std::fegetexceptflag(&flags, raised);
        std::fesetenv(&saved);
        std::fesetexceptflag(&flags, raised);
    }
}"""


class _Emitter:
    def __init__(self, module, physical):
        self.module = module
        self.regions = physical.get("regions", {})
        self.workers = physical["workers"]
        self.threshold = physical["minimum_parallel_elements"]
        if type(self.workers) is not int or self.workers < 1:
            raise ValueError("physical worker count must be positive")
        if self.threshold is not None and (
            type(self.threshold) is not int or self.threshold < 0
        ):
            raise ValueError("physical parallel extent threshold must be nonnegative")
        self.libraries = Libraries()
        self.objects = ObjectModel(module, names=_identifier)
        self.receiver = None
        self.headers = self.libraries.headers
        self.openmp = False
        self.map_count = 0
        self.buffer_count = 0
        self.helpers = set()
        self.uses_fenv_guard = False
        self.layout_scope = {}
        self.functions = {}
        self.member_functions = {
            fn["name"] for fn in module["functions"] if fn.get("method")
        }
        self.externs = {f["name"]: f for f in module.get("externs", ())}
        for function in module["functions"]:
            name = function["name"]
            if name in self.functions:
                raise ValueError(f"duplicate function definition {name!r}")
            # Plain link symbols retain their ABI. A non-identifier private symbol
            # (for example, a translation-unit-local function) gets a stable name.
            self.functions[name] = (
                name
                if _IDENTIFIER.fullmatch(name) and name not in _KEYWORDS
                else _identifier(name, "fn")
            )

    def callee(self, symbol):
        if symbol in self.member_functions:
            raise ValueError(f"member procedure {symbol!r} requires an object receiver")
        if symbol in self.functions:
            return self.functions[symbol]
        if symbol not in self.externs:
            raise ValueError(
                f"call has no IR definition or external declaration: {symbol}"
            )
        return self.libraries.function(self.externs[symbol], self.type_name)

    def type_name(self, spec):
        return _type(spec, self.libraries)

    def expression(self, expr):
        op = expr["op"]
        E = self.expression
        if op in (
            "library_construct",
            "library_call",
            "library_ref",
            "library_member",
            "library_value",
            "variadic",
        ):
            return self.libraries.cpp_expression(expr, E, self.type_name)
        if op.startswith("object_") or op == "base_cast":
            value = self.objects.expression(expr, self)
            if value is not None:
                return value
        if op == "literal":
            return _literal(expr)
        if op == "string":
            return _string(expr["value"])
        if op == "runtime_array_size":
            name = expr["name"]
            return f"(sizeof(*{_identifier(name,'v')}) * {_identifier(name,'stack_count')})"
        if op == "fp_traps_masked":
            self.uses_fenv_guard = True
            return "scar::fp_traps_masked()"
        if op == "function_ref":
            return self.callee(expr["name"])
        if op == "var":
            if (
                self.receiver is not None
                and expr["name"] == self.receiver
                and expr.get("scope") != "global"
            ):
                return "this"
            scope = "::" if expr.get("scope") == "global" else ""
            return scope + _identifier(expr["name"], "v")
        if op == "binary":
            return f"({E(expr['left'])} {_BINARY[expr['operator']]} {E(expr['right'])})"
        if op == "unary":
            operand, operator = E(expr["value"]), expr["operator"]
            if operator in ("postinc", "postdec"):
                return f"({operand}{'++' if operator == 'postinc' else '--'})"
            return f"({_UNARY[operator]}{operand})"
        if op == "update":
            return (
                f"({E(expr['target'])} {_UPDATE[expr['operator']]} {E(expr['value'])})"
            )
        if op == "indirect_call":
            return (
                "(" + E(expr["target"]) + ")(" + ", ".join(map(E, expr["args"])) + ")"
            )
        if op == "allocate":
            element = self.type_name(expr["type"])
            suffix = f"[{E(expr['count'])}]" if expr.get("count") is not None else ""
            init = expr.get("init")
            if init is not None:
                text = E(init)
                suffix += text if init.get("op") == "init_list" else "(" + text + ")"
            return f"(new {element}{suffix})"
        if op == "release":
            return f"(delete{'[]' if expr['array'] else ''} {E(expr['target'])})"
        if op == "intrinsic":
            return self.intrinsic(expr)
        if op == "load_bytes":
            self.headers.add("cstring")
            self.helpers.add("byte_access")
            return f"scar::load_bytes<{self.type_name(expr['type'])}>({E(expr['base'])}, {E(expr['offset'])})"
        if op == "index":
            return (
                "("
                + E(expr["base"])
                + ")"
                + "".join(f"[{E(i)}]" for i in expr["indices"])
            )
        if op == "cast":
            mode = expr.get("mode", "static")
            target = {
                "static": "static_cast",
                "qualification": "const_cast",
                "reinterpret": "reinterpret_cast",
                "explicit_pointer": "scar::pointer_cast",
            }.get(mode)
            if target is None:
                raise ValueError(f"unsupported conversion {mode!r}")
            if mode == "explicit_pointer":
                self.helpers.add("pointer_cast")
            return f"{target}<{self.type_name(expr['type'])}>({E(expr['value'])})"
        if op == "call":
            symbol = expr["callee"]
            callee = self.callee(symbol)
            if symbol not in self.functions:
                callee = (
                    self.libraries.format_operator(self.externs[symbol], expr["args"])
                    or callee
                )
            return callee + "(" + ", ".join(map(E, expr["args"])) + ")"
        if op == "init_list":
            return "{" + ", ".join(map(E, expr["items"])) + "}"
        if op == "select":
            return f"({E(expr['condition'])} ? {E(expr['then'])} : {E(expr['else'])})"
        if op == "member":
            name = (
                self.libraries.field(expr["provider"], expr["record"], expr["field"])
                if expr.get("provider")
                else _identifier(expr["field"], "m")
            )
            return f"({E(expr['base'])}).{name}"
        raise ValueError(
            f"unsupported executable expression {op!r} at {expr.get('id', '?')}"
        )

    def intrinsic(self, expr):
        name, args = expr["name"], list(map(self.expression, expr["args"]))
        result_type = self.type_name(expr["type"])
        arity = len(args)
        if name in ("min", "max") and arity >= 2:
            self.headers.add("algorithm")
            return f"std::{name}<{result_type}>({{{', '.join(args)}}})"
        if name in ("powi", "sign", "mod", "modulo") and arity == 2:
            self.helpers.add(name)
            self.headers.add("cmath")
            invocation = f"scar::{name}({', '.join(args)})"
        elif name == "complex" and arity in (1, 2):
            return f"{result_type}({', '.join(args)})"
        elif name in ("real", "imag", "conjg") and arity == 1:
            member = {"real": "real", "imag": "imag", "conjg": "conj"}[name]
            invocation = f"std::{member}({args[0]})"
        else:
            self.headers.add("cmath")
            target = {"ceiling": "ceil", "nint": "round"}.get(name, name)
            if name == "atan" and arity == 2:
                target = "atan2"
            if not (
                target in _UNARY_MATH
                and arity == 1
                or target in _BINARY_MATH
                and arity == 2
            ):
                raise ValueError(
                    f"no target binding for intrinsic {name!r} with {arity} arguments"
                )
            invocation = f"std::{target}({', '.join(args)})"
        return f"static_cast<{result_type}>({invocation})"

    def declaration(self, statement):
        name = _identifier(statement["name"], "v")
        storage = statement.get("storage")
        if storage not in (None, "auto", "static", "thread_local"):
            raise ValueError(f"unsupported storage duration {storage!r}")
        prefix = storage + " " if storage in ("static", "thread_local") else ""
        text = (
            f"{_alignment(statement)}{prefix}{self.type_name(statement['type'])} {name}"
        )
        if "init" in statement and statement["init"] is not None:
            text += " = " + self.expression(statement["init"])
        return text + ";"

    def statements(self, body, depth, *, function=None):
        result = []
        for index, stmt in enumerate(body):
            if stmt["op"] == "runtime_array":
                result.extend(
                    self.scoped_array(stmt, body[index + 1 :], depth, function)
                )
                break
            result.extend(self.statement(stmt, depth))
        return result

    def scoped_array(self, stmt, suffix, depth, function):
        if (
            stmt.get("storage") != "scoped"
            or stmt.get("initialization") != "uninitialized"
        ):
            raise ValueError("unsupported runtime array storage or initialization")

        def check_flow(body, loop=False, breakable=False):
            for child in body:
                op = child["op"]
                if (
                    op in ("goto", "label")
                    or (op == "return" and function is None)
                    or (op == "break" and not breakable)
                    or (op == "continue" and not loop)
                    or (op in ("case", "default") and not breakable)
                ):
                    raise ValueError(
                        f"runtime array scope crosses an external {op} edge"
                    )
                nested_loop = op in ("for", "each", "while", "do_while", "map")
                for branch in ("body", "then", "else"):
                    check_flow(
                        child.get(branch, ()),
                        loop or nested_loop,
                        breakable or nested_loop or op == "switch",
                    )

        check_flow(suffix)
        self.headers.update(("new", "climits"))
        self.helpers.add("stack_array")
        indent = "    " * depth
        element, extent = self.type_name(stmt["element"]), self.expression(
            stmt["extent"]
        )
        result_type = self.type_name(function["result"]) if function else "void"
        returns_value = result_type != "void"
        start = "return " if returns_value else ""
        name = _identifier(stmt["name"], "v")
        count = _identifier(stmt["name"], "stack_count")
        lines = [
            f"{indent}const std::size_t {count} = static_cast<std::size_t>({extent});",
            f"{indent}{start}scar::with_stack_array<{element}>({count}, [&]({element}* const {name}) -> {result_type} {{",
        ]
        lines.extend(self.statements(suffix, depth + 1, function=function))
        if function and function["name"] == "main":
            lines.append(indent + "    return 0;")
        return [*lines, indent + "});"]

    def statement(self, stmt, depth):
        op = stmt["op"]
        indent = "    " * depth
        E = self.expression
        if op == "decl":
            return [indent + self.declaration(stmt)]
        if op == "buffer":
            self.headers.add("memory")
            self.buffer_count += 1
            owner = f"scar_buffer_{self.buffer_count}"
            name, element = _identifier(stmt["name"], "v"), self.type_name(stmt["type"])
            return [
                f"{indent}std::unique_ptr<{element}[]> {owner}(new {element}[{E(stmt['extent'])}]);",
                f"{indent}{element}* const {name} = {owner}.get();",
            ]
        if op in ("label", "goto"):
            label = _identifier(stmt["name"], "label")
            return [
                indent + (label + ": ;" if op == "label" else "goto " + label + ";")
            ]
        if op == "do_while":
            return [
                indent + "do {",
                *self.statements(stmt["body"], depth + 1),
                f"{indent}}} while ({E(stmt['condition'])});",
            ]
        if op == "release":
            return [indent + E(stmt) + ";"]
        if op in ("case", "default"):
            return [
                indent
                + (f"case {E(stmt['value'])}: ;" if op == "case" else "default: ;")
            ]
        if op == "assign":
            return [f"{indent}{E(stmt['target'])} = {E(stmt['value'])};"]
        if op == "store_bytes":
            self.headers.add("cstring")
            self.helpers.add("byte_access")
            return [
                f"{indent}scar::store_bytes<{self.type_name(stmt['type'])}>("
                f"{E(stmt['base'])}, {E(stmt['offset'])}, {E(stmt['value'])});"
            ]
        if op == "expr":
            value = stmt["value"]
            external = (
                self.externs.get(value.get("callee")) if value["op"] == "call" else None
            )
            if external and external["result"] != "void":
                self.headers.add("tuple")
                return [f"{indent}std::ignore = {E(value)};"]
            return [f"{indent}{E(value)};"]
        if op == "return":
            value = stmt.get("value")
            return [indent + ("return;" if value is None else f"return {E(value)};")]
        if op in ("break", "continue"):
            return [indent + op + ";"]
        if op == "for":
            result = [indent + "{", *self.statements(stmt["init"], depth + 1)]
            condition = (
                E(stmt["condition"]) if stmt.get("condition") is not None else "true"
            )
            step = E(stmt["step"]) if stmt.get("step") is not None else ""
            result.append(f"{indent}    for (; {condition}; {step}) {{")
            result.extend(self.statements(stmt["body"], depth + 2))
            return [*result, indent + "    }", indent + "}"]
        if op == "map":
            return self.map_statement(stmt, depth)
        if op == "each":
            declaration = (
                f"{self.type_name(stmt['type'])} {_identifier(stmt['name'], 'v')}"
            )
            header = f"for ({declaration} : {E(stmt['range'])})"
        elif op in ("if", "while", "switch"):
            header = f"{op} ({E(stmt['condition'])})"
        elif op == "block":
            header = ""
        else:
            raise ValueError(
                f"unsupported executable statement {op!r} at {stmt.get('id', '?')}"
            )
        result = [indent + (header + " " if header else "") + "{"]
        result.extend(
            self.statements(stmt["then"] if op == "if" else stmt["body"], depth + 1)
        )
        if op == "if" and stmt.get("else"):
            result.append(indent + "} else {")
            result.extend(self.statements(stmt["else"], depth + 1))
        return [*result, indent + "}"]

    def map_statement(self, stmt, depth):
        self.map_count += 1
        limit = f"scar_extent_{self.map_count}"
        index = _identifier(stmt["index"], "v")
        indent = "    " * depth
        result = [
            indent + "{",
            f"{indent}    const std::int64_t {limit} = {self.expression(stmt['extent'])};",
        ]
        region = self.regions.get(stmt.get("site"), {})
        widths = region.get("widths")
        if widths is None:
            widths = [{"minimum_elements": 0, "workers": 1}]
            if self.threshold is not None and self.workers > 1:
                if self.threshold == 0:
                    widths = []
                widths.append(
                    {"minimum_elements": self.threshold, "workers": self.workers}
                )
        if not widths or widths[0]["minimum_elements"] != 0:
            raise ValueError(
                "physical map schedule must cover an empty iteration domain"
            )
        previous = -1
        selected_width = "1"
        for band in widths:
            lower, workers = band["minimum_elements"], band["workers"]
            if (
                type(lower) is not int
                or lower <= previous
                or type(workers) is not int
                or workers < 1
            ):
                raise ValueError(
                    "invalid worker count or overlapping physical map bands"
                )
            previous = lower
            selected_width = (
                str(workers)
                if lower == 0
                else f"({limit} >= {lower} ? {workers} : {selected_width})"
            )
        if stmt.get("independent") is True and any(b["workers"] > 1 for b in widths):
            self.openmp = True
            self.headers.update(("omp.h", "sched.h"))
            result.append(
                f"{indent}    scar::map({limit}, {selected_width}, [&](std::int64_t {index}) {{"
            )
            result.extend(self.statements(stmt["body"], depth + 2))
            result.append(indent + "    });")
        else:
            result.append(
                f"{indent}    for (std::int64_t {index} = 0; {index} < {limit}; ++{index}) {{"
            )
            result.extend(self.statements(stmt["body"], depth + 2))
            result.append(indent + "    }")
        return [*result, indent + "}"]

    def signature(self, function):
        name = function["name"]
        arguments = [
            f"{self.type_name(param['type'])} {_identifier(param['name'], 'v')}"
            for param in function["params"]
        ]
        if function.get("variadic"):
            arguments.append("...")
        prefix = "" if name == "main" else 'extern "C" '
        return f"{prefix}{self.type_name(function['result'])} {self.functions[name]}({', '.join(arguments)})"

    def entry_wrapper(self):
        name = self.module["entry"]
        symbol = self.functions.get(name)
        if symbol is None:
            raise ValueError(f"missing entry definition {name!r}")
        entry = next(f for f in self.module["functions"] if f["name"] == name)
        abi = self.module.get("entry_abi")
        if name == "main":
            if abi:
                raise ValueError(
                    "a process main entry cannot also specify an array calling convention"
                )
            return []
        if "main" in self.functions:
            raise ValueError(f"entry {name!r} conflicts with a second process entry")
        if abi is None:
            if entry["params"]:
                raise ValueError(f"entry {name!r} needs an explicit executable ABI")
            if entry["result"] == "void":
                return ["int main() {", f"    {symbol}();", "}"]
            if entry["result"] in ("i32", "i16", "i8", "bool"):
                return ["int main() {", f"    return {symbol}();", "}"]
            raise ValueError(
                f"entry {name!r} has no process exit ABI for {entry['result']!r}"
            )
        if abi.get("kind") != "binary_f64_arrays":
            raise ValueError(f"unknown process entry ABI {abi.get('kind')!r}")
        shape, inputs, count = abi["shape"], abi["inputs"], abi["count"]
        if (
            not isinstance(shape, list)
            or any(type(n) is not int or n < 0 for n in shape)
            or type(inputs) is not int
            or inputs < 1
            or type(count) is not int
            or not 0 <= count < 1 << 63
        ):
            raise ValueError("invalid binary array entry dimensions or trip count")
        elements = math.prod(shape)
        if elements * 8 * inputs >= 1 << 63:
            raise ValueError("binary array input exceeds the stream offset range")
        read = {"kind": "ptr", "to": {"kind": "const", "of": "f64"}}
        write = {"kind": "ptr", "to": "f64"}
        expected = ["i64", "i64", *([read] * inputs), write]
        if [p["type"] for p in entry["params"]] != expected or entry[
            "result"
        ] != "void":
            raise ValueError(
                "binary array entry signature differs from its declared ABI"
            )
        self.headers.update(("array", "bit", "fstream", "vector"))
        args = [
            "elements",
            str(count),
            *(f"input[{i}].data()" for i in range(inputs)),
            "output.data()",
        ]
        return [
            "int main(int argc, char** argv) {",
            "    static_assert(std::endian::native == std::endian::little);",
            "    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);",
            "    if (argc != 3) return 2;",
            f"    constexpr std::int64_t elements = {elements};",
            "    constexpr std::streamsize bytes = elements * sizeof(double);",
            "    std::ifstream source(argv[1], std::ios::binary);",
            "    if (!source) return 3;",
            f"    std::array<std::vector<double>, {inputs}> input;",
            "    for (auto& buffer : input) {",
            "        buffer.resize(elements);",
            "        if (bytes && !source.read(reinterpret_cast<char*>(buffer.data()), bytes)) return 3;",
            "    }",
            "    if (source.peek() != std::char_traits<char>::eof() || source.bad()) return 3;",
            "    std::vector<double> output(elements);",
            f"    {symbol}({', '.join(args)});",
            "    std::ofstream sink(argv[2], std::ios::binary);",
            "    if (!sink) return 4;",
            "    if (bytes) sink.write(reinterpret_cast<const char*>(output.data()), bytes);",
            "    sink.close();",
            "    return sink ? 0 : 4;",
            "}",
        ]

    def record_declarations(self):
        forwards, declarations = [], []
        records = {record["name"]: record for record in self.module.get("records", ())}
        if len(records) != len(self.module.get("records", ())):
            raise ValueError("duplicate named record definition")
        order = TopologicalSorter()
        for name, record in records.items():
            kind = record.get("kind", "struct")
            keyword = "struct" if kind in ("record", "struct", "class") else kind
            if keyword not in ("struct", "union"):
                raise ValueError(f"unsupported record layout {kind!r}")
            if record.get("packed"):
                raise ValueError(
                    f"target layout for packed record {name!r} is unsupported"
                )
            if record.get("opaque") or record.get("provider"):
                tag = _identifier(name, "rec")
                target = self.libraries.opaque(record)
                forwards.append(
                    f"using {tag} = {target};" if target else f"{keyword} {tag};"
                )
                if target:
                    declarations.extend(self.layout_checks(record))
                continue
            forwards.append(f"{keyword} {_identifier(name, 'rec')};")
            embedded = {base["name"] for base in record.get("bases", ())}
            for field in record["fields"]:
                spec = field["type"]
                while isinstance(spec, dict) and spec["kind"] in (
                    "array",
                    "const",
                    "volatile",
                ):
                    spec = spec["of"]
                if isinstance(spec, dict) and spec["kind"] == "record":
                    embedded.add(spec["name"])
            if any(
                dep not in records
                or (records[dep].get("opaque") and not records[dep].get("complete"))
                for dep in embedded
            ):
                raise ValueError(f"incomplete by-value field in record {name!r}")
            order.add(
                name,
                *sorted(dep for dep in embedded if not records[dep].get("provider")),
            )
        for name in order.static_order():
            record = records[name]
            keyword = "union" if record.get("kind") == "union" else "struct"
            alignment = _alignment(record)
            bases = self.objects.bases(record)
            declarations.append(
                f"{keyword} {alignment}{_identifier(name, 'rec')}{bases} {{"
            )
            self.receiver = "this"
            for field in record["fields"]:
                init = (
                    " = " + self.expression(field["init"])
                    if field.get("init") is not None
                    else ""
                )
                declarations.append(
                    f"    {self.type_name(field['type'])} {_identifier(field['name'], 'm')}{init};"
                )
            self.receiver = None
            declarations.extend(
                "    " + method for method in self.objects.declarations(record, self)
            )
            declarations.append("};")
            declarations.extend(self.layout_checks(record))
        return forwards, declarations

    def layout_checks(self, record):
        checks, offsets, tag = [], [], _identifier(record["name"], "rec")
        if record.get("size_bytes") is not None:
            checks.append(
                f"static_assert(sizeof({tag}) == {int(record['size_bytes'])});"
            )
        if record.get("align_bytes") is not None:
            checks.append(
                f"static_assert(alignof({tag}) == {int(record['align_bytes'])});"
            )
        for field in record.get("fields", ()):
            if "offset_bits" in field:
                offset = int(field["offset_bits"])
                member = (
                    self.libraries.field(
                        record["provider"], record["symbol"], field["name"]
                    )
                    if record.get("provider")
                    else _identifier(field["name"], "m")
                )
                self.headers.add("climits")
                offsets.append(f"offsetof(T, {member}) * CHAR_BIT == {offset}")
        if offsets:
            must_verify = bool(
                record.get("provider")
                or record.get("requires_byte_layout")
                or record.get("standard_layout") is True
            )
            if must_verify:
                checks.append(f"static_assert(std::is_standard_layout_v<{tag}>);")
            # offsetof is portable only for standard-layout types. Keeping T
            # dependent prevents even instantiation on classes using a vtable.
            checks.extend(
                (
                    "static_assert([]<class T>() {",
                    "    if constexpr (std::is_standard_layout_v<T>) return",
                    "        " + " &&\n        ".join(offsets) + ";",
                    "    else return true;",
                    f"}}.operator()<{tag}>());",
                )
            )
            self.layout_scope[record["name"]] = (
                "required" if must_verify else "when_standard_layout"
            )
        return checks

    def write_sources(self, root):
        header_dir = root / "include/scar"
        header_dir.mkdir(parents=True)
        (root / "src").mkdir()
        forwards, records = self.record_declarations()
        prototypes, globals_, definitions = [], [], []
        for decl in self.module.get("globals", ()):
            name = _identifier(decl["name"], "v")
            if decl.get("external"):
                value = self.libraries.global_value(decl)
                globals_.append(f"inline auto& {name} = {value};")
            else:
                storage = decl.get("storage")
                if storage not in (None, "auto", "static", "thread_local"):
                    raise ValueError(f"unsupported global storage duration {storage!r}")
                # Global names already have unique graph identities. Declare a
                # single storage instance shared across the generated units.
                duration = "thread_local " if storage == "thread_local" else ""
                globals_.append(
                    f"{_alignment(decl)}extern {duration}{self.type_name(decl['type'])} {name};"
                )
                definitions.append(
                    self.declaration(
                        {**decl, "storage": "thread_local" if duration else None}
                    )
                )
        entry = self.module["entry"]
        functions = self.module["functions"]
        ordered = (
            fn
            for first in (True, False)
            for fn in functions
            if (fn["name"] == entry) == first
        )
        groups = []
        include = '#include "scar/program.hpp"\n\n'
        parts = [*definitions, *self.entry_wrapper(), ""]
        length = sum(map(len, parts))

        def write_group():
            name = f"src/unit_{len(groups):04}.cpp" if groups else "src/program.cpp"
            (root / name).write_text(
                include + "\n".join(parts) + "\n", encoding="utf-8"
            )
            groups.append(name)
            parts.clear()

        for fn in ordered:
            method = fn.get("method", {})
            if (
                method.get("defaulted")
                or method.get("deleted")
                or method.get("implicit")
            ):
                if fn["body"] or fn.get("initializers"):
                    raise ValueError(
                        f"body supplied for compiler-defined method {fn['name']!r}"
                    )
                continue
            signature = self.objects.signature(fn, self)
            if signature is None:
                self.receiver = None
                signature = self.signature(fn)
                prototypes.append(signature + ";")
            else:
                self.receiver = fn["params"][0]["name"]
                signature += self.objects.initializers(fn, self)
            body = "\n".join(
                (
                    signature + " {",
                    *self.statements(fn["body"], 1, function=fn),
                    "}",
                    "",
                )
            )
            if length and length + len(body) > 128 * 1024:
                write_group()
                length = 0
            parts.append(body)
            length += len(body)
        write_group()
        if self.openmp or self.uses_fenv_guard:
            self.headers.update(("cfenv", "fenv.h"))
        headers = sorted(
            self.headers | {"complex", "cstddef", "cstdint", "limits", "type_traits"}
        )
        helpers = [_TARGET_HELPERS[name] for name in sorted(self.helpers)]
        helpers.extend(sorted(self.libraries.helpers))
        if self.openmp or self.uses_fenv_guard:
            helpers.append(_FP_TRAPS)
        if self.openmp:
            helpers.append(_PARALLEL_MAP)
        header = "\n".join(
            [
                "#pragma once",
                *(f"#include <{h}>" for h in headers),
                "",
                "namespace scar {",
                "template<class T, std::size_t N> using array = T[N];",
                "template<class R, class... A> using function = R(A...);",
                "template<class R, class... A> using variadic_function = R(A..., ...);",
                *helpers,
                "}",
                "",
                *forwards,
                *self.libraries.declarations.values(),
                *prototypes,
                *globals_,
                *records,
                "",
            ]
        )
        (header_dir / "program.hpp").write_text(header, encoding="utf-8")
        return groups


def emit(module: dict, physical: dict, destination: Path) -> dict:
    """Produce a standalone project and executable from the execution IR alone."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    project = destination / "project"
    if project.exists():
        raise ValueError(f"output project already exists at {project}")
    emitter = _Emitter(module, physical)
    with TemporaryDirectory(prefix=".scar-project-", dir=destination) as tmp:
        staging = Path(tmp) / "project"
        sources = emitter.write_sources(staging)
        build(
            staging,
            sources=sources,
            openmp=emitter.openmp,
            libraries=emitter.libraries.links,
        )
        staging.rename(project)
    return {
        "project": str(project),
        "executable": str(project / "bin/scar_entry"),
        "source": str(project / "src/program.cpp"),
        "sources": [str(project / name) for name in sources],
        "header": str(project / "include/scar/program.hpp"),
        "openmp": emitter.openmp,
        "runtime_libraries": sorted(emitter.libraries.links),
        "record_field_offset_checks": emitter.layout_scope,
    }

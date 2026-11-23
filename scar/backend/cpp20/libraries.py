"""Bind declared operator types and calls to the selected C++20 target libraries."""

import re

from .model import Attribute, Decl, Expr, Type, call

_STANDARD_HEADERS = {
    "stdio.h": set(
        "printf fprintf sprintf snprintf vfprintf vprintf vsnprintf vsprintf scanf fscanf sscanf puts putchar getchar "
        "fopen fclose fread fwrite fgets fputs fputc fseek ftell rewind fflush "
        "ferror feof perror remove rename".split()
    ),
    "stdlib.h": set(
        "malloc calloc realloc free abort exit atexit abs labs llabs atof atoi "
        "atol strtod strtof strtol strtoul getenv system rand srand qsort bsearch".split()
    ),
    "time.h": set(
        "clock time difftime mktime localtime gmtime asctime ctime strftime".split()
    ),
    "fenv.h": set(
        "feclearexcept fetestexcept feraiseexcept fegetround fesetround".split()
    ),
    "ctype.h": set(
        "isalnum isalpha isblank iscntrl isdigit isgraph islower isprint ispunct isspace "
        "isupper isxdigit tolower toupper".split()
    ),
    "string.h": set(
        "memcpy memmove memcmp memset strlen strcpy strncpy strcmp strncmp "
        "strcat strncat strchr strrchr strstr strerror".split()
    ),
}
_MATH = "acos acosh asin asinh atan atan2 atanh cbrt ceil copysign cos cosh erf erfc exp exp2 expm1 fabs fdim floor fma fmax fmin fmod frexp hypot ldexp lgamma log log10 log1p log2 modf nearbyint nextafter pow remainder rint round scalbn sin sinh sqrt tan tanh tgamma trunc"
_STANDARD_HEADERS["math.h"] = {
    name + suffix for name in _MATH.split() for suffix in ("", "f", "l")
}
_STANDARD_HEADERS["math.h"].update(("finite", "finitef", "finitel"))
_HEADER_BY_FUNCTION = {
    sym: header for header, names in _STANDARD_HEADERS.items() for sym in names
}
# Public SDK ownership can be preserved in a declaration even when its operation
# has no dedicated spelling rule. Only target headers may cross the IR boundary.
_C_HEADERS = set(
    (
        "assert.h complex.h ctype.h errno.h fenv.h float.h inttypes.h limits.h locale.h "
        "math.h setjmp.h signal.h stdarg.h stddef.h stdint.h stdio.h stdlib.h string.h "
        "time.h wchar.h wctype.h dirent.h dlfcn.h fcntl.h fnmatch.h glob.h grp.h netdb.h "
        "poll.h pthread.h pwd.h regex.h sched.h semaphore.h spawn.h strings.h sys/mman.h "
        "sys/resource.h sys/select.h sys/socket.h sys/stat.h sys/statvfs.h sys/time.h "
        "sys/times.h sys/types.h sys/uio.h sys/utsname.h sys/wait.h termios.h unistd.h"
    ).split()
)
_GNU_IO = {
    "_gfortran_st_open",
    "_gfortran_st_close",
    "_gfortran_st_read",
    "_gfortran_st_read_done",
    "_gfortran_st_write",
    "_gfortran_st_write_done",
    "_gfortran_stop_numeric",
    "_gfortran_stop_string",
}
for _kind in ("integer", "real", "complex", "logical", "character"):
    _GNU_IO.update((f"_gfortran_transfer_{_kind}", f"_gfortran_transfer_{_kind}_write"))

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
_C23_CONVERSIONS = {
    "strtol": "long",
    "strtoul": "unsigned long",
    "strtoll": "long long",
    "strtoull": "unsigned long long",
    "strtoimax": "std::intmax_t",
    "strtoumax": "std::uintmax_t",
}


_TYPE_HEADERS = {
    name: header
    for header, names in {
        "string": "basic_string char_traits",
        "memory": "allocator",
        "functional": "less greater",
        "utility": "pair",
        "map": "map multimap",
        "vector": "vector",
        "ios": "basic_ios",
        "istream": "basic_istream",
        "ostream": "basic_ostream",
        "iostream": "basic_iostream",
        "fstream": "basic_ifstream basic_ofstream basic_fstream",
        "sstream": "basic_istringstream basic_ostringstream basic_stringstream",
        "complex": "complex",
        "iterator": "reverse_iterator",
    }.items()
    for name in names.split()
}
_STREAM_ENUMS = {
    "ios_openmode": "openmode",
    "ios_fmtflags": "fmtflags",
    "ios_iostate": "iostate",
    "ios_seekdir": "seekdir",
}
_STREAM_CONSTANTS = set(
    (
        "in out ate app trunc binary beg cur end left right internal fixed scientific "
        "dec oct hex skipws boolalpha showbase showpoint showpos unitbuf uppercase "
        "goodbit eofbit failbit badbit adjustfield basefield floatfield"
    ).split()
)
_MANIPULATORS = set(
    "ends endl flush ws boolalpha noboolalpha showbase noshowbase showpoint noshowpoint "
    "showpos noshowpos skipws noskipws unitbuf nounitbuf uppercase nouppercase left "
    "right internal dec oct hex fixed scientific defaultfloat hexfloat".split()
)
_MEMBERS = set(
    (
        "c_str data empty size length capacity max_size clear reserve resize push_back pop_back "
        "at front back begin end cbegin cend rbegin rend crbegin crend find rfind find_first_of "
        "find_first_not_of find_last_of find_last_not_of substr compare append assign erase "
        "insert emplace count lower_bound upper_bound equal_range good eof fail bad open is_open "
        "close get put getline read write seekg tellg seekp tellp flush flags setf unsetf "
        "precision width fill setstate rdstate ignore peek gcount str base swap real imag"
    ).split()
)
_FREE_HEADERS = {
    name: header
    for header, names in {
        "iostream": "getline endl ends flush ws",
        "iomanip": "setw setprecision setfill setbase setiosflags resetiosflags",
        "iterator": "begin end rbegin rend cbegin cend advance next prev distance",
        "utility": "make_pair move forward swap",
        "algorithm": "min max sort stable_sort fill copy find find_if reverse",
        "functional": "not1 ptr_fun",
        "complex": "abs real imag conj norm arg polar proj",
        "cmath": _MATH + " isfinite isinf isnan isnormal signbit fpclassify",
    }.items()
    for name in names.split()
}
_OPERATORS = set(
    (
        "<< >> == != <= >= < > = + - * / % & | ^ && || ! ~ += -= *= /= "
        "%= &= |= ^= <<= >>= [] -> ++ -- ()"
    ).split()
)
_PUBLIC_FIELDS = {
    ("libc", "tm"): set(
        "tm_sec tm_min tm_hour tm_mday tm_mon tm_year tm_wday tm_yday tm_isdst tm_gmtoff tm_zone".split()
    ),
    ("mpi", "MPI_Status"): {"MPI_SOURCE", "MPI_TAG", "MPI_ERROR"},
    ("mpi", "ompi_status_public_t"): {"MPI_SOURCE", "MPI_TAG", "MPI_ERROR"},
}
_PUBLIC_FIELDS.update(
    {
        ("libc", name): set(fields.split())
        for name, fields in {
            "timeval": "tv_sec tv_usec",
            "timespec": "tv_sec tv_nsec",
            "itimerval": "it_interval it_value",
            "sigaction": "sa_handler sa_sigaction sa_mask sa_flags sa_restorer",
            "rusage": "ru_utime ru_stime ru_maxrss ru_ixrss ru_idrss ru_isrss ru_minflt ru_majflt ru_nswap ru_inblock ru_oublock ru_msgsnd ru_msgrcv ru_nsignals ru_nvcsw ru_nivcsw",
            "fd_set": "fds_bits __fds_bits",
            "sigset_t": "__val",
            "rlimit": "rlim_cur rlim_max",
        }.items()
    }
)
_RECORD_HEADERS = {
    "tm": "time.h",
    "FILE": "stdio.h",
    "timeval": "sys/time.h",
    "itimerval": "sys/time.h",
    "timespec": "time.h",
    "sigaction": "signal.h",
    "sigset_t": "signal.h",
    "rusage": "sys/resource.h",
    "rlimit": "sys/resource.h",
    "fd_set": "sys/select.h",
}

_LEGACY_TYPES = {
    "unary_negate": "unary_negation",
    "pointer_to_unary_function": "unary_function_pointer",
    "pointer_to_binary_function": "binary_function_pointer",
}


class Libraries:
    def __init__(self):
        self.headers = set()
        self.links = set()
        self.declarations = {}

    def format_operator(self, decl, args):
        """Select the target's checked dynamic-format entry point."""
        counts = {"printf": 1, "fprintf": 2, "sprintf": 2, "snprintf": 3}
        name = decl["name"]
        if decl.get("provider") != "libc" or counts.get(name) != len(args):
            return None
        fmt = args[-1]
        while fmt["op"] == "cast":
            fmt = fmt["value"]
        return None if fmt["op"] == "string" else "scar::dynamic_" + name

    def prototype(
        self, symbol, result, params, *, variadic=False, noexcept=False, noreturn=False
    ):
        self.declarations[symbol] = (
            Decl(
                "function",
                name=symbol,
                type=result,
                params=tuple(Decl("param", type=ty) for ty in params)
                + ((Decl("param", definition="variadic"),) if variadic else ()),
                linkage="C",
                qualifiers=("noexcept",) if noexcept else (),
                attributes=(Attribute("noreturn"),) if noreturn else (),
            ),
        )

    def function(self, decl, spell_type):
        symbol = decl["name"]
        provider = decl.get("provider")
        if provider == "libc" and symbol == "__assert_fail":
            text = {"kind": "ptr", "to": {"kind": "const", "of": "char"}}
            if (
                [p["type"] for p in decl["params"]] != [text, text, "u32", text]
                or decl["result"] != "void"
                or decl.get("variadic")
            ):
                raise ValueError("invalid glibc assertion failure interface")
            self.prototype(
                symbol,
                Type("name", name="void"),
                [spell_type(p["type"]) for p in decl["params"]],
                noexcept=True,
                noreturn=True,
            )
            return "::" + symbol
        if provider == "libgfortran" and symbol in _GNU_IO:
            if decl.get("soname") != "libgfortran.so.5":
                raise ValueError(f"incompatible GNU runtime for {symbol}")
            types, layout = [], []
            for param in decl["params"]:
                spec = param["type"]
                types.append(spell_type(spec))
                layout.append(
                    "ptr" if isinstance(spec, dict) and spec["kind"] == "ptr" else spec
                )
            if symbol.startswith("_gfortran_st_"):
                required = ["ptr"]
            elif symbol == "_gfortran_stop_numeric":
                required = ["i32", "bool"]
            elif symbol == "_gfortran_stop_string":
                required = ["ptr", "u64", "bool"]
            else:
                length = (
                    "i64"
                    if symbol.startswith("_gfortran_transfer_character")
                    else "i32"
                )
                required = ["ptr", "ptr", length]
            if layout != required or decl.get("variadic") or decl["result"] != "void":
                raise ValueError(f"invalid GNU I/O procedure interface {symbol}")
            self.prototype(symbol, Type("name", name="void"), types)
            self.links.add("libgfortran.so.5")
            return "::" + symbol
        if provider == "libc" and re.fullmatch(
            r"__isoc(?:99|23)_(?:f|s)?scanf", symbol
        ):
            # The linker symbol identifies the glibc format-string dialect.
            routine = symbol.split("_")[-1]
            expected = 1 if routine == "scanf" else 2
            if (
                len(decl["params"]) != expected
                or not decl.get("variadic")
                or decl["result"] != "i32"
            ):
                raise ValueError(f"invalid glibc scanning interface {symbol}")
            types = [spell_type(p["type"]) for p in decl["params"]]
            self.prototype(symbol, Type("name", name="int"), types, variadic=True)
            return "::" + symbol
        routine = symbol.removeprefix("__isoc23_")
        if provider == "libc" and symbol != routine and routine in _C23_CONVERSIONS:
            expected = [
                {"kind": "ptr", "to": {"kind": "const", "of": "char"}},
                {"kind": "ptr", "to": {"kind": "ptr", "to": "char"}},
                "i32",
            ]
            params = [p["type"] for p in decl["params"]]
            if (
                params != expected
                or decl.get("variadic")
                or decl["result"] not in ("i32", "i64", "u32", "u64")
            ):
                raise ValueError(f"invalid glibc integer conversion interface {symbol}")
            result, logical_type = _C23_CONVERSIONS[routine], spell_type(decl["result"])
            target = Type("name", name=result)
            self.prototype(
                symbol, target, [spell_type(t) for t in expected], noexcept=True
            )
            size = Expr(
                "binary",
                operator="==",
                args=(
                    Expr("sizeof_type", type=target),
                    Expr("sizeof_type", type=logical_type),
                ),
            )
            sign = Expr(
                "binary",
                operator="==",
                args=tuple(
                    Expr("template_name", name="std::is_signed_v", args=(t,))
                    for t in (target, logical_type)
                ),
            )
            self.declarations[symbol] += (
                Decl(
                    "static_assert",
                    initializer=Expr("binary", operator="&&", args=(size, sign)),
                ),
            )
            return "::" + symbol
        if symbol in _HEADER_BY_FUNCTION and provider in (None, "libc", "libm"):
            self.headers.add(_HEADER_BY_FUNCTION[symbol])
            return "::" + symbol
        if (
            provider in ("libc", "libm", "posix")
            and decl.get("header") in _C_HEADERS
            and _IDENTIFIER.fullmatch(symbol)
        ):
            header = decl["header"]
            self.headers.add(header)
            if header in ("pthread.h", "semaphore.h"):
                self.links.add("Threads")
            if header == "dlfcn.h":
                self.links.add("DynamicLoader")
            return "::" + symbol
        if provider == "mpi" and re.fullmatch(r"P?MPI_[A-Za-z0-9_]+", symbol):
            self.links.add("MPI")
            self.headers.add("mpi.h")
            return "::" + symbol
        raise ValueError(f"no target library binding for declared external {symbol!r}")

    def opaque(self, record):
        provider, symbol = record.get("provider"), record.get("symbol")
        if provider is None:
            return None
        if provider == "libc" and symbol in _RECORD_HEADERS:
            self.headers.add(_RECORD_HEADERS[symbol])
            return Type(
                "tagged" if symbol == "sigaction" else "name", name="::" + symbol
            )
        if provider == "mpi" and symbol and _IDENTIFIER.fullmatch(symbol):
            self.links.add("MPI")
            self.headers.add("mpi.h")
            return Type(
                "name",
                name=(
                    "::MPI_Status"
                    if symbol == "ompi_status_public_t"
                    else "::" + symbol
                ),
            )
        raise ValueError(f"no target binding for opaque record {record['name']!r}")

    def field(self, provider, record, field):
        if field not in _PUBLIC_FIELDS.get((provider, record), ()):
            raise ValueError(f"no public {provider} field {record}.{field}")
        return field

    def global_value(self, decl):
        name, provider = decl["name"], decl.get("provider")
        spec = decl["type"]
        if isinstance(spec, dict) and spec.get("kind") == "const":
            spec = spec["of"]
        if (
            isinstance(spec, dict)
            and spec.get("provider") == "cxx_standard"
            and spec.get("name") in _STREAM_ENUMS
        ):
            for flag in _STREAM_CONSTANTS:
                if name == f"_ZNSt8ios_base{len(flag)}{flag}E":
                    self.headers.add("ios")
                    return "std::ios_base::" + flag
        if provider == "libc" and name in ("stdin", "stdout", "stderr"):
            self.headers.add("stdio.h")
        elif provider == "mpi" and _IDENTIFIER.fullmatch(name):
            self.headers.add("mpi.h")
            self.links.add("MPI")
        else:
            raise ValueError(f"no target binding for external global {name!r}")
        return "::" + name

    def cpp_type(self, spec, spell_type):
        if (
            spec.get("provider") == "c_abi"
            and spec["name"] == "va_list"
            and not spec.get("args")
        ):
            self.headers.add("cstdarg")
            return Type("name", name="std::va_list")
        if spec.get("provider") != "cxx_standard":
            raise ValueError(f"no target operator library for {spec.get('provider')!r}")
        name = spec["name"]
        args = tuple(spell_type(t) for t in spec.get("args", ()))
        if name in _LEGACY_TYPES:
            return Type("template", name="scar::" + _LEGACY_TYPES[name], args=args)
        if name in _STREAM_ENUMS and not args:
            self.headers.add("ios")
            return Type("name", name="std::ios_base::" + _STREAM_ENUMS[name])
        if name in ("iterator", "const_iterator") and len(args) == 1:
            return Type("member", parent=args[0], name=name)
        for family in ("map", "vector", "string"):
            if name in (f"{family}_iterator", f"{family}_const_iterator"):
                self.headers.add(family)
                owner = Type(
                    "template",
                    name="std::" + ("basic_string" if family == "string" else family),
                    args=args,
                )
                return Type("member", parent=owner, name=name[len(family) + 1 :])
        if name not in _TYPE_HEADERS:
            raise ValueError(f"no target binding for standard library type {name!r}")
        self.headers.add(_TYPE_HEADERS[name])
        return Type("template", name="std::" + name, args=args)

    def cpp_expression(self, expr, expression, spell_type):
        op = expr["op"]

        if op == "library_value":
            if expr.get("provider") != "cxx_standard":
                raise ValueError("unbound library value")
            value = expression(expr["value"])
            if expr["mode"] == "qualify":
                return value
            if expr["mode"] == "copy":
                ty = Type(
                    "template",
                    name="std::decay_t",
                    args=(Type("decltype", args=(value,)),),
                )
                return Expr("construct", type=ty, args=(value,), qualifier="paren")
            raise ValueError(f"unsupported value construction {expr['mode']!r}")
        if op == "variadic":
            self.headers.add("cstdarg")
            name, value = expr["name"], expression(expr["list"])
            if name == "start" and expr["last"]["op"] == "var":
                return call("va_start", value, expression(expr["last"]))
            if name == "end":
                return call("va_end", value)
            if name == "copy":
                return call("va_copy", value, expression(expr["source"]))
            if name == "arg":
                ty = spell_type(expr["type"])
                identity = Expr(
                    "construct",
                    type=Type("template", name="std::type_identity", args=(ty,)),
                    args=(),
                    qualifier="brace",
                )
                return call(
                    "va_arg",
                    value,
                    Type(
                        "member", parent=Type("decltype", args=(identity,)), name="type"
                    ),
                )
            raise ValueError(f"unsupported variadic operation {name!r}")
        if op == "library_construct":
            ty = expr["type"]
            if not isinstance(ty, dict) or ty["kind"] != "library":
                raise ValueError("library construction requires a declared type")
            form = expr.get("form", "paren")
            if form not in ("list", "paren", "value"):
                raise ValueError(f"unsupported construction form {form!r}")
            return Expr(
                "construct",
                type=spell_type(ty),
                args=tuple(map(expression, expr.get("args", ()))),
                qualifier="brace" if form == "list" else "paren",
            )
        if expr.get("provider") != "cxx_standard":
            raise ValueError(f"unknown library {expr.get('provider')}")
        name = expr["name"]
        if op == "library_member":
            if name not in ("first", "second"):
                raise ValueError(f"unsupported library member {name}")
            return Expr("member", args=(expression(expr["receiver"]),), name=name)
        if op == "library_ref":
            if name == "npos":
                owner = expr.get("owner", expr.get("type"))
                if not isinstance(owner, dict) or owner.get("name") != "basic_string":
                    raise ValueError("npos requires a string owner")
                return Expr("type_member", type=spell_type(owner), name="npos")
            if name in ("cout", "cerr", "cin", "clog") or name in _MANIPULATORS:
                self.headers.add("iostream")
                return Expr("name", name="std::" + name)
            if name.startswith("ios_") and name[4:] in _STREAM_CONSTANTS:
                self.headers.add("ios")
                return Expr("name", name="std::ios_base::" + name[4:])
            raise ValueError(f"unsupported library reference {name!r}")
        args = tuple(map(expression, expr["args"]))
        receiver = (
            expression(expr["receiver"]) if expr.get("receiver") is not None else None
        )
        if name.startswith("operator"):
            values = (receiver, *args) if receiver is not None else args
            token = name[len("operator") :].strip()
            if token == "bool" and len(values) == 1:
                return Expr(
                    "cast",
                    operator="static",
                    type=Type("name", name="bool"),
                    args=values,
                )
            if token not in _OPERATORS or not values:
                raise ValueError(f"unsupported operator {token!r}")
            left, *rest = values
            if token == "[]" and len(rest) == 1:
                return Expr("subscript", args=values)
            if token == "()":
                return call(left, *rest)
            if token == "->" and not rest:
                return call(Expr("member", args=(left,), name="operator->"))
            if token in ("++", "--"):
                if not rest:
                    return Expr("unary", operator=token, args=(left,))
                if len(rest) == 1:
                    return (
                        call(Expr("member", args=(left,), name=name), *rest)
                        if receiver is not None
                        else call(name, *values)
                    )
            if not rest and token in ("+", "-", "*", "&", "!", "~"):
                return Expr("unary", operator=token, args=(left,))
            if len(rest) == 1 and token not in ("!", "~", "[]", "->"):
                return Expr("binary", operator=token, args=values)
            raise ValueError(f"invalid arity for operator {token!r}")
        if receiver is not None and name in _MEMBERS:
            return call(Expr("member", args=(receiver,), name=name), *args)
        if receiver is None and name in _FREE_HEADERS:
            self.headers.add(_FREE_HEADERS[name])
            symbol = {
                "ptr_fun": "scar::legacy_function",
                "not1": "scar::legacy_not",
            }.get(name, "std::" + name)
            target = (
                Expr(
                    "template_name",
                    name=symbol,
                    args=tuple(map(spell_type, expr["template_args"])),
                )
                if expr.get("template_args")
                else Expr("name", name=symbol)
            )
            return call(target, *args)
        raise ValueError(f"no binding for library operation {name!r}")

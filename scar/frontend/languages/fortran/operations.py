"""F2: normalize operations using captured Fortran symbols, syntax and ABI data."""

import math
import re

from .facts import Gap, kind, ptr


def _syntax_nodes(root, kinds):
    """Traverse already-captured nodes; no parsing or name resolution."""
    pending = [root]
    while pending:
        node = pending.pop()
        if kind(node) in kinds:
            yield node
        children = (
            node if isinstance(node, (tuple, list)) else getattr(node, "children", ())
        )
        if children:
            pending.extend(reversed(children))


def var(name):
    return {"op": "var", "name": name}


def literal(dtype, value):
    return {"op": "literal", "type": dtype, "value": value}


def binary(operator, left, right):
    return {"op": "binary", "operator": operator, "left": left, "right": right}


def unary(operator, value):
    return {"op": "unary", "operator": operator, "value": value}


def index(base, offset):
    return {"op": "index", "base": base, "indices": [offset]}


def assign(target, value):
    if target["op"] == "load_bytes":
        return {**target, "op": "store_bytes", "value": value}
    return {"op": "assign", "target": target, "value": value}


def cast(dtype, value):
    if value.get("op") == "literal" and value.get("type") == dtype:
        return value
    return {"op": "cast", "type": dtype, "value": value}


def decl(name, dtype, init=None):
    result = {"op": "decl", "name": name, "type": dtype}
    if init is not None:
        result["init"] = init
    return result


_OPERATORS = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "==": "eq",
    ".eq.": "eq",
    "/=": "ne",
    ".ne.": "ne",
    "<=": "le",
    ".le.": "le",
    "<": "lt",
    ".lt.": "lt",
    ">=": "ge",
    ".ge.": "ge",
    ">": "gt",
    ".gt.": "gt",
    ".and.": "land",
    ".or.": "lor",
    ".eqv.": "eq",
    ".neqv.": "ne",
}
_RELATIONS = {"eq", "ne", "le", "lt", "ge", "gt", "land", "lor"}
_MATH = {
    "abs",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "exp",
    "log",
    "log10",
    "max",
    "min",
    "sign",
    "mod",
    "conjg",
}
_LEGACY_MATH = {
    "iabs": "abs",
    "dabs": "abs",
    "cdabs": "abs",
    "cabs": "abs",
    "dsqrt": "sqrt",
    "dsin": "sin",
    "dcos": "cos",
    "dtan": "tan",
    "dacos": "acos",
    "datan": "atan",
    "datan2": "atan2",
    "dsinh": "sinh",
    "dcosh": "cosh",
    "dexp": "exp",
    "cdexp": "exp",
    "zexp": "exp",
    "dlog": "log",
    "dlog10": "log10",
    "max0": "max",
    "min0": "min",
    "dmax1": "max",
    "dmin1": "min",
    "dsign": "sign",
    "dmod": "mod",
    "dconjg": "conjg",
}


def numeric_type(a, b):
    # Fortran integer operands acquire the real operand's kind in mixed arithmetic.
    def precedence(t):
        return ("ifc".index(t[0]), int(t[1:]))

    if a == b:
        return a
    if a == "bool" or b == "bool":
        return "bool"
    if a.startswith("c") or b.startswith("c"):
        size = max(
            (
                int(t[1:]) // 2
                if t.startswith("c")
                else int(t[1:]) if t.startswith("f") else 0
            )
            for t in (a, b)
        )
        return f"c{2*size}"
    return max((a, b), key=precedence)


def number(value, dtype):
    token = re.sub(r"_\d+$", "", value.strip()).replace("D", "e").replace("d", "e")
    if dtype[0] == "i":
        if token.lower() in (".true.", ".false."):
            return literal(dtype, int(token.lower() == ".true."))
        return literal(dtype, int(token))
    if dtype[0] == "f":
        parsed = float(token)
        if not math.isfinite(parsed):
            raise Gap("constant", f"nonfinite {dtype} constant")
        return literal(dtype, parsed)
    raise Gap("constant", f"constant of type {dtype}")


def choose(condition, yes, no):
    return {"op": "select", "condition": condition, "then": yes, "else": no}


def product(values):
    result = values[0]
    for value in values[1:]:
        result = binary("mul", result, value)
    return result


def integer(expr):
    return (
        expr.get("value")
        if expr and expr.get("op") == "literal" and expr["type"].startswith("i")
        else None
    )


class Body:
    def __init__(
        self,
        symbols,
        function_name,
        result_name,
        link_name,
        location,
        defaults,
        target,
        *,
        interoperable=False,
    ):
        self.target = target
        self.interoperable = interoperable
        self.symbols = symbols
        self.function_name, self.result_name = function_name, result_name
        self.link_name, self.location = link_name, location
        self.default_integer, self.default_real, self.double = defaults
        self.prologue = []
        self._bounds = {}
        self._temporaries = 0
        self._prelude = []
        self.externs = {}
        self.records = {}
        self.formats = {}
        self.calls = set()
        self._views = {}
        self.aliases = {}
        self.storage = {}
        self.character_lengths = {}

    def bind_character(self, name, hidden):
        declared = self.fact(name).char_length
        if str(declared) == "*":
            if not hidden:
                raise Gap("character_abi", "missing assumed character length")
            self.character_lengths[name] = cast("i64", var(hidden))
        elif declared is None:
            self.character_lengths[name] = literal("i64", 1)
        else:
            extent = self.capture_bound(declared)
            self.character_lengths[name] = {
                "op": "intrinsic",
                "name": "max",
                "type": "i64",
                "args": [literal("i64", 0), extent],
            }

    def character_view(self, node):
        tag = kind(node)
        if tag == "Char_Literal_Constant":
            text = self.characters(node)
            return {"op": "string", "value": text}, literal("i64", len(text.encode()))
        if tag not in {"Name", "Part_Ref", "Substring"}:
            return None
        name = str(node if tag == "Name" else node.items[0]).lower()
        if name not in self.character_lengths:
            return None
        source = var(name)
        length = self.character_lengths[name]
        if tag == "Name":
            return source, length
        tail = node.items[1]
        if kind(tail) == "Section_Subscript_List" and len(tail.items) == 1:
            tail = tail.items[0]
        if kind(tail) not in {"Subscript_Triplet", "Substring_Range"} or (
            len(tail.items) > 2 and tail.items[2] is not None
        ):
            raise Gap("character_value", "invalid substring selection", node)
        lo = (
            self._once(self.expression(tail.items[0])[0])
            if tail.items[0] is not None
            else literal("i64", 1)
        )
        hi = (
            self._once(self.expression(tail.items[1])[0])
            if tail.items[1] is not None
            else length
        )
        length = self._once(
            {
                "op": "intrinsic",
                "name": "max",
                "type": "i64",
                "args": [
                    literal("i64", 0),
                    binary("add", binary("sub", hi, lo), literal("i64", 1)),
                ],
            }
        )
        return binary("add", source, binary("sub", lo, literal("i64", 1))), length

    def libc(self, name, types, args):
        self.externs[name] = dict(
            name=name,
            result=ptr("void"),
            provider="libc",
            params=[dict(name=f"arg{i}", type=t) for i, t in enumerate(types)],
        )
        return {"op": "expr", "value": {"op": "call", "callee": name, "args": args}}

    def character_assignment(self, dest, rhs):
        source = self.character_view(rhs)
        if source is None:
            raise Gap("character_value", kind(rhs), rhs)
        dst, length = dest
        src, available = source
        n = self._once(
            {
                "op": "intrinsic",
                "name": "min",
                "type": "i64",
                "args": [length, available],
            }
        )
        size = self.target.size_type
        copy = self.libc(
            "memmove", [ptr("void"), ptr("void", True), size], [dst, src, cast(size, n)]
        )
        pad = self.libc(
            "memset",
            [ptr("void"), "i32", size],
            [
                binary("add", dst, n),
                literal("i32", 32),
                cast(size, binary("sub", length, n)),
            ],
        )
        return {
            "op": "block",
            "body": [
                {
                    "op": "if",
                    "condition": binary("gt", n, literal("i64", 0)),
                    "then": [copy],
                    "else": [],
                },
                {
                    "op": "if",
                    "condition": binary("gt", length, n),
                    "then": [pad],
                    "else": [],
                },
            ],
        }

    def fresh(self, stem):
        self._temporaries += 1
        return f"_f_{stem}_{self._temporaries}"

    def fact(self, name, node=None):
        try:
            return self.symbols[name.lower()]
        except KeyError:
            raise Gap("symbol", f"unresolved symbol {name}", node) from None

    def capture_bound(self, node):
        if node is None:
            return literal("i64", 1)
        key = str(node).lower()
        if key not in self._bounds:
            value, dtype = self.expression(node)
            if not dtype.startswith("i"):
                raise Gap("array_bound", f"noninteger array bound {key}", node)
            n = integer(value)
            if n is not None:
                self._bounds[key] = literal("i64", n)
            else:
                name = self.fresh("bound")
                self.prologue.append(decl(name, "i64", cast("i64", value)))
                self._bounds[key] = var(name)
        return self._bounds[key]

    def bind_dimensions(self, name):
        symbol = self.fact(name)
        if symbol.bounds:
            return
        for lo, hi in symbol.dims:
            lower = self.capture_bound(lo)
            upper = self.capture_bound(hi) if hi is not None else None
            if upper is None:
                extent = None
            elif integer(lower) is not None and integer(upper) is not None:
                extent = literal("i64", max(0, integer(upper) - integer(lower) + 1))
            else:
                key = (str(lo or 1).lower(), str(hi).lower())
                if key not in self._bounds:
                    name = self.fresh("extent")
                    span = binary("add", binary("sub", upper, lower), literal("i64", 1))
                    self.prologue.append(decl(name, "i64", span))
                    self.prologue.append(
                        {
                            "op": "if",
                            "condition": binary("lt", var(name), literal("i64", 0)),
                            "then": [assign(var(name), literal("i64", 0))],
                            "else": [],
                        }
                    )
                    self._bounds[key] = var(name)
                extent = self._bounds[key]
            symbol.bounds.append((lower, extent))

    def size(self, name):
        self.bind_dimensions(name)
        extents = [d[1] for d in self.fact(name).bounds]
        if not extents or any(x is None for x in extents):
            raise Gap("array_extent", f"runtime extent unavailable for {name}")
        result = extents[0]
        for other in extents[1:]:
            result = binary("mul", result, other)
        return result

    def memory_index(self, name, subscripts):
        self.bind_dimensions(name)
        dims = self.fact(name).bounds
        if len(subscripts) != len(dims):
            raise Gap(
                "array_rank",
                f"{name} expects {len(dims)} subscripts, got {len(subscripts)}",
            )
        offset = None
        for arg, (lower, extent) in reversed(tuple(zip(subscripts, dims))):
            at = binary("sub", cast("i64", arg), lower)
            if offset is not None:
                if extent is None:
                    raise Gap("array_extent", f"missing stride in {name}")
                at = binary("add", at, binary("mul", extent, offset))
            offset = at
        return self.element(name, offset)

    def element(self, name, offset):
        if name not in self.storage:
            return index(var(name), offset)
        s = self.storage[name]
        displacement = binary("mul", offset, literal("i64", s["bytes"]))
        return {
            "op": "load_bytes",
            "type": self.fact(name).dtype,
            "base": s["base"],
            "offset": binary("add", literal("i64", s["offset"]), displacement),
        }

    def scalar(self, name, index_value=None):
        symbol = self.fact(name)
        if "PARAMETER" in symbol.attrs:
            if symbol.dims or symbol.value is None:
                raise Gap("parameter", f"unavailable scalar constant {name}")
            try:
                return number(symbol.value, symbol.dtype)
            except ValueError as exc:
                raise Gap("parameter", f"constant syntax for {name}") from exc
        if symbol.dims:
            if index_value is None:
                raise Gap(
                    "array_value", f"whole array {name} outside elemental context"
                )
            return self.element(name, index_value)
        if name in self.storage:
            return self.element(name, literal("i64", 0))
        address = var(self.aliases.get(name, name))
        if symbol.byref and name not in self.aliases:
            return unary("deref", address)
        return address

    def _once(self, expression, dtype="i64"):
        if expression.get("op") == "literal":
            return literal(dtype, expression["value"])
        name = self.fresh("value")
        self._prelude.append(decl(name, dtype, cast(dtype, expression)))
        return var(name)

    def section(self, node):
        """Return column-major section bounds; scalar subscripts remove a dimension."""
        key = id(node)
        if key in self._views:
            return self._views[key]
        name = str(node.items[0]).lower()
        self.bind_dimensions(name)
        bounds = self.fact(name).bounds
        subscripts = node.items[1].items
        if len(bounds) != len(subscripts):
            raise Gap("array_rank", name, node)
        axes = []
        for spec, (lo, length) in zip(subscripts, bounds):
            if kind(spec) != "Subscript_Triplet":
                if self.shape_of(spec):
                    raise Gap("vector_subscript", name, spec)
                at = self._once(self.expression(spec)[0])
                axes.append((at, None, None))
                continue
            first, last, stride = spec.items
            step = (
                self._once(self.expression(stride)[0]) if stride else literal("i64", 1)
            )
            if length is None and (first is None or last is None):
                raise Gap("array_extent", f"unknown section bound of {name}", spec)
            if first is None and last is None and stride is None:
                axes.append((lo, step, length))
                continue
            hi = (
                binary("sub", binary("add", lo, length), literal("i64", 1))
                if length
                else None
            )
            positive = binary("gt", step, literal("i64", 0))
            start = (
                self._once(self.expression(first)[0])
                if first
                else choose(positive, lo, hi)
            )
            end = (
                self._once(self.expression(last)[0])
                if last
                else choose(positive, hi, lo)
            )
            count = binary("div", binary("add", binary("sub", end, start), step), step)
            count = self._once(
                {
                    "op": "intrinsic",
                    "name": "max",
                    "args": [literal("i64", 0), count],
                    "type": "i64",
                }
            )
            axes.append((start, step, count))
        self._views[key] = axes
        return axes

    def shape_of(self, node):
        if kind(node) == "Name":
            name = str(node).lower()
            symbol = self.fact(name)
            if symbol.dims:
                self.bind_dimensions(name)
                return tuple(n for _, n in symbol.bounds)
            return ()
        parts = getattr(node, "items", ())
        if kind(node) in {
            "Part_Ref",
            "Function_Reference",
            "Structure_Constructor",
            "Intrinsic_Function_Reference",
        }:
            name = str(parts[0]).lower()
            if name in {
                "size",
                "sum",
                "product",
                "any",
                "all",
                "maxval",
                "minval",
                "count",
                "lbound",
                "ubound",
            }:
                return ()
            symbol = self.symbols.get(name)
            if symbol and symbol.dims:
                args = parts[1].items if parts[1] else ()
                if not any(kind(a) == "Subscript_Triplet" for a in args):
                    if any(self.shape_of(a) for a in args):
                        raise Gap("vector_subscript", name, node)
                    return ()
                return tuple(
                    count for _, step, count in self.section(node) if step is not None
                )
            if (
                symbol
                and "PROCEDURE" in symbol.attrs
                and "INTRINSIC-PROC" not in symbol.attrs
            ):
                return ()
            parts = parts[1].items if parts[1] else ()
        for child in parts:
            if hasattr(child, "items") or kind(child) == "Name":
                answer = self.shape_of(child)
                if answer:
                    return answer
        return ()

    def section_index(self, node, position):
        name = str(node.items[0]).lower()
        coordinates = []
        for start, step, count in self.section(node):
            if step is None:
                coordinates.append(start)
            else:
                if position is None:
                    raise Gap(
                        "array_value", f"section of {name} outside array context", node
                    )
                if integer(position) == 0:
                    coordinates.append(start)
                else:
                    axis = binary("mod", position, count)
                    coordinates.append(binary("add", start, binary("mul", step, axis)))
                    position = binary("div", position, count)
        return self.memory_index(name, coordinates)

    def _intrinsic_result(
        self,
        source_name,
        normalized_name,
        expression,
        dtype,
        argument_types,
        conversion_basis,
        **details,
    ):
        families = {normalized_name, _LEGACY_MATH.get(source_name, source_name)}
        bindings = [
            name
            for name in getattr(self, "intrinsic_bindings", ())
            if any(name.startswith("__" + family + "_") for family in families)
        ]
        return {
            **expression,
            "source_intrinsic_contract": {
                "semantics_version": 1,
                "provider": "fortran_intrinsic_or_operator",
                "source_name": source_name,
                "normalized_name": normalized_name,
                "argument_types": list(argument_types),
                "result_type": dtype,
                "conversion_basis": conversion_basis,
                "validation": "accepted_by_captured_gnu_semantics",
                "gnu_resolution_candidates": bindings,
                "gnu_resolution_scope": "procedure_catalog_not_callsite_binding",
                **details,
            },
        }, dtype

    def reduction(self, name, args):
        if len(args) != 1:
            raise Gap("array_reduction", name + " with DIM, MASK or KIND")
        shape = self.shape_of(args[0])
        if not shape or any(x is None for x in shape):
            raise Gap(
                "array_reduction", name + " needs a known iteration domain", args[0]
            )
        extent = product(shape)
        iterator = self.fresh("reduction_index")
        accumulator = self.fresh("reduction")
        value, dtype = self.expression(args[0], var(iterator))
        source_dtype = dtype
        if name in {"sum", "product"}:
            initial = literal(
                dtype,
                (
                    ([0, 0] if name == "sum" else [1, 0])
                    if dtype.startswith("c")
                    else int(name == "product")
                ),
            )
            update = assign(
                var(accumulator),
                binary("add" if name == "sum" else "mul", var(accumulator), value),
            )
        elif name in {"any", "all"}:
            dtype = "bool"
            initial = literal("bool", name == "all")
            update = assign(
                var(accumulator),
                binary("lor" if name == "any" else "land", var(accumulator), value),
            )
        else:
            raise Gap("array_reduction", name)
        self._prelude.extend(
            [
                decl(accumulator, dtype, initial),
                {
                    "op": "for",
                    "init": [decl(iterator, "i64", literal("i64", 0))],
                    "condition": binary("lt", var(iterator), extent),
                    "step": unary("preinc", var(iterator)),
                    "body": [update],
                },
            ]
        )
        return self._intrinsic_result(
            name,
            name,
            var(accumulator),
            dtype,
            [source_dtype],
            "fortran_array_reduction",
            argument_forms=["array"],
            source_rank=len(shape),
        )

    def intrinsic(self, name, args, element):
        source_name = name
        if name in {"sum", "product", "any", "all", "maxval", "minval", "count"}:
            return self.reduction(name, args)
        if name in {"lbound", "ubound", "size"}:
            if not args or kind(args[0]) != "Name" or not self.fact(str(args[0])).dims:
                raise Gap("array_inquiry", name)
            target = str(args[0]).lower()
            self.bind_dimensions(target)
            if len(args) == 1 and name == "size":
                return self._intrinsic_result(
                    name,
                    name,
                    cast(self.default_integer, self.size(target)),
                    self.default_integer,
                    [self.fact(target).dtype],
                    "fortran_array_inquiry_default_integer",
                    argument_forms=["array"],
                    source_rank=len(self.fact(target).bounds),
                )
            if len(args) != 2:
                raise Gap("array_inquiry", name + " needs scalar DIM")
            dim = args[1].items[1] if kind(args[1]) == "Actual_Arg_Spec" else args[1]
            dimension_value, dimension_type = self.expression(dim)
            dimension = integer(dimension_value)
            bounds = self.fact(target).bounds
            if dimension is None or not 1 <= dimension <= len(bounds):
                raise Gap("array_inquiry", name + " dynamic DIM")
            lo, length = bounds[dimension - 1]
            if length is None:
                raise Gap("array_extent", name + " on assumed-size last dimension")
            # The bounds of an empty dimension are 1 and 0 respectively.
            val = {
                "size": length,
                "lbound": lo,
                "ubound": binary("sub", binary("add", lo, length), literal("i64", 1)),
            }[name]
            if name in {"lbound", "ubound"}:
                val = choose(
                    binary("gt", length, literal("i64", 0)),
                    val,
                    literal("i64", int(name == "lbound")),
                )
            return self._intrinsic_result(
                name,
                name,
                cast(self.default_integer, val),
                self.default_integer,
                [self.fact(target).dtype, dimension_type],
                "fortran_array_inquiry_default_integer",
                argument_forms=["array", "scalar"],
                source_rank=len(bounds),
            )
        keywords = {}
        positional = []
        for arg in args:
            if kind(arg) == "Actual_Arg_Spec":
                keyword = str(arg.items[0]).lower()
                if (
                    keyword != "kind"
                    or keyword in keywords
                    or name not in {"real", "int", "cmplx"}
                ):
                    raise Gap("intrinsic_keyword", f"{name}({keyword}=...)", arg)
                keywords[keyword] = arg.items[1]
            else:
                positional.append(arg)
        selected_kind = (
            integer(self.expression(keywords["kind"])[0])
            if "kind" in keywords
            else None
        )
        if keywords and selected_kind not in (1, 2, 4, 8):
            raise Gap("kind", name)
        values = [self.expression(n, element) for n in positional]
        if not values:
            raise Gap("intrinsic", name)
        xs, types = zip(*values)
        dtype = types[0]
        if name in {"cmplx", "dcmplx"}:
            # With three arguments the final argument is KIND; a complex first
            # argument plus a kind selector is handled through an explicit keyword.
            xs = list(xs)
            target = "c" + str(
                2 * int((self.double if name == "dcmplx" else self.default_real)[1:])
            )
            if len(xs) == 3 or selected_kind is not None:
                width = (
                    selected_kind if selected_kind is not None else integer(xs.pop())
                )
                if width not in (4, 8):
                    raise Gap("kind", name)
                selected_kind = width
                target = "c" + str(16 * width)
            if dtype.startswith("c") and len(xs) == 1:
                return self._intrinsic_result(
                    source_name,
                    "complex",
                    cast(target, xs[0]),
                    target,
                    types,
                    "fortran_complex_kind_selection",
                    selected_kind=selected_kind,
                )
            return self._intrinsic_result(
                source_name,
                "complex",
                {
                    "op": "intrinsic",
                    "name": "complex",
                    "args": xs,
                    "type": target,
                },
                target,
                types,
                "fortran_complex_kind_selection",
                selected_kind=selected_kind,
            )
        if name in {"dreal", "aimag", "dimag"}:
            if not dtype.startswith("c"):
                raise Gap("intrinsic", name + " requires COMPLEX")
            target = "f" + str(int(dtype[1:]) // 2)
            component = "real" if name == "dreal" else "imag"
            return self._intrinsic_result(
                source_name,
                component,
                {
                    "op": "intrinsic",
                    "name": component,
                    "args": list(xs),
                    "type": target,
                },
                target,
                types,
                "fortran_complex_component_kind",
            )

        if name in {"real", "float", "sngl", "dble", "int", "ifix", "idint"} and len(
            xs
        ) in (1, 2):
            category = "i" if name in {"int", "ifix", "idint"} else "f"
            width = integer(xs[1]) if len(xs) == 2 else selected_kind
            if len(xs) == 2 and width not in (1, 2, 4, 8):
                raise Gap("kind", name + " dynamic kind")
            target = (
                f"{category}{8*width}"
                if width is not None
                else (
                    self.double
                    if name == "dble"
                    else self.default_integer if category == "i" else self.default_real
                )
            )
            value = (
                {
                    "op": "intrinsic",
                    "name": "real",
                    "args": [xs[0]],
                    "type": "f" + str(int(dtype[1:]) // 2),
                }
                if dtype.startswith("c")
                else xs[0]
            )
            return self._intrinsic_result(
                source_name,
                "int" if category == "i" else "real",
                cast(target, value),
                target,
                types,
                "fortran_conversion_intrinsic_kind_selection",
                selected_kind=width,
            )
        if name in {"aint", "dint"}:
            return self._intrinsic_result(
                source_name,
                "trunc",
                {
                    "op": "intrinsic",
                    "name": "trunc",
                    "type": dtype,
                    "args": [xs[0]],
                },
                dtype,
                types,
                "fortran_real_integral_value_same_kind",
            )
        name = _LEGACY_MATH.get(name, name)
        if name not in _MATH:
            raise Gap("intrinsic", name)
        result = dtype
        if name == "abs" and dtype.startswith("c"):
            result = "f" + str(int(dtype[1:]) // 2)
        return self._intrinsic_result(
            source_name,
            name,
            {
                "op": "intrinsic",
                "name": name,
                "args": list(xs),
                "type": result,
            },
            result,
            types,
            "fortran_intrinsic_argument_and_result_kinds",
        )

    def expression(self, node, element=None):
        expression, dtype = self._expression(node, element)
        if "type" in expression and expression["type"] != dtype:
            raise Gap(
                "expression_type",
                f"expression type {expression['type']} conflicts with inferred {dtype}",
                node,
            )
        return {"location": self.location(node), **expression, "type": dtype}, dtype

    def _expression(self, node, element=None):
        tag, parts = kind(node), getattr(node, "items", ())
        if tag == "Name":
            name = str(node).lower()
            return self.scalar(name, element), self.fact(name).dtype
        if tag in {
            "Int_Literal_Constant",
            "Signed_Int_Literal_Constant",
            "Real_Literal_Constant",
            "Signed_Real_Literal_Constant",
        }:
            numeric, explicit_kind = parts
            is_int = "Int" in tag
            dtype = (
                self.default_integer
                if is_int
                else self.double if "d" in numeric.lower() else self.default_real
            )
            if explicit_kind:
                size = str(explicit_kind)
                if not size.isdecimal():
                    size = (self.fact(size).value or "").split("_")[0]
                if size not in {"1", "2", "4", "8"}:
                    raise Gap("kind", f"unresolved literal kind {explicit_kind}", node)
                dtype = ("i" if is_int else "f") + str(8 * int(size))
            return number(numeric, dtype), dtype
        if tag == "Complex_Literal_Constant":
            real, imag = (self.expression(n)[0] for n in parts)
            dtype = "c" + str(2 * max(int(x["type"][1:]) for x in (real, imag)))
            return {
                "op": "intrinsic",
                "name": "complex",
                "args": [real, imag],
                "type": dtype,
            }, dtype
        if tag == "Logical_Literal_Constant":
            return literal("bool", str(parts[0]).lower() == ".true."), "bool"
        if tag == "Parenthesis":
            return self.expression(parts[1], element)
        if tag in {
            "Part_Ref",
            "Function_Reference",
            "Structure_Constructor",
            "Intrinsic_Function_Reference",
        }:
            name = str(parts[0]).lower()
            args = list(parts[1].items) if parts[1] else []
            resolved = self.symbols.get(name)
            if (
                tag == "Intrinsic_Function_Reference"
                or resolved
                and "INTRINSIC-PROC" in resolved.attrs
            ):
                return self.intrinsic(name, args, element)
            symbol = self.fact(name, node)
            if symbol.dims:
                if any(kind(a) == "Subscript_Triplet" for a in args):
                    return self.section_index(node, element), symbol.dtype
                if any(self.shape_of(a) for a in args):
                    raise Gap("vector_subscript", name, node)
                return (
                    self.memory_index(name, [self.expression(a)[0] for a in args]),
                    symbol.dtype,
                )
            if "PROCEDURE" in symbol.attrs:
                return self.call(name, args, symbol.dtype), symbol.dtype
            raise Gap("reference", name, node)
        if (
            len(parts) == 3
            and isinstance(parts[1], str)
            and parts[1].lower() in {*_OPERATORS, "**"}
        ):
            operator = parts[1].lower()
            left, a = self.expression(parts[0], element)
            right, b = self.expression(parts[2], element)
            if operator == "**":
                # Integer exponents retain the base type, including an integral result.
                dtype = a if b.startswith("i") else numeric_type(a, b)
                name = "powi" if b.startswith("i") else "pow"
                if name == "pow":
                    # Materialize Fortran's operand conversions in the graph;
                    # C++ overload promotion is a different language rule.
                    left = cast(dtype, left) if a != dtype else left
                    right = cast(dtype, right) if b != dtype else right
                return self._intrinsic_result(
                    "**",
                    name,
                    {
                        "op": "intrinsic",
                        "name": name,
                        "args": [left, right],
                        "type": dtype,
                    },
                    dtype,
                    [a, b],
                    (
                        "fortran_integer_exponent_preserves_base_kind"
                        if b.startswith("i")
                        else "fortran_numeric_exponent_kind_promotion"
                    ),
                    normalized_argument_types=[left["type"], right["type"]],
                )
            op = _OPERATORS[operator]
            dtype = numeric_type(a, b)
            if dtype.startswith("c"):
                left = cast(dtype, left) if a != dtype else left
                right = cast(dtype, right) if b != dtype else right
            return binary(op, left, right), "bool" if op in _RELATIONS else dtype
        if len(parts) == 2 and str(parts[0]).lower() in {"+", "-", ".not."}:
            value, dtype = self.expression(parts[1], element)
            operator = str(parts[0]).lower()
            return (
                value
                if operator == "+"
                else unary("neg" if operator == "-" else "not", value)
            ), ("bool" if operator == ".not." else dtype)
        raise Gap("expression", tag, node)

    def call(self, name, args, result_type):
        target = self.link_name(name)
        self.calls.add(target)
        fact = self.symbols.get(name)
        if fact and fact.dummy:
            raise Gap("procedure_argument", f"indirect call via {name}")
        actuals = []
        types = []
        writebacks = []
        lengths = []
        interface = self.interfaces.get(name)
        if interface and len(args) != len(interface["params"]):
            raise Gap("call_arity", name)

        def pointer(value, symbol, dtype, readonly):
            # A native INTENT(IN) dummy still has a mutable address when it
            # is forwarded. Only an interoperable IN dummy has a const base.
            if self.interoperable and symbol and symbol.intent == "in" and not readonly:
                return {
                    "op": "cast",
                    "mode": "qualification",
                    "source_type": ptr(dtype, True),
                    "type": ptr(dtype),
                    "value": value,
                }
            return value

        for position, arg in enumerate(args):
            formal = interface["params"][position] if interface else {}
            readonly = bool(
                interface and interface["bind_c"] and formal.get("intent") == "in"
            )
            arg_name = str(arg).lower() if kind(arg) == "Name" else ""
            symbol = self.symbols.get(arg_name)
            origin = arg_name or (
                str(arg.items[0]).lower() if kind(arg) == "Part_Ref" else ""
            )
            if origin in self.storage and not formal.get("value"):
                raise Gap(
                    "storage_escape", "reference to associated storage " + origin, arg
                )
            if formal.get("value"):
                if self.character_view(arg):
                    raise Gap(
                        "character_abi", "VALUE character procedure argument", arg
                    )
                value, dtype = self.expression(arg)
                actuals.append(value)
                types.append(dtype)
                continue
            if symbol and "PROCEDURE" in symbol.attrs:
                raise Gap("procedure_argument", "procedure callback " + arg_name, arg)
            chars = self.character_view(arg)
            if chars is not None:
                text, length = chars
                if kind(arg) == "Char_Literal_Constant" and not readonly:
                    text = {
                        "op": "cast",
                        "mode": "qualification",
                        "type": ptr("char"),
                        "value": text,
                    }
                elif kind(arg) != "Char_Literal_Constant":
                    source = str(arg if kind(arg) == "Name" else arg.items[0]).lower()
                    text = pointer(text, self.fact(source), "char", readonly)
                actuals.append(text)
                types.append(ptr("char", readonly))
                if not (interface and interface["bind_c"]):
                    lengths.append(cast(self.target.size_type, length))
                continue
            if symbol and symbol.dims:
                actuals.append(pointer(var(arg_name), symbol, symbol.dtype, readonly))
                types.append(ptr(symbol.dtype, readonly))
                continue
            if kind(arg) == "Part_Ref" and self.shape_of(arg):
                base = str(arg.items[0]).lower()
                symbol = self.fact(base)
                axes = self.section(arg)
                # A consecutive interval can be passed directly. The first
                # partial dimension may only be followed by scalar subscripts.
                contiguous = True
                partial = False
                for spec, (start, step, count) in zip(arg.items[1].items, axes):
                    if step is None:
                        partial = True
                    else:
                        if integer(step) != 1 or partial:
                            contiguous = False
                            break
                        if not all(p is None for p in spec.items):
                            partial = True
                length = product([n for _, step, n in axes if step is not None])
                if contiguous:
                    first = self.section_index(arg, literal("i64", 0))
                    actuals.append(
                        pointer(
                            choose(
                                binary("gt", length, literal("i64", 0)),
                                unary("addr", first),
                                var(base),
                            ),
                            symbol,
                            symbol.dtype,
                            readonly,
                        )
                    )
                else:
                    buffer = self.fresh("argument_buffer")
                    idx = self.fresh("copy_index")
                    cell = self.section_index(arg, var(idx))
                    tmp = index(var(buffer), var(idx))
                    self._prelude.append(
                        {
                            "op": "buffer",
                            "name": buffer,
                            "type": symbol.dtype,
                            "extent": length,
                        }
                    )
                    if formal.get("intent") != "out":
                        self._prelude.append(
                            {
                                "op": "map",
                                "index": idx,
                                "extent": length,
                                "independent": True,
                                "proof": "gather into fresh argument storage",
                                "body": [assign(tmp, cell)],
                            }
                        )
                    actuals.append(var(buffer))
                    # An implicit interface can modify an actual even when the
                    # caller's native dummy has INTENT(IN). Preserve copy-out.
                    if formal.get("intent") != "in" and not (
                        self.interoperable and symbol.intent == "in"
                    ):
                        writebacks.append(
                            {
                                "op": "map",
                                "index": idx,
                                "extent": length,
                                "independent": True,
                                "proof": "distinct destination addresses of a regular strided section",
                                "body": [assign(cell, tmp)],
                            }
                        )
                types.append(ptr(symbol.dtype, readonly))
                continue
            value, dtype = self.expression(arg)
            if value["op"] not in {"index", "var"} and not (
                value["op"] == "unary" and value["operator"] == "deref"
            ):
                tmp = self.fresh("arg")
                self._prelude.append(decl(tmp, dtype, value))
                value = var(tmp)
            if kind(arg) == "Part_Ref":
                symbol = self.symbols.get(str(arg.items[0]).lower())
            actuals.append(pointer(unary("addr", value), symbol, dtype, readonly))
            types.append(ptr(dtype, readonly))
        actuals.extend(lengths)
        if lengths:
            types.extend([self.target.size_type] * len(lengths))
        self.externs[target] = {
            "name": target,
            "params": [dict(name=f"arg{i}", type=t) for i, t in enumerate(types)],
            "result": result_type,
        }
        operation = {
            "op": "call",
            "callee": target,
            "args": actuals,
            "parameter_types": types,
            "parameter_type_basis": (
                "actual_types_with_explicit_passing"
                if interface
                else "actual_argument_abi"
            ),
        }
        if not writebacks:
            return operation
        if result_type == "void":
            self._prelude.extend([{"op": "expr", "value": operation}, *writebacks])
            return None
        saved = self.fresh("call_result")
        self._prelude.extend([decl(saved, result_type, operation), *writebacks])
        return var(saved)

    def masked_assignments(self, mask, arms):
        # Freeze the WHERE mask once; each successive statement reads the state
        # left by its predecessor. Evaluate RHS only for selected elements.
        shape = self.shape_of(mask)
        if not shape or any(n is None for n in shape):
            raise Gap("where", "unknown mask shape", mask)
        extent = product(shape)
        idx = self.fresh("where_index")
        mv = self.fresh("mask")
        at = var(idx)

        def mapping(body):
            return {
                "op": "map",
                "extent": extent,
                "index": idx,
                "independent": False,
                "body": body,
            }

        def element(node):
            old, views = self._prelude, self._views
            self._prelude = []
            self._views = {}
            try:
                value, dtype = self.expression(node, at)
                return self._prelude, value, dtype
            finally:
                self._prelude, self._views = old, views

        selected = index(var(mv), at)
        prefix, test, _ = element(mask)
        steps = [
            {
                "op": "buffer",
                "name": mv,
                "type": "bool",
                "extent": extent,
                "storage": "compiler_temporary",
            },
            mapping([*prefix, assign(selected, test)]),
        ]
        for include, assignments in arms:
            guard = selected if include else unary("not", selected)
            for assignment in assignments:
                if kind(assignment) != "Assignment_Stmt":
                    raise Gap("where", kind(assignment), assignment)
                target, _, rhs = assignment.items
                before, dest, dtype = element(target)
                pre, value, _ = element(rhs)
                scratch = self.fresh("masked_value")
                cell = index(var(scratch), at)
                steps.extend(
                    [
                        {
                            "op": "buffer",
                            "name": scratch,
                            "type": dtype,
                            "extent": extent,
                            "storage": "compiler_temporary",
                        },
                        mapping(
                            [
                                {
                                    "op": "if",
                                    "condition": guard,
                                    "then": [*pre, assign(cell, value)],
                                    "else": [],
                                }
                            ]
                        ),
                        mapping(
                            [
                                {
                                    "op": "if",
                                    "condition": guard,
                                    "then": [*before, assign(dest, cell)],
                                    "else": [],
                                }
                            ]
                        ),
                    ]
                )
        return {"op": "block", "body": steps}

    def forall_assignment(self, node):
        header, assignment = node.items
        triplets, mask = header.items
        if len(triplets.items) != 1 or mask is not None:
            raise Gap("forall", "multi-index or masked FORALL", node)
        idx, lower, upper, stride = triplets.items[0].items
        src = str(idx).lower()
        dt = self.fact(src).dtype
        idx = self.fresh("forall_index")
        iteration = self.fresh("forall_iteration")
        lo = self._once(self.expression(lower)[0])
        hi = self._once(self.expression(upper)[0])
        step = self._once(self.expression(stride)[0]) if stride else literal("i64", 1)
        length = self._once(
            {
                "op": "intrinsic",
                "name": "max",
                "type": "i64",
                "args": [
                    literal("i64", 0),
                    binary("div", binary("add", binary("sub", hi, lo), step), step),
                ],
            }
        )
        bound = decl(
            idx, dt, cast(dt, binary("add", lo, binary("mul", var(iteration), step)))
        )
        previous = self.aliases.copy()
        self.aliases[src] = idx
        try:
            target, _, rhs = assignment.items
            dest, typ = self.expression(target)
            value = self.expression(rhs)[0]
        finally:
            self.aliases = previous
        tmp = self.fresh("forall_values")
        v = index(var(tmp), var(iteration))

        def loop(stmt):
            return {
                "op": "map",
                "extent": length,
                "index": iteration,
                "independent": False,
                "body": [bound, stmt],
            }

        return {
            "op": "block",
            "body": [
                {"op": "buffer", "name": tmp, "type": typ, "extent": length},
                loop(assign(v, value)),
                loop(assign(dest, v)),
            ],
        }

    def _assignment(self, target, rhs):
        chars = self.character_view(target)
        if chars:
            return self.character_assignment(chars, rhs)
        target_shape = self.shape_of(target)
        if target_shape:
            name = str(target if kind(target) == "Name" else target.items[0]).lower()
            target_sym = self.fact(name)
            if any(length is None for length in target_shape):
                raise Gap("array_extent", f"unbound extent of {name}", target)
            indexed = list(
                _syntax_nodes(
                    rhs, {"Part_Ref", "Function_Reference", "Structure_Constructor"}
                )
            )
            procedures = [self.symbols.get(str(n.items[0]).lower()) for n in indexed]
            if any(
                s and "PROCEDURE" in s.attrs and "INTRINSIC-PROC" not in s.attrs
                for s in procedures
            ):
                raise Gap("array_effect", "array RHS contains a user procedure", rhs)
            referenced = {str(n).lower() for n in _syntax_nodes(rhs, {"Name"})}
            snapshot = name in referenced and (
                kind(target) != "Name"
                or any(str(n.items[0]).lower() == name for n in indexed)
            )
            for other in referenced & self.symbols.keys():
                s = self.fact(other)
                if (
                    other != name
                    and other in self.storage
                    and name in self.storage
                    and self.storage[other]["base"] == self.storage[name]["base"]
                ):
                    snapshot = True
                    continue
                if other == name or "PARAMETER" in s.attrs or not (s.byref or s.dims):
                    continue
                if (
                    target_sym.dummy
                    and s.dummy
                    and target_sym.restricted
                    and s.restricted
                ):
                    continue
                if (
                    not target_sym.attrs & {"POINTER", "TARGET", "IN-COMMON"}
                    and not s.attrs & {"POINTER", "TARGET", "IN-COMMON"}
                    and (not target_sym.dummy or not s.dummy)
                ):
                    continue
                snapshot = True
            element = var("_f_element")
            value, _ = self.expression(rhs, element)
            to = self.expression(target, element)[0]
            extent = product(target_shape)

            def mapping(dest, source, proof):
                return {
                    "op": "map",
                    "index": "_f_element",
                    "extent": extent,
                    "independent": True,
                    "proof": proof,
                    "body": [assign(dest, source)],
                }

            if snapshot:
                scratch = self.fresh("snapshot")
                cell = index(var(scratch), element)
                return {
                    "op": "block",
                    "body": [
                        {
                            "op": "buffer",
                            "name": scratch,
                            "type": target_sym.dtype,
                            "extent": extent,
                        },
                        mapping(cell, value, "disjoint local result buffer"),
                        mapping(
                            to, cell, "distinct elements of the destination section"
                        ),
                    ],
                }
            return mapping(
                to,
                value,
                "same-index elemental accesses and Fortran definable-dummy association",
            )
        value, _ = self.expression(rhs)
        return assign(self.expression(target)[0], value)

    @staticmethod
    def characters(node):
        if kind(node) != "Char_Literal_Constant":
            raise Gap("character_value", kind(node), node)
        value, kind_param = node.items
        if kind_param is not None and str(kind_param) != "1":
            raise Gap("character_kind", str(kind_param), node)
        quote = value[0]
        return value[1:-1].replace(quote + quote, quote)

    def runtime_call(self, leaf, args, profile=None):
        symbol = "_gfortran_" + leaf
        profile = profile or self.target.data_transfer()
        types = profile.signatures.get(symbol)
        if types is None or len(args) != len(types):
            raise Gap("io_abi", f"runtime argument declaration for {symbol}")
        self.externs[symbol] = dict(
            name=symbol,
            provider="libgfortran",
            soname=profile.soname,
            result="void",
            params=[dict(name=f"arg{i}", type=t) for i, t in enumerate(types)],
        )
        return {"op": "expr", "value": {"op": "call", "callee": symbol, "args": args}}

    def io_descriptor(self, node, profile, mode, unit):
        self.records[profile.record["name"]] = profile.record
        name = self.fresh("io")
        dt = var(name)

        def set_field(field, value):
            return assign({"op": "member", "base": dt, "field": field}, value)

        source = self.location(node)
        steps = [
            decl(
                name,
                {"kind": "record", "name": profile.record["name"]},
                {"op": "init_list", "items": []},
            ),
            set_field("filename", {"op": "string", "value": source["file"]}),
            set_field("line", literal("i32", source["line"])),
            set_field("flags", literal("i32", profile.flags[mode])),
            set_field("unit", unit),
        ]
        return steps, set_field, unary("addr", dt)

    def file_control(self, node):
        operation, specs = node.items
        operation = operation.lower()
        fields = {}
        for spec in specs.items:
            key, value = spec.items
            fields[str(key or "unit").lower()] = value
        if "unit" not in fields:
            raise Gap("io_control", operation + " requires UNIT", node)
        unit = self.expression(fields.pop("unit"))[0]
        profile = self.target.file_control(operation, tuple(sorted(fields)))
        steps, set_field, addr = self.io_descriptor(node, profile, operation, unit)
        for key, value in fields.items():
            if key == "recl":
                steps.append(set_field("recl_in", self.expression(value)[0]))
                continue
            chars = self.character_view(value)
            if chars is None:
                raise Gap("character_value", key, value)
            steps.extend([set_field(key, chars[0]), set_field(key + "_len", chars[1])])
        steps.append(self.runtime_call("st_" + operation, [addr], profile))
        return {"op": "block", "effects": ["IO", "ORDER"], "body": steps}

    def write(self, node):
        controls, items = node.items
        if kind(node) == "Print_Stmt":
            opts = {"unit": "*", "fmt": controls}
        else:
            opts = {}
            for i, spec in enumerate(controls.items):
                key, val = spec.items
                name = str(key).lower() if key else {0: "unit", 1: "fmt"}.get(i)
                if name not in {"unit", "fmt"}:
                    raise Gap("io_control", str(name), node)
                opts[name] = val
        if "unit" not in opts or self.character_view(opts["unit"]):
            raise Gap("io_format", "missing or internal unit", node)
        profile = self.target.data_transfer()
        unit_val = (
            literal("i32", profile.default_unit)
            if str(opts["unit"]) == "*"
            else self.expression(opts["unit"])[0]
        )
        fmt = opts.get("fmt")
        fmt_text = None
        if fmt is not None and str(fmt) != "*":
            if kind(fmt) == "Char_Literal_Constant":
                fmt_text = self.characters(fmt)
            elif str(fmt) in self.formats:
                fmt_text = self.formats[str(fmt)]
            else:
                raise Gap("io_format", f"runtime format {fmt}", node)
        mode = (
            "unformatted" if fmt is None else "list" if fmt_text is None else "format"
        )
        steps, field, addr = self.io_descriptor(node, profile, mode, unit_val)
        if fmt_text is not None:
            steps.extend(
                [
                    field("format", {"op": "string", "value": fmt_text}),
                    field("format_len", literal("i64", len(fmt_text.encode()))),
                ]
            )
        steps.append(self.runtime_call("st_write", [addr]))

        def transfer(item, element):
            characters = self.character_view(item)
            if characters:
                return [
                    self.runtime_call("transfer_character_write", [addr, *characters])
                ]
            value, dtype = self.expression(item, element)
            item_name = str(item).lower() if kind(item) == "Name" else ""
            logical = dtype == "bool" or (
                item_name in self.symbols
                and self.symbols[item_name].category == "LOGICAL"
            )
            if logical and dtype == "bool":
                dtype = self.default_integer
            if dtype[0] not in "ifc":
                raise Gap("io_type", dtype, item)
            leaf = (
                "logical"
                if logical
                else {"i": "integer", "f": "real", "c": "complex"}[dtype[0]]
            )
            size = int(dtype[1:]) // (16 if dtype.startswith("c") else 8)
            name = self.fresh("item")
            return [
                decl(name, dtype, value),
                self.runtime_call(
                    "transfer_" + leaf + "_write",
                    [addr, unary("addr", var(name)), literal("i32", size)],
                ),
            ]

        for item in (items.items if items else ()):
            before = self._prelude
            self._prelude = []
            shape = self.shape_of(item)
            if shape:
                if any(n is None for n in shape):
                    raise Gap("array_extent", "unknown array I/O bound", item)
                i = self.fresh("io_index")
                written = transfer(item, var(i))
                batch = [
                    {
                        "op": "for",
                        "init": [decl(i, "i64", literal("i64", 0))],
                        "condition": binary("lt", var(i), product(shape)),
                        "step": unary("preinc", var(i)),
                        "body": written,
                    }
                ]
            else:
                batch = transfer(item, None)
            prefix, self._prelude = self._prelude, before
            steps.extend([*prefix, *batch])
        steps.append(self.runtime_call("st_write_done", [addr]))
        return {"op": "block", "effects": ["IO", "ORDER"], "body": steps}

    def case_construct(self, node):
        selector, dtype = self.expression(node.content[0].items[0])
        local = self.fresh("selector")
        arms = []
        default = []
        for child in node.content[1:-1]:
            if kind(child) == "Case_Stmt":
                values = child.items[0].items[0]
                arm = []
                if values is None:
                    default = arm
                else:
                    arms.append((values.items, arm))
            else:
                arm.append(child)
        tail = self.statements(default)
        for values, arm in reversed(arms):
            cond = None
            for spec in values:
                if kind(spec) == "Case_Value_Range":
                    lo, hi = spec.items
                    test = None
                    if lo:
                        test = binary("ge", var(local), self.expression(lo)[0])
                    if hi:
                        high = binary("le", var(local), self.expression(hi)[0])
                        test = binary("land", test, high) if test else high
                else:
                    test = binary("eq", var(local), self.expression(spec)[0])
                cond = binary("lor", cond, test) if cond else test
            tail = [
                {
                    "op": "if",
                    "condition": cond,
                    "then": self.statements(arm),
                    "else": tail,
                }
            ]
        return {"op": "block", "body": [decl(local, dtype, selector), *tail]}

    def statements(self, nodes):
        body = []
        for node in nodes:
            before, views = self._prelude, self._views
            self._prelude = []
            self._views = {}
            try:
                steps = self.statement(node)
            except Gap as gap:
                if not getattr(getattr(gap.node, "item", None), "span", None):
                    gap.node = node
                raise
            prefix, self._prelude, self._views = self._prelude, before, views
            line = getattr(node, "item", None)
            label = getattr(line, "label", None)
            if label:
                body.append({"op": "label", "name": "f_label_" + str(label)})
            for step in steps:
                step.setdefault("location", self.location(node))
            # Statement-local temporaries cannot extend across a Fortran label.
            if prefix:
                body.append({"op": "block", "body": [*prefix, *steps]})
            else:
                body.extend(steps)
        return body

    def statement(self, node):
        tag, parts = kind(node), getattr(node, "items", ())
        if tag == "Assignment_Stmt":
            return [self._assignment(parts[0], parts[2])]
        if tag == "If_Stmt":
            return [
                {
                    "op": "if",
                    "condition": self.expression(parts[0])[0],
                    "then": self.statements([parts[1]]),
                    "else": [],
                }
            ]
        if tag == "If_Construct":
            clauses = []
            branch = None
            for child in node.content:
                name = kind(child)
                if name in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt"}:
                    branch = []
                    clauses.append(
                        (None if name == "Else_Stmt" else child.items[0], branch)
                    )
                    continue
                if name != "End_If_Stmt":
                    branch.append(child)
            tail = []
            for cond, contents in reversed(clauses):
                steps = self.statements(contents)
                if cond is None:
                    tail = steps
                else:
                    before = self._prelude
                    self._prelude = []
                    cond_value = self.expression(cond)[0]
                    prefix, self._prelude = self._prelude, before
                    tail = [
                        *prefix,
                        {
                            "op": "if",
                            "condition": cond_value,
                            "then": steps,
                            "else": tail,
                        },
                    ]
            return tail
        if tag in {
            "Block_Nonlabel_Do_Construct",
            "Block_Label_Do_Construct",
            "Action_Term_Do_Construct",
        }:
            first = node.content[0]
            loop = first.items[-1]
            if loop is None or not loop.items[1]:
                raise Gap("loop", "unbounded or conditional DO", first)
            iv, bounds = loop.items[1]
            iv = str(iv).lower()
            dtype = self.fact(iv).dtype
            last = node.content[-1]
            nested = (
                node.content[1:-1] if kind(last) == "End_Do_Stmt" else node.content[1:]
            )
            lo, hi, *stride = [self.expression(a)[0] for a in bounds]
            step = stride[0] if stride else literal(dtype, 1)
            limit = self.fresh("limit")
            increment = self.fresh("step")
            target = self.scalar(iv)
            constant = integer(step)
            if constant is None:
                condition = binary(
                    "lor",
                    binary(
                        "land",
                        binary("gt", var(increment), literal(dtype, 0)),
                        binary("le", target, var(limit)),
                    ),
                    binary(
                        "land",
                        binary("lt", var(increment), literal(dtype, 0)),
                        binary("ge", target, var(limit)),
                    ),
                )
            else:
                condition = binary("le" if constant > 0 else "ge", target, var(limit))
            steps = self.statements(nested)
            return [
                {
                    "op": "for",
                    "init": [
                        decl(limit, dtype, hi),
                        decl(increment, dtype, step),
                        assign(target, lo),
                    ],
                    "condition": condition,
                    "step": {
                        "op": "update",
                        "operator": "add",
                        "target": target,
                        "value": var(increment),
                    },
                    "body": steps,
                }
            ]
        if tag == "Forall_Stmt":
            return [self.forall_assignment(node)]
        if tag == "Where_Stmt":
            return [self.masked_assignments(parts[0], [(True, [parts[1]])])]
        if tag == "Where_Construct":
            mask = node.content[0].items[0]
            arms = [(True, [])]
            for item in node.content[1:-1]:
                if kind(item) == "Elsewhere_Stmt":
                    arms.append((False, []))
                else:
                    arms[-1][1].append(item)
            return [self.masked_assignments(mask, arms)]
        if tag in {"Write_Stmt", "Print_Stmt"}:
            return [self.write(node)]
        if tag in {"Open_Stmt", "Close_Stmt"}:
            return [self.file_control(node)]
        if tag == "Format_Stmt":
            return []
        if tag == "Stop_Stmt":
            code = parts[1]
            if kind(code) == "Char_Literal_Constant":
                text = self.characters(code)
                return [
                    self.runtime_call(
                        "stop_string",
                        [
                            {"op": "string", "value": text},
                            literal("u64", len(text.encode())),
                            literal("bool", False),
                        ],
                    )
                ]
            if code is None:
                return [
                    self.runtime_call(
                        "stop_string",
                        [
                            literal(ptr("void"), None),
                            literal("u64", 0),
                            literal("bool", False),
                        ],
                    )
                ]
            if re.fullmatch(r"[+-]?\d+", str(code)):
                return [
                    self.runtime_call(
                        "stop_numeric",
                        [literal("i32", int(str(code))), literal("bool", False)],
                    )
                ]
            raise Gap("stop_code", str(code), node)
        if tag == "Case_Construct":
            return [self.case_construct(node)]
        if tag == "Call_Stmt":
            call = self.call(
                str(parts[0]).lower(), list(parts[1].items) if parts[1] else [], "void"
            )
            return [{"op": "expr", "value": call}] if call is not None else []
        if tag == "Return_Stmt":
            if parts and parts[0] is not None:
                raise Gap("alternate_return", "alternate return", node)
            return [
                {
                    "op": "return",
                    **(
                        {"value": var(self.result_name)}
                        if self.result_name
                        else {"value": literal("i32", 0)} if self.main else {}
                    ),
                }
            ]
        if tag == "Continue_Stmt":
            return []
        if tag == "Goto_Stmt":
            return [{"op": "goto", "name": "f_label_" + str(parts[0])}]
        raise Gap("statement", tag, node)

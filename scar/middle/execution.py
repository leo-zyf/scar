"""Derive target-neutral lowering contracts and seal the execution IR product."""

from copy import deepcopy
from scar.contract import IR, document, member, write_json, walk_operations as _nodes


def _private_extents(function, global_names):
    """Consume M2 ownership facts; retain initialized i64 domain sources."""
    state = function.get("state_contract", {})
    return {
        name
        for name in state.get("private_scalars", ())
        if name not in global_names
        and state["scalars"][name]["type"] == "i64"
        and state["scalars"][name]["initialized"]
    }


def _at_paths(node, path=()):
    if isinstance(node, dict):
        if "op" in node:
            yield path, node
        members = node.items()
    elif isinstance(node, list):
        members = enumerate(node)
    else:
        return
    for key, child in members:
        yield from _at_paths(child, (*path, key))


def _coordinate_layout(stmt, private, definitions):
    """Offer a coordinate traversal only for proven stable integer radices."""

    def atom(expr):
        op = expr.get("op")
        if (
            op == "var"
            and expr.get("scope") != "global"
            and expr["name"] in private
            and expr["name"] != stmt["index"]
        ):
            return dict(op=op, name=expr["name"])
        if op == "literal" and expr["type"] == "i64" and type(expr["value"]) is int:
            return dict(op=op, type="i64", value=expr["value"])
        return None

    def factors(expr, expanded=frozenset()):
        if expr.get("op") == "binary" and expr["operator"] == "mul":
            return factors(expr["left"], expanded) + factors(expr["right"], expanded)
        if expr.get("op") == "var" and expr.get("scope") != "global":
            name = expr["name"]
            definition = definitions.get(name)
            if (
                name in private
                and name not in expanded
                and definition
                and stmt.get("site") in definition["uses"]
            ):
                return factors(definition["value"], expanded | {name})
        return [atom(expr)]

    axes = factors(stmt["extent"])
    if any(axis is None for axis in axes):
        return None
    # The original sequence is observable. Exclude control transfers, calls
    # and writes through an escaped extent before replacing any operation.
    if any(
        node["op"] != "assign" or node["target"]["op"] != "index"
        for node in stmt["body"]
    ):
        return None
    body = list(_at_paths(stmt["body"]))
    operations = {
        "assign",
        "index",
        "var",
        "literal",
        "binary",
        "unary",
        "cast",
        "select",
        "intrinsic",
    }
    if any(
        node["op"] not in operations
        or (
            node["op"] == "unary"
            and node["operator"] not in {"pos", "neg", "not", "bnot", "deref"}
        )
        for _, node in body
    ):
        return None
    coordinates, equalities = [], []
    for path, expr in body:
        if expr.get("op") != "binary" or expr["operator"] != "mod":
            continue
        lhs, radices = expr["left"], [atom(expr["right"])]
        while lhs.get("op") == "binary" and lhs["operator"] == "div":
            radices.append(atom(lhs["right"]))
            lhs = lhs["left"]
        if (
            lhs.get("op") != "var"
            or lhs["name"] != stmt["index"]
            or lhs.get("scope") == "global"
        ):
            continue
        if len(radices) > len(axes) or any(radix is None for radix in radices):
            continue
        coordinates.append(dict(path=list(path), axis=len(radices) - 1))
        for axis, radix in enumerate(reversed(radices)):
            equality = dict(axis=axis, radix=radix)
            if radix != axes[axis] and equality not in equalities:
                equalities.append(equality)
    if coordinates:
        return dict(
            kind="mixed_radix",
            order="first_axis_fastest",
            axes=axes,
            equalities=equalities,
            coordinates=coordinates,
            fallback="linear",
        )
    return None


def annotate_iteration_layouts(function, global_names=()):
    """M4 seals legal target-neutral traversal alternatives into the execution IR."""
    maps = [op for op in _nodes(function["body"]) if op["op"] == "map"]
    if not maps:
        return 0
    private = _private_extents(function, global_names)
    definitions = function.get("state_contract", {}).get("value_definitions", {})
    count = 0
    for stmt in maps:
        layout = _coordinate_layout(stmt, private, definitions)
        if layout:
            stmt["iteration_layout"] = layout
            count += 1
    return count


def annotate_scoped_arrays(function):
    """Prove whether a lexical continuation can own an automatic array lifetime."""

    def inspect(items, at_function_scope, loop=False, breakable=False):
        for node in items:
            op = node["op"]
            if (
                op in {"goto", "label"}
                or (op == "return" and not at_function_scope)
                or (op == "break" and not breakable)
                or (op == "continue" and not loop)
                or (op in {"case", "default"} and not breakable)
            ):
                raise ValueError(f"automatic array continuation crosses a {op} edge")
            nested_loop = op in {"for", "each", "while", "do_while", "map"}
            for field in ("body", "then", "else"):
                inspect(
                    node.get(field, ()),
                    at_function_scope,
                    loop or nested_loop,
                    breakable or nested_loop or op == "switch",
                )

    def sequence(items, at_function_scope=False):
        count = 0
        for index, node in enumerate(items):
            if node["op"] == "runtime_array":
                if (
                    node.get("storage") != "scoped"
                    or node.get("initialization") != "uninitialized"
                ):
                    raise ValueError(
                        "unsupported automatic array duration or initialization"
                    )
                inspect(items[index + 1 :], at_function_scope)
                node["scope_control"] = "lexical_continuation"
                count += 1
            for field in ("body", "then", "else", "reference"):
                count += sequence(node.get(field, ()))
            if node.get("op") == "schedule_choice":
                count += sequence([node["fused"]])
        return count

    return sequence(function["body"], True)


def prepare(module):
    prepared = deepcopy(module)
    globals_ = {g["name"] for g in prepared["globals"]}
    layouts = sum(
        annotate_iteration_layouts(fn, globals_) for fn in prepared["functions"]
    )
    lifetimes = sum(annotate_scoped_arrays(fn) for fn in prepared["functions"])
    return prepared, dict(iteration_layouts=layouts, scoped_lifetimes=lifetimes)


def seal(module, out, *, status, graph_sha256, contract, logical_optimization):
    payload = dict(module)
    functions = (
        [member(out, "functions", f) for f in payload.pop("functions", [])]
        if status == "READY"
        else []
    )
    product = document(
        IR,
        status=status,
        graph_sha256=graph_sha256,
        contract=contract,
        module=payload if status == "READY" else {"entry": module["entry"]},
        functions=functions,
        logical_optimization=logical_optimization,
    )
    path = out / "exec-ir.json"
    write_json(path, product)
    return path

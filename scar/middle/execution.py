"""Derive target-neutral lowering contracts and seal the execution IR product."""

from copy import deepcopy
from scar.contract import (
    ARTIFACT_FLOW,
    IR,
    VERSION,
    digest,
    document,
    json_bytes,
    member,
    validate_resource_flow,
    write_json,
    walk_operations as _nodes,
    control_regions,
)


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
            for owner, field, _ in control_regions(node):
                inspect(
                    owner[field],
                    at_function_scope,
                    loop or nested_loop,
                    breakable or nested_loop or op == "switch",
                )

    def sequence(items, at_function_scope=False):
        count = 0
        for index, node in enumerate(items):
            if node["op"] == "for" and any(
                s["op"] == "runtime_array" for s in node.get("init", ())
            ):
                loop = dict(node, init=[])
                initializers = node["init"]
                node.clear()
                node.update(op="block", body=[*initializers, loop])
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
            for owner, field, _ in control_regions(node):
                count += sequence(owner[field])
            if node.get("op") == "schedule_choice":
                count += sequence(node["reference"])
                count += sequence([node["fused"]])
        return count

    return sequence(function["body"], True)


def prepare(module, *, executable=True):
    from .resource_flow import analyze

    prepared = deepcopy(module) if executable else dict(module)
    layouts = lifetimes = 0
    if executable:
        globals_ = {g["name"] for g in prepared["globals"]}
        layouts = sum(
            annotate_iteration_layouts(fn, globals_) for fn in prepared["functions"]
        )
        lifetimes = sum(annotate_scoped_arrays(fn) for fn in prepared["functions"])
    inventory = dict(prepared.get("resource_model", {}))
    prepared["resource_model"] = inventory
    flow = validate_resource_flow(
        analyze(prepared, partial=not executable), inventory.get("resources", [])
    )
    prepared["resource_analysis"] = dict(
        graph=flow,
        inventory_sha256=digest(
            json_bytes(
                {
                    key: inventory.get(key)
                    for key in ("resources", "topology", "platform")
                }
            )
        ),
        coverage=flow["coverage"],
        scope="logical copies and source resource domains; runtime placement is not established",
    )
    inventory["resource_flow"] = flow
    return prepared, dict(
        iteration_layouts=layouts,
        scoped_lifetimes=lifetimes,
        resource_copies=flow["coverage"]["copies"],
        resource_coverage=flow["coverage"]["kind"],
    )


def seal(module, out, *, status, graph_sha256, contract, logical_optimization):
    if status == "READY" and any(
        fn.get("execution_kind") == "python_protocol" for fn in module["functions"]
    ):
        raise ValueError("execution IR cannot contain unresolved source protocols")
    payload = dict(module)
    # Only the versioned file graph crosses the execution boundary. Incomplete
    # protocol fragments remain in the logical planning report.
    payload.pop("artifact_analysis", None)
    payload.pop("resource_analysis", None)
    inventory = dict(payload.get("resource_model", {}))
    if "artifact_flow" in inventory:
        flow = inventory["artifact_flow"]
        if not isinstance(flow, dict) or (
            flow.get("schema"),
            flow.get("schema_version"),
        ) != (ARTIFACT_FLOW, VERSION):
            raise ValueError("M4 requires an artifact flow graph before sealing")
        inventory["artifact_flow"] = member(out, "analysis", flow)
    if status == "READY" and "resource_flow" in inventory:
        inventory["resource_flow"] = member(
            out, "resources", inventory["resource_flow"]
        )
    if "resource_model" in payload:
        payload["resource_model"] = inventory
    if status != "READY":
        payload = {"entry": module["entry"]}
        if "artifact_flow" in inventory:
            payload["resource_model"] = {"artifact_flow": inventory["artifact_flow"]}
    functions = [member(out, "functions", f) for f in payload.pop("functions", [])]
    product = document(
        IR,
        status=status,
        graph_sha256=graph_sha256,
        contract=contract,
        module=payload,
        functions=functions,
        logical_optimization=logical_optimization,
    )
    path = out / "exec-ir.json"
    write_json(path, product)
    return path

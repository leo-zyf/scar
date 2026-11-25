"""Derive target-neutral lowering contracts and seal the execution IR product."""

from collections import Counter
from copy import deepcopy
from math import prod
from .contracts import is_complete, publish_program, require_program
from scar.contract import (
    ARTIFACT_FLOW,
    IR,
    VERSION,
    digest,
    document,
    json_bytes,
    member,
    validate_resource_flow,
    validate_access_contract,
    validate_numeric_contract,
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
    if "extent_factors" in stmt:
        counts = stmt["extent_factors"]
        extent = stmt["extent"]
        if not (
            isinstance(counts, list)
            and len(counts) > 1
            and all(type(n) is int and n > 0 for n in counts)
            and extent.get("op") == "literal"
            and extent.get("type") == "i64"
            and type(extent.get("value")) is int
            and 0 < extent["value"] < 1 << 63
            and prod(counts) == extent["value"]
        ):
            raise ValueError("invalid folded map extent factorization")
        axes = [dict(op="literal", type="i64", value=n) for n in counts]
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


_SCALAR_TYPES = (
    "bool",
    "char",
    "i8",
    "u8",
    "i16",
    "u16",
    "i32",
    "u32",
    "i64",
    "u64",
    "f32",
    "f64",
)


def annotate_private_scratch(function, external_names):
    """Prove uses of a function-scope native buffer only through scalar elements."""
    body = function["body"]
    buffers = [node for node in body if node["op"] == "buffer"]
    if not buffers:
        return
    candidates = {
        node["name"]: node
        for node in buffers
        if node["type"] in _SCALAR_TYPES
        and node["name"] not in external_names
        and node.get("storage") in (None, "compiler_temporary")
        and set(node) <= {"op", "name", "type", "extent", "site", "location", "storage"}
    }
    if not candidates:
        return
    types = {}

    def bind(name, spec):
        types[name] = spec if name not in types or types[name] == spec else None

    for param in function["params"]:
        bind(param["name"], param["type"])
        candidates.pop(param["name"], None)
    nodes = list(_nodes(body))
    for node in nodes:
        if node["op"] in {"try", "goto", "label"}:
            return
        if node["op"] in {"decl", "each", "runtime_array", "buffer"}:
            ty = node["type"]
            if node["op"] in {"buffer", "runtime_array"}:
                ty = {"kind": "ptr", "to": ty}
            bind(node["name"], ty)
        elif node["op"] == "map":
            bind(node["index"], "i64")

    def local_type(node):
        if node.get("op") == "var" and node.get("scope") != "global":
            return types.get(node["name"])

    def scalar(node):
        op = node.get("op")
        if op == "var":
            return local_type(node) in _SCALAR_TYPES
        if op == "literal":
            return node.get("type") in _SCALAR_TYPES
        if op == "index" or op == "unary" and node["operator"] == "deref":
            base = node["base"] if op == "index" else node["value"]
            ty = local_type(base)
            return (
                isinstance(ty, dict)
                and ty.get("kind") == "ptr"
                and ty.get("to") in _SCALAR_TYPES
                and (
                    op != "index"
                    or len(node["indices"]) == 1
                    and scalar(node["indices"][0])
                )
            )
        if op == "cast":
            return (
                node.get("type") in _SCALAR_TYPES
                and node.get("mode", "static") == "static"
                and scalar(node["value"])
            )
        if op == "unary":
            return node["operator"] in {"pos", "neg", "not", "bnot"} and scalar(
                node["value"]
            )
        return (
            op == "binary"
            and node["operator"]
            in {
                "add",
                "sub",
                "mul",
                "div",
                "mod",
                "lt",
                "le",
                "gt",
                "ge",
                "eq",
                "ne",
                "lor",
                "land",
                "bor",
                "band",
                "bxor",
                "shl",
                "shr",
            }
            and scalar(node["left"])
            and scalar(node["right"])
        )

    # Count name occurrences, including binding positions and other implicit
    # operands. An unrecognized use or shadow cannot accidentally count as a load.
    mentions = Counter()

    def count_names(tree):
        if isinstance(tree, dict):
            for child in tree.values():
                count_names(child)
        elif isinstance(tree, list):
            for child in tree:
                count_names(child)
        elif isinstance(tree, str) and tree in candidates:
            mentions[tree] += 1

    for stmt in body:
        if stmt["op"] == "buffer" and mentions[stmt["name"]]:
            candidates.pop(stmt["name"], None)
        count_names(stmt)
    indexed = Counter()
    for node in nodes:
        if node["op"] != "assign" or not scalar(node["target"]):
            continue
        for expression in (node["target"], node["value"]):
            if scalar(expression):
                for use in _nodes(expression):
                    if use["op"] == "index" and use["base"].get("name") in candidates:
                        indexed[use["base"]["name"]] += 1
    for name, node in candidates.items():
        if indexed[name] and mentions[name] == 1 + indexed[name]:
            node["private_scratch"] = "indexed_scalar_access_only"


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
            if node["op"] == "buffer":
                node.pop("private_scratch", None)
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

    require_program(module, "M3")
    if executable and not is_complete(module):
        raise ValueError("M4 cannot prepare a partial program for execution")
    prepared = deepcopy(module) if executable else dict(module)
    layouts = lifetimes = 0
    if executable:
        access_roots = sum(
            len(validate_access_contract(fn)) for fn in prepared["functions"]
        )
        globals_ = {g["name"] for g in prepared["globals"]}
        layouts = sum(
            annotate_iteration_layouts(fn, globals_) for fn in prepared["functions"]
        )
        lifetimes = sum(annotate_scoped_arrays(fn) for fn in prepared["functions"])
        symbols = globals_ | {
            f["name"] for f in (*prepared["functions"], *prepared["externs"])
        }
        # A replacement allocator can observe the generated storage even when the
        # source never takes its address. Explicit source allocations have their
        # own IR operation and do not enter the native buffer proof.
        allocators = ("operator new", "operator delete", "_Znw", "_Zna", "_Zdl", "_Zda")
        observers = {
            "mallinfo",
            "mallinfo2",
            "malloc_info",
            "malloc_stats",
            "malloc_usable_size",
            "mtrace",
            "std::set_new_handler",
            "_ZSt15set_new_handlerPFvvE",
        }
        custom_heap = (
            any(name.startswith(allocators) for name in symbols)
            or bool(symbols & observers)
            or any(
                f["name"] in {"malloc", "free", "calloc", "realloc"}
                for f in prepared["functions"]
            )
        )
        candidates = [
            fn
            for fn in prepared["functions"]
            if any(n["op"] == "buffer" for n in fn["body"])
        ]
        if (
            candidates
            and not custom_heap
            and not any(
                node["op"] == "try"
                for fn in prepared["functions"]
                for node in _nodes(fn["body"])
            )
        ):
            for fn in candidates:
                annotate_private_scratch(fn, symbols)
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
    return publish_program(prepared, "M4", complete=executable), dict(
        iteration_layouts=layouts,
        scoped_lifetimes=lifetimes,
        invocation_access_roots=access_roots if executable else 0,
        resource_copies=flow["coverage"]["copies"],
        resource_coverage=flow["coverage"]["kind"],
    )


def seal(module, out, *, status, graph_sha256, contract, issues=()):
    require_program(module, "M4")
    if status not in {"READY", "INCOMPLETE"}:
        raise ValueError("invalid execution IR completion status")
    if status == "READY" and (issues or not is_complete(module)):
        raise ValueError("M4 cannot seal a partial program as executable")
    if status == "READY":
        for function in module["functions"]:
            validate_access_contract(function)
            validate_numeric_contract(function)
    if status == "READY" and any(
        fn.get("execution_kind") == "python_protocol" for fn in module["functions"]
    ):
        raise ValueError("execution IR cannot contain unresolved source protocols")
    payload = dict(module)
    # Only the versioned file graph crosses the execution boundary. Incomplete
    # protocol fragments remain in the logical planning report.
    payload.pop("artifact_analysis", None)
    payload.pop("resource_analysis", None)
    imports = payload.get("source_imports")
    if imports is not None:
        payload["source_imports"] = member(out, "analysis", imports)
    if isinstance(payload.get("source_obligations"), dict):
        payload["source_obligations"] = member(
            out, "analysis", payload["source_obligations"]
        )
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
        payload = {
            "entry": module["entry"],
            **(
                {"source_obligations": payload["source_obligations"]}
                if "source_obligations" in payload
                else {}
            ),
            **(
                {"source_imports": payload["source_imports"]}
                if imports is not None
                else {}
            ),
        }
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
        logical_optimization=True,
        **({"unresolved": member(out, "analysis", issues)} if issues else {}),
    )
    path = out / "exec-ir.json"
    write_json(path, product)
    return path

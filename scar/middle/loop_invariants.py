"""M3 motion of total integer expressions across structured loop boundaries."""

from collections import ChainMap
from functools import lru_cache

from scar.contract import (
    EXECUTION_ANNOTATIONS,
    control_regions,
    json_bytes,
    semantic_form,
    walk_operations,
)

_INTEGER = {
    prefix
    + str(bits): (
        -(1 << (bits - 1)) if prefix == "i" else 0,
        (1 << (bits - (prefix == "i"))) - 1,
    )
    for prefix in ("i", "u")
    for bits in (8, 16, 32, 64)
}
_INTEGER["bool"] = (0, 1)
_ARITHMETIC = {"i32", "u32", "i64", "u64"}
_COMPARE = {"lt", "le", "gt", "ge", "eq", "ne"}
_MUTATION = {"preinc", "postinc", "predec", "postdec"}
_LOOPS = {"for", "while", "do_while", "map", "each"}
_OPERANDS = {
    "assign": ("target", "value"),
    "binary": ("left", "right"),
    "call": ("args",),
    "cast": ("value",),
    "decl": ("init",),
    "expr": ("value",),
    "index": ("base", "indices"),
    "init_list": ("items",),
    "intrinsic": ("args",),
    "return": ("value",),
    "select": ("condition", "then", "else"),
    "unary": ("value",),
    "update": ("target", "value"),
}


def scalar_write(node):
    """Pick the directly mutated scalar slot of an ordinary canonical operator."""
    target = None
    if node["op"] in {"assign", "update"}:
        target = node["target"]
    elif node["op"] == "unary" and node.get("operator") in _MUTATION:
        target = node["value"]
    if target and target.get("op") == "var" and target.get("scope") != "global":
        return target["name"]
    return None


def _integer_type(spec):
    if isinstance(spec, dict) and spec.get("kind") == "const":
        spec = spec["of"]
    return spec if isinstance(spec, str) and spec in _INTEGER else None


def _bounded(kind, low, high):
    lower, upper = _INTEGER[kind]
    if lower <= low <= high <= upper:
        return kind, low, high
    if kind in {"u32", "u64"}:
        return kind, lower, upper
    return None


def _integer_binary(operator, left, right, result):
    kind, low, high = left
    right_kind, a, b = right
    if operator in _COMPARE and result == "bool":
        if kind == right_kind:
            return "bool", 0, 1
    if operator in {"land", "lor"} and result == "bool":
        return "bool", 0, 1
    if operator in {"shl", "shr"} and kind == result and kind in _ARITHMETIC:
        width = int(kind[1:])
        if 0 <= a <= b < width and low >= 0:
            if operator == "shl":
                return _bounded(kind, low << a, high << b)
            return kind, low >> b, high >> a
    if kind != right_kind or kind != result or kind not in _ARITHMETIC:
        return None
    if operator == "add":
        return _bounded(kind, low + a, high + b)
    if operator == "sub":
        return _bounded(kind, low - b, high - a)
    if operator == "mul":
        corners = (low * a, low * b, high * a, high * b)
        return _bounded(kind, min(corners), max(corners))
    if operator in {"band", "bor", "bxor"}:
        return kind, *_INTEGER[kind]
    if operator in {"div", "mod"} and (a > 0 or b < 0):
        if kind.startswith("i") and low == _INTEGER[kind][0] and b == -1:
            return None
        return kind, *_INTEGER[kind]
    return None


def _integer_value(node, incoming, called=None):
    """Return the result type and bounds only for a total, effect-free expression."""
    if not isinstance(node, dict):
        return None
    if node.get("effects") or node.get("state_writes") or node.get("may_raise"):
        return None
    op, kind = node.get("op"), _integer_type(node.get("type"))
    if op == "var" and node.get("scope") in {None, "local", "parameter"}:
        value = incoming.get(node["name"])
        if value and ("type" not in node or value[0] == kind):
            return value
    if op == "literal" and kind and type(node["value"]) in (int, bool):
        value = int(node["value"])
        low, high = _INTEGER[kind]
        if low <= value <= high:
            return kind, value, value
    if op == "binary" and (kind or "type" not in node):
        left = _integer_value(node["left"], incoming, called)
        right = _integer_value(node["right"], incoming, called)
        if left and right:
            operator = node["operator"]
            result = kind or (
                "bool"
                if operator in _COMPARE or operator in {"land", "lor"}
                else left[0]
            )
            return _integer_binary(operator, left, right, result)
    if op in {"cast", "unary"} and (kind or "type" not in node):
        source = _integer_value(node["value"], incoming, called)
        if source is None:
            return None
        source_kind, low, high = source
        if op == "cast":
            if not kind or node.get("mode", "static") != "static":
                return None
            if kind == "bool":
                return kind, 0, 1
            lower, upper = _INTEGER[kind]
            if lower <= low <= high <= upper:
                return kind, low, high
        elif node["operator"] == "not" and kind in {None, "bool"}:
            return "bool", 0, 1
        elif source_kind in _ARITHMETIC and (not kind or source_kind == kind):
            kind = source_kind
            if node["operator"] == "bnot":
                return kind, *_INTEGER[kind]
            if node["operator"] == "neg":
                return _bounded(kind, -high, -low)
    if op == "call" and called is not None:
        args = tuple(_integer_value(arg, incoming, called) for arg in node["args"])
        if all(args):
            value = called(node["callee"], args)
            if value and ("type" not in node or kind == value[0]):
                return value
    return None


def summarize_calls(functions):
    """Prove total integer callees from the same sealed program without running them."""
    summaries = {}
    for fn in functions:
        result = _integer_type(fn["result"])
        if not result or fn.get("method") or fn.get("initializers"):
            continue
        private = set(fn.get("state_contract", {}).get("private_scalars", ()))
        params = [(p["name"], _integer_type(p["type"])) for p in fn["params"]]
        if any(name not in private or kind is None for name, kind in params):
            continue
        body = fn["body"]
        if not body or body[-1]["op"] != "return" or "value" not in body[-1]:
            continue
        if any(
            step["op"] != "decl"
            or step["name"] not in private
            or not _integer_type(step["type"])
            or "init" not in step
            for step in body[:-1]
        ):
            continue
        if any(step.get("effects") or step.get("state_writes") for step in body):
            continue
        summaries[fn["name"]] = (params, body, result)

    active = set()

    @lru_cache(maxsize=2048)
    def evaluate(name, arguments):
        summary = summaries.get(name)
        if summary is None or name in active:
            return None
        params, body, result = summary
        if [value[0] for value in arguments] != [kind for _, kind in params]:
            return None
        incoming = {p[0]: value for p, value in zip(params, arguments)}
        active.add(name)
        try:
            for step in body[:-1]:
                value = _integer_value(step["init"], incoming, evaluate)
                if value is None or value[0] != _integer_type(step["type"]):
                    return None
                incoming[step["name"]] = value
            value = _integer_value(body[-1]["value"], incoming, evaluate)
            return value if value and value[0] == result else None
        finally:
            active.remove(name)

    return evaluate


def _project_aliases(node, aliases, visible):
    """Use a reaching snapshot only where its preheader definition is visible."""
    op = node["op"]
    if op == "var" and node.get("scope") in {None, "local", "parameter"}:
        origin = aliases.get(node["name"])
        if origin and origin[0] in visible:
            return {**node, "name": origin[1]}
    if op in {"binary", "cast", "unary"}:
        operands = {
            k: _project_aliases(node[k], aliases, visible) for k in _OPERANDS[op]
        }
        if any(value is not node[k] for k, value in operands.items()):
            return {**node, **operands}
    if op == "call":
        args = [_project_aliases(arg, aliases, visible) for arg in node["args"]]
        if any(new is not old for new, old in zip(args, node["args"])):
            return {**node, "args": args}
    return node


def _writes(value, cache, private, inventory):
    """Compute may-definitions once, including backedges and nested loop bindings."""
    if isinstance(value, list):
        return set().union(
            *(_writes(node, cache, private, inventory) for node in value)
        )
    if not isinstance(value, dict):
        return set()
    op = value.get("op")
    names = set()
    for key, child in value.items():
        if key not in EXECUTION_ANNOTATIONS:
            names.update(_writes(child, cache, private, inventory))
    if op is None:
        return names
    if op in _LOOPS or op in {"goto", "label"}:
        inventory["control"].add(op)
    inventory["names"].update(
        value[key] for key in ("name", "index") if isinstance(value.get(key), str)
    )
    if value.get("state_writes") or "OPAQUE" in (value.get("effects") or ()):
        names.update(private)
    if op in {"decl", "each", "buffer", "runtime_array"}:
        names.add(value["name"])
    elif op == "map":
        names.add(value["index"])
    elif name := scalar_write(value):
        names.add(name)
    if names:
        cache[id(value)] = names
    return names


def hoist(function, changes, register, called):
    """Move a value to its outermost proven preheader, preserving every use."""
    state = function.get("state_contract", {})
    private = {
        name: kind
        for name in state.get("private_scalars", ())
        if (kind := _integer_type(state["scalars"][name]["type"]))
    }
    occupied = {p["name"] for p in function["params"]}
    inventory = {"names": occupied, "control": set()}
    writes = {}
    _writes(function["body"], writes, private, inventory)
    control = inventory["control"]
    if not (control & _LOOPS) or control & {"goto", "label"}:
        return
    active = []
    slots, origins = {}, {}

    def expression(node, aliases):
        if isinstance(node, list):
            return [expression(item, aliases) for item in node]
        if not isinstance(node, dict):
            return node
        op = node.get("op")
        if op in {"binary", "unary", "cast", "call"}:
            visible = set()
            for loop, invariant, definitions, cached in active:
                visible.add(id(loop))
                projected = (
                    _project_aliases(node, aliases, visible) if aliases else node
                )
                bounds = ChainMap(invariant, slots) if aliases else invariant
                if not (value := _integer_value(projected, bounds, called)):
                    continue
                kind = value[0]
                if (
                    op == "cast"
                    and _integer_value(projected["value"], bounds, called)[0] == kind
                ):
                    break
                form = semantic_form(projected)
                dependencies = sorted(
                    {
                        item["name"]
                        for item in walk_operations(form)
                        if item["op"] == "var"
                    }
                )
                callees = sorted(
                    {
                        item["callee"]
                        for item in walk_operations(form)
                        if item["op"] == "call"
                    }
                )
                if not dependencies and not callees:
                    break
                key = json_bytes(form)
                if key not in cached:
                    name = f"scar_invariant_{len(occupied)}"
                    while name in occupied:
                        name += "_"
                    occupied.add(name)
                    cached[key] = name
                    slots[name], origins[name] = value, (id(loop), name)
                    definitions.append(
                        dict(
                            op="decl",
                            name=name,
                            type={"kind": "const", "of": kind},
                            init=projected,
                        )
                    )
                    register(
                        function, name, kind, "loop_invariant_motion", node.get("site")
                    )
                    proof = "total typed integer expression over initialized frame-private values without backedge writes"
                    function["state_contract"].setdefault("value_definitions", {})[
                        name
                    ] = dict(
                        value=form,
                        loop=loop.get("site"),
                        dependencies=dependencies,
                        bounds=list(value[1:]),
                        **({"callees": callees} if callees else {}),
                        proof=proof,
                    )
                    changes.append(
                        dict(
                            rule="loop_invariant_motion",
                            site=node.get("site"),
                            loop=loop.get("site"),
                            slot=name,
                            dependencies=dependencies,
                            **({"callees": callees} if callees else {}),
                            proof=proof,
                        )
                    )
                # Preserve the value category even when a consumer binds a reference.
                return dict(
                    op="cast",
                    type=kind,
                    value=dict(op="var", name=cached[key]),
                    **{k: node[k] for k in ("location", "site") if k in node},
                )
        for key in _OPERANDS.get(op, ()):
            if key in node:
                node[key] = expression(node[key], aliases)
        return node

    def bind(name, node, incoming, aliases, previous, previous_aliases):
        kind = private[name]
        incoming[name] = kind, *_INTEGER[kind]
        if not isinstance(node, dict):
            return
        value = _integer_value(node, ChainMap(previous, slots), called)
        if value and value[0] == kind:
            incoming[name] = value
            if node["op"] == "var" and node.get("scope") != "global":
                origin = previous_aliases.get(node["name"])
            elif node["op"] == "cast" and node["value"]["op"] == "var":
                origin = origins.get(node["value"]["name"])
                if origin and slots[origin[1]][0] != kind:
                    origin = None
            else:
                origin = None
            if origin:
                aliases[name] = origin

    def sequence(body, incoming, aliases):
        result = []
        for statement in body:
            op = statement["op"]
            previous = incoming.copy()
            previous_aliases = aliases.copy()
            changed = writes.get(id(statement), set())
            for name in changed & incoming.keys():
                kind = incoming[name][0]
                incoming[name] = kind, *_INTEGER[kind]
                aliases.pop(name, None)
            if op in _LOOPS:
                entry, entry_aliases = previous, previous_aliases
                if op == "for":
                    statement["init"] = sequence(
                        statement["init"], entry, entry_aliases
                    )
                    changing = writes.get(id(statement.get("condition")), set()).union(
                        writes.get(id(statement.get("step")), ()),
                        *(writes.get(id(child), ()) for child in statement["body"]),
                    )
                else:
                    changing = changed
                    for field in ("extent", "range"):
                        if field in statement:
                            statement[field] = expression(
                                statement[field], previous_aliases
                            )
                invariant = {
                    name: value for name, value in entry.items() if name not in changing
                }
                definitions = []
                active.append((statement, invariant, definitions, {}))
                loop_aliases = {
                    name: origin
                    for name, origin in entry_aliases.items()
                    if name not in changing
                }
                for field in ("condition", "step"):
                    if field in statement:
                        statement[field] = expression(statement[field], loop_aliases)
                loop_body = invariant.copy()
                loop_body.update(
                    {
                        name: (entry[name][0], *_INTEGER[entry[name][0]])
                        for name in changing & entry.keys()
                    }
                )
                statement["body"] = sequence(statement["body"], loop_body, loop_aliases)
                active.pop()
                if op == "for":
                    statement["init"].extend(definitions)
                else:
                    result.extend(definitions)
            elif op != "switch":
                if op == "if":
                    statement["condition"] = expression(statement["condition"], aliases)
                if active:
                    # Assignment operands can themselves mutate locals. Drop
                    # any projected alias the statement might redefine first.
                    expression(statement, aliases)
                for owner, field, _ in control_regions(statement):
                    owner[field] = sequence(
                        owner[field], incoming.copy(), aliases.copy()
                    )
                if (
                    op == "decl"
                    and (name := statement["name"]) in private
                    and "init" in statement
                ):
                    bind(
                        name,
                        statement["init"],
                        incoming,
                        aliases,
                        previous,
                        previous_aliases,
                    )
                elif op == "assign" and (name := scalar_write(statement)) in private:
                    bind(
                        name,
                        statement["value"],
                        incoming,
                        aliases,
                        previous,
                        previous_aliases,
                    )
            result.append(statement)
        return result

    parameters = {
        name: (private[name], *_INTEGER[private[name]])
        for p in function["params"]
        if (name := p["name"]) in private
    }
    function["body"] = sequence(function["body"], parameters, {})

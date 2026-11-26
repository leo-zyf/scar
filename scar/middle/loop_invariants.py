"""M3 scalar value proofs and motion across structured loop boundaries."""

from collections import ChainMap, Counter, deque
from functools import lru_cache
import math

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
_EXACT_FLOAT = {"f32": 1 << 24, "f64": 1 << 53}
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


def scalar_type(spec):
    if isinstance(spec, dict) and spec.get("kind") == "const":
        spec = spec["of"]
    if isinstance(spec, str) and (spec in _INTEGER or spec in _EXACT_FLOAT):
        return spec
    return None


def integer_domain(kind):
    """Return a full type range only when it makes an exact-value guarantee."""
    if isinstance(kind, str) and kind in _INTEGER:
        return kind, *_INTEGER[kind]
    return None


def _bounded(kind, low, high):
    limit = _EXACT_FLOAT.get(kind)
    lower, upper = (-limit, limit) if limit is not None else _INTEGER[kind]
    if lower <= low <= high <= upper:
        return kind, low, high
    if kind in {"u32", "u64"}:
        return kind, lower, upper
    return None


def _binary_value(operator, left, right, result):
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
    if kind != right_kind or kind != result:
        return None
    if kind in _EXACT_FLOAT:
        # All proved operands are integral and finite. A cancellation to zero
        # in add/sub can still pick a different sign under FE_DOWNWARD.
        if operator in {"add", "sub"}:
            first, last = (
                (low + a, high + b) if operator == "add" else (low - b, high - a)
            )
            if not first <= 0 <= last:
                return _bounded(kind, first, last)
        if operator == "mul":
            corners = (low * a, low * b, high * a, high * b)
            return _bounded(kind, min(corners), max(corners))
        if operator == "div" and a == b and a in {-1, 1}:
            return (kind, low, high) if a == 1 else (kind, -high, -low)
        return None
    if kind not in _ARITHMETIC:
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


def _has_effects(node):
    return any(
        node.get(field)
        for field in (
            "effects",
            "volatile",
            "atomic",
            "reads",
            "writes",
            "state_writes",
            "may_raise",
            "ordering_effect",
            "external_effect",
        )
    )


def exact_scalar_value(node, incoming, called=None):
    """Return the result type and bounds only for a total, effect-free expression."""
    if not isinstance(node, dict):
        return None
    if _has_effects(node):
        return None
    op, kind = node.get("op"), scalar_type(node.get("type"))
    if op == "var" and node.get("scope") in {None, "local", "parameter"}:
        value = incoming.get(node["name"])
        if value and ("type" not in node or value[0] == kind):
            return value
    if op == "literal" and kind:
        value = node["value"]
        if kind in _EXACT_FLOAT:
            if type(value) is float and math.isfinite(value) and value.is_integer():
                value = int(value)
            if type(value) is int and abs(value) <= _EXACT_FLOAT[kind]:
                return kind, value, value
        elif type(value) in (int, bool):
            low, high = _INTEGER[kind]
            if low <= value <= high:
                return kind, int(value), int(value)
    if op == "binary" and (kind or "type" not in node):
        left = exact_scalar_value(node["left"], incoming, called)
        right = exact_scalar_value(node["right"], incoming, called)
        if left and right:
            operator = node["operator"]
            result = kind or (
                "bool"
                if operator in _COMPARE or operator in {"land", "lor"}
                else left[0]
            )
            return _binary_value(operator, left, right, result)
    if op in {"cast", "unary"} and (kind or "type" not in node):
        source = exact_scalar_value(node["value"], incoming, called)
        if source is None:
            return None
        source_kind, low, high = source
        if op == "cast":
            if (
                not kind
                or node.get("mode", "static") != "static"
                or node.get("source_type", source_kind) != source_kind
                or node.get("target_type", kind) != kind
            ):
                return None
            if kind == "bool":
                return kind, 0, 1
            result = _bounded(kind, low, high)
            if result and result[1:] == (low, high):
                return result
        elif node["operator"] == "not" and kind in {None, "bool"}:
            return "bool", 0, 1
        elif (source_kind in _ARITHMETIC or source_kind in _EXACT_FLOAT) and (
            not kind or source_kind == kind
        ):
            kind = source_kind
            if node["operator"] == "bnot" and kind in _INTEGER:
                return kind, *_INTEGER[kind]
            if node["operator"] == "neg":
                return _bounded(kind, -high, -low)
    if op == "call" and called is not None:
        args = tuple(exact_scalar_value(arg, incoming, called) for arg in node["args"])
        if all(args):
            value = called(node["callee"], args)
            if value and ("type" not in node or kind == value[0]):
                return value
    return None


def summarize_calls(functions):
    """Prove total scalar callees from the same sealed program without running them."""
    summaries = {}
    for fn in functions:
        result = scalar_type(fn["result"])
        if not result or fn.get("method") or fn.get("initializers"):
            continue
        private = set(fn.get("state_contract", {}).get("private_scalars", ()))
        params = [(p["name"], scalar_type(p["type"])) for p in fn["params"]]
        if any(name not in private or kind is None for name, kind in params):
            continue
        body = fn["body"]
        if not body or body[-1]["op"] != "return" or "value" not in body[-1]:
            continue
        if any(
            step["op"] != "decl"
            or step["name"] not in private
            or not scalar_type(step["type"])
            or "init" not in step
            for step in body[:-1]
        ):
            continue
        if any(map(_has_effects, body)):
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
                value = exact_scalar_value(step["init"], incoming, evaluate)
                if value is None or value[0] != scalar_type(step["type"]):
                    return None
                incoming[step["name"]] = value
            value = exact_scalar_value(body[-1]["value"], incoming, evaluate)
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


def _discard_unused_snapshots(function, candidates, changes):
    """Cascade zero-consumer deletions through the proved scalar definition graph."""
    if not candidates:
        return

    def reads(tree):
        return (
            op["name"]
            for op in walk_operations(tree, exclude_fields=EXECUTION_ANNOTATIONS)
            if op["op"] == "var" and op.get("scope") != "global"
        )

    remaining = Counter(reads(function["body"]))
    pending = deque(name for name in candidates if not remaining[name])
    removals = {}
    state = function["state_contract"]
    removed_names = set()
    while pending:
        name = pending.popleft()
        body, declaration = candidates.pop(name)
        removed_names.add(name)
        for field in ("scalars", "introduced", "value_definitions"):
            state.get(field, {}).pop(name, None)
        removals.setdefault(id(body), (body, set()))[1].add(id(declaration))
        changes.append(
            dict(
                rule="discard_unused_scalar_snapshot",
                site=declaration.get("site") or declaration["init"].get("site"),
                slot=name,
                proof="total private scalar definition with zero remaining consumers",
            )
        )
        for dependency in reads(declaration["init"]):
            remaining[dependency] -= 1
            if dependency in candidates and remaining[dependency] == 0:
                pending.append(dependency)
    for body, members in removals.values():
        body[:] = [statement for statement in body if id(statement) not in members]
    if removed_names:
        state["private_scalars"] = [
            name for name in state["private_scalars"] if name not in removed_names
        ]


def hoist(function, changes, register, called):
    """Move a value to its outermost proven preheader, preserving every use."""
    state = function.get("state_contract", {})
    private = {
        name: kind
        for name in state.get("private_scalars", ())
        if (kind := scalar_type(state["scalars"][name]["type"]))
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
    candidates = {}

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
                if not (value := exact_scalar_value(projected, bounds, called)):
                    continue
                kind = value[0]
                if (
                    op == "cast"
                    and exact_scalar_value(projected["value"], bounds, called)[0]
                    == kind
                ):
                    break
                form = semantic_form(projected)
                inputs, called_symbols = set(), set()
                for item in walk_operations(form):
                    if item["op"] == "var":
                        inputs.add(item["name"])
                    elif item["op"] == "call":
                        called_symbols.add(item["callee"])
                dependencies, callees = sorted(inputs), sorted(called_symbols)
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
                    proof = "total typed expression with exact values over initialized frame-private inputs without backedge writes"
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
        if domain := integer_domain(kind):
            incoming[name] = domain
        else:
            incoming.pop(name, None)
        if not isinstance(node, dict):
            return
        value = exact_scalar_value(node, ChainMap(previous, slots), called)
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
                if domain := integer_domain(incoming[name][0]):
                    incoming[name] = domain
                else:
                    del incoming[name]
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
                        name: domain
                        for name in changing & entry.keys()
                        if (domain := integer_domain(entry[name][0]))
                    }
                )
                statement["body"] = sequence(statement["body"], loop_body, loop_aliases)
                active.pop()
                destination = statement["init"] if op == "for" else result
                destination.extend(definitions)
                candidates.update(
                    (decl["name"], (destination, decl)) for decl in definitions
                )
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
                    if name in aliases and not _has_effects(statement):
                        candidates[name] = (result, statement)
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
        name: domain
        for p in function["params"]
        if (name := p["name"]) in private
        if (domain := integer_domain(private[name]))
    }
    function["body"] = sequence(function["body"], parameters, {})
    _discard_unused_snapshots(function, candidates, changes)

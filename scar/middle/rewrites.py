"""Local equivalence rules and legality proofs for pointwise iteration domains."""

from scar.contract import semantic_form as semantics


def _fold(expr, changes):
    if isinstance(expr, list):
        return [_fold(v, changes) for v in expr]
    if not isinstance(expr, dict):
        return expr
    expr = {k: _fold(v, changes) for k, v in expr.items()}
    if expr.get("op") != "binary" or expr["operator"] not in {"add", "sub", "mul"}:
        return expr
    a, b = expr["left"], expr["right"]
    t = a.get("type")
    if (
        a["op"] != "literal"
        or b["op"] != "literal"
        or t != b.get("type")
        or t not in ("i32", "i64")
    ):
        return expr
    x, y = a["value"], b["value"]
    if type(x) is not int or type(y) is not int:
        return expr
    value = {"add": x + y, "sub": x - y, "mul": x * y}[expr["operator"]]
    limit = 1 << (int(t[1:]) - 1)
    if not -limit <= value < limit:
        return expr
    changes.append(
        dict(
            rule="constant_fold",
            site=expr.get("site"),
            proof="exact representable integer",
        )
    )
    return dict(op="literal", type=t, value=value)


def _indexed(expr, index):
    return (
        expr.get("op") == "index"
        and expr["base"].get("op") == "var"
        and semantics(expr["indices"]) == [{"op": "var", "name": index}]
    )


def _reads(expr, index, params):
    if _indexed(expr, index):
        name = expr["base"]["name"]
        return {name} if name in params else None
    if expr.get("op") == "literal":
        return set()
    if expr.get("op") == "binary" and expr["operator"] in {"add", "mul", "sub", "div"}:
        left, right = (_reads(expr[k], index, params) for k in ("left", "right"))
        return left | right if left is not None and right is not None else None
    if expr.get("op") == "var" and expr["name"] in params.get(None, ()):
        return set()
    return None


def _map(stmt, params):
    if stmt["op"] != "map" or not stmt.get("independent") or len(stmt["body"]) != 1:
        return None
    assignment = stmt["body"][0]
    index = stmt["index"]
    if assignment["op"] != "assign" or not _indexed(assignment["target"], index):
        return None
    dest = assignment["target"]["base"]["name"]
    pt = params.get(dest, {})
    if (
        not pt.get("elementwise_noalias")
        or _reads(assignment["value"], index, params) is None
    ):
        return None
    if pt.get("type", {}).get("to") not in ("f32", "f64"):
        return None
    return dest


def _forward(value, index, stored):
    if isinstance(value, list):
        return [_forward(v, index, stored) for v in value]
    if not isinstance(value, dict):
        return value
    if _indexed(value, index) and value["base"]["name"] in stored:
        return dict(op="var", name=stored[value["base"]["name"]])
    return {k: _forward(v, index, stored) for k, v in value.items()}


def _nodes(value):
    if isinstance(value, dict):
        if "op" in value:
            yield value
        for child in value.values():
            yield from _nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nodes(child)


def _introduced_scalar(fn, name, type_, rule, site):
    state = fn.setdefault("state_contract", {})
    private = state.setdefault("private_scalars", [])
    private.append(name)
    private.sort()
    state.setdefault("scalars", {})[name] = dict(type=type_, initialized=True)
    state.setdefault("introduced", {})[name] = dict(rule=rule, site=site)


def _share_extents(fn, changes):
    """Value-number repeated integer iteration domains at their first evaluation."""
    bindings = {s["name"]: s["type"] for s in fn["body"] if s["op"] == "decl"}
    reserved = {n["name"] for n in _nodes(fn["body"]) if "name" in n}
    reserved.update(p["name"] for p in fn["params"])
    private = set(fn.get("state_contract", {}).get("private_scalars", ()))
    bindings = {name: spec for name, spec in bindings.items() if name in private}

    def scalar(e):
        if (
            e["op"] == "var"
            and e.get("scope") != "global"
            and bindings.get(e["name"]) in ("i64", "i32")
        ):
            return bindings[e["name"]]
        if e["op"] == "literal" and e["type"] in ("i64", "i32"):
            return e["type"]
        if e["op"] == "binary" and e["operator"] in {"mul", "add", "sub"}:
            a, b = scalar(e["left"]), scalar(e["right"])
            return a if a and a == b else None
        return None

    body = fn["body"]
    begin = 0
    while begin < len(body):
        first = body[begin]
        if (
            first["op"] != "map"
            or first["extent"]["op"] != "binary"
            or not (t := scalar(first["extent"]))
        ):
            begin += 1
            continue
        expr = semantics(first["extent"])
        end = begin
        while (
            end < len(body)
            and body[end]["op"] == "map"
            and semantics(body[end]["extent"]) == expr
        ):
            if any(
                n["op"] not in {"assign", "var", "index", "literal", "binary", "cast"}
                or n["op"] == "assign"
                and n["target"]["op"] != "index"
                for n in _nodes(body[end]["body"])
            ):
                break
            end += 1
        if end - begin < 2:
            begin += 1
            continue
        slot = "scar_domain_size"
        while slot in reserved:
            slot += "_"
        reserved.add(slot)
        changes.append(
            dict(
                rule="common_iteration_extent",
                site=first["site"],
                uses=end - begin,
                proof="same pure typed integer expression over unescaped locals; first use dominates the group",
            )
        )
        decl = dict(
            op="decl", name=slot, type={"kind": "const", "of": t}, init=first["extent"]
        )
        _introduced_scalar(fn, slot, t, "common_iteration_extent", first["site"])
        fn["state_contract"].setdefault("value_definitions", {})[slot] = dict(
            value=expr,
            uses=[each["site"] for each in body[begin:end]],
            proof="snapshot dominates these uses; referenced frame-private scalars are not written in the group",
        )
        for each in body[begin:end]:
            each["extent"] = {"op": "var", "name": slot}
        body.insert(begin, decl)
        begin = end + 1


def _bounded_coordinates(fn, changes):
    """Use immutable iteration domains to remove redundant coordinate arithmetic."""
    integer_types = {p + str(n) for p in ("i", "u") for n in (8, 16, 32, 64)}
    types = {p["name"]: p["type"] for p in fn["params"]}
    types.update(
        {n["name"]: n["type"] for n in _nodes(fn["body"]) if n["op"] == "decl"}
    )
    private = set(fn.get("state_contract", {}).get("private_scalars", ()))
    epochs = {}

    def scalar_type(name):
        spec = types.get(name)
        while isinstance(spec, dict) and spec.get("kind") == "const":
            spec = spec["of"]
        return spec if isinstance(spec, str) and spec in integer_types else None

    def term(value, env):
        op = value.get("op")
        if (
            op == "literal"
            and isinstance(value["type"], str)
            and value["type"] in integer_types
        ):
            return ("literal", value["type"], value["value"])
        if (
            op == "var"
            and value.get("scope") != "global"
            and scalar_type(value["name"])
            and value["name"] in private
        ):
            name = value["name"]
            return env.get(name, ("value", name, epochs.get(name, 0)))
        if (
            op == "cast"
            and isinstance(value["type"], str)
            and value["type"] in integer_types
        ):
            operand = term(value["value"], env)
            return ("cast", value["type"], operand) if operand is not None else None
        if op == "binary":
            a, b = term(value["left"], env), term(value["right"], env)
            if a is not None and b is not None:
                return ("binary", value["operator"], a, b)
        if (
            op == "intrinsic"
            and value["name"] in {"max", "min"}
            and value["type"] in integer_types
        ):
            args = [term(a, env) for a in value["args"]]
            if all(a is not None for a in args):
                return ("intrinsic", value["name"], value["type"], *args)
        return None

    def writes(value):
        names = set()
        for node in _nodes(value):
            target = node.get("target") if node["op"] in {"assign", "update"} else None
            if node["op"] == "unary" and node["operator"] in {
                "preinc",
                "postinc",
                "predec",
                "postdec",
            }:
                target = node["value"]
            if isinstance(target, dict) and target.get("op") == "var":
                names.add(target["name"])
        return names

    def invalidate(env, names):
        for name in names:
            env.pop(name, None)
            epochs[name] = epochs.get(name, 0) + 1

    def expression(value, env, bounds):
        if isinstance(value, list):
            return [expression(v, env, bounds) for v in value]
        if not isinstance(value, dict):
            return value
        result = {k: expression(v, env, bounds) for k, v in value.items()}
        if result.get("op") == "binary" and result["operator"] in {"mod", "div"}:
            left = result["left"]
            if (
                left.get("op") == "var"
                and left.get("scope") != "global"
                and left["name"] in bounds
            ):
                domain = bounds[left["name"]]
                if term(result["right"], env) == domain:
                    changes.append(
                        dict(
                            rule="bounded_integer_coordinate",
                            site=result.get("site"),
                            operator=result["operator"],
                            proof="0 <= index < immutable iteration extent; positive divisor equals that extent in every executed iteration",
                        )
                    )
                    return (
                        left
                        if result["operator"] == "mod"
                        else dict(op="literal", type="i64", value=0)
                    )
        return result

    def sequence(body, env, bounds):
        for statement in body:
            op = statement["op"]
            if op in {"for", "while", "do_while", "each", "map"}:
                invalidate(env, writes(statement))
                inner = env.copy()
                domains = bounds.copy()
                if op == "map":
                    types[statement["index"]] = "i64"
                    extent = term(statement["extent"], env)
                    pure = all(
                        n["op"]
                        in {
                            "assign",
                            "var",
                            "index",
                            "literal",
                            "binary",
                            "unary",
                            "cast",
                            "intrinsic",
                        }
                        and not (n["op"] == "assign" and n["target"]["op"] != "index")
                        and not (
                            n["op"] == "unary" and n["operator"] not in {"pos", "neg"}
                        )
                        for n in _nodes(statement["body"])
                    )
                    if extent is not None and pure:
                        domains[statement["index"]] = extent
                sequence(statement["body"], inner, domains)
                invalidate(env, writes(statement))
            elif op == "block":
                sequence(statement["body"], env, bounds)
            elif op == "if":
                sequence(statement["then"], env.copy(), bounds)
                sequence(statement.get("else", []), env.copy(), bounds)
                env.clear()
                invalidate(env, writes(statement))
            else:
                replacement = expression(statement, env, bounds)
                statement.clear()
                statement.update(replacement)
                if op == "decl":
                    name = statement["name"]
                    invalidate(env, {name})
                    value = term(statement.get("init", {}), env)
                    if value is not None:
                        env[name] = value
                else:
                    invalidate(env, writes(statement))
                if any(
                    n["op"]
                    in {"call", "indirect_call", "object_dispatch", "library_call"}
                    for n in _nodes(statement)
                ):
                    env.clear()

    sequence(fn["body"], {}, {})


def optimize(fn):
    changes = []
    fn["body"] = _fold(fn["body"], changes)
    _bounded_coordinates(fn, changes)
    _share_extents(fn, changes)
    params = {p["name"]: p for p in fn["params"]}
    # Only immutable scalar coefficients can be shared throughout the region.
    params[None] = {
        v["name"]
        for v in fn["body"]
        if v["op"] == "decl"
        and isinstance(v.get("type"), dict)
        and v["type"]["kind"] == "const"
    }
    body = fn["body"]
    occupied = {n["name"] for n in _nodes(body) if "name" in n} | set(params)
    start = 0
    while start < len(body):
        first = body[start]
        dest = _map(first, params)
        if dest is None:
            start += 1
            continue
        end = start + 1
        destinations = [dest]
        while end < len(body) and (name := _map(body[end], params)) is not None:
            item = body[end]
            if item["index"] != first["index"] or semantics(
                item["extent"]
            ) != semantics(first["extent"]):
                break
            destinations.append(name)
            end += 1
        if end - start > 1:
            sources = body[start:end]
            stored, merged = {}, []
            reserved = occupied | {first["index"]}
            for position, (item, name) in enumerate(zip(sources, destinations)):
                assignment = item["body"][0]
                local = f"scar_element_value_{position}"
                while local in reserved:
                    local += "_"
                reserved.add(local)
                occupied.add(local)
                _introduced_scalar(
                    fn,
                    local,
                    params[name]["type"]["to"],
                    "legal_map_fusion",
                    first["site"],
                )
                merged.append(
                    dict(
                        op="decl",
                        name=local,
                        type=params[name]["type"]["to"],
                        init=_forward(assignment["value"], first["index"], stored),
                    )
                )
                merged.append({**assignment, "value": dict(op="var", name=local)})
                stored[name] = local
            # This is optional at the target. Strict physical mode can retain
            # the original traversal (including its floating-environment order).
            choice = dict(
                op="schedule_choice",
                site=first["site"],
                reference=sources,
                fused={**first, "body": merged, "site": first["site"] + "/fusion"},
                guard="masked_floating_exceptions",
                proof="same index domain, all accesses pointwise, writes to disjoint definable storage",
            )
            changes.append(
                dict(
                    rule="legal_map_fusion",
                    site=first["site"],
                    members=[x["site"] for x in sources],
                    proof=choice["proof"],
                )
            )
            body[start:end] = [choice]
        start += 1
    return changes

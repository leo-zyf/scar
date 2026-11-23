"""Local equivalence rules and legality proofs for pointwise iteration domains."""

from .link import semantics


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


def _share_extents(fn, changes):
    """Value-number repeated integer iteration domains at their first evaluation."""
    bindings = {s["name"]: s["type"] for s in fn["body"] if s["op"] == "decl"}
    reserved = {n["name"] for n in _nodes(fn["body"]) if "name" in n}
    reserved.update(p["name"] for p in fn["params"])
    # Only privately owned scalar locals whose address never leaves are eligible.
    for n in _nodes(fn["body"]):
        if (
            n["op"] == "unary"
            and n["operator"] == "addr"
            or n["op"] == "decl"
            and isinstance(n["type"], dict)
            and n["type"]["kind"] == "ref"
        ):
            for term in _nodes(n.get("value", n.get("init"))):
                if term["op"] == "var":
                    bindings.pop(term["name"], None)

    def scalar(e):
        if e["op"] == "var" and bindings.get(e["name"]) in ("i64", "i32"):
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
        names = {v["name"] for v in _nodes(expr) if v["op"] == "var"}
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
                or n["op"] == "var"
                and n["name"] in names
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
        for each in body[begin:end]:
            each["extent"] = {"op": "var", "name": slot}
        body.insert(begin, decl)
        begin = end + 1


def optimize(fn):
    changes = []
    fn["body"] = _fold(fn["body"], changes)
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
            reserved = set(params) | params[None] | {first["index"]}
            for position, (item, name) in enumerate(zip(sources, destinations)):
                assignment = item["body"][0]
                local = f"scar_element_value_{position}"
                while local in reserved:
                    local += "_"
                reserved.add(local)
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

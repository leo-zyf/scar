"""Local equivalence rules and legality proofs for pointwise iteration domains."""

from scar.contract import control_regions, semantic_form as semantics


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

    def invalidate_effects(value, env):
        """Kill may-written facts before entering a control continuation."""
        invalidate(env, writes(value))
        if any(
            node["op"]
            in {
                "call",
                "indirect_call",
                "object_dispatch",
                "library_call",
                "goto",
                "label",
            }
            or set(node.get("effects", ())) & {"STATE", "OPAQUE"}
            or node.get("state_writes")
            for node in _nodes(value)
        ):
            env.clear()

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
                # A body may run after init/condition effects and on any later
                # iteration after step/body effects. Never carry an earlier
                # scalar alias through any of those possible writes.
                invalidate_effects(statement, env)
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
                invalidate_effects(statement, env)
            elif op == "block":
                sequence(statement["body"], env, bounds)
            elif op == "if":
                # The condition precedes both arms. This includes may-effects
                # inside short-circuit or selected expressions.
                invalidate_effects(statement["condition"], env)
                sequence(statement["then"], env.copy(), bounds)
                sequence(statement.get("else", []), env.copy(), bounds)
                env.clear()
                invalidate(env, writes(statement))
            else:
                replacement = expression(statement, env, bounds)
                statement.clear()
                statement.update(replacement)
                invalidate_effects(statement, env)
                if op == "decl":
                    name = statement["name"]
                    invalidate(env, {name})
                    value = term(statement.get("init", {}), env)
                    if value is not None:
                        env[name] = value

    sequence(fn["body"], {}, {})


def _forward_materializations(fn, changes):
    """Elide a private pointwise copy stage, retaining the frozen mask stage.

    Only value-only compiler temporaries qualify, never source allocations.
    The producer contains no arithmetic or conversion: moving its loads across
    disjoint stores therefore preserves floating-environment evaluation order.
    """
    nodes = list(_nodes(fn["body"]))
    params = {p["name"]: p for p in fn["params"]}
    private = set(fn.get("state_contract", {}).get("private_scalars", ()))
    bindings = {}
    uses = {}
    changed = set(fn.get("state_contract", {}).get("escaped", ()))
    for node in nodes:
        op = node["op"]
        if op in {"decl", "buffer", "runtime_array", "map", "each"}:
            name = node["index"] if op == "map" else node["name"]
            bindings.setdefault(name, []).append(node)
        target = node.get("target") if op in {"assign", "update"} else None
        if op == "unary" and node["operator"] in {
            "addr",
            "preinc",
            "postinc",
            "predec",
            "postdec",
        }:
            target = node["value"]
        if target and target.get("op") == "var" and target.get("scope") != "global":
            changed.add(target["name"])
        if node["op"] == "var" and node.get("scope") != "global":
            uses[node["name"]] = uses.get(node["name"], 0) + 1
    if any(n["op"] in {"goto", "label"} for n in nodes):
        return

    def plain(spec):
        if isinstance(spec, dict) and spec.get("kind") == "const":
            return spec["of"]
        return spec

    def temporary(name):
        declarations = bindings.get(name, ())
        if len(declarations) != 1 or name in params:
            return None
        node = declarations[0]
        return (
            node
            if node["op"] == "buffer" and node.get("storage") == "compiler_temporary"
            else None
        )

    def domain(value):
        if value["op"] == "literal":
            return (
                value["type"]
                if value["type"] in ("i32", "i64") and type(value["value"]) is int
                else None
            )
        if value["op"] == "var" and value.get("scope") != "global":
            scalar = (
                fn.get("state_contract", {}).get("scalars", {}).get(value["name"], {})
            )
            if (
                value["name"] in private
                and scalar.get("initialized")
                and scalar.get("type") in ("i32", "i64")
            ):
                return scalar["type"]
        if value["op"] == "binary" and value["operator"] in {"mul", "add", "sub"}:
            left, right = domain(value["left"]), domain(value["right"])
            return left if left and left == right else None
        return None

    def guarded_assignment(mapping):
        if mapping["op"] != "map" or len(mapping["body"]) != 1:
            return None
        branch = mapping["body"][0]
        if (
            branch["op"] != "if"
            or branch.get("else")
            or len(branch["then"]) != 1
            or branch["then"][0]["op"] != "assign"
        ):
            return None
        return branch, branch["then"][0]

    def mask_cell(value):
        if value["op"] == "unary" and value["operator"] == "not":
            value = value["value"]
        return (
            value
            if value["op"] == "index" and value["base"].get("op") == "var"
            else None
        )

    # A frozen mask has one complete producer and only direct condition reads.
    # Taking its element address or passing it by reference cannot be mistaken
    # for reading a boolean value, even when the buffer itself has a unique name.
    mask_sources, mask_uses = {}, {}
    for node in nodes:
        if node["op"] == "map" and len(node["body"]) == 1:
            assignment = node["body"][0]
            if assignment["op"] == "assign" and _indexed(
                assignment["target"], node["index"]
            ):
                base = assignment["target"]["base"]
                owner = temporary(base["name"])
                if (
                    base.get("scope") != "global"
                    and owner is not None
                    and owner["type"] == "bool"
                    and semantics(owner["extent"]) == semantics(node["extent"])
                ):
                    mask_sources.setdefault(base["name"], []).append(node)
                    mask_uses[base["name"]] = mask_uses.get(base["name"], 0) + 1
        if node["op"] == "if" and (cell := mask_cell(node["condition"])) is not None:
            base = cell["base"]
            if base.get("scope") != "global":
                mask_uses[base["name"]] = mask_uses.get(base["name"], 0) + 1

    def mask(value, index, dominating):
        cell = mask_cell(value)
        if cell is None or not _indexed(cell, index):
            return False
        name = cell["base"]["name"]
        owner = temporary(name)
        producers = mask_sources.get(name, ())
        return (
            owner is not None
            and cell["base"].get("scope") != "global"
            and len(producers) == 1
            and id(owner) in dominating
            and id(producers[0]) in dominating
            and mask_uses.get(name) == uses.get(name)
        )

    def parameter(name):
        return (
            params.get(name, {}) if name not in bindings and name not in changed else {}
        )

    def copy_type(value, index, destination):
        op = value["op"]
        if op == "literal":
            return value["type"]
        if op == "var" and value.get("scope") != "global":
            name = value["name"]
            if name in private:
                return plain(fn["state_contract"]["scalars"][name]["type"])
        base = None
        if _indexed(value, index):
            base = value["base"]
        elif op == "unary" and value["operator"] == "deref":
            base = value["value"]
        if base and base["op"] == "var" and base.get("scope") != "global":
            if base["name"] == destination and not _indexed(value, index):
                return None
            spec = parameter(base["name"]).get("type", {})
            if isinstance(spec, dict) and spec.get("kind") == "ptr":
                return plain(spec["to"])
        return None

    def sequence(body, dominating=frozenset()):
        dominating = set(dominating)
        position = 0
        while position < len(body):
            candidate = body[position : position + 3]
            if len(candidate) == 3:
                buffer, producer, consumer = candidate
                temp = buffer.get("name")
                first, last = guarded_assignment(producer), guarded_assignment(consumer)
                if (
                    temporary(temp) is buffer
                    and buffer["type"] in {"bool", "i32", "i64", "f32", "f64"}
                    and first
                    and last
                    and uses.get(temp) == 2
                    and producer["index"] == consumer["index"]
                    and domain(buffer["extent"])
                    and semantics(buffer["extent"])
                    == semantics(producer["extent"])
                    == semantics(consumer["extent"])
                ):
                    index = producer["index"]
                    source_guard, source = first
                    dest_guard, dest = last
                    target = dest["target"]
                    output = (
                        parameter(target["base"]["name"])
                        if _indexed(target, index)
                        and target["base"].get("scope") != "global"
                        else {}
                    )
                    spec = output.get("type", {})
                    if (
                        _indexed(source["target"], index)
                        and source["target"]["base"].get("scope") != "global"
                        and source["target"]["base"]["name"] == temp
                        and semantics(dest["value"]) == semantics(source["target"])
                        and semantics(source_guard["condition"])
                        == semantics(dest_guard["condition"])
                        and mask(source_guard["condition"], index, dominating)
                        and output.get("elementwise_noalias") is True
                        and isinstance(spec, dict)
                        and spec.get("kind") == "ptr"
                        and spec["to"] == buffer["type"]
                        and copy_type(source["value"], index, output.get("name"))
                        == buffer["type"]
                    ):
                        # Keep the source value inside the mask. A zero extent
                        # or an entirely false mask must never evaluate a load.
                        forwarded = {**dest, "value": source["value"]}
                        body[position : position + 3] = [
                            {
                                **consumer,
                                "independent": True,
                                "body": [{**dest_guard, "then": [forwarded]}],
                            }
                        ]
                        changes.append(
                            dict(
                                rule="pointwise_materialization_forwarding",
                                site=consumer.get("site"),
                                temporary=temp,
                                element_type=buffer["type"],
                                removed_traversals=1,
                                proof="value-only unescaped compiler temporary; matching frozen mask and index domain; same-type loads across disjoint pointwise stores; no arithmetic reordered",
                            )
                        )
            node = body[position]
            for owner, field, _ in control_regions(node):
                sequence(owner[field], dominating)
            dominating.add(id(node))
            position += 1

    sequence(fn["body"])


def optimize(fn):
    changes = []
    fn["body"] = _fold(fn["body"], changes)
    _bounded_coordinates(fn, changes)
    _share_extents(fn, changes)
    _forward_materializations(fn, changes)
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

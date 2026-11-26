"""M3 rewrites canonical products using local equivalence and legality proofs."""

from copy import deepcopy
from scar.contract import (
    LOOP_OPERATIONS,
    control_regions,
    external_math_declaration,
    semantic_form as semantics,
)
from .contracts import is_complete, publish_program, require_program
from .loop_invariants import (
    hoist as hoist_loop_invariants,
    scalar_write,
    summarize_calls,
)
from .math_operands import analyze as analyze_math_operands


def rewrite(module):
    """Return an M3 product without mutating M2 bodies or ownership evidence."""
    require_program(module, "M2")
    result, changes = dict(module), []
    if is_complete(module):
        result["functions"] = []
        definitions = {fn["name"] for fn in module["functions"]}
        math = {
            d["name"]: abi
            for d in module["externs"]
            if d["name"] not in definitions
            and (abi := external_math_declaration(d)) is not None
        }
        scalar_calls = summarize_calls(module["functions"])
        for source in module["functions"]:
            function = dict(source)
            # optimize rebuilds the body before changing it. Its ownership
            # annotations also need a private copy; the remaining fields are read-only.
            if "state_contract" in source:
                function["state_contract"] = deepcopy(source["state_contract"])
            changes.extend(optimize(function, scalar_calls=scalar_calls))
            changes.extend(analyze_math_operands(function, math, _introduced_scalar))
            result["functions"].append(function)
    return publish_program(result, "M3", complete=is_complete(module)), changes


def _fold(expr, changes):
    if isinstance(expr, list):
        return [_fold(v, changes) for v in expr]
    if not isinstance(expr, dict):
        return expr
    domain = expr.get("extent") if expr.get("op") == "map" else None
    expr = {k: _fold(v, changes) for k, v in expr.items()}
    if domain and domain.get("operator") == "mul" and expr["extent"]["op"] == "literal":

        def factors(node):
            if node.get("op") == "binary" and node["operator"] == "mul":
                yield from factors(node["left"])
                yield from factors(node["right"])
            else:
                yield (
                    node.get("value")
                    if node.get("op") == "literal" and node["type"] == "i64"
                    else None
                )

        # Retain the source factorization before folding the extent to one
        # integer. The order of factors is the order of radix reconstruction.
        axes = list(factors(domain))
        if all(type(n) is int and n > 0 for n in axes):
            expr["extent_factors"] = axes
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
        return {name for node in _nodes(value) if (name := scalar_write(node))}

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
                extent, direct = bounds[left["name"]]
                if direct and result.get("type", "i64") != "i64":
                    return result
                divisor = (
                    semantics(result["right"]) if direct else term(result["right"], env)
                )
                if divisor == extent:
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

    def counted_domain(loop):
        # The condition is re-evaluated on every iteration. A private signed
        # unit induction from zero cannot overflow before reaching its bound.
        init = loop.get("init", ())
        if len(init) != 1 or init[0].get("op") != "decl":
            return None
        decl = init[0]
        index = decl["name"]
        condition, step = loop.get("condition") or {}, loop.get("step") or {}
        if (
            index not in private
            or decl["type"] != "i64"
            or term(decl.get("init") or {}, {}) != ("literal", "i64", 0)
            or condition.get("operator") != "lt"
            or condition.get("op") != "binary"
            or semantics(condition["left"]) != {"op": "var", "name": index}
            or step.get("op") != "unary"
            or step.get("operator") not in {"preinc", "postinc"}
            or semantics(step["value"]) != semantics(condition["left"])
        ):
            return None
        bound = condition["right"]
        # Use an atom so integer conversions of the operands cannot change
        # the comparison domain. Exclude mutation or escape of either operand.
        if (
            bound.get("op") == "var"
            and bound.get("scope") != "global"
            and bound["name"] in private
            and scalar_type(bound["name"]) == "i64"
        ):
            protected = {index, bound["name"]}
        elif bound.get("op") == "literal" and bound.get("type") == "i64":
            protected = {index}
        else:
            return None
        if writes(loop["body"]) & protected or any(
            n["op"]
            not in {"assign", "index", "var", "literal", "binary", "cast", "unary"}
            or n["op"] == "unary"
            and n["operator"] not in {"pos", "neg", "not", "bnot"}
            or n.get("state_writes")
            or set(n.get("effects", ())) & {"STATE", "OPAQUE"}
            for n in _nodes(loop["body"])
        ):
            return None
        return index, semantics(bound)

    def sequence(body, env, bounds):
        for statement in body:
            op = statement["op"]
            if op == "module_object_for":
                # The iterator or its target mapping may call arbitrary source code
                # between any two iterations. No incoming scalar bound proves a use.
                env.clear()
                sequence(statement["body"], {}, {})
                sequence(statement.get("else", []), {}, {})
            elif op in LOOP_OPERATIONS:
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
                        domains[statement["index"]] = (extent, False)
                elif op == "for" and (counted := counted_domain(statement)):
                    index, extent = counted
                    domains[index] = (extent, True)
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
    External masked stores only move loads. An unescaped frame-local target can
    also receive its same-type arithmetic at the original producer position;
    that target is inaccessible while floating operations evaluate in order.
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

    # Count uses through operand roles. An element passed to a reference
    # parameter can escape even when the IR contains no address-of operator.
    element_uses = {}
    for node in nodes:
        op = node["op"]
        if op == "assign":
            operands = (node["target"], node["value"])
        elif op == "binary" and node["operator"] in {
            "add",
            "sub",
            "mul",
            "div",
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
        }:
            operands = (node["left"], node["right"])
        elif op == "unary" and node["operator"] in {"pos", "neg", "not"}:
            operands = (node["value"],)
        else:
            continue
        for operand in operands:
            if operand.get("op") == "index":
                base = operand["base"]
                if base.get("op") == "var" and base.get("scope") != "global":
                    name = base["name"]
                    element_uses[name] = element_uses.get(name, 0) + 1

    def private_buffer(name, dominating):
        declarations = bindings.get(name, ())
        return (
            declarations[0]
            if len(declarations) == 1
            and declarations[0]["op"] == "buffer"
            and id(declarations[0]) in dominating
            and name not in params
            and element_uses.get(name, 0) == uses.get(name)
            else None
        )

    def affine(expr, index):
        """A unit-stride index proof; the original expression is not rewritten."""
        op = expr["op"]
        if (
            op == "literal"
            and expr.get("type") in {"i32", "i64"}
            and type(expr.get("value")) is int
        ):
            return 0, expr["value"]
        if op == "var" and expr.get("scope") != "global" and expr["name"] == index:
            return 1, 0
        if op == "cast" and expr.get("type") == "i64":
            return affine(expr["value"], index)
        if op == "unary" and expr.get("operator") in {"neg", "pos"}:
            value = affine(expr["value"], index)
            if value is not None:
                sign = -1 if expr["operator"] == "neg" else 1
                return sign * value[0], sign * value[1]
        if op == "binary" and expr.get("operator") in {"add", "sub", "mul"}:
            left, right = affine(expr["left"], index), affine(expr["right"], index)
            if left is not None and right is not None:
                if expr["operator"] == "add":
                    return left[0] + right[0], left[1] + right[1]
                if expr["operator"] == "sub":
                    return left[0] - right[0], left[1] - right[1]
                if left[0] == 0:
                    return left[1] * right[0], left[1] * right[1]
                if right[0] == 0:
                    return right[1] * left[0], right[1] * left[1]
        return None

    def address(expr, index):
        if affine(expr, index) is not None or domain(expr) in {"i32", "i64"}:
            return True
        if expr["op"] == "cast" and expr.get("type") == "i64":
            return address(expr["value"], index)
        return (
            expr["op"] == "binary"
            and expr.get("operator") in {"add", "sub", "mul", "div", "mod"}
            and all(address(expr[key], index) for key in ("left", "right"))
        )

    def proof_index(value, index, extent, guards):
        if (
            value["op"] == "binary"
            and value.get("operator") == "mod"
            and semantics(value["left"]) == {"op": "var", "name": index}
        ):
            radix = value["right"]
            if radix["op"] in {"var", "literal"} and domain(radix) in {"i32", "i64"}:
                if semantics(radix) != semantics(extent):
                    guards.setdefault(str(semantics(radix)), radix)
                return value["left"]
        return {
            key: (
                proof_index(child, index, extent, guards)
                if isinstance(child, dict) and "op" in child
                else child
            )
            for key, child in value.items()
        }

    def arithmetic_type(value, index, target, target_index, dominating, extent, guards):
        if value.get("external_effect") == "UNKNOWN" or value.get("writes") not in (
            None,
            [],
        ):
            return None
        op = value["op"]
        if op == "literal":
            return value.get("type")
        if op == "binary" and value.get("operator") in {"add", "sub", "mul", "div"}:
            left, right = (
                arithmetic_type(
                    value[key], index, target, target_index, dominating, extent, guards
                )
                for key in ("left", "right")
            )
            return left if left == right else None
        if op == "var" and value.get("scope") != "global" and value["name"] in private:
            return plain(fn["state_contract"]["scalars"][value["name"]]["type"])
        if op != "index" or len(value.get("indices", ())) != 1:
            return None
        base = value["base"]
        if (
            base.get("op") != "var"
            or base.get("scope") == "global"
            or not address(value["indices"][0], index)
        ):
            return None
        name = base["name"]
        if name == target:
            actual = affine(
                proof_index(value["indices"][0], index, extent, guards), index
            )
            expected = affine(proof_index(target_index, index, extent, guards), index)
            if actual is None or actual != expected:
                return None
        local = private_buffer(name, dominating)
        if local is not None:
            return local["type"]
        spec = parameter(name).get("type", {})
        return (
            plain(spec["to"])
            if isinstance(spec, dict) and spec.get("kind") == "ptr"
            else None
        )

    def local_forward(buffer, producer, consumer, dominating):
        if (
            producer.get("op") != "map"
            or consumer.get("op") != "map"
            or len(producer["body"]) != 1
            or len(consumer["body"]) != 1
            or producer["index"] != consumer["index"]
            or not domain(buffer["extent"])
            or semantics(buffer["extent"]) != semantics(producer["extent"])
            or semantics(buffer["extent"]) != semantics(consumer["extent"])
        ):
            return None
        source, dest = producer["body"][0], consumer["body"][0]
        index = producer["index"]
        if (
            source.get("op") != "assign"
            or dest.get("op") != "assign"
            or not _indexed(source["target"], index)
            or source["target"]["base"].get("scope") == "global"
            or source["target"]["base"]["name"] != buffer["name"]
            or semantics(dest["value"]) != semantics(source["target"])
        ):
            return None
        target = dest["target"]
        if (
            target.get("op") != "index"
            or len(target.get("indices", ())) != 1
            or target["base"].get("op") != "var"
            or target["base"].get("scope") == "global"
        ):
            return None
        name, selected = target["base"]["name"], target["indices"][0]
        owner = private_buffer(name, dominating)
        guards = {}
        layout = affine(proof_index(selected, index, producer["extent"], guards), index)
        if (
            owner is None
            or owner["type"] != buffer["type"]
            or layout is None
            or abs(layout[0]) != 1
            or arithmetic_type(
                source["value"],
                index,
                name,
                selected,
                dominating,
                producer["extent"],
                guards,
            )
            != buffer["type"]
        ):
            return None
        changes.append(
            dict(
                rule="pointwise_materialization_forwarding",
                site=consumer.get("site"),
                temporary=buffer["name"],
                element_type=buffer["type"],
                removed_traversals=1,
                storage_proof=dict(
                    destination=name,
                    unique_local_allocation=True,
                    direct_element_uses=uses[name],
                    index_coefficient=layout[0],
                    index_offset=layout[1],
                    self_reads="same_destination_index_only",
                    runtime_equalities=[
                        dict(left=producer["extent"], right=radix)
                        for radix in guards.values()
                    ],
                ),
                proof="unique value-only temporary; unescaped local destination with injective index; same-type arithmetic remains in original producer order; only inaccessible private stores move",
            )
        )
        forwarded = {
            **producer,
            "proof": "M3 verified private destination and same-index self reads",
            "body": [{**source, "target": target}],
        }
        if not guards:
            return forwarded
        condition = None
        for radix in guards.values():
            equal = dict(
                op="binary",
                operator="eq",
                type="bool",
                left=producer["extent"],
                right=radix,
            )
            condition = (
                equal
                if condition is None
                else dict(
                    op="binary",
                    operator="land",
                    type="bool",
                    left=condition,
                    right=equal,
                )
            )
        fallback = [buffer, producer, consumer]
        guard_fallbacks.add(id(fallback))
        return dict(
            op="if", condition=condition, then=[forwarded], **{"else": fallback}
        )

    guard_fallbacks = set()

    def sequence(body, dominating=frozenset()):
        if id(body) in guard_fallbacks:
            return
        dominating = set(dominating)
        position = 0
        while position < len(body):
            candidate = body[position : position + 3]
            if len(candidate) == 3:
                buffer, producer, consumer = candidate
                temp = buffer.get("name")
                if (
                    temporary(temp) is buffer
                    and buffer["type"] in {"f32", "f64"}
                    and uses.get(temp) == 2
                ):
                    forwarded = local_forward(buffer, producer, consumer, dominating)
                    if forwarded is not None:
                        body[position : position + 3] = [forwarded]
                        dominating.add(id(forwarded))
                        position += 1
                        continue
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


def optimize(fn, *, scalar_calls=None):
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
    hoist_loop_invariants(fn, changes, _introduced_scalar, scalar_calls)
    return changes

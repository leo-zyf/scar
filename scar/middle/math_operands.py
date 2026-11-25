"""M3 operand dependence and exact scalar sharing at resolved external math calls."""

from collections import Counter

from scar.contract import EXECUTION_ANNOTATIONS, control_regions, walk_operations

_INTEGER = {f"{s}{n}" for s in "iu" for n in (8, 16, 32, 64)}
_LOOPS = {"for", "while", "do_while", "each", "map"}
# Expressions evaluated by these statements. Nested statement lists are handled
# separately, so definitions inside an unentered control arm cannot dominate a use.
_OPERANDS = (
    "args",
    "base",
    "condition",
    "count",
    "extent",
    "index",
    "indices",
    "init",
    "left",
    "receiver",
    "right",
    "step",
    "target",
    "value",
    "values",
)


def _nodes(value):
    return walk_operations(value, exclude_fields=EXECUTION_ANNOTATIONS)


def _exact(node, private):
    """Return a typed, bit-exact value number for a closed integer expression."""
    if any(
        node.get(field)
        for field in (
            "volatile",
            "atomic",
            "writes",
            "state_writes",
            "may_raise",
            "ordering_effect",
            "external_effect",
        )
    ):
        return None
    op = node.get("op")
    if op == "var" and node.get("scope") in (None, "local"):
        name = node["name"]
        dtype = private.get(name)
        if dtype in _INTEGER and node.get("type", dtype) == dtype:
            return dtype, ("var", name)
    if op == "literal":
        dtype, value = node.get("type"), node.get("value")
        if isinstance(dtype, str) and dtype in _INTEGER and type(value) is int:
            width = int(dtype[1:])
            bound = 1 << (width - (dtype[0] == "i"))
            if (-bound if dtype[0] == "i" else 0) <= value < bound:
                return dtype, ("literal", value)
    if op == "binary" and node.get("operator") in {
        "add",
        "sub",
        "mul",
        "band",
        "bor",
        "bxor",
    }:
        left, right = _exact(node["left"], private), _exact(node["right"], private)
        if (
            left
            and right
            and left[0] == right[0]
            and left[0] in {"u32", "u64"}
            and node.get("type", left[0]) == left[0]
        ):
            return left[0], (node["operator"], left, right)
    if op == "cast" and node.get("mode") == "static":
        arg = _exact(node["value"], private)
        dtype = node.get("type")
        if (
            arg
            and isinstance(dtype, str)
            and dtype in {"f32", "f64"}
            and arg[0] in _INTEGER
            and node.get("source_type", arg[0]) == arg[0]
            and node.get("target_type", dtype) == dtype
        ):
            bits = int(arg[0][1:]) - (arg[0][0] == "i")
            if bits <= (24 if dtype == "f32" else 53):
                return dtype, ("exact_integer_conversion", arg)
    return None


def _expression_nodes(statement):
    for key in _OPERANDS:
        value = statement.get(key)
        if key == "init" and isinstance(value, list):
            continue
        if isinstance(value, (dict, list)):
            yield from _nodes(value)


def _inputs(arg, private):
    reads, closed = set(), True
    for node in _nodes(arg):
        if (
            node["op"] == "var"
            and node.get("scope") in (None, "local")
            and node["name"] in private
        ):
            reads.add(node["name"])
        elif node["op"] not in {"literal", "cast", "binary"}:
            closed = False
    return reads, closed


def _written(node):
    op = node["op"]
    if op == "decl" and node.get("init") is not None:
        return node["name"]
    if op in {"map", "each"}:
        return node.get("index") if op == "map" else node.get("name")
    target = None
    if op in {"update", "assign"}:
        target = node["target"]
    elif op == "unary" and node.get("operator") in {
        "postinc",
        "postdec",
        "preinc",
        "predec",
    }:
        target = node["value"]
    if (
        isinstance(target, dict)
        and target.get("op") == "var"
        and target.get("scope") in (None, "local")
    ):
        return target["name"]
    return None


def analyze(fn, declarations, introduce):
    """Preserve external evaluations and share only proven exact operands."""
    fn.pop("math_operand_analysis", None)
    if not declarations:
        return []

    def external(node):
        return node.get("op") == "call" and node.get("callee") in declarations

    if not any(external(n) for n in _nodes(fn["body"])):
        return []
    state = fn.get("state_contract", {})
    private = {
        name: dtype
        for name in state.get("private_scalars", ())
        if isinstance(dtype := state.get("scalars", {}).get(name, {}).get("type"), str)
        and name not in state.get("escaped", ())
    }
    reserved = {p["name"] for p in fn["params"]}
    explicit_jumps = False
    writes = Counter()
    for node in _nodes(fn["body"]):
        if isinstance(node.get("name"), str):
            reserved.add(node["name"])
        if isinstance(name := _written(node), str):
            writes[name] += 1
        # A jump could cross an introduced C++ initialization, even when it
        # targets a label after both calls.
        if node["op"] in {"label", "goto", "switch"}:
            explicit_jumps = True
    # Record single assignments in the straight-line entry prefix. Their
    # operands describe the stored value at definition, not a recomputation.
    definitions = {}
    for stmt in fn["body"]:
        if explicit_jumps or stmt["op"] not in {"decl", "assign", "expr"}:
            break
        name = _written(stmt)
        if name in private and writes[name] == 1:
            value = stmt.get("init") if stmt["op"] == "decl" else stmt.get("value")
            if isinstance(value, dict):
                reads, closed = _inputs(value, private)
                if closed:
                    definitions[name] = dict(
                        site=stmt.get("site"),
                        location=stmt.get("location"),
                        private_reads_at_definition=sorted(reads),
                    )
    facts, changes = [], []

    def record(call, loops):
        args = call.get("args", [])
        operands = []
        for arg in args:
            reads, scalar_expression = _inputs(arg, private)
            operands.append(
                dict(
                    private_reads=sorted(reads),
                    all_inputs_named_private_scalars_or_literals=scalar_expression,
                    evaluation=(
                        "bit_exact_integer_expression"
                        if _exact(arg, private)
                        else "unproven"
                    ),
                    loops=[
                        dict(
                            site=site,
                            syntactically_written_reads=sorted(writes & reads),
                            other_private_reads=sorted(reads - writes),
                        )
                        for site, writes in loops
                    ],
                )
            )
        facts.append(
            dict(site=call.get("site"), callee=call["callee"], operands=operands)
        )

    def direct(statement):
        op = statement["op"]
        if op == "assign":
            target = statement["target"]
            if (
                target.get("op") != "var"
                or target.get("scope") not in (None, "local")
                or target["name"] not in private
            ):
                return None
        if op not in {"assign", "decl", "expr", "return"}:
            return None
        if op == "decl" and (
            statement.get("storage") not in (None, "auto")
            or statement.get("cell")
            or statement["name"] not in private
        ):
            return None
        call = statement.get("init" if op == "decl" else "value", {})
        if (
            not isinstance(call, dict)
            or not external(call)
            or len(call.get("args", ())) != 1
        ):
            return None
        arg = call["args"][0]
        key = _exact(arg, private)
        abi = declarations[call["callee"]]
        if (
            not key
            or arg["op"] in {"literal", "var"}
            or abi["variadic"]
            or abi["parameter_types"] != [key[0]]
            or call.get("parameter_types", [key[0]]) != [key[0]]
        ):
            return None
        used = {n["name"] for n in _nodes(arg) if n["op"] == "var"}
        written = (
            statement["target"]["name"]
            if op == "assign"
            else statement.get("name") if op == "decl" else None
        )
        return call, key, written in used

    def visit(body, loops=()):
        group = []
        output = []

        def flush():
            if len(group) > 1:
                first, key, _ = group[0][1]
                name = "scar_math_operand_" + str(len(changes))
                while name in reserved:
                    name += "_"
                reserved.add(name)
                site = first.get("site")
                output.append(
                    dict(op="decl", name=name, type=key[0], init=first["args"][0])
                )
                introduce(fn, name, key[0], "share_exact_math_operand", site)
                for stmt, (call, _, _) in group:
                    call["args"] = [dict(op="var", name=name, type=key[0])]
                changes.append(
                    dict(
                        rule="share_exact_math_operand",
                        site=site,
                        uses=[info[0].get("site") for _, info in group],
                        preserved_calls=len(group),
                        type=key[0],
                        proof="exact integer operations and lossless conversion; frame-private inputs; adjacent unary calls in the same control arm",
                    )
                )
            output.extend(stmt for stmt, _ in group)
            group.clear()

        for stmt in body:
            for n in _expression_nodes(stmt):
                if external(n):
                    record(n, loops)
            nested = loops
            if stmt["op"] in _LOOPS:
                modified = {
                    name for n in _nodes(stmt) if isinstance(name := _written(n), str)
                }
                nested = (*loops, (stmt.get("site"), modified))
            for owner, field, _ in control_regions(stmt):
                visit(owner[field], nested if field == "body" else loops)
            item = None if explicit_jumps else direct(stmt)
            if not item or (group and item[1] != group[-1][1][1]):
                flush()
            if item:
                group.append((stmt, item))
                if item[2] or stmt["op"] == "return":
                    flush()
            else:
                output.append(stmt)
        flush()
        body[:] = output

    visit(fn["body"])
    if facts:
        relevant = {
            name
            for fact in facts
            for operand in fact["operands"]
            for name in operand["private_reads"]
        }
        for name, definition in reversed(definitions.items()):
            if name in relevant:
                relevant.update(definition["private_reads_at_definition"])
        fn["math_operand_analysis"] = dict(
            scope="resolved external calls in ordinary statement operands; loop fields list direct variable writes, not a transitive purity proof",
            entry_scalar_definitions={
                name: row for name, row in definitions.items() if name in relevant
            },
            result_policy="preserve_each_call_at_original_control_point",
            unproven_result_reuse_obligations=[
                "stable_implementation_and_private_state",
                "same_input_bits",
                "stable_rounding_mode",
                "preserve_errno_exceptions_flags_and_traps",
                "dominance_and_effect_order",
            ],
            sites=facts,
        )
    return changes

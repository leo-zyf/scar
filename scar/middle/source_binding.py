"""M1 validation of optional source instruction evidence against graph ports."""

from scar.contract import (
    digest,
    json_bytes,
    python_constant_rhs_sha256,
    python_literal_sequence_sha256,
    walk_operations,
)


from scar.source_contracts import python_binary_operator


def _source_position(location):
    return [
        location["line"],
        location["end_line"],
        location["column"] - 1,
        location["end_column"] - 1,
    ]


def _namespace_target(instruction, target, scope):
    name = instruction.get("name")
    location = target.get("location") if isinstance(target, dict) else None
    return (
        isinstance(name, str)
        and name.isidentifier()
        and isinstance(location, dict)
        and location.get("file") == scope["source_code"]["source_key"]
        and instruction.get("position") == _source_position(location)
        and target.get("op") == "var"
        and target.get("scope") == "global"
        and not target.get("cell")
        and target.get("name") == f"py:{scope['module']}:{name}"
    )


def _sequence_operands(node, source_key):
    """Index uniquely positioned display operands within one statement."""
    result = {}
    for item in walk_operations(node, exclude_fields={"source_expression"}):
        loc = item.get("location")
        if (
            item["op"] == "aggregate"
            and item.get("kind") in ("tuple", "list")
            and isinstance(item.get("items"), list)
            and isinstance(loc, dict)
            and loc.get("file") == source_key
        ):
            position = tuple(_source_position(loc))
            result[position] = None if position in result else item
    return result


def bind_constant_store(node, scope):
    """Match the constant load, its expression and its publication in the module."""
    proof = node["source_constant_store"]
    code = scope.get("source_code") or {}
    target, value = node.get("target", {}), node.get("value", {})
    if not (
        isinstance(proof, dict)
        and type(proof.get("semantics_version")) is int
        and proof["semantics_version"] == 1
        and code.get("kind") == "module"
        and proof.get("code") == code
        and node["op"] == "assign"
        and node["location"]["file"] == code["source_key"]
    ):
        raise ValueError("source constant store disagrees with its module")
    load, store = proof.get("load"), proof.get("store")
    for operand, instruction in ((value, load), (target, store)):
        if not (
            isinstance(instruction, dict)
            and operand.get("location", {}).get("file") == code["source_key"]
            and instruction.get("position") == _source_position(operand["location"])
            and all(
                type(instruction.get(k)) is int and instruction[k] >= 0
                for k in ("offset", "index")
            )
        ):
            raise ValueError(
                "source constant store disagrees with its positioned operand"
            )
    if not (
        load.get("opcode") == "LOAD_CONST"
        and store.get("opcode") in {"STORE_NAME", "STORE_GLOBAL"}
        and load["offset"] < store["offset"]
        and _namespace_target(store, target, scope)
    ):
        raise ValueError("source constant store disagrees with its destination")
    if "rhs_sha256" in proof:
        fingerprint = python_constant_rhs_sha256(value)
        content = load.get("value_sha256")
        if (
            value.get("op") not in {"binary", "unary"}
            or fingerprint is None
            or fingerprint != proof["rhs_sha256"]
            or load.get("type") not in {"int", "float", "str", "bool", "NoneType"}
            or not isinstance(content, str)
            or len(content) != 64
            or any(c not in "0123456789abcdef" for c in content)
        ):
            raise ValueError("source folded constant disagrees with its expression")
    elif not (
        value.get("op") == "literal"
        and value.get("type") == {"kind": "python_object", "class": "str"}
        and type(value.get("value")) is str
        and load.get("type") == "str"
        and load.get("value_sha256") == digest(json_bytes(value["value"]))
    ):
        raise ValueError("source constant load disagrees with its literal")


def _graph_position(operation, code):
    loc = operation.get("location") if isinstance(operation, dict) else None
    return (
        _source_position(loc) if loc and loc.get("file") == code["source_key"] else None
    )


def _inside_source(span, spot):
    return (
        span is not None
        and isinstance(spot, list)
        and len(spot) == 4
        and all(type(x) is int for x in spot)
        and (span[0], span[2])
        <= (spot[0], spot[2])
        <= (spot[1], spot[3])
        <= (span[1], span[3])
    )


def _edge_enters(edge, statement, kind, code):
    span = _graph_position(statement, code)
    return (
        edge.get("kind") == kind
        and edge.get("statement_position") == span
        and _inside_source(span, edge.get("position"))
    )


def _source_return(other, operand_pos):
    return (
        other.get("kind") == "implicit_return"
        and other.get("statement_position") is None
        and other.get("position") == operand_pos
        and type(other.get("constant_index")) is int
        and other["constant_index"] >= 0
        and type(other.get("return_offset")) is int
        and (
            other["opcode"] == "RETURN_CONST"
            and other["return_offset"] == other["offset"]
            or other["opcode"] == "LOAD_CONST"
            and other["return_offset"] > other["offset"]
        )
    )


def _folded_value_type(value):
    if isinstance(value, dict):
        if set(value) == {"float_hex"} and value["float_hex"] in ("inf", "-inf", "nan"):
            return "float"
        if set(value) == {"integer_hex"} and isinstance(value["integer_hex"], str):
            token = value["integer_hex"]
            try:
                return "int" if hex(int(token, 16)) == token else None
            except ValueError:
                pass
        return None
    return type(value).__name__


def bind_module_exit(scope):
    """Validate the captured terminal region independently of paths reaching it."""
    exit_ = scope["frame_exit"]
    code = scope.get("source_code", {})
    rows = exit_.get("instructions") if isinstance(exit_, dict) else None
    if not (
        scope.get("kind") == code.get("kind") == "module"
        and code.get("code_sha256")
        and isinstance(rows, list)
        and code["code_sha256"] == exit_.get("code_sha256")
        and all(
            isinstance(row, dict)
            and type(row.get("offset")) is int
            and row["offset"] >= 0
            and row["offset"] % 2 == 0
            for row in rows
        )
        and [r.get("opcode") for r in rows]
        in (["RETURN_CONST"], ["LOAD_CONST", "RETURN_VALUE"])
        and (len(rows) == 1 or rows[0]["offset"] + 2 == rows[1]["offset"])
        and rows[0].get("type") == "NoneType"
        and rows[0].get("value_sha256") == digest(json_bytes(None))
        and type(rows[0].get("index")) is int
        and type(rows[0].get("oparg")) is int
        and 0 <= rows[0]["index"] == rows[0]["oparg"] <= 255
    ):
        raise ValueError("source module exit disagrees with its code or None return")


def _is_loop_backedge(edge, loop, code):
    if loop is None or edge.get("kind") != "loop_backedge":
        return False
    ports = loop.get("source_expression", {}).get("iteration", {})
    backs = ports.get("backedges", [ports.get("backedge")])
    prefixes = edge.get("prefixes", [])
    if not isinstance(prefixes, list) or not all(isinstance(p, dict) for p in prefixes):
        return False
    for back in backs:
        if not isinstance(back, dict) or edge.get("backedge_offset") != back.get(
            "offset"
        ):
            continue
        first = (prefixes or [back])[0]
        return (
            edge.get("statement_position") == _graph_position(loop, code)
            and edge.get("target") == back.get("target")
            and all(
                edge.get(k) == first.get(k) for k in ("offset", "opcode", "position")
            )
            and all(
                p.get("opcode") == "EXTENDED_ARG"
                and p.get("offset") == back["offset"] - 2 * (len(prefixes) - i)
                and p.get("oparg") == back["oparg"] >> (8 * (len(prefixes) - i))
                and p.get("position") == back.get("position")
                for i, p in enumerate(prefixes)
            )
        )
    return False


def _bind_suite_exits(branch, node, successor, loop, code):
    """Bind both arms to one following statement or the enclosing iterator header."""
    join = branch.get("join", {})
    exits = branch["exits"]
    if not (
        isinstance(exits, dict)
        and set(exits) == {"then", "else"}
        and node.get("else")
        and "then_exit" not in branch
        and "else_exit" not in branch
    ):
        raise ValueError("source branch has incompatible exit tables")
    iterator = (loop or {}).get("source_expression", {}).get("iteration", {})
    backward = join.get("kind") == "loop_header"
    if backward:
        header = (iterator.get("next_prefix") or [iterator.get("next")])[0]
        if (
            successor is not None
            or not header
            or join.get("statement_position") != _graph_position(loop, code)
            or any(join.get(k) != header[k] for k in ("opcode", "offset", "position"))
        ):
            raise ValueError("source branch does not join its enclosing iterator")
    elif not successor or not _edge_enters(join, successor, "continuation", code):
        raise ValueError("source branch does not join its graph successor")
    seen = set()
    for side in ("then", "else"):
        sites = [_graph_position(statement, code) for statement in node[side]]
        rows = exits[side]
        if not isinstance(rows, list) or not rows:
            raise ValueError("source branch arm has no normal exit")
        for row in rows:
            if not (
                isinstance(row, dict)
                and type(row.get("offset")) is int
                and row["offset"] >= branch[side]["offset"]
                and row["offset"] not in seen
                and any(_inside_source(span, row.get("position")) for span in sites)
            ):
                raise ValueError("source branch exit lies outside its arm")
            seen.add(row["offset"])
            if backward:
                backs = iterator.get("backedges", [iterator.get("backedge")])
                valid = row.get("target") == join["offset"] and any(
                    back
                    and all(
                        row.get(k) == back.get(k)
                        for k in ("opcode", "offset", "position", "oparg", "target")
                    )
                    for back in backs
                )
            elif "target" in row:
                valid = (
                    row.get("opcode")
                    in {"JUMP_FORWARD", "POP_JUMP_IF_TRUE", "POP_JUMP_IF_FALSE"}
                    and row["target"] == join["offset"] > row["offset"]
                )
            else:
                valid = row.get("fallthrough") == join["offset"] > row[
                    "offset"
                ] and row.get("opcode") not in {
                    "JUMP_FORWARD",
                    "JUMP_BACKWARD",
                    "RETURN_VALUE",
                    "RETURN_CONST",
                    "RAISE_VARARGS",
                    "RERAISE",
                }
            if not valid:
                raise ValueError("source branch exit disagrees with its join")
        if backward:
            expected = {
                back["offset"]
                for back in iterator.get("backedges", [iterator.get("backedge")])
                if back
                and any(_inside_source(span, back.get("position")) for span in sites)
            }
            if {row["offset"] for row in rows} != expected:
                raise ValueError("source branch arm omits an enclosing loop backedge")


def _bind_condition_edges(node, proof, code, successor, loop):
    branch = proof.get("branch")
    condition = node.get("condition", {})
    negated = condition.get("op") == "unary" and condition.get("operator") == "not"
    operand = condition.get("value", {}) if negated else condition

    rows = proof["instructions"]
    jump = rows[-1]
    operand_pos = _graph_position(operand, code)
    taken = branch.get("taken", "else") if isinstance(branch, dict) else None
    fallthrough = "then" if taken == "else" else "else"
    if not (
        isinstance(branch, dict)
        and taken in ("then", "else")
        and branch.get("fallthrough", "then") == fallthrough
        and operand_pos
        and branch.get("condition_position") == _graph_position(condition, code)
        and branch.get("operand_position") == operand_pos
        and branch.get("negated") is negated
        and jump["opcode"]
        == ("POP_JUMP_IF_TRUE" if (taken == "then") != negated else "POP_JUMP_IF_FALSE")
        and jump["position"] == operand_pos
        and type(jump.get("target")) is int
    ):
        raise ValueError("source condition disagrees with the graph condition or jump")
    then, other = branch.get("then"), branch.get("else")
    if not all(
        isinstance(edge, dict)
        and type(edge.get("offset")) is int
        and isinstance(edge.get("opcode"), str)
        and edge["opcode"]
        for edge in (then, other)
    ):
        raise ValueError("source condition has invalid successor instructions")
    marker = branch.get("fallthrough_marker")
    if not (
        jump["offset"]
        < branch[fallthrough]["offset"]
        < branch[taken]["offset"]
        == jump["target"]
        and (
            marker is None
            or (
                isinstance(marker, dict)
                and marker.get("opcode") == "NOT_TAKEN"
                and marker.get("position") == operand_pos
                and type(marker.get("offset")) is int
                and jump["offset"] < marker["offset"] < branch[fallthrough]["offset"]
            )
        )
        and node.get("then")
        and _edge_enters(then, node["then"][0], "suite", code)
    ):
        raise ValueError(
            "source condition disagrees with its fallthrough or jump target"
        )
    if node.get("else") or successor is not None:
        suite = node.get("else")
        if not _edge_enters(
            other,
            suite[0] if suite else successor,
            "suite" if suite else "continuation",
            code,
        ):
            raise ValueError(
                "source condition disagrees with its alternative or continuation"
            )
    elif not (
        _is_loop_backedge(other, loop, code)
        if loop is not None
        else _source_return(other, operand_pos)
    ):
        raise ValueError("source condition lacks its enclosing continuation")
    if "exits" in branch:
        _bind_suite_exits(branch, node, successor, loop, code)
    elif any(key in branch for key in ("join", "then_exit", "else_exit")):
        destination = branch.get("join", {})
        yes, no = branch.get("then_exit", {}), branch.get("else_exit", {})
        if not (
            node.get("else")
            and successor
            and _edge_enters(destination, successor, "continuation", code)
            and _inside_source(
                _graph_position(node["then"][-1], code), yes.get("position")
            )
            and _inside_source(
                _graph_position(node["else"][-1], code), no.get("position")
            )
            and yes.get("opcode") == "JUMP_FORWARD"
            and yes.get("target") == no.get("fallthrough") == destination.get("offset")
            and then["offset"]
            <= yes["offset"]
            < other["offset"]
            <= no["offset"]
            < destination["offset"]
        ):
            raise ValueError(
                "source branch exits disagree with the normal continuation"
            )


def _bind_iteration_edges(node, proof, scope, successor, enclosing_loop):
    code = scope["source_code"]
    loop = proof.get("iteration")
    body = node.get("body")
    if not (
        isinstance(loop, dict)
        and all(
            isinstance(loop.get(k), dict)
            for k in ("next", "store", "body", "exhausted")
        )
        and isinstance(loop.get("cleanup"), list)
        and node.get("protocol") == "iterator"
        and node.get("init") == []
        and isinstance(body, list)
        and body
        and isinstance(node.get("else"), list)
        and not any(
            op["op"] in {"break", "continue"}
            for op in walk_operations([body, node["else"]])
        )
    ):
        raise ValueError("source iteration lacks its ordinary target or control ports")
    step, store, entry, exit_edge = (
        loop[k] for k in ("next", "store", "body", "exhausted")
    )
    acquire = proof["instructions"][-1]
    iterable_pos = _graph_position(node.get("range"), code)
    pos = acquire["position"]
    cleanup = loop["cleanup"]
    backedges = [loop["backedge"]] if "backedge" in loop else loop.get("backedges")
    prefixes = loop.get("next_prefix", [])
    if not (
        isinstance(backedges, list)
        and backedges
        and not ("backedge" in loop and "backedges" in loop)
        and isinstance(prefixes, list)
        and all(isinstance(r, dict) for r in [*prefixes, *backedges])
    ):
        raise ValueError("source iteration lacks unique backedge records")
    if not (
        acquire["opcode"] == "GET_ITER"
        and pos in (proof["position"], iterable_pos)
        and len(proof["instructions"]) >= 2
        and all(
            _inside_source(iterable_pos, row["position"])
            for row in proof["instructions"][:-1]
        )
        and [r.get("opcode") for r in cleanup if isinstance(r, dict)]
        in (["END_FOR"], ["END_FOR", "POP_ITER"])
        and all(isinstance(r, dict) and r.get("position") == pos for r in cleanup)
        and all(
            type(r.get("offset")) is int
            and r["offset"] >= 0
            and isinstance(r.get("opcode"), str)
            for r in (*prefixes, step, store, entry, *backedges, *cleanup, exit_edge)
        )
        and all(
            type(r.get("target")) is int
            and type(r.get("oparg")) is int
            and r["oparg"] >= 0
            for r in (step, *backedges)
        )
    ):
        raise ValueError(
            "source iteration disagrees with its iterable or cleanup instructions"
        )
    slots = scope.get("definition_code", {}).get("varnames", ())
    target = node["target"]
    local = (
        code["kind"] == "function"
        and store["opcode"] == "STORE_FAST"
        and type(store.get("index")) is int
        and 0 <= store["index"] < len(slots)
        and slots[store["index"]] == store.get("name")
        and _fast_operand(store, target, code)
    )
    mapping = store["opcode"] in {"STORE_NAME", "STORE_GLOBAL"} and _namespace_target(
        store, target, scope
    )
    head = (prefixes or [step])[0]["offset"]
    back_width = cleanup[0]["offset"] - backedges[-1]["offset"]
    if not (
        step["opcode"] == "FOR_ITER"
        and step.get("position") == pos
        and (local or mapping)
        and type(store.get("oparg")) is int
        and type(store.get("index")) is int
        and store["oparg"] == store["index"] >= 0
        and all(
            p["opcode"] == "EXTENDED_ARG"
            and p["offset"] == head + 2 * i
            and p.get("oparg") == step["oparg"] >> (8 * (len(prefixes) - i))
            and p.get("position") == pos
            for i, p in enumerate(prefixes)
        )
        and step["offset"] == head + 2 * len(prefixes)
        and back_width in (2, 4)
        and [b["offset"] for b in backedges] == sorted({b["offset"] for b in backedges})
        and all(
            b["opcode"] == "JUMP_BACKWARD"
            and b["target"] == head
            and b["offset"] + back_width - 2 * b["oparg"] == head
            and _inside_source(proof["position"], b.get("position"))
            for b in backedges
        )
        and step["target"] == cleanup[0]["offset"]
        and step["target"] == store["offset"] + 2 * step["oparg"]
        and acquire["offset"] + 2 == head
        and step["offset"] < store["offset"] < entry["offset"] <= backedges[0]["offset"]
        and all(
            b["offset"] == a["offset"] + 2
            for a, b in zip(cleanup, [*cleanup[1:], exit_edge])
        )
        and _edge_enters(entry, body[0], "suite", code)
        and _inside_source(
            _graph_position(body[-1], code), backedges[-1].get("position")
        )
    ):
        raise ValueError(
            "source iteration disagrees with its target, loop body or backedge"
        )
    alternative = node.get("else") or []
    if alternative or successor is not None:
        if not _edge_enters(
            exit_edge,
            alternative[0] if alternative else successor,
            "suite" if alternative else "continuation",
            code,
        ):
            raise ValueError(
                "source iteration disagrees with its exhaustion continuation"
            )
    elif not (
        _is_loop_backedge(exit_edge, enclosing_loop, code)
        if enclosing_loop is not None
        else _source_return(exit_edge, pos)
    ):
        raise ValueError("source iteration lacks its enclosing continuation")


def _fast_operand(row, node, code):
    return (
        node.get("op") == "var"
        and not node.get("scope")
        and not node.get("cell")
        and node.get("name") == row["name"]
        and _graph_position(node, code) == row["position"]
    )


def bind_source_expression(node, scope, successor=None, loop=None):
    """Bind positioned instruction evidence; M2 verifies semantic evaluation order."""
    proof, code = node["source_expression"], scope.get("source_code") or {}
    loc = node["location"]
    position = _source_position(loc)
    if not (
        isinstance(proof, dict)
        and type(proof.get("semantics_version")) is int
        and proof["semantics_version"] == 1
        and code.get("kind") == scope.get("kind")
        and code["kind"] in {"function", "module"}
        and proof.get("code_sha256") == code.get("code_sha256")
        and loc.get("file") == code.get("source_key")
        and proof.get("position") == position
        and isinstance(proof.get("instructions"), list)
        and proof["instructions"]
    ):
        raise ValueError("source expression region disagrees with its module")
    operation = node["op"]
    if operation == "raise":
        present, cause = node.get("has_exception"), node.get("has_cause")
        if not (
            type(present) is bool
            and type(cause) is bool
            and (present or not cause)
            and all(isinstance(node.get(k), dict) for k in ("value", "cause"))
        ):
            raise ValueError("source raise disagrees with its operand ports")
        for supplied, key in ((present, "value"), (cause, "cause")):
            operand = node[key]
            if not supplied and operand != {
                "op": "literal",
                "type": {"kind": "python_object", "class": "NoneType"},
                "value": None,
            }:
                raise ValueError("source raise omits an evaluated operand")
    elif operation == "literal":
        if not (
            code["kind"] == "function"
            and len(proof["instructions"]) == 1
            and proof["instructions"][0].get("opcode") == "NOP"
            and proof["instructions"][0].get("position") == position
        ):
            raise ValueError("discarded literal lacks its source line instruction")
    elif operation not in {"assign", "unbind", "if", "for", "return"} and not (
        isinstance(node.get("target"), dict)
        and isinstance(node.get("args"), list)
        and isinstance(node.get("keywords"), list)
    ):
        raise ValueError("source call region disagrees with its call port")
    first = position[0], position[2]
    last = position[1], position[3]
    previous = -1
    sequences = None
    binary_ports = fast_reads = None
    folded_ports = None
    for instruction in proof["instructions"]:
        if not isinstance(instruction, dict):
            raise ValueError("invalid source instruction in call region")
        offset, opcode = instruction.get("offset"), instruction.get("opcode")
        span = instruction.get("position")
        if not (
            type(offset) is int
            and offset > previous
            and isinstance(opcode, str)
            and opcode
            and isinstance(span, list)
            and len(span) == 4
            and all(type(v) is int and v >= 0 for v in span)
            and first <= (span[0], span[2]) <= (span[1], span[3]) <= last
            and all(
                type(instruction[k]) is int and instruction[k] >= 0
                for k in ("oparg", "index")
                if k in instruction
            )
        ):
            raise ValueError("invalid source instruction position or operand")
        previous = offset
        if "folded_value" in instruction or "rhs_sha256" in instruction:
            if folded_ports is None:
                folded_ports = {}
                for operand in walk_operations(
                    node, exclude_fields={"source_expression"}
                ):
                    if operand["op"] not in {"binary", "unary"}:
                        continue
                    pos = _graph_position(operand, code)
                    if pos is not None:
                        key = tuple(pos)
                        folded_ports[key] = None if key in folded_ports else operand
            expression = folded_ports.get(tuple(span))
            if not (
                expression
                and opcode in {"LOAD_CONST", "LOAD_SMALL_INT", "RETURN_CONST"}
                and "folded_value" in instruction
                and instruction.get("type")
                == _folded_value_type(instruction["folded_value"])
                and instruction.get("rhs_sha256")
                and instruction["rhs_sha256"] == python_constant_rhs_sha256(expression)
                and instruction.get("value_sha256")
                == digest(json_bytes(instruction["folded_value"]))
            ):
                raise ValueError(
                    "compiler-folded constant disagrees with its graph operands"
                )
        aggregate_opcode = opcode in {
            "BUILD_LIST",
            "BUILD_TUPLE",
            "LIST_EXTEND",
            "LIST_APPEND",
            "CALL_INTRINSIC_1",
        }
        pool_aggregate = instruction.get("type") == "tuple" and (
            "item_count" in instruction or "items_sha256" in instruction
        )
        if opcode in {
            "BINARY_SUBSCR",
            "STORE_SUBSCR",
            "BINARY_SLICE",
            "STORE_SLICE",
        } or (opcode == "BINARY_OP" and instruction.get("operator") == "subscript"):
            if opcode == "BINARY_OP" and instruction.get("oparg") != 26:
                raise ValueError("source subscription has invalid binary opcode")
            if not any(
                item["op"] == "index"
                and item.get("protocol") == "python_subscription"
                and _graph_position(item, code) == span
                for item in walk_operations(node, exclude_fields={"source_expression"})
            ):
                raise ValueError("source subscription lacks its matching graph port")
            continue
        if opcode == "BINARY_OP":
            if binary_ports is None:
                binary_ports = {}
                for item in walk_operations(node, exclude_fields={"source_expression"}):
                    loc = item.get("location")
                    if (
                        item["op"] == "binary"
                        and loc
                        and loc.get("file") == code["source_key"]
                    ):
                        key = tuple(_source_position(loc))
                        binary_ports[key] = None if key in binary_ports else item
            expr = binary_ports.get(tuple(span))
            operator = python_binary_operator(instruction.get("oparg"))
            if not (
                expr
                and operator
                and operator[0] == expr.get("operator") == instruction.get("operator")
                and all(isinstance(expr.get(k), dict) for k in ("left", "right"))
            ):
                raise ValueError(
                    "source binary instruction disagrees with its operands or operator"
                )
        if aggregate_opcode or pool_aggregate:
            if sequences is None:
                sequences = _sequence_operands(node, code["source_key"])
            display = sequences.get(tuple(span))
            if display is None:
                raise ValueError(
                    "source sequence instruction lacks a unique display port"
                )
            count = len(display["items"])
            arg = instruction.get("oparg")
            if pool_aggregate:
                fingerprint = python_literal_sequence_sha256(display["items"])
                if not (
                    opcode == "LOAD_CONST"
                    and "names" not in instruction
                    and type(arg) is int
                    and instruction.get("index") == arg
                    and type(instruction.get("item_count")) is int
                    and instruction["item_count"] == count
                    and fingerprint is not None
                    and instruction.get("items_sha256") == fingerprint
                ):
                    raise ValueError(
                        "source constant tuple disagrees with its ordered display items"
                    )
            elif not (
                type(arg) is int
                and (
                    opcode == "BUILD_LIST"
                    and arg <= count
                    or opcode == "BUILD_TUPLE"
                    and display["kind"] == "tuple"
                    and arg == count
                    or opcode in {"LIST_EXTEND", "LIST_APPEND"}
                    and arg == 1
                    or opcode == "CALL_INTRINSIC_1"
                    and display["kind"] == "tuple"
                    and arg == 6
                    and instruction.get("intrinsic") == "INTRINSIC_LIST_TO_TUPLE"
                )
            ):
                raise ValueError(
                    "source sequence instruction disagrees with its display operands"
                )
            continue
        if opcode in {"LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_BORROW", "STORE_FAST"}:
            index = instruction.get("index")
            slots = scope.get("definition_code", {}).get("varnames", ())
            if (
                code["kind"] != "function"
                or type(index) is not int
                or instruction.get("oparg") != index
                or not 0 <= index < len(slots)
                or slots[index] != instruction.get("name")
            ):
                raise ValueError("source fast local disagrees with its function slot")
            if opcode != "STORE_FAST":
                if fast_reads is None:
                    fast_reads = {
                        tuple(_graph_position(v, code) or ()): v
                        for v in walk_operations(node)
                        if v["op"] == "var" and v.get("source_name_load")
                    }
                operand = fast_reads.get(tuple(span), {})
                if not _fast_operand(instruction, operand, code) or operand.get(
                    "source_name_load"
                ) != dict(
                    instruction, semantics_version=1, code_sha256=code["code_sha256"]
                ):
                    raise ValueError(
                        "source fast local disagrees with its lexical operand"
                    )
        if opcode in {
            "LOAD_NAME",
            "LOAD_ATTR",
            "LOAD_GLOBAL",
            "STORE_NAME",
            "STORE_GLOBAL",
            "DELETE_NAME",
            "DELETE_GLOBAL",
        }:
            arg = instruction.get("oparg")
            if not (
                type(arg) is int
                and isinstance(instruction.get("name"), str)
                and instruction.get("index")
                == (arg >> 1 if opcode in {"LOAD_GLOBAL", "LOAD_ATTR"} else arg)
            ):
                raise ValueError("source call name operand disagrees with its index")
        elif opcode in {"LOAD_CONST", "KW_NAMES", "LOAD_SMALL_INT", "RETURN_CONST"}:
            immediate = opcode == "LOAD_SMALL_INT"
            arg = instruction.get("oparg")
            index_matches = (
                "index" not in instruction
                if immediate
                else instruction.get("index") == arg
            )
            if type(arg) is not int or not index_matches:
                raise ValueError("source call literal operand disagrees with its index")
            if "constant_key" in instruction:
                if not (
                    opcode == "LOAD_CONST"
                    and instruction.get("type") in {"tuple", "slice"}
                    and isinstance(instruction["constant_key"], dict)
                    and instruction["constant_key"].get("kind") == instruction["type"]
                ):
                    raise ValueError("invalid source subscription constant")
            elif instruction.get("type") == "tuple":
                names = instruction.get("names")
                if (
                    immediate
                    or not isinstance(names, list)
                    or not all(isinstance(n, str) for n in names)
                ):
                    raise ValueError("invalid source keyword tuple")
            else:
                fingerprint = instruction.get("value_sha256")
                if (
                    opcode == "KW_NAMES"
                    or instruction.get("type")
                    not in {"int", "float", "str", "bool", "NoneType"}
                    or not isinstance(fingerprint, str)
                    or len(fingerprint) != 64
                    or any(c not in "0123456789abcdef" for c in fingerprint)
                ):
                    raise ValueError("missing literal identity in source call")
    last = proof["instructions"][-1]
    if operation == "raise":
        if not (
            last["opcode"] == "RAISE_VARARGS"
            and last.get("oparg") == int(present) + int(cause)
            and last.get("position") == position
        ):
            raise ValueError(
                "source raise terminator disagrees with its operand count or position"
            )
    elif operation == "return":
        value = node.get("value", {})
        if not (
            code["kind"] == "function"
            and last["position"] in (position, _graph_position(value, code))
            and (
                last["opcode"] == "RETURN_VALUE"
                and len(proof["instructions"]) > 1
                or last["opcode"] == "RETURN_CONST"
                and len(proof["instructions"]) == 1
                and (
                    "folded_value" in last
                    and last["rhs_sha256"] == python_constant_rhs_sha256(value)
                    or value.get("op") == "literal"
                    and value.get("type", {}).get("class") == last.get("type")
                    and digest(json_bytes(value.get("value")))
                    == last.get("value_sha256")
                )
            )
        ):
            raise ValueError("source return disagrees with its function operand")
    elif operation == "assign":
        target = node.get("target", {})
        if not (
            isinstance(node.get("value"), dict)
            and len(proof["instructions"]) > 1
            and (
                last["opcode"] in {"STORE_NAME", "STORE_GLOBAL"}
                and _namespace_target(last, target, scope)
                or last["opcode"] == "STORE_FAST"
                and _fast_operand(last, target, code)
                or last["opcode"] in {"STORE_SUBSCR", "STORE_SLICE"}
                and target.get("op") == "index"
                and target.get("protocol") == "python_subscription"
                and _graph_position(target, code) == last["position"]
            )
        ):
            raise ValueError("source expression store disagrees with its target")
    elif operation == "unbind":
        targets = node.get("targets")
        if not (
            isinstance(targets, list)
            and len(targets) == len(proof["instructions"])
            and all(
                ins["opcode"] in {"DELETE_NAME", "DELETE_GLOBAL"}
                and _namespace_target(ins, target, scope)
                for ins, target in zip(proof["instructions"], targets)
            )
        ):
            raise ValueError("source deletion disagrees with its ordered targets")
    elif operation == "for":
        _bind_iteration_edges(node, proof, scope, successor, loop)
    elif operation == "if":
        _bind_condition_edges(node, proof, code, successor, loop)
    elif operation != "literal" and last["opcode"] != "POP_TOP":
        raise ValueError("source call region has no final result discard")


def bind_source_definition(node, scope, declarations):
    """Tie the code constant and the published object to the declared function."""
    proof = node["source_definition"]
    parent = scope.get("source_code", {})
    child = proof.get("child") if isinstance(proof, dict) else None
    declared = declarations.get(node.get("symbol"), {})
    nested = declared.get("python_scope", {})
    location = node["location"]
    if not (
        isinstance(child, dict)
        and node["op"] == "python_define"
        and parent.get("kind") == "module"
        and parent.get("source_key") == location["file"]
        and type(proof.get("semantics_version")) is int
        and proof["semantics_version"] == 1
        and proof.get("code_sha256") == parent.get("code_sha256")
        and proof.get("position") == _source_position(location)
        and nested.get("kind") == "function"
        and nested.get("module") == scope["module"]
        and nested.get("source_key") == location["file"]
        and declared.get("location") == location
        and declared.get("call_protocol", {}).get("kind") == "regular"
        and child == nested.get("definition_code")
        and type(child.get("code_sha256")) is str
        and len(child["code_sha256"]) == 64
        and all(c in "0123456789abcdef" for c in child["code_sha256"])
        and child.get("first_line") == location["line"]
        and child.get("name") == child.get("qualname")
        and not any(
            node.get(k)
            for k in ("decorators", "defaults", "annotations", "type_params")
        )
        and not any(v is not None for v in node.get("keyword_defaults", ()))
    ):
        raise ValueError(
            "source definition disagrees with its child code or declaration"
        )
    if "parameters" in child:
        parameters = [[p.get("binding"), p.get("name")] for p in declared["params"]]
        frame_args = [
            name
            for kinds in (
                {"posonly", "positional"},
                {"keywordonly"},
                {"varargs"},
                {"kwargs"},
            )
            for kind, name in parameters
            if kind in kinds
        ]
        if not (
            child["parameters"] == parameters
            and isinstance(child.get("varnames"), list)
            and child["varnames"][: len(parameters)] == frame_args
            and nested.get("source_code")
            == dict(parent, kind="function", code_sha256=child["code_sha256"])
        ):
            raise ValueError(
                "published child code disagrees with its parameters or compilation unit"
            )
    rows = proof.get("instructions")
    if not (
        isinstance(rows, list)
        and len(rows) == 3
        and all(
            isinstance(row, dict)
            and row.get("position") == proof["position"]
            and type(row.get("offset")) is int
            for row in rows
        )
    ):
        raise ValueError("source definition has invalid instruction positions")
    load, make, store = rows
    if not (
        0 <= load["offset"] < make["offset"] < store["offset"]
        and load.get("opcode") == "LOAD_CONST"
        and make.get("opcode") == "MAKE_FUNCTION"
        and ("oparg" not in make or type(make["oparg"]) is int and make["oparg"] == 0)
        and store.get("opcode") in {"STORE_NAME", "STORE_GLOBAL"}
        and all(
            type(r.get("index")) is int
            and type(r.get("oparg")) is int
            and 0 <= r["index"] == r["oparg"]
            for r in (load, store)
        )
        and store.get("name") == child["name"]
        and _namespace_target(store, node.get("target"), scope)
    ):
        raise ValueError(
            "source definition disagrees with its construction or publication"
        )


def _bind_import_request(proof, node, scope, *, named):
    request, loc = node["request"], node["location"]
    if not isinstance(proof, dict) or not all(
        isinstance(proof.get(k), dict) for k in ("level", "fromlist", "import_name")
    ):
        raise ValueError("invalid source import operands")
    level, fromlist, imported = (proof[k] for k in ("level", "fromlist", "import_name"))
    code = scope.get("source_code") or {}
    depth = request.get("level")
    immediate = level.get("opcode") == "LOAD_SMALL_INT"
    indexed = (fromlist, imported) if immediate else (level, fromlist, imported)
    if not (
        type(proof.get("semantics_version")) is int
        and proof["semantics_version"] == 1
        and code.get("kind") == "module"
        and proof.get("code_sha256") == code["code_sha256"]
        and loc.get("file") == code["source_key"]
        and proof.get("position") == _source_position(loc)
        and request.get("form") == ("from" if named else "import")
        and isinstance(request.get("module"), str)
        and request["module"] == imported.get("name")
        and type(depth) is int
        and depth >= 0
        and (named or depth == 0)
        and type(level.get("value")) is int
        and level["value"] == depth
        and (
            type(level.get("oparg")) is int
            and level["oparg"] == depth
            and "index" not in level
            if immediate
            else level.get("opcode") == "LOAD_CONST" and "oparg" not in level
        )
        and fromlist.get("opcode") == "LOAD_CONST"
        and "value" in fromlist
        and isinstance(request.get("fromlist"), list)
        and (
            fromlist["value"] == request["fromlist"]
            if named
            else fromlist["value"] is None and not request["fromlist"]
        )
        and imported.get("opcode") == "IMPORT_NAME"
        and all(type(p.get("index")) is int and p["index"] >= 0 for p in indexed)
        and all(
            type(p.get("offset")) is int and p["offset"] >= 0
            for p in (level, fromlist, imported)
        )
        and level["offset"] < fromlist["offset"] < imported["offset"]
    ):
        raise ValueError("source import operands disagree with their graph ports")


def bind_import_store(node, scope):
    """Check an optional bytecode witness against the reconstructed graph ports."""
    projected = "source_import_projection" in node
    if projected and "source_import_store" in node:
        raise ValueError("source import has conflicting publication paths")
    proof = node["source_import_projection" if projected else "source_import_store"]
    _bind_import_request(proof, node, scope, named=False)
    request, target, loc = node["request"], node["binding"], node["location"]
    module = node["module"]
    slot = node.get("alias") or module.partition(".")[0]
    stored = proof.get("store")
    if not (
        isinstance(stored, dict)
        and module == request["module"]
        and projected == ("." in module and node.get("alias") is not None)
        and stored.get("opcode") in {"STORE_NAME", "STORE_GLOBAL"}
        and stored.get("name") == slot
        and type(stored.get("index")) is int
        and stored["index"] >= 0
        and type(stored.get("offset")) is int
        and stored["offset"] > proof["import_name"]["offset"]
        and target.get("op") == "var"
        and target.get("scope") == "global"
        and target.get("name") == f"py:{scope['module']}:{slot}"
    ):
        raise ValueError("source import operands disagree with their graph ports")
    if projected:
        if (
            node.get("member") != module
            or request.get("binding_index") != 0
            or not slot.isidentifier()
            or target.get("location") != loc
        ):
            raise ValueError(
                "source import projection disagrees with its alias binding"
            )
        _bind_import_projection(proof, module.split(".")[1:])
    return proof


def bind_named_fromlist(body, start, scope, references, function_id):
    """Bind the single IMPORT_NAME result to the sibling read/store ports in source order."""
    first = body[start]
    if first.get("op") != "python_import" or not isinstance(first.get("request"), dict):
        raise ValueError("source fromlist requires an import statement")
    proof, request, loc = first["source_fromlist"], first["request"], first["location"]
    _bind_import_request(proof, first, scope, named=True)
    members, bindings, release = (
        request["fromlist"],
        proof.get("bindings"),
        proof.get("release"),
    )
    group = body[start : start + len(members)]
    if not (
        members
        and len(group) == len(members)
        and isinstance(bindings, list)
        and len(bindings) == len(group)
        and isinstance(release, dict)
        and set(release) == {"opcode", "offset"}
        and release["opcode"] == "POP_TOP"
    ):
        raise ValueError("source fromlist is missing a binding or final release")
    if request["level"] == 0 and request["module"] != first.get("module"):
        raise ValueError("source fromlist disagrees with its requested module")
    ordered = [proof["level"], proof["fromlist"], proof["import_name"]]
    sites = []
    for i, (node, member, pair) in enumerate(
        zip(group, members, bindings, strict=True)
    ):
        if not (
            isinstance(node, dict)
            and node.get("op") == "python_import"
            and isinstance(ref := references[start + i], dict)
            and set(ref) == {"port"}
            and type(ref["port"]) is int
            and node.get("site") == f"{function_id}/{ref['port']}"
            and node.get("module") == first["module"]
            and node.get("location") == loc
            and node.get("member") == member
            and isinstance(member, str)
            and member.isidentifier()
            and node.get("request") == dict(request, binding_index=i)
            and type(node["request"]["binding_index"]) is int
            and not (i and "source_fromlist" in node)
            and "source_import_store" not in node
            and "source_import_projection" not in node
            and isinstance(pair, dict)
            and set(pair) == {"read", "store"}
        ):
            raise ValueError("source fromlist disagrees with its ordered binding ports")
        local = node.get("alias") or member
        read, store = pair["read"], pair["store"]
        if not (
            isinstance(local, str)
            and local.isidentifier()
            and all(
                isinstance(row, dict)
                and set(row) == {"opcode", "offset", "index", "name"}
                and type(row["index"]) is int
                and row["index"] >= 0
                for row in (read, store)
            )
            and read["opcode"] == "IMPORT_FROM"
            and read["name"] == member
            and store["opcode"] in {"STORE_NAME", "STORE_GLOBAL"}
            and store["name"] == local
            and _namespace_target(
                dict(store, position=proof["position"]), node.get("binding"), scope
            )
        ):
            raise ValueError("source fromlist disagrees with its member read or store")
        ordered.extend((read, store))
        sites.append(node["site"])
    if len(set(sites)) != len(sites):
        raise ValueError("source fromlist repeats a binding port")
    _bind_import_offsets([*ordered, release])
    first["fromlist_binding_sites"] = sites


def _bind_import_projection(proof, members):
    steps, release = proof.get("projection"), proof.get("release")
    if (
        not isinstance(steps, list)
        or len(steps) != 3 * len(members) - 2
        or not isinstance(release, dict)
        or set(release) != {"opcode", "offset"}
        or release["opcode"] != "POP_TOP"
        or not all(isinstance(row, dict) for row in steps)
    ):
        raise ValueError("invalid source import projection")
    for i, member in enumerate(members):
        fetched = steps[3 * i]
        if (
            set(fetched) != {"opcode", "offset", "index", "name"}
            or fetched["opcode"] != "IMPORT_FROM"
            or fetched["name"] != member
            or type(fetched["index"]) is not int
            or fetched["index"] < 0
        ):
            raise ValueError("source import projection disagrees with its member")
        if i:
            swap, pop = steps[3 * i - 2 : 3 * i]
            if (
                set(swap) != {"opcode", "offset", "oparg"}
                or swap["opcode"] != "SWAP"
                or type(swap["oparg"]) is not int
                or swap["oparg"] != 2
                or set(pop) != {"opcode", "offset"}
                or pop["opcode"] != "POP_TOP"
            ):
                raise ValueError("source import projection lost its parent release")
    ordered = [
        proof["level"],
        proof["fromlist"],
        proof["import_name"],
        *steps,
        proof["store"],
        release,
    ]
    _bind_import_offsets(ordered)


def _bind_import_offsets(ordered):
    # These wordcode instructions have no inline caches. An operand wider than
    # one byte owns its preceding EXTENDED_ARG words.
    for earlier, row in zip(ordered, ordered[1:]):
        arg = row.get("index", row.get("oparg", 0))
        width = max(1, (arg.bit_length() + 7) // 8)
        if (
            type(row.get("offset")) is not int
            or row["offset"] != earlier["offset"] + 2 * width
        ):
            raise ValueError("source import has a discontinuous instruction path")


def loader_ports(node, site, declarations, proof):
    """Connect a resolved source body to its optional activation inside IMPORT_NAME."""
    symbols = node.get("initializers", ())
    if not node.get("graph_backed") or not symbols:
        return None
    candidates = []
    for symbol in symbols:
        scope = declarations.get(symbol, {}).get("python_scope", {})
        code = scope.get("source_code", {})
        if not (
            scope.get("initializer") == symbol
            and scope.get("kind") == code.get("kind") == "module"
            and code.get("source_key") == scope.get("source_key")
            and code.get("code_sha256")
            and code.get("source_sha256")
        ):
            return None
        candidates.append(
            dict(
                initializer=symbol,
                module=scope["module"],
                source_key=code["source_key"],
                code_sha256=code["code_sha256"],
            )
        )
    if candidates[-1]["module"] != node["module"]:
        return None
    return dict(
        site=site,
        importer=dict(frame="builtins", key="__import__"),
        dispatch=dict(
            instruction=proof["import_name"]["offset"], result="import_result"
        ),
        body=dict(
            selection="each_actual_matching_code_activation",
            inputs=["active_frame", "actual_module_globals"],
            completion="return_to_loader_or_propagate",
            candidates=candidates,
        ),
        binding=dict(
            instruction=proof["store"]["offset"],
            input=(
                "projected_import_member" if "projection" in proof else "import_result"
            ),
            on_import_exception="skip",
            **({"on_projection_exception": "skip"} if "projection" in proof else {}),
        ),
    )


def _exception_edges(proof, following):
    """Check references among the captured region's entry, branches and unwinds."""
    entry, load, check, branch, miss = (
        proof[key] for key in ("entry", "load", "check", "branch", "mismatch")
    )
    unwind, normal, cleanup = (
        proof[key] for key in ("unwind", "normal_exit", "exceptional_cleanup")
    )
    table = proof["exception_table"]

    def reaches(offset, target, depth):
        return any(
            e["start"] <= offset < e["end"]
            and e["target"] == target
            and e["depth"] == depth
            and e["lasti"] is bool(depth)
            for e in table
        )

    if not (
        entry["opcode"] == "PUSH_EXC_INFO"
        and load["opcode"] in {"LOAD_NAME", "LOAD_GLOBAL"}
        and check["opcode"] == "CHECK_EXC_MATCH"
        and branch["opcode"] == "POP_JUMP_IF_FALSE"
        and branch["target"] == miss["offset"]
        and entry["offset"]
        < load["offset"]
        < check["offset"]
        < branch["offset"]
        < proof["matched"][-1]["offset"]
        < proof["body_offset"]
        <= miss["offset"]
        and (
            len(proof["matched"]) == 1
            or (
                len(proof["matched"]) == 2
                and proof["matched"][0]["opcode"] == "NOT_TAKEN"
                and branch["offset"]
                < proof["matched"][0]["offset"]
                < proof["matched"][1]["offset"]
                and proof["matched"][0]["position"] == branch["position"]
            )
        )
        and [r["opcode"] for r in unwind] == ["COPY", "POP_EXCEPT", "RERAISE"]
        and any(
            e["target"] == entry["offset"] and e["depth"] == 0 and e["lasti"] is False
            for e in table
        )
        and all(
            reaches(r["offset"], unwind[0]["offset"], 1)
            for r in (load, check, branch, proof["matched"][-1], *cleanup)
        )
    ):
        raise ValueError("source exception dispatch edges are inconsistent")
    if following is None:
        if (miss["opcode"], miss.get("oparg")) != ("RERAISE", 0):
            raise ValueError(
                "final exception mismatch must propagate the caught exception"
            )
    else:
        if (
            following.get("offset") != miss["offset"]
            or following.get("opcode") != miss["opcode"]
        ):
            raise ValueError("source exception mismatch must reach the next matcher")
    binding = proof["matched"][-1]
    if normal and normal[0]["opcode"] != "POP_EXCEPT":
        raise ValueError("source handler exit does not restore exception state")
    if bool(proof["body_raise"]) == bool(normal):
        raise ValueError("source exception handler has an ambiguous exit")
    if binding["opcode"] == "POP_TOP":
        if cleanup or len(normal) > 1:
            raise ValueError("unnamed exception handler cannot clear an alias")
        return
    for rows in ([normal[1:]] if normal else []) + [cleanup[:-1]]:
        if len(rows) != 3:
            raise ValueError("source exception alias lacks its cleanup")
        constant, store, delete = rows
        if not (
            constant["opcode"] == "LOAD_CONST"
            and "value" in constant
            and constant["value"] is None
            and store["opcode"] == binding["opcode"]
            and delete["opcode"] == binding["opcode"].replace("STORE_", "DELETE_")
            and store["name"] == delete["name"] == binding["name"]
            and store["index"] == delete["index"] == binding["index"]
            and constant["offset"] < store["offset"] < delete["offset"]
        ):
            raise ValueError("source exception cleanup disagrees with the bound alias")
    if not cleanup or (cleanup[-1]["opcode"], cleanup[-1].get("oparg")) != (
        "RERAISE",
        1,
    ):
        raise ValueError("source exception cleanup must resume unwinding")


def bind_exception_arms(region, scope):
    """Bind each source handler to its own ordered clause and lexical destinations."""
    arms = region.get("handlers", ())
    for index, arm in enumerate(arms):
        if "source_exception_handler" not in arm:
            continue
        proof = arm["source_exception_handler"]
        code = scope.get("source_code") or {}
        test, alias = arm.get("test"), arm.get("name")
        location = arm["location"]
        if not (
            isinstance(proof, dict)
            and type(proof.get("semantics_version")) is int
            and proof["semantics_version"] == 1
            and code.get("kind") == "module"
            and location.get("file") == code.get("source_key")
            and proof.get("code_sha256") == code.get("code_sha256")
            and proof.get("position") == _source_position(location)
            and type(proof.get("arm_index")) is int
            and proof["arm_index"] == index
            and type(proof.get("arm_count")) is int
            and proof["arm_count"] == len(arms)
            and isinstance(test, dict)
            and test.get("op") in {"python_builtin", "var"}
        ):
            raise ValueError(
                "source exception evidence disagrees with its ordered handler"
            )
        load = proof.get("load")
        if not isinstance(load, dict):
            raise ValueError("source exception handler lacks the matching operand")
        name = load.get("name")
        if not (
            isinstance(name, str)
            and load == test.get("source_name_load")
            and load.get("code_sha256") == code["code_sha256"]
            and load.get("position") == _source_position(test["location"])
            and test["location"]["file"] == location["file"]
            and test.get("name")
            == (f"py:{scope['module']}:{name}" if test.get("op") == "var" else name)
        ):
            raise ValueError("source exception match disagrees with its lookup port")
        binding = arm.get("binding")
        matched = proof.get("matched")
        if (
            not isinstance(matched, list)
            or not matched
            or not isinstance(matched[-1], dict)
        ):
            raise ValueError("invalid entry to source exception handler")
        store = matched[-1]
        if alias:
            if not (
                isinstance(binding, dict)
                and binding.get("scope") == "global"
                and binding.get("op") == "var"
                and binding.get("name") == f"py:{scope['module']}:{alias}"
                and store.get("opcode") in {"STORE_NAME", "STORE_GLOBAL"}
                and store.get("name") == alias
            ):
                raise ValueError("source exception alias disagrees with its graph slot")
        elif binding is not None or store.get("opcode") != "POP_TOP":
            raise ValueError("unnamed exception handler describes an alias")
        try:
            following = (
                arms[index + 1]["test"]["source_name_load"]
                if index + 1 < len(arms)
                else None
            )
            _exception_edges(proof, following)
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("incomplete source exception dispatch evidence") from error

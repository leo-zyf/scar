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


def _bind_condition_edges(node, proof, code, successor):
    branch = proof.get("branch")
    condition = node.get("condition", {})
    negated = condition.get("op") == "unary" and condition.get("operator") == "not"
    operand = condition.get("value", {}) if negated else condition

    def position(operation):
        loc = operation.get("location") if isinstance(operation, dict) else None
        return (
            _source_position(loc)
            if loc and loc.get("file") == code["source_key"]
            else None
        )

    def enters(edge, statement, kind):
        span = position(statement)
        spot = edge.get("position")
        return (
            span is not None
            and edge.get("kind") == kind
            and edge.get("statement_position") == span
            and isinstance(spot, list)
            and len(spot) == 4
            and all(type(x) is int for x in spot)
            and (span[0], span[2])
            <= (spot[0], spot[2])
            <= (spot[1], spot[3])
            <= (span[1], span[3])
        )

    rows = proof["instructions"]
    jump = rows[-1]
    operand_pos = position(operand)
    if not (
        isinstance(branch, dict)
        and operand_pos
        and branch.get("condition_position") == position(condition)
        and branch.get("operand_position") == operand_pos
        and branch.get("negated") is negated
        and jump["opcode"] == ("POP_JUMP_IF_TRUE" if negated else "POP_JUMP_IF_FALSE")
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
        jump["offset"] < then["offset"] < other["offset"] == jump["target"]
        and (
            marker is None
            or (
                isinstance(marker, dict)
                and marker.get("opcode") == "NOT_TAKEN"
                and marker.get("position") == operand_pos
                and type(marker.get("offset")) is int
                and jump["offset"] < marker["offset"] < then["offset"]
            )
        )
        and node.get("then")
        and enters(then, node["then"][0], "suite")
    ):
        raise ValueError(
            "source condition disagrees with its fallthrough or jump target"
        )
    if node.get("else") or successor is not None:
        suite = node.get("else")
        if not enters(
            other,
            suite[0] if suite else successor,
            "suite" if suite else "continuation",
        ):
            raise ValueError(
                "source condition disagrees with its alternative or continuation"
            )
    elif not (
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
    ):
        raise ValueError("source condition lacks the module's implicit return")


def bind_source_expression(node, scope, successor=None):
    """Bind positioned instruction evidence; M2 verifies semantic evaluation order."""
    proof, code = node["source_expression"], scope.get("source_code") or {}
    loc = node["location"]
    position = _source_position(loc)
    if not (
        isinstance(proof, dict)
        and type(proof.get("semantics_version")) is int
        and proof["semantics_version"] == 1
        and code.get("kind") == "module"
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
    elif operation not in {"assign", "unbind", "if"} and not (
        isinstance(node.get("target"), dict)
        and isinstance(node.get("args"), list)
        and isinstance(node.get("keywords"), list)
    ):
        raise ValueError("source call region disagrees with its call port")
    first = position[0], position[2]
    last = position[1], position[3]
    previous = -1
    sequences = None
    binary_ports = None
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
        elif opcode in {"LOAD_CONST", "KW_NAMES", "LOAD_SMALL_INT"}:
            immediate = opcode == "LOAD_SMALL_INT"
            arg = instruction.get("oparg")
            index_matches = (
                "index" not in instruction
                if immediate
                else instruction.get("index") == arg
            )
            if type(arg) is not int or not index_matches:
                raise ValueError("source call literal operand disagrees with its index")
            if instruction.get("type") == "tuple":
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
    elif operation == "assign":
        if not (
            isinstance(node.get("value"), dict)
            and len(proof["instructions"]) > 1
            and last["opcode"] in {"STORE_NAME", "STORE_GLOBAL"}
            and _namespace_target(last, node.get("target"), scope)
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
    elif operation == "if":
        _bind_condition_edges(node, proof, code, successor)
    elif last["opcode"] != "POP_TOP":
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


def bind_import_store(node, scope):
    """Check an optional bytecode witness against the reconstructed graph ports."""
    projected = "source_import_projection" in node
    if projected and "source_import_store" in node:
        raise ValueError("source import has conflicting publication paths")
    proof = node["source_import_projection" if projected else "source_import_store"]
    request, target, loc = node["request"], node["binding"], node["location"]
    module = node["module"]
    slot = node.get("alias") or module.partition(".")[0]
    ports = ("level", "fromlist", "import_name", "store")
    if not isinstance(proof, dict) or not all(
        isinstance(proof.get(key), dict) for key in ports
    ):
        raise ValueError("invalid source import operands")
    level, fromlist, imported, stored = (proof[key] for key in ports)
    code = scope.get("source_code") or {}
    position = proof.get("position")
    # CPython may load the import level from its constant table or encode
    # zero in the instruction itself. Only the former owns a table index.
    immediate = level.get("opcode") == "LOAD_SMALL_INT"
    indexed = (
        (fromlist, imported, stored)
        if immediate
        else (level, fromlist, imported, stored)
    )
    level_matches = (
        type(level.get("oparg")) is int and level["oparg"] == 0 and "index" not in level
        if immediate
        else level.get("opcode") == "LOAD_CONST" and "oparg" not in level
    )
    if not (
        type(proof.get("semantics_version")) is int
        and proof["semantics_version"] == 1
        and code.get("kind") == "module"
        and proof.get("code_sha256") == code["code_sha256"]
        and loc.get("file") == code["source_key"]
        and position == _source_position(loc)
        and request.get("form") == "import"
        and request.get("module") == imported.get("name") == module
        and projected == ("." in module and node.get("alias") is not None)
        and request.get("level") == 0
        and request.get("fromlist") == []
        and level_matches
        and fromlist.get("opcode") == "LOAD_CONST"
        and type(level.get("value")) is int
        and level["value"] == 0
        and "value" in fromlist
        and fromlist["value"] is None
        and imported.get("opcode") == "IMPORT_NAME"
        and stored.get("opcode") in {"STORE_NAME", "STORE_GLOBAL"}
        and stored.get("name") == slot
        and target.get("op") == "var"
        and target.get("scope") == "global"
        and target.get("name") == f"py:{scope['module']}:{slot}"
        and all(type(p.get("index")) is int and p["index"] >= 0 for p in indexed)
        and all(
            type(p.get("offset")) is int and p["offset"] >= 0
            for p in (level, fromlist, imported, stored)
        )
        and level["offset"] < fromlist["offset"] < imported["offset"] < stored["offset"]
    ):
        raise ValueError("source import operands disagree with their graph ports")
    if projected:
        members = module.split(".")[1:]
        if (
            node.get("member") != module
            or request.get("binding_index") != 0
            or not slot.isidentifier()
            or target.get("location") != loc
        ):
            raise ValueError(
                "source import projection disagrees with its alias binding"
            )
        _bind_import_projection(proof, members)
    return proof


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
    # These wordcode instructions have no inline caches. An operand wider than
    # one byte owns its preceding EXTENDED_ARG words.
    for earlier, row in zip(ordered, ordered[1:]):
        arg = row.get("index", row.get("oparg", 0))
        width = max(1, (arg.bit_length() + 7) // 8)
        if (
            type(row.get("offset")) is not int
            or row["offset"] != earlier["offset"] + 2 * width
        ):
            raise ValueError(
                "source import projection has a discontinuous instruction path"
            )


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

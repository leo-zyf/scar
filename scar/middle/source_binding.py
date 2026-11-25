"""M1 validation of optional source instruction evidence against graph ports."""


def _source_position(location):
    return [
        location["line"],
        location["end_line"],
        location["column"] - 1,
        location["end_column"] - 1,
    ]


def bind_call_expression(node, scope):
    """Bind positioned instruction evidence; M2 verifies semantic evaluation order."""
    proof, code = node["source_expression"], scope.get("source_code") or {}
    loc = node["location"]
    position = _source_position(loc)
    if not (
        isinstance(proof, dict)
        and type(proof.get("semantics_version")) is int
        and proof["semantics_version"] == 1
        and code.get("kind") == "module"
        and isinstance(node.get("target"), dict)
        and isinstance(node.get("args"), list)
        and isinstance(node.get("keywords"), list)
        and proof.get("code_sha256") == code.get("code_sha256")
        and loc.get("file") == code.get("source_key")
        and proof.get("position") == position
        and isinstance(proof.get("instructions"), list)
        and proof["instructions"]
    ):
        raise ValueError("source call region disagrees with its module and call port")
    first = position[0], position[2]
    last = position[1], position[3]
    previous = -1
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
        if opcode in {"LOAD_NAME", "LOAD_ATTR", "LOAD_GLOBAL"}:
            arg = instruction.get("oparg")
            if not (
                type(arg) is int
                and isinstance(instruction.get("name"), str)
                and instruction.get("index")
                == (arg if opcode == "LOAD_NAME" else arg >> 1)
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
    if proof["instructions"][-1]["opcode"] != "POP_TOP":
        raise ValueError("source call region has no final result discard")


def bind_import_store(node, scope):
    """Check an optional bytecode witness against the reconstructed graph ports."""
    proof = node["source_import_store"]
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
        and [r["opcode"] for r in unwind] == ["COPY", "POP_EXCEPT", "RERAISE"]
        and any(
            e["target"] == entry["offset"] and e["depth"] == 0 and e["lasti"] is False
            for e in table
        )
        and all(
            reaches(r["offset"], unwind[0]["offset"], 1)
            for r in (load, check, branch, *proof["matched"], *cleanup)
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

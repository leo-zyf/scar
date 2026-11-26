"""Schedule independently invocable source-frame regions without granting an entry ABI."""


def _expression(node, code, minor):
    if node.get("op") != "module_object_expression" or node.get("source_code") != code:
        return []
    receipt = node.get("source_expression", {})
    region = node.get("expression", {})
    if (
        receipt.get("code_sha256") != code["code_sha256"]
        or region.get("op") != "source_object_region"
    ):
        return []
    steps = {s["offset"]: s for s in region["steps"]}
    used, stack, program = set(), [], []
    for ins in receipt["instructions"]:
        opcode, pc = ins["opcode"], ins["offset"]
        step = steps.get(pc, {})
        kind = step.get("op")
        arg = ins.get("index", ins.get("oparg", 0))
        if opcode == "PUSH_NULL":
            if step:
                return []
            stack.append(None)
        elif (
            opcode == "LOAD_NAME"
            and kind == "module_name_load"
            and step["instruction"]["index"] == arg
        ):
            stack.append(pc)
        elif (
            opcode in {"LOAD_CONST", "LOAD_SMALL_INT"}
            and kind == "source_object_constant"
            and step.get("operand") == arg
        ):
            stack.append(pc)
        elif (
            opcode == "LOAD_ATTR"
            and kind == "source_object_attribute"
            and step["result"] == "attribute_object"
        ):
            if (
                ins["oparg"] & 1
                or step["name_index"] != arg
                or not stack
                or step["base"] != {"from_offset": stack.pop()}
            ):
                return []
            stack.append(pc)
        elif (
            opcode == "CALL" and kind == "source_object_call" and len(stack) >= arg + 2
        ):
            target, marker = stack[-arg - 2 : -arg] if arg else stack[-2:]
            if minor == 12:
                marker, target = target, marker
            args = stack[-arg:] if arg else []
            if (
                marker is not None
                or target is None
                or step["target"] != {"from_offset": target}
                or step["args"] != [{"from_offset": v} for v in args]
                or step.get("keyword_names") is not None
            ):
                return []
            del stack[-arg - 2 :]
            stack.append(pc)
        elif (
            opcode == "POP_TOP"
            and kind == "source_object_discard"
            and stack
            and step["value"] == {"from_offset": stack.pop()}
        ):
            pass
        else:
            return []
        if step:
            used.add(pc)
        program.append([opcode, pc, arg, ins.get("name", "")])
    return program if not stack and used == steps.keys() else []


def regions(dispatch, checked):
    """A normal continuation starts immediately after the last included opcode."""
    result = []
    for fn in dispatch["functions"]:
        body = fn["body"]["body"]
        for pos, stmt in enumerate(body):
            op = stmt.get("value", {})
            if stmt.get("op") != "expr" or op.get("op") != "module_import_publish":
                continue
            key = fn["symbol"], op["site"]
            review = checked.get(key)
            if not review or not review["import_binding_verified"]:
                continue
            proof = op.get("source_import_store")
            name = op["request"]["module"]
            code = op["source_code"]
            compiler = code["compiler"].split(".")
            if (
                not proof
                or not name.isidentifier()
                or op["request"].get("level") != 0
                or op["request"].get("fromlist") != []
                or proof["code_sha256"] != code["code_sha256"]
                or compiler[0] != "cpython 3"
                or compiler[1] not in {"12", "14"}
            ):
                continue
            minor = int(compiler[1])
            instructions = [
                proof[k] for k in ("level", "fromlist", "import_name", "store")
            ]
            program = [
                [
                    i["opcode"],
                    i["offset"],
                    i.get("index", i.get("oparg", 0)),
                    i.get("name", ""),
                ]
                for i in instructions
            ]
            # A later unresolved callable ends this component at the source
            # store. Its host resumes with the actual published object.
            if (
                not review["missing"]
                and pos + 1 < len(body)
                and body[pos + 1].get("op") == "expr"
            ):
                suffix = _expression(body[pos + 1]["value"], code, minor)
                if suffix and suffix[0][1] == program[-1][1] + 2:
                    program.extend(suffix)
            result.append(
                dict(
                    owner=key[0],
                    site=key[1],
                    occurrence=op["occurrence"],
                    code_sha256=code["code_sha256"],
                    minor=minor,
                    instructions=program,
                    resume_offset=program[-1][1] + 2,
                )
            )
    return result


def _module_body(loader):
    from scar.contract import digest, json_bytes, source_function_publication

    entry = loader["entry"]
    code = entry["code"]["code_sha256"]
    steps = []
    for stmt in loader["function"]["body"]:
        op = stmt.get("value", {})
        published = (
            source_function_publication(op, code) if stmt["op"] == "expr" else None
        )
        if not published:
            return steps
        make, store, child = published
        load, instruction, _ = op["source_definition"]["instructions"]
        constant = op["evaluation"][0]
        if (
            op.get("source_code") != entry["code"]
            or constant.get("source") != "current_frame_co_consts"
            or constant.get("operand") != load["index"]
            or load["index"] != load["oparg"]
            or instruction.get("oparg", 0) != 0
            or store["index"] != store["oparg"]
            or load["offset"] != (steps[-1][1] + 2 if steps else 2)
            or instruction["offset"] != load["offset"] + 2
            or store["offset"] != instruction["offset"] + 2
        ):
            return steps
        steps.extend(
            [
                [
                    "LOAD_CHILD_CODE",
                    load["offset"],
                    load["index"],
                    child["code_sha256"],
                ],
                ["MAKE_FUNCTION", make["offset"], 0, child["code_sha256"]],
                [store["opcode"], store["offset"], store["index"], store["name"]],
            ]
        )
    frame_exit = entry.get("frame_exit", {})
    tail = frame_exit.get("instructions", [])
    if (
        frame_exit.get("code_sha256") != code
        or not tail
        or tail[0]["offset"] != (steps[-1][1] + 2 if steps else 2)
        or tail[0].get("index") != tail[0].get("oparg")
    ):
        return steps
    if tail[0].get("type") != "NoneType" or tail[0].get("value_sha256") != digest(
        json_bytes(None)
    ):
        return steps
    opcodes = [t["opcode"] for t in tail]
    if opcodes == ["RETURN_CONST"]:
        steps.append(["RETURN_CONST", tail[0]["offset"], tail[0]["index"], ""])
    elif (
        opcodes == ["LOAD_CONST", "RETURN_VALUE"]
        and tail[1]["offset"] == tail[0]["offset"] + 2
    ):
        steps.extend(
            [
                ["LOAD_CONST", tail[0]["offset"], tail[0]["index"], ""],
                ["RETURN_VALUE", tail[1]["offset"], 0, ""],
            ]
        )
    return steps


def _callee_return(fragment):
    if fragment.get("completion") != "conditional_source_body":
        return []
    body = fragment["function"]["body"]
    if len(body) != 1 or body[0]["op"] != "source_object_region":
        return []
    steps = body[0]["steps"]
    if (
        not steps
        or steps[0]["offset"] != 2
        or steps[-1]["op"] != "frame_return"
        or steps[-1].get("destination") != "actual_caller"
    ):
        return []
    returned = steps[-1]
    value = returned["value"]
    if (
        len(steps) == 1
        and returned["opcode"] == "RETURN_CONST"
        and value.get("source") == "current_frame_co_consts"
    ):
        return [["RETURN_CONST", returned["offset"], value["index"], ""]]
    if len(steps) != 2 or returned["opcode"] != "RETURN_VALUE":
        return []
    load = steps[0]
    if (
        value != {"from_offset": load["offset"]}
        or returned["offset"] != load["offset"] + 2
    ):
        return []
    if load["op"] == "source_object_constant":
        opcode = {
            "current_frame_co_consts": "LOAD_CONST",
            "source_small_integer": "LOAD_SMALL_INT",
        }.get(load["source"])
        if opcode:
            return [
                [opcode, load["offset"], load["operand"], ""],
                ["RETURN_VALUE", returned["offset"], 0, ""],
            ]
    return []


def activation_bodies(dispatch, checked):
    """Bind independently enterable body instructions to their original live code."""
    bodies = {}
    callees = {
        name
        for site in checked.values()
        for call in site["calls"]
        for name in call["callees"]
    }
    for role in ("loaders", "callees"):
        for row in dispatch[role]:
            symbol = row["symbol"]
            if role == "loaders":
                site = checked.get((row["owner"], row["site"]), {})
                if symbol not in site.get("loaders", ()):
                    continue
                steps = _module_body(row)
            else:
                if symbol not in callees:
                    continue
                steps = _callee_return(row)
            if not steps:
                continue
            code = row["entry"]["code"]
            slots = row["function"].get("constants", {})
            used = {}
            for op, _, index, _ in steps:
                if op in {"LOAD_CONST", "RETURN_CONST"}:
                    literal = (
                        {"kind": "NoneType", "value": None}
                        if role == "loaders"
                        else slots.get(str(index), {})
                    )
                    kind = literal.get("kind")
                    if type(literal.get("value")).__name__ != kind or kind not in {
                        "NoneType",
                        "bool",
                        "int",
                        "str",
                        "float",
                    }:
                        break
                    if kind == "str" and any(
                        0xD800 <= ord(ch) <= 0xDFFF for ch in literal["value"]
                    ):
                        # The native constant-slot ABI represents text as UTF-8.
                        break
                    used[str(index)] = literal
            else:
                key = symbol, code["code_sha256"]
                body = dict(
                    symbol=symbol,
                    kind=code["kind"],
                    code_sha256=code["code_sha256"],
                    instructions=steps,
                    constants=used,
                    resume_offset=(
                        steps[-1][1] + 2
                        if not steps[-1][0].startswith("RETURN_")
                        else None
                    ),
                )
                if key in bodies and bodies[key] != body:
                    raise ValueError(
                        "conflicting schedules for the same source activation"
                    )
                bodies[key] = body
    return list(bodies.values())

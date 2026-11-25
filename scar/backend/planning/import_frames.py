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
            if key not in checked or checked[key]["missing"]:
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
            if pos + 1 < len(body) and body[pos + 1].get("op") == "expr":
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
                )
            )
    return result

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


def regions(candidates, checked):
    """A normal continuation starts immediately after the last included opcode."""
    result = []
    for key, (op, following) in candidates.items():
        review = checked[key]
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
        instructions = [proof[k] for k in ("level", "fromlist", "import_name", "store")]
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
        if not review["missing"] and following is not None:
            suffix = _expression(following, code, minor)
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


def _constant_store(op, code, offset, slots):
    from scar.contract import digest, json_bytes

    port = op.get("source_constant_store", {})
    load, store = port.get("load", {}), port.get("store", {})
    literal = op.get("constant_value", {})
    idx = load.get("index")
    if (
        op.get("op") != "module_global_store"
        or port.get("code") != code
        or "value" not in literal
        or type(literal["value"]).__name__ != literal.get("kind")
        or literal["kind"] != load.get("type")
        or digest(json_bytes(literal["value"])) != load.get("value_sha256")
        or load.get("opcode") != "LOAD_CONST"
        or type(idx) is not int
        or idx < 0
        or load.get("offset") != offset
        or store.get("offset") != load["offset"] + 2
        or store.get("opcode") not in {"STORE_NAME", "STORE_GLOBAL"}
        or store["opcode"] != op.get("store_opcode")
        or store.get("name") != op.get("name")
        or not isinstance(store.get("name"), str)
        or op.get("namespace")
        != ("frame.locals" if store["opcode"] == "STORE_NAME" else "frame.globals")
        or type(store.get("index")) is not int
        or store["index"] < 0
    ):
        return []
    if str(idx) in slots and json_bytes(slots[str(idx)]) != json_bytes(literal):
        raise ValueError("conflicting values for a module constant slot")
    slots[str(idx)] = literal
    return [
        ["LOAD_CONST", load["offset"], idx, ""],
        [store["opcode"], store["offset"], store["index"], store["name"]],
    ]


def _function_store(op, code, offset):
    from scar.contract import source_function_publication

    publication = source_function_publication(op, code["code_sha256"])
    if not publication or op.get("source_code") != code:
        return []
    make, store, child = publication
    proof = op["source_definition"]
    source = proof["instructions"]
    if any(ins["offset"] != offset + i * 2 for i, ins in enumerate(source)):
        return []
    defaults = proof.get("positional_defaults")
    begin = int(defaults is not None)
    load, creation = source[begin : begin + 2]
    if store["index"] != store["oparg"]:
        return []
    group = []
    if defaults is not None:
        compiler = code.get("compiler", "").split(".")
        if (
            compiler[:2]
            not in (["cpython 3", "12"], ["cpython 3", "13"], ["cpython 3", "14"])
            or (len(source) == 4) != (compiler[1] == "12")
            or defaults["index"] == load["index"]
        ):
            return []
        group.append(
            ["LOAD_DEFAULT_TUPLE", offset, defaults["index"], defaults["count"]]
        )
    group.extend(
        [
            ["LOAD_CHILD_CODE", load["offset"], load["index"], child["code_sha256"]],
            [
                "MAKE_FUNCTION",
                make["offset"],
                creation.get("oparg", 0),
                child["code_sha256"],
            ],
        ]
    )
    for attr in source[begin + 2 : -1]:
        group.append([attr["opcode"], attr["offset"], attr["oparg"], "defaults"])
    group.append([store["opcode"], store["offset"], store["index"], store["name"]])
    return group


def _subscript_store(op, code):
    """Match each live operand in a complete module STORE_SUBSCR statement."""
    from scar.contract import valid_python_source_successor

    receipt = op.get("source_expression", {})
    region = op.get("expression", {})
    source = receipt.get("instructions", ())
    steps = region.get("steps", ())
    if (
        op.get("op") != "module_object_expression"
        or op.get("source_code") != code
        or code.get("kind") != "module"
        or receipt.get("semantics_version") != 1
        or receipt.get("code_sha256") != code["code_sha256"]
        or region.get("op") != "source_object_region"
        or region.get("type") != "unit"
        or not source
        or len(steps) != len(source)
        or source[-1]["opcode"] != "STORE_SUBSCR"
        or not valid_python_source_successor(
            source[-1]["offset"], receipt.get("normal_successor")
        )
    ):
        return []
    stack, lists, program = [], set(), []
    for ins, step in zip(source, steps):
        pc, opcode = ins["offset"], ins["opcode"]
        kind, arg, name = step.get("op"), ins.get("oparg", 0), ""
        # No supported interior opcode has inline caches. A hole could contain
        # another producer or EXTENDED_ARG, so do not admit that interval.
        if (
            type(arg) is not int
            or not 0 <= arg < 2**32
            or type(pc) is not int
            or pc < 2
            or pc % 2
            or step.get("offset") != pc
            or (program and pc != program[-1][1] + 2)
        ):
            return []
        if opcode in {"LOAD_CONST", "LOAD_SMALL_INT"}:
            pool = opcode == "LOAD_CONST"
            if (
                kind != "source_object_constant"
                or step.get("source")
                != ("current_frame_co_consts" if pool else "source_small_integer")
                or step.get("operand") != arg
                or pool
                and ins.get("index") != arg
                or not pool
                and (not code["compiler"].startswith("cpython 3.14.") or arg > 255)
            ):
                return []
            if pool:
                opcode = "LOAD_FRAME_CONST"
                name = (
                    ins.get("items_sha256")
                    if ins.get("type") == "tuple"
                    else ins.get("value_sha256")
                )
                if not isinstance(name, str) or len(name) != 64:
                    return []
            stack.append({"from_offset": pc})
        elif opcode == "LOAD_NAME":
            field = step.get("instruction", {})
            name = ins.get("name")
            if (
                kind != "module_name_load"
                or step.get("lookup_order")
                != ["frame.locals", "frame.globals", "frame.builtins"]
                or field.get("code_sha256") != code["code_sha256"]
                or any(
                    field.get(k) != ins.get(k)
                    for k in ("opcode", "offset", "position", "oparg", "index", "name")
                )
                or ins.get("index") != arg
                or not isinstance(name, str)
                or not name.isidentifier()
            ):
                return []
            stack.append({"from_offset": pc})
        elif opcode == "BUILD_LIST":
            if (
                kind != "source_object_build_sequence"
                or step.get("opcode") != opcode
                or step.get("may_raise") is not True
                or type(arg) is not int
                or not 0 <= arg <= len(stack)
                or step.get("items") != (stack[-arg:] if arg else [])
            ):
                return []
            if arg:
                del stack[-arg:]
            stack.append({"from_offset": pc})
            lists.add(pc)
        elif opcode == "LIST_EXTEND":
            if (
                kind != "source_object_list_update"
                or step.get("opcode") != opcode
                or step.get("may_raise") is not True
                or arg != 1
                or len(stack) < 2
                or stack[-2]["from_offset"] not in lists
                or (step.get("target"), step.get("value")) != tuple(stack[-2:])
            ):
                return []
            stack.pop()
        elif opcode == "STORE_SUBSCR":
            if (
                kind != "source_object_setitem"
                or step.get("opcode") != opcode
                or step.get("may_raise") is not True
                or len(stack) != 3
                or [step.get(k) for k in ("value", "base", "key")] != stack
                or ins is not source[-1]
            ):
                return []
            stack.clear()
        else:
            return []
        program.append([opcode, pc, arg, name])
    return program


def _module_intervals(loader, slots):
    """Yield contiguous, independently enterable runs of whole source statements."""
    from scar.contract import digest, json_bytes

    entry = loader["entry"]
    code = entry["code"]["code_sha256"]
    steps = []
    body = loader["function"]["body"]
    for stmt in body:
        op = stmt.get("value", {}) if stmt["op"] == "expr" else {}
        constant = op.get("source_constant_store", {})
        if constant:
            first = constant["load"]["offset"]
            group = _constant_store(op, entry["code"], first, slots)
        elif proof := op.get("source_definition"):
            first = proof["instructions"][0]["offset"]
            group = _function_store(op, entry["code"], first)
        else:
            group = []
        if steps and (not group or group[0][1] != steps[-1][1] + 2):
            yield steps, None
            steps = []
        steps.extend(group)
        if op.get("op") == "module_object_expression":
            subscript = _subscript_store(op, entry["code"])
            if subscript:
                yield subscript, op["source_expression"]["normal_successor"]
    # Only attach a return reached directly from the last admitted statement,
    # or the entry of an empty module. A source gap needs its own continuation.
    if body and not steps:
        return
    frame_exit = entry.get("frame_exit", {})
    tail = frame_exit.get("instructions", [])
    if (
        frame_exit.get("code_sha256") != code
        or not tail
        or tail[0]["offset"] != (steps[-1][1] + 2 if steps else 2)
        or tail[0].get("index") != tail[0].get("oparg")
    ):
        if steps:
            yield steps, None
        return
    if tail[0].get("type") != "NoneType" or tail[0].get("value_sha256") != digest(
        json_bytes(None)
    ):
        if steps:
            yield steps, None
        return
    slot = str(tail[0]["index"])
    if slot in slots and slots[slot]["value"] is not None:
        raise ValueError("module return slot conflicts with its published constant")
    slots[slot] = {"kind": "NoneType", "value": None}
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
    if steps:
        yield steps, None


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
            slots = dict(row["function"].get("constants", {}))
            if role == "loaders":
                site = checked.get((row["owner"], row["site"]), {})
                if symbol not in site.get("loaders", ()):
                    continue
                intervals = _module_intervals(row, slots)
            else:
                if symbol not in callees:
                    continue
                intervals = [(_callee_return(row), None)]
            for steps, successor in intervals:
                if not steps:
                    continue
                code = row["entry"]["code"]
                used = {}
                for op, _, index, _ in steps:
                    if op in {"LOAD_CONST", "RETURN_CONST"}:
                        literal = slots.get(str(index), {})
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
                    key = symbol, code["code_sha256"], steps[0][1]
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
                    if successor:
                        body["normal_successor"] = successor
                        body["resume_offset"] = successor["offset"]
                    if steps[0][1] != 2:
                        body["entry_offset"] = steps[0][1]
                    if any(step[0] == "LOAD_DEFAULT_TUPLE" for step in steps):
                        body["minor"] = int(code["compiler"].split(".")[1])
                    if key in bodies and bodies[key] != body:
                        raise ValueError(
                            "conflicting schedules for the same source activation"
                        )
                    bodies[key] = body
    return list(bodies.values())

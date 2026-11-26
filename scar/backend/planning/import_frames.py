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


def _frame_operand(ins, step, code):
    """Read one positioned stack operand from the original module frame."""
    opcode, arg, pc = ins["opcode"], ins.get("oparg"), ins["offset"]
    if type(arg) is not int or not 0 <= arg < 2**32 or step.get("offset") != pc:
        return None
    if opcode == "LOAD_NAME":
        field = step.get("instruction", {})
        name = ins.get("name")
        if (
            step.get("op") == "module_name_load"
            and step.get("lookup_order")
            == ["frame.locals", "frame.globals", "frame.builtins"]
            and field.get("code_sha256") == code["code_sha256"]
            and all(
                field.get(k) == ins.get(k)
                for k in ("opcode", "offset", "position", "oparg", "index", "name")
            )
            and ins.get("index") == arg
            and isinstance(name, str)
            and name.isidentifier()
        ):
            return [opcode, pc, arg, name]
    elif step.get("op") == "source_object_constant" and step.get("operand") == arg:
        if (
            opcode == "LOAD_CONST"
            and ins.get("index") == arg
            and step.get("source") == "current_frame_co_consts"
        ):
            identity = (
                ins.get("items_sha256")
                if ins.get("type") == "tuple"
                else ins.get("value_sha256")
            )
            if isinstance(identity, str) and len(identity) == 64:
                return ["LOAD_FRAME_CONST", pc, arg, identity]
        if (
            opcode == "LOAD_SMALL_INT"
            and code["compiler"].startswith("cpython 3.14.")
            and arg < 256
            and step.get("source") == "source_small_integer"
        ):
            return [opcode, pc, arg, ""]
    return None


def _function_store(op, code, offset):
    from scar.contract import source_function_publication

    publication = source_function_publication(op, code["code_sha256"])
    if not publication or op.get("source_code") != code:
        return []
    make, store, child = publication
    proof = op["source_definition"]
    source = proof["instructions"]
    if len(op["evaluation"]) != len(source) - 1 or any(
        ins["offset"] != offset + i * 2 for i, ins in enumerate(source)
    ):
        return []
    defaults = proof.get("positional_defaults")
    split = source[-2]["opcode"] == "SET_FUNCTION_ATTRIBUTE"
    begin = len(source) - 3 - int(split)
    load, creation = source[begin : begin + 2]
    namespace = "frame.locals" if store["opcode"] == "STORE_NAME" else "frame.globals"
    if (
        store["index"] != store["oparg"]
        or op.get("namespace") != namespace
        or op["target"].get("namespace") != namespace
        or op.get("store_opcode") != store["opcode"]
        or op.get("name") != store["name"]
    ):
        return []
    group = []
    if defaults is not None:
        compiler = code.get("compiler", "").split(".")
        if compiler[:2] not in (
            ["cpython 3", "12"],
            ["cpython 3", "13"],
            ["cpython 3", "14"],
        ) or split != (compiler[1] != "12"):
            return []
        count = defaults["count"]
        if "index" in defaults:
            if begin != 1 or defaults["index"] == load["index"]:
                return []
            group.append(["LOAD_DEFAULT_TUPLE", offset, defaults["index"], count])
        else:
            if begin != count + 1:
                return []
            for ins, step in zip(source[:count], op["evaluation"]):
                operand = _frame_operand(ins, step, code)
                if not operand:
                    return []
                group.append(operand)
            build, step = source[count], op["evaluation"][count]
            if (
                build["opcode"] != "BUILD_TUPLE"
                or build.get("oparg") != count
                or build["offset"] != defaults.get("build_offset")
                or step.get("op") != "source_object_build_sequence"
                or step.get("offset") != build["offset"]
                or step.get("opcode") != "BUILD_TUPLE"
                or step.get("may_raise") is not True
                or step.get("items") != [{"from_offset": row[1]} for row in group]
            ):
                return []
            group.append(["BUILD_TUPLE", build["offset"], count, ""])
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
        if opcode in {"LOAD_CONST", "LOAD_SMALL_INT", "LOAD_NAME"}:
            operand = _frame_operand(ins, step, code)
            if not operand:
                return []
            stack.append({"from_offset": pc})
            program.append(operand)
            continue
        if opcode == "BUILD_LIST":
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
                    if any(
                        step[0] in {"LOAD_DEFAULT_TUPLE", "BUILD_TUPLE"}
                        for step in steps
                    ):
                        body["minor"] = int(code["compiler"].split(".")[1])
                    if key in bodies and bodies[key] != body:
                        raise ValueError(
                            "conflicting schedules for the same source activation"
                        )
                    bodies[key] = body
    return list(bodies.values())


def _isolated_declaration(module, ledger, dispatch):
    """Resolve the declared deployment separately from the analysis process."""
    env = module.get("python_entry_environment") or {}
    entries = env.get("entries", ())
    if (
        len(entries) != 1
        or entries[0].get("image") != module.get("current_image")
        or ledger.get("entry_environment") != env
        or module.get("layer_contract", {}).get("completion") != "complete"
        or module.get("entry_abi") is not None
        or not dispatch
        or dispatch.get("completion") != "modeled_source_entry"
    ):
        return None
    declaration = entries[0]["declaration"]
    if (declaration.get("kind"), declaration.get("deployment")) != (
        "standalone_script",
        "isolated_project_modules_v1",
    ) or declaration.get("flags", []) not in ([], ["-E", "-S", "-B"]):
        return None
    return declaration


def _project_name(key):
    import _imp
    import sys

    return (
        isinstance(key, str)
        and key.isidentifier()
        and key not in sys.stdlib_module_names
        and not _imp.is_builtin(key)
        and not _imp.is_frozen(key)
    )


def isolated_project_entry(module, ledger, dispatch, rows, regions, bodies):
    """Prove total coverage of a closed, single-threaded local import program."""
    import platform
    import sys

    from scar.contract import digest, json_bytes

    decl = _isolated_declaration(module, ledger, dispatch)
    if decl is None or "entry_frame" not in dispatch:
        return None
    env = module["python_entry_environment"]
    by_code = {row["code_sha256"]: row for row in bodies}
    if not bodies or len(by_code) != len(bodies):
        return None
    entry = dispatch["entry_frame"]["code"]
    # Built-in and frozen names can pre-empt disk resolution. The executing
    # compiler supplies only this exclusion list, never initial application state.
    if (
        entry.get("compiler") != "cpython " + platform.python_version()
        or sys.implementation.name != "cpython"
    ):
        return None
    entry_exit = dispatch["entry_frame"].get("frame_exit", {})
    exits = entry_exit.get("instructions", ())
    if (
        dispatch["completion"] != "modeled_source_entry"
        or [step.get("opcode") for step in exits]
        not in (["RETURN_CONST"], ["LOAD_CONST", "RETURN_VALUE"])
        or len(exits) == 2
        and exits[1]["offset"] != exits[0]["offset"] + 2
        or exits[0].get("type") != "NoneType"
        or exits[0].get("value_sha256") != digest(json_bytes(None))
        or entry_exit.get("code_sha256") != entry["code_sha256"]
        or not regions
        or not (len(regions) == len(rows) == len(ledger["occurrences"]))
    ):
        return None
    loader_at = {(row["owner"], row["site"]): row for row in dispatch["loaders"]}
    occurrences = {row["id"]: row for row in ledger["occurrences"]}
    if set(occurrences) != {region["occurrence"] for region in regions}:
        return None
    if any(
        row["activation"] != "visited"
        or (row.get("source_dispatch") or {}).get("status")
        != "modeled_source_path_requires_native_implementation"
        for row in rows
    ):
        return None
    # Only literal-return zero-argument functions and plain module publications
    # are admitted. Calls, custom objects, hooks and handler transitions would
    # require a different set of process/frame services.
    functions = {}
    for row in dispatch["callees"]:
        code = row["entry"]["code"]["code_sha256"]
        body = by_code.get(code, {})
        steps = body.get("instructions", ())
        if (
            row["entry"].get("parameters") != []
            or body.get("kind") != "function"
            or [step[0] for step in steps]
            not in (
                ["RETURN_CONST"],
                ["LOAD_CONST", "RETURN_VALUE"],
                ["LOAD_SMALL_INT", "RETURN_VALUE"],
            )
            or body.get("resume_offset") is not None
        ):
            return None
        result = (
            {"kind": "int", "value": steps[0][2]}
            if steps[0][0] == "LOAD_SMALL_INT"
            else body["constants"][str(steps[0][2])]
        )
        if result["kind"] not in {"str", "NoneType", "bool", "int"}:
            return None
        if type(result["value"]) is int and result["value"].bit_length() > 63:
            return None
        if isinstance(result["value"], str) and not result["value"].isascii():
            return None
        functions[code] = "scalar"
    modules, slots, previous, names = {}, {}, 2, {}
    for region in sorted(regions, key=lambda row: row["instructions"][0][1]):
        steps = region["instructions"]
        occurrence = occurrences[region["occurrence"]]
        effect = occurrence["import_effects"]
        identity = effect["source_identity"]
        loader = loader_at.get((region["owner"], region["site"]), {})
        key = identity["module"]
        code = loader.get("entry", {}).get("code", {})
        body = by_code.get(code.get("code_sha256"), {})
        if (
            (occurrence["owner"], occurrence["site"])
            != (region["owner"], region["site"])
            or not _project_name(key)
            or identity["status"] != "project"
            or code.get("source_key") != key + ".py"
            or body.get("kind") != "module"
            or body["symbol"] != loader["symbol"]
            or body["resume_offset"] is not None
            or not body["instructions"]
            or body["instructions"][-1][0] not in {"RETURN_CONST", "RETURN_VALUE"}
            or region["owner"] != module["entry"]
            or region["code_sha256"] != entry["code_sha256"]
            or steps[0][1] != previous
            or len(steps) < 4
            or steps[2][0] != "IMPORT_NAME"
            or steps[2][3] != key
            or steps[3][0] not in {"STORE_NAME", "STORE_GLOBAL"}
        ):
            return None
        exports = {}
        tail = 1 if body["instructions"][-1][0] == "RETURN_CONST" else 2
        if tail == 2 and (
            len(body["instructions"]) < 2 or body["instructions"][-2][0] != "LOAD_CONST"
        ):
            return None
        publications = body["instructions"][:-tail]
        if len(publications) % 3:
            return None
        for i in range(0, len(publications), 3):
            make_code, create, store = publications[i : i + 3]
            if (
                make_code[0] != "LOAD_CHILD_CODE"
                or make_code[3] not in functions
                or create[0] != "MAKE_FUNCTION"
                or create[2] != 0
                or create[3] != make_code[3]
                or store[0] not in {"STORE_NAME", "STORE_GLOBAL"}
            ):
                return None
            exports[store[3]] = make_code[3]
        info = dict(name=key, symbol=body["symbol"], code_sha256=body["code_sha256"])
        if key in modules and modules[key] != info:
            return None
        modules[key], slots[key] = info, exports
        names[steps[3][3]] = ("module", key)
        if not _isolated_expression(region, names, slots):
            return None
        previous = region["resume_offset"]
    if previous != exits[0]["offset"]:
        return None
    return dict(
        deployment=decl["deployment"],
        code_sha256=entry["code_sha256"],
        ledger_sha256=digest(json_bytes(ledger)),
        entry_environment_sha256=digest(json_bytes(env)),
        modules=list(modules.values()),
        exit_offset=previous,
        frame_tables_sha256=digest(json_bytes([regions, bodies])),
    )


def _isolated_expression(region, names, exports):
    """Prove each qualified source lookup/call before using the native object ABI."""
    stack = []
    for op, _, arg, spelling in region["instructions"][4:]:
        if op == "PUSH_NULL":
            stack.append(None)
        elif op == "LOAD_NAME":
            value = names.get(
                spelling, ("builtin", "print") if spelling == "print" else None
            )
            if value is None:
                return False
            stack.append(value)
        elif op == "LOAD_ATTR" and stack and stack[-1] and stack[-1][0] == "module":
            child = exports[stack.pop()[1]].get(spelling)
            if child is None:
                return False
            stack.append(("function", child))
        elif op == "CALL" and len(stack) >= arg + 2:
            group = stack[-(arg + 2) :]
            del stack[-(arg + 2) :]
            called, null = (
                (group[1], group[0]) if region["minor"] == 12 else (group[0], group[1])
            )
            if (
                null is not None
                or called is None
                or not (
                    called[0] == "function"
                    and arg == 0
                    or called == ("builtin", "print")
                    and all(v == ("scalar", None) for v in group[2:])
                )
            ):
                return False
            stack.append(("scalar", None))
        elif op == "POP_TOP" and stack:
            stack.pop()
        else:
            return False
    return not stack


def compiled_project_entry(module, ledger, dispatch, reviews):
    """Bind typed import sites to actual local loader bodies in a closed entry."""
    import builtins
    import platform
    import sys

    from scar.contract import EXECUTION_ANNOTATIONS, digest, json_bytes, walk_operations

    declaration = _isolated_declaration(module, ledger, dispatch)
    if (
        declaration is None
        or sys.implementation.name != "cpython"
        or sys.version_info.minor not in {12, 14}
        or module.get("globals")
        or module.get("externs")
        or module.get("records")
    ):
        return None
    env = module["python_entry_environment"]

    def operations(body):
        return walk_operations(body, exclude_fields=EXECUTION_ANNOTATIONS)

    functions = {fn["name"]: fn for fn in module["functions"]}
    root = module["entry"]
    occurrences = {r["id"]: r for r in ledger["occurrences"]}
    loaders = {(r["owner"], r["site"], r["symbol"]): r for r in dispatch["loaders"]}
    originals = {
        (scope["symbol"], op["occurrence"]): op
        for scope in dispatch["functions"]
        for op in operations(scope["body"]["body"])
        if op["op"] == "module_import_publish"
    }
    bindings, modules, sites = set(), {}, {}
    for name, fn in functions.items():
        if fn["params"] or fn["result"] != "void" or fn.get("initializers"):
            return None
        for op in operations(fn["body"]):
            if op["op"] != "project_module_import":
                continue
            key, site, symbol = op["module"], op["site"], op["callee"]
            identity = op["occurrence"]
            src = originals.get((name, identity), {})
            row = occurrences.get(identity, {})
            loader = loaders.get((name, site, symbol), {})
            code = loader.get("entry", {}).get("code", {})
            reviewed = reviews.get((name, site), {})
            source_identity = row.get("import_effects", {}).get("source_identity", {})
            if (
                name != root
                or identity in sites
                or symbol not in functions
                or row.get("owner") != name
                or row.get("site") != site
                or row.get("request")
                != dict(form="import", module=key, fromlist=[], level=0)
                or (source_identity.get("status"), source_identity.get("module"))
                != ("project", key)
                or not _project_name(key)
                or not src
                or op.get("source_import_occurrence") != identity
                or op.get("loader_ports") != row.get("loader_ports")
                or op.get("on_hit") != "use_actual_cached_module"
                or op.get("on_miss")
                != "publish_initializing_module_then_execute_initializer"
                or op.get("normal") != "publish_actual_import_result_at_source_store"
                or op.get("exceptional_exit_semantics")
                != "on_initializer_error_remove_its_cache_entry_skip_caller_store_and_propagate"
                or not op.get("source_code")
                or op["source_code"].get("compiler")
                != "cpython " + platform.python_version()
                or any(
                    op.get(k) != src.get(k)
                    for k in ("source_code", "site", "source_import_store", "binding")
                )
                or op.get("source_import_store") is None
                or not reviewed.get("import_binding_verified")
                or reviewed.get("missing")
                or reviewed.get("loaders") != [symbol]
                or code.get("source_key") != key + ".py"
                or code.get("compiler") != op["source_code"]["compiler"]
            ):
                return None
            provider = (symbol, code["code_sha256"])
            if modules.setdefault(key, provider) != provider:
                return None
            store = src["source_import_store"]["store"]
            binding = store["name"]
            bindings.add(binding)
            sites[identity] = dict(
                owner=name,
                module=key,
                binding=binding,
                callee=symbol,
                code_sha256=code["code_sha256"],
                operation_sha256=digest(json_bytes(op)),
            )
    if (
        not sites
        or sites.keys() != occurrences.keys()
        or set(functions) != {root, *(s for s, _ in modules.values())}
    ):
        return None
    # Narrow primitive contract. All reachable module writes come from named
    # imports; these cannot replace any built-in used by printing or handlers.
    if any(
        hasattr(builtins, binding) or binding.startswith("__") for binding in bindings
    ):
        return None
    failures = {}
    for symbol, _ in modules.values():
        effects = _local_exits(functions[symbol]["body"], {})
        if effects is None:
            return None
        failures[symbol] = effects
    if _local_exits(functions[root]["body"], failures) != set():
        return None
    return dict(
        deployment=declaration["deployment"],
        strategy="compiled_initializers",
        minor=sys.version_info.minor,
        initializer_errors={name: sorted(kinds) for name, kinds in failures.items()},
        entry=root,
        sites=sites,
        ledger_sha256=digest(json_bytes(ledger)),
        entry_environment_sha256=digest(json_bytes(env)),
        functions_sha256={
            name: digest(json_bytes(fn)) for name, fn in functions.items()
        },
    )


def _local_exits(body, callees):
    """Compute escaping language errors for the closed scalar/IO import profile."""

    locals_ = {}

    def abstract(expr):
        if not isinstance(expr, dict):
            return None
        op, ty = expr.get("op"), expr.get("type")
        if (
            op == "literal"
            and ty in ("text", "bool", "i64", "unit")
            and (not isinstance(expr.get("value"), str) or expr["value"].isascii())
        ):
            return ("scalar", ty)
        if op == "var":
            value = locals_.get(expr.get("name"))
            return value[1] if value and value[0] == ty else None
        if (
            op == "format_value"
            and expr.get("conversion") == "str"
            and expr.get("spec", {}).get("empty") is True
            and abstract(expr.get("value")) == ("scalar", "text")
        ):
            return ("scalar", "text")
        if (
            op == "error_make"
            and expr.get("category") in {"value_error", "type_error", "runtime_error"}
            and abstract(expr.get("message")) == ("scalar", "text")
        ):
            return ("error", expr["category"])
        return None

    failures = set()
    for stmt in body:
        op = stmt["op"]
        value = stmt.get("value", {})
        if op == "decl":
            initial = abstract(stmt.get("init"))
            if initial is None or stmt["name"] in locals_:
                return None
            locals_[stmt["name"]] = (stmt["type"], initial)
            continue
        if (
            op == "expr"
            and value.get("op") == "project_module_import"
            and value.get("callee") in callees
        ):
            failures.update(callees[value["callee"]])
        elif (
            op == "expr"
            and value.get("op") == "text_print"
            and "stream" not in value
            and all(
                (item := abstract(arg)) and item[0] == "scalar"
                for arg in [*value["args"], value["sep"], value["end"], value["flush"]]
            )
        ):
            continue
        elif op == "throw" and "cause" not in stmt:
            error = abstract(value)
            if not error or error[0] != "error":
                return None
            failures.add(error[1])
        elif op == "try" and not stmt.get("finally") and not stmt.get("else"):
            pending = _local_exits(stmt["body"], callees)
            if pending is None:
                return None
            for handler in stmt["handlers"]:
                if handler.get("binding") is not None or any(
                    k in handler for k in ("match", "on_match", "cleanup")
                ):
                    return None
                categories = handler.get("categories")
                if (
                    not isinstance(categories, list)
                    or not categories
                    or not all(isinstance(k, str) for k in categories)
                ):
                    return None
                effects = _local_exits(handler["body"], callees)
                if effects is None:
                    return None
                caught = {
                    kind
                    for kind in pending
                    if kind in categories
                    or {"exception", "base_exception"}.intersection(categories)
                }
                if caught:
                    pending -= caught
                    failures.update(effects)
            failures.update(pending)
        elif op == "return" and stmt.get("value") is None:
            continue
        else:
            return None
    return failures

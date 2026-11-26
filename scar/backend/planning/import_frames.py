"""Schedule independently invocable source-frame regions without granting an entry ABI."""


def _expression(node, code, minor):
    if node.get("op") != "module_object_expression" or node.get("source_code") != code:
        return []
    receipt = node.get("source_expression", {})
    region = node.get("expression", {})
    if (
        receipt.get("code_sha256") != code["code_sha256"]
        or region.get("op") != "source_object_region"
        or node.get("code_match")
        != "qualify_current_frame_code_and_expression_instruction_order_stack_operands_spans_and_jumps_before_first_operand"
        or node.get("on_code_mismatch")
        != "resume_source_equivalent_execution_before_first_operand"
        or node.get("unwind")
        != "on_failure_unwind_only_live_source_stack_references_at_failing_opcode_no_duplicate_effects"
        or node.get("exceptional_exit")
        != "propagate_at_failing_source_operation_preserving_completed_effects"
    ):
        return []
    steps = {s["offset"]: s for s in region["steps"]}
    used, stack, program = set(), [], []
    for ins in receipt["instructions"]:
        opcode, pc = ins["opcode"], ins["offset"]
        step = steps.get(pc, {})
        kind = step.get("op")
        arg = ins.get("index", ins.get("oparg", 0))
        # The two supported bytecodes have nine cache words after LOAD_ATTR
        # and three after CALL. Reject gaps (including unmodeled EXTENDED_ARG).
        if type(pc) is not int or pc < 2 or pc % 2 or type(arg) is not int or arg < 0:
            return []
        if program:
            prior, offset, _, _ = program[-1]
            if pc != offset + {"LOAD_ATTR": 20, "CALL": 8}.get(prior, 2):
                return []
        if opcode == "PUSH_NULL":
            if step:
                return []
            stack.append(None)
        elif (
            opcode == "LOAD_NAME"
            and kind == "module_name_load"
            and step["instruction"]["index"] == arg == ins.get("oparg")
            and step["instruction"].get("code_sha256") == code["code_sha256"]
            and step["instruction"].get("name") == ins.get("name")
            and step["instruction"].get("offset") == pc
            and step.get("lookup_order")
            == ["frame.locals", "frame.globals", "frame.builtins"]
        ):
            stack.append(pc)
        elif (
            opcode in {"LOAD_CONST", "LOAD_SMALL_INT"}
            and kind == "source_object_constant"
            and step.get("operand") == arg
            and (opcode != "LOAD_SMALL_INT" or minor == 14 and arg < 256)
        ):
            stack.append(pc)
        elif (
            opcode == "LOAD_ATTR"
            and kind == "source_object_attribute"
            and step["result"] == "attribute_object"
        ):
            if (
                ins["oparg"] != arg * 2
                or step["name_index"] != arg
                or step.get("name") != ins.get("name")
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
                or None in args
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


def regions(candidates, checked, calls):
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
    included = {
        (r["owner"], r["code_sha256"], pc)
        for r in result
        for opcode, pc, _, _ in r["instructions"]
        if opcode == "CALL"
    }
    # Re-enter independently at the actual caller PC. No jump from the
    # import to a later consumer, or across an enclosing control arm.
    for key, occurrence, expression, call in calls:
        if key not in candidates or checked[key]["missing"]:
            continue
        code = candidates[key][0]["source_code"]
        identity = key[0], code["code_sha256"], call["offset"]
        compiler = code["compiler"].split(".")[:2]
        if identity in included or compiler not in (
            ["cpython 3", "12"],
            ["cpython 3", "14"],
        ):
            continue
        minor = int(compiler[1])
        program = _expression(expression, code, minor)
        if (
            not program
            or program[-1][0] != "POP_TOP"
            or not any(
                op == "CALL" and pc == call["offset"] for op, pc, _, _ in program
            )
        ):
            continue
        result.append(
            dict(
                owner=key[0],
                site=call["site"],
                occurrence=occurrence,
                import_site=key[1],
                code_sha256=code["code_sha256"],
                minor=minor,
                profile="live_call",
                call_offset=call["offset"],
                instructions=program,
                resume_offset=program[-1][1] + 2,
            )
        )
        included.add(identity)
    return result


def named_region(owner, code, op, occurrence):
    """Bind a from-list to one IMPORT_NAME, ordered lookup/store pairs and POP_TOP."""
    port, call = op.get("source_fromlist", {}), op.get("import_call", {})
    request, bindings = op.get("request", {}), op.get("bindings", [])
    members, level = request.get("fromlist", []), port.get("level", {})
    names, enter, release = (
        port.get(key, {}) for key in ("fromlist", "import_name", "release")
    )
    if (
        op.get("source_code") != code
        or port.get("code_sha256") != code.get("code_sha256")
        or port.get("semantics_version") != 1
        or not isinstance(code.get("compiler"), str)
        or code["compiler"].split(".")[:2]
        not in (["cpython 3", "12"], ["cpython 3", "14"])
        or occurrence.get("id") != op.get("source_import_occurrence")
        or op.get("occurrence") != occurrence.get("id")
        or (op.get("site"), owner) != (occurrence.get("site"), occurrence.get("owner"))
        or occurrence.get("activation") not in {"visited", "unvisited"}
        or occurrence.get("request") != request
        or request.get("form") != "from"
        or type(request.get("level")) is not int
        or request["level"] < 0
        or level.get("value") != request["level"]
        or not isinstance(request.get("module"), str)
        or not isinstance(members, list)
        or not members
        or not all(isinstance(n, str) and n.isidentifier() for n in members)
        or names.get("opcode") != "LOAD_CONST"
        or names.get("value") != members
        or enter.get("opcode") != "IMPORT_NAME"
        or enter.get("name") != request["module"]
        or release.get("opcode") != "POP_TOP"
        or not (
            len(bindings)
            == len(port.get("bindings", []))
            == len(occurrence.get("bindings", []))
            == len(members)
        )
        or op.get("after_store")
        != {"instruction": release, "release": {"result": "import_result"}}
        or op.get("unwind")
        != "on_import_lookup_or_store_failure_unwind_live_references_at_that_opcode_preserve_completed_effects_and_skip_later_bindings"
        or op.get("lookup")
        != "successive_IMPORT_FROM_on_retained_actual_return_including_circular_registry_fallback"
        or op.get("publication")
        != "for_each_binding_read_then_store_its_value_in_the_source_mapping_before_next_read"
        or op.get("code_match")
        != "require_current_activation_code_and_instruction_operands_match_before_first_source_import_operand"
        or op.get("exceptional_exit")
        != "propagate_at_failing_source_operation_preserving_completed_effects"
        or op.get("dispatch")
        != "call_current_frame_importer_once_with_original_arguments"
        or op.get("initialization")
        != "only_as_invoked_by_actual_importer_including_ordered_fromlist_prechecks"
        or op.get("on_code_mismatch")
        != "resume_source_equivalent_execution_before_first_source_import_operand"
    ):
        return None
    minor = int(code["compiler"].split(".")[1])
    if (
        level.get("opcode") == "LOAD_SMALL_INT"
        and minor == 14
        and level.get("oparg") == request["level"]
    ):
        argument = {"opcode": "LOAD_SMALL_INT", "oparg": level["oparg"]}
    elif level.get("opcode") == "LOAD_CONST":
        argument = {"table": "co_consts", "index": level.get("index")}
    else:
        return None
    if (
        call.get("opcode") != "IMPORT_NAME"
        or call.get("importer") != {"mapping": "frame.builtins", "key": "__import__"}
        or call.get("result") != "import_result"
        or call.get("registry")
        != "actual_importer_controls_cache_initialization_hooks_and_audit"
        or call.get("arguments")
        != [
            {"table": "co_names", "index": enter.get("index")},
            {"frame": "globals"},
            {"frame": "locals"},
            {"table": "co_consts", "index": names.get("index")},
            argument,
        ]
    ):
        return None
    instructions = [level, names, enter]
    projections = []
    for i, (spelling, binding, pair, record) in enumerate(
        zip(members, bindings, port["bindings"], occurrence["bindings"], strict=True)
    ):
        read, store = pair["read"], pair["store"]
        target = {"from_offset": read.get("offset")}
        if (
            read.get("opcode") != "IMPORT_FROM"
            or read.get("name") != spelling
            or binding.get("member") != spelling
            or binding.get("binding_index") != i
            or binding.get("read")
            != {
                "receiver": {"result": "import_result"},
                "instruction": read,
                "result": target,
            }
            or store.get("opcode") not in {"STORE_NAME", "STORE_GLOBAL"}
            or binding.get("value") != target
            or binding.get("namespace")
            != ("frame.locals" if store["opcode"] == "STORE_NAME" else "frame.globals")
            or any(binding.get(k) != store.get(k) for k in ("opcode", "offset", "name"))
            or binding.get("key") != {"table": "co_names", "index": store.get("index")}
            or (record.get("member"), record.get("site"), record.get("name"))
            != (spelling, binding.get("site"), binding.get("slot"))
        ):
            return None
        instructions.extend((read, store))
        projections.append(
            dict(site=binding["site"], member=spelling, read=read, store=store)
        )
    instructions.append(release)
    if "body" in call and not occurrence.get("loader_ports"):
        return None
    if (ports := occurrence.get("loader_ports")) is not None:
        binding = ports.get("binding", {})
        if (
            ports.get("site") != op["site"]
            or ports.get("dispatch", {}).get("instruction") != enter["offset"]
            or call.get("body") != ports.get("body")
            or binding.get("input") != "import_result"
            or binding.get("reads") != projections
            or binding.get("on_import_exception") != "skip"
            or binding.get("release") != release["offset"]
        ):
            return None
    start = level.get("offset")
    if (
        type(start) is not int
        or start < 2
        or start % 2
        or any(
            ins.get("offset") != start + 2 * i
            or type(ins.get("index", ins.get("oparg", 0))) is not int
            or not 0 <= ins.get("index", ins.get("oparg", 0)) < 2**32
            for i, ins in enumerate(instructions)
        )
    ):
        return None
    steps = [
        [
            ins["opcode"],
            ins["offset"],
            ins.get("index", ins.get("oparg", 0)),
            ins.get("name", ""),
        ]
        for ins in instructions
    ]
    return dict(
        profile="live_fromlist",
        owner=owner,
        site=op["site"],
        occurrence=op["occurrence"],
        code_sha256=code["code_sha256"],
        minor=minor,
        instructions=steps,
        resume_offset=release["offset"] + 2,
    )


def named_loaders(op, occurrence, loaders):
    """Yield conditional source bodies selected by this verified from-list."""
    for candidate in (
        occurrence.get("loader_ports", {}).get("body", {}).get("candidates", ())
    ):
        child = loaders.get((occurrence["owner"], op["site"], candidate["initializer"]))
        entry = child["entry"] if child and child.get("route") == "fromlist" else {}
        selected = entry.get("code", {})
        if (
            selected.get("source_key") == candidate.get("source_key")
            and selected.get("code_sha256") == candidate["code_sha256"]
            and entry.get("selection") == "each_actual_matching_code_activation"
            and entry.get("on_mismatch")
            == "resume_source_equivalent_execution_before_module_body"
        ):
            yield child


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


def unpack_body(owner, code, op):
    """Schedule one reached module destructuring statement, including partial writes."""
    from scar.contract import valid_python_source_successor

    receipt = op.get("source_expression", {})
    source = receipt.get("instructions", [])
    evaluation = op.get("evaluation", [])
    successor = receipt.get("normal_successor")
    version = code.get("compiler", "").split(".")[:2]
    if (
        op.get("op") != "module_unpack_store"
        or op.get("type") != "unit"
        or op.get("semantics_version") != 1
        or op.get("frame_globals", {}).get("owner") != owner
        or op.get("frame_globals", {}).get("identity") != "current_activation_globals"
        or op.get("exceptional_exit")
        != "propagate_at_failing_source_operation_preserving_completed_effects"
        or op.get("on_code_mismatch")
        != "resume_source_equivalent_execution_before_first_instruction_with_original_stack_and_exception_state"
        or op.get("ownership")
        != "retain_each_owned_operand_until_consumed_at_its_source_opcode"
        or op.get("release")
        != "preserve_replaced_or_deleted_values_release_reentry_and_unraisable_finalizer_errors"
        or op.get("callbacks")
        != "preserve_instruction_tracing_reentry_and_release_of_live_source_stack_references"
        or op.get("source_code") != code
        or code.get("kind") != "module"
        or version not in (["cpython 3", "12"], ["cpython 3", "14"])
        or receipt.get("code_sha256") != code["code_sha256"]
        or receipt.get("semantics_version") != 1
        or op.get("code_match")
        != "qualify_current_activation_code_instruction_stack_and_exception_table_before_first_source_instruction"
        or op.get("assignment")
        != "consume_each_owned_unpack_result_at_its_source_target_preserving_completed_stores_on_failure"
        or op.get("unwind")
        != "on_failure_unwind_at_failing_opcode_follow_source_exception_table_and_preserve_completed_effects"
        or not source
        or not valid_python_source_successor(source[-1].get("offset"), successor)
    ):
        return None
    minor = int(version[1])
    steps = iter(evaluation)
    stack, program = [], []
    pc = source[0].get("offset")
    if type(pc) is not int or pc < 2 or pc % 2:
        return None
    width = stored = 0
    targets = sum(
        a.get("opcode") in {"STORE_SUBSCR", "STORE_NAME", "STORE_GLOBAL"}
        for a in source
    )
    for ins in source:
        opcode, n, text = ins["opcode"], ins.get("oparg", 0), ""
        if type(n) is not int or not 0 <= n < 2**32 or ins.get("offset") != pc:
            return None
        if opcode == "PUSH_NULL":
            if n:
                return None
            stack.append(None)
        else:
            step = next(steps, {})
            if step.get("offset") != pc:
                return None
            kind = step.get("op")
            if operand := _frame_operand(ins, step, code):
                opcode, _, n, text = operand
                stack.append({"from_offset": pc})
            elif opcode == "LOAD_ATTR" and kind == "source_object_attribute":
                text = ins.get("name")
                if (
                    not stack
                    or stack[-1] is None
                    or step.get("base") != stack[-1]
                    or step.get("result") != "attribute_object"
                    or ins.get("index") != n >> 1
                    or n & 1
                    or step.get("name_index") != n >> 1
                    or step.get("name") != text
                    or not isinstance(text, str)
                    or not text.isidentifier()
                ):
                    return None
                n >>= 1
                stack[-1] = {"from_offset": pc}
            elif (
                opcode == "CALL"
                and kind == "source_object_call"
                and len(stack) >= n + 2
            ):
                group = stack[-(n + 2) :]
                target, null = (
                    (group[1], group[0]) if minor == 12 else (group[0], group[1])
                )
                if (
                    target is None
                    or null is not None
                    or None in group[2:]
                    or step.get("target") != target
                    or step.get("args") != group[2:]
                    or step.get("keyword_names") is not None
                    or step.get("keyword_values") != []
                    or step.get("target_layout") != "callable_with_explicit_NULL"
                    or step.get("dispatch")
                    != "consume_source_call_slots_including_method_receiver_at_most_once"
                ):
                    return None
                stack[-(n + 2) :] = [{"from_offset": pc}]
            elif opcode == "UNPACK_SEQUENCE" and kind == "source_object_unpack":
                if (
                    width
                    or n != targets
                    or n < 1
                    or len(stack) != 1
                    or stack[0] is None
                    or step.get("value") != stack[0]
                    or step.get("arity") != n
                    or step.get("opcode") != opcode
                    or step.get("may_raise") is not True
                ):
                    return None
                width = n
                stack = [
                    dict(from_offset=pc, result_index=i) for i in reversed(range(n))
                ]
            elif opcode == "STORE_SUBSCR" and kind == "source_object_setitem":
                if (
                    n
                    or len(stack) < 3
                    or not stack[-3]
                    or stack[-3].get("result_index") != stored
                    or [step.get(k) for k in ("value", "base", "key")] != stack[-3:]
                    or step.get("opcode") != opcode
                    or step.get("may_raise") is not True
                    or step.get("write_effect", {}).get("exceptional_completion")
                    != "may_have_written"
                ):
                    return None
                del stack[-3:]
                stored += 1
            elif (
                opcode in {"STORE_NAME", "STORE_GLOBAL"}
                and kind == "source_frame_namespace_store"
            ):
                text = ins.get("name")
                record = step.get("instruction", {})
                if (
                    not stack
                    or not stack[-1]
                    or stack[-1].get("result_index") != stored
                    or step.get("value") != stack[-1]
                    or not isinstance(text, str)
                    or not text.isidentifier()
                    or step.get("name") != text
                    or ins.get("index") != n
                    or step.get("index") != n
                    or step.get("may_raise") is not True
                    or step.get("namespace")
                    != ("frame.locals" if opcode == "STORE_NAME" else "frame.globals")
                    or any(
                        record.get(k) != ins.get(k)
                        for k in (
                            "offset",
                            "opcode",
                            "oparg",
                            "index",
                            "name",
                            "position",
                        )
                    )
                ):
                    return None
                stack.pop()
                stored += 1
            else:
                return None
        program.append([opcode, pc, n, text])
        pc += {"LOAD_ATTR": 20, "CALL": 8, "UNPACK_SEQUENCE": 4, "STORE_SUBSCR": 4}.get(
            opcode, 2
        )
    if (
        stack
        or not width
        or stored != width
        or next(steps, None) is not None
        or successor["offset"] != pc
    ):
        return None
    return dict(
        symbol=owner,
        kind="module",
        profile="live_unpack_store",
        code_sha256=code["code_sha256"],
        minor=minor,
        entry_offset=program[0][1],
        instructions=program,
        constants={},
        normal_successor=successor,
        resume_offset=pc,
    )


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


def _callee_calls(fragment):
    """Join complete, straight-line call regions to the sealed terminal return."""
    from scar.contract import digest, json_bytes, valid_python_source_successor

    code = fragment.get("entry", {}).get("code", {})
    version = code.get("compiler", "").split(".")[:2]
    if version not in (["cpython 3", "12"], ["cpython 3", "14"]):
        return []
    minor = int(version[1])
    body = fragment["function"]["body"]
    slots = fragment["function"].get("constants", {})
    pc, successor, program = 2, None, []
    for segment in body:
        source = segment.get("source_expression", {})
        instructions, steps = source.get("instructions", []), segment.get("steps", [])
        if (
            segment.get("op") != "source_object_region"
            or source.get("code_sha256") != code["code_sha256"]
            or not instructions
            or (
                segment is body[-1]
                and len(body) > 1
                and [i.get("opcode") for i in instructions]
                in (["RETURN_CONST"], ["LOAD_CONST", "RETURN_VALUE"])
                and successor is None
            )
            or len(steps) != len(instructions)
            or successor is not None
            and successor.get("opcode") != instructions[0].get("opcode")
        ):
            return []
        stack = []
        for ins, step in zip(instructions, steps, strict=True):
            opcode, op = ins.get("opcode"), step.get("op")
            arg, name = ins.get("oparg", 0), ""
            if (
                type(arg) is not int
                or not 0 <= arg < 2**32
                or ins.get("offset") != pc
                or step.get("offset") != pc
            ):
                return []
            if opcode == "LOAD_GLOBAL" and op == "module_name_load":
                load = step.get("instruction", {})
                name = ins.get("name")
                if (
                    not isinstance(name, str)
                    or not name.isidentifier()
                    or ins.get("index") != arg >> 1
                    or load.get("code_sha256") != code["code_sha256"]
                    or any(
                        load.get(k) != ins.get(k)
                        for k in (
                            "offset",
                            "opcode",
                            "oparg",
                            "index",
                            "position",
                            "name",
                        )
                    )
                    or step.get("lookup_order") != ["frame.globals", "frame.builtins"]
                    or step.get("lookup")
                    != "source_mapping_lookup_propagate_nonmissing_errors_or_NameError"
                ):
                    return []
                if arg & 1:
                    opcode = "LOAD_GLOBAL_CALL"
                    stack.extend([None, pc] if minor == 12 else [pc, None])
                else:
                    stack.append(pc)
                arg >>= 1
            elif opcode in {"LOAD_CONST", "LOAD_SMALL_INT", "RETURN_CONST"}:
                if opcode == "LOAD_SMALL_INT":
                    if minor != 14 or arg >= 256:
                        return []
                    value, kind = arg, "int"
                else:
                    literal = slots.get(str(arg), {})
                    if ins.get("index") != arg or "value" not in literal:
                        return []
                    value, kind = literal["value"], literal.get("kind")
                if (
                    ins.get("value_sha256") != digest(json_bytes(value))
                    or ins.get("type") != kind
                ):
                    return []
                if opcode == "RETURN_CONST":
                    if (
                        minor != 12
                        or stack
                        or step.get("value")
                        != {"source": "current_frame_co_consts", "index": arg}
                    ):
                        return []
                else:
                    source_kind = (
                        "source_small_integer"
                        if opcode == "LOAD_SMALL_INT"
                        else "current_frame_co_consts"
                    )
                    if (
                        op != "source_object_constant"
                        or step.get("source") != source_kind
                        or step.get("operand") != arg
                    ):
                        return []
                    stack.append(pc)
            elif (
                opcode == "CALL"
                and op == "source_object_call"
                and len(stack) >= arg + 2
            ):
                group = stack[-(arg + 2) :]
                target, marker = (
                    (group[1], group[0]) if minor == 12 else (group[0], group[1])
                )
                if (
                    target is None
                    or marker is not None
                    or None in group[2:]
                    or step.get("target") != {"from_offset": target}
                    or step.get("args") != [{"from_offset": i} for i in group[2:]]
                    or step.get("target_layout") != "callable_with_implicit_NULL"
                    or step.get("keyword_names") is not None
                    or step.get("keyword_values") != []
                    or step.get("dispatch")
                    != "consume_source_call_slots_including_method_receiver_at_most_once"
                ):
                    return []
                stack[-(arg + 2) :] = [pc]
            elif (opcode, op) in {
                ("POP_TOP", "source_object_discard"),
                ("RETURN_VALUE", "frame_return"),
            }:
                if (
                    arg
                    or len(stack) != 1
                    or stack[0] is None
                    or step.get("value") != {"from_offset": stack.pop()}
                ):
                    return []
            else:
                return []
            if opcode.startswith("RETURN_") and (
                segment is not body[-1]
                or ins is not instructions[-1]
                or op != "frame_return"
                or step.get("opcode") != opcode
                or step.get("destination") != "actual_caller"
                or step.get("lifetime")
                != "transfer_result_and_release_frame_locals_at_source_exit"
            ):
                return []
            program.append([opcode, pc, arg, name])
            pc += {"LOAD_GLOBAL": 10, "LOAD_GLOBAL_CALL": 10, "CALL": 8}.get(opcode, 2)
        if stack or (segment is not body[-1] and program[-1][0] != "POP_TOP"):
            return []
        successor = source.get("normal_successor")
        if successor is not None and (
            not valid_python_source_successor(program[-1][1], successor)
            or successor["offset"] != pc
        ):
            return []
    if not program or program[-1][0] not in {"RETURN_CONST", "RETURN_VALUE"}:
        return []
    if any(i[0] == "CALL" for i in program) or [i[0] for i in program] in (
        ["RETURN_CONST"],
        ["LOAD_CONST", "RETURN_VALUE"],
        ["LOAD_SMALL_INT", "RETURN_VALUE"],
    ):
        return program
    return []


def _callee_return(fragment):
    if fragment.get("completion") != "conditional_source_body":
        return []
    body = fragment["function"]["body"]
    if any(r.get("source_expression") for r in body):
        return _callee_calls(fragment)
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
                    if role == "callees" and any(step[0] == "CALL" for step in steps):
                        body.update(
                            profile="live_function_calls",
                            minor=int(code["compiler"].split(".")[1]),
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


def _native_scalar(slot):
    """Values represented exactly by the isolated import object's display ABI."""
    kind, value = slot.get("kind"), slot.get("value")
    return type(value).__name__ == kind and (
        kind in {"NoneType", "bool"}
        or kind == "int"
        and value.bit_length() <= 63
        or kind == "str"
        and value.isascii()
    )


def _module_exports(body, functions):
    """Evaluate literal and function publications of a complete plain module."""
    steps = body["instructions"]
    tail = 1 if steps[-1][0] == "RETURN_CONST" else 2
    if (
        steps[0][1] != 2
        or body.get("entry_offset", 2) != 2
        or (
            tail == 2
            and (
                len(steps) < 2
                or steps[-2][0] != "LOAD_CONST"
                or steps[-1][0] != "RETURN_VALUE"
            )
        )
        or body["constants"].get(str(steps[-tail][2]))
        != {"kind": "NoneType", "value": None}
    ):
        return None
    exports = {}
    cursor, stop = 0, len(steps) - tail
    while cursor < stop:
        load = steps[cursor]
        if load[0] == "LOAD_CONST" and _native_scalar(
            body["constants"].get(str(load[2]), {})
        ):
            value = ("scalar", None)
            cursor += 1
        elif (
            load[0] == "LOAD_CHILD_CODE" and load[3] in functions and cursor + 1 < stop
        ):
            create = steps[cursor + 1]
            if create[0] != "MAKE_FUNCTION" or create[2] != 0 or create[3] != load[3]:
                return None
            value = ("function", load[3])
            cursor += 2
        else:
            return None
        if cursor >= stop or steps[cursor][0] not in {"STORE_NAME", "STORE_GLOBAL"}:
            return None
        key = steps[cursor][3]
        if key.startswith("__") and key.endswith("__") and key != "__doc__":
            return None
        exports[key] = value
        cursor += 1
    return exports


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
    imports = [r for r in regions if "profile" not in r]
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
        or not (len(imports) == len(rows) == len(ledger["occurrences"]))
    ):
        return None
    loader_at = {(row["owner"], row["site"]): row for row in dispatch["loaders"]}
    occurrences = {row["id"]: row for row in ledger["occurrences"]}
    if set(occurrences) != {region["occurrence"] for region in imports}:
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
        if not _native_scalar(result):
            return None
        functions[code] = "scalar"
    modules, slots, previous, names = {}, {}, 2, {}
    for region in sorted(regions, key=lambda row: row["instructions"][0][1]):
        steps = region["instructions"]
        occurrence = occurrences[region["occurrence"]]
        if (
            region["owner"] != module["entry"]
            or region["code_sha256"] != entry["code_sha256"]
            or steps[0][1] != previous
        ):
            return None
        previous = region["resume_offset"]
        if region.get("profile") == "live_call":
            if (region["owner"], region["import_site"]) != (
                occurrence["owner"],
                occurrence["site"],
            ) or not _isolated_expression(region, names, slots, start=0):
                return None
            continue
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
            or len(steps) < 4
            or steps[2][0] != "IMPORT_NAME"
            or steps[2][3] != key
            or steps[3][0] not in {"STORE_NAME", "STORE_GLOBAL"}
        ):
            return None
        exports = _module_exports(body, functions)
        if exports is None:
            return None
        info = dict(name=key, symbol=body["symbol"], code_sha256=body["code_sha256"])
        if key in modules and modules[key] != info:
            return None
        modules[key], slots[key] = info, exports
        names[steps[3][3]] = ("module", key)
        if not _isolated_expression(region, names, slots):
            return None
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


def _isolated_expression(region, names, exports, *, start=4):
    """Prove each qualified source lookup/call before using the native object ABI."""
    stack = []
    for op, _, arg, spelling in region["instructions"][start:]:
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
            stack.append(child)
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

"""M2 operations on a live module namespace, including lookup and object release."""

from collections import ChainMap
import re

from scar.contract import (
    EXECUTION_ANNOTATIONS,
    MODULE_REGISTRY_OPERATIONS,
    digest,
    json_bytes,
    python_constant_rhs_sha256,
    source_call_operands,
    walk_operations,
)
from . import source_frames
from .values import literal, source_site, written_targets

REGISTRY_PROFILE = "cpython-importlib-registry-v1"


def require_registry_operator(project, operation):
    if operation not in MODULE_REGISTRY_OPERATIONS:
        raise ValueError(f"unregistered module-registry IR operation: {operation}")
    project.require_operator(
        "module_registry",
        operation,
        profile=REGISTRY_PROFILE,
        scope="process_module_registry",
        entry_contract={
            "initial_cache": "unobserved_by_static_capture",
            "graph_module_first_import": "require_absent_entry_before_each_scheduled_graph_initializer",
            "hooks": "preserve_import_callbacks_audit_and_failures",
            "interference": "exclude_unmodelled_registry_replacement_across_each_folded_lookup",
            "bound_callable": "qualify_the_recorded_source_revision_and_unmodified_CPython_import_module_cache_path",
        },
    )


def global_delete_slots(frame, deletion):
    """DELETE_NAME operands must name the current activation's lexical globals."""
    if deletion.get("op") != "unbind" or not frame.initializing:
        return []
    slots = []
    prefix = f"py:{frame.module}:"
    for target in deletion.get("targets", ()):
        slot = target.get("name", "")
        if (
            target.get("op") != "var"
            or target.get("scope") != "global"
            or target.get("cell")
            or not isinstance(slot, str)
            or not slot.startswith(prefix)
            or not slot[len(prefix) :].isidentifier()
        ):
            return []
        slots.append(slot)
    return slots


def _fromlist_group(frame, body, index):
    """Match a single source request with its ordered lexical bindings."""
    first = body[index]
    request = first.get("request", {})
    members = request.get("fromlist", ())
    end = index + len(members)
    module = first.get("module")
    if (
        not frame.initializing
        or request.get("form") != "from"
        or request.get("binding_index") != 0
        or not request.get("level")
        and request.get("module") != module
        or not members
        or end > len(body)
    ):
        return None
    port = first.get("source_fromlist")
    # This operation can itself open the namespace. Its source instruction
    # still selects the mapping before any later callback has run.
    code = (
        _module_code(frame, source_code=frame.scope.get("source_code"))
        if port is not None
        else None
    )
    verified = (
        code is not None
        and port["code_sha256"] == code["code_sha256"]
        and first.get("fromlist_binding_sites")
        == [n.get("site") for n in body[index:end]]
    )
    if not verified and module not in frame.project.initializers:
        return None
    if not frame.project.imports.open_namespace and (
        frame.dynamic
        or frame.region is not frame.root_region
        or module != frame.module
        or not frame.scope.get("package")
        or end == len(body)
        or body[end].get("op") != "unbind"
    ):
        return None
    level = request.get("level", 0)
    requested = request.get("module")
    if type(level) is not int or level < 0 or not isinstance(requested, str):
        return None
    if level and not verified:
        package = (
            frame.module
            if frame.scope.get("package")
            else frame.module.rpartition(".")[0]
        )
        parents = package.split(".")
        if not package or level > len(parents):
            return None
        resolved = ".".join(
            parents[: len(parents) - level + 1] + ([requested] if requested else [])
        )
        if resolved != module:
            return None
    parts = module.split(".")
    expected = [
        frame.project.initializers[parent]
        for depth in range(1, len(parts) + 1)
        if (parent := ".".join(parts[:depth])) in frame.project.initializers
    ]
    if first.get("initializers") != expected:
        return None
    if any(
        frame.project.functions[symbol]["python_scope"].get("graph_backed") is not True
        for symbol in expected
    ):
        return None
    occurrence = frame.project.source_imports.get(frame.name, first)
    bindings = []
    for position, (node, member) in enumerate(zip(body[index:end], members)):
        target = node.get("binding", {})
        local = node.get("alias") or member
        if (
            node.get("op") != "python_import"
            or not verified
            and node.get("graph_backed") is not True
            or node.get("module") != module
            or node.get("member") != member
            or not isinstance(member, str)
            or not member.isidentifier()
            or not isinstance(local, str)
            or not local.isidentifier()
            or position
            and (
                node.get("initializers")
                or node.get("conditional_initializers")
                or node.get("fromlist_checks")
            )
            or node.get("request", {}).get("binding_index") != position
            or frame.project.source_imports.get(frame.name, node) is not occurrence
            or target.get("op") != "var"
            or target.get("scope") != "global"
            or target.get("cell")
            or target.get("name") != f"py:{frame.module}:{local}"
        ):
            return None
        bindings.append(
            dict(
                binding_index=position,
                member=member,
                name=local,
                slot=target["name"],
                source_evidence=dict(
                    scope="source_candidates_only",
                    requested_module=module,
                    bound_value=node.get("bound_value", {}),
                ),
                **source_site(node),
            )
        )
    if not frame.project.imports.open_namespace and set(
        global_delete_slots(frame, body[end])
    ) != {b["slot"] for b in bindings}:
        return None
    return occurrence, bindings, port if verified else None, code


def publish_fromlist(frame, body, index, steps):
    group = _fromlist_group(frame, body, index)
    if group is None:
        return 0
    occurrence, bindings, port, code = group
    first = body[index]
    frame.import_origin(first)
    occurrence["activation"] = "visited"
    evidence = frame.package_fromlist(first, occurrence, dynamic=True)
    publication = "bind_each_successful_member_in_source_order_to_its_lexical_slot"
    publications = [dict(occurrence=occurrence["id"], **b) for b in bindings]
    source = {
        "mapping_qualification": "require_source_store_or_proven_module_locals_globals_identity"
    }
    if port is not None:
        for binding, source_port in zip(bindings, port["bindings"], strict=True):
            read = _import_member(source_port["read"], {"result": "import_result"})
            binding.update(
                read=read, **_store_import_value(source_port["store"], read["result"])
            )
        source = _import_operands(first, port, code)
        source.update(
            source_fromlist=port,
            writes=tuple(dict.fromkeys(b["namespace"] for b in bindings)),
            after_store=dict(
                instruction=port["release"], release={"result": "import_result"}
            ),
            unwind="on_import_lookup_or_store_failure_unwind_live_references_at_that_opcode_preserve_completed_effects_and_skip_later_bindings",
        )
        publication = "for_each_binding_read_then_store_its_value_in_the_source_mapping_before_next_read"
    namespace_effect(
        frame,
        first,
        steps,
        "module_fromlist_publish",
        source_protocol="python.import",
        occurrence=occurrence["id"],
        request=occurrence["request"],
        resolved_module=first["module"],
        bindings=bindings,
        fromlist_checks=evidence["ordered_checks"],
        initializers=first["initializers"],
        source_effects=occurrence["source_effects"],
        dispatch="call_current_frame_importer_once_with_original_arguments",
        initialization="only_as_invoked_by_actual_importer_including_ordered_fromlist_prechecks",
        lookup="successive_IMPORT_FROM_on_retained_actual_return_including_circular_registry_fallback",
        publication=publication,
        release="preserve_replaced_values_operand_stack_unwind_and_final_import_result_release",
        callbacks="preserve_source_opcode_tracing_reentry_and_unraisable_finalizer_errors",
        **source,
    )
    for published in publications:
        frame.project.imports.publications[published["slot"]] = published
    return len(bindings)


def _module_code(frame, *, source_code=None):
    code = frame.scope.get("source_code")
    if (
        (source_code is None and not frame.project.imports.open_namespace)
        or source_code is not None
        and source_code != code
        or not frame.initializing
        or frame.scope.get("graph_backed") is not True
        or frame.name != frame.scope.get("initializer")
        or not isinstance(code, dict)
        or code.get("kind") != "module"
        or not isinstance(code.get("source_key"), str)
        or code.get("source_key") != frame.scope.get("source_key")
        or not isinstance(code.get("compiler"), str)
        or not code["compiler"].startswith("cpython ")
        or type(code.get("optimize")) is not int
        or code["optimize"] not in (0, 1, 2)
        or not all(
            isinstance(code.get(key), str) and re.fullmatch("[0-9a-f]{64}", code[key])
            for key in ("source_sha256", "code_sha256")
        )
    ):
        return None
    return code


def _import_operands(node, port, code):
    level = port["level"]
    level_operand = (
        {"opcode": "LOAD_SMALL_INT", "oparg": level["oparg"]}
        if level["opcode"] == "LOAD_SMALL_INT"
        else {"table": "co_consts", "index": level["index"]}
    )
    return dict(
        source_code=code,
        reads=("frame.code", "frame.locals"),
        import_call=dict(
            opcode="IMPORT_NAME",
            importer={"mapping": "frame.builtins", "key": "__import__"},
            arguments=[
                {"table": "co_names", "index": port["import_name"]["index"]},
                {"frame": "globals"},
                {"frame": "locals"},
                {"table": "co_consts", "index": port["fromlist"]["index"]},
                level_operand,
            ],
            result="import_result",
            **(
                {"body": node["loader_ports"]["body"]} if "loader_ports" in node else {}
            ),
            registry="actual_importer_controls_cache_initialization_hooks_and_audit",
        ),
        code_match="require_current_activation_code_and_instruction_operands_match_before_first_source_import_operand",
        on_code_mismatch="resume_source_equivalent_execution_before_first_source_import_operand",
    )


def _store_import_value(store, value):
    return dict(
        namespace=(
            "frame.locals" if store["opcode"] == "STORE_NAME" else "frame.globals"
        ),
        opcode=store["opcode"],
        offset=store["offset"],
        key={"table": "co_names", "index": store["index"]},
        value=value,
    )


def _import_member(instruction, receiver):
    return dict(
        instruction=instruction,
        receiver=receiver,
        result={"from_offset": instruction["offset"]},
    )


def _project_import_members(port, result):
    """Connect the verified IMPORT_FROM, SWAP and POP_TOP operations to live references."""
    stack = [result]
    steps = []
    for instruction in port["projection"]:
        opcode = instruction["opcode"]
        step = dict(instruction=instruction)
        if opcode == "IMPORT_FROM":
            step = _import_member(instruction, stack[-1])
            stack.append(step["result"])
        elif opcode == "SWAP":
            stack[-2], stack[-1] = stack[-1], stack[-2]
        elif opcode == "POP_TOP":
            step["release"] = stack.pop()
        else:
            raise ValueError("unrecognized import projection instruction")
        steps.append(step)
    receiver, member = stack
    return member, dict(
        input=result,
        steps=steps,
        lookup="source_IMPORT_FROM_on_actual_receiver_including_circular_import_fallback",
        after_store=dict(instruction=port["release"], release=receiver),
        unwind="on_failure_unwind_live_source_stack_at_failing_opcode_and_preserve_prior_effects",
    )


def publish_plain_import(frame, node, occurrence, steps, *, source_code=None):
    """Consume the actual importer result through its verified projection and store."""
    request = occurrence["request"]
    name = request["module"]
    target = node.get("binding", {})
    slot = target.get("name")
    alias = node.get("alias")
    if not isinstance(name, str) or not all(p.isidentifier() for p in name.split(".")):
        return False
    bound_name = alias or name.partition(".")[0]
    projected = alias is not None and "." in name
    field = "source_import_projection" if projected else "source_import_store"
    port = node.get(field)
    if (
        not (code := _module_code(frame, source_code=source_code))
        or request.get("form") != "import"
        or request.get("level") != 0
        or request.get("fromlist") != []
        or node.get("member") != name
        or target.get("cell")
        or not node.get("site")
        or slot != f"py:{frame.module}:{bound_name}"
        or len(occurrence["bindings"]) != 1
        or occurrence["bindings"][0]["site"] != node["site"]
        or occurrence["bindings"][0]["name"] != slot
        or not isinstance(port, dict)
        or port.get("code_sha256") != code["code_sha256"]
    ):
        return False
    # M1 verifies the source instructions, their order, names and stack contract.
    store = port["store"]
    imported = {"result": "import_result"}
    value, projection = (
        _project_import_members(port, imported) if projected else (imported, None)
    )
    binding = dict(
        slot=slot,
        name=bound_name,
        **_store_import_value(store, value),
        **source_site(node),
    )
    namespace_effect(
        frame,
        node,
        steps,
        "module_import_publish",
        source_protocol="python.import",
        occurrence=occurrence["id"],
        request=request,
        **_import_operands(node, port, code),
        **{field: port},
        **({"projection": projection} if projected else {}),
        writes=(binding["namespace"],),
        binding=binding,
        unwind="on_import_failure_skip_store_on_store_failure_preserve_completed_effects_and_unwind_source_stack",
        release="preserve_source_stack_ownership_replaced_binding_release_reentry_and_unraisable_finalizer_errors",
        callbacks="preserve_source_opcode_tracing_and_exception_state",
    )
    frame.project.imports.publications[slot] = dict(
        occurrence=occurrence["id"], slot=slot, name=bound_name, **source_site(node)
    )
    return True


def store_open_module_constant(frame, node, steps):
    """Retain the source constant object and store opcode across callback effects."""
    port = node.get("source_constant_store")
    if (
        not isinstance(port, dict)
        or type(port.get("semantics_version")) is not int
        or port["semantics_version"] != 1
        or not (code := _module_code(frame))
        or port.get("code") != code
    ):
        return False
    load, store = port.get("load"), port.get("store")
    if not isinstance(load, dict) or not isinstance(store, dict):
        return False
    value, target = node["value"], node["target"]
    if "rhs_sha256" in port:
        fingerprint = python_constant_rhs_sha256(value)
        if (
            not fingerprint
            or value.get("op") not in {"binary", "unary"}
            or port["rhs_sha256"] != fingerprint
        ):
            return False
    elif not (
        load.get("type") == "str"
        and value.get("op") == "literal"
        and value.get("type") == {"kind": "python_object", "class": "str"}
        and type(value.get("value")) is str
        and load.get("value_sha256") == digest(json_bytes(value["value"]))
    ):
        return False
    if (
        load.get("opcode") != "LOAD_CONST"
        or store.get("opcode") not in ("STORE_NAME", "STORE_GLOBAL")
        or not isinstance(store.get("name"), str)
        or not store["name"].isidentifier()
        or target.get("op") != "var"
        or target.get("scope") != "global"
        or target.get("cell")
        or target.get("name") != f"py:{frame.module}:{store['name']}"
        or not source_frames.matches_instruction(load, value, code["source_key"])
        or not source_frames.matches_instruction(store, target, code["source_key"])
        or store["offset"] <= load["offset"]
    ):
        return False
    mapping = "frame.locals" if store["opcode"] == "STORE_NAME" else "frame.globals"
    namespace_effect(
        frame,
        node,
        steps,
        "module_global_store",
        reads=(
            ("frame.code", "frame.locals")
            if mapping == "frame.locals"
            else ("frame.code",)
        ),
        writes=[mapping],
        source_constant_store=port,
        name=store["name"],
        slot=target["name"],
        store_opcode=store["opcode"],
        namespace=mapping,
        code_match="require_current_activation_code_and_instruction_operands_at_sealed_offsets_match_before_LOAD_CONST",
        on_code_mismatch="resume_source_equivalent_execution_before_LOAD_CONST",
        constant="load_same_object_from_current_frame_code_co_consts_at_index",
        publication="use_current_frame_code_co_names_key_and_source_store_opcode",
        release="preserve_replaced_binding_release_reentry_and_stack_cleanup_on_failure",
        callbacks="preserve_opcode_tracing_intervening_callbacks_and_unraisable_finalizer_errors",
    )
    return True


def _source_region(frame, receipt):
    return source_frames.SourceExpressionRegion(
        frame.scope,
        receipt,
        publications=frame.project.imports.publications,
        definitions=ChainMap(
            frame.project.functions, frame.project.function_definitions
        ),
    )


def _object_operand(frame, node, code):
    return source_frames.object_operand(
        frame.module, frame.project.imports.publications, node, code
    )


def open_object_statement(frame, node, steps):
    """Consume a complete source evaluation stack at its actual store, delete or raise."""
    code = _module_code(frame)
    field = (
        "source_definition" if node["op"] == "python_define" else "source_expression"
    )
    receipt = node.get(field)
    if not code or not receipt:
        return False
    try:
        region = _source_region(frame, receipt)
        opcode, ports = region.statement(node)
    except (ValueError, IndexError, KeyError, TypeError) as error:
        frame.fail(node, "source_expression_operands", str(error))
    _register_expression(frame, region.steps)
    namespace_effect(
        frame,
        node,
        steps,
        opcode,
        source_code=code,
        **{field: receipt},
        evaluation=region.steps,
        **ports,
        reads=("frame.code", "frame.locals", "python.exception_state"),
        code_match="qualify_current_activation_code_instruction_stack_and_exception_table_before_first_source_instruction",
        **(
            {
                "child_code_match": "qualify_actual_code_constant_type_identity_descriptor_and_empty_freevars_before_first_source_instruction"
            }
            if field == "source_definition"
            else {}
        ),
        on_code_mismatch="resume_source_equivalent_execution_before_first_instruction_with_original_stack_and_exception_state",
        ownership="retain_each_owned_operand_until_consumed_at_its_source_opcode",
        release="preserve_replaced_or_deleted_values_release_reentry_and_unraisable_finalizer_errors",
        unwind="on_failure_unwind_at_failing_opcode_follow_source_exception_table_and_preserve_completed_effects",
        callbacks="preserve_instruction_tracing_reentry_and_release_of_live_source_stack_references",
    )
    return True


def _object_truth(frame, node, code):
    if node["op"] == "short_circuit":
        terms, operator = node["operands"], node["operator"]
        if len(terms) < 2 or operator not in ("and", "or"):
            return None
        tests = [_object_truth(frame, term, code) for term in terms]
        if any(test is None for test in tests):
            return None
        result = tests[-1]
        stop = operator == "or"
        for test in reversed(tests[:-1]):
            result = dict(
                op="select",
                type="bool",
                condition=test,
                then=literal(True).value if stop else result,
                **{"else": result if stop else literal(False).value},
                **source_site(node),
            )
        return result
    value = _object_operand(frame, node, code)
    if value is not None:
        return dict(
            op="source_object_truth",
            type="bool",
            value=value,
            dispatch="actual_truth_protocol_once_then_release",
            **source_site(node),
        )


def _register_expression(frame, expression):
    for item in walk_operations(expression, exclude_fields={"source_evidence"}):
        if item["op"] in MODULE_REGISTRY_OPERATIONS:
            require_registry_operator(frame.project, item["op"])
        if item.get("site") is not None:
            frame.project.normalized_sites.add((frame.name, item["site"]))
            frame.project.live_sites.add((frame.name, item["site"]))


def open_object_loop(frame, node, steps):
    """A live source iterator owns its item, exhaustion and exceptional edges."""
    receipt = node.get("source_expression")
    if not (
        receipt and node.get("protocol") == "iterator" and (code := _module_code(frame))
    ):
        return None
    ports = receipt.get("iteration")
    try:
        if not ports or any(
            part["op"]
            in {
                "break",
                "continue",
                "return",
                "try",
                "while",
                "for",
                "if",
                "with_context",
            }
            for part in walk_operations([node["body"], node.get("else", [])])
        ):
            raise ValueError(
                "source iterator needs verified straight-line suites and exits"
            )
        region = _source_region(frame, receipt)
        iterable = region.value(node["range"])
        get = region.take("GET_ITER")
        region.finish()
        store = ports["store"]
        if store["opcode"] not in {
            "STORE_NAME",
            "STORE_GLOBAL",
        } or not source_frames.matches_position(
            store, node["target"], code["source_key"]
        ):
            raise ValueError("source iterator item has no matching mapping store")
        target = region.mapping_target(node["target"], store)
        advance = ports["next"]
        if (
            advance["opcode"] != "FOR_ITER"
            or ports["backedge"]["target"] != advance["offset"]
            or advance["target"] != ports["cleanup"][0]["offset"]
        ):
            raise ValueError(
                "source iterator edges disagree with the verified header or exit"
            )
    except (ValueError, IndexError, KeyError, TypeError) as error:
        frame.fail(node, "source_iteration_operands", str(error))

    _register_expression(frame, region.steps)
    iterator = {"from_offset": get["offset"]}
    item = {"from_offset": advance["offset"]}
    action = _namespace_operation(
        frame,
        node,
        "module_object_for",
        source_code=code,
        source_expression=receipt,
        reads=("frame.code", "frame.locals", "python.iterator"),
        writes=(target["namespace"],),
        evaluation=region.steps,
        acquire=dict(
            instruction=get,
            iterable=iterable,
            result=iterator,
            dispatch="apply_source_GET_ITER_once_and_release_consumed_iterable",
        ),
        advance=dict(
            instruction=advance,
            iterator=iterator,
            item=item,
            dispatch="apply_source_FOR_ITER_to_retained_iterator_on_each_header_visit",
            on_item="consume_this_visits_owned_item_at_mapping_store_before_body",
            on_exhaustion="release_iterator_once_then_enter_else",
            on_other_exception="unwind_and_propagate_without_store_body_or_else",
        ),
        target=dict(target, value=item),
        edges=dict(
            body=ports["body"],
            backedge=ports["backedge"],
            cleanup=ports["cleanup"],
            exhausted=ports["exhausted"],
        ),
        body=[],
        **{"else": []},
        code_match="require_active_code_and_iteration_operands_and_edges_before_first_operand",
        on_code_mismatch="resume_source_equivalent_execution_before_first_iterable_operand",
        ownership="retain_source_stack_per_visit_and_release_at_source_edges_including_store_replacement_and_reentry",
        unwind="propagate_failures_of_preparation_store_or_body_with_completed_effects_and_source_stack_unwind",
        tracing="preserve_source_control_events_including_version_specific_exhaustion_cleanup",
    )
    # Keep a failed body or else normalization attached to its actual control region.
    steps.append(action)
    observe_handler_effect(frame, action)
    zero_visits = frame.snapshot()
    allocation = frame.project.array_state.enter_loop(frame, node)
    frame.widen_loop(node, None, None)
    # The body may follow earlier visits. A publication overwritten on one of
    # them cannot serve as unconditional historical evidence for a later read.
    publications = frame.project.imports.publications
    publications.pop(target["slot"], None)
    for part in walk_operations(node["body"]):
        for dest in written_targets(part):
            if dest.get("op") == "var" and dest.get("scope") == "global":
                publications.pop(dest["name"], None)
    frame.dynamic += 1
    frame.enter_control("loop", node)
    try:
        falls = frame.sequence(node["body"], action["body"])
    finally:
        frame.leave_control()
        frame.dynamic -= 1
        frame.project.array_state.leave_loop(allocation)
    exits = [zero_visits]
    if falls:
        exits.append(frame.snapshot())
    frame.merge(exits)
    # The final advance and iterator disposal may reenter on every normal path.
    observe_handler_effect(frame, action)
    return frame.sequence(node.get("else", []), action["else"])


def open_object_expression(frame, node, steps, *, truth_context, discard):
    """Only a truth value or completed discard may leave this source evaluation stack."""
    if not (truth_context or discard) or not (code := _module_code(frame)):
        return None
    conditional = truth_context and node.get("op") == "if"
    receipt = (
        node.get("source_expression")
        if conditional
        or discard
        and not truth_context
        and node.get("op") in {"call", "python_invoke", "library_call"}
        else None
    )
    if receipt is not None:
        try:
            region = _source_region(frame, receipt)
            expression = (
                region.condition(node, receipt["branch"])
                if conditional
                else region.discard(node)
            )
        except (ValueError, IndexError, KeyError, TypeError) as error:
            frame.fail(node, "source_expression_operands", str(error))
    elif truth_context:
        expression = _object_truth(frame, node, code)
    else:
        value = _object_operand(frame, node, code)
        expression = (
            None
            if value is None
            else dict(
                op="source_object_discard",
                type="unit",
                value=value,
                **source_site(node),
            )
        )
    if expression is None:
        return None
    _register_expression(frame, expression)
    return namespace_effect(
        frame,
        node,
        steps,
        "module_object_expression",
        result=expression["type"],
        reads=("frame.code", "frame.locals"),
        source_code=code,
        expression=expression,
        **({"source_expression": receipt} if receipt is not None else {}),
        code_match="qualify_current_frame_code_and_expression_instruction_order_stack_operands_spans_and_jumps_before_first_operand",
        on_code_mismatch="resume_source_equivalent_execution_before_first_operand",
        lifetime="transfer_actual_result_to_parent_and_release_consumed_stack_references_at_each_source_opcode",
        unwind="on_failure_unwind_only_live_source_stack_references_at_failing_opcode_no_duplicate_effects",
        callbacks="preserve_current_frame_tracing_exception_state_reentry_and_unraisable_finalizer_errors",
    )


def _namespace_operation(
    frame, node, opcode, *, result="unit", reads=(), writes=(), **operands
):
    action = dict(
        op=opcode,
        type=result,
        operator_family="module_registry",
        semantics_version=1,
        frame_globals=dict(
            owner=frame.name, module=frame.module, identity="current_activation_globals"
        ),
        effects=["CTRL", "STATE", "IO", "MEM", "OPAQUE"],
        may_raise=True,
        reads=[
            "frame.globals",
            "frame.builtins",
            "module.registry",
            "module.namespace",
            "python.object.state",
            *reads,
        ],
        writes=[
            "module.registry",
            "module.namespace",
            "python.object.state",
            "UNKNOWN",
            *writes,
        ],
        external_effect="UNKNOWN",
        allocates="UNKNOWN",
        frees="UNKNOWN",
        aliases="UNKNOWN",
        escapes="UNKNOWN",
        rng_effect="UNKNOWN",
        exceptional_exit="propagate_at_failing_source_operation_preserving_completed_effects",
        **source_site(node),
        **operands,
    )
    require_registry_operator(frame.project, opcode)
    return action


def _open_namespace_after(frame, action):
    frame.project.names.clear()
    if action["op"] == "module_global_store":
        frame.project.imports.publications.pop(action["slot"], None)
    elif action["op"] == "module_global_delete":
        for target in action["targets"]:
            frame.project.imports.publications.pop(target["slot"], None)
    frame.project.imports.open_namespace = dict(
        owner=frame.name, operation=action["op"], **source_site(action)
    )


def observe_handler_effect(frame, action):
    """Transfer analysis state along an actual arm edge without emitting a second operation."""
    frame.observe_effect(action)
    _open_namespace_after(frame, action)


def namespace_effect(frame, node, steps, opcode, *, result="unit", **operands):
    action = _namespace_operation(frame, node, opcode, result=result, **operands)
    if opcode in {"module_fromlist_publish", "module_import_publish"}:
        row = frame.project.source_imports.get(frame.name, node)
        frame.project.source_imports.producer(row, action, "source_import")
    value = frame.effect(action, result, steps, discard=result == "unit")
    _open_namespace_after(frame, action)
    return value


def open_exception_handler(frame, node, index, count):
    """Build the conditional entry and cleanup operations for one verified arm."""
    code = _module_code(frame)
    if not code or node.get("test") is None:
        return None
    receipt = node.get("source_exception_handler")
    if receipt is None:
        return None
    test = node["test"]
    matcher = _object_operand(frame, test, code)
    if not (
        matcher is not None
        and matcher["op"] == "module_name_load"
        and receipt["code_sha256"] == code["code_sha256"]
        and receipt["load"] == matcher["instruction"]
        and receipt["arm_index"] == index
        and receipt["arm_count"] == count
    ):
        frame.fail(
            test,
            "exception_match_operands",
            "the handler's position or name lookup differs from its source exception dispatch",
        )
    _register_expression(frame, matcher)
    active = {
        "source": "currently_dispatched_exception",
        "identity": "original_exception_object",
        "lifetime": "per_exception_dispatch_in_current_activation",
        "entry_offset": receipt["entry"]["offset"],
    }
    match = _namespace_operation(
        frame,
        node,
        "source_exception_match",
        result="bool",
        reads=("frame.code", "frame.locals", "python.exception_state"),
        writes=("python.exception_state",),
        exception=active,
        matcher=matcher,
        source_code=code,
        source_exception_handler=receipt,
        code_match="qualify_current_activation_code_exception_table_and_current_handler_stack_before_lookup",
        on_code_mismatch="resume_source_exception_dispatch_before_this_matcher_with_original_exception_traceback_and_stack",
        dispatch="source_CHECK_EXC_MATCH_validate_actual_class_or_tuple_then_match_against_original_exception",
        exception_state="enter_dispatch_once_preserve_previous_handled_exception_and_restore_at_source_POP_EXCEPT_or_unwind",
        on_match="run_on_match_then_body_with_source_active_exception",
        on_miss="continue_ordered_dispatch_with_same_exception_and_post_lookup_state_or_reraise",
        on_error="unwind_match_failure_follow_source_exception_table_and_preserve_original_exception_context",
    )
    header = dict(match=match, binding=None, on_match=[], cleanup=[])
    alias = node.get("binding")
    if alias is None:
        return header
    if alias.get("cell"):
        frame.fail(
            alias,
            "handler_binding",
            "exception alias requires its current module mapping",
        )
    store = receipt["matched"][-1]
    mapping = "frame.locals" if store["opcode"] == "STORE_NAME" else "frame.globals"
    common = dict(
        slot=alias["name"],
        name=store["name"],
        key_index=store["index"],
        namespace=mapping,
        arm_index=index,
        source_code_sha256=code["code_sha256"],
        reads=("frame.code", "frame.locals", "python.exception_state"),
        writes=(mapping, "python.exception_state"),
    )
    bind = _namespace_operation(
        frame,
        node,
        "source_exception_bind",
        **common,
        instruction=store,
        value=active,
        publication="source_store_of_original_exception_on_matched_edge_only",
        release="preserve_replaced_binding_release_reentry_and_unraisable_finalizer_errors",
        on_error="follow_source_unwind_from_store_without_entering_body",
    )
    clear = _namespace_operation(
        frame,
        node,
        "source_exception_clear",
        **common,
        normal_exit=receipt["normal_exit"],
        exceptional_cleanup=receipt["exceptional_cleanup"],
        cleanup="follow_exit_specific_source_opcodes_including_None_store_then_delete_with_release_and_reentry",
        on_error="follow_source_exception_table_at_failing_cleanup_instruction_preserving_completed_effects",
    )
    header["on_match"].append(dict(op="expr", value=bind))
    header["cleanup"].append(dict(op="expr", value=clear))
    return header


def _published_functions(project, fragments):
    """Index only MAKE_FUNCTION results consumed by their original module STORE."""
    exports = {}
    for fragment in fragments:
        owner = fragment["symbol"]
        code = fragment["activation_handoff"]["entry"]["code"]
        for store in walk_operations(
            fragment["function"]["body"], exclude_fields=EXECUTION_ANNOTATIONS
        ):
            proof = store.get("source_definition")
            if (
                store["op"] != "module_global_store"
                or not proof
                or proof["code_sha256"] != code["code_sha256"]
                or len(store.get("evaluation", ())) != 2
            ):
                continue
            load, make = store["evaluation"]
            target = store["target"]["instruction"]
            if not (
                load["op"] == "source_object_constant"
                and load["expected_code"] == proof["child"]
                and make["op"] == "source_object_make_function"
                and make["code"] == {"from_offset": load["offset"]}
                and store["value"] == {"from_offset": make["offset"]}
                and target == proof["instructions"][-1]
            ):
                continue
            symbol = make["symbol"]
            child = dict(
                initializer=owner,
                creation_site=make["site"],
                make_offset=make["offset"],
                store_offset=target["offset"],
                symbol=symbol,
                code=proof["child"],
            )
            body = project.function_definitions.get(symbol, {}).get("source_body", {})
            if body.get("code_sha256") == proof["child"]["code_sha256"]:
                child["source_body"] = body
            key = owner, target["name"]
            group = exports.setdefault(key, [])
            if child not in group:
                group.append(child)
    return exports


def _call_import_bindings(op, owners):
    """Yield the qualified source of each published binding, including aliased members."""
    if op["op"] == "module_import_publish":
        port = op.get("source_import_projection") or op.get("source_import_store")
        if not port:
            return
        name = port["import_name"]["name"]
        module = name if "projection" in port else name.partition(".")[0]
        for candidate in op["import_call"].get("body", {}).get("candidates", ()):
            owner = candidate["initializer"]
            if candidate["module"] == module and owners.get(owner) == (
                module,
                candidate["code_sha256"],
            ):
                yield op["binding"], owner, None, port["import_name"]["offset"]
    elif op["op"] == "module_fromlist_publish" and (port := op.get("source_fromlist")):
        # IMPORT_FROM reads this member from the actual returned object. Parent
        # package initializers in the same request are distinct candidates.
        for owner in op["initializers"]:
            if owner in owners and owners[owner][0] == op["resolved_module"]:
                for binding in op["bindings"]:
                    yield binding, owner, binding["member"], port["import_name"][
                        "offset"
                    ]


def link_conditional_function_calls(project, conditional):
    """Connect already evaluated callees to optional function creations in sealed modules."""
    exports = _published_functions(project, conditional)
    if not exports:
        return set()
    owners = {
        f["symbol"]: (
            project.functions[f["symbol"]]["python_scope"]["module"],
            f["activation_handoff"]["entry"]["code"]["code_sha256"],
        )
        for f in conditional
    }
    selected = set()
    scopes = [
        *project.closed.values(),
        *project.prefixes,
        *(f["function"] for f in conditional),
    ]
    for scope in scopes:
        imports, calls = {}, []
        for op in walk_operations(scope["body"], exclude_fields=EXECUTION_ANNOTATIONS):
            if op["op"] in {"module_import_publish", "module_fromlist_publish"}:
                for binding, owner, member, offset in _call_import_bindings(op, owners):
                    key = op["occurrence"], binding["site"], binding["slot"]
                    imports.setdefault(key, []).append(
                        (owner, member, offset, binding, op["frame_globals"]["owner"])
                    )
            calls.extend(source_call_operands(op))
        for call, name, attribute, previous in calls:
            key = tuple(previous.get(k) for k in ("occurrence", "site", "slot"))
            for owner, member, offset, binding, importer in imports.get(key, ()):
                if (attribute is None) != (member is not None):
                    continue
                candidates = exports.get((owner, member or attribute["name"]), ())
                if not candidates:
                    continue
                call["conditional_activation"] = dict(
                    semantics_version=1,
                    origin=dict(
                        import_owner=importer,
                        occurrence=key[0],
                        binding_site=key[1],
                        import_offset=offset,
                        store_offset=binding["offset"],
                        load_site=name["site"],
                        **(
                            {"attribute_site": attribute["site"]}
                            if attribute
                            else {
                                "read_offset": binding["read"]["instruction"]["offset"]
                            }
                        ),
                    ),
                    candidates=list(candidates),
                    when="original_call_directly_selects_exact_function_from_witnessed_successful_creation_store_and_frame_uses_that_same_child_code_object",
                    witness="no_extra_references_callbacks_or_reuse_of_dead_object_identity",
                    inputs=[
                        "selected_function",
                        "active_frame",
                        "actual_code",
                        "actual_globals",
                        "actual_builtins",
                        "bound_arguments",
                    ],
                    on_other="continue_original_call_or_entered_source_frame_without_repeating_lookup_arguments_or_dispatch",
                )
                selected.update(c["symbol"] for c in candidates)
    return selected

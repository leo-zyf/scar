"""M2 operations on a live module namespace, including lookup and object release."""

import re

from scar.contract import (
    MODULE_REGISTRY_OPERATIONS,
    digest,
    json_bytes,
    walk_operations,
)
from .values import literal, source_site

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


def global_fromlist(frame, body, index):
    """Model source-ordered publication in an open module namespace."""
    first = body[index]
    request = first.get("request", {})
    members = request.get("fromlist", ())
    end = index + len(members)
    module = first.get("module")
    if (
        not frame.initializing
        or module not in frame.project.initializers
        or request.get("form") != "from"
        or request.get("binding_index") != 0
        or not request.get("level")
        and request.get("module") != module
        or not members
        or end > len(body)
    ):
        return None
    if not frame.project.open_python_namespace and (
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
    if level:
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
            or node.get("graph_backed") is not True
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
    if not frame.project.open_python_namespace and set(
        global_delete_slots(frame, body[end])
    ) != {b["slot"] for b in bindings}:
        return None
    return occurrence, bindings


def _matches_instruction(instruction, operand, source_key):
    location = operand.get("location")
    position = instruction.get("position")
    fields = ("line", "end_line", "column", "end_column")
    if (
        not isinstance(location, dict)
        or location.get("file") != source_key
        or not all(type(location.get(k)) is int and location[k] >= 1 for k in fields)
        or not isinstance(position, list)
        or not all(type(part) is int for part in position)
    ):
        return False
    line, end_line, column, end_column = (location[k] for k in fields)
    return (
        position == [line, end_line, column - 1, end_column - 1]
        and type(instruction.get("index")) is int
        and instruction["index"] >= 0
        and type(instruction.get("offset")) is int
        and instruction["offset"] >= 0
    )


def _open_module_code(frame):
    code = frame.scope.get("source_code")
    if (
        not frame.project.open_python_namespace
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


def store_open_module_constant(frame, node, steps):
    """Retain the source constant object and store opcode across callback effects."""
    port = node.get("source_constant_store")
    if (
        not isinstance(port, dict)
        or type(port.get("semantics_version")) is not int
        or port["semantics_version"] != 1
        or not (code := _open_module_code(frame))
        or port.get("code") != code
    ):
        return False
    load, store = port.get("load"), port.get("store")
    if not isinstance(load, dict) or not isinstance(store, dict):
        return False
    value, target = node["value"], node["target"]
    if (
        load.get("opcode") != "LOAD_CONST"
        or load.get("type") != "str"
        or value.get("op") != "literal"
        or value.get("type") != {"kind": "python_object", "class": "str"}
        or type(value.get("value")) is not str
        or load.get("value_sha256") != digest(json_bytes(value["value"]))
        or store.get("opcode") not in ("STORE_NAME", "STORE_GLOBAL")
        or not isinstance(store.get("name"), str)
        or not store["name"].isidentifier()
        or target.get("op") != "var"
        or target.get("scope") != "global"
        or target.get("cell")
        or target.get("name") != f"py:{frame.module}:{store['name']}"
        or not _matches_instruction(load, value, code["source_key"])
        or not _matches_instruction(store, target, code["source_key"])
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


def _object_operand(frame, node, code):
    """Build an owned operand tree. No lookup or call is hoisted out of its consumer."""
    if not isinstance(node, dict):
        return None
    op = node.get("op")
    owned = dict(
        type={"kind": "handle", "resource": "python_object"}, **source_site(node)
    )
    if op in {"var", "python_builtin"}:
        load = node.get("source_name_load")
        if (
            not isinstance(load, dict)
            or type(load.get("semantics_version")) is not int
            or load["semantics_version"] != 1
            or load.get("code_sha256") != code["code_sha256"]
            or not _matches_instruction(load, node, code["source_key"])
            or load.get("opcode") not in ("LOAD_NAME", "LOAD_GLOBAL")
            or type(load.get("oparg")) is not int
            or load["index"]
            != (
                load["oparg"] >> 1 if load["opcode"] == "LOAD_GLOBAL" else load["oparg"]
            )
            or not isinstance(load.get("name"), str)
            or not load["name"].isidentifier()
        ):
            return None
        slot = f"py:{frame.module}:{load['name']}"
        if op == "var":
            if (
                node.get("scope") != "global"
                or node.get("cell")
                or node.get("name") != slot
            ):
                return None
        elif node.get("name") != load["name"]:
            return None
        scopes = ["frame.globals", "frame.builtins"]
        if load["opcode"] == "LOAD_NAME":
            scopes.insert(0, "frame.locals")
        previous = frame.project.fromlist_publications.get(slot)
        return dict(
            op="module_name_load",
            instruction=load,
            lookup_order=scopes,
            lookup="source_mapping_lookup_propagate_nonmissing_errors_or_NameError",
            **(
                {"source_evidence": {"prior_publication": previous}} if previous else {}
            ),
            **owned,
        )
    if (
        op in {"call", "python_invoke", "library_call"}
        and isinstance(node.get("args"), list)
        and node.get("keywords", []) == []
    ):
        operands = [
            _object_operand(frame, child, code)
            for child in (node.get("target"), *node["args"])
        ]
        if all(item is not None for item in operands):
            return dict(
                op="source_object_call",
                target=operands[0],
                args=operands[1:],
                evaluation="retain_target_then_each_positional_argument_left_to_right",
                dispatch="invoke_actual_callable_with_retained_argument_objects",
                **owned,
            )
    if (
        op == "compare_chain"
        and len(node.get("operators", ())) == len(node.get("operands", ())) == 1
    ):
        operator = node["operators"][0]
        if operator in {"lt", "lte", "gt", "gte", "eq", "noteq"}:
            left = _object_operand(frame, node["left"], code)
            right = _object_operand(frame, node["operands"][0], code)
            if left is not None and right is not None:
                return dict(
                    op="source_object_compare",
                    operator=operator,
                    left=left,
                    right=right,
                    evaluation="retain_left_then_right",
                    dispatch="rich_comparison_including_reflected_dispatch_and_NotImplemented_fallback",
                    **owned,
                )
    return None


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


def open_object_expression(frame, node, steps, *, truth_context, discard):
    """Only a truth value or completed discard may leave this source evaluation stack."""
    if not (truth_context or discard) or not (code := _open_module_code(frame)):
        return None
    if truth_context:
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
    for item in walk_operations(expression, exclude_fields={"source_evidence"}):
        if item["op"] in MODULE_REGISTRY_OPERATIONS:
            require_registry_operator(frame.project, item["op"])
        if item.get("site") is not None:
            frame.project.normalized_sites.add((frame.name, item["site"]))
            frame.project.live_sites.add((frame.name, item["site"]))
    return namespace_effect(
        frame,
        node,
        steps,
        "module_object_expression",
        result=expression["type"],
        reads=("frame.code", "frame.locals"),
        source_code=code,
        expression=expression,
        code_match="qualify_current_frame_code_and_expression_instruction_order_stack_operands_spans_and_jumps_before_first_operand",
        on_code_mismatch="resume_source_equivalent_execution_before_first_operand",
        lifetime="transfer_actual_result_to_parent_and_release_consumed_stack_references_at_each_source_opcode",
        unwind="on_failure_unwind_only_live_source_stack_references_at_failing_opcode_no_duplicate_effects",
        callbacks="preserve_current_frame_tracing_exception_state_reentry_and_unraisable_finalizer_errors",
    )


def namespace_effect(
    frame, node, steps, opcode, *, result="unit", reads=(), writes=(), **operands
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
    if opcode == "module_fromlist_publish":
        row = frame.project.source_imports.get(frame.name, node)
        frame.project.source_imports.producer(row, action, "source_import")
    value = frame.effect(action, result, steps, discard=result == "unit")
    # Import callbacks and releasing arbitrary objects can reenter Python and
    # replace bindings, including names that never appeared in the source.
    frame.project.names.clear()
    frame.project.open_python_namespace = dict(
        owner=frame.name, operation=opcode, **source_site(node)
    )
    return value

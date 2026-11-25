"""M2 operations on a live module namespace, including lookup and object release."""

import re

from scar.contract import (
    MODULE_REGISTRY_OPERATIONS,
    digest,
    json_bytes,
    python_constant_rhs_sha256,
    python_literal_sequence_sha256,
    walk_operations,
)
from scar.source_contracts import python_binary_operator
from . import objects
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


def _matches_position(instruction, operand, source_key):
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
    return position == [line, end_line, column - 1, end_column - 1]


def _matches_instruction(instruction, operand, source_key):
    return (
        _matches_position(instruction, operand, source_key)
        and type(instruction.get("index")) is int
        and instruction["index"] >= 0
        and type(instruction.get("offset")) is int
        and instruction["offset"] >= 0
    )


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
        previous = frame.project.imports.publications.get(slot)
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


class _SourceExpressionRegion:
    """Consume the graph operands and the sealed instructions in lockstep."""

    def __init__(self, frame, code, receipt):
        if receipt["code_sha256"] != code["code_sha256"]:
            raise ValueError("source expression belongs to another code object")
        self.frame = frame
        self.code = code
        self.instructions = receipt["instructions"]
        self.index = 0
        self.steps = []
        version = re.match(r"cpython 3\.(12|13|14)\.", code["compiler"])
        if version is None:
            raise ValueError("unsupported source call stack convention")
        self.minor = int(version[1])

    def next_opcode(self):
        if self.index < len(self.instructions):
            return self.instructions[self.index]["opcode"]

    def take(self, *opcodes, source=None):
        if self.next_opcode() not in opcodes:
            raise ValueError(
                f"expected {' or '.join(opcodes)} at instruction {self.index}"
            )
        instruction = self.instructions[self.index]
        if source is not None and not _matches_position(
            instruction, source, self.code["source_key"]
        ):
            raise ValueError(
                f"source position disagrees with {instruction['opcode']} operand"
            )
        self.index += 1
        return instruction

    def emit(self, instruction, op, node, **fields):
        offset = instruction["offset"]
        self.steps.append(dict(op=op, offset=offset, **source_site(node), **fields))
        return {"from_offset": offset}

    def name(self, node, callee):
        instruction = self.take("LOAD_NAME", "LOAD_GLOBAL")
        load = _object_operand(self.frame, node, self.code)
        if load is None or any(
            instruction.get(key) != load["instruction"].get(key)
            for key in ("opcode", "oparg", "offset", "name", "index", "position")
        ):
            raise ValueError("name operand disagrees with source lookup")
        pair = instruction["opcode"] == "LOAD_GLOBAL" and instruction["oparg"] & 1
        if pair and not callee:
            raise ValueError("unexpected callable stack slot in name lookup")
        self.steps.append(dict(load, offset=instruction["offset"]))
        return {"from_offset": instruction["offset"]}, (
            "callable_with_implicit_NULL" if pair else None
        )

    def constant(self, node):
        ins = self.take("LOAD_CONST", "LOAD_SMALL_INT", source=node)
        typ = node.get("type")
        kind = typ.get("class") if isinstance(typ, dict) else None
        if (
            kind not in {"str", "int", "float", "bool", "NoneType"}
            or kind != ins.get("type")
            or "value" not in node
            or digest(json_bytes(node["value"])) != ins.get("value_sha256")
            or ins["opcode"] == "LOAD_SMALL_INT"
            and (self.minor < 14 or kind != "int" or node["value"] != ins["oparg"])
        ):
            raise ValueError("literal disagrees with source constant")
        return self.emit(
            ins,
            "source_object_constant",
            node,
            source=(
                "current_frame_co_consts"
                if ins["opcode"] == "LOAD_CONST"
                else "source_small_integer"
            ),
            operand=ins["index"] if ins["opcode"] == "LOAD_CONST" else ins["oparg"],
        )

    def tuple_constant(self, node, items):
        ins = self.take("LOAD_CONST", source=node)
        fingerprint = python_literal_sequence_sha256(items)
        if (
            not fingerprint
            or ins.get("type") != "tuple"
            or ins.get("item_count") != len(items)
            or ins.get("items_sha256") != fingerprint
        ):
            raise ValueError(
                "constant tuple slot disagrees with ordered display elements"
            )
        return self.emit(
            ins,
            "source_object_constant",
            node,
            source="current_frame_co_consts",
            operand=ins["index"],
        )

    def build_sequence(self, node, opcode, values):
        ins = self.take(opcode, source=node)
        if ins["oparg"] != len(values):
            raise ValueError("sequence build arity disagrees with source display")
        return self.emit(
            ins,
            "source_object_build_sequence",
            node,
            opcode=opcode,
            items=values,
            may_raise=True,
            allocation="source_build_after_evaluation_of_owned_elements",
        )

    def list_update(self, node, opcode, target, value):
        ins = self.take(opcode, source=node)
        if ins["oparg"] != 1:
            raise ValueError("list update does not target its display's retained list")
        self.emit(
            ins,
            "source_object_list_update",
            node,
            opcode=opcode,
            target=target,
            value=value,
            may_raise=True,
            mutation="update_same_list_before_next_source_instruction",
        )

    def sequence(self, node):
        kind, items = node["kind"], node.get("items")
        if kind not in {"list", "tuple"} or not isinstance(items, list):
            raise ValueError("expected an ordered list or tuple display")
        first = self.instructions[self.index]
        at_display = _matches_position(first, node, self.code["source_key"])
        if kind == "tuple" and at_display and first["opcode"] == "LOAD_CONST":
            return self.tuple_constant(node, items)
        if at_display and first["opcode"] == "BUILD_LIST" and first["oparg"] == 0:
            target = self.build_sequence(node, "BUILD_LIST", [])
            # Constant-only lists extend an already allocated distinct list
            # with the actual elements of an exact tuple in the active code.
            if (
                items
                and self.next_opcode() == "LOAD_CONST"
                and self.instructions[self.index].get("type") == "tuple"
            ):
                immutable = self.tuple_constant(node, items)
                self.list_update(node, "LIST_EXTEND", target, immutable)
            else:
                for item in items:
                    self.list_update(node, "LIST_APPEND", target, self.value(item))
            if kind == "list":
                return target
            ins = self.take("CALL_INTRINSIC_1", source=node)
            if (
                ins.get("intrinsic") != "INTRINSIC_LIST_TO_TUPLE"
                or ins.get("oparg") != 6
            ):
                raise ValueError("tuple display has no source list-to-tuple conversion")
            return self.emit(
                ins,
                "source_object_build_sequence",
                node,
                opcode="CALL_INTRINSIC_1",
                intrinsic=ins["intrinsic"],
                source_list=target,
                may_raise=True,
            )
        values = [self.value(item) for item in items]
        return self.build_sequence(
            node, "BUILD_LIST" if kind == "list" else "BUILD_TUPLE", values
        )

    def operand(self, node, *, callee=False):
        if not isinstance(node, dict):
            raise ValueError("missing source expression operand")
        op = node.get("op")
        if op in {"var", "python_builtin"}:
            return self.name(node, callee)
        if op == "member":
            base = self.value(node.get("base"))
            ins = self.take("LOAD_ATTR", source=node)
            method = bool(ins["oparg"] & 1)
            if ins["name"] != node.get("field") or method and not callee:
                raise ValueError(
                    "attribute name or call operand disagrees with source lookup"
                )
            value = self.emit(
                ins,
                "source_object_attribute",
                node,
                base=base,
                name_index=ins["index"],
                name=ins["name"],
                result=(
                    "callable_and_receiver_or_NULL" if method else "attribute_object"
                ),
                lookup="source_LOAD_ATTR_including_descriptor_and_receiver_stack_convention",
            )
            return value, "source_method_pair" if method else None
        if op in {"call", "python_invoke", "library_call"}:
            return self.call(node), None
        if op == "literal":
            return self.constant(node), None
        if op == "aggregate":
            return self.sequence(node), None
        if op == "binary":
            left = self.value(node.get("left"))
            right = self.value(node.get("right"))
            ins = self.take("BINARY_OP", source=node)
            source_op = python_binary_operator(ins["oparg"])
            operator = node.get("operator")
            if (
                source_op is None
                or source_op[0] != operator
                or ins.get("operator") != operator
            ):
                raise ValueError(
                    "binary instruction disagrees with the source operation"
                )
            return (
                self.emit(
                    ins,
                    "source_object_binary",
                    node,
                    oparg=ins["oparg"],
                    operator=operator,
                    left=left,
                    right=right,
                    may_raise=True,
                    dispatch="source_binary_slots_with_subtype_and_reflected_priority_NotImplemented_and_operator_specific_fallback",
                    ownership="retain_left_through_right_evaluation_then_consume_operands_and_transfer_actual_result",
                ),
                None,
            )
        if op == "format_field":
            return self.field(node), None
        if op == "format_string":
            if not all(
                isinstance(child, dict)
                and (
                    child.get("op") == "format_field"
                    or child.get("op") == "literal"
                    and child.get("type") == {"kind": "python_object", "class": "str"}
                )
                for child in node["items"]
            ):
                raise ValueError(
                    "joined string requires string constants and formatted fields"
                )
            items = [self.value(child) for child in node["items"]]
            if len(items) == 1:
                return items[0], None
            ins = self.take("BUILD_STRING", source=node)
            if ins["oparg"] != len(items):
                raise ValueError(
                    "formatted string item count disagrees with BUILD_STRING"
                )
            return (
                self.emit(ins, "source_object_build_string", node, items=items),
                None,
            )
        raise ValueError(f"unsupported source operand {op}")

    def value(self, node):
        return self.operand(node)[0]

    def field(self, node):
        value = self.value(node.get("value"))
        conversion = {-1: 0, ord("s"): 1, ord("r"): 2, ord("a"): 3}.get(
            node.get("conversion")
        )
        if conversion is None:
            raise ValueError("unidentified format conversion")
        if conversion and self.minor >= 13:
            ins = self.take("CONVERT_VALUE", source=node)
            if ins["oparg"] != conversion:
                raise ValueError("format conversion disagrees with source operand")
            value = self.emit(
                ins, "source_object_convert", node, value=value, conversion=conversion
            )
        spec = self.value(node["format"]) if node.get("format") is not None else None
        if self.minor == 12:
            ins = self.take("FORMAT_VALUE", source=node)
            if ins["oparg"] != conversion | (4 if spec is not None else 0):
                raise ValueError("FORMAT_VALUE flags disagree with the source field")
        else:
            ins = self.take(
                "FORMAT_SIMPLE" if spec is None else "FORMAT_WITH_SPEC", source=node
            )
            conversion = 0
        return self.emit(
            ins,
            "source_object_format",
            node,
            value=value,
            spec=spec,
            conversion=conversion,
            dispatch="source_format_opcode_with_actual_value_and_specification",
        )

    def call(self, node):
        positional, keywords = node.get("args"), node.get("keywords", [])
        if not isinstance(positional, list) or not isinstance(keywords, list):
            raise ValueError("missing ordered call operands")
        explicit_null = self.minor == 12 and self.next_opcode() == "PUSH_NULL"
        if explicit_null:
            self.take("PUSH_NULL")
        target, pair = self.operand(node.get("target"), callee=True)
        if self.minor >= 13 and not pair:
            self.take("PUSH_NULL")
        elif self.minor == 12 and bool(pair) == explicit_null:
            raise ValueError(
                "source call requires one callable and receiver or NULL pair"
            )
        args = [self.value(arg) for arg in positional]
        names, values = [], []
        for kw in keywords:
            if (
                not isinstance(kw, dict)
                or not isinstance(kw.get("name"), str)
                or kw["name"] in names
            ):
                raise ValueError("call requires unique explicit keyword names")
            names.append(kw["name"])
            values.append(self.value(kw.get("value")))
        kwref = None
        if names:
            ins = self.take("KW_NAMES" if self.minor == 12 else "LOAD_CONST")
            if ins.get("type") != "tuple" or ins.get("names") != names:
                raise ValueError("keyword order disagrees with the code constant tuple")
            kwref = self.emit(
                ins,
                "source_object_constant",
                node,
                source="current_frame_co_consts",
                operand=ins["index"],
            )
        ins = self.take(
            "CALL_KW" if names and self.minor >= 13 else "CALL", source=node
        )
        if ins["oparg"] != len(args) + len(names):
            raise ValueError("argument count disagrees with source call")
        return self.emit(
            ins,
            "source_object_call",
            node,
            target=target,
            target_layout=pair or "callable_with_explicit_NULL",
            args=args,
            keyword_names=kwref,
            keyword_values=values,
            dispatch="consume_source_call_slots_including_method_receiver_at_most_once",
        )

    def discard(self, node):
        value = self.call(node)
        ins = self.take("POP_TOP", source=node)
        self.emit(ins, "source_object_discard", node, value=value)
        self.finish()
        return dict(
            op="source_object_region",
            type="unit",
            steps=self.steps,
            ownership="consume_each_input_at_its_source_opcode_and_unwind_remaining_stack_on_failure",
            tracing="preserve_all_source_instruction_events_including_call_stack_setup",
            **source_site(node),
        )

    def condition(self, node, ports):
        condition = node["condition"]
        negated = condition["op"] == "unary" and condition.get("operator") == "not"
        operand = condition["value"] if negated else condition
        if (
            operand["op"] not in {"call", "python_invoke", "library_call"}
            or ports.get("negated") is not negated
            or not _matches_position(
                {"position": ports.get("condition_position")},
                condition,
                self.code["source_key"],
            )
            or not _matches_position(
                {"position": ports.get("operand_position")},
                operand,
                self.code["source_key"],
            )
        ):
            raise ValueError(
                "condition and branch operands disagree with the source graph"
            )
        value = self.value(operand)
        conversion = self.take("TO_BOOL", source=operand) if self.minor >= 13 else None
        jump = self.take(
            "POP_JUMP_IF_TRUE" if negated else "POP_JUMP_IF_FALSE", source=operand
        )
        self.finish()
        if jump["target"] != ports["else"]["offset"]:
            raise ValueError("conditional jump does not enter the sealed alternative")
        truth = self.emit(
            conversion or jump,
            "source_object_truth",
            operand,
            type="bool",
            value=value,
            opcode=(conversion or jump)["opcode"],
            dispatch="consume_actual_call_result_at_source_truth_instruction_including_bool_or_len_and_release",
        )
        return dict(
            op="source_object_region",
            type="bool",
            steps=self.steps,
            # The surrounding structured if consumes this exact boolean. The
            # source conditional jumps to the alternative when it does not match.
            result=dict(value=truth, equals=not negated),
            branch=dict(
                instruction=jump,
                input=truth,
                taken=ports["else"],
                fallthrough=ports["then"],
                fallthrough_marker=ports["fallthrough_marker"],
            ),
            ownership="source_truth_consumes_owned_result_and_only_exact_boolean_reaches_the_two_edges",
            tracing="preserve_jump_event_and_run_fallthrough_marker_only_on_fallthrough_before_suite",
            **source_site(condition),
        )

    def finish(self):
        if self.index != len(self.instructions):
            raise ValueError(
                "unexpected instruction after source expression terminator"
            )

    def target(self, target, *opcodes):
        return self.mapping_target(target, self.take(*opcodes, source=target))

    def mapping_target(self, target, instruction):
        name = instruction["name"]
        if (
            target.get("op") != "var"
            or target.get("scope") != "global"
            or target.get("cell")
            or target.get("name") != f"py:{self.frame.module}:{name}"
        ):
            raise ValueError("source mapping target disagrees with graph binding")
        namespace = (
            "frame.locals"
            if instruction["opcode"].endswith("_NAME")
            else "frame.globals"
        )
        return dict(
            slot=target["name"],
            name=name,
            namespace=namespace,
            instruction=instruction,
            **source_site(target),
        )

    def function(self, node):
        symbol = node["symbol"]
        project = self.frame.project
        definition = project.functions.get(symbol) or project.function_definitions.get(
            symbol, {}
        )
        scope = definition.get("python_scope", {})
        child = node["source_definition"]["child"]
        if (
            not objects.plain_definition(node)
            or definition.get("name") != symbol
            or definition.get("owner_class")
            or definition.get("call_protocol", {}).get("kind") != "regular"
            or scope.get("kind") != "function"
            or scope.get("initializer") != self.frame.name
            or scope.get("module") != self.frame.module
            or scope.get("source_key") != self.code["source_key"]
            or scope.get("definition_code") != child
            or not isinstance(child, dict)
            or child.get("qualname") != child.get("name")
        ):
            raise ValueError(
                "function creation requires the sealed ordinary definition in this module"
            )
        load = self.take("LOAD_CONST", source=node)
        code = self.emit(
            load,
            "source_object_constant",
            node,
            source="current_frame_co_consts",
            operand=load["index"],
            expected_code=child,
        )
        make = self.take("MAKE_FUNCTION", source=node)
        if (
            self.minor == 12
            and make.get("oparg") != 0
            or self.minor >= 13
            and "oparg" in make
        ):
            raise ValueError(
                "function creation requires the source MAKE_FUNCTION without optional attributes"
            )
        return self.emit(
            make,
            "source_object_make_function",
            node,
            code=code,
            symbol=symbol,
            globals="current_frame_globals",
            builtins=dict(
                lookup="source_lookup_of___builtins___in_current_globals_at_function_creation",
                absent="current_frame_builtins",
                module="module_dict",
                other="actual_object",
            ),
            allocation="source_MAKE_FUNCTION_creates_distinct_function_with_source_metadata_each_execution",
            may_raise=True,
        )

    def statement(self, node):
        if node["op"] == "raise":
            exception = self.value(node["value"]) if node["has_exception"] else None
            cause = self.value(node["cause"]) if node["has_cause"] else None
            instruction = self.take("RAISE_VARARGS", source=node)
            if instruction["oparg"] != int(node["has_exception"]) + int(
                node["has_cause"]
            ):
                raise ValueError(
                    "RAISE_VARARGS operand count disagrees with source raise"
                )
            opcode = "source_exception_raise"
            ports = dict(
                exception=exception,
                cause=cause,
                instruction=instruction,
                noreturn=True,
                writes=["python.exception_state"],
                dispatch="source_RAISE_VARARGS_normalize_actual_exception_then_explicit_cause_using_source_class_instance_None_and_invalid_operand_rules",
                reraise="zero_operands_reraises_actual_handled_exception_with_existing_traceback_or_raises_RuntimeError_if_absent",
                exception_state="preserve_traceback_cause_context_suppression_and_handled_exception_at_source_transfer",
            )
        elif node["op"] in {"assign", "python_define"}:
            value = (
                self.function(node)
                if node["op"] == "python_define"
                else self.value(node["value"])
            )
            target = self.target(node["target"], "STORE_NAME", "STORE_GLOBAL")
            if (
                node["op"] == "python_define"
                and node["source_definition"]["child"]["name"] != target["name"]
            ):
                raise ValueError("the source function name and store target differ")
            opcode = "module_global_store"
            ports = dict(
                slot=target["slot"],
                name=target["name"],
                namespace=target["namespace"],
                target=target,
                value=value,
                store_opcode=target["instruction"]["opcode"],
                writes=[target["namespace"]],
                publication="consume_actual_rhs_object_with_source_store_and_current_co_names_key",
            )
        elif node["op"] == "unbind" and node["targets"]:
            targets = [
                self.target(t, "DELETE_NAME", "DELETE_GLOBAL") for t in node["targets"]
            ]
            opcode = "module_global_delete"
            ports = dict(
                targets=targets,
                names=[t["name"] for t in targets],
                writes=list(dict.fromkeys(t["namespace"] for t in targets)),
                deletion="execute_each_sealed_mapping_delete_and_release_in_source_order",
                missing="propagate_source_delete_opcode_failure_after_completed_prefix",
            )
        else:
            raise ValueError("unrecognized source statement termination")
        self.finish()
        return opcode, ports


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
        region = _SourceExpressionRegion(frame, code, receipt)
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
        region = _SourceExpressionRegion(frame, code, receipt)
        iterable = region.value(node["range"])
        get = region.take("GET_ITER")
        region.finish()
        store = ports["store"]
        if store["opcode"] not in {
            "STORE_NAME",
            "STORE_GLOBAL",
        } or not _matches_position(store, node["target"], code["source_key"]):
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
            region = _SourceExpressionRegion(frame, code, receipt)
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

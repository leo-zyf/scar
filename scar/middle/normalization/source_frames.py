"""Decode bound function-frame and expression ports into ordered object operations."""

import re

from scar.contract import digest, json_bytes, python_literal_sequence_sha256
from scar.source_contracts import python_binary_operator
from . import objects
from .values import source_site


def matches_position(instruction, operand, source_key):
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


def matches_instruction(instruction, operand, source_key):
    return (
        matches_position(instruction, operand, source_key)
        and type(instruction.get("index")) is int
        and instruction["index"] >= 0
        and type(instruction.get("offset")) is int
        and instruction["offset"] >= 0
    )


def _name_operand(module, publications, node, code):
    load = node.get("source_name_load")
    if (
        not isinstance(load, dict)
        or type(load.get("semantics_version")) is not int
        or load["semantics_version"] != 1
        or load.get("code_sha256") != code["code_sha256"]
        or not matches_instruction(load, node, code["source_key"])
        or load.get("opcode") not in ("LOAD_NAME", "LOAD_GLOBAL")
        or type(load.get("oparg")) is not int
        or load["index"]
        != (load["oparg"] >> 1 if load["opcode"] == "LOAD_GLOBAL" else load["oparg"])
        or not isinstance(load.get("name"), str)
        or not load["name"].isidentifier()
    ):
        return None
    slot = f"py:{module}:{load['name']}"
    if node["op"] == "var":
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
    previous = publications.get(slot)
    return dict(
        op="module_name_load",
        instruction=load,
        lookup_order=scopes,
        lookup="source_mapping_lookup_propagate_nonmissing_errors_or_NameError",
        **({"source_evidence": {"prior_publication": previous}} if previous else {}),
        type={"kind": "handle", "resource": "python_object"},
        **source_site(node),
    )


def object_operand(module, publications, node, code):
    """Build an owned operand tree. No lookup or call is hoisted out of its consumer."""
    if not isinstance(node, dict):
        return None
    op = node.get("op")
    owned = dict(
        type={"kind": "handle", "resource": "python_object"}, **source_site(node)
    )
    if op in {"var", "python_builtin"}:
        return _name_operand(module, publications, node, code)
    if (
        op in {"call", "python_invoke", "library_call"}
        and isinstance(node.get("args"), list)
        and node.get("keywords", []) == []
    ):
        operands = [
            object_operand(module, publications, child, code)
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
            left = object_operand(module, publications, node["left"], code)
            right = object_operand(module, publications, node["operands"][0], code)
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


class SourceExpressionRegion:
    """Consume the graph operands and the sealed instructions in lockstep."""

    def __init__(self, scope, receipt, *, publications=None, definitions=None):
        code = scope["source_code"]
        if receipt["code_sha256"] != code["code_sha256"]:
            raise ValueError("source expression belongs to another code object")
        self.scope = scope
        self.publications = {} if publications is None else publications
        self.definitions = definitions
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
        if source is not None and not matches_position(
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
        if self.next_opcode() in {"LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_BORROW"}:
            ins = self.take(self.next_opcode(), source=node)
            local = self.fast_local(node, ins)
            load = node.get("source_name_load", {})
            if (
                load.get("semantics_version") != 1
                or load.get("code_sha256") != self.code["code_sha256"]
                or any(
                    load.get(field) != ins.get(field)
                    for field in (
                        "opcode",
                        "name",
                        "index",
                        "oparg",
                        "offset",
                        "position",
                    )
                )
            ):
                raise ValueError("local operand disagrees with its source load")
            return (
                self.emit(
                    ins,
                    "frame_local_load",
                    node,
                    **local,
                    opcode=ins["opcode"],
                    ownership=(
                        "borrowed" if ins["opcode"] == "LOAD_FAST_BORROW" else "owned"
                    ),
                    unbound=(
                        "raise_UnboundLocalError"
                        if ins["opcode"] == "LOAD_FAST_CHECK"
                        else "source_initialized_slot"
                    ),
                ),
                None,
            )
        instruction = self.take("LOAD_NAME", "LOAD_GLOBAL")
        load = _name_operand(self.scope["module"], self.publications, node, self.code)
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

    def fast_local(self, node, instruction):
        names = self.scope.get("definition_code", {}).get("varnames", ())
        slot = instruction.get("index")
        if not (
            self.scope.get("kind") == "function"
            and node.get("op") == "var"
            and node.get("scope") in {None, "local"}
            and not node.get("cell")
            and type(slot) is int
            and 0 <= slot < len(names)
            and instruction.get("oparg") == slot
            and instruction.get("name") == names[slot] == node.get("name")
        ):
            raise ValueError("local operand disagrees with its function frame slot")
        return dict(slot=slot, name=names[slot])

    def constant(self, node):
        ins = self.take("LOAD_CONST", "LOAD_SMALL_INT", source=node)
        self.check_literal(node, ins)
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

    def check_literal(self, node, ins):
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

    def return_value(self, node):
        if self.next_opcode() == "RETURN_CONST":
            ins = self.take("RETURN_CONST")
            self.check_literal(node["value"], ins)
            value = dict(source="current_frame_co_consts", index=ins["index"])
        else:
            value = self.value(node["value"])
            ins = self.take("RETURN_VALUE")
        self.emit(
            ins,
            "frame_return",
            node,
            value=value,
            opcode=ins["opcode"],
            destination="actual_caller",
            lifetime="transfer_result_and_release_frame_locals_at_source_exit",
        )
        self.finish()

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
        at_display = matches_position(first, node, self.code["source_key"])
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
            # In 3.12 a NULL below the base survives successive ordinary
            # LOAD_ATTRs. A LOAD_ATTR that produces a method pair owns both slots.
            base, prefix = self.operand(
                node.get("base"), callee=callee and self.minor == 12
            )
            ins = self.take("LOAD_ATTR", source=node)
            method = bool(ins["oparg"] & 1)
            if (
                ins["name"] != node.get("field")
                or method
                and (not callee or prefix)
                or prefix not in (None, "callable_with_implicit_NULL")
            ):
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
            return value, "source_method_pair" if method else prefix
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
            or not matches_position(
                {"position": ports.get("condition_position")},
                condition,
                self.code["source_key"],
            )
            or not matches_position(
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
            or target.get("name") != f"py:{self.scope['module']}:{name}"
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
        definition = self.definitions.get(symbol, {})
        scope = definition.get("python_scope", {})
        child = node["source_definition"]["child"]
        if (
            not objects.plain_definition(node)
            or definition.get("name") != symbol
            or definition.get("owner_class")
            or definition.get("call_protocol", {}).get("kind") != "regular"
            or scope.get("kind") != "function"
            or scope.get("initializer") != self.scope["initializer"]
            or scope.get("module") != self.scope["module"]
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


def conditional_function_body(definition, descriptor):
    """Read a matched callee from the bound graph, starting after argument binding."""
    scope = definition["python_scope"]
    symbol = definition["name"]
    code = scope.get("source_code", {})
    child = scope.get("definition_code", {})
    body_id = descriptor.get("source_body", {})
    regions = []
    result = dict(
        symbol=symbol,
        completion="conditional_source_prefix",
        function=dict(symbol=symbol, body=regions),
    )

    def stop(node, reason):
        result["frontier"] = dict(
            operator=node.get("op", "frame_entry"),
            requirement=reason,
            **source_site(node),
        )
        for role in ("value", "condition", "range"):
            operand = node.get(role)
            if isinstance(operand, dict) and "op" in operand:
                result["frontier"][role] = dict(
                    operator=operand["op"], **source_site(operand)
                )
        return result

    if not (
        definition.get("call_protocol", {}).get("kind") == "regular"
        and scope.get("kind") == code.get("kind") == "function"
        and body_id.get("graph_id")
        and code.get("code_sha256")
        and code["code_sha256"]
        == child.get("code_sha256")
        == body_id.get("code_sha256")
        and code.get("source_key") == scope.get("source_key")
        and isinstance(child.get("varnames"), list)
        and isinstance(definition.get("body"), list)
    ):
        return stop(definition, "bound_function_body_and_matching_frame_layout")
    slots = {name: i for i, name in enumerate(child["varnames"])}
    params = definition["params"]
    if [[p["binding"], p["name"]] for p in params] != child.get("parameters") or any(
        p["name"] not in slots for p in params
    ):
        return stop(definition, "formal_parameter_to_compiled_frame_slot")
    result["entry"] = dict(
        graph=body_id,
        code=code,
        timing="after_actual_dispatch_and_successful_source_argument_binding",
        inputs=["selected_function", "active_frame", "frame.globals", "frame.builtins"],
        parameters=[
            dict(name=p["name"], binding=p["binding"], slot=slots[p["name"]])
            for p in params
        ],
        state="read_bound_arguments_from_this_activation_without_rebinding_or_reloading_caller_operands",
        exceptional_exit="propagate_at_source_instruction_with_its_live_stack_and_completed_effects",
    )
    for node in definition["body"]:
        if (
            node["op"] == "decl"
            and node.get("initial_binding") == "unbound"
            and "init" not in node
        ):
            continue
        expr = node.get("value", {}) if node["op"] == "expr" else node
        receipt = expr.get("source_expression")
        if not receipt:
            return stop(node, "instruction_region_for_current_function_statement")
        try:
            region = SourceExpressionRegion(scope, receipt)
            if node["op"] == "expr" and expr["op"] in {
                "call",
                "python_invoke",
                "library_call",
            }:
                region.discard(expr)
            elif node["op"] == "assign":
                value = region.value(node["value"])
                target = node["target"]
                ins = region.take("STORE_FAST", source=target)
                region.emit(
                    ins,
                    "frame_local_store",
                    target,
                    **region.fast_local(target, ins),
                    value=value,
                    ownership="install_result_before_releasing_previous_slot_object",
                )
                region.finish()
            elif node["op"] == "return":
                region.return_value(node)
            else:
                return stop(node, "source_function_statement_or_control_transfer")
        except (ValueError, IndexError, KeyError, TypeError) as error:
            return stop(node, str(error))
        regions.append(
            dict(op="source_object_region", steps=region.steps, **source_site(node))
        )
        if node["op"] == "return":
            result["completion"] = "conditional_source_body"
            return result
    return stop(definition, "source_return_and_frame_exit")

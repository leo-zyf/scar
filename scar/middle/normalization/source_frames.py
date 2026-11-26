"""Decode bound function-frame and expression ports into ordered object operations."""

import re

from scar.contract import (
    digest,
    json_bytes,
    python_constant_rhs_sha256,
    python_pool_items,
)
from scar.source_contracts import python_binary_operator, python_unary_operator
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


def class_body_activation(symbol, scope):
    """Describe entry into the actual created body, conditional on its invocation."""
    code = scope["source_code"]
    result = dict(
        selection="when_the_actual_created_body_is_invoked_in_a_matching_source_frame",
        symbol=symbol,
        code=code,
        locals="actual_entered_frame_locals_including_builder_prepared_mapping",
        globals="actual_created_function_globals",
        builtins="actual_created_function_builtins",
        completion="source_body_required",
        on_mismatch="continue_actual_body_in_its_source_equivalent_frame",
    )
    proof = scope.get("source_class_body")
    if not proof:
        return result
    rows = proof["instructions"]
    if (
        proof["code_sha256"] != code["code_sha256"]
        or not rows
        or rows[0]["opcode"] != "RESUME"
    ):
        raise ValueError("class body program does not match its class frame")
    steps, stack = [], []
    for row in rows[1:]:
        opcode, pc = row["opcode"], row["offset"]
        step = dict(offset=pc, instruction=row)
        if opcode == "LOAD_NAME":
            step.update(
                op="module_name_load",
                instruction=dict(
                    row, semantics_version=1, code_sha256=code["code_sha256"]
                ),
                type={"kind": "handle", "resource": "python_object"},
                lookup_order=["frame.locals", "frame.globals", "frame.builtins"],
                lookup="source_mapping_lookup_propagate_nonmissing_errors_or_NameError",
            )
            stack.append({"from_offset": pc})
        elif opcode in {"LOAD_CONST", "LOAD_SMALL_INT"}:
            step.update(
                op="source_object_constant",
                source=(
                    "current_frame_co_consts"
                    if opcode == "LOAD_CONST"
                    else "source_small_integer"
                ),
                operand=row["oparg"],
            )
            stack.append({"from_offset": pc})
        elif opcode == "STORE_NAME" and stack:
            step.update(
                op="source_frame_namespace_store",
                namespace="frame.locals",
                name=row["name"],
                index=row["index"],
                value=stack.pop(),
                may_raise=True,
                publication="use_source_STORE_NAME_on_actual_frame_locals_with_current_co_names_key",
            )
        elif opcode == "NOP":
            step.update(op="pass")
        elif opcode in {"RETURN_CONST", "RETURN_VALUE"} and row is rows[-1]:
            value = (
                dict(source="current_frame_co_consts", index=row["index"])
                if opcode == "RETURN_CONST"
                else stack.pop()
            )
            step.update(
                op="frame_return",
                opcode=opcode,
                value=value,
                destination="actual_caller",
                lifetime="transfer_result_and_release_frame_locals_at_source_exit",
            )
        else:
            raise ValueError("unsupported class namespace instruction or operand stack")
        steps.append(step)
    if stack or not steps or steps[-1]["op"] != "frame_return":
        raise ValueError("class namespace program has no complete source return")
    result.update(
        completion="conditional_source_body",
        entry_instruction=rows[0],
        steps=steps,
        callbacks="preserve_RESUME_and_every_instruction_event_including_source_NOP",
        unwind="propagate_from_failing_class_instruction_in_its_actual_frame_after_completed_mapping_effects",
    )
    return result


class SourceExpressionRegion:
    """Consume the graph operands and the sealed instructions in lockstep."""

    def __init__(
        self, scope, receipt, *, publications=None, definitions=None, constants=None
    ):
        code = scope["source_code"]
        if receipt["code_sha256"] != code["code_sha256"]:
            raise ValueError("source expression belongs to another code object")
        self.scope = scope
        self.publications = {} if publications is None else publications
        self.definitions = definitions
        self.constants = constants
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
        if "folded_value" in ins:
            if not (
                ins.get("rhs_sha256")
                and python_constant_rhs_sha256(node) == ins["rhs_sha256"]
            ):
                raise ValueError("folded operand disagrees with its source expression")
            node = {"type": {"class": ins["type"]}, "value": ins["folded_value"]}
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
        if self.constants is not None and "index" in ins:
            self.record_constant(ins["index"], dict(kind=kind, value=node["value"]))

    def record_constant(self, index, value):
        # A slot identifies an object within its actual code object. Equal values
        # in different slots or functions do not establish an identity relation.
        key = str(index)
        old = self.constants.setdefault(key, value)
        if old is not value and json_bytes(old) != json_bytes(value):
            raise ValueError("one code constant slot has conflicting typed values")

    def compiled_key(self, node):
        """Read a compiler-folded slice or tuple from the owning code object's pool."""

        def describe(part):
            if part["op"] == "literal":
                return dict(kind=part["type"]["class"], value=part["value"])
            if part["op"] == "slice":
                return dict(
                    kind="slice",
                    **{k: describe(part[k]) for k in ("lower", "upper", "step")},
                )
            if part["op"] == "aggregate" and part["kind"] == "tuple":
                return dict(
                    kind="tuple", items=[describe(item) for item in part["items"]]
                )
            raise ValueError("nonconstant operand in a compiled subscription key")

        ins = self.take("LOAD_CONST", source=node)
        value = describe(node)
        if ins.get("type") != value["kind"] or json_bytes(
            ins.get("constant_key")
        ) != json_bytes(value):
            raise ValueError("compiled subscription key disagrees with graph operands")
        if self.constants is not None:
            self.record_constant(ins["index"], value)
        return self.emit(
            ins,
            "source_object_constant",
            node,
            source="current_frame_co_consts",
            operand=ins["index"],
        )

    def slice_bounds(self, node):
        bounds = []
        for key in ("lower", "upper", "step"):
            child = node[key]
            omitted = (
                child["op"] == "literal"
                and child.get("value") is None
                and not child.get("location")
            )
            if key == "step" and omitted:
                break
            if omitted:
                ins = self.take("LOAD_CONST", source=node)
                self.check_literal(child, ins)
                bounds.append(
                    self.emit(
                        ins,
                        "source_object_constant",
                        node,
                        source="current_frame_co_consts",
                        operand=ins["index"],
                    )
                )
            else:
                bounds.append(self.value(child))
        return bounds

    def build_slice(self, node, bounds):
        ins = self.take("BUILD_SLICE", source=node)
        if ins["oparg"] != len(bounds) or len(bounds) not in (2, 3):
            raise ValueError("compiled slice arity disagrees with source bounds")
        return self.emit(
            ins, "source_object_build_slice", node, bounds=bounds, may_raise=True
        )

    def subscription(self, node, *, stored=None):
        if node.get("protocol") != "python_subscription" or len(node["indices"]) != 1:
            raise ValueError("subscription requires its original single key expression")
        base = self.value(node["base"])
        key = node["indices"][0]
        if key["op"] == "slice" and not self.instructions[self.index].get(
            "constant_key"
        ):
            bounds = self.slice_bounds(key)
            if self.next_opcode() in {"BINARY_SLICE", "STORE_SLICE"}:
                ins = self.take(
                    "BINARY_SLICE" if stored is None else "STORE_SLICE", source=node
                )
                if len(bounds) != 2:
                    raise ValueError("direct slice requires lower and upper operands")
                operands = dict(bounds=bounds)
            else:
                item = self.build_slice(key, bounds)
                operands = dict(key=item)
                ins = self.subscription_opcode(node, stored)
        else:
            operands = dict(key=self.value(key))
            ins = self.subscription_opcode(node, stored)
        if stored is not None:
            operands.update(
                value=stored,
                write_effect={
                    "receiver": "base",
                    "selector": "key" if "key" in operands else "bounds",
                    "extent": "receiver_protocol",
                    "exceptional_completion": "may_have_written",
                },
            )
        return self.emit(
            ins,
            "source_object_getitem" if stored is None else "source_object_setitem",
            node,
            opcode=ins["opcode"],
            base=base,
            **operands,
            may_raise=True,
        )

    def subscription_opcode(self, node, stored):
        if stored is not None:
            return self.take("STORE_SUBSCR", source=node)
        ins = self.take(
            "BINARY_OP" if self.minor >= 14 else "BINARY_SUBSCR", source=node
        )
        if ins["opcode"] == "BINARY_OP" and (
            ins.get("oparg") != 26 or ins.get("operator") != "subscript"
        ):
            raise ValueError("binary operation does not implement source subscription")
        return ins

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
        values = python_pool_items(ins, items)
        if values is None or ins.get("type") != "tuple":
            raise ValueError(
                "constant tuple slot disagrees with ordered display elements"
            )
        if self.constants is not None:
            self.record_constant(ins["index"], dict(kind="tuple", items=values))
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
        if (
            op in {"unary", "binary"}
            and self.index < len(self.instructions)
            and "folded_value" in self.instructions[self.index]
            and matches_position(
                self.instructions[self.index], node, self.code["source_key"]
            )
        ):
            return self.constant(node), None
        if (
            op in {"aggregate", "slice"}
            and self.instructions[self.index].get("constant_key")
            and matches_position(
                self.instructions[self.index], node, self.code["source_key"]
            )
        ):
            return self.compiled_key(node), None
        if op == "slice":
            return (
                self.build_slice(node, self.slice_bounds(node)),
                None,
            )
        if op == "index":
            return self.subscription(node), None
        if (
            op == "compare_chain"
            and len(node["operands"]) == len(node["operators"]) == 1
        ):
            left = self.value(node["left"])
            right = self.value(node["operands"][0])
            ins = self.take("COMPARE_OP", source=node)
            operator = node["operators"][0]
            operations = ("lt", "lte", "eq", "noteq", "gt", "gte")
            arg = ins["oparg"]
            boolean = self.minor >= 13 and bool(arg & 16)
            if (
                operator not in operations
                or operations.index(operator) != arg >> (4 if self.minor == 12 else 5)
                or ins.get("operator") != operator
                or ins.get("force_bool") is not boolean
            ):
                raise ValueError(
                    "comparison disagrees with compiled operation or truth conversion"
                )
            return (
                self.emit(
                    ins,
                    "source_object_compare",
                    node,
                    left=left,
                    right=right,
                    operator=operator,
                    force_bool=boolean,
                    opcode=ins["opcode"],
                ),
                None,
            )
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
        if op == "unary":
            value = self.value(node["value"])
            ins = self.take(
                "UNARY_NEGATIVE", "UNARY_INVERT", "CALL_INTRINSIC_1", source=node
            )
            operator = python_unary_operator(ins["opcode"], ins.get("oparg"))
            if (
                not operator
                or operator != node["operator"]
                or ins.get("operator") != operator
            ):
                raise ValueError(
                    "source unary slot disagrees with its operand or opcode"
                )
            return (
                self.emit(
                    ins,
                    "source_object_unary",
                    node,
                    operator=operator,
                    value=value,
                    may_raise=True,
                    dispatch="source_unary_numeric_slot",
                    ownership="consume_operand_and_transfer_actual_result",
                ),
                None,
            )
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
            if self.constants is not None:
                self.record_constant(
                    ins["index"],
                    dict(
                        kind="tuple",
                        items=[dict(kind="str", value=name) for name in names],
                    ),
                )
            kwref = self.emit(
                ins,
                "source_object_constant",
                node,
                source="current_frame_co_consts",
                operand=ins["index"],
            )
        return self.call_result(node, target, pair, args, kwref, values)

    def call_result(self, node, target, pair, args, kwref=None, values=()):
        ins = self.take(
            "CALL_KW" if kwref is not None and self.minor >= 13 else "CALL", source=node
        )
        if ins["oparg"] != len(args) + len(values):
            raise ValueError("argument count disagrees with source call")
        return self.emit(
            ins,
            "source_object_call",
            node,
            target=target,
            target_layout=pair or "callable_with_explicit_NULL",
            args=args,
            keyword_names=kwref,
            keyword_values=list(values),
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

    def iteration(self, node, ports):
        """Bind one acquired iterator to the loop's repeat, item and exit ports."""
        if node.get("protocol") != "iterator" or node.get("init"):
            raise ValueError("source loop requires the graph's iterator protocol")
        iterable = self.value(node["range"])
        acquire = self.take("GET_ITER")
        self.finish()
        store = ports["store"]
        dest = node["target"]
        if not matches_position(store, dest, self.code["source_key"]):
            raise ValueError("source iterator item has no matching target")
        if store["opcode"] == "STORE_FAST":
            target = dict(
                **self.fast_local(dest, store), instruction=store, **source_site(dest)
            )
            on_item = "consume_this_visits_owned_item_at_frame_slot_before_body"
        elif store["opcode"] in {"STORE_NAME", "STORE_GLOBAL"}:
            target = self.mapping_target(dest, store)
            on_item = "consume_this_visits_owned_item_at_mapping_store_before_body"
        else:
            raise ValueError("unsupported source iterator target")
        advance = ports["next"]
        prefixes = ports.get("next_prefix", [])
        backedges = ports.get("backedges") or [ports["backedge"]]
        if (
            advance["opcode"] != "FOR_ITER"
            or any(p["opcode"] != "EXTENDED_ARG" for p in prefixes)
            or any(
                back["opcode"] != "JUMP_BACKWARD"
                or back["target"] != (prefixes or [advance])[0]["offset"]
                for back in backedges
            )
            or not ports["cleanup"]
            or ports["cleanup"][0]["opcode"] != "END_FOR"
            or advance["target"] != ports["cleanup"][0]["offset"]
        ):
            raise ValueError(
                "source iterator edges disagree with the verified header or exit"
            )
        iterator = {"from_offset": acquire["offset"]}
        item = {"from_offset": advance["offset"]}
        return dict(
            evaluation=self.steps,
            acquire=dict(
                instruction=acquire,
                iterable=iterable,
                result=iterator,
                dispatch="apply_source_GET_ITER_once_and_release_consumed_iterable",
            ),
            advance=dict(
                instruction=advance,
                iterator=iterator,
                item=item,
                dispatch="apply_source_FOR_ITER_to_retained_iterator_on_each_header_visit",
                on_item=on_item,
                on_exhaustion="release_iterator_once_then_enter_else",
                on_other_exception="unwind_and_propagate_without_store_body_or_else",
            ),
            target=dict(target, value=item),
            edges={
                k: ports[k]
                for k in (
                    "body",
                    "backedge",
                    "backedges",
                    "next_prefix",
                    "cleanup",
                    "exhausted",
                )
                if k in ports
            },
        )

    def condition(self, node, ports):
        condition = node["condition"]
        negated = condition["op"] == "unary" and condition.get("operator") == "not"
        operand = condition["value"] if negated else condition
        if (
            ports.get("negated") is not negated
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
        already_bool = self.steps[-1].get("force_bool") is True
        conversion = (
            self.take("TO_BOOL", source=operand)
            if self.minor >= 13 and not already_bool
            else None
        )
        taken = ports.get("taken", "else")
        fall = ports.get("fallthrough", "then")
        if {taken, fall} != {"then", "else"}:
            raise ValueError("source branch requires two distinct control successors")
        jump = self.take(
            "POP_JUMP_IF_TRUE" if (taken == "then") != negated else "POP_JUMP_IF_FALSE",
            source=operand,
        )
        self.finish()
        if jump["target"] != ports[taken]["offset"]:
            raise ValueError("conditional jump does not enter its sealed successor")
        truth = (
            value
            if already_bool
            else self.emit(
                conversion or jump,
                "source_object_truth",
                operand,
                type="bool",
                value=value,
                opcode=(conversion or jump)["opcode"],
                dispatch=(
                    "consume_actual_call_result_at_source_truth_instruction_including_bool_or_len_and_release"
                    if operand["op"] in {"call", "python_invoke", "library_call"}
                    else "apply_source_truth_to_actual_operand_including_bool_or_len_and_release"
                ),
            )
        )
        return dict(
            op="source_object_region",
            type="bool",
            steps=self.steps,
            # Source truth and physical jump direction are independent.
            result=dict(value=truth, equals=not negated),
            branch=dict(
                instruction=jump,
                input=truth,
                taken=ports[taken],
                fallthrough=ports[fall],
                taken_arm=taken,
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
        receipt = node["source_definition"]
        child = receipt["child"]
        defaults = receipt.get("positional_defaults")
        keywords = receipt.get("keyword_defaults")
        if (
            not objects.plain_definition(
                node,
                positional_defaults=defaults is not None,
                keyword_defaults=keywords is not None,
            )
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
        default_ref = None
        if defaults is not None and "build_offset" in defaults:
            values = [self.value(operand) for operand in node["defaults"]]
            default_ref = self.build_sequence(node, "BUILD_TUPLE", values)
            if defaults != {
                "count": len(values),
                "build_offset": default_ref["from_offset"],
            }:
                raise ValueError(
                    "function defaults disagree with the defining tuple build"
                )
        elif defaults is not None:
            pool = self.take("LOAD_CONST", source=node)
            if (
                pool.get("type") != "tuple"
                or defaults
                != {"index": pool.get("index"), "count": pool.get("item_count")}
                or not node.get("defaults")
                or defaults["count"] != len(node["defaults"])
            ):
                raise ValueError(
                    "function defaults disagree with the defining frame's constant tuple"
                )
            default_ref = self.emit(
                pool,
                "source_object_constant",
                node,
                source="current_frame_co_consts",
                operand=pool["index"],
                expected_tuple={"count": defaults["count"]},
            )
        keyword_ref = (
            self.keyword_default_map(node, keywords) if keywords is not None else None
        )
        return self.make_function(node, symbol, child, default_ref, keyword_ref)

    def keyword_default_map(self, node, port):
        names = port["names"]
        defaults = [v for v in node["keyword_defaults"] if v is not None]
        build = next(
            (
                r
                for r in self.instructions[self.index :]
                if r["offset"] == port["build_offset"]
            ),
            {},
        )
        if (
            build.get("opcode") not in {"BUILD_MAP", "BUILD_CONST_KEY_MAP"}
            or build.get("oparg") != len(names)
            or len(names) != len(defaults)
        ):
            raise ValueError("keyword defaults need their ordered source map")
        packed = build["opcode"] == "BUILD_CONST_KEY_MAP"
        keys, values = [], []
        for name, operand in zip(names, defaults):
            if not packed:
                key = self.take("LOAD_CONST", source=node)
                if key.get("type") != "str" or key.get("value_sha256") != digest(
                    json_bytes(name)
                ):
                    raise ValueError("keyword default uses the wrong key")
                keys.append(
                    self.emit(
                        key,
                        "source_object_constant",
                        node,
                        source="current_frame_co_consts",
                        operand=key["index"],
                    )
                )
            values.append(self.value(operand))
        if packed:
            key = self.take("LOAD_CONST", source=node)
            if key.get("type") != "tuple" or key.get("names") != names:
                raise ValueError("keyword default uses the wrong source key tuple")
            pairs = dict(
                keys_tuple=self.emit(
                    key,
                    "source_object_constant",
                    node,
                    source="current_frame_co_consts",
                    operand=key["index"],
                )
            )
        else:
            pairs = dict(keys=keys)
        ins = self.take(build["opcode"], source=node)
        if ins["offset"] != port["build_offset"]:
            raise ValueError("keyword default map belongs to another operation")
        return self.emit(
            ins,
            "source_object_build_keyword_defaults",
            node,
            opcode=ins["opcode"],
            values=values,
            **pairs,
            may_raise=True,
            allocation="fresh_mapping_with_actual_key_and_value_objects_in_source_order",
        )

    def make_function(
        self, node, symbol, child, default_ref=None, keyword_ref=None, **ports
    ):
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
            and make.get("oparg")
            != int(default_ref is not None) + 2 * int(keyword_ref is not None)
            or self.minor >= 13
            and "oparg" in make
        ):
            raise ValueError("MAKE_FUNCTION flags disagree with the captured operands")
        function = self.emit(
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
            **ports,
            **(
                {"kwdefaults": keyword_ref}
                if self.minor == 12 and keyword_ref is not None
                else {}
            ),
            **(
                {"defaults": default_ref}
                if self.minor == 12 and default_ref is not None
                else {}
            ),
        )
        if self.minor >= 13:
            for flag, op, attr, reference in (
                (2, "source_object_set_function_kwdefaults", "kwdefaults", keyword_ref),
                (1, "source_object_set_function_defaults", "defaults", default_ref),
            ):
                if reference is None:
                    continue
                instruction = self.take("SET_FUNCTION_ATTRIBUTE", source=node)
                if instruction.get("oparg") != flag:
                    raise ValueError(
                        "source defaults attach to the wrong function attribute"
                    )
                function = self.emit(
                    instruction,
                    op,
                    node,
                    function=function,
                    **{attr: reference},
                    same_identity=True,
                )
        return function

    def class_value(self, node):
        receipt = node["source_class"]
        symbol = node["body_symbol"]
        definition = self.definitions[symbol]
        scope = definition["python_scope"]
        child = receipt["child"]
        if (
            self.scope["kind"] != "module"
            or definition.get("owner_class") != node["symbol"]
            or scope.get("kind") != "class"
            or scope.get("definition_code") != child
            or scope.get("source_code")
            != dict(self.code, kind="class", code_sha256=child["code_sha256"])
            or child.get("name") != node["source_name"]
            or any(node.get(k) for k in ("decorators", "keywords", "type_params"))
        ):
            raise ValueError("class construction requires its bound namespace code")
        if self.minor == 12:
            self.take("PUSH_NULL", source=node)
        builder = self.emit(
            self.take("LOAD_BUILD_CLASS", source=node),
            "source_object_load_build_class",
            node,
            namespace="current_frame_builtins",
            name="__build_class__",
            may_raise=True,
            lookup="source_LOAD_BUILD_CLASS_lookup_and_missing_or_mapping_exception",
        )
        if self.minor >= 13:
            self.take("PUSH_NULL", source=node)
        function = self.make_function(
            node, symbol, child, body_activation=class_body_activation(symbol, scope)
        )
        label = self.take("LOAD_CONST", source=node)
        if label.get("value") != child["name"]:
            raise ValueError("class name constant differs from the bound source label")
        name = self.emit(
            label,
            "source_object_constant",
            node,
            source="current_frame_co_consts",
            operand=label["index"],
        )
        args = [function, name, *(self.value(base) for base in node["bases"])]
        return self.call_result(node, builder, None, args)

    def statement(self, node):
        if node["op"] == "assign" and node["target"]["op"] == "index":
            value = self.value(node["value"])
            self.subscription(node["target"], stored=value)
            opcode = "module_object_expression"
            ports = dict(
                expression=dict(
                    op="source_object_region",
                    type="unit",
                    steps=self.steps,
                    **source_site(node),
                )
            )
        elif node["op"] == "raise":
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
        elif node["op"] in {"assign", "python_define", "python_define_class"}:
            if node["op"] == "python_define_class":
                value = self.class_value(node)
            elif node["op"] == "python_define":
                value = self.function(node)
            else:
                value = self.value(node["value"])
            target = self.target(node["target"], "STORE_NAME", "STORE_GLOBAL")
            definition = node.get("source_class") or node.get("source_definition")
            if definition and target["name"] != definition["child"]["name"]:
                raise ValueError("the source definition name and store target differ")
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

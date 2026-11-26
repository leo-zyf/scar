"""Structure the bound statements and control edges of one selected source frame."""

from scar.contract import json_bytes
from .source_frames import SourceExpressionRegion
from .values import source_site


class _FrameGap(ValueError):
    def __init__(self, node, detail):
        super().__init__(detail)
        self.node = node


def _suite(nodes, scope, constants, output):
    """Append a source suite once. Return True when control can fall through."""
    for node in nodes:
        op = node["op"]
        if (
            op == "decl"
            and node.get("initial_binding") == "unbound"
            and "init" not in node
        ):
            continue
        expr = node["value"] if op == "expr" else node
        receipt = expr.get("source_expression")
        if not receipt:
            raise _FrameGap(node, "instruction_region_for_current_function_statement")
        used = {}
        record = dict(op="source_object_region", **source_site(node))
        try:
            region = SourceExpressionRegion(scope, receipt, constants=used)
            if op == "expr" and expr["op"] in {"call", "python_invoke", "library_call"}:
                region.discard(expr)
            elif op == "expr" and expr["op"] == "literal":
                marker = region.take("NOP", source=expr)
                region.finish()
                region.emit(marker, "frame_instruction_event", expr, opcode="NOP")
            elif op == "assign":
                value = region.value(node["value"])
                target = node["target"]
                if target["op"] == "index":
                    region.subscription(target, stored=value)
                else:
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
            elif op == "return":
                region.return_value(node)
            elif op == "if":
                control = region.condition(node, receipt["branch"])
                record.update(
                    op="source_frame_if",
                    branch=control["branch"],
                    result=control["result"],
                    then=[],
                    **{"else": []},
                )
            elif op == "for":
                loop = region.iteration(node, receipt["iteration"])
                loop.pop("evaluation")
                record.update(op="source_frame_for", **loop, body=[], **{"else": []})
            else:
                raise ValueError("source_function_statement_or_control_transfer")
            for index, value in used.items():
                if index in constants and json_bytes(constants[index]) != json_bytes(
                    value
                ):
                    raise ValueError(
                        "function statements disagree on one code constant"
                    )
        except (ValueError, IndexError, KeyError, TypeError) as error:
            raise _FrameGap(node, str(error)) from error
        constants.update(used)
        record["steps"] = region.steps
        output.append(record)
        if op == "return":
            return False
        if op == "if":
            yes = _suite(node["then"], scope, constants, record["then"])
            no = _suite(node.get("else", []), scope, constants, record["else"])
            if yes and node.get("else") and not receipt["branch"].get("join"):
                raise _FrameGap(node, "branch_join_after_true_suite")
            if "join" in receipt["branch"]:
                record["branch"].update(
                    {
                        k: v
                        for k, v in receipt["branch"].items()
                        if k in ("then_exit", "else_exit", "exits", "join")
                    }
                )
            if not (yes or no):
                return False
        if op == "for":
            _suite(node["body"], scope, constants, record["body"])
            if not _suite(node.get("else", []), scope, constants, record["else"]):
                return False
    return True


def conditional_function_body(definition, descriptor):
    """Read a matched callee from the bound graph, starting after argument binding."""
    scope = definition["python_scope"]
    symbol = definition["name"]
    code = scope.get("source_code", {})
    child = scope.get("definition_code", {})
    body_id = descriptor.get("source_body", {})
    regions = []
    constants = {}
    result = dict(
        symbol=symbol,
        completion="conditional_source_prefix",
        function=dict(symbol=symbol, body=regions, constants=constants),
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
    try:
        falls = _suite(definition["body"], scope, constants, regions)
    except _FrameGap as error:
        return stop(error.node, str(error))
    if not falls:
        result["completion"] = "conditional_source_body"
        return result
    return stop(definition, "source_return_and_frame_exit")

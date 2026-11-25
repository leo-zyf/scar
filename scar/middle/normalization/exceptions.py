"""M2 contracts for exceptional control flow and lexical resource cleanup."""

from scar.contract import walk_operations
from . import host_protocols, namespace_operations
from .values import Value, literal, runtime, wire, written_targets

ERROR = "handle:error"
_CLASSES = {
    "BaseException": "base_exception",
    "Exception": "exception",
    "ValueError": "value_error",
    "TypeError": "type_error",
    "IndexError": "index_error",
    "KeyError": "key_error",
    "AttributeError": "attribute_error",
    "ImportError": "import_error",
    "ModuleNotFoundError": "module_not_found_error",
    "OSError": "os_error",
    "EOFError": "eof",
    "RuntimeError": "runtime_error",
    "SystemExit": "system_exit",
    "ZeroDivisionError": "zero_division_error",
    "OverflowError": "overflow_error",
    "ArithmeticError": "arithmetic_error",
    "LookupError": "lookup_error",
    "UnicodeError": "unicode_error",
    "UnicodeDecodeError": "unicode_decode_error",
}


def builtin(name):
    return Value("exception_class", _CLASSES[name]) if name in _CLASSES else None


def construct(frame, node, target, args, kwargs, steps, discard=False):
    if target.value == "unicode_decode_error":
        frame.fail(
            node,
            "exception_arguments",
            "constructing a UnicodeDecodeError requires its five-field byte-range contract",
        )
    if (
        kwargs
        or len(args) > 1
        or any(v.kind not in {"bool", "i64", "f64", "unit", "text"} for v in args)
    ):
        frame.fail(
            node,
            "exception_arguments",
            "the exception requires zero or one closed scalar argument",
        )
    category = target.value
    message = literal("")
    if args:
        arg = args[0]
        if category == "key_error" and arg.kind == "text":
            if type(arg.exact) is not str:
                frame.fail(
                    node,
                    "exception_repr",
                    "a dynamic string key requires a repr formatting contract",
                )
            message = literal(repr(arg.exact))
        else:
            message = host_protocols.invoke(
                frame,
                node,
                Value("host_callable", "builtins.str"),
                [arg],
                {},
                steps,
                False,
            )
    value = dict(
        op="error_make", category=category, message=message.value, type=wire(ERROR)
    )
    if category == "system_exit":
        arg = args[0] if args else literal(None)
        if arg.kind not in {"unit", "i64", "bool", "text"}:
            frame.fail(
                node, "exit_contract", "exit requires a status or a text message"
            )
        code = literal(1 if arg.kind == "text" else 0).value
        if arg.kind == "i64":
            code = arg.value
        elif arg.kind == "bool":
            code = dict(op="cast", type="i64", value=arg.value)
        value.update(
            exit_code=code,
            exit_message=arg.value if arg.kind == "text" else literal(None).value,
        )
    return frame.effect(value, ERROR, steps, discard)


def throw(frame, node, steps):
    if (
        type(node.get("has_exception")) is not bool
        or type(node.get("has_cause")) is not bool
    ):
        frame.fail(
            node,
            "raise_form",
            "a throw requires the explicit exception and cause presence flags",
        )
    if node["has_cause"] and not node["has_exception"]:
        frame.fail(node, "raise_form", "an explicit cause requires an exception")
    if frame.initializing and frame.project.imports.open_namespace:
        if not namespace_operations.open_object_statement(frame, node, steps):
            frame.fail(
                node,
                "source_raise_evidence",
                "the open module raise requires its current code, ordered operands and RAISE_VARARGS",
            )
        return
    if not node["has_exception"]:
        if not frame.handler_depth:
            frame.fail(node, "rethrow_scope", "no lexically active exception handler")
        steps.append(dict(op="rethrow"))
        return
    error = frame.evaluate(node["value"], steps)
    if node["has_cause"] and error.kind == ERROR:
        # The cause expression may rebind the original exception's source slot.
        error = frame.effect(error.value, ERROR, steps, facts=error.facts)
    # Python evaluates both expressions before normalizing exception classes.
    cause = frame.evaluate(node["cause"], steps) if node["has_cause"] else None
    if error.kind == "exception_class":
        error = construct(frame, node, error, [], {}, steps)
    if error.kind != ERROR:
        frame.fail(
            node,
            "raise_target",
            "only a closed exception object or exception class can be raised",
        )
    operation = dict(op="throw", value=error.value)
    if cause is not None:
        if cause.kind == "exception_class":
            cause = construct(frame, node, cause, [], {}, steps)
        if cause.kind not in {ERROR, "unit"}:
            frame.fail(
                node,
                "exception_cause",
                "an explicit cause requires a closed exception object, class or None",
            )
        operation.update(cause_mode="explicit", cause=cause.value)
    steps.append(operation)


def _affected(body):
    names = {}
    for node in walk_operations(body):
        removed = node["op"] in {"exception_handler", "unbind"}
        for target in written_targets(node):
            if target.get("op") == "var":
                key = (target.get("scope") == "global", target["name"])
                names[key] = removed or names.get(key, False)
    return names


def _prefix(frame, start, body, after, effects):
    """A throw may precede any assignment, or follow a write to an existing slot."""
    state = frame.exceptional_snapshot(start, after, effects)
    for (global_, name), deleted in _affected(body).items():
        scope = state[int(global_)]
        old = scope.get(name)
        if old is None:
            continue
        if deleted or not runtime(old.kind) or old.value.get("op") != "var":
            scope.pop(name)
        else:
            scope[name] = Value(old.kind, old.value)
    return state


def _categories(frame, node):
    test = node["test"]
    if test is None:
        return ["base_exception"]
    parts = (
        test["items"]
        if test["op"] == "aggregate" and test.get("kind") == "tuple"
        else [test]
    )
    cats = []
    for part in parts:
        if part["op"] not in {"var", "python_builtin"}:
            frame.fail(
                part, "handler_type", "handler matching needs a fixed exception class"
            )
        value = frame.ref(part)
        if value.kind != "exception_class":
            frame.fail(
                part, "handler_type", "handler matching needs a fixed exception class"
            )
        cats.append(value.value)
    return cats


def _marks(frame):
    return [
        (flow, tag, len(flow[tag]))
        for flow in frame.loops
        for tag in ("break", "continue")
    ]


def _take(marks):
    transfers = []
    for flow, tag, start in marks:
        transfers.extend((flow, tag, state) for state in flow[tag][start:])
        del flow[tag][start:]
    return transfers


def _unbind(alias, state):
    if alias:
        state[int(alias.get("scope") == "global")].pop(alias["name"], None)


def _handler(frame, node, exceptional, index, count):
    frame.restore(exceptional)
    header = namespace_operations.open_exception_handler(frame, node, index, count)
    dynamic = header is not None
    alias = node.get("binding")
    if dynamic:
        namespace_operations.observe_handler_effect(frame, header["match"])
        # Lookup can reenter or change bindings even when CHECK_EXC_MATCH is false.
        on_miss = frame.snapshot()
        for statement in header["on_match"]:
            namespace_operations.observe_handler_effect(frame, statement["value"])
        frame.region.handler_entries[index] = header
    else:
        on_miss = exceptional
        cats = _categories(frame, node)
        if node.get("name") and (
            not alias or alias.get("op") != "var" or alias.get("cell")
        ):
            frame.fail(
                node,
                "handler_binding",
                "the handler needs an identified lexical or module error slot",
            )
        binding = None
        if alias:
            name = alias["name"]
            global_ = alias.get("scope") == "global"
            slot = Value(ERROR, None).at(
                name, **({"scope": "global"} if global_ else {})
            )
            frame.store(alias, slot, [])
            binding = slot.value
        header = dict(categories=cats, binding=binding)
    code = []
    marks = _marks(frame)
    frame.handler_depth += 1
    frame.enter_control("handler", node)
    try:
        falls = frame.sequence(node["body"], code)
    finally:
        frame.leave_control()
        frame.handler_depth -= 1
    if dynamic and (
        body_raise := header["match"]["source_exception_handler"].get("body_raise")
    ):
        terminal = code[-1].get("value", {}) if code else {}
        if (
            terminal.get("op") != "source_exception_raise"
            or terminal.get("instruction") != body_raise
        ):
            frame.fail(
                node,
                "handler_raise",
                "the handler's terminating raise differs from its sealed exception dispatch",
            )
    if (
        dynamic
        and falls
        and not header["match"]["source_exception_handler"]["normal_exit"]
    ):
        frame.fail(
            node,
            "handler_exit",
            "the source arm has no normal continuation for this handler body",
        )

    def exited(state):
        if dynamic and header["cleanup"]:
            frame.restore(state)
            namespace_operations.observe_handler_effect(
                frame, header["cleanup"][0]["value"]
            )
            state = frame.snapshot()
        _unbind(alias, state)
        return state

    state = exited(frame.snapshot()) if falls else None
    for flow, tag, before in marks:
        pending = flow[tag]
        pending[before:] = [exited(path) for path in pending[before:]]
    return dict(header, body=code), state, on_miss


def _project_final(frame, path, modifications, before, after):
    result = frame.project_final_state(path, before, after)
    for global_, name in modifications:
        index = int(global_)
        if name in after[index]:
            result[index][name] = after[index][name]
        else:
            result[index].pop(name, None)
    return result


def protected(frame, node, steps):
    start = frame.snapshot()
    normal = []
    body, otherwise, final, handlers = [], [], [], []
    marks = _marks(frame)
    frame.dynamic += 1
    try:
        frame.enter_control("try_body", node)
        try:
            falls = frame.sequence(node["body"], body)
        finally:
            frame.leave_control()
        body_state = frame.snapshot()
        if falls:
            frame.enter_control("try_else", node)
            try:
                if frame.sequence(node["else"], otherwise):
                    normal.append(frame.snapshot())
            finally:
                frame.leave_control()
        exceptional = _prefix(frame, start, node["body"], body_state, body)
        for index, arm in enumerate(node["handlers"]):
            handler, state, exceptional = _handler(
                frame, arm, exceptional, index, len(node["handlers"])
            )
            handlers.append(handler)
            if state is not None:
                normal.append(state)
        pending = _take(marks)
        if node["finally"]:
            # Cleanup includes exceptional prefixes and pending transfers.
            # Only changes made by the cleanup project back to each path.
            candidates = [
                _prefix(
                    frame,
                    start,
                    [node["body"], node["handlers"], node["else"]],
                    frame.snapshot(),
                    [body, handlers, otherwise],
                ),
                *normal,
                *[state for _, _, state in pending],
            ]
            frame.merge(candidates)
            before = frame.snapshot()
            frame.enter_control("finally", node)
            try:
                falls = frame.sequence(node["finally"], final)
            finally:
                frame.leave_control()
            if falls:
                after = frame.snapshot()
                written = _affected(node["finally"])
                normal = [
                    _project_final(frame, state, written, before, after)
                    for state in normal
                ]
                pending = [
                    (flow, tag, _project_final(frame, state, written, before, after))
                    for flow, tag, state in pending
                ]
            else:
                normal, pending = [], []
        for flow, tag, state in pending:
            flow[tag].append(state)
    finally:
        frame.dynamic -= 1
    steps.append(
        dict(
            op="try",
            body=body,
            handlers=handlers,
            **{"else": otherwise, "finally": final},
        )
    )
    if normal:
        frame.merge(normal)
    return bool(normal)


def resources(frame, node, steps):
    if node.get("asynchronous") is not False or not node["resources"]:
        frame.fail(node, "context_protocol", "a synchronous stream context is required")
    body = steps
    for item in node["resources"]:
        resource = frame.evaluate(item["manager"], body)
        if resource.kind != "handle:text_stream":
            frame.fail(
                node,
                "context_protocol",
                "an identified text-stream enter and close protocol is required",
            )
        handle = frame.effect(resource.value, resource.kind, body, facts=resource.facts)
        nested = []
        body.append(dict(op="resource_scope", resource=handle.value, body=nested))
        body = nested
        if item.get("binding") is not None:
            frame.store(item["binding"], handle, body)
    return frame.sequence(node["body"], body)

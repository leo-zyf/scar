"""M2 contracts for exceptional control flow and lexical resource cleanup."""

from scar.contract import walk_operations
from . import host_protocols
from .values import Value, literal, runtime, wire

ERROR = "handle:error"
_CLASSES = {
    "BaseException": "base_exception",
    "Exception": "exception",
    "ValueError": "value_error",
    "TypeError": "type_error",
    "IndexError": "index_error",
    "KeyError": "key_error",
    "OSError": "os_error",
    "EOFError": "eof",
    "RuntimeError": "runtime_error",
    "SystemExit": "system_exit",
}


def builtin(name):
    return Value("exception_class", _CLASSES[name]) if name in _CLASSES else None


def construct(frame, node, target, args, kwargs, steps, discard=False):
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
    if node["has_cause"]:
        frame.fail(
            node,
            "exception_cause",
            "explicit exception chaining or context suppression needs a cause contract",
        )
    if not node["has_exception"]:
        if not frame.handler_depth:
            frame.fail(node, "rethrow_scope", "no lexically active exception handler")
        steps.append(dict(op="rethrow"))
        return
    error = frame.evaluate(node["value"], steps)
    if error.kind == "exception_class":
        error = construct(frame, node, error, [], {}, steps)
    if error.kind != ERROR:
        frame.fail(
            node,
            "raise_target",
            "only a closed exception object or exception class can be raised",
        )
    steps.append(dict(op="throw", value=error.value))


def _affected(body):
    names = {}

    def record(target, remove=False):
        if isinstance(target, dict) and target.get("op") == "var":
            key = (target.get("scope") == "global", target["name"])
            names[key] = remove or names.get(key, False)

    for node in walk_operations(body):
        op = node["op"]
        if op in {"assign", "update", "for", "python_define", "python_define_class"}:
            record(node.get("target"))
        elif op == "python_import":
            record(node.get("binding"))
        elif op == "exception_handler":
            record(node.get("binding"), True)
        elif op == "with_context":
            for resource in node["resources"]:
                record(resource.get("binding"))
        elif op == "unbind":
            for target in node["targets"]:
                record(target, True)
    return names


def _prefix(start, body):
    """A throw may precede any assignment, or follow a write to an existing slot."""
    state = tuple(slots.copy() for slots in start)
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


def _handler(frame, node, exceptional):
    frame.restore(exceptional)
    cats = _categories(frame, node)
    alias = node.get("binding")
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
        # The handler operation binds and clears this real slot, including global
        # aliases visible to a called function. Other saved handles retain ownership.
        name = alias["name"]
        global_ = alias.get("scope") == "global"
        slot = Value(ERROR, None).at(name, **({"scope": "global"} if global_ else {}))
        # The binding operation itself performs the checked assignment.
        frame.store(alias, slot, [])
        binding = slot.value
    code = []
    marks = _marks(frame)
    frame.handler_depth += 1
    falls = frame.sequence(node["body"], code)
    frame.handler_depth -= 1
    state = frame.snapshot() if falls else None
    if state:
        _unbind(alias, state)
    for flow, tag, before in marks:
        for pending in flow[tag][before:]:
            _unbind(alias, pending)
    return dict(categories=cats, binding=binding, body=code), state


def _project_final(path, modifications, after):
    result = tuple(scope.copy() for scope in path)
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
    if frame.sequence(node["body"], body) and frame.sequence(node["else"], otherwise):
        normal.append(frame.snapshot())
    exceptional = _prefix(start, node["body"])
    for arm in node["handlers"]:
        handler, state = _handler(frame, arm, exceptional)
        handlers.append(handler)
        if state is not None:
            normal.append(state)
    pending = _take(marks)
    if node["finally"]:
        # Cleanup also runs for an unhandled exception, return or loop transfer.
        # Prefixes are conservative and cannot introduce a definitely assigned slot.
        candidates = [
            _prefix(start, [node["body"], node["handlers"], node["else"]]),
            *normal,
            *[state for _, _, state in pending],
        ]
        frame.merge(candidates)
        if frame.sequence(node["finally"], final):
            after = frame.snapshot()
            written = _affected(node["finally"])
            normal = [_project_final(state, written, after) for state in normal]
            pending = [
                (flow, tag, _project_final(state, written, after))
                for flow, tag, state in pending
            ]
        else:
            normal, pending = [], []
    for flow, tag, state in pending:
        flow[tag].append(state)
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

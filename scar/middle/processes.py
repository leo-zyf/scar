"""M2 process identity, declared image references and ordered pipe observations."""

from .values import Value, literal, wire

PROCESS = "handle:process"
STREAM = "handle:text_stream"


def is_process(kind):
    return kind == PROCESS or kind.startswith(PROCESS + ":")


def _host(frame, interface, result, args, steps, discard=False):
    projection = interface == "process.stdout"
    return frame.effect(
        dict(
            op="host_call",
            interface=interface,
            semantics_version=1,
            args=[v.value for v in args],
            type=wire(result),
            effects=["STATE"] if projection else ["IO", "STATE", "ORDER"],
            reads=["process.state"],
            writes=[] if projection else ["process.state", "process.completion"],
            may_raise=True,
            operator_family="process",
        ),
        result,
        steps,
        discard,
    )


def member(frame, node, base, name, steps):
    if not is_process(base.kind):
        return None
    if name == "wait":
        return Value("process_method", (base, name))
    if name in {"stdout", "stderr", "stdin"}:
        routes = base.facts.get("process_routes", {})
        if base.kind.startswith(PROCESS + ":"):
            output, errors = base.kind[len(PROCESS) + 1 :].split(":")
            routes = dict(stdin="inherit", stdout=output, stderr=errors)
        if name not in routes:
            frame.fail(
                node, "process_channel", "the process stream route is not proven"
            )
        if routes[name] != "pipe":
            return literal(None)
        if name != "stdout":
            frame.fail(
                node, "process_channel", "only a single readable output pipe is closed"
            )
        return _host(frame, "process.stdout", STREAM, [base], steps)
    frame.fail(node, "process_member", f"no closed process observation for {name}")


def persist_method(frame, target, method, steps):
    receiver, name = method.value
    slot = "@process-method/" + target["name"]
    frame.project.globals[slot] = receiver.kind
    captured = receiver.at(slot, scope="global")
    steps.append(dict(op="assign", target=captured.value, value=receiver.value))
    return Value(method.kind, (captured, name))


def invoke(frame, node, target, args, keywords, steps, discard):
    if target.kind == "process_method":
        if args or keywords:
            frame.fail(
                node, "process_arguments", "wait timeout needs its own timing contract"
            )
        return _host(frame, "process.wait", "i64", [target.value[0]], steps, discard)
    if target.kind != "host_callable" or target.value != "subprocess.Popen":
        return None
    if len(args) > 1 or (args and "args" in keywords):
        frame.fail(
            node, "process_arguments", "process launch needs one argument vector"
        )
    options = dict(keywords)
    command = args[0] if args else options.pop("args", None)
    if command is None:
        frame.fail(node, "process_arguments", "process launch needs an argument vector")
    if command.kind == "text":
        argv = [command]
    elif command.kind in {"list", "tuple"}:
        argv = list(command.value)
    else:
        frame.fail(
            node,
            "process_arguments",
            "process arguments need a fixed vector of text values",
        )
    if not argv or any(v.kind != "text" for v in argv):
        frame.fail(
            node, "process_arguments", "process argument vector must be nonempty text"
        )
    source_target = argv[0].exact
    matches = [
        i for i in frame.project.execution_images if source_target == "./" + i["output"]
    ]
    if len(matches) != 1:
        frame.fail(
            node,
            "process_image_binding",
            "the executable has no unique declared application image; external tools, build actions and launchers need separate bindings",
        )
    image = matches[0]
    shell = options.pop("shell", literal(False))
    if shell.exact is not False:
        frame.fail(
            node,
            "process_shell",
            "a shell invocation requires its own bound command and status contract",
        )
    stdin = options.pop("stdin", literal(None))
    stdout = options.pop("stdout", literal(None))
    stderr = options.pop("stderr", literal(None))
    if (
        stdin.kind != "unit"
        or stdout.exact not in (None, -1)
        or stderr.exact not in (None, -2)
    ):
        frame.fail(
            node,
            "process_channel",
            "launch supports inherited input, one output pipe and inherited or merged errors",
        )
    if "text" in options and "universal_newlines" in options:
        frame.fail(
            node,
            "process_options",
            "combined text aliases need a single consistent mode",
        )
    text_mode = options.pop("text", options.pop("universal_newlines", literal(False)))
    encoding = options.pop("encoding", literal(None))
    errors = options.pop("errors", literal(None))
    default_encoding = node.get("source_runtime", {}).get("default_text_encoding", "")
    encoding_name = default_encoding if encoding.kind == "unit" else encoding.exact
    if stdout.exact == -1 and (
        not isinstance(encoding_name, str)
        or encoding_name.lower().replace("-", "").replace("_", "") != "utf8"
        or errors.exact not in (None, "strict")
        or text_mode.exact is not True
        and encoding.kind == "unit"
    ):
        frame.fail(
            node,
            "process_encoding",
            "process text requires a proven strict UTF-8 decoding contract",
        )
    defaults = {
        "bufsize": -1,
        "cwd": None,
        "env": None,
        "close_fds": True,
        "restore_signals": True,
        "start_new_session": False,
    }
    if (
        any(k in options and options.pop(k).exact != v for k, v in defaults.items())
        or options
    ):
        frame.fail(
            node,
            "process_options",
            "nondefault process environment, lifetime or buffering needs another contract",
        )
    routes = dict(
        stdin="inherit",
        stdout="pipe" if stdout.exact == -1 else "inherit",
        stderr="stdout" if stderr.exact == -2 else "inherit",
    )
    frame.project.process_ops.add("process_spawn")
    frame.project.decisions.append(
        dict(
            code="bound_process_image",
            symbol=frame.name,
            location=node.get("location"),
            image_id=image["id"],
            output=image["output"],
            source_target=source_target,
            proof="exact logical executable reference in the explicitly declared application image namespace; no filesystem-name guessing or source-language subprocess fallback",
        )
    )
    kind = PROCESS + ":" + routes["stdout"] + ":" + routes["stderr"]
    return frame.effect(
        dict(
            op="process_spawn",
            operator_family="process",
            type=wire(kind),
            image_id=image["id"],
            argv=[v.value for v in argv],
            **routes,
            encoding="utf-8",
            errors="strict",
            newline="universal",
            cwd="inherit",
            effects=["CTRL", "STATE", "IO"],
            reads=["process.cwd", "process.environment", "program.images"],
            writes=["process.children", "external.state"],
            external_effect="UNKNOWN",
            may_raise=True,
        ),
        kind,
        steps,
        discard,
        facts={"process_routes": routes},
    )


def next_line(frame, stream, steps):
    """The empty result denotes EOF; an unterminated final line is still yielded."""
    return frame.effect(
        dict(
            op="host_call",
            interface="stream.readline",
            semantics_version=1,
            args=[stream.value, literal(-1).value],
            type="text",
            effects=["IO", "STATE"],
            reads=["stream.state", "filesystem"],
            writes=["stream.state"],
            may_raise=True,
        ),
        "text",
        steps,
    )

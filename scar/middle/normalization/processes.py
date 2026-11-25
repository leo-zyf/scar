"""M2 process protocols, explicit target obligations and ordered pipe observations."""

import re

from scar.contract import digest, json_bytes
from . import sequences
from .values import Value, is_sequence, literal, source_site, wire

PROCESS = "handle:process"
STREAM = "handle:text_stream"
_SHELL_COMMAND = re.compile(
    r"[A-Za-z0-9_./,:@%+=-]+(?:[ \t]+[A-Za-z0-9_./,:@%+=-]+)*\Z"
)
_SHELL_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
_MAX_RANK = str((1 << 63) - 1)


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


def _command(frame, node, command, shell):
    """Describe a source command without authorizing or interpreting its target."""
    if shell.exact not in (True, False) or type(shell.exact) is not bool:
        frame.fail(node, "process_shell", "shell needs a fixed boolean contract")
    if shell.exact:
        if command.kind != "text":
            frame.fail(
                node,
                "process_arguments",
                "a shell command requires a text value",
            )
        return dict(form="posix_shell", text=command.value)
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
    return dict(form="argv", argv=[value.value for value in argv])


def _image(frame, target):
    return [
        image
        for image in frame.project.execution_images
        if target == "./" + image["output"]
    ]


def _launch(frame, node, command, shell):
    """Prove a declared-image binding; otherwise retain a logical request."""
    if shell.exact:
        text = command.exact
        runtime = node.get("source_runtime", {})
        if (
            not isinstance(text, str)
            or not _SHELL_COMMAND.fullmatch(text.strip(" \t"))
            or runtime.get("os_name") != "posix"
            or runtime.get("platform") not in {"linux", "linux2"}
        ):
            return None
        words = text.split()
        if _SHELL_ASSIGNMENT.match(words[0]):
            return None
        argv = [literal(word) for word in words]
    else:
        argv = [command] if command.kind == "text" else list(command.value)
    source_argv = argv
    matches = _image(frame, argv[0].exact)
    launcher = None
    if not matches:
        words = [value.exact for value in argv]
        if (
            len(words) < 4
            or any(not isinstance(word, str) for word in words)
            or words[1] not in {"-n", "-np", "--np"}
            or not words[2].isascii()
            or not words[2].isdecimal()
            or ":" in words
        ):
            return None
        ranks = words[2].lstrip("0")
        if (
            not ranks
            or len(ranks) > len(_MAX_RANK)
            or len(ranks) == len(_MAX_RANK)
            and ranks > _MAX_RANK
        ):
            return None
        launcher = dict(
            kind="mpi",
            ranks=literal(int(ranks)).value,
            semantics_version=1,
            source_reference=words[0],
            rank_option=words[1],
        )
        argv = argv[3:]
        matches = _image(frame, argv[0].exact)
    if len(matches) != 1:
        return None
    fields = {}
    if launcher is not None or shell.exact:
        runtime = node.get("source_runtime", {})
        fields.update(
            shell=shell.exact,
            source_command=dict(
                form="posix_shell" if shell.exact else "argv",
                argv=[value.exact for value in source_argv],
                text=command.exact if shell.exact else None,
                shell_path="/bin/sh" if shell.exact else None,
                runtime={
                    key: runtime[key]
                    for key in ("interpreter", "platform", "os_name")
                    if key in runtime
                },
            ),
        )
    if launcher is not None:
        fields["launcher"] = launcher
    return matches[0], argv, fields


def _process_error(frame, node, args, keywords, steps, discard):
    parameters = ("returncode", "cmd", "output", "stderr")
    if len(args) > len(parameters) or set(keywords) - set(parameters):
        frame.fail(node, "process_error_arguments", "unclosed process error fields")
    fields = dict(zip(parameters, args))
    if fields.keys() & keywords.keys():
        frame.fail(node, "process_error_arguments", "repeated process error field")
    fields.update(keywords)
    if not {"returncode", "cmd"} <= fields.keys():
        frame.fail(node, "process_error_arguments", "status and command are required")
    fields.setdefault("output", literal(None))
    fields.setdefault("stderr", literal(None))
    if (
        fields["returncode"].kind != "i64"
        or fields["cmd"].kind != "text"
        or any(fields[key].kind not in {"unit", "text"} for key in ("output", "stderr"))
    ):
        frame.fail(
            node,
            "process_error_fields",
            "process errors require integer status, text command and optional text output",
        )
    return frame.effect(
        dict(
            op="process_error_make",
            operator_family="process_error",
            category="called_process_error",
            **source_site(node),
            **{key: value.value for key, value in fields.items()},
            type=wire("handle:error"),
            effects=["MEM", "STATE"],
            reads=[],
            writes=["object.allocation"],
            may_raise=True,
        ),
        "handle:error",
        steps,
        discard,
    )


def invoke(frame, node, target, args, keywords, steps, discard):
    if target.kind == "process_method":
        if args or keywords:
            frame.fail(
                node, "process_arguments", "wait timeout needs its own timing contract"
            )
        return _host(frame, "process.wait", "i64", [target.value[0]], steps, discard)
    if target.kind != "host_callable":
        return None
    if target.value == "subprocess.CalledProcessError":
        return _process_error(frame, node, args, keywords, steps, discard)
    if target.value != "subprocess.Popen":
        return None
    if len(args) > 1 or (args and "args" in keywords):
        frame.fail(
            node, "process_arguments", "process launch needs one argument vector"
        )
    options = dict(keywords)
    command = args[0] if args else options.pop("args", None)
    if command is None:
        frame.fail(node, "process_arguments", "process launch needs an argument vector")
    shell = options.pop("shell", literal(False))
    if is_sequence(command.kind):
        members = sequences.elements(frame, node, command, steps)
        if members is None:
            frame.fail(
                node,
                "process_arguments",
                "process arguments need a current finite vector of text values",
            )
        command = Value("tuple", members)
    request = _command(frame, node, command, shell)
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
    launch = _launch(frame, node, command, shell)
    if launch is None:
        identity = dict(symbol=frame.name, **source_site(node))
        if not source_site(node):
            identity["ordinal"] = len(frame.project.process_requests)
        request_id = "process-request:" + digest(json_bytes(identity))
        fields = dict(
            op="process_request",
            request_id=request_id,
            command=request,
            shell=shell.exact,
            target_binding=None,
            binding_obligations=[
                "qualified_executable_target",
                "target_effect_and_completion_contract",
                "preserved_command_environment_and_working_directory",
            ],
        )
        frame.project.process_requests.setdefault(
            request_id,
            dict(
                **identity,
                request_id=request_id,
                binding_obligations=fields["binding_obligations"],
                status="UNBOUND",
            ),
        )
    else:
        image, argv, launch_fields = launch
        fields = dict(
            op="process_spawn",
            image_id=image["id"],
            argv=[value.value for value in argv],
            **launch_fields,
        )
        frame.project.decisions.append(
            dict(
                code="bound_process_image",
                symbol=frame.name,
                location=node.get("location"),
                image_id=image["id"],
                output=image["output"],
                source_target=argv[0].exact,
                proof="exact logical executable reference in the explicitly declared application image namespace; no filesystem-name guessing or source-language subprocess fallback",
            )
        )
    frame.project.process_ops.add(fields["op"])
    kind = PROCESS + ":" + routes["stdout"] + ":" + routes["stderr"]
    return frame.effect(
        dict(
            **fields,
            capture=digest(
                json_bytes(
                    [frame.name, fields["op"], source_site(node), frame.temporary()]
                )
            ),
            operator_family="process",
            **source_site(node),
            type=wire(kind),
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

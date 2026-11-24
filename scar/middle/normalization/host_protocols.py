"""M2: bind verified source host APIs to language-independent effect contracts."""

import math

from . import scalars, text
from .values import Value, argument_evidence, literal, source_site, wire

_STREAM = "handle:text_stream"
_REQUIRED = object()
# Names and defaults belong to source binding. The emitted interfaces do not.
# Each item is (source parameter, value type, default); None is a real default.
_SIGNATURES = {
    "os.path.exists": ("fs.exists", "bool", [("path", "text", _REQUIRED)]),
    "os.path.isfile": ("fs.is_file", "bool", [("path", "text", _REQUIRED)]),
    "os.path.isdir": ("fs.is_dir", "bool", [("path", "text", _REQUIRED)]),
    "os.mkdir": (
        "fs.mkdir",
        "unit",
        [("path", "text", _REQUIRED), ("mode", "i64", 0o777), ("dir_fd", "unit", None)],
    ),
    "os.makedirs": (
        "fs.makedirs",
        "unit",
        [
            ("name", "text", _REQUIRED),
            ("mode", "i64", 0o777),
            ("exist_ok", "bool", False),
        ],
    ),
    "os.chdir": ("process.chdir", "unit", [("path", "text", _REQUIRED)]),
    "os.getcwd": ("process.getcwd", "text", []),
    "shutil.rmtree": (
        "fs.remove_tree",
        "unit",
        [("path", "text", _REQUIRED), ("ignore_errors", "bool", False)],
    ),
    "shutil.copyfile": (
        "fs.copy_file",
        "text",
        [
            ("src", "text", _REQUIRED),
            ("dst", "text", _REQUIRED),
            ("follow_symlinks", "bool", True),
        ],
    ),
    "shutil.copy": (
        "fs.copy",
        "text",
        [
            ("src", "text", _REQUIRED),
            ("dst", "text", _REQUIRED),
            ("follow_symlinks", "bool", True),
        ],
    ),
    "shutil.copy2": (
        "fs.copy2",
        "text",
        [
            ("src", "text", _REQUIRED),
            ("dst", "text", _REQUIRED),
            ("follow_symlinks", "bool", True),
        ],
    ),
    "shutil.copytree": (
        "fs.copy_tree",
        "text",
        [("src", "text", _REQUIRED), ("dst", "text", _REQUIRED)],
    ),
    "time.time": ("time.wall", "f64", []),
    "time.sleep": ("time.sleep", "unit", [("secs", "f64", _REQUIRED)]),
}
_METHODS = {
    "write": ("stream.write", "i64", [("s", "text", _REQUIRED)]),
    "flush": ("stream.flush", "unit", []),
    "close": ("stream.close", "unit", []),
    "read": ("stream.read", "text", [("size", "i64", -1)]),
    "readline": ("stream.readline", "text", [("size", "i64", -1)]),
}
_MEMBERS = {
    "os": {"path", "mkdir", "makedirs", "chdir", "getcwd"},
    "os.path": {"join", "exists", "isfile", "isdir"},
    "shutil": {"rmtree", "copyfile", "copy", "copy2", "copytree"},
    "time": {"time", "sleep"},
    "sys": {"exit"},
    "math": {"pi", "e", "tau"}
    | scalars.MATH_FUNCTIONS.keys()
    | scalars.MATH_CLASSIFICATIONS,
    "subprocess": {"Popen", "PIPE", "STDOUT", "CalledProcessError"},
}


def supports_import(module):
    """Whether this layer has a namespace and initialization contract for a module."""
    return module in _MEMBERS


def array_effect_free(target):
    """These closed interfaces may affect IO/state, but never write array storage."""
    if target.kind == "host_method":
        return target.value[1] in _METHODS
    return target.kind == "host_callable" and target.value in (
        _SIGNATURES.keys()
        | {
            "builtins.open",
            "builtins.input",
            "builtins.str",
            "builtins.repr",
            "builtins.ascii",
            "builtins.format",
            "sys.exit",
            "os.path.join",
            "os.makedirs",
            "os.chdir",
            "os.getcwd",
        }
        | {
            "math." + name
            for name in scalars.MATH_FUNCTIONS.keys() | scalars.MATH_CLASSIFICATIONS
        }
    )


def readonly_callable(base, name):
    """Resolve an array-readonly namespace member without executing an operation."""
    if name not in _MEMBERS.get(base.value, ()):
        return None
    candidate = Value("host_callable", base.value + "." + name, base.facts)
    return candidate if array_effect_free(candidate) else None


def builtin(frame, node, name):
    if name in {"open", "str", "repr", "ascii", "format", "input"}:
        return Value(
            "host_callable", "builtins." + name, node.get("source_runtime", {})
        )
    return None


def import_provider(frame, node):
    source = node.get("source_provider", {})
    module = node["module"]
    if (
        source.get("status") != "verified_stdlib"
        or source.get("module") != module.split(".")[0]
        or source.get("platform") not in {"linux", "linux2"}
        or not source.get("sha256")
    ):
        frame.fail(
            node,
            "host_provider",
            "host APIs require a pinned POSIX standard-library identity",
        )
    identity = source["module"]
    key = "stdlib:" + identity
    prior = frame.project.providers.get(key)
    if prior and prior["sha256"] != source["sha256"]:
        frame.fail(
            node,
            "host_provider",
            "one program resolves conflicting host provider identities",
        )
    frame.project.providers[key] = dict(
        source, contract="closed host APIs with explicit resource effects"
    )
    if module not in _MEMBERS:
        frame.fail(node, "host_provider", f"no host namespace contract for {module}")
    if module == "os.path":
        _path_identity(frame, node, source)
    namespace = Value("host_namespace", module, source)
    if node["request"]["form"] == "from":
        return member(frame, node, namespace, node["member"])
    if module == "os.path" and not node.get("alias"):
        return Value("host_namespace", "os", source)
    return namespace


def member(frame, node, base, name):
    if base.kind == _STREAM:
        if name not in _METHODS:
            frame.fail(node, "stream_member", f"no closed stream method for {name}")
        return Value("host_method", (base, name))
    if base.kind != "host_namespace":
        return None
    if name not in _MEMBERS.get(base.value, ()):
        frame.fail(node, "host_member", f"no host API contract for {base.value}.{name}")
    qualified = base.value + "." + name
    if qualified in {"subprocess.PIPE", "subprocess.STDOUT"}:
        return literal(-1 if name == "PIPE" else -2)
    if qualified == "os.path":
        _path_identity(frame, node, base.facts)
        return Value("host_namespace", qualified, base.facts)
    if base.value == "math" and name in {"pi", "e", "tau"}:
        return literal(getattr(math, name))
    return Value("host_callable", qualified, base.facts)


def _path_identity(frame, node, source):
    alias = source.get("aliases", {}).get("os.path", {})
    if alias.get("status") != "verified_stdlib" or not alias.get("sha256"):
        frame.fail(
            node,
            "host_provider",
            "the imported path implementation requires its own verified identity",
        )


def _bind(frame, node, args, keywords, parameters):
    if len(args) > len(parameters):
        frame.fail(node, "host_arguments", "too many positional arguments")
    unknown = set(keywords) - {p[0] for p in parameters}
    if unknown:
        frame.fail(
            node,
            "host_arguments",
            f"unclosed host keyword arguments: {sorted(unknown)}",
        )
    bound = []
    for i, (name, kind, default) in enumerate(parameters):
        if i < len(args):
            if name in keywords:
                frame.fail(node, "host_arguments", f"multiple values for {name}")
            value = args[i]
        elif name in keywords:
            value = keywords[name]
        elif default is not _REQUIRED:
            value = literal(default)
        else:
            frame.fail(node, "host_arguments", f"missing argument {name}")
        if (
            kind == "f64"
            and value.kind == "i64"
            and max(map(abs, value.bounds)) <= 1 << 53
        ):
            value = Value(
                "f64", dict(op="cast", type="f64", value=value.value), value.facts
            )
        allowed = kind if isinstance(kind, tuple) else (kind,)
        if value.kind not in allowed:
            frame.fail(
                node, "host_arguments", f"{name} requires {kind}, got {value.kind}"
            )
        bound.append(value)
    return bound


def _emit(frame, node, interface, result, args, steps, discard, *, result_facts=None):
    filesystem = interface.startswith("fs.")
    reads = ["process.cwd"] if filesystem else []
    writes = []
    if filesystem:
        reads.append("filesystem")
        if interface not in {"fs.exists", "fs.is_file", "fs.is_dir"}:
            writes.append("filesystem")
    if interface == "process.chdir":
        writes.append("process.cwd")
    if interface == "process.getcwd":
        reads.append("process.cwd")
    if interface.startswith("stream."):
        reads.append("stream.state")
        writes.append("stream.state")
        if interface == "stream.open":
            reads += ["process.cwd", "filesystem"]
            if args[1].exact[:1] != "r":
                writes.append("filesystem")
        elif interface in {"stream.write", "stream.flush", "stream.close"}:
            writes.append("filesystem")
        else:
            reads.append("filesystem")
    if interface.startswith("time."):
        reads.append("host.clock")
    if interface == "console.input":
        reads.append("stdin.state")
        writes += ["stdin.state", "stdout.state", "stderr.state"]
    if interface == "process.exit":
        writes.append("process.completion")
    effects = ["VAL"] if interface == "path.join" else ["IO", "STATE"]
    return frame.effect(
        dict(
            op="host_call",
            interface=interface,
            semantics_version=1,
            args=[frame.scalar(value, node) for value in args],
            type=wire(result),
            effects=effects,
            reads=reads,
            writes=writes,
            may_raise=True,
            **source_site(node),
            **argument_evidence(args),
            **({"result_facts": result_facts} if result_facts else {}),
            **({"noreturn": True} if interface == "process.exit" else {}),
        ),
        result,
        steps,
        discard,
        facts=result_facts,
    )


def _open(frame, node, target, args, keywords, steps, discard):
    parameters = [
        ("file", "text", _REQUIRED),
        ("mode", "text", "r"),
        ("buffering", "i64", -1),
        ("encoding", ("unit", "text"), None),
        ("errors", ("unit", "text"), None),
        ("newline", ("unit", "text"), None),
        ("closefd", "bool", True),
        ("opener", "unit", None),
    ]
    bound = _bind(frame, node, args, keywords, parameters)
    path, mode, buffering, encoding, errors, newline, closefd, opener = bound
    if mode.exact not in {"r", "rt", "w", "wt", "a", "at", "x", "xt"}:
        frame.fail(
            node, "stream_mode", "text stream mode must be a closed r/w/a/x mode"
        )
    default_encoding = target.facts.get("default_text_encoding", "")
    encoding_name = default_encoding if encoding.kind == "unit" else encoding.exact
    if not isinstance(encoding_name, str) or encoding_name.lower().replace(
        "-", ""
    ) not in {"utf8", "utf_8"}:
        frame.fail(
            node, "stream_encoding", "text streams require a proven UTF-8 encoding"
        )
    if (
        buffering.exact != -1
        or errors.kind != "unit"
        and errors.exact != "strict"
        or newline.kind != "unit"
        or closefd.exact is not True
        or opener.kind != "unit"
    ):
        frame.fail(
            node,
            "stream_options",
            "nondefault buffering, errors, newline or opener requires another stream contract",
        )
    return _emit(
        frame,
        node,
        "stream.open",
        _STREAM,
        [path, mode, literal(encoding_name), literal("universal")],
        steps,
        discard,
        result_facts={"stream_origin": {"kind": "named_text_file"}},
    )


def invoke(frame, node, target, args, keywords, steps, discard):
    if target.kind == "host_method":
        receiver, method = target.value
        interface, result, parameters = _METHODS[method]
        values = _bind(frame, node, args, keywords, parameters)
        return _emit(
            frame, node, interface, result, [receiver, *values], steps, discard
        )
    if target.kind != "host_callable":
        return None
    name = target.value
    if name.startswith("math."):
        return scalars.math_call(
            frame, node, name.removeprefix("math."), args, keywords, steps, discard
        )
    if name == "builtins.open":
        return _open(frame, node, target, args, keywords, steps, discard)
    if name == "builtins.input":
        return _input(frame, node, target, args, keywords, steps, discard)
    if name == "sys.exit":
        return _exit(frame, node, args, keywords, steps, discard)
    if name in {"builtins.str", "builtins.repr", "builtins.ascii", "builtins.format"}:
        builtin_name = name.removeprefix("builtins.")
        if builtin_name == "str":
            if not args and set(keywords) == {"object"}:
                args, keywords = [keywords["object"]], {}
            if not args and not keywords:
                return literal("")
        arities = {1, 2} if builtin_name == "format" else {1}
        if keywords or len(args) not in arities:
            frame.fail(
                node,
                "text_arguments",
                f"{builtin_name} needs its scalar positional signature",
            )
        conversion = "none" if builtin_name == "format" else builtin_name
        spec = args[1] if len(args) == 2 else literal("")
        return text.render(frame, node, args[0], conversion, spec, steps, discard)
    if name == "os.path.join":
        if not args or keywords or any(v.kind != "text" for v in args):
            frame.fail(
                node,
                "path_arguments",
                "path.join requires positional text path components",
            )
        facts = {}
        origin = target.facts.get("aliases", {}).get("os.path", {})
        if (
            origin.get("module") == "posixpath"
            and origin.get("status") == "verified_stdlib"
            and origin.get("sha256")
            and all(type(v.exact) is str for v in args)
        ):
            # POSIX join resets at each absolute component and preserves a final
            # empty component. It neither normalizes dots nor consults the filesystem.
            joined = args[0].exact
            for value in args[1:]:
                part = value.exact
                if part.startswith("/"):
                    joined = part
                else:
                    joined += (
                        "/" if joined and not joined.endswith("/") else ""
                    ) + part
            facts["exact"] = joined
        return _emit(
            frame, node, "path.join", "text", args, steps, discard, result_facts=facts
        )
    interface, result, parameters = _SIGNATURES[name]
    values = _bind(frame, node, args, keywords, parameters)
    if name in {"os.mkdir", "os.makedirs"}:
        if values[1].exact != 0o777:
            frame.fail(
                node,
                "filesystem_mode",
                "nondefault directory permissions need an explicit mode contract",
            )
        if name == "os.mkdir" and values[2].kind != "unit":
            frame.fail(
                node,
                "filesystem_directory",
                "relative directory descriptors need a separate filesystem contract",
            )
        values = values[:1] if name == "os.mkdir" else [values[0], values[2]]
    if name in {"shutil.copyfile", "shutil.copy", "shutil.copy2"}:
        if values[-1].exact is not True:
            frame.fail(
                node,
                "filesystem_links",
                "non-following symbolic links need a separate copy contract",
            )
        values = values[:2]
    return _emit(frame, node, interface, result, values, steps, discard)


def _format(frame, node, value, steps, discard):
    return frame.effect(
        dict(
            op="text_format",
            value=frame.scalar(value, node),
            type="text",
            scalar_format=text.SCALAR_FORMAT,
        ),
        "text",
        steps,
        discard,
    )


def _input(frame, node, target, args, keywords, steps, discard):
    if keywords or len(args) > 1 or args and args[0].kind != "text":
        frame.fail(
            node,
            "console_arguments",
            "input requires at most one positional text prompt",
        )
    facts = target.facts
    encodings = [facts.get(key, "") for key in ("stdin_encoding", "stdout_encoding")]
    if any(
        not isinstance(value, str)
        or value.lower().replace("-", "") not in {"utf8", "utf_8"}
        for value in encodings
    ):
        frame.fail(
            node,
            "console_encoding",
            "console input requires proven UTF-8 input and output encodings",
        )
    errors = facts.get("stdin_errors")
    if (
        errors not in {"strict", "surrogateescape"}
        or errors == "surrogateescape"
        and facts.get("stdout_errors") != errors
    ):
        frame.fail(
            node,
            "console_encoding",
            "the input error policy requires a compatible native text-output contract",
        )
    return _emit(
        frame,
        node,
        "console.input",
        "text",
        [args[0] if args else literal(""), literal(errors)],
        steps,
        discard,
    )


def _exit(frame, node, args, keywords, steps, discard):
    if keywords or len(args) > 1:
        frame.fail(
            node, "exit_arguments", "exit requires at most one positional status"
        )
    value = args[0] if args else literal(None)
    if value.kind == "unit":
        status, message, output = literal(0), literal(""), literal(None)
    elif value.kind in {"i64", "bool"}:
        status = (
            value
            if value.kind == "i64"
            else Value(
                "i64", dict(op="cast", type="i64", value=value.value), value.facts
            )
        )
        message = _format(frame, node, value, steps, False)
        output = literal(None)
    elif value.kind == "text":
        status, message, output = literal(1), value, value
    else:
        frame.fail(
            node,
            "exit_arguments",
            "exit status requires a closed scalar display and integer-code contract",
        )
    error = frame.effect(
        dict(
            op="error_make",
            category="system_exit",
            message=message.value,
            exit_code=status.value,
            exit_message=output.value,
            type=wire("handle:error"),
        ),
        "handle:error",
        steps,
    )
    return _emit(frame, node, "process.exit", "unit", [error], steps, discard)

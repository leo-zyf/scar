"""F2 conditional IO projections referencing existing call operands by path."""

import ast

STREAM_METHODS = {"write", "read", "readline", "close", "flush"}
_REQUIRED = object()
_SIGNATURES = {
    "builtins.open": (
        "stream.open",
        (
            ("path", "file", 0, _REQUIRED),
            ("mode", "mode", 1, "r"),
            ("buffering", "buffering", 2, -1),
            ("encoding", "encoding", 3, None),
            ("errors", "errors", 4, None),
            ("newline", "newline", 5, None),
            ("closefd", "closefd", 6, True),
            ("opener", "opener", 7, None),
        ),
        ("source_open_identity", "path_resolution", "open_mode_and_encoding"),
    ),
    "os.chdir": (
        "process.chdir",
        (("path", "path", 0, _REQUIRED),),
        ("path_resolution", "directory_access_success"),
    ),
    "shutil.copyfile": (
        "fs.copy_file",
        (
            ("src", "src", 0, _REQUIRED),
            ("dst", "dst", 1, _REQUIRED),
            ("follow_symlinks", "follow_symlinks", None, True),
        ),
        ("source_and_destination_locator", "copy_contents_and_metadata"),
    ),
    "shutil.copy": (
        "fs.copy",
        (
            ("src", "src", 0, _REQUIRED),
            ("dst", "dst", 1, _REQUIRED),
            ("follow_symlinks", "follow_symlinks", None, True),
        ),
        ("source_and_destination_locator", "copy_contents_and_metadata"),
    ),
    "shutil.copy2": (
        "fs.copy2",
        (
            ("src", "src", 0, _REQUIRED),
            ("dst", "dst", 1, _REQUIRED),
            ("follow_symlinks", "follow_symlinks", None, True),
        ),
        ("source_and_destination_locator", "copy_contents_and_metadata"),
    ),
    "shutil.copytree": (
        "fs.copy_tree",
        (
            ("src", "src", 0, _REQUIRED),
            ("dst", "dst", 1, _REQUIRED),
            ("symlinks", "symlinks", 2, False),
            ("ignore", "ignore", 3, None),
            (
                "copy_function",
                "copy_function",
                4,
                {"symbol": "shutil.copy2", "binding": "definition_default"},
            ),
            ("ignore_dangling_symlinks", "ignore_dangling_symlinks", 5, False),
            ("dirs_exist_ok", "dirs_exist_ok", 6, False),
        ),
        (
            "source_and_destination_locator",
            "copy_contents_and_metadata",
            "recursive_tree_membership",
            "source_default_bindings",
        ),
    ),
}


def _argument(syntax, fields, role, name, position, default, effect):
    keywords = [
        index
        for index, item in enumerate(fields["keywords"])
        if name is not None and item["name"] == name
    ]
    if len(keywords) == 1:
        effect["operands"][role] = ["keywords", keywords[0], "value"]
    elif (
        position is not None
        and len(syntax.args) > position
        and not any(
            isinstance(item, ast.Starred) for item in syntax.args[: position + 1]
        )
    ):
        effect["operands"][role] = ["args", position]
    elif (
        not any(isinstance(item, ast.Starred) for item in syntax.args)
        and not any(item.arg is None for item in syntax.keywords)
        and default is not _REQUIRED
    ):
        effect["defaults"][role] = (
            dict(default) if isinstance(default, dict) else default
        )
    else:
        effect["obligations"].append("argument_binding:" + role)


def project(symbol, syntax, fields, *, providers, stream_handles=()):
    """Describe a candidate effect; unresolved guards never prove it occurred."""
    stream_method = (
        symbol[len("io.IOBase.") :] if symbol.startswith("io.IOBase.") else None
    )
    if stream_method in STREAM_METHODS:
        if not stream_handles:
            return None
        interface = "stream." + stream_method
        signature = (
            (("data", None, 0, _REQUIRED),)
            if stream_method == "write"
            else (
                (("size", "size", 0, -1),)
                if stream_method in {"read", "readline"}
                else ()
            )
        )
        obligations = (
            "bound_stream_receiver",
            "stream_instance_identity",
            "stream_mode_and_encoding",
        )
    elif symbol == "builtins.print":
        interface = "text.print"
        signature = (
            ("file", "file", None, None),
            ("sep", "sep", None, " "),
            ("end", "end", None, "\n"),
            ("flush", "flush", None, False),
        )
        obligations = ("stream_destination_identity", "source_text_formatting")
    elif symbol == "os.path.join":
        interface, signature = "path.join", ()
        obligations = ("source_path_protocols",)
    elif symbol in _SIGNATURES:
        interface, signature, obligations = _SIGNATURES[symbol]
    else:
        return None
    root = symbol.partition(".")[0]
    if (
        root in {"os", "shutil"}
        and providers.get(root, {}).get("status") != "verified_stdlib"
    ):
        return None
    effect = dict(
        semantics_version=1,
        interface=interface,
        operands={},
        defaults={},
        obligations=[
            "target_binding_guard",
            "call_argument_binding",
            "normal_completion",
            *obligations,
        ],
    )
    for spec in signature:
        _argument(syntax, fields, *spec, effect)
    if stream_method in STREAM_METHODS:
        target = fields["target"]
        direct = target.get("op") == "member" and "base" in target
        operand = ["target", "base"] if direct else ["target"]
        effect["operands"]["stream"] = operand
        fields["stream_handle"] = dict(
            operand=operand,
            binding="identity" if direct else "evaluated_bound_method",
            sites=list(stream_handles),
            instance="receiver_at_this_invocation",
        )
    elif symbol == "builtins.print":
        effect["operands"]["values"] = ["args"]
    elif symbol == "os.path.join":
        effect["operands"]["parts"] = ["args"]
        if syntax.keywords:
            effect["obligations"].append("path_keyword_binding")
    return effect

"""F2 turns sealed build facts into compilation profiles and artifact dependencies.

This layer does not inspect a checkout or execute build commands. File identities
come from F1 receipts; paths absent from those receipts remain declared paths.
"""

from hashlib import sha256
import json
import posixpath
import re

_SOURCE_LANGUAGES = {
    ".c": "c",
    ".C": "c++",
    ".cc": "c++",
    ".cp": "c++",
    ".cpp": "c++",
    ".cxx": "c++",
    ".c++": "c++",
    ".f": "fortran",
    ".for": "fortran",
    ".ftn": "fortran",
    ".f77": "fortran",
    ".f90": "fortran",
    ".f95": "fortran",
    ".f03": "fortran",
    ".f08": "fortran",
    ".f18": "fortran",
}
_DRIVER = re.compile(
    r"(?:^|-)(?:gcc|g\+\+|gfortran|clang(?:\+\+)?|cc|c\+\+)" r"(?:-\d+(?:\.\d+)*)?$"
)
_PATH_OPTIONS = {
    "-I",
    "-isystem",
    "-iquote",
    "-idirafter",
    "-include",
    "-imacros",
    "-isysroot",
    "--sysroot",
    "-B",
}
_VALUE_OPTIONS = {
    "-o",
    "-x",
    "-D",
    "-U",
    "-L",
    "-l",
    "-MF",
    "-MT",
    "-MQ",
    "-J",
    "-B",
    "-Xlinker",
    "-Xassembler",
    "-Xpreprocessor",
    *_PATH_OPTIONS,
}
_DEPENDENCY_OPTIONS = {"-M", "-MM", "-MD", "-MMD", "-MP", "-MG"}


def _identity(kind, value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "build:" + kind + ":" + sha256(raw).hexdigest()[:24]


def _path(name, directory):
    return posixpath.normpath(
        name if posixpath.isabs(name) else posixpath.join(directory, name)
    )


def _suffix(path):
    suffix = posixpath.splitext(path)[1]
    return suffix if suffix == ".C" else suffix.lower()


def _option_path(value, option, directory):
    # GCC interprets these prefixes relative to its sysroot, not the build cwd.
    if option in {"-I", "-isystem", "-iquote", "-idirafter"} and value.startswith(
        ("=", "$SYSROOT")
    ):
        return value
    return _path(value, directory)


def _operands(arguments):
    """Split driver positionals without mistaking option arguments for inputs."""
    result, i = [], 1
    while i < len(arguments):
        token = arguments[i]
        if token == "--":
            result.extend(arguments[i + 1 :])
            break
        if token in _VALUE_OPTIONS:
            i += 2
        else:
            if not token.startswith("-"):
                result.append(token)
            i += 1
    return result


def _option(arguments, option):
    value = None
    for i, token in enumerate(arguments[1:], 1):
        if token == option and i + 1 < len(arguments):
            value = arguments[i + 1]
        elif token.startswith(option) and len(token) > len(option):
            value = token[len(option) :]
    return value


def _language(source, arguments):
    declared = _option(arguments, "-x")
    if declared and declared != "none":
        return {
            "c": "c",
            "c++": "c++",
            "f77": "fortran",
            "f77-cpp-input": "fortran",
            "f95": "fortran",
            "f95-cpp-input": "fortran",
        }.get(declared)
    if _suffix(source) in {
        ".c",
        ".cc",
        ".cp",
        ".cpp",
        ".cxx",
        ".c++",
        ".C",
    } and re.search(
        r"(?:^|-)(?:g\+\+|clang\+\+|c\+\+)(?:-\d+(?:\.\d+)*)?$",
        posixpath.basename(arguments[0]),
    ):
        return "c++"
    return _SOURCE_LANGUAGES.get(_suffix(source))


def _profile_flags(arguments, directory):
    result, i = [], 1
    while i < len(arguments):
        token = arguments[i]
        if token in {"-o", "-MF", "-MT", "-MQ", "-L", "-l", "-Xlinker"}:
            i += 2
            continue
        if token in _DEPENDENCY_OPTIONS or token in {"-c", "--"}:
            i += 1
            continue
        if any(token.startswith(p) for p in ("-Wl,", "-MF", "-MT", "-MQ")):
            i += 1
            continue
        if token.startswith("-o") and len(token) > 2:
            i += 1
            continue
        if token.startswith(("-L", "-l")):
            i += 1
            continue
        if token in _PATH_OPTIONS | {"-J"}:
            if i + 1 >= len(arguments):
                raise ValueError(f"missing build option argument: {token}")
            result.extend(
                [
                    "-I" if token == "-J" else token,
                    _option_path(arguments[i + 1], token, directory),
                ]
            )
            i += 2
            continue
        joined = next(
            (
                prefix
                for prefix in sorted(_PATH_OPTIONS | {"-J"}, key=len, reverse=True)
                if token.startswith(prefix) and len(token) > len(prefix)
            ),
            None,
        )
        if joined:
            value = token[len(joined) :]
            if joined == "--sysroot":
                value = value.removeprefix("=")
            prefix = "-I" if joined == "-J" else joined
            result.extend([prefix, _option_path(value, joined, directory)])
            i += 1
            continue
        if token in _VALUE_OPTIONS:
            if i + 1 >= len(arguments):
                raise ValueError(f"missing build option argument: {token}")
            result.extend(arguments[i : i + 2])
            i += 2
            continue
        if token.startswith("-"):
            result.append(token)
        i += 1
    return result


def normalize_build(facts, *, targets=None, entry="main", require_root=True):
    """Return F3-ready source profiles and a separately sealable build graph.

    The first explicit target is the root image. Without a selection, a capture
    containing several executables needs a root unless a containing project
    already supplies its controller image.
    """
    root = posixpath.normpath(facts["root"])
    files = facts.get("files", {})
    aliases = facts.get("path_aliases", {})
    nodes, edges, diagnostics, resources = {}, [], [], {}
    profile_table, profile_ids, objects, compilations = [], {}, {}, []
    links, archives, project_libraries = {}, {}, set()

    def canonical_path(name, directory):
        path = _path(name, directory)
        return aliases.get(path, path)

    def issue(code, detail, **attributes):
        diagnostics.append({"code": "build." + code, "detail": detail, **attributes})

    for row in facts.get("issues", []):
        issue(
            row["code"].removeprefix("build."),
            row["detail"],
            **{k: v for k, v in row.items() if k not in {"code", "detail"}},
        )

    def artifact(path, kind=None):
        identity = _identity("artifact", path)
        receipt = files.get(path)
        node = nodes.setdefault(
            identity,
            {"id": identity, "kind": "BUILD_ARTIFACT", "path": path},
        )
        if receipt:
            node.update(
                artifact_kind=receipt.get("kind", kind),
                receipt={k: receipt[k] for k in ("sha256", "size") if k in receipt},
                observed_at="capture",
            )
        elif kind:
            node.setdefault("artifact_kind", kind)
        return identity

    def edge(kind, source, target, **attributes):
        edges.append({"kind": kind, "source": source, "target": target, **attributes})

    def operation(row):
        identity = "build:invocation:" + str(row["id"])
        nodes.setdefault(identity, {"id": identity, "kind": "BUILD_OPERATION", **row})
        nodes[identity]["id"] = identity
        nodes[identity]["invocation_id"] = row["id"]
        executable = row.get("canonical_executable", row["executable"])
        tool = artifact(executable, "tool")
        edge("USES_TOOL", identity, tool)
        resource = _identity("tool", [executable, files.get(executable)])
        resources.setdefault(
            resource,
            {
                "id": resource,
                "kind": "TOOLCHAIN",
                "capacity": None,
                "attributes": {
                    "path": executable,
                    "artifact": tool,
                    "observed_at": "capture",
                },
            },
        )
        nodes.setdefault(
            resource, {"id": resource, "kind": "TOOLCHAIN", "artifact": tool}
        )
        edge("USES_RESOURCE", identity, resource)
        return identity

    def add_object(path, indices, producer):
        if path in objects and objects[path] != (indices, producer):
            issue(
                "multiple_producers",
                "artifact has several captured producers",
                path=path,
            )
        objects[path] = (indices, producer)

    for row in facts.get("compilations", []):
        directory = row["directory"]
        source = row.get("canonical_file", canonical_path(row["file"], directory))
        if not source.startswith(root.rstrip("/") + "/"):
            issue(
                "source_outside_root",
                "compilation source is outside source_root",
                path=source,
            )
            continue
        arguments = row["arguments"]
        if not _DRIVER.search(posixpath.basename(arguments[0])):
            issue(
                "compiler_provider",
                "source driver is outside the qualified GNU/Clang capture profile",
                path=source,
            )
            continue
        if any(token.startswith(("@", "-Wl,@")) for token in arguments[1:]):
            issue(
                "response_grammar",
                "response file arguments require a qualified decoder",
                path=source,
            )
            continue
        language = _language(source, arguments)
        if language is None:
            issue(
                "source_language",
                "no qualified source language for compilation",
                path=source,
            )
            continue
        try:
            flags = _profile_flags(arguments, directory)
        except ValueError as error:
            issue("compiler_options", str(error), path=source)
            continue
        profile = {
            "path": posixpath.relpath(source, root),
            "language": language,
            "flags": flags,
            "source_compiler": {
                "driver": arguments[0],
                "candidates": [],
                "observed_at": "capture_not_historical",
            },
        }
        driver = arguments[0]
        if "/" in driver:
            candidates = {canonical_path(driver, directory)}
        else:
            candidates = {
                row.get("canonical_executable")
                or canonical_path(row["executable"], directory)
                for row in facts.get("invocations", ())
                if row["directory"] == directory
                and posixpath.basename(row["executable"]) == driver
            }
        for path in sorted(candidates):
            receipt = files.get(path, {})
            profile["source_compiler"]["candidates"].append(
                dict(
                    path=path,
                    receipt={
                        key: receipt[key]
                        for key in ("sha256", "size")
                        if key in receipt
                    },
                )
            )
        key = json.dumps(profile, sort_keys=True)
        if key not in profile_ids:
            profile_ids[key] = len(profile_table)
            profile_table.append(profile)
        index = profile_ids[key]
        output = row.get("output") or _option(arguments, "-o")
        if output:
            output = canonical_path(output, directory)
        elif "-c" in arguments:
            output = canonical_path(
                posixpath.splitext(posixpath.basename(source))[0] + ".o", directory
            )
        else:
            output = canonical_path("a.out", directory)
        compilations.append(
            {
                "source": source,
                "output": output,
                "index": index,
                "arguments": arguments,
                "directory": directory,
            }
        )
        artifact(source, "source")

    for row in facts.get("invocations", []):
        invocation = operation(row)
        arguments, directory = row["arguments"], row["directory"]
        if not arguments:
            issue(
                "empty_arguments",
                "captured invocation has no argument vector",
                invocation=invocation,
            )
            continue
        if any(token.startswith(("@", "-Wl,@")) for token in arguments[1:]):
            for token in arguments[1:]:
                if token.startswith(("@", "-Wl,@")):
                    edge(
                        "BUILD_INPUT",
                        artifact(
                            canonical_path(token.partition("@")[2], directory),
                            "response",
                        ),
                        invocation,
                    )
            issue(
                "response_grammar",
                "response file dependencies are not decoded",
                invocation=invocation,
            )
            continue
        executable = posixpath.basename(row["executable"])
        if not _DRIVER.search(executable):
            if executable in {"ar", "llvm-ar", "gcc-ar"} or re.search(
                r"-(?:gcc-)?ar(?:-\d+)?$", executable
            ):
                operands = [v for v in arguments[1:] if not v.startswith("-")]
                if operands and re.fullmatch("[a-zA-Z]+", operands[0]):
                    operands = operands[1:]
                if operands:
                    path = canonical_path(operands[0], directory)
                    members = [canonical_path(v, directory) for v in operands[1:]]
                    archives[path] = (members, invocation)
                    edge("BUILD_OUTPUT", invocation, artifact(path, "archive"))
                    for ordinal, name in enumerate(members):
                        edge(
                            "ARCHIVE_MEMBER",
                            artifact(name),
                            invocation,
                            ordinal=ordinal,
                        )
            continue
        operands = [canonical_path(v, directory) for v in _operands(arguments)]
        sources = [p for p in operands if _suffix(p) in _SOURCE_LANGUAGES]
        output = _option(arguments, "-o")
        if output is None and "-c" in arguments and len(sources) == 1:
            output = posixpath.splitext(posixpath.basename(sources[0]))[0] + ".o"
        if output is None and "-c" not in arguments and (sources or operands):
            output = "a.out"
        if output is None or any(
            v in arguments for v in ("-E", "-S", "-fsyntax-only", "-M", "-MM")
        ):
            continue
        output = canonical_path(output, directory)
        matches = [
            c for c in compilations if c["source"] in sources and c["output"] == output
        ]
        environment = row.get("environment", {})
        unprojected = {
            key: environment[key]
            for key in (
                "CPATH",
                "C_INCLUDE_PATH",
                "CPLUS_INCLUDE_PATH",
                "OBJC_INCLUDE_PATH",
                "OBJCPLUS_INCLUDE_PATH",
                "COMPILER_PATH",
                "GCC_EXEC_PREFIX",
            )
            if environment.get(key)
        }
        if matches and unprojected:
            issue(
                "compiler_environment",
                "recorded include/tool search environment has not been projected into the source analysis profile",
                invocation=invocation,
                environment=unprojected,
            )
        for c in matches:
            edge("COMPILES", invocation, artifact(c["source"]), profile=c["index"])
        for ordinal, name in enumerate(operands):
            edge("BUILD_INPUT", artifact(name), invocation, ordinal=ordinal)
        edge("BUILD_OUTPUT", invocation, artifact(output))
        if "-c" in arguments:
            if sources and not matches:
                issue(
                    "missing_compilation",
                    "compiler event has no matching source profile",
                    invocation=invocation,
                )
            add_object(output, [c["index"] for c in matches], invocation)
        elif "-r" in arguments:
            archives[output] = (operands, invocation)
        elif "-shared" in arguments:
            project_libraries.add(output)
        elif "-shared" not in arguments:
            if output in links:
                issue(
                    "multiple_producers",
                    "executable has several captured links",
                    path=output,
                )
            links[output] = {
                "invocation": invocation,
                "inputs": operands,
                "direct_units": [c["index"] for c in matches],
                "arguments": arguments,
                "directory": directory,
            }
        _library_requests(
            arguments,
            directory,
            invocation,
            nodes,
            edges,
            resources,
            library_path=environment.get("LIBRARY_PATH"),
        )
        dependency = _option(arguments, "-MF")
        if dependency:
            path = canonical_path(dependency, directory)
            edge("DEPENDENCY_MANIFEST", artifact(path, "dependency"), invocation)
            text = files.get(path, {}).get("text")
            if text is not None:
                try:
                    for name in _dependency_inputs(text):
                        edge(
                            "DECLARED_BUILD_INPUT",
                            artifact(canonical_path(name, directory)),
                            invocation,
                            evidence=artifact(path, "dependency"),
                        )
                except ValueError as error:
                    issue("dependency_grammar", str(error), path=path)

    for row in facts.get("invocations", []):
        if parent := row.get("parent_invocation_id"):
            source, target = (
                "build:invocation:" + parent,
                "build:invocation:" + row["id"],
            )
            if source in nodes and target in nodes:
                edge("PROCESS_PARENT", source, target, evidence="unique_pid_lifetime")

    selected = []
    if targets:
        for target in targets:
            candidates = [
                p
                for p in links
                if target in {p, posixpath.relpath(p, root), posixpath.basename(p)}
            ]
            if len(candidates) != 1:
                issue(
                    "target_selection",
                    "target must identify exactly one captured executable",
                    target=target,
                    matches=candidates,
                )
            elif candidates[0] not in selected:
                selected.append(candidates[0])
    else:
        selected = sorted(links)
        if require_root and len(selected) > 1:
            issue(
                "ambiguous_root_image",
                "select the root executable explicitly",
                targets=selected,
            )
    if not selected:
        issue("missing_executable", "capture contains no selected executable link")

    images, used = [], []
    for path in selected:
        link = links[path]
        indices = []
        for request in _requested_libraries(link["arguments"]):
            candidates = [
                p
                for p in (*archives, *project_libraries)
                if posixpath.basename(p)
                in (
                    {request[1:]}
                    if request.startswith(":")
                    else {"lib" + request + ".a", "lib" + request + ".so"}
                )
            ]
            if candidates:
                issue(
                    "project_library_binding",
                    "source-built library needs an exact linked member/provider binding",
                    request=request,
                    candidates=candidates,
                    target=path,
                )
        for ordinal, name in enumerate(link["inputs"]):
            if name in objects:
                indices.extend(objects[name][0])
                edge("LINK_INPUT", artifact(name), link["invocation"], ordinal=ordinal)
            elif name in archives:
                issue(
                    "archive_selection",
                    "lazy archive member selection is not proven",
                    path=name,
                    target=path,
                )
                for item in archives[name][0]:
                    if item in objects:
                        indices.extend(objects[item][0])
                    else:
                        issue(
                            "missing_object_producer",
                            "archive member has no captured compilation",
                            path=item,
                        )
            elif name in project_libraries:
                issue(
                    "project_library_binding",
                    "source-built shared library needs an exact provider binding",
                    path=name,
                    target=path,
                )
            elif files.get(name, {}).get("kind") == "library":
                edge(
                    "EXTERNAL_LIBRARY_INPUT",
                    artifact(name),
                    link["invocation"],
                    ordinal=ordinal,
                )
            elif name not in {c["source"] for c in compilations}:
                issue(
                    "missing_object_producer",
                    "link input has no captured source compilation",
                    path=name,
                    target=path,
                )
        indices.extend(link["direct_units"])
        indices = list(dict.fromkeys(indices))
        if not indices:
            issue(
                "missing_image_units",
                "selected executable has no source profiles",
                path=path,
            )
            continue
        output = posixpath.basename(path)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", output) or any(
            i["output"] == output for i in images
        ):
            issue(
                "image_namespace",
                "executable basename is invalid or collides in the deployment namespace",
                path=path,
            )
            continue
        images.append(
            {
                "id": output,
                "output": output,
                "entry": entry,
                "units": indices,
                "build_artifact": artifact(path),
            }
        )
        used.extend(indices)

    for path, facts_row in facts.get("libraries", {}).items():
        output = artifact(path)
        for ordinal, soname in enumerate(facts_row.get("needed", [])):
            identity = _identity("elf-needed", [path, files.get(path), soname, ordinal])
            attrs = {
                "soname": soname,
                "artifact": output,
                "loaded": None,
                "observed_at": facts_row.get("observed_at", "capture"),
            }
            resources[identity] = {
                "id": identity,
                "kind": "ELF_LIBRARY_DEPENDENCY",
                "capacity": None,
                "attributes": attrs,
            }
            nodes[identity] = {
                "id": identity,
                "kind": "ELF_LIBRARY_DEPENDENCY",
                **attrs,
            }
            edge("ELF_NEEDED", output, identity, ordinal=ordinal)

    # A profile shared by several object outputs is parsed once; link order lives
    # on build edges, rather than duplicating the translation-unit graph.
    used = list(dict.fromkeys(used))
    remap = {old: new for new, old in enumerate(used)}
    for image in images:
        image["units"] = [remap[i] for i in image["units"]]
    sources = {
        p: files.get(p)
        for p in files
        if p.startswith(root.rstrip("/") + "/") and files[p].get("kind") == "source"
    }
    for index in used:
        path = canonical_path(profile_table[index]["path"], root)
        receipt = files.get(path)
        if not receipt or "sha256" not in receipt:
            issue(
                "source_receipt",
                "compilation source has no F1 content receipt",
                path=path,
            )
        sources[path] = receipt
    hashes = {
        posixpath.relpath(p, root): receipt["sha256"]
        for p, receipt in sources.items()
        if receipt and "sha256" in receipt
    }
    graph = {
        "provider": facts["provider"],
        "trace_sha256": facts["trace_sha256"],
        "nodes": list(nodes.values()),
        "edges": edges,
        "diagnostics": diagnostics,
        "compilation_profiles": profile_table,
        "tools": facts.get("tools", {}),
        "coverage": {
            "runtime_library_loading": "not_observed",
            "artifact_receipts": "capture_time_not_historical",
            "generated_input_history": "not_observed",
        },
    }
    if "process_observation" in facts:
        graph["process_observation"] = facts["process_observation"]
    if "process_diagnostics" in facts:
        graph["process_diagnostics"] = facts["process_diagnostics"]
    return {
        "profiles": [profile_table[i] for i in used],
        "images": images,
        "root_image": images[0]["id"] if images else None,
        "build_graph": graph,
        "source_hashes": hashes,
        "software_resources": list(resources.values()),
    }


def _dependency_inputs(text):
    """Decode literal GNU dependency rules; do not evaluate Make expressions."""
    text = text.replace("\\\r\n", "").replace("\\\n", "")
    for rule in text.splitlines():
        token, words, colon, escaped, i = "", [], False, False, 0
        while i < len(rule):
            ch = rule[i]
            if escaped:
                token += ch
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "#":
                break
            elif ch == "$":
                if i + 1 >= len(rule) or rule[i + 1] != "$":
                    raise ValueError(
                        "dependency file contains an unevaluated Make expression"
                    )
                token += "$"
                i += 1
            elif ch == ":" and not colon:
                token, words, colon = "", [], True
            elif ch.isspace():
                if token:
                    words.append(token)
                    token = ""
            else:
                token += ch
            i += 1
        if escaped:
            raise ValueError("dependency file ends in an unfinished escape")
        if token:
            words.append(token)
        if colon:
            yield from words
        elif words:
            raise ValueError("dependency record has no target separator")


def _library_requests(
    arguments, directory, invocation, nodes, edges, resources, *, library_path=None
):
    """Preserve ordered link requirements without claiming resolved binaries."""
    search, requests, i = [], [], 1
    if library_path:
        identity = _identity("library-search", [invocation, library_path])
        attrs = {
            "environment_variable": "LIBRARY_PATH",
            "value": library_path,
            "paths": [_path(p, directory) for p in library_path.split(":")],
            "invocation": invocation,
            "evidence": "recorded_environment",
        }
        resources[identity] = {
            "id": identity,
            "kind": "LIBRARY_SEARCH_CONTEXT",
            "capacity": None,
            "attributes": attrs,
        }
        nodes[identity] = {"id": identity, "kind": "LIBRARY_SEARCH_CONTEXT", **attrs}
        edges.append(
            {"kind": "USES_LIBRARY_SEARCH", "source": invocation, "target": identity}
        )
    while i < len(arguments):
        token = arguments[i]
        if token in {"-L", "-l", "-Xlinker"} and i + 1 < len(arguments):
            value = arguments[i + 1]
            if token == "-L":
                search.append(_path(value, directory))
            else:
                requests.append([token, value])
            i += 2
            continue
        if token.startswith("-L"):
            search.append(_path(token[2:], directory))
        elif token.startswith(("-l", "-Wl,")):
            requests.append([token])
        i += 1
    for ordinal, request in enumerate(requests):
        identity = _identity("library-request", [invocation, ordinal, request, search])
        attrs = {
            "arguments": request,
            "search_paths": search,
            "invocation": invocation,
            "resolved_binary": None,
            "loaded": None,
            "evidence": "driver_arguments",
        }
        resources[identity] = {
            "id": identity,
            "kind": "LIBRARY_REQUEST",
            "capacity": None,
            "attributes": attrs,
        }
        nodes[identity] = {"id": identity, "kind": "LIBRARY_REQUEST", **attrs}
        edges.append(
            {
                "kind": "REQUESTS_LIBRARY",
                "source": invocation,
                "target": identity,
                "ordinal": ordinal,
            }
        )


def _requested_libraries(arguments):
    """Find explicit driver library names without resolving search paths."""
    args = iter(arguments[1:])
    for token in args:
        if token == "-l":
            value = next(args, None)
            if value is not None:
                yield value
        elif token.startswith("-l"):
            yield token[2:]
        elif token.startswith("-Wl,"):
            linker = iter(token[4:].split(","))
            for option in linker:
                if option == "-l":
                    value = next(linker, None)
                    if value is not None:
                        yield value
                elif option.startswith("-l"):
                    yield option[2:]

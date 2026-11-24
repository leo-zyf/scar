"""M2 file-content provenance over already-qualified canonical operations."""

from copy import deepcopy
from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
import posixpath

from scar.contract import (
    ARTIFACT_FLOW,
    byte_value_id,
    digest,
    document,
    json_bytes,
    validate_artifact_flow,
    walk_operations,
)
from .values import (
    identical,
    result_summaries,
    apply_result,
    scalar_spelling,
    exact_scalar,
    evidence_scalar,
    _UNKNOWN,
)

_FILE_FREE_OPERATIONS = {
    "literal",
    "string",
    "zero",
    "var",
    "unary",
    "binary",
    "cast",
    "index",
    "member",
    "select",
    "intrinsic",
    "function_ref",
    "sizeof",
    "alignof",
    "decl",
    "assign",
    "update",
    "expr",
    "return",
    "block",
    "if",
    "for",
    "while",
    "do_while",
    "map",
    "each",
    "switch",
    "case",
    "break",
    "continue",
    "pass",
    "try",
    "exception_handler",
    "throw",
    "raise",
    "error_make",
    "process_error_make",
    "aggregate",
    "buffer",
    "runtime_array",
    "allocate",
    "release",
    "load_bytes",
    "store_bytes",
    "text_concat",
    "text_format",
    "array_create",
    "array_iota",
    "array_literal",
    "array_copy",
    "array_broadcast_copy",
    "array_unary",
    "array_binary",
    "array_view",
    "array_reshape",
    "array_store",
    "array_update",
    "array_load",
    "array_shape",
    "array_dtype",
    "array_broadcast",
    "array_comparison",
    "array_power_dispatch",
}


@dataclass
class _Stream:
    identity: str
    path: object
    mode: object
    open_operation: str
    chunks: list = field(default_factory=list)
    writes: list = field(default_factory=list)
    closed: bool = False
    unresolved: set = field(default_factory=set)
    locator: str | None = None
    path_key: tuple | None = None
    last_operation: str | None = None


def analyze(module):
    """Preserve proven normal-path bytes without authorizing an effect rewrite.

    A constant byte value is independent of file identity and history. Equality
    with a parser snapshot therefore joins content, never a historical write to
    a read. Loops and incomplete control regions stay summarized, not unrolled.
    """
    original = module.get("resource_model", {}).get("artifact_flow")
    graph = (
        deepcopy(original)
        if original
        else document(
            ARTIFACT_FLOW,
            nodes=[],
            edges=[],
            source_effects=[],
            open_fields=[],
            coverage={},
        )
    )
    nodes = {node["id"]: node for node in graph["nodes"]}
    edges, materializations, boundaries, templates = graph["edges"], [], [], []
    file_transitions = []
    source_operations = {
        str(node["graph_id"]) + "/" + str(node["port"]): node["id"]
        for node in nodes.values()
        if node["kind"] == "SOURCE_OPERATION"
    }
    candidates = [
        dict(symbol=fn["name"], completion="complete", function=fn)
        for fn in module.get("functions", ())
        if fn.get("execution_kind") != "python_protocol"
    ]
    protocol = module.get("protocol_analysis", {})
    candidates.extend(protocol.get("closed_fragments", ()))
    candidates.extend(
        {**fragment, "function": dict(name=fragment["symbol"], body=fragment["body"])}
        for fragment in protocol.get("prefix_fragments", ())
    )
    functions = {
        fragment["symbol"]: fragment["function"]
        for fragment in candidates
        if fragment["completion"] == "complete"
    }
    stable_globals = set(protocol.get("single_bind_globals", ()))
    summaries = result_summaries(functions)
    file_calls, file_effects, dependents = {}, {}, defaultdict(set)
    summary_operations = 0
    for name, fn in functions.items():
        dependencies, effects = set(), set()
        for node in walk_operations(fn["body"]):
            summary_operations += 1
            op = node["op"]
            if op == "call" and node.get("callee") in functions:
                dependencies.add(node["callee"])
            elif op == "host_call":
                interface = node["interface"]
                if (
                    interface.startswith(("fs.", "stream.", "process."))
                    or interface == "console.input"
                ):
                    effects.add("FILE")
                elif interface not in {"path.join", "time.wall", "time.sleep"}:
                    effects.add("OPAQUE")
            elif op in {"text_print", "process_spawn", "process_request"}:
                effects.add("FILE")
            elif op in {
                "call",
                "indirect_call",
                "library_call",
                "object_construct",
                "object_new",
                "object_destroy",
                "object_dispatch",
                "unsupported",
                "opaque",
            }:
                effects.add("OPAQUE")
            elif (
                {
                    "filesystem",
                    "stream.state",
                    "process.cwd",
                    "external.state",
                }.intersection(node.get("reads", ()))
                or {
                    "filesystem",
                    "stream.state",
                    "process.cwd",
                    "external.state",
                }.intersection(node.get("writes", ()))
                or node.get("external_effect") not in (None, "NONE")
                or "OPAQUE" in node.get("effects", ())
                or op not in _FILE_FREE_OPERATIONS
            ):
                effects.add("OPAQUE")
        file_calls[name], file_effects[name] = dependencies, effects
        for callee in dependencies:
            dependents[callee].add(name)
    # Iterative Kosaraju: include every member of a recursive component,
    # including cross edges into an already finished DFS branch. Neither
    # traversal depends on Python's recursion limit or unrolls program calls.
    inspected, finished = set(), []
    for root in functions:
        pending_calls = [(root, False)]
        while pending_calls:
            name, expanded = pending_calls.pop()
            if expanded:
                finished.append(name)
            elif name not in inspected:
                inspected.add(name)
                pending_calls.append((name, True))
                pending_calls.extend((callee, False) for callee in file_calls[name])
    recursive, assigned = set(), set()
    for root in reversed(finished):
        component, pending_calls = set(), [root]
        while pending_calls:
            name = pending_calls.pop()
            if name not in assigned:
                assigned.add(name)
                component.add(name)
                pending_calls.extend(dependents[name])
        if component and (len(component) > 1 or root in file_calls[root]):
            recursive.update(component)
    for name in recursive:
        file_effects[name].add("OPAQUE")
    work = deque(name for name, effects in file_effects.items() if effects)
    while work:
        callee = work.popleft()
        for caller in dependents[callee]:
            merged = file_effects[caller] | file_effects[callee]
            if merged != file_effects[caller]:
                file_effects[caller] = merged
                work.append(caller)
    file_contexts = set()
    file_call_counts = dict(non_file=0, opaque_barrier=0, stateful=0)
    # One abstract input tuple per static call port. Each scalar formal moves
    # monotonically from a constant to unknown on disagreement, so recursive or
    # polymorphic input paths cannot create an unbounded set of value contexts.
    context_inputs, queued = {}, set()

    def add(kind, key, **fields):
        identity = (
            byte_value_id(key)
            if kind == "BYTE_VALUE"
            else "artifact:" + kind.lower() + ":" + digest(json_bytes(key))
        )
        value = dict(id=identity, kind=kind, **fields)
        if identity not in nodes:
            nodes[identity] = value
        elif kind != "BYTE_VALUE" and nodes[identity] != value:
            raise ValueError("conflicting canonical artifact identity")
        return identity

    pending = deque(candidates)

    def bind_context(callee, site, arguments, activation):
        key = callee["name"], site
        previous = context_inputs.get(key)
        joined = (
            arguments
            if previous is None
            else [
                left if identical(left, right) else _UNKNOWN
                for left, right in zip(previous, arguments)
            ]
        )
        if previous is not None and all(
            identical(left, right) for left, right in zip(previous, joined)
        ):
            return
        context_inputs[key] = joined
        stamp = json_bytes(
            [
                key,
                [
                    (
                        dict(exact=value)
                        if value is not _UNKNOWN
                        else dict(unresolved=True)
                    )
                    for value in joined
                ],
            ]
        )
        if stamp not in queued:
            queued.add(stamp)
            pending.append(
                dict(
                    symbol=callee["name"],
                    completion="complete",
                    function=callee,
                    bindings=joined,
                    call_site=site,
                    activation=activation,
                )
            )

    # A qualified call fact instantiates a conditional summary even when the
    # caller's preceding control region remains open. It does not assert that
    # a branch was taken or a loop actually executed.
    for fragment in candidates:
        for call in walk_operations(fragment["function"]["body"]):
            callee = functions.get(call.get("callee")) if call["op"] == "call" else None
            evidence = call.get("argument_facts", {}).get("args")
            if (
                callee is None
                or not evidence
                or len(evidence) != len(callee.get("params", ()))
                or not isinstance(call.get("site"), str)
            ):
                continue
            arguments = []
            for argument, fact, parameter in zip(
                call["args"], evidence, callee["params"]
            ):
                value = evidence_scalar(fact, parameter["type"])
                if argument.get("op") == "literal" and not identical(
                    argument.get("value"), value
                ):
                    value = _UNKNOWN
                arguments.append(value)
            if any(value is not _UNKNOWN for value in arguments):
                bind_context(
                    callee,
                    call["site"],
                    arguments,
                    dict(
                        owner=fragment["symbol"],
                        site=call["site"],
                        condition="source_call_reached_and_argument_facts_hold",
                    ),
                )

    def evaluate_fragment(fragment, incoming=None, call_chain=()):
        function, completion = fragment["function"], fragment["completion"]
        symbol, streams, environment, ordinal = fragment["symbol"], [], {}, 0
        bindings = fragment.get("bindings", ())
        context = [
            symbol,
            fragment.get("call_site"),
            [
                dict(exact=v) if v is not _UNKNOWN else dict(unresolved=True)
                for v in bindings
            ],
        ]
        if "file_state_context" in fragment:
            context.append(dict(file_state_context=fragment["file_state_context"]))
        for parameter, value in zip(function.get("params", ()), bindings):
            environment[None, parameter["name"]] = exact_scalar(
                value, parameter["type"]
            )
        state = incoming or {}
        fact_epoch_clean = state.get("fact_epoch_clean", True)
        live_process_effects = state.get("live_process_effects", False)
        cwd = state.get("cwd", ("invocation.cwd", "."))
        file_versions, filesystem_epoch = dict(
            state.get("file_versions", {})
        ), state.get("epoch", 0)
        last_file_operation, operation_ids = state.get("last_operation"), {}
        root_activation = state.get("root_activation", digest(json_bytes(context)))
        return_value = _UNKNOWN

        def operation(node):
            nonlocal ordinal
            if id(node) in operation_ids:
                return operation_ids[id(node)]
            ordinal += 1
            site = node.get("site")
            if site in source_operations:
                return source_operations[site]
            identity = add(
                "CANONICAL_OPERATION",
                [symbol, completion, site, ordinal, context],
                symbol=symbol,
                site=site,
                operation=node["op"],
                instance_scope="per_dynamic_invocation_and_iteration",
            )
            operation_ids[id(node)] = identity
            return identity

        def ordered(site):
            nonlocal last_file_operation
            if last_file_operation is not None and last_file_operation != site:
                edges.append(
                    dict(
                        kind="FILESYSTEM_ORDER_IF_NORMAL",
                        source=last_file_operation,
                        target=site,
                        context=context,
                    )
                )
            last_file_operation = site

        def invalidate(reason, site=None, *, changes_cwd=False, retain=()):
            nonlocal filesystem_epoch, cwd
            filesystem_epoch += 1
            if site is not None:
                ordered(site)
                for key, version in file_versions.items():
                    if key in retain:
                        continue
                    edges.append(
                        dict(
                            kind="MAY_INVALIDATE_FILE_VERSION",
                            source=site,
                            target=version,
                            reason=reason,
                            epoch=filesystem_epoch,
                        )
                    )
            for key in list(file_versions):
                if key not in retain:
                    del file_versions[key]
            if changes_cwd:
                cwd = None
            for stream in streams:
                if not stream.closed:
                    stream.unresolved.add(reason)

        def locate(path, site):
            key = None
            # Resolving '..' through symlinks requires directory identity,
            # not string normalization. Keep it as a field-level boundary.
            if type(path) is str and "\0" not in path and ".." not in path.split("/"):
                if posixpath.isabs(path):
                    key = (None, posixpath.normpath(path))
                elif cwd is not None:
                    key = (cwd[0], posixpath.normpath(posixpath.join(cwd[1], path)))
            location = dict(
                path=key[1] if key is not None else None,
                cwd=key[0] if key is not None else "unresolved.cwd",
                activation=root_activation,
            )
            locator = add(
                "FILE_LOCATOR",
                [location, site if key is None else None],
                **location,
                identity_scope="normal_path_expression",
                inode_aliases="unresolved",
            )
            if key is None:
                boundaries.append(
                    dict(
                        symbol=symbol,
                        field="file_locator",
                        source=site,
                        reason="path_or_cwd_resolution_unproven",
                    )
                )
            return key, locator

        def requires(version, site, **fields):
            edges.append(
                dict(
                    kind="REQUIRES_FILE_VERSION",
                    source=version,
                    target=site,
                    conditional=True,
                    filesystem_epoch=filesystem_epoch,
                    requires=[
                        "normal_completion",
                        "no_external_filesystem_interference",
                    ],
                    **fields,
                )
            )

        def exact(node, role, value, index=None):
            facts = node.get("argument_facts", {}).get(role)
            if index is not None:
                facts = (
                    facts[index]
                    if isinstance(facts, list) and index < len(facts)
                    else None
                )
            if isinstance(facts, dict) and "exact" in facts:
                candidate = evidence_scalar(facts, facts.get("type"))
                if candidate is _UNKNOWN:
                    return _UNKNOWN
                if value is not _UNKNOWN and not identical(value, candidate):
                    boundaries.append(
                        dict(
                            symbol=symbol,
                            field="argument_facts",
                            reason="contradicts_expression_value",
                        )
                    )
                    invalidate("contradictory_argument_facts")
                    return _UNKNOWN
                operand = node.get(role)
                if index is not None:
                    operand = (
                        operand[index]
                        if isinstance(operand, list) and index < len(operand)
                        else None
                    )
                stable = (
                    isinstance(operand, dict)
                    and operand.get("op") == "var"
                    and operand.get("scope") == "global"
                    and operand.get("name") in stable_globals
                )
                if value is not _UNKNOWN or fact_epoch_clean or stable:
                    return candidate
            return value

        def finish(stream, closing):
            stream.closed = True
            if stream.mode not in {"w", "wt", "x", "xt", "a", "at"}:
                return
            if stream.mode in {"a", "at"}:
                stream.unresolved.add("prior_file_content")
            content = None if stream.unresolved else "".join(stream.chunks)
            raw = None
            try:
                if content is not None:
                    raw = content.encode("utf-8")
            except UnicodeEncodeError:
                boundaries.append(
                    dict(
                        symbol=symbol,
                        field="encoded_bytes",
                        reason="text_contains_surrogates",
                    )
                )
                stream.unresolved.add("text_contains_surrogates")
            value = (
                add("BYTE_VALUE", digest(raw), sha256=digest(raw))
                if raw is not None
                else None
            )
            if value is not None:
                nodes[value]["literal_text"] = content
            locator = stream.locator
            version = add(
                "FILE_VERSION_TEMPLATE",
                [stream.identity, closing, context],
                locator=locator,
                value=value,
                producer=stream.open_operation,
                completion=closing,
                instance_scope="per_successful_dynamic_invocation",
                valid_at="successful_close",
                historical_instance="unresolved",
                value_status=(
                    "conditional_file_content"
                    if value is not None
                    else "unresolved_content"
                ),
                unresolved_fields=sorted(stream.unresolved),
                filesystem_epoch=filesystem_epoch,
                requires=[
                    "regular_byte_file",
                    "exclusive_file_writer",
                    "no_unmodeled_concurrent_effects",
                    "no_unmodeled_file_alias_writes",
                ],
            )
            edges.extend(
                [
                    dict(
                        kind="WRITES_VERSION_IF_NORMAL",
                        source=stream.open_operation,
                        target=version,
                    ),
                    dict(
                        kind="COMPLETES_VERSION_IF_NORMAL",
                        source=closing,
                        target=version,
                    ),
                    dict(kind="LOCATED_BY", source=version, target=locator),
                ]
            )
            if value is not None:
                edges.append(dict(kind="HAS_CONTENT", source=version, target=value))
            for writer in stream.writes:
                edges.append(
                    dict(
                        kind="CONTRIBUTES_CONTENT_IF_NORMAL",
                        source=writer,
                        target=version,
                    )
                )
            templates.append(
                dict(
                    symbol=symbol,
                    version=version,
                    context=context,
                    value=value,
                    source_sites=stream.writes,
                )
            )
            if (
                stream.path_key is not None
                and not stream.unresolved
                and not live_process_effects
            ):
                file_versions[stream.path_key] = version
            if value is None:
                return
            materializations.append(
                dict(
                    symbol=symbol,
                    version=version,
                    value=value,
                    bytes=len(raw),
                    source_sites=stream.writes,
                    proof="qualified text operations, exact operands, ordered writes and successful close",
                    region_completion=completion,
                    activation=fragment.get("activation"),
                    context=context,
                    context_condition=(
                        "formal_parameters_match_and_source_call_is_reached"
                        if bindings
                        else "normal_function_invocation"
                    ),
                    transformation="not_applied",
                    proof_scope="bytes_sent_through_the_qualified_text_stream",
                    file_content_status="conditional_on_storage_and_writer_contracts",
                    obligations=[
                        "regular_byte_file",
                        "exclusive_file_writer",
                        "no_unmodeled_concurrent_effects",
                        "normal_completion",
                        "no_external_filesystem_interference",
                        "preserve_observable_io_effects",
                        "prove_consumer_version_before_replacement",
                    ],
                )
            )

        def copy_file(interface, args, site):
            source_key, _ = locate(args[0], site)
            target_key, target = locate(args[1], site)
            source_version = file_versions.get(source_key)
            if source_version is not None:
                requires(source_version, site, role="copy_source")
            if interface != "fs.copy_file" and target_key not in file_versions:
                # A string names either a directory or a file. A content
                # version at that exact locator is our only closed file-kind
                # evidence here; no directory is inferred from its spelling.
                invalidate("copy_destination_kind_unproven", site)
                boundaries.append(
                    dict(
                        symbol=symbol,
                        field="copy_destination_kind",
                        source=site,
                        interface=interface,
                        reason="directory_or_file_unproven",
                    )
                )
                return _UNKNOWN
            if source_key is None or target_key is None or source_version is None:
                invalidate("copy_source_version_unproven", site)
                return _UNKNOWN
            if source_key == target_key:
                invalidate("copy_source_destination_alias", site)
                boundaries.append(
                    dict(
                        symbol=symbol,
                        field="copy_alias",
                        source=site,
                        reason="same_locator_cannot_complete_as_a_regular_file_copy",
                    )
                )
                return _UNKNOWN
            invalidate("copy_may_alias_another_file", site, retain=(source_key,))
            if live_process_effects:
                return _UNKNOWN
            value = nodes[source_version]["value"]
            version = add(
                "FILE_VERSION_TEMPLATE",
                [site, source_version, target, context],
                locator=target,
                value=value,
                producer=site,
                completion=site,
                source_version=source_version,
                interface=interface,
                copy_contract=dict(
                    semantics_version=1,
                    follow_symlinks=(
                        args[2] if len(args) > 2 and type(args[2]) is bool else None
                    ),
                    metadata={
                        "fs.copy_file": "content_only",
                        "fs.copy": "mode",
                        "fs.copy2": "stat",
                    }[interface],
                    failures="preserve_source_interface",
                ),
                filesystem_epoch=filesystem_epoch,
                instance_scope="per_successful_dynamic_invocation",
                valid_at="successful_copy_return",
                historical_instance="unresolved",
                value_status="conditional_file_content",
                unresolved_fields=[],
                requires=[
                    "source_and_destination_are_regular_byte_files",
                    "source_and_destination_are_distinct_inodes",
                    "source_version_remains_current_through_copy",
                    "exclusive_file_writer",
                    "normal_completion",
                    "no_unmodeled_concurrent_effects",
                    "preserve_interface_metadata_and_failure_effects",
                ],
            )
            edges.extend(
                [
                    dict(
                        kind="COPIES_FILE_VERSION",
                        source=source_version,
                        target=version,
                        conditional=True,
                        interface=interface,
                    ),
                    dict(
                        kind="PRODUCES_VERSION_IF_NORMAL", source=site, target=version
                    ),
                    dict(kind="LOCATED_BY", source=version, target=target),
                ]
            )
            if value is not None:
                edges.append(dict(kind="HAS_CONTENT", source=version, target=value))
            file_versions[target_key] = version
            templates.append(
                dict(
                    symbol=symbol,
                    version=version,
                    context=context,
                    value=value,
                    source_sites=[site],
                    source_version=source_version,
                    derivation="copy",
                    interface=interface,
                )
            )
            return args[1]

        def expression(node):
            nonlocal fact_epoch_clean, live_process_effects, cwd
            nonlocal file_versions, filesystem_epoch, last_file_operation
            if not isinstance(node, dict):
                return _UNKNOWN
            op = node["op"]
            if op == "literal":
                return exact_scalar(node["value"], node.get("type"))
            if op == "var":
                return environment.get((node.get("scope"), node["name"]), _UNKNOWN)
            if op == "call" and node.get("callee") in functions:
                callee = functions[node["callee"]]
                args = [
                    exact(node, "args", expression(arg), i)
                    for i, arg in enumerate(node["args"])
                ]
                args = (
                    [
                        exact_scalar(arg, parameter["type"])
                        for arg, parameter in zip(args, callee.get("params", ()))
                    ]
                    if len(args) == len(callee.get("params", ()))
                    else args
                )
                if summaries[callee["name"]] is not None and len(args) == len(
                    callee.get("params", ())
                ):
                    return apply_result(summaries[callee["name"]], args, summaries)
                if not file_effects[callee["name"]] and callee["name"] not in recursive:
                    # Numeric loops and memory-only callees do not need a
                    # second traversal merely to establish file independence.
                    file_call_counts["non_file"] += 1
                    fact_epoch_clean = False
                    environment.clear()
                    return evidence_scalar(
                        node.get("result_facts"), callee.get("result")
                    )
                site = operation(node)
                static_site = node.get("site") or (symbol + "/call/" + str(ordinal))
                file_key = root_activation, callee["name"], static_site
                scalar_parameters = all(
                    isinstance(parameter.get("type"), str)
                    for parameter in callee.get("params", ())
                )
                supported = (
                    len(args) == len(callee.get("params", ()))
                    and scalar_parameters
                    and "FILE" in file_effects[callee["name"]]
                    and (bindings or not function.get("params"))
                    and callee["name"] not in recursive
                    and callee["name"] not in (*call_chain, symbol)
                    and file_key not in file_contexts
                )
                if supported:
                    file_call_counts["stateful"] += 1
                    file_contexts.add(file_key)
                    # A caller's still-open handle may alias any file touched
                    # by its callee; no ownership transfer is inferred here.
                    if any(not stream.closed for stream in streams):
                        invalidate("interprocedural_open_stream_alias", site)
                    ordered(site)
                    input_versions = sorted(file_versions.values())
                    state_context = digest(
                        json_bytes(
                            [
                                context,
                                static_site,
                                cwd,
                                sorted(file_versions.items()),
                                filesystem_epoch,
                            ]
                        )
                    )
                    child_state, result, closed = evaluate_fragment(
                        dict(
                            symbol=callee["name"],
                            function=callee,
                            completion="complete",
                            bindings=args,
                            call_site=static_site,
                            file_state_context=state_context,
                            activation=dict(
                                owner=symbol,
                                site=node.get("site"),
                                condition="source_call_reached_and_normal_completion",
                            ),
                        ),
                        dict(
                            cwd=cwd,
                            file_versions=file_versions,
                            epoch=filesystem_epoch,
                            last_operation=last_file_operation,
                            fact_epoch_clean=fact_epoch_clean,
                            live_process_effects=live_process_effects,
                            root_activation=root_activation,
                        ),
                        (*call_chain, symbol),
                    )
                    cwd, file_versions = (
                        child_state["cwd"],
                        child_state["file_versions"],
                    )
                    filesystem_epoch = child_state["epoch"]
                    last_file_operation = child_state["last_operation"]
                    live_process_effects = child_state["live_process_effects"]
                    fact_epoch_clean = child_state["fact_epoch_clean"]
                    locals_ = {
                        key: value
                        for key, value in environment.items()
                        if key[0] != "global" and not isinstance(value, _Stream)
                    }
                    environment.clear()
                    environment.update(locals_)
                    if not closed:
                        invalidate(
                            "incomplete_interprocedural_file_state", changes_cwd=True
                        )
                    file_transitions.append(
                        dict(
                            owner=symbol,
                            target=callee["name"],
                            site=static_site,
                            context=state_context,
                            input_versions=input_versions,
                            output_versions=sorted(file_versions.values()),
                            normal_path_closed=closed,
                            condition="source_call_reached_and_normal_completion",
                            historical_instance="not_established",
                        )
                    )
                    return result
                if (
                    len(args) == len(callee.get("params", ()))
                    and args
                    and (bindings or not function.get("params"))
                ):
                    # Streams/arrays are not scalar constants, and must never
                    # be copied into context facts or tensor-value snapshots.
                    # An uninstantiated parameterized frame is a template, not
                    # an invocation whose unknown inputs widen observed inputs.
                    bind_context(
                        callee,
                        static_site,
                        args,
                        dict(
                            owner=symbol,
                            site=node.get("site"),
                            condition="source_call_reached_and_argument_facts_hold",
                        ),
                    )
                fact_epoch_clean = False
                file_call_counts["opaque_barrier"] += 1
                environment.clear()
                invalidate(
                    "interprocedural_file_effect", operation(node), changes_cwd=True
                )
                boundaries.append(
                    dict(
                        symbol=symbol,
                        field="interprocedural_file_state_summary",
                        target=callee["name"],
                        reason=(
                            "recursive_call_component"
                            if callee["name"] in recursive
                            or callee["name"] in (*call_chain, symbol)
                            else (
                                "static_call_context_widened"
                                if file_key in file_contexts
                                else "parameter_or_activation_state_unclosed"
                            )
                        ),
                    )
                )
                return evidence_scalar(node.get("result_facts"), callee.get("result"))
            if op == "text_concat":
                parts = [expression(part) for part in node["parts"]]
                return (
                    "".join(parts)
                    if all(type(part) is str for part in parts)
                    else _UNKNOWN
                )
            if op == "host_call":
                args = [
                    exact(node, "args", expression(arg), i)
                    for i, arg in enumerate(node["args"])
                ]
                interface, site = node["interface"], operation(node)
                if interface == "path.join":
                    return (
                        posixpath.join(*args)
                        if args and all(type(arg) is str for arg in args)
                        else _UNKNOWN
                    )
                if interface == "process.chdir":
                    key, _ = locate(args[0], site)
                    ordered(site)
                    cwd = key
                    return None
                if interface == "process.getcwd":
                    ordered(site)
                    return cwd[1] if cwd is not None and cwd[0] is None else _UNKNOWN
                if interface in {"fs.copy_file", "fs.copy", "fs.copy2"}:
                    ordered(site)
                    return copy_file(interface, args, site)
                if interface == "stream.open":
                    overlapping = any(not stream.closed for stream in streams)
                    if overlapping:
                        invalidate("possible_overlapping_file_handle")
                    if args[1] not in {"r", "rt"}:
                        # Different path spellings may still be hard links.
                        # Opening a writer kills prior content facts until a
                        # distinct-storage relationship has actually been proved.
                        invalidate("file_open_may_write_alias", site)
                    stream = _Stream(site, args[0], args[1], site)
                    stream.path_key, stream.locator = locate(args[0], site)
                    stream.last_operation = site
                    ordered(site)
                    if overlapping:
                        stream.unresolved.add("possible_overlapping_file_handle")
                    if live_process_effects:
                        stream.unresolved.add("unmodeled_live_process")
                    if args[2:] != ["utf-8", "universal"]:
                        stream.unresolved.add("text_encoding")
                    streams.append(stream)
                    if args[1] in {"r", "rt"} and stream.path_key in file_versions:
                        requires(file_versions[stream.path_key], site, role="open_read")
                    return stream
                if (
                    interface.startswith("stream.")
                    and args
                    and isinstance(args[0], _Stream)
                ):
                    stream = args[0]
                    ordered(site)
                    if stream.last_operation != site:
                        edges.append(
                            dict(
                                kind="STREAM_ORDER_IF_NORMAL",
                                source=stream.last_operation,
                                target=site,
                            )
                        )
                    stream.last_operation = site
                    if stream.closed:
                        stream.unresolved.add("operation_after_close")
                        return _UNKNOWN
                    if interface == "stream.write":
                        stream.writes.append(site)
                        if type(args[1]) is str:
                            stream.chunks.append(args[1])
                            return len(args[1])
                        stream.unresolved.add("nonconstant_write")
                    elif interface == "stream.close":
                        finish(stream, site)
                    elif interface not in {
                        "stream.flush",
                        "stream.read",
                        "stream.readline",
                    }:
                        stream.unresolved.add("unclosed_stream_protocol")
                    elif interface in {"stream.read", "stream.readline"}:
                        version = file_versions.get(stream.path_key)
                        if (
                            version is not None
                            and not stream.unresolved
                            and not live_process_effects
                        ):
                            requires(
                                version,
                                site,
                                role="stream_read",
                                range="runtime_position_and_requested_size",
                            )
                        else:
                            boundaries.append(
                                dict(
                                    symbol=symbol,
                                    field="file_read_version",
                                    source=site,
                                    reason="current_file_version_unproven",
                                )
                            )
                        boundaries.append(
                            dict(
                                symbol=symbol,
                                field="read_content_range",
                                source=site,
                                reason="runtime_position_and_requested_size",
                            )
                        )
                    return _UNKNOWN
                if "filesystem" in node.get("writes", ()) or "stream.state" in node.get(
                    "writes", ()
                ):
                    invalidate("unresolved_filesystem_effect", site)
                elif interface.startswith("process.") and node.get("writes"):
                    invalidate("external_process_effect", site, changes_cwd=True)
                elif {"stdin.state", "stdout.state", "stderr.state"}.intersection(
                    node.get("writes", ())
                ):
                    invalidate("unresolved_console_file_alias", site)
                return _UNKNOWN
            if op == "text_print":
                args = [
                    exact(node, "args", expression(arg), i)
                    for i, arg in enumerate(node["args"])
                ]
                sep = exact(node, "sep", expression(node["sep"]))
                end = exact(node, "end", expression(node["end"]))
                rendered = [
                    scalar_spelling(value, node.get("scalar_format", {}))
                    for value in args
                ]
                stream = expression(node["stream"]) if "stream" in node else None
                if isinstance(stream, _Stream):
                    site = operation(node)
                    ordered(site)
                    if stream.last_operation != site:
                        edges.append(
                            dict(
                                kind="STREAM_ORDER_IF_NORMAL",
                                source=stream.last_operation,
                                target=site,
                            )
                        )
                    stream.last_operation = site
                    stream.writes.append(site)
                    if stream.closed:
                        stream.unresolved.add("operation_after_close")
                    elif (
                        type(sep) is str
                        and type(end) is str
                        and all(type(arg) is str for arg in rendered)
                    ):
                        stream.chunks.append(sep.join(rendered) + end)
                    else:
                        stream.unresolved.add("nonconstant_print")
                elif "stream" in node:
                    invalidate("unresolved_print_receiver", operation(node))
                else:
                    invalidate("unresolved_console_file_alias", operation(node))
                return None
            # A closed scalar expression need not imply filesystem effects.
            # Unknown calls can still reach a stream or an aliased file.
            if op in {"process_spawn", "process_request"}:
                live_process_effects = True
                fact_epoch_clean = False
                environment.clear()
                invalidate("unmodeled_live_process", operation(node), changes_cwd=True)
            elif op in {"call", "indirect_call", "library_call"}:
                fact_epoch_clean = False
                environment.clear()
                invalidate(
                    "interprocedural_file_effect", operation(node), changes_cwd=True
                )
            elif "filesystem" in node.get("writes", ()):
                invalidate("unresolved_filesystem_effect", operation(node))
            elif any(
                child is not node
                and (
                    child["op"]
                    in {
                        "call",
                        "indirect_call",
                        "library_call",
                        "process_spawn",
                        "process_request",
                        "text_print",
                    }
                    or {"filesystem", "stream.state"}.intersection(
                        child.get("writes", ())
                    )
                )
                for child in walk_operations(node)
            ):
                invalidate(
                    "unexpanded_effectful_expression", operation(node), changes_cwd=True
                )
            return _UNKNOWN

        def stored(value, spec):
            if isinstance(value, _Stream):
                return (
                    value
                    if spec is None
                    or spec == dict(kind="handle", resource="text_stream")
                    else _UNKNOWN
                )
            return exact_scalar(value, spec)

        def sequence(body):
            nonlocal fact_epoch_clean, return_value
            for statement in body:
                op = statement["op"]
                if op == "decl":
                    if statement.get("storage") in {"static", "thread_local", "extern"}:
                        # An initializer is not the current value on every call.
                        # Its first-evaluation effects also need their own guard.
                        environment[statement.get("scope"), statement["name"]] = (
                            _UNKNOWN
                        )
                        if (
                            "init" in statement
                            and statement["init"].get("op") != "literal"
                        ):
                            invalidate("persistent_initialization_effect")
                            fact_epoch_clean = False
                            environment.clear()
                        continue
                    environment[statement.get("scope"), statement["name"]] = (
                        stored(expression(statement["init"]), statement.get("type"))
                        if "init" in statement
                        else _UNKNOWN
                    )
                elif op == "assign":
                    value = expression(statement["value"])
                    target = statement["target"]
                    if target["op"] == "var":
                        environment[target.get("scope"), target["name"]] = stored(
                            value, target.get("type")
                        )
                    elif isinstance(value, _Stream):
                        value.unresolved.add("escaped_stream")
                elif op == "expr":
                    expression(statement["value"])
                elif op == "block":
                    shadows = {
                        (item.get("scope"), item["name"]): environment.get(
                            (item.get("scope"), item["name"]), _UNKNOWN
                        )
                        for item in statement["body"]
                        if item["op"] == "decl"
                    }
                    flow = sequence(statement["body"])
                    environment.update(shadows)
                    if flow != "normal":
                        return flow
                elif op == "return":
                    if statement.get("value"):
                        returned = expression(statement["value"])
                        if isinstance(returned, _Stream):
                            returned.unresolved.add("escaped_returned_stream")
                            invalidate("escaped_returned_stream", operation(statement))
                        else:
                            return_value = exact_scalar(
                                returned, function.get("result")
                            )
                    else:
                        return_value = None
                    return "return"
                else:
                    # Preserve loops/branch/cleanup boundaries. Their writes
                    # cannot be lifted out as unconditional prefix actions.
                    invalidate("unexpanded_control_region")
                    boundaries.append(
                        dict(
                            symbol=symbol,
                            field="control_region",
                            operation=op,
                            site=statement.get("site"),
                        )
                    )
                    return "unresolved"
            return "normal"

        flow = sequence(function["body"])
        if flow == "normal" and function.get("result") in {"unit", "void"}:
            return_value = None
        for stream in streams:
            if not stream.closed or stream.unresolved:
                boundaries.append(
                    dict(
                        symbol=symbol,
                        field="file_content",
                        source=stream.open_operation,
                        reasons=sorted(
                            stream.unresolved
                            | ({"close_not_proven"} if not stream.closed else set())
                        ),
                    )
                )
        if fragment.get("frontier"):
            boundaries.append(
                dict(
                    symbol=symbol,
                    field="prefix_frontier",
                    frontier=fragment["frontier"],
                )
            )
        return (
            dict(
                cwd=cwd,
                file_versions=file_versions,
                epoch=filesystem_epoch,
                last_operation=last_file_operation,
                live_process_effects=live_process_effects,
                fact_epoch_clean=fact_epoch_clean,
                root_activation=root_activation,
            ),
            return_value,
            flow in {"normal", "return"}
            and not any(not stream.closed for stream in streams),
        )

    while pending:
        evaluate_fragment(pending.popleft())

    graph["nodes"] = list(nodes.values())
    graph["edges"] = [
        json.loads(encoded) for encoded in sorted({json_bytes(edge) for edge in edges})
    ]
    graph.setdefault("coverage", {})["normal_path_file_state"] = dict(
        scope="bounded_normal_path_call_contexts",
        initial_cwd="symbolic_invocation_cwd",
        interprocedural_file_state="conditional_scalar_parameter_state_transitions",
        context_limit="one_per_root_activation_and_static_call_site",
        recursive_components="opaque_state_barrier",
        directory_kind="not_inferred_from_path_spelling",
        read_value="not_constant_folded",
    )
    validate_artifact_flow(graph)
    return dict(
        graph=graph,
        constant_materializations=materializations,
        file_templates=templates,
        call_contexts=len(context_inputs),
        file_call_contexts=len(file_contexts),
        file_state_transitions=file_transitions,
        file_effect_summary=dict(
            functions=len(functions),
            scanned_operations=summary_operations,
            call_edges=sum(len(edges) for edges in file_calls.values()),
            file_functions=sum("FILE" in effect for effect in file_effects.values()),
            opaque_functions=sum(
                "OPAQUE" in effect for effect in file_effects.values()
            ),
            recursive_functions=len(recursive),
            calls=file_call_counts,
            propagation="two_monotone_effect_bits_per_function",
        ),
        result_summaries={
            name: summary for name, summary in summaries.items() if summary is not None
        },
        boundaries=boundaries,
        historical_read_version_proof="not_established_by_content_equality",
    )

"""M2 file-content provenance over already-qualified canonical operations."""

from copy import deepcopy
from collections import deque
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

    while pending:
        fragment = pending.popleft()
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
        for parameter, value in zip(function.get("params", ()), bindings):
            environment[None, parameter["name"]] = exact_scalar(
                value, parameter["type"]
            )
        fact_epoch_clean, live_process_effects = True, False

        def operation(node):
            nonlocal ordinal
            ordinal += 1
            site = node.get("site")
            if site in source_operations:
                return source_operations[site]
            return add(
                "CANONICAL_OPERATION",
                [symbol, completion, site, ordinal, context],
                symbol=symbol,
                site=site,
                operation=node["op"],
                instance_scope="per_dynamic_invocation_and_iteration",
            )

        def invalidate(reason):
            for stream in streams:
                if not stream.closed:
                    stream.unresolved.add(reason)

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
            location = dict(
                path=stream.path if isinstance(stream.path, str) else None,
                cwd=(
                    "invocation.cwd"
                    if not isinstance(stream.path, str)
                    or not posixpath.isabs(stream.path)
                    else None
                ),
                symbol=symbol,
            )
            locator = add(
                "FILE_LOCATOR",
                [location, context],
                **location,
                identity_scope="normal_path_expression",
                inode_aliases="unresolved",
            )
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
                requires=[
                    "regular_byte_file",
                    "exclusive_file_writer",
                    "no_unmodeled_concurrent_effects",
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

        def expression(node):
            nonlocal fact_epoch_clean, live_process_effects
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
                if (
                    len(args) == len(callee.get("params", ()))
                    and args
                    and (bindings or not function.get("params"))
                ):
                    # Streams/arrays are not scalar constants, and must never
                    # be copied into context facts or tensor-value snapshots.
                    # An uninstantiated parameterized frame is a template, not
                    # an invocation whose unknown inputs widen observed inputs.
                    site = operation(node)
                    static_site = node.get("site") or (symbol + "/call/" + str(ordinal))
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
                environment.clear()
                invalidate("interprocedural_file_effect")
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
                if interface == "stream.open":
                    overlapping = any(not stream.closed for stream in streams)
                    if overlapping:
                        invalidate("possible_overlapping_file_handle")
                    stream = _Stream(site, args[0], args[1], site)
                    if overlapping:
                        stream.unresolved.add("possible_overlapping_file_handle")
                    if live_process_effects:
                        stream.unresolved.add("unmodeled_live_process")
                    if args[2:] != ["utf-8", "universal"]:
                        stream.unresolved.add("text_encoding")
                    streams.append(stream)
                    return stream
                if (
                    interface.startswith("stream.")
                    and args
                    and isinstance(args[0], _Stream)
                ):
                    stream = args[0]
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
                        stream.unresolved.add("read_position")
                    return _UNKNOWN
                if "filesystem" in node.get("writes", ()) or "stream.state" in node.get(
                    "writes", ()
                ):
                    invalidate("unresolved_filesystem_effect")
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
                    invalidate("unresolved_print_receiver")
                else:
                    invalidate("unresolved_console_file_alias")
                return None
            # A closed scalar expression need not imply filesystem effects.
            # Unknown calls can still reach a stream or an aliased file.
            if op == "process_spawn":
                live_process_effects = True
                fact_epoch_clean = False
                environment.clear()
                invalidate("unmodeled_live_process")
            elif op in {"call", "indirect_call", "library_call"}:
                fact_epoch_clean = False
                environment.clear()
                invalidate("interprocedural_file_effect")
            elif "filesystem" in node.get("writes", ()):
                invalidate("unresolved_filesystem_effect")
            elif any(
                child is not node
                and (
                    child["op"]
                    in {
                        "call",
                        "indirect_call",
                        "library_call",
                        "process_spawn",
                        "text_print",
                    }
                    or {"filesystem", "stream.state"}.intersection(
                        child.get("writes", ())
                    )
                )
                for child in walk_operations(node)
            ):
                invalidate("unexpanded_effectful_expression")
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
            nonlocal fact_epoch_clean
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
                        expression(statement["value"])
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

        sequence(function["body"])
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

    graph["nodes"] = list(nodes.values())
    graph["edges"] = [
        json.loads(encoded) for encoded in sorted({json_bytes(edge) for edge in edges})
    ]
    validate_artifact_flow(graph)
    return dict(
        graph=graph,
        constant_materializations=materializations,
        file_templates=templates,
        call_contexts=len(context_inputs),
        result_summaries={
            name: summary for name, summary in summaries.items() if summary is not None
        },
        boundaries=boundaries,
        historical_read_version_proof="not_established_by_content_equality",
    )

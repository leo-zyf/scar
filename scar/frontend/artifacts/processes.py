"""F5 joins sealed launch observations to source ports and process invocations."""

import posixpath

from scar.contract import digest, json_bytes, json_loads


def link_observations(flow, build, units, root):
    observation = (build or {}).get("process_observation")
    if observation is None:
        return flow
    nodes = {node["id"]: node for node in flow["nodes"]}
    edges = list(flow["edges"])
    invocations = {
        node["build_node"]: node["id"]
        for node in nodes.values()
        if node["kind"] == "BUILD_OPERATION"
    }
    bindings = {row["sequence"]: row for row in observation.get("bindings", ())}
    if len(bindings) != len(observation.get("bindings", ())):
        raise ValueError("duplicate observed process binding")
    attribution = []
    source_root_matches = not observation.get("diagnostics") and posixpath.normpath(
        observation.get("source_root", "")
    ) == posixpath.normpath(str(root))
    runtime = observation.get("interpreter", {})
    interpreter = runtime.get("implementation", "") + " " + runtime.get("version", "")

    def edge(kind, source, target, **fields):
        edges.append(dict(kind=kind, source=source, target=target, **fields))

    for ordinal, event in enumerate(observation.get("events", ())):
        sequence = event["sequence"]
        identity = "process_intent:" + digest(
            json_bytes([build["trace_sha256"], observation["sha256"], ordinal])
        )
        nodes[identity] = dict(
            id=identity,
            kind="PROCESS_INTENT",
            event_kind=event["kind"],
            **{key: value for key, value in event.items() if key != "kind"},
            observation_sha256=observation["sha256"],
            trace_sha256=build["trace_sha256"],
            interpretation="pre_spawn_intent_not_process_completion",
            process_id_domain="python_published",
        )
        caller = event["caller"]
        source_candidates = []
        unresolved = list(event["unresolved"])
        if (
            caller is not None
            and caller["source_code_matches_snapshot"]
            and source_root_matches
        ):
            location = caller["location"]
            for process in flow["source_processes"]:
                unit = units[process["unit"]]
                expected = unit["source_facts"].get("runtime", {}).get("interpreter")
                site = process.get("location", {})
                if (
                    process["operation_kind"] == "process_launch"
                    and unit["source_facts"]["files"].get(caller["path"])
                    == caller["source_sha256"]
                    and expected == interpreter
                    and site.get("file") == caller["path"]
                    and all(
                        site.get(key) == value
                        for key, value in location.items()
                        if value is not None
                    )
                ):
                    source_candidates.append(process["operation"])
        operation = source_candidates[0] if len(source_candidates) == 1 else None
        if operation is None:
            unresolved.append("source_operation_not_uniquely_qualified")
        else:
            edge(
                "OBSERVED_LAUNCH_INTENT",
                operation,
                identity,
                evidence="matching_source_snapshot_code_and_position",
                source_sha256=caller["source_sha256"],
            )
        binding = bindings.get(sequence)
        invocation = None
        if binding is not None:
            invocation = invocations.get("build:invocation:" + binding["invocation_id"])
            parent = invocations.get(
                "build:invocation:" + binding["parent_invocation_id"]
            )
            if invocation is None or parent is None:
                raise ValueError(
                    "observed spawn binding references an absent invocation"
                )
            edge(
                "OBSERVED_PROCESS_START",
                identity,
                invocation,
                evidence=binding["evidence"],
                **(
                    {"instrumentation_mapping": binding["instrumentation_mapping"]}
                    if "instrumentation_mapping" in binding
                    else {}
                ),
            )
            edge("INTENT_PARENT_INVOCATION", parent, identity)
        else:
            unresolved.append("process_invocation_not_uniquely_qualified")
        attribution.append(
            dict(
                sequence=sequence,
                intent=identity,
                source_operation=operation,
                source_candidates=source_candidates,
                invocation=invocation,
                status=(
                    "BOUND"
                    if operation is not None and invocation is not None
                    else "UNRESOLVED"
                ),
                unresolved=unresolved,
            )
        )
    bound = sum(row["status"] == "BOUND" for row in attribution)
    return dict(
        flow,
        nodes=list(nodes.values()),
        edges=[json_loads(raw) for raw in sorted({json_bytes(edge) for edge in edges})],
        process_observations=dict(
            provider=observation.get("provider"),
            sha256=observation["sha256"],
            trace_sha256=build["trace_sha256"],
            event_limit=observation.get("event_limit"),
            dropped_events=observation.get("dropped_events"),
            launch_bindings=attribution,
            diagnostics=observation.get("diagnostics", []),
            unresolved=observation.get("unresolved", []),
            **(
                {"process_tree": observation["process_tree"]}
                if "process_tree" in observation
                else {}
            ),
        ),
        coverage=dict(
            flow["coverage"],
            process_observation_intents=len(attribution),
            observed_source_launch_bindings=bound,
            process_execution_events=(
                "captured_process_starts" if bindings else "not_observed"
            ),
            process_instance_binding=(
                "partial_observed_source_links" if bound else "source_templates_only"
            ),
            process_build_invocation_matching=(
                "partial_observed" if bound else "not_established"
            ),
        ),
    )

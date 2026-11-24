"""Join existing subgraphs without reparsing source or asserting unobserved history."""

import json
import posixpath

from scar.contract import (
    ARTIFACT_FLOW,
    byte_value_id,
    digest,
    document,
    json_bytes,
    validate_artifact_flow,
)


def artifact_graph(units, payloads, build, images, root):
    """F5 artifact provenance: operation references, file versions and byte values.

    Paths identify locators, not inodes. Captured file bytes identify a snapshot,
    not the bytes consumed by an earlier invocation. Those historical read
    requirements remain separate version nodes with an unresolved value field.
    """
    nodes, edges, pending = {}, [], []

    def identity(kind, key):
        if kind == "byte_value":
            return byte_value_id(key)
        return kind + ":" + digest(json_bytes(key))

    def node(kind, key, **fields):
        key = identity(kind.lower(), key)
        value = dict(id=key, kind=kind, **fields)
        if nodes.setdefault(key, value) != value:
            raise ValueError("conflicting artifact graph identity")
        return key

    def edge(kind, source, target, **fields):
        edges.append(dict(kind=kind, source=source, target=target, **fields))

    def locator(path):
        return node(
            "FILE_LOCATOR",
            path,
            path=path,
            identity_scope="captured_resolved_path",
            inode_aliases="unresolved",
        )

    def snapshot(file, sha256, observation, size=None):
        value = node("BYTE_VALUE", sha256, sha256=sha256)
        version = node(
            "FILE_VERSION",
            [file, observation, sha256],
            locator=file,
            value=value,
            observation=observation,
            size=size,
            history="capture_snapshot",
        )
        edge("HAS_SNAPSHOT", file, version)
        edge("HAS_CONTENT", version, value)
        return version

    profiles = {}
    source_sites = []
    for index, (unit, payload) in enumerate(zip(units, payloads)):
        reader = node(
            "SOURCE_CAPTURE",
            [index, unit["revision"]],
            unit=index,
            compiler=unit["source_facts"]["compiler"],
        )
        if profile := unit.get("compilation_profile_sha256"):
            profiles.setdefault(profile, []).append(reader)
        for name, sha256 in unit["source_facts"]["files"].items():
            path = posixpath.normpath(posixpath.join(str(root), name))
            file = locator(path)
            version = snapshot(file, sha256, reader)
            edge(
                "READS_CAPTURED_VERSION",
                reader,
                version,
                evidence="parser_owned_buffer",
            )
        for path, receipt in unit["source_facts"].get("external_headers", {}).items():
            file = locator(path)
            version = snapshot(file, receipt["sha256"], reader, receipt.get("size"))
            edge(
                "READS_CAPTURED_VERSION",
                reader,
                version,
                evidence="parser_owned_buffer",
            )
        functions = [payload["global_graph"], *payload["functions"]]
        functions.extend(
            record["initializer_graph"]
            for record in payload["records"]
            if "initializer_graph" in record
        )
        for function in functions:
            rows = function["nodes"]

            def operation(port):
                if type(port) is not int or not 0 <= port < len(rows):
                    raise ValueError(
                        "artifact operand references an absent source port"
                    )
                return node(
                    "SOURCE_OPERATION",
                    [function["id"], port],
                    unit=index,
                    graph_id=function["id"],
                    port=port,
                    operation=rows[port]["op"],
                    instance_scope="per_dynamic_invocation_and_iteration",
                )

            for port, row in enumerate(rows):
                effect = row.get("artifact_effect")
                if not effect:
                    continue
                if effect.get("semantics_version") != 1:
                    raise ValueError("unsupported source artifact effect contract")
                action = operation(port)
                operands = {}

                def references(value, role):
                    if isinstance(value, dict) and set(value) == {"port"}:
                        reference = operation(value["port"])
                        edge("ARGUMENT", reference, action, role=role)
                        return reference
                    if isinstance(value, list):
                        return [
                            references(item, role + "/" + str(i))
                            for i, item in enumerate(value)
                        ]
                    raise ValueError(
                        "artifact operands must identify graph value ports"
                    )

                for role, path in effect["operands"].items():
                    value = row
                    for key in path:
                        # F3 has replaced nested expressions by port edges.
                        # A route follows those edges before selecting its next
                        # field; it never indexes a port record as an AST node.
                        if isinstance(value, dict) and set(value) == {"port"}:
                            target_port = value["port"]
                            operation(target_port)
                            value = rows[target_port]
                        value = value[key]
                    operands[role] = references(value, role)
                source_sites.append(
                    dict(
                        operation=action,
                        interface=effect["interface"],
                        operands=operands,
                        defaults=effect["defaults"],
                        obligations=effect["obligations"],
                        binding_guard=row.get("binding_guard"),
                        stream_handle=row.get("stream_handle"),
                        **(
                            {"contract": effect["contract"]}
                            if "contract" in effect
                            else {}
                        ),
                    )
                )

    build_refs, build_snapshots = {}, {}
    if build:
        for row in build["nodes"]:
            key = row["id"]
            if row["kind"] == "BUILD_ARTIFACT":
                file = locator(row["path"])
                build_refs[key] = file
                if receipt := row.get("receipt"):
                    if sha256 := receipt.get("sha256"):
                        build_snapshots[key] = snapshot(
                            file,
                            sha256,
                            "build-capture:" + build["trace_sha256"],
                            receipt.get("size"),
                        )
            elif row["kind"] == "BUILD_OPERATION":
                build_refs[key] = node(
                    "BUILD_OPERATION",
                    key,
                    build_node=key,
                    invocation_id=row["invocation_id"],
                )
        requirements = set()
        for relation in build["edges"]:
            kind, source, target = (
                relation[key] for key in ("kind", "source", "target")
            )
            if kind == "COMPILES":
                profile = build["compilation_profiles"][relation["profile"]]
                for reader in profiles.get(digest(json_bytes(profile)), ()):
                    edge("ANALYZES_COMPILATION_PROFILE", build_refs[source], reader)
                source, target = target, source
            if kind in {
                "COMPILES",
                "BUILD_INPUT",
                "LINK_INPUT",
                "EXTERNAL_LIBRARY_INPUT",
                "DECLARED_BUILD_INPUT",
            }:
                pair = source, target
                if (
                    pair in requirements
                    or source not in build_refs
                    or target not in build_refs
                ):
                    continue
                requirements.add(pair)
                file = build_refs[source]
                version = node(
                    "FILE_VERSION_REQUIREMENT",
                    pair,
                    locator=file,
                    value=None,
                    unresolved_field="historical_consumed_bytes",
                    consumer=build_refs[target],
                )
                edge("REQUIRES_VERSION", build_refs[target], version, evidence=kind)
                edge("LOCATED_BY", version, file)
                if source in build_snapshots:
                    edge(
                        "CAPTURED_AT_SAME_LOCATOR",
                        version,
                        build_snapshots[source],
                        establishes_version_equality=False,
                    )
            elif (
                kind == "BUILD_OUTPUT" and source in build_refs and target in build_refs
            ):
                edge("DECLARES_OUTPUT", build_refs[source], build_refs[target])
        for image in images or ():
            artifact = image.get("build_artifact")
            if artifact not in build_refs:
                continue
            image_node = node(
                "EXECUTION_IMAGE",
                image["id"],
                image_id=image["id"],
                units=image["units"],
                entry=image["entry"],
            )
            edge("BUILT_IMAGE_SOURCE", build_refs[artifact], image_node)

    for site in source_sites:
        interface = site["interface"]
        if interface.startswith("stream.") or interface == "text.print":
            pending.append(
                dict(
                    operation=site["operation"],
                    fields=[
                        "runtime_stream_instance",
                        "file_locator_version",
                        "value_content",
                    ],
                    basis="source_binding_and_control_obligations",
                )
            )
    if any(e["source"] not in nodes or e["target"] not in nodes for e in edges):
        raise ValueError("artifact graph has an unbound endpoint")
    # Stable order and duplicate removal keep this cross-unit index compact.
    edges = [json.loads(value) for value in sorted({json_bytes(e) for e in edges})]
    return validate_artifact_flow(
        document(
            ARTIFACT_FLOW,
            nodes=list(nodes.values()),
            edges=edges,
            source_effects=source_sites,
            open_fields=pending,
            coverage=dict(
                source_effect_sites=len(source_sites),
                parser_snapshot_reads=sum(
                    e["kind"] == "READS_CAPTURED_VERSION" for e in edges
                ),
                historical_read_versions="not_observed",
                path_to_inode_aliasing="not_inferred",
            ),
        )
    )

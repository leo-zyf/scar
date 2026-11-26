"""M4 cross-image entry and lifecycle graph over independently sealed image IRs."""

from collections import defaultdict
from pathlib import Path

from scar.contract import (
    IR,
    WORKFLOW,
    digest,
    document,
    json_bytes,
    read_image_product,
    read_member,
    validate_artifact_flow,
    validate_images,
    validate_workflow,
)

_ENTRY_STATES = ("argv", "cwd", "environment", "filesystem_namespace")


def link_images(folder, images, root_image):
    """Bind initial process state, never a post-mutation argv or a file version."""
    folder = Path(folder)
    validate_images(images, root_image)
    image_ids = {image["id"] for image in images}
    nodes, edges, catalog = {}, [], []
    launch_files, copies, reads = [], {}, defaultdict(list)

    def add(kind, image, key, **fields):
        identity = "workflow:" + digest(json_bytes([kind, image, key]))
        nodes[identity] = dict(id=identity, kind=kind, image_id=image, **fields)
        return identity

    def image_node(image):
        return add("IMAGE", image, image, instance_scope="per_dynamic_image_entry")

    def entry(image, state):
        return add(
            "ENTRY_STATE",
            image,
            state,
            state=state,
            instance_scope="per_dynamic_image_entry",
            version="initial_entry_state_not_later_mutations",
        )

    def expand(image, node):
        edges.append(dict(kind="EXPANDS_TO", source=image_node(image), target=node))

    for image in images:
        identity = image["id"]
        image_node(identity)
        for state in _ENTRY_STATES:
            expand(identity, entry(identity, state))

    for image in images:
        identity = image["id"]
        path, revision, ir = read_image_product(folder, image, "ir", "exec-ir.json", IR)
        if ir.get("module", {}).get("entry") != image["entry"]:
            raise ValueError("workflow image entry differs from its sealed IR")
        receipt = ir.get("module", {}).get("resource_model", {}).get("artifact_flow")
        catalog.append(
            dict(
                id=identity,
                entry=image["entry"],
                output=image["output"],
                source_ir_sha256=revision,
                artifact_flow_sha256=receipt["sha256"] if receipt else None,
            )
        )
        if receipt is None:
            continue
        flow = validate_artifact_flow(read_member(path.parent, receipt))
        records = {node["id"]: node for node in flow["nodes"]}

        def origin(node):
            return dict(artifact_flow_sha256=receipt["sha256"], node=node)

        process_launches = {}
        for environment in flow.get("launch_environments", ()):
            child = environment.get("image_id")
            if child is not None and child not in image_ids:
                raise ValueError("launch targets an undeclared workflow image")
            launch = add(
                "LAUNCH",
                identity,
                environment["id"],
                origin=origin(environment["operation"]),
                launch_environment=environment["id"],
                process_template=environment.get("process_template"),
                child_image=child,
                target_binding=environment["target_binding"],
                replication=(
                    environment.get("launcher")
                    or {"kind": "single_process", "count": 1}
                    if child is not None
                    else {"kind": "unbound"}
                ),
                instance_scope="per_dynamic_launch_and_control_context",
                historical_instance="not_established",
            )
            expand(identity, launch)
            if environment.get("process_template") is not None:
                process_launches[environment["process_template"]] = launch
            if child is None:
                continue
            if environment.get("directory") and environment.get(
                "conditional_candidates"
            ):
                launch_files.append((identity, child, launch, environment))
            edges.append(
                dict(
                    kind="LAUNCHES_IMAGE",
                    source=launch,
                    target=image_node(child),
                    conditional=True,
                    requires=["qualified_backend_image_and_launcher"],
                )
            )
            for state in _ENTRY_STATES:
                edges.append(
                    dict(
                        kind="BINDS_ENTRY_STATE",
                        source=launch,
                        target=entry(child, state),
                        conditional=True,
                        binding_time="process_entry",
                        source_ports=(
                            environment["argument_ports"] if state == "argv" else []
                        ),
                        **(
                            {"argument_vector": environment["argument_vector"]}
                            if state == "argv" and "argument_vector" in environment
                            else {}
                        ),
                        parent_directory=(
                            environment.get("directory") if state == "cwd" else None
                        ),
                        requires=[
                            "preserve_declared_worker_entry_protocol",
                            "backend_qualified_environment_cwd_and_namespace_inheritance",
                        ],
                    )
                )

        for node in records.values():
            if node["kind"] == "FILE_VERSION_TEMPLATE" and node.get("interface") in {
                "fs.copy_file",
                "fs.copy",
                "fs.copy2",
            }:
                selected = add(
                    "FILE_COPY",
                    identity,
                    node["id"],
                    origin=origin(node["id"]),
                    destination_case=node.get("destination_case"),
                    locator=node["locator"],
                    source_requirement=node.get("source_requirement"),
                    source_version=node.get("source_version"),
                    requires=node.get("requires", []),
                )
                expand(identity, selected)
                locator = records[node["locator"]]
                path = locator.get("path")
                if isinstance(path, str) and (name := path.rsplit("/", 1)[-1]):
                    copies[identity, node["id"]] = (
                        selected,
                        locator.get("directory"),
                        name,
                    )
            elif (
                node["kind"] == "FILE_VERSION_REQUIREMENT"
                and node.get("role") == "runtime_stream_read"
            ):
                selected = add(
                    "FILE_READ",
                    identity,
                    node["id"],
                    origin=origin(node["id"]),
                    locator=node["locator"],
                    path=records[node["locator"]].get("path"),
                    acquisition=node.get("acquisition"),
                    requires=node.get("requires", []),
                )
                expand(identity, selected)
                locator = records[node["locator"]]
                path = locator.get("path")
                if (
                    isinstance(path, str)
                    and path not in {"", ".", ".."}
                    and "/" not in path
                    and locator.get("cwd") == "process_cwd_at_open"
                ):
                    reads[identity, path].append(selected)
            elif (
                node["kind"] == "COMMAND_REQUEST"
                and node.get("request_kind") != "processor_availability_query"
            ):
                selected = add(
                    "LAUNCH",
                    identity,
                    node["id"],
                    origin=origin(node["id"]),
                    child_image=None,
                    target_binding="unbound",
                    activation_condition=node.get("request_kind"),
                    historical_instance="not_established",
                )
                expand(identity, selected)

        for relation in flow["edges"]:
            if relation["kind"] != "COMPLETES_DIRECT_PROCESS_IF_NORMAL":
                continue
            launch = process_launches.get(relation["source"])
            if launch is None:
                raise ValueError("workflow wait lacks its process launch environment")
            wait = add(
                "WAIT",
                identity,
                [relation["source"], relation["target"]],
                origin=origin(relation["target"]),
                process_template=relation["source"],
                descendants_completed=False,
            )
            expand(identity, wait)
            edges.append(
                dict(
                    kind="WAITS_FOR",
                    source=launch,
                    target=wait,
                    conditional=True,
                    descendants_completed=False,
                    time_constraint="finish(wait) >= exit(direct_child)",
                )
            )

    for parent, child, launch, environment in launch_files:
        directory = environment["directory"]
        for candidate in environment["conditional_candidates"]:
            copy = copies.get((parent, candidate["version"]))
            if copy is None or copy[1] != directory:
                continue
            source, _, basename = copy
            for target in reads[child, basename]:
                edges.append(
                    dict(
                        kind="MAY_SUPPLY_CHILD_READ",
                        source=source,
                        target=target,
                        launch=launch,
                        conditional=True,
                        establishes_current_version=False,
                        parent_directory=directory,
                        invalidated_by=candidate["invalidated_by"],
                        requires=sorted(
                            set(candidate["requires"])
                            | set(environment["requires"])
                            | {
                                "child_read_resolves_in_inherited_entry_directory",
                                "selected_copy_instance_is_current_at_child_read",
                            }
                        ),
                    )
                )

    return validate_workflow(
        document(
            WORKFLOW,
            root_image=root_image,
            images=catalog,
            nodes=list(nodes.values()),
            edges=edges,
            binding_scope="initial_process_state_and_direct_child_lifetimes",
            file_version_equivalence_established=False,
            reference_scope="image_artifact_flow_identity; no source package paths",
        )
    )

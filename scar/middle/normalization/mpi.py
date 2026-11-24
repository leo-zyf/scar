"""M2 contracts for provider-bound MPI C calls; no communication is reordered.

The conservative lifecycle and completion rules follow MPI-4.1, sections
4.7.3, 4.9, 7 and 12.2. In particular, Request_free is not a completion proof,
and a general collective is not an implicit barrier for unrelated operations.
"""

import re

from scar.contract import walk_operations

_INTERFACE = re.compile(r"P?MPI_[A-Za-z][A-Za-z0-9_]*\Z")
_OPERATIONS = {"call", "indirect_call"}
_FIELDS = {"api", "semantics_version", "operations", "interfaces"}
_COLLECTIVES = {
    "Barrier",
    "Bcast",
    "Gather",
    "Gatherv",
    "Scatter",
    "Scatterv",
    "Allgather",
    "Allgatherv",
    "Alltoall",
    "Alltoallv",
    "Alltoallw",
    "Reduce",
    "Allreduce",
    "Reduce_scatter",
    "Reduce_scatter_block",
    "Scan",
    "Exscan",
    "Neighbor_allgather",
    "Neighbor_allgatherv",
    "Neighbor_alltoall",
    "Neighbor_alltoallv",
    "Neighbor_alltoallw",
}


def _requirement(value):
    if (
        not isinstance(value, dict)
        or set(value) != _FIELDS
        or value["api"] != "c"
        or type(value["semantics_version"]) is not int
        or value["semantics_version"] != 1
        or not isinstance(value["operations"], list)
        or any(
            not isinstance(op, str) or op not in _OPERATIONS
            for op in value["operations"]
        )
        or not isinstance(value["interfaces"], list)
        or any(
            not isinstance(name, str) or not _INTERFACE.fullmatch(name)
            for name in value["interfaces"]
        )
    ):
        raise ValueError("MPI requires a C API v1 contract with closed interfaces")
    return set(value["operations"]), set(value["interfaces"])


def _point_to_point(name):
    mode, operation = "blocking", name.removesuffix("_c")
    if operation.endswith("_init"):
        mode, operation = "persistent_initialization", operation[:-5]
    elif operation.startswith("I"):
        mode, operation = "nonblocking", operation[1:2].upper() + operation[2:]
    send_modes = {
        "Send": "standard",
        "Ssend": "synchronous",
        "Bsend": "buffered",
        "Rsend": "ready",
    }
    if operation not in {*send_modes, "Recv"}:
        return None
    receive = operation == "Recv"
    arguments = dict(buffer=0, count=1, datatype=2, peer=3, tag=4, communicator=5)
    if mode != "blocking":
        arguments["request"] = 6
    elif receive:
        arguments["status"] = 6
    contract = dict(
        category="point_to_point",
        direction="receive" if receive else "send",
        mode=mode,
        argument_indexing="zero_based",
        arguments=arguments,
        buffer_access=dict(
            argument=0,
            mode="write" if receive else "read",
            region="datatype_typemap_and_count",
            coverage="may_access",
        ),
        buffer_lifetime={
            "blocking": "call_until_successful_return",
            "nonblocking": "call_until_request_completion",
            "persistent_initialization": "request_start_until_completion",
        }[mode],
        communication_started_by_call=mode != "persistent_initialization",
        remote_call_completion_guaranteed=False,
    )
    if not receive:
        contract["send_mode"] = send_modes[operation]
        if operation == "Rsend":
            contract["precondition"] = "matching_receive_posted_before_send_start"
    if mode != "blocking":
        contract["request_effect"] = (
            "create_inactive"
            if mode == "persistent_initialization"
            else "create_active"
        )
        contract["application_buffer_access_while_active"] = (
            "no_reads_or_writes" if receive else "no_writes"
        )
    if mode == "persistent_initialization":
        contract.update(
            activation="MPI_Start_or_MPI_Startall",
            buffer_binding="address_bound_by_initialization",
            application_buffer_access_while_inactive="not_restricted_by_this_request",
        )
    return contract


def _interface_contract(symbol):
    # PMPI has the corresponding MPI routine's semantics; preserve its spelling.
    name = symbol[5:] if symbol.startswith("PMPI_") else symbol[4:]
    contract = dict(category="opaque_mpi", reads="unknown", writes="unknown")
    point_to_point = _point_to_point(name)
    if point_to_point is not None:
        contract.update(point_to_point)
    elif name in {"Init", "Init_thread", "Finalize", "Initialized", "Finalized"}:
        contract.update(category="lifecycle", scope="world_model")
        if name in {"Init", "Init_thread"}:
            contract.update(transition="initialize", restart_after_finalize=False)
            if name == "Init_thread":
                contract["thread_support"] = "preserve_requested_and_provided_levels"
        elif name == "Finalize":
            contract.update(
                transition="finalize",
                precondition="complete_local_operations_and_matching_obligations",
                satisfies_pending_completion_obligations=False,
                session_state_finalized=False,
            )
        else:
            contract.update(transition="query", query=name.lower())
    elif name.startswith("Comm_"):
        contract.update(category="communicator", handle="communicator")
        if name in {"Comm_rank", "Comm_size", "Comm_compare", "Comm_test_inter"}:
            contract["handle_effect"] = "query"
        elif name in {"Comm_dup", "Comm_dup_with_info", "Comm_split", "Comm_create"}:
            contract.update(handle_effect="create", participation="collective")
        elif name in {"Comm_idup", "Comm_idup_with_info"}:
            contract.update(handle_effect="create", request_effect="create_active")
        elif name in {"Comm_free", "Comm_disconnect"}:
            contract.update(
                handle_effect="release",
                pending_completion_guaranteed=(name == "Comm_disconnect"),
            )
    elif name in {"Request_free", "Request_get_status", "Start", "Startall", "Cancel"}:
        contract.update(category="request", handle="request")
        if name == "Request_free":
            contract.update(
                handle_effect="release_handle",
                operation_completion="not_implied",
                operation_storage_release="may_be_deferred",
            )
        elif name in {"Start", "Startall"}:
            contract.update(
                handle_effect="activate_persistent_request",
                precondition="request_is_inactive",
            )
        elif name == "Request_get_status":
            contract.update(
                handle_effect="query", operation_completion="flag_dependent"
            )
        else:
            contract.update(
                handle_effect="request_cancellation",
                operation_completion="requires_completion_operation",
                cancellation_success="not_guaranteed",
            )
    elif name in {
        "Wait",
        "Waitall",
        "Waitany",
        "Waitsome",
        "Test",
        "Testall",
        "Testany",
        "Testsome",
    }:
        contract.update(
            category="completion",
            handle="request",
            operation_completion=(
                "reported_requests_complete_on_success"
                if name.startswith("Wait")
                else "only_requests_reported_complete"
            ),
            request_lifetime="nonpersistent_released_persistent_becomes_inactive",
            unrelated_requests_completed=False,
        )
    else:
        mode, collective = "blocking", name.removesuffix("_c")
        if collective.endswith("_init"):
            mode, collective = "persistent_initialization", collective[:-5]
        elif collective.startswith("I"):
            mode, collective = "nonblocking", collective[1:2].upper() + collective[2:]
        if collective in _COLLECTIVES:
            contract.update(
                category="collective",
                participation="communicator_group_contract",
                mode=mode,
                synchronization=(
                    {
                        "blocking": "on_call_return",
                        "nonblocking": "on_request_completion",
                        "persistent_initialization": "after_start_and_request_completion",
                    }[mode]
                    if collective == "Barrier"
                    else "not_a_general_barrier"
                ),
                unrelated_requests_completed=False,
            )
            if collective == "Barrier":
                contract["synchronization_scope"] = "communicator_barrier"
            if mode != "blocking":
                contract["request_effect"] = (
                    "create_inactive"
                    if mode == "persistent_initialization"
                    else "create_active"
                )
                contract["buffer_lifetime"] = "preserve_start_and_completion_contract"
                contract["communication_started_by_call"] = (
                    mode != "persistent_initialization"
                )
                if mode == "persistent_initialization":
                    contract["activation"] = "MPI_Start_or_MPI_Startall"
    return contract


def qualify(module):
    """Annotate proven MPI targets and merge their native operator requirement."""
    requirements = module.get("operator_requirements", {})
    if not isinstance(requirements, dict):
        return [], [
            dict(code="mpi.requirement", detail="operator requirements must be a table")
        ]
    operations, interfaces = set(), set()
    try:
        if "mpi" in requirements:
            operations, interfaces = _requirement(requirements["mpi"])
    except ValueError as error:
        return [], [dict(code="mpi.requirement", detail=str(error))]

    definitions = {fn["name"] for fn in module.get("functions", ())}
    effective_externs = [
        declaration
        for declaration in module.get("externs", ())
        if declaration.get("name") not in definitions
    ]
    externs, issues = {}, []
    for declaration in effective_externs:
        if declaration.get("provider") != "mpi":
            continue
        name = declaration.get("name")
        if not isinstance(name, str) or not _INTERFACE.fullmatch(name):
            issues.append(
                dict(
                    code="mpi.interface",
                    symbol=name,
                    detail="provider-bound MPI interface requires its exact C symbol",
                )
            )
        else:
            externs[name] = declaration
    interfaces.update(externs)
    # Only executable products and their bound types establish a dependency.
    # Source evidence can still mention a provider whose declaration was
    # replaced by an application definition or removed from the active image.
    pending = [
        module.get("functions", ()),
        module.get("records", ()),
        module.get("globals", ()),
        effective_externs,
    ]
    dependency = bool(externs) or "mpi" in requirements
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            dependency |= value.get("provider") == "mpi"
            pending.extend(value.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)

    annotations = []
    bodies = [module.get(field, ()) for field in ("functions", "globals", "records")]
    for node in walk_operations(bodies):
        op = node["op"]
        bound = []
        non_mpi = []
        if op == "call":
            if node.get("callee") in externs:
                bound = [node["callee"]]
        elif op == "indirect_call":
            targets = node.get("targets", ())
            if not isinstance(targets, (list, tuple)) or any(
                not isinstance(t, str) for t in targets
            ):
                issues.append(
                    dict(
                        code="mpi.targets",
                        detail="indirect targets must be a closed symbol list",
                        location=node.get("location"),
                    )
                )
                continue
            bound = sorted(set(targets).intersection(externs))
            non_mpi = sorted(set(targets).difference(externs))
        if not bound:
            if (
                node.get("operator_family") == "mpi"
                or "mpi_interface" in node
                or "mpi_interfaces" in node
            ):
                issues.append(
                    dict(
                        code="mpi.binding",
                        detail="MPI call annotations require a proven MPI external target",
                        location=node.get("location"),
                    )
                )
            continue
        effects = node.get("effects", [])
        if (
            node.get("operator_family") not in (None, "mpi")
            or not isinstance(effects, list)
            or any(not isinstance(effect, str) for effect in effects)
        ):
            issues.append(
                dict(
                    code="mpi.effects",
                    detail="MPI calls require compatible operator and effect fields",
                    location=node.get("location"),
                )
            )
            continue
        if (
            "mpi_interface" in node
            and (len(bound) != 1 or node["mpi_interface"] != bound[0])
            or "mpi_interfaces" in node
            and node["mpi_interfaces"] != bound
        ):
            issues.append(
                dict(
                    code="mpi.binding",
                    detail="MPI interface annotations disagree with bound targets",
                    location=node.get("location"),
                )
            )
            continue
        annotations.append((node, bound, non_mpi, effects))
        operations.add(op)
    if issues:
        return [], issues
    if not dependency and not annotations:
        return [], []
    for node, bound, non_mpi, effects in annotations:
        node.update(
            operator_family="mpi",
            effects=list(dict.fromkeys([*effects, "STATE", "ORDER", "XFER"])),
            mpi_contract=dict(
                semantics_version=1,
                binding="direct" if node["op"] == "call" else "closed_target_set",
                may_call=True,
                reorderable=False,
                non_mpi_targets=non_mpi,
                interfaces={name: _interface_contract(name) for name in bound},
            ),
        )
        if len(bound) == 1:
            node["mpi_interface"] = bound[0]
        else:
            node["mpi_interfaces"] = bound
    requirement = dict(
        api="c",
        semantics_version=1,
        operations=sorted(operations),
        interfaces=sorted(interfaces),
    )
    module.setdefault("operator_requirements", {})["mpi"] = requirement
    return [
        dict(
            code="qualified_mpi_c_interfaces",
            **requirement,
            proof="provider-bound external C interfaces; preserve calls, effects and control order",
        )
    ], []

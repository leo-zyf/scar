"""B1 selects target resources and seals a relocatable physical execution plan."""

from pathlib import Path

from scar.contract import (
    IR,
    PHYSICAL,
    array_rank_domain,
    copy_member,
    digest,
    document,
    file_receipt,
    json_bytes,
    member,
    open_product,
    read_image_product,
    read_member,
    validate_images,
    write_json,
    walk_operations,
)
from scar.host import array_sdk
from . import mpi

_ARRAY_OPERATIONS = {
    "array_create",
    "array_iota",
    "array_literal",
    "array_copy",
    "array_unary",
    "array_view",
    "array_size",
    "array_rank",
    "array_extent",
    "array_reshape",
    "array_broadcast_copy",
    "array_load",
    "array_store",
    "array_binary",
    "array_update",
    "array_format",
}


def _observed_operators(module):
    # Collect tagged operations and the type-only dependencies of older
    # packages in one traversal of the sealed IR.
    observed, elements, process_requests = {}, set(), []
    defined = {fn["name"] for fn in module.get("functions", ())}
    external = [
        decl for decl in module.get("externs", ()) if decl["name"] not in defined
    ]
    pending = [
        module.get(key, ()) for key in ("functions", "globals", "records", "entry_abi")
    ]
    pending.append(external)
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            op = node.get("op")
            if op is not None:
                family = node.get("operator_family")
                if family is not None:
                    if (
                        not isinstance(family, str)
                        or not family
                        or not isinstance(op, str)
                    ):
                        raise ValueError("invalid operator family or operation tag")
                    observed.setdefault(family, set()).add(op)
                if isinstance(op, str) and op in _ARRAY_OPERATIONS:
                    observed.setdefault("ndarray", set()).add(op)
                if op in {"array_text_read", "array_source_open"}:
                    observed.setdefault(
                        (
                            "text_array_io"
                            if op == "array_text_read"
                            else "text_array_source"
                        ),
                        set(),
                    ).add(op)
                if (
                    op in ("process_spawn", "process_request")
                    or op == "host_call"
                    and node.get("interface") in {"process.stdout", "process.wait"}
                ):
                    observed.setdefault("process", set()).add(op)
                if op == "process_request":
                    process_requests.append(node)
            if node.get("kind") == "ndarray":
                observed.setdefault("ndarray", set())
                element = node.get("element")
                elements.add(element if isinstance(element, str) else "unspecified")
            if mpi.dependency(node):
                observed.setdefault("mpi", set())
            pending.extend(node.values())
        elif isinstance(node, (tuple, list)):
            pending.extend(node)
    return observed, elements, process_requests


def _ndarray_binding(requirement, observed):
    if (
        not isinstance(requirement, dict)
        or requirement.keys() - {"device", "element", "operations"}
        or requirement.get("device", "cpu") != "cpu"
        or requirement.get("element", "f64") != "f64"
        or not isinstance(requirement.get("operations", []), list)
        or any(not isinstance(op, str) for op in requirement.get("operations", ()))
    ):
        raise ValueError("requires CPU binary64 arrays and registered array operations")
    missing = sorted(
        observed.union(requirement.get("operations", ())) - _ARRAY_OPERATIONS
    )
    if missing:
        raise ValueError("unavailable CPU ndarray operations: " + ", ".join(missing))
    return dict(
        array_sdk(required=True),
        device="cpu",
        selection="fixed semantic provider; no global performance optimum claimed",
    )


def _text_array_binding(requirement, observed, module):
    import ctypes
    import platform

    profile = "numpy-2.5.5-whitespace-f64-v1"
    if (
        not isinstance(requirement, dict)
        or requirement.keys() - {"operations", "profile"}
        or requirement.get("profile", profile) != profile
        or not isinstance(requirement.get("operations", []), list)
        or observed.union(requirement.get("operations", ())) - {"array_text_read"}
    ):
        raise ValueError("requires the registered whitespace binary64 text parser")
    for node in walk_operations(
        [module.get(key, ()) for key in ("functions", "globals", "records")]
    ):
        if node["op"] != "array_text_read":
            continue
        if node.get("source_origin") != {"kind": "named_text_file"}:
            raise ValueError(
                "text input requires a proven borrowed named UTF-8 file stream; unresolved source/diagnostic identity cannot select this provider"
            )
        parser = node.get("parser", {})
        if (
            parser.get("profile") != profile
            or parser.get("encoding") != "unicode"
            or parser.get("delimiter") != "whitespace_runs"
            or parser.get("comments") != "#"
            or type(parser.get("ndmin")) is not int
            or parser["ndmin"] != 0
            or node.get("source", {}).get("type")
            != dict(kind="handle", resource="text_stream")
            or set(node.get("effects", ())) != {"IO", "STATE", "VAL", "MEM"}
            or array_rank_domain(node.get("type")) != (0, 1, 2)
            or node["type"]["element"] != "f64"
        ):
            raise ValueError("unregistered text parsing contract")
        location, warning = node.get("location", {}), node.get("warning", {})
        if (
            not isinstance(location.get("diagnostic_file"), str)
            or not location["diagnostic_file"]
            or type(location.get("line")) is not int
            or location["line"] <= 0
            or warning.get("operation") != "loadtxt"
            or not isinstance(warning.get("module"), str)
            or not warning["module"]
            or not isinstance(warning.get("source_line"), str)
            or not warning["source_line"]
        ):
            raise ValueError("text input requires its captured warning site and line")
    if platform.system() != "Linux":
        raise ValueError("the registered text parser requires Linux locale APIs")
    library = ctypes.CDLL(None)
    if any(
        not hasattr(library, name) for name in ("strtod_l", "newlocale", "freelocale")
    ):
        raise ValueError("native C-locale number conversion is unavailable")
    return dict(
        provider="scar-cpu-text-input",
        profile=profile,
        semantics_version=1,
        source="borrowed_named_utf8_stream",
        numeric_converter="system-libc:strtod_l:C",
    )


_OPERATOR_BINDINGS = {"ndarray": _ndarray_binding}


def _process_binding(requirement, observed, module, context):
    import ctypes
    import platform

    if platform.system() != "Linux" or not Path("/proc/self/exe").is_file():
        raise ValueError("image processes require the Linux executable namespace")
    try:
        getattr(ctypes.CDLL(None), "posix_spawn_file_actions_addclosefrom_np")
    except (OSError, AttributeError) as exc:
        raise ValueError(
            "process deployment lacks the required native spawn file actions"
        ) from exc
    if (
        not isinstance(requirement, dict)
        or requirement.keys() - {"operations", "binding"}
        or requirement.get("binding", "declared_images")
        not in ("declared_images", "explicit_targets")
        or not isinstance(requirement.get("operations", []), list)
        or any(not isinstance(op, str) for op in requirement.get("operations", ()))
        or observed.union(requirement.get("operations", ()))
        - {"process_spawn", "host_call"}
    ):
        raise ValueError("requires declared image process operations")
    if not context:
        raise ValueError("process targets require the enclosing sealed image bundle")
    images = context["images"]
    declarations = [
        {k: image[k] for k in ("id", "entry", "output")} for image in images
    ]
    if (
        module.get("execution_images") != declarations
        or module.get("current_image") != context["current_image"]
    ):
        raise ValueError(
            "process image declarations disagree with the enclosing bundle"
        )
    targets = {
        image["id"]: dict(
            image_id=image["id"], output=image["output"], ir_sha256=image["ir_sha256"]
        )
        for image in images
    }
    references, launches = set(), {}
    for node in walk_operations(
        [module.get(k, ()) for k in ("functions", "globals", "records")]
    ):
        if node["op"] == "process_spawn":
            if node.get("image_id") not in targets:
                raise ValueError(
                    "process launch references an undeclared or unsealed image"
                )
            image = node["image_id"]
            references.add(image)
            if "launcher" in node:
                qualify_worker = context.get("qualify_mpi")
                if qualify_worker is None:
                    raise ValueError("MPI launch requires the sealed worker IR package")
                _, worker = qualify_worker(image)
                site, binding = mpi.qualify_launcher(
                    node, targets[image], worker, context["launcher_cache"]
                )
                if site in launches and launches[site] != binding:
                    raise ValueError("MPI launch site has competing physical bindings")
                launches[site] = binding
                node["launch_binding"] = site
            elif node.get("shell", False) is not False or "source_command" in node:
                raise ValueError(
                    "shell image launch requires an explicitly qualified launcher"
                )
        elif node.get("operator_family") == "process" and (
            node["op"] != "host_call"
            or node.get("interface") not in {"process.stdout", "process.wait"}
        ):
            raise ValueError("unregistered process interface")
    return dict(
        provider="posix_spawn",
        target="linux-cxx20",
        binding="declared_images",
        images={name: targets[name] for name in sorted(references)},
        **({"launches": launches} if launches else {}),
        environment_contract="immutable executable generation during execution; inherited cwd, environment and input",
    )


def _external_execution_boundaries(module):
    """Native ABI calls need the same image closure as source process protocols."""
    defined = {fn["name"] for fn in module.get("functions", ())}
    interfaces = {
        "system",
        "popen",
        "fork",
        "vfork",
        "execl",
        "execle",
        "execlp",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "fexecve",
        "posix_spawn",
        "posix_spawnp",
        "dlopen",
        "dlmopen",
    }
    external = {
        decl["name"]: decl
        for decl in module.get("externs", ())
        if decl["name"] not in defined
        and decl.get("provider") in {"libc", "posix"}
        and decl["name"].rsplit("::", 1)[-1] in interfaces
    }
    issues, seen = [], set()
    for node in walk_operations(
        [module.get("functions", ()), module.get("globals", ())]
    ):
        targets = (
            [node.get("callee")]
            if node["op"] == "call"
            else node.get("targets", ()) if node["op"] == "indirect_call" else ()
        )
        for name in targets:
            if name not in external or (name, node.get("site")) in seen:
                continue
            seen.add((name, node.get("site")))
            issues.append(
                dict(
                    stage="B1",
                    code="missing_execution_boundary_binding",
                    symbol=name,
                    provider=external[name]["provider"],
                    site=node.get("site"),
                    location=node.get("location"),
                    detail="native process execution or dynamic loading requires an explicit image/tool binding; a library ABI alone does not close the application graph",
                )
            )
    return issues


def _qualify(module, image_context=None):
    requested = module.get("operator_requirements", {})
    if not isinstance(requested, dict):
        raise ValueError("operator requirements must be a family-indexed table")
    requested = dict(requested)
    observed, array_elements, process_requests = _observed_operators(module)
    for family, operators in observed.items():
        if family not in requested:
            requirement = dict(operations=sorted(operators))
            if family == "process" and process_requests:
                requirement["binding"] = "explicit_targets"
            if family == "ndarray":
                requirement["device"] = "cpu"
                if len(array_elements) > 1:
                    requirement["elements"] = sorted(array_elements)
                else:
                    requirement["element"] = next(iter(array_elements), "f64")
            requested[family] = requirement
    bindings, issues = {}, _external_execution_boundaries(module)
    process = requested.get("process")
    if (
        not process_requests
        and isinstance(process, dict)
        and isinstance(process.get("operations"), list)
        and "process_request" in process["operations"]
    ):
        # A family declaration alone supplies no executable target either.
        process_requests = [{}]
    if process_requests:
        for request in process_requests:
            issues.append(
                dict(
                    stage="B1",
                    code="missing_process_target_binding",
                    requirement="process",
                    target="cpu-cxx20",
                    request_id=request.get("request_id"),
                    site=request.get("site"),
                    location=request.get("location"),
                    binding_obligations=request.get("binding_obligations", []),
                    detail="logical process request requires a qualified executable target before physical scheduling",
                )
            )
    for family, requirement in sorted(requested.items()):
        resolve = _OPERATOR_BINDINGS.get(family)
        if (process_requests or "text_array_source" in observed) and (
            resolve is not None or family in {"mpi", "process", "text_array_io"}
        ):
            # Deployment cannot proceed. Report other unregistered families
            # below without probing any physical providers.
            continue
        if family == "mpi":
            try:
                resolve_worker = (image_context or {}).get("qualify_mpi")
                result = (
                    resolve_worker(image_context["current_image"], module)
                    if resolve_worker
                    else mpi.qualify(
                        module,
                        requirement,
                        explicit=family in module.get("operator_requirements", {}),
                    )
                )
                requested[family], bindings[family] = result
                continue
            except ValueError as exc:
                detail = str(exc)
        elif family == "process":
            try:
                bindings[family] = _process_binding(
                    requirement, observed.get(family, set()), module, image_context
                )
                continue
            except ValueError as exc:
                detail = str(exc)
        elif family == "text_array_io":
            try:
                bindings[family] = _text_array_binding(
                    requirement, observed.get(family, set()), module
                )
                continue
            except ValueError as exc:
                detail = str(exc)
        elif family == "text_array_source":
            detail = "NumPy path resolution includes local/compressed alternatives and URL caching; no qualified native resolver is registered"
        elif resolve is None:
            detail = "no target implementation is registered for this operator family"
        elif family == "ndarray" and array_elements - {"f64"}:
            detail = "unsupported CPU array element types: " + ", ".join(
                sorted(array_elements - {"f64"})
            )
        else:
            try:
                bindings[family] = resolve(requirement, observed.get(family, set()))
                continue
            except ValueError as exc:
                detail = str(exc)
        issues.append(
            dict(
                stage="B1",
                code="missing_operator_binding",
                requirement=family,
                target="cpu-cxx20",
                detail=detail,
            )
        )
    return requested, bindings, issues


def _load_module(path, ir):
    module = dict(
        ir["module"], functions=[read_member(path.parent, r) for r in ir["functions"]]
    )
    if "resource_flow" in module.get("resource_model", {}):
        inventory = dict(module["resource_model"])
        inventory["resource_flow"] = read_member(
            path.parent, inventory["resource_flow"]
        )
        module["resource_model"] = inventory
    return module


def _schedule_image(
    path, revision, ir, destination, policy, workers, image_context=None
):
    module = _load_module(path, ir)
    inventory = dict(module.get("resource_model", {}))
    if "artifact_flow" in inventory:
        inventory["artifact_flow"] = copy_member(
            path.parent, inventory["artifact_flow"], destination, "analysis"
        )
        module["resource_model"] = inventory
    module.pop("image_bindings", None)
    module.pop("process_launch_bindings", None)
    requirements, bindings, issues = _qualify(module, image_context)
    if "process" in bindings:
        module["image_bindings"] = bindings["process"]["images"]
        if "launches" in bindings["process"]:
            module["process_launch_bindings"] = {
                site: digest(json_bytes(binding))
                for site, binding in bindings["process"]["launches"].items()
            }
    status = "INCOMPLETE" if issues else "READY"
    fields = dict(stage="B1", status=status, ir_sha256=revision)
    if issues:
        fields["issues"] = issues
        physical = document(PHYSICAL, target="cpu-cxx20", **fields)
        scheduled = dict(entry=module["entry"])
        if "artifact_flow" in inventory:
            scheduled["module"] = dict(
                entry=module["entry"],
                resource_model={"artifact_flow": inventory["artifact_flow"]},
            )
    else:
        from .cpu import schedule

        mpi = bindings.get("mpi", {})
        uses_mpi = bool(mpi.get("called_or_possible") or mpi.get("address_taken"))
        selected, physical = schedule(module, policy, 1 if uses_mpi else workers)
        if uses_mpi:
            # No rank placement or guaranteed MPI thread level is sealed yet.
            # The cap applies to SCAR's generated teams, including calibration.
            worker_policy = physical["worker_policy"]
            caps = worker_policy["caps"]
            caps["mpi_intra_rank_limit"] = 1
            if workers is None:
                del caps["requested_worker_limit"]
            else:
                caps["requested_worker_limit"] = workers
            worker_policy["mpi"] = dict(
                scope="scar_generated_intra_rank_workers",
                caller_limit=workers,
                reason="rank_placement_and_provided_thread_level_unverified",
            )
        physical.update(fields)
        functions = [
            member(destination, "scheduled-units", fn)
            for fn in selected.pop("functions")
        ]
        if "resource_flow" in selected.get("resource_model", {}):
            inventory = dict(selected["resource_model"])
            inventory["resource_flow"] = member(
                destination, "resources", inventory["resource_flow"]
            )
            selected["resource_model"] = inventory
        scheduled = dict(module=selected, functions=functions)
    if requirements:
        physical["operator_requirements"] = requirements
    if bindings:
        physical["operator_backends"] = bindings
    inventory = module.get("resource_model")
    if inventory:
        physical["source_resources"] = dict(
            sha256=digest(json_bytes(inventory)),
            count=len(inventory.get("resources", ())),
        )
        if "resource_flow" in inventory:
            physical["resource_flow"] = member(
                destination, "resources", inventory["resource_flow"]
            )
    physical_member = member(destination, "scheduled-units", physical)
    write_json(destination / "physical-plan.json", physical)
    package = document(
        "scar.scheduled-ir",
        **fields,
        **scheduled,
        physical=physical_member,
    )
    path = destination / "scheduled-ir.json"
    write_json(path, package)
    return path, status, issues


def schedule_ir(source, destination, *, policy="auto", workers=None):
    if workers is not None and (type(workers) is not int or workers < 1):
        raise ValueError("worker_limit must be a positive integer")
    path, revision, ir = open_product(source, "exec-ir.json", IR)
    if ir.get("status") != "READY":
        raise ValueError("incomplete execution IR; consult its logical plan")
    destination = Path(destination).resolve()
    if ir.get("kind") != "image_bundle":
        scheduled, _, _ = _schedule_image(
            path, revision, ir, destination, policy, workers
        )
        return scheduled

    images, root = ir["images"], ir["root_image"]
    validate_images(images, root)
    audit = (
        {"workflow": copy_member(path.parent, ir["workflow"], destination, "analysis")}
        if "workflow" in ir
        else {}
    )
    inputs = []
    for image in images:
        child = read_image_product(path.parent, image, "ir", "exec-ir.json", IR)
        child_ir = child[2]
        if (
            child_ir.get("status") != "READY"
            or child_ir.get("kind") == "image_bundle"
            or child_ir.get("module", {}).get("entry") != image["entry"]
        ):
            raise ValueError(f"image {image['id']} requires a ready IR for its entry")
        inputs.append((image, child))

    scheduled, plans, issues = [], [], []
    catalog = [
        {k: image[k] for k in ("id", "entry", "output")} | {"ir_sha256": child[1]}
        for image, child in inputs
    ]
    child_inputs = {image["id"]: child for image, child in inputs}
    qualified, launcher_cache = {}, {}

    def qualify_mpi(image_id, module=None):
        # A controller may precede its worker in the input manifest. Qualify
        # the worker itself, reusing the result for its own schedule.
        if image_id not in qualified:
            if module is None:
                child_path, _, child_ir = child_inputs[image_id]
                module = _load_module(child_path, child_ir)
            claims = module.get("operator_requirements", {})
            try:
                result = mpi.qualify(
                    module, claims.get("mpi"), explicit="mpi" in claims
                )
                qualified[image_id] = result, None
            except ValueError as exc:
                qualified[image_id] = None, str(exc)
        result, error = qualified[image_id]
        if error is not None:
            raise ValueError(f"MPI capability of image {image_id}: {error}")
        return result

    for image, child in inputs:
        identity = {key: image[key] for key in ("id", "entry", "output")}
        child_dir = destination / "images" / image["id"]
        child_path, child_status, child_issues = _schedule_image(
            *child,
            child_dir,
            policy,
            workers,
            image_context=dict(
                current_image=image["id"],
                images=catalog,
                qualify_mpi=qualify_mpi,
                launcher_cache=launcher_cache,
            ),
        )
        identity["status"] = child_status
        issues.extend(dict(issue, image_id=image["id"]) for issue in child_issues)
        scheduled.append(
            dict(identity, scheduled=file_receipt(destination, child_path))
        )
        plans.append(
            dict(
                identity,
                physical=file_receipt(destination, child_dir / "physical-plan.json"),
            )
        )
    fields = dict(
        kind="image_bundle",
        stage="B1",
        status="INCOMPLETE" if issues else "READY",
        root_image=root,
        ir_sha256=revision,
    )
    if issues:
        fields["issues"] = issues
    physical_path = destination / "physical-plan.json"
    write_json(
        physical_path,
        document(
            PHYSICAL,
            **fields,
            planning_scope="per_image",
            interprocess_schedule="unmodeled",
            images=plans,
        ),
    )
    package = document(
        "scar.scheduled-ir",
        **fields,
        **audit,
        images=scheduled,
        physical=file_receipt(destination, physical_path),
    )
    result = destination / "scheduled-ir.json"
    write_json(result, package)
    return result

"""B1 selects target resources and seals a relocatable physical execution plan."""

from pathlib import Path

from scar.contract import (
    IR,
    PHYSICAL,
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

_ARRAY_OPERATIONS = {
    "array_create",
    "array_iota",
    "array_literal",
    "array_copy",
    "array_unary",
    "array_view",
    "array_size",
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
    observed, elements = {}, set()
    pending = [module]
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
                if (
                    op == "process_spawn"
                    or op == "host_call"
                    and node.get("interface") in {"process.stdout", "process.wait"}
                ):
                    observed.setdefault("process", set()).add(op)
            if node.get("kind") == "ndarray":
                observed.setdefault("ndarray", set())
                element = node.get("element")
                elements.add(element if isinstance(element, str) else "unspecified")
            pending.extend(node.values())
        elif isinstance(node, (tuple, list)):
            pending.extend(node)
    return observed, elements


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
        or requirement.get("binding", "declared_images") != "declared_images"
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
    references = set()
    for node in walk_operations(module):
        if node["op"] == "process_spawn":
            if node.get("image_id") not in targets:
                raise ValueError(
                    "process launch references an undeclared or unsealed image"
                )
            references.add(node["image_id"])
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
    observed, array_elements = _observed_operators(module)
    for family, operators in observed.items():
        if family not in requested:
            requirement = dict(operations=sorted(operators))
            if family == "ndarray":
                requirement["device"] = "cpu"
                if len(array_elements) > 1:
                    requirement["elements"] = sorted(array_elements)
                else:
                    requirement["element"] = next(iter(array_elements), "f64")
            requested[family] = requirement
    bindings, issues = {}, _external_execution_boundaries(module)
    for family, requirement in sorted(requested.items()):
        resolve = _OPERATOR_BINDINGS.get(family)
        if family == "process":
            try:
                bindings[family] = _process_binding(
                    requirement, observed.get(family, set()), module, image_context
                )
                continue
            except ValueError as exc:
                detail = str(exc)
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


def _schedule_image(
    path, revision, ir, destination, policy, workers, image_context=None
):
    module = dict(
        ir["module"], functions=[read_member(path.parent, r) for r in ir["functions"]]
    )
    module.pop("image_bindings", None)
    requirements, bindings, issues = _qualify(module, image_context)
    if "process" in bindings:
        module["image_bindings"] = bindings["process"]["images"]
    status = "INCOMPLETE" if issues else "READY"
    fields = dict(stage="B1", status=status, ir_sha256=revision)
    if issues:
        fields["issues"] = issues
        physical = document(PHYSICAL, target="cpu-cxx20", **fields)
        scheduled = dict(entry=module["entry"])
    else:
        from .cpu import schedule

        selected, physical = schedule(module, policy, workers)
        physical.update(fields)
        functions = [
            member(destination, "scheduled-units", fn)
            for fn in selected.pop("functions")
        ]
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
    for image, child in inputs:
        identity = {key: image[key] for key in ("id", "entry", "output")}
        child_dir = destination / "images" / image["id"]
        child_path, child_status, child_issues = _schedule_image(
            *child,
            child_dir,
            policy,
            workers,
            image_context=dict(current_image=image["id"], images=catalog),
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
        images=scheduled,
        physical=file_receipt(destination, physical_path),
    )
    result = destination / "scheduled-ir.json"
    write_json(result, package)
    return result

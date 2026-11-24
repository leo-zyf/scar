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
)
from scar.host import array_sdk

_ARRAY_OPERATIONS = {
    "array_create",
    "array_iota",
    "array_literal",
    "array_copy",
    "array_unary",
    "array_view",
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


def _qualify(module):
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
    bindings, issues = {}, []
    for family, requirement in sorted(requested.items()):
        resolve = _OPERATOR_BINDINGS.get(family)
        if resolve is None:
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


def _schedule_image(path, revision, ir, destination, policy, workers):
    module = dict(
        ir["module"], functions=[read_member(path.parent, r) for r in ir["functions"]]
    )
    requirements, bindings, issues = _qualify(module)
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
    for image, child in inputs:
        identity = {key: image[key] for key in ("id", "entry", "output")}
        child_dir = destination / "images" / image["id"]
        child_path, child_status, child_issues = _schedule_image(
            *child, child_dir, policy, workers
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

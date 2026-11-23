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


def _schedule_image(path, revision, ir, destination, policy, workers):
    from .cpu import schedule

    module = dict(
        ir["module"], functions=[read_member(path.parent, r) for r in ir["functions"]]
    )
    selected, physical = schedule(module, policy, workers)
    physical["ir_sha256"] = revision
    inventory = module.get("resource_model")
    if inventory:
        physical["source_resources"] = dict(
            sha256=digest(json_bytes(inventory)),
            count=len(inventory.get("resources", ())),
        )
    functions = [
        member(destination, "scheduled-units", fn) for fn in selected.pop("functions")
    ]
    physical_member = member(destination, "scheduled-units", physical)
    write_json(destination / "physical-plan.json", physical)
    package = document(
        "scar.scheduled-ir",
        stage="B1",
        status="READY",
        ir_sha256=revision,
        module=selected,
        functions=functions,
        physical=physical_member,
    )
    path = destination / "scheduled-ir.json"
    write_json(path, package)
    return path


def schedule_ir(source, destination, *, policy="auto", workers=None):
    path, revision, ir = open_product(source, "exec-ir.json", IR)
    if ir.get("status") != "READY":
        raise ValueError("incomplete execution IR; consult its logical plan")
    destination = Path(destination).resolve()
    if ir.get("kind") != "image_bundle":
        return _schedule_image(path, revision, ir, destination, policy, workers)

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

    scheduled, plans = [], []
    for image, child in inputs:
        identity = {key: image[key] for key in ("id", "entry", "output")}
        child_dir = destination / "images" / image["id"]
        child_path = _schedule_image(*child, child_dir, policy, workers)
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
        status="READY",
        root_image=root,
        ir_sha256=revision,
    )
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

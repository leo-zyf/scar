"""B1 selects target resources and seals a relocatable physical execution plan."""

from pathlib import Path

from scar.contract import (
    IR,
    digest,
    document,
    json_bytes,
    member,
    open_product,
    read_member,
    write_json,
)


def schedule_ir(source, destination, *, policy="auto", workers=None):
    from .cpu import schedule

    path, revision, ir = open_product(source, "exec-ir.json", IR)
    if ir.get("status") != "READY":
        raise ValueError("incomplete execution IR; consult its logical plan")
    destination = Path(destination).resolve()
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

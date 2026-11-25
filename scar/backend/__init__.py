"""Run the independent CPU scheduling, C++ lowering, printing, and build layers."""

from pathlib import Path
from tempfile import TemporaryDirectory
from scar.contract import (
    PHYSICAL,
    document,
    file_receipt,
    open_product,
    read_member,
    write_json,
)


def _emit(schedule, folder):
    from .cpp20 import lower_scheduled, print_target
    from .build import build

    target = lower_scheduled(schedule, folder)
    project = print_target(target, folder)
    executable = build(project["cpp20_manifest"])
    return project, executable


def _deliver_components(out, physical, receipt):
    package = read_member(out, receipt)
    _, _, application = open_product(out, "scheduled-ir.json", "scar.scheduled-ir")
    component_plan = read_member(out, package["physical"])
    ready = {image["id"] for image in physical["images"] if image["status"] == "READY"}
    selected = {image["id"] for image in package["images"]}
    if (
        package.get("delivery_scope") != "closed_components"
        or package.get("ir_sha256") != physical["ir_sha256"]
        or package.get("application_root") != physical["root_image"]
        or physical["root_image"] in selected
        or not selected <= ready
        or package["images"]
        != [row for row in application["images"] if row["id"] in selected]
        or component_plan["images"]
        != [row for row in physical["images"] if row["id"] in selected]
    ):
        raise ValueError("component schedule disagrees with the application plan")
    destination = out / "components"
    with TemporaryDirectory(prefix=".scar-components-", dir=out) as temporary:
        staging = Path(temporary) / "components"
        _emit(out / receipt["path"], staging)
        _, _, built = open_product(staging, "build.json", "scar.cpp20-executable")
        if set(built["executables"]) != selected:
            raise ValueError("component delivery differs from the scheduled image set")
        staging.rename(destination)
    return dict(
        scheduled=receipt,
        target=file_receipt(out, destination / "target-ir.json"),
        sources=file_receipt(out, destination / "cpp20.json"),
        build=file_receipt(out, destination / "build.json"),
        executables={
            name: {**binary, "path": "components/" + binary["path"]}
            for name, binary in built["executables"].items()
        },
    )


def compile_ir(source, out, *, policy="auto", workers=None):
    from .planning import schedule_ir

    out = Path(out).resolve()
    if (out / "project").exists() or (out / "components").exists():
        raise ValueError(f"delivery already contains a generated project: {out}")
    schedule = schedule_ir(source, out, policy=policy, workers=workers)
    physical_path, _, physical = open_product(out, "physical-plan.json", PHYSICAL)
    if physical["status"] == "INCOMPLETE":
        result = document(
            "scar.cpp20-delivery",
            status="INCOMPLETE",
            scheduled_ir=str(schedule),
            physical_plan=str(physical_path),
            issues=physical["issues"],
        )
        if physical.get("kind") == "image_bundle":
            result.update(
                kind="image_bundle",
                root_image=physical["root_image"],
                images=[
                    {k: row[k] for k in ("id", "entry", "output", "status")}
                    for row in physical["images"]
                ],
            )
            if receipt := physical.get("component_schedule"):
                result["component_delivery"] = _deliver_components(
                    out, physical, receipt
                )
        write_json(out / "result.json", result)
        return result
    project, executable = _emit(schedule, out)
    images = {}
    if project.get("kind") == "image_bundle":
        _, _, built = open_product(out, "build.json", "scar.cpp20-executable")
        images["executables"] = {
            key: str(out / row["path"]) for key, row in built["executables"].items()
        }
    result = document(
        "scar.cpp20-delivery",
        status="COMPILED",
        scheduled_ir=str(schedule),
        **project,
        **images,
        executable=str(executable),
        physical_plan=str(physical_path),
    )
    write_json(out / "result.json", result)
    return result

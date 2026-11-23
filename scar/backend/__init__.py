"""Run the independent CPU scheduling, C++ lowering, printing, and build layers."""

from pathlib import Path
from scar.contract import PHYSICAL, document, open_product, write_json


def compile_ir(source, out, *, policy="auto", workers=None):
    from .planning import schedule_ir

    out = Path(out).resolve()
    if (out / "project").exists():
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
        write_json(out / "result.json", result)
        return result
    from .cpp20 import lower_scheduled, print_target
    from .build import build

    target = lower_scheduled(schedule, out)
    project = print_target(target, out)
    executable = build(project["cpp20_manifest"])
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

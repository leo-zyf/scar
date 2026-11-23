"""Run the independent CPU scheduling, C++ lowering, printing, and build layers."""

from pathlib import Path
from scar.contract import document, write_json


def compile_ir(source, out, *, policy="auto", workers=None):
    from .planning import schedule_ir
    from .cpp20 import lower_scheduled, print_target
    from .build import build

    out = Path(out).resolve()
    if (out / "project").exists():
        raise ValueError(f"delivery already contains a generated project: {out}")
    schedule = schedule_ir(source, out, policy=policy, workers=workers)
    target = lower_scheduled(schedule, out)
    project = print_target(target, out)
    executable = build(project["cpp20_manifest"])
    result = document(
        "scar.cpp20-delivery",
        status="COMPILED",
        scheduled_ir=str(schedule),
        **project,
        executable=str(executable),
        physical_plan=str(out / "physical-plan.json"),
    )
    write_json(out / "result.json", result)
    return result

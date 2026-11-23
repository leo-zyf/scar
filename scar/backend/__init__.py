"""Select a target schedule, then realize it from a self-contained execution IR."""

from pathlib import Path

from scar.contract import IR, document, open_product, read_member, write_json
from .cpu import schedule
from .emission import emit


def compile_ir(source, out, *, policy="auto", workers=None):
    path, revision, ir = open_product(source, "exec-ir.json", IR)
    if ir.get("status") != "READY":
        raise ValueError(
            "the execution IR is incomplete; see the producing stage's plan.json"
        )
    out = Path(out).resolve()
    if (out / "project").exists():
        raise ValueError(f"delivery already contains a generated project: {out}")
    module = dict(
        ir["module"], functions=[read_member(path.parent, r) for r in ir["functions"]]
    )
    selected, physical = schedule(module, policy, workers)
    physical["ir_sha256"] = revision
    write_json(out / "physical-plan.json", physical)
    project = emit(selected, physical, out)
    result = document(
        "scar.cpp20-delivery",
        status="COMPILED",
        ir_sha256=revision,
        **project,
        physical_plan=str(out / "physical-plan.json"),
    )
    write_json(out / "result.json", result)
    return result

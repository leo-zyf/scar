"""Realize a sealed SCAR plan as a C++20 project and validate its entry."""

from dataclasses import asdict, replace
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from ..contract import artifact, open_plan, verify_artifact, write_json
from .cmake_build import build_project
from .validation import validate_cases, verify_exports


def _toolchain():
    compiler = shutil.which("g++")
    if compiler is None:
        raise ValueError("g++ is required for the CPU C++20 backend")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("LD_") and key not in {"CXXFLAGS", "CPPFLAGS", "LDFLAGS"}
    }
    return str(Path(compiler).resolve(strict=True)), dict(environment, LC_ALL="C")


def compile_plan(
    path: Path,
    out: Path,
    *,
    case_seconds: int = 30,
    repetitions: int = 1,
    cpu_threads: int | None = None,
    warmups: int = 1,
) -> dict:
    """Read a SCAR plan and publish its backend result without loading another compiler stage."""
    path, out = Path(path), Path(out).resolve()
    if path.is_dir():
        path /= "plan.json"
    if (
        type(case_seconds) is not int
        or case_seconds < 1
        or type(repetitions) is not int
        or repetitions < 1
    ):
        raise ValueError(
            "native delivery requires a positive case time budget and repetition count"
        )
    if type(warmups) is not int or warmups < 0:
        raise ValueError("warmups must be a nonnegative integer")
    if cpu_threads is not None and (type(cpu_threads) is not int or cpu_threads < 1):
        raise ValueError("the worker limit must be a positive integer")
    plan_ref, plan = open_plan(path)
    if out.exists():
        raise FileExistsError(f"backend output already exists: {out}")
    from .cpu_schedule import physical_plan
    from .cpp20_codegen import emit_cpp20_project

    compiler, environment = _toolchain() if plan.lowering.bodies else (None, {})
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".scar-cpp20-", dir=out.parent) as temporary:
        staging = Path(temporary) / "delivery"
        staging.mkdir()
        cpus, physical = physical_plan(
            plan, Path(temporary), cpu_threads, compiler, environment
        )
        physical_ref = write_json(
            staging / "physical-plan.json", dict(physical, plan=asdict(plan_ref))
        )
        outcome = dict(
            schema="scar.cpp20-delivery",
            schema_version=1,
            plan=asdict(plan_ref),
            status="KEEP_ORIGINAL",
            decision=plan.decision,
            validation=[],
            physical_plan=dict(
                asdict(physical_ref), path=str(out / "physical-plan.json")
            ),
        )
        if plan.lowering.bodies and physical["decision"] != "KEEP_ORIGINAL":
            project = staging / "project"
            configuration, exports, invocation = emit_cpp20_project(
                replace(plan.lowering, cpu_ids=cpus), project, path.parent.resolve()
            )
            executable = build_project(project, configuration, compiler, environment)
            if probe := physical.get("probe"):
                verify_artifact(probe["compiler"])
            verify_exports(executable, exports)
            cases = (
                []
                if invocation
                else validate_cases(
                    plan.validation,
                    path.parent.resolve(),
                    executable,
                    staging,
                    cpus,
                    case_seconds,
                    repetitions,
                    warmups,
                )
            )
            result = dict(
                asdict(artifact(executable)),
                path=str(out / executable.relative_to(staging)),
            )
            status = "TRIAL_COMPILED" if invocation else "BUILT"
            if cases:
                status = (
                    "VALIDATED"
                    if all(case["matches"] for case in cases)
                    else "VALIDATION_FAILED"
                )
            outcome.update(
                status=status,
                project=str(out / "project"),
                executable=result,
                cpus=list(cpus),
                compiler=compiler,
                exports=exports,
                validation=cases,
            )
            if invocation:
                outcome.update(invocation=invocation, deployment_authorized=False)
        write_json(staging / "result.json", outcome)
        verify_artifact(plan_ref)
        staging.rename(out)
    return outcome

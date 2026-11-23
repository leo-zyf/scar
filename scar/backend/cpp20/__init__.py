"""B2 produces a relocatable, formatted C++20 project from a scheduled execution IR."""

from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

from scar.contract import (
    digest,
    document,
    member,
    open_product,
    read_member,
    write_json,
)


def _format_workers(jobs, file_count):
    affinity = getattr(os, "sched_getaffinity", None)
    available = len(affinity(0)) if affinity else os.cpu_count() or 1
    limit = min(jobs, 4, file_count, available)
    if limit <= 1 or not Path("/proc/self/cgroup").exists():
        return max(1, limit)

    try:
        groups = {}
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, controllers, group = line.split(":", 2)
            if not controllers:
                groups["cgroup2"] = Path(group)
            elif "cpu" in controllers.split(","):
                groups["cgroup"] = Path(group)
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            before, after = line.split(" - ", 1)
            kind, _, options = after.split()
            if (
                kind not in groups
                or kind == "cgroup"
                and "cpu" not in options.split(",")
            ):
                continue
            # A namespace can mount a subtree rather than the hierarchy root.
            mount_root, mount = (
                Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value))
                for value in before.split()[3:5]
            )
            group = groups[kind]
            relative = (
                group.relative_to(mount_root)
                if group.is_relative_to(mount_root)
                else group.relative_to("/")
            )
            folder = (mount / relative).resolve()
            if not folder.is_relative_to(mount):
                folder = mount
            while folder.is_relative_to(mount):
                quota = folder / (
                    "cpu.max" if kind == "cgroup2" else "cpu.cfs_quota_us"
                )
                if quota.is_file():
                    if kind == "cgroup2":
                        amount, period = quota.read_text().split()
                    else:
                        amount = quota.read_text().strip()
                        period = (folder / "cpu.cfs_period_us").read_text().strip()
                    if amount not in {"max", "-1"}:
                        limit = min(limit, max(1, int(amount) // int(period)))
                if folder == mount:
                    break
                folder = folder.parent
    except (OSError, ValueError, ZeroDivisionError):
        return 1
    return limit


def _format_sources(formatter, files, jobs):
    files = sorted(files, key=lambda path: (-path.stat().st_size, str(path)))
    workers = _format_workers(jobs, len(files))
    command = [formatter, "--style=file", "-i"]
    if workers == 1:
        subprocess.run([*command, *files], check=True)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = [
            pool.submit(subprocess.run, [*command, path], check=True) for path in files
        ]
        _, pending = wait(tasks, return_when=FIRST_EXCEPTION)
        for task in pending:
            task.cancel()
    # All active writers have exited before an error can release the staging directory.
    for task in tasks:
        if not task.cancelled():
            task.result()


def _package(root, sources, openmp, libraries, jobs):
    unknown = libraries - {"MPI", "Threads", "DynamicLoader", "libgfortran.so.5"}
    if unknown:
        raise ValueError(
            f"unsupported C++ project link requirements: {sorted(unknown)}"
        )
    formatter = shutil.which("clang-format-18") or shutil.which("clang-format")
    if formatter is None:
        raise ValueError("clang-format is required for LLVM-style C++20 delivery")
    (root / ".clang-format").write_text(
        "BasedOnStyle: LLVM\nStandard: c++20\n", encoding="utf-8"
    )
    generated = [root / name for name in sources]
    generated.extend(sorted((root / "include").rglob("*.hpp")))
    _format_sources(formatter, generated, jobs)
    lines = [
        "cmake_minimum_required(VERSION 3.20)",
        "project(scar_delivery LANGUAGES CXX)",
        "add_executable(scar_entry",
        *(f"    {source}" for source in sources),
        ")",
        "target_include_directories(scar_entry PRIVATE include)",
        "set_target_properties(scar_entry PROPERTIES CXX_STANDARD 20",
        "    CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)",
        'if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")',
        "    target_compile_options(scar_entry PRIVATE -Werror -fno-fast-math -frounding-math -ffp-contract=off)",
        'elseif(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")',
        "    target_compile_options(scar_entry PRIVATE -Werror -Kieee -Mnofma)",
        "endif()",
    ]
    for enabled, package in ((openmp, "OpenMP"), ("MPI" in libraries, "MPI")):
        if enabled:
            lines.extend(
                (
                    f"find_package({package} REQUIRED COMPONENTS CXX)",
                    f"target_link_libraries(scar_entry PRIVATE {package}::{package}_CXX)",
                )
            )
    if "MPI" in libraries:
        lines.append(
            "target_compile_definitions(scar_entry PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)"
        )
    if "Threads" in libraries:
        lines.extend(
            (
                "find_package(Threads REQUIRED)",
                "target_link_libraries(scar_entry PRIVATE Threads::Threads)",
            )
        )
    if "DynamicLoader" in libraries:
        lines.append('target_link_libraries(scar_entry PRIVATE "${CMAKE_DL_LIBS}")')
    if "libgfortran.so.5" in libraries:
        lines.extend(
            (
                "find_library(SCAR_GNU_IO_LIBRARY NAMES libgfortran.so.5 REQUIRED)",
                'target_link_libraries(scar_entry PRIVATE "${SCAR_GNU_IO_LIBRARY}")',
            )
        )
    lines.append("install(TARGETS scar_entry RUNTIME DESTINATION bin)")
    (root / "CMakeLists.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def lower_scheduled(source, destination):
    """B2a converts the sealed B1 product to C++ target trees."""
    from .lowering import Lowerer
    from .model import encode_unit

    path, revision, scheduled = open_product(
        source, "scheduled-ir.json", "scar.scheduled-ir"
    )
    if scheduled.get("status") != "READY":
        raise ValueError("incomplete physical execution schedule")
    physical = read_member(path.parent, scheduled["physical"])
    if (
        physical.get("schema"),
        physical.get("schema_version"),
        physical.get("ir_sha256"),
    ) != ("scar.physical-plan", scheduled["schema_version"], scheduled["ir_sha256"]):
        raise ValueError("physical plan does not belong to its scheduled IR package")
    module = dict(
        scheduled["module"],
        functions=[read_member(path.parent, f) for f in scheduled["functions"]],
    )
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    lowerer = Lowerer(module, physical)
    units = lowerer.units()
    members = [
        dict(output=name, tree=member(destination, "target-units", encode_unit(unit)))
        for name, unit in units.items()
    ]
    support = Path(__file__).with_name("runtime.hpp").read_bytes()
    metadata = dict(
        openmp=lowerer.openmp,
        runtime_libraries=sorted(lowerer.libraries.links),
        coordinate_maps=lowerer.coordinate_maps,
        record_field_offset_checks=lowerer.layout_scope,
    )
    product = document(
        "scar.cpp20-target-ir",
        stage="B2a",
        scheduled_ir_sha256=revision,
        units=members,
        ir_sha256=physical.get("ir_sha256"),
        runtime_sha256=digest(support),
        **metadata,
    )
    path = destination / "target-ir.json"
    write_json(path, product)
    return path


def print_target(target, destination, *, jobs=4):
    """Print a standalone project using at most jobs formatter processes (up to four)."""
    from .model import from_data
    from .printer import print_unit

    if type(jobs) is not int or jobs < 1:
        raise ValueError("formatting jobs must be a positive integer")
    path, revision, product = open_product(
        target, "target-ir.json", "scar.cpp20-target-ir"
    )
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    project = destination / "project"
    if project.exists():
        raise ValueError(f"output project already exists at {project}")
    support = Path(__file__).with_name("runtime.hpp").read_bytes()
    if digest(support) != product["runtime_sha256"]:
        raise ValueError("target package requires a different C++20 runtime revision")
    sources = []
    with TemporaryDirectory(prefix=".scar-project-", dir=destination) as tmp:
        staging = Path(tmp) / "project"
        header = "include/scar/program.hpp"
        (staging / "include/scar").mkdir(parents=True)
        (staging / "src").mkdir()
        (staging / "include/scar/runtime.hpp").write_bytes(support)
        for part in product["units"]:
            output = part["output"]
            if output != header and not re.fullmatch(
                r"src/(program|unit_[0-9]+)\.cpp", output
            ):
                raise ValueError(f"invalid C++ translation unit path {output!r}")
            file = staging / output
            if file.exists():
                raise ValueError(f"duplicate C++ translation unit {output!r}")
            tree = from_data(read_member(path.parent, part["tree"]))
            file.write_text(print_unit(tree), encoding="utf-8")
            if output != header:
                sources.append(output)
        if not sources or not (staging / header).is_file():
            raise ValueError("incomplete target translation-unit set")
        _package(
            staging, sources, product["openmp"], set(product["runtime_libraries"]), jobs
        )
        members = []
        for f in sorted(staging.rglob("*")):
            if f.is_file():
                data = f.read_bytes()
                members.append(
                    dict(
                        path=str(f.relative_to(staging)),
                        sha256=digest(data),
                        size=len(data),
                    )
                )
        staging.rename(project)
    metadata = {
        key: product[key]
        for key in (
            "ir_sha256",
            "openmp",
            "runtime_libraries",
            "coordinate_maps",
            "record_field_offset_checks",
        )
    }
    receipt = document(
        "scar.cpp20-project",
        stage="B2b",
        target_ir_sha256=revision,
        project="project",
        sources=sources,
        header=header,
        members=members,
        **metadata,
    )
    manifest = destination / "cpp20.json"
    write_json(manifest, receipt)
    return dict(
        target_ir=str(path),
        cpp20_manifest=str(manifest),
        project=str(project),
        source=str(project / sources[0]),
        sources=[str(project / s) for s in sources],
        header=str(project / header),
        **metadata,
    )

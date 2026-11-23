"""B2 produces a relocatable, formatted C++20 project from a scheduled execution IR."""

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


def _package(root, sources, openmp, libraries):
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
    subprocess.run([formatter, "--style=file", "-i", *map(str, generated)], check=True)
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


def print_target(target, destination):
    """B2b prints the target-tree package and seals the standalone CMake project."""
    from .model import from_data
    from .printer import print_unit

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
        _package(staging, sources, product["openmp"], set(product["runtime_libraries"]))
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

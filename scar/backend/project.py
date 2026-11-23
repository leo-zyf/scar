"""Package generated C++20 units and build the target with CMake."""

from pathlib import Path
import os
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory


def build(root: Path, *, sources: list[str], openmp: bool, libraries: set[str]) -> Path:
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
    with TemporaryDirectory(prefix="scar-cmake-", dir=root.parent) as scratch:
        compile_command = ["cmake", "--build", scratch]
        if not os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL"):
            parallel = min(len(sources), len(os.sched_getaffinity(0)), 8)
            compile_command.extend(("--parallel", str(parallel)))
        commands = (
            ["cmake", "-S", str(root), "-B", scratch, "-DCMAKE_BUILD_TYPE=Release"],
            compile_command,
            ["cmake", "--install", scratch, "--prefix", str(root)],
        )
        with (root / "build.log").open("w", encoding="utf-8") as log:
            for argv in commands:
                result = subprocess.run(
                    argv, stdout=log, stderr=subprocess.STDOUT, check=False
                )
                if result.returncode:
                    log.flush()
                    tail = (root / "build.log").read_text(errors="replace")[-8000:]
                    raise ValueError(f"C++20 project build failed\n{tail}")
    with (root / "build.log").open(errors="replace") as log:
        for line in log:
            if re.search(r"\bwarning(?:\s*#\d+)?\s*:|^CMake Warning\b", line, re.I):
                raise ValueError(
                    f"C++20 project emitted a build warning: {line.strip()}"
                )
    executable = root / "bin/scar_entry"
    if not os.access(executable, os.X_OK):
        raise ValueError("CMake did not install the executable")
    return executable

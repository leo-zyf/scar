"""Build the emitted C++20 translation units with CMake."""

import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

_TARGET = "scar_entry"
_SOURCES = {".cpp", ".cc", ".cxx", ".C", ".c++", ".cp"}


def _cmake_arg(value):
    if any(token in value for token in (";", "$<", "\n", "\r")):
        raise ValueError(f"unsupported CMake input: {value!r}")
    delimiter = "="
    while f"]{delimiter}]" in value:
        delimiter += "="
    return f"[{delimiter}[{value}]{delimiter}]"


def _cmake_project(root, config):
    sources = config["sources"]
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("C++20 delivery requires distinct source units")
    for name in sources:
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix not in _SOURCES
            or not (root / relative).is_file()
        ):
            raise ValueError(f"missing C++20 translation unit: {name}")
    lines = [
        "cmake_minimum_required(VERSION 3.20)",
        "project(scar_entry LANGUAGES CXX)",
        "add_executable(scar_entry " + " ".join(map(_cmake_arg, sources)) + ")",
        "set_target_properties(scar_entry PROPERTIES CXX_STANDARD 20 "
        "CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)",
    ]
    includes = ("include/scar_generated", "src", *config.get("include_dirs", ()))
    for command, arguments in (
        ("target_include_directories", includes),
        ("target_compile_options", config.get("compile_options", ())),
        ("target_link_libraries", config.get("libraries", ())),
        ("target_link_options", config.get("link_options", ())),
    ):
        if arguments:
            lines.append(
                f"{command}(scar_entry PRIVATE "
                + " ".join(map(_cmake_arg, arguments))
                + ")"
            )
    for name, (directories, flags) in config.get("source_settings", {}).items():
        if name not in sources:
            raise ValueError(f"source settings have no C++20 unit: {name}")
        source = _cmake_arg(name)
        if directories:
            lines.append(
                "set(scar_source_includes "
                + " ".join(map(_cmake_arg, directories))
                + ")"
            )
            lines.append(
                'list(TRANSFORM scar_source_includes PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")'
            )
            lines.append(
                f"set_property(SOURCE {source} PROPERTY INCLUDE_DIRECTORIES ${{scar_source_includes}})"
            )
        if flags:
            lines.append(
                f"set_property(SOURCE {source} PROPERTY COMPILE_OPTIONS "
                + " ".join(map(_cmake_arg, flags))
                + ")"
            )
    if config.get("runtime_dirs"):
        paths = " ".join(map(_cmake_arg, config["runtime_dirs"]))
        lines.extend(
            (
                f"set_property(TARGET scar_entry PROPERTY BUILD_RPATH {paths})",
                f"set_property(TARGET scar_entry PROPERTY INSTALL_RPATH {paths})",
            )
        )
    lines.append("install(TARGETS scar_entry RUNTIME DESTINATION bin)")
    (root / "CMakeLists.txt").write_text("\n".join(lines) + "\n")


def build_project(project, config, compiler, env):
    _cmake_project(project, config)
    logs = []
    with TemporaryDirectory(prefix=".build-", dir=project.parent) as build_dir:
        parallel = min(len(os.sched_getaffinity(0)), 8)
        commands = (
            [
                "cmake",
                "-S",
                str(project),
                "-B",
                build_dir,
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_CXX_COMPILER={compiler}",
            ],
            ["cmake", "--build", build_dir, "--parallel", str(parallel)],
            ["cmake", "--install", build_dir, "--prefix", str(project)],
        )
        for command in commands:
            run = subprocess.run(
                command, env=env, capture_output=True, text=True, timeout=900
            )
            logs.extend((run.stdout, run.stderr))
            if run.returncode:
                raise ValueError(
                    f"C++20 project build failed\n{run.stderr}\n{run.stdout}"
                )
    (project / "build.log").write_text("".join(logs))
    executable = project / "bin" / _TARGET
    listing = subprocess.run(
        ["ldd", str(executable)], env=env, capture_output=True, text=True, timeout=30
    )
    if listing.returncode:
        raise ValueError(
            f"could not inspect linked C++20 dependencies\n{listing.stderr}"
        )
    forbidden = ("libpython", "libtorch_python", "libgfortran")
    if "not found" in listing.stdout or any(
        line.lstrip().startswith(forbidden) for line in listing.stdout.splitlines()
    ):
        raise ValueError(
            "C++20 executable has unresolved or Python/Fortran runtime dependencies"
        )
    (project / "ldd.txt").write_text(listing.stdout)
    return executable

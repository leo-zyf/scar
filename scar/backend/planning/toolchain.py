"""B1 target-compiler capabilities and whole-image compilation decisions."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
from tempfile import TemporaryDirectory

from scar.contract import digest, json_bytes, validate_access_contract, walk_operations
from scar.host import cpu_availability


def _file(path):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    return dict(path=str(path), size=len(data), sha256=digest(data))


def _access_feature(cache):
    """Compile a fixed feature probe; never execute an application or the probe."""
    arguments = shlex.split(os.environ.get("CXX") or "c++")
    compiler = shutil.which(arguments[0]) if arguments else None
    if not compiler:
        return dict(supported=False, reason="target C++ compiler unavailable")
    source = Path(__file__).with_name("access_probe.cpp")
    qualification = dict(
        compiler=_file(compiler),
        compiler_arguments=arguments[1:],
        probe_sha256=digest(source.read_bytes()),
        definition="SCAR_CXX_ACCESS_PROBE=1",
        cxx_standard=20,
        evidence="compile_only",
        application_executed=False,
        scope="qualification host; deployment compiler must support this parameter spelling",
    )
    key = digest(json_bytes(qualification))
    if key not in cache:
        with TemporaryDirectory(prefix="scar-cxx-access-") as build:
            try:
                result = subprocess.run(
                    [
                        compiler,
                        *arguments[1:],
                        "-std=c++20",
                        "-Werror",
                        "-pedantic-errors",
                        "-DSCAR_CXX_ACCESS_PROBE=1",
                        "-c",
                        str(source),
                        "-o",
                        str(Path(build) / "probe.o"),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                qualification["supported"] = result.returncode == 0
                if result.returncode:
                    qualification["reason"] = (
                        "target compiler rejected the parameter feature"
                    )
                    qualification["diagnostic"] = result.stderr[-4000:]
            except (OSError, subprocess.TimeoutExpired) as exc:
                qualification.update(supported=False, reason=str(exc))
        cache[key] = qualification
    return cache[key]


def parameter_access(module, *, cache=None):
    """Bind only invocation-wide neutral proofs, not loop independence flags."""
    functions = {}
    for function in module["functions"]:
        roots = validate_access_contract(function)
        if roots:
            functions[function["name"]] = dict(
                contract_sha256=digest(json_bytes(function["access_contract"])),
                parameters=list(roots),
            )
    if not functions:
        return None
    qualification = _access_feature(cache if cache is not None else {})
    return dict(
        semantics_version=1,
        scope="invocation",
        feature="gnu_restrict_parameter",
        qualifier="__restrict__" if qualification["supported"] else None,
        qualification=qualification,
        functions=functions,
    )


_IPO_PROJECT = """cmake_minimum_required(VERSION 3.20)
project(scar_compiler_capability LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CheckIPOSupported)
check_ipo_supported(RESULT scar_ipo OUTPUT scar_ipo_error LANGUAGES CXX)
file(WRITE "${CMAKE_BINARY_DIR}/identity.txt"
    "${scar_ipo}\\n${CMAKE_CXX_COMPILER_ID}\\n${CMAKE_CXX_COMPILER_VERSION}\\n${CMAKE_CXX_COMPILER}\\n")
file(WRITE "${CMAKE_BINARY_DIR}/diagnostic.txt" "${scar_ipo_error}")
"""


def _ipo_feature(cache):
    """Use CMake's maintained compile/link probe; never run its executable."""
    cmake = shutil.which("cmake")
    if not cmake:
        return dict(supported=False, reason="CMake is unavailable")
    environment = {
        name: os.environ.get(name) for name in ("CXX", "CXXFLAGS", "LDFLAGS", "PATH")
    }
    cmake_identity = _file(cmake)
    key = ("ipo", digest(json_bytes([cmake_identity, environment, _IPO_PROJECT])))
    if key not in cache:
        result = dict(
            supported=False,
            cmake=cmake_identity,
            evidence="cmake_compile_and_link_only",
            probe_sha256=digest(_IPO_PROJECT.encode()),
            application_executed=False,
            environment_sha256=digest(json_bytes(environment)),
            scope="qualification host; delivered project validates the selected compiler family",
        )
        with TemporaryDirectory(prefix="scar-ipo-capability-") as temp:
            source = Path(temp)
            (source / "CMakeLists.txt").write_text(_IPO_PROJECT)
            build = source / "build"
            try:
                probe = subprocess.run(
                    [cmake, "-S", str(source), "-B", str(build)],
                    capture_output=True,
                    text=True,
                    timeout=45,
                )
                identity = build / "identity.txt"
                if probe.returncode or not identity.is_file():
                    result.update(
                        reason="target compiler capability configuration failed",
                        diagnostic=(probe.stdout + probe.stderr)[-4000:],
                    )
                else:
                    available, family, version, compiler = (
                        identity.read_text().splitlines()
                    )
                    result.update(
                        supported=available.upper() in {"YES", "TRUE", "ON", "1"},
                        compiler_id=family,
                        compiler_version=version,
                        compiler=_file(compiler),
                    )
                    if not result["supported"]:
                        result.update(
                            reason="target compiler rejected interprocedural optimization",
                            diagnostic=(build / "diagnostic.txt").read_text()[-4000:],
                        )
            except (OSError, subprocess.TimeoutExpired, ValueError) as error:
                result.update(supported=False, reason=str(error))
        cache[key] = result
    return cache[key]


def compilation(module, *, cache=None):
    """Keep a complete image visible across target translation-unit boundaries.

    Compiler optimization is delegated to the qualified toolchain. This policy
    does not relax floating-point behavior or claim a measured runtime saving.
    """
    functions = {function["name"] for function in module["functions"]}
    calls = sum(
        node["op"] == "call" and node.get("callee") in functions
        for node in walk_operations([fn["body"] for fn in module["functions"]])
    )
    budget = cpu_availability().worker_budget
    requested = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL")
    if requested and (not requested.isdecimal() or int(requested) < 1):
        raise ValueError("CMAKE_BUILD_PARALLEL_LEVEL must be a positive integer")
    budget = min(budget, int(requested) if requested else 8)
    decision = dict(
        semantics_version=1,
        scope="link_image",
        mode="none",
        workers=1,
        evidence=dict(
            defined_functions=len(functions),
            direct_internal_calls=calls,
            worker_budget=budget,
        ),
        qualification=dict(
            supported=False, reason="no cross-function compilation scope required"
        ),
        reason="no cross-function compilation scope required",
    )
    if len(functions) < 2 or not calls:
        return decision
    arguments = " ".join(
        os.environ.get(name, "") for name in ("CXX", "CXXFLAGS", "LDFLAGS")
    )
    if "-fno-lto" in shlex.split(arguments):
        decision["reason"] = (
            "interprocedural optimization explicitly disabled by compiler arguments"
        )
        decision["qualification"]["reason"] = decision["reason"]
        return decision
    qualified = _ipo_feature(cache if cache is not None else {})
    decision["qualification"] = qualified
    if not qualified["supported"]:
        decision["reason"] = qualified["reason"]
    elif qualified["compiler_id"] != "GNU":
        decision["reason"] = (
            "no bounded interprocedural compilation policy registered for this compiler family"
        )
    else:
        decision.update(
            mode="gnu_lto",
            workers=budget,
            reason="retain whole-image optimization scope across generated source partitions",
        )
    return decision

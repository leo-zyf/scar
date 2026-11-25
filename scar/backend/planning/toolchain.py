"""B1 target-compiler capabilities and whole-image compilation decisions."""

import os
from pathlib import Path
import re
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


def _source_math_declarations(contract, headers):
    """Index source interface declarations by spelling, retaining competing predicates."""
    capture = contract.get("semantic_capture", {})
    capture = capture if isinstance(capture, dict) else {}
    inputs = {
        row["path"]: row
        for row in (capture.get("inputs") or ())
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    declarations = {}
    for row in contract.get("source_math_interfaces") or ():
        if not isinstance(row, dict) or not isinstance(row.get("interface"), str):
            continue
        proof = row.get("evidence", {})
        proof = proof if isinstance(proof, dict) else {}
        path = proof.get("file")
        if not isinstance(path, str):
            path = None
        selected, header = inputs.get(path, {}), headers.get(path, {})
        sha = proof.get("sha256")
        verified = (
            contract.get("semantics_version") == 1
            and type(capture.get("returncode")) is int
            and capture["returncode"] == 0
            and row.get("semantics_version") == 1
            and row.get("execution") == "not_observed"
            and row.get("declaration") == "gnu_builtin"
            and isinstance(sha, str)
            and re.fullmatch(r"[0-9a-f]{64}", sha) is not None
            and selected.get("sha256") == header.get("sha256") == sha
            and type(selected.get("size")) is int
            and selected["size"] > 0
            and selected["size"] == header.get("size")
            and selected.get("evidence")
            == proof.get("provider")
            == "gnu_semantic_preinclude"
            and type(proof.get("line")) is int
            and proof["line"] > 0
        )
        declarations.setdefault(row["interface"], []).append(
            dict(declaration=row, evidence_verified=verified)
        )
    return declarations


_MATH_OPERANDS = frozenset(
    (
        "args base body condition count else extent finally handlers index indices init "
        "initializers items left range receiver right shape step target then value values "
        "resource output delimiter sep end flush stream reference fused exponent start "
        "stop inputs argv"
    ).split()
)


def _math_operations(function, names, external_math):
    # Only operation operands carry executable syntax. Source contracts, layout
    # proofs, type descriptions and other provenance are not calls.
    pending = [
        (("initializers",), function.get("initializers", [])),
        (("body",), function["body"]),
    ]
    while pending:
        path, node = pending.pop()
        if isinstance(node, list):
            pending.extend(((*path, i), node[i]) for i in reversed(range(len(node))))
        elif isinstance(node, dict):
            op, symbol = node.get("op"), node.get("name")
            if op == "call":
                symbol = node.get("callee")
            if isinstance(symbol, str) and (
                op == "intrinsic"
                and symbol in names
                or op == "library_call"
                and node.get("provider") == "cxx_standard"
                and "receiver" not in node
                and symbol in names
                or op == "call"
                and symbol in external_math
                and symbol in names
            ):
                yield list(path), node, symbol
            pending.extend(
                ((*path, key), child)
                for key, child in reversed(node.items())
                if key in _MATH_OPERANDS and isinstance(child, (list, dict))
            )


def math_interfaces(module, source_members, scheduled_members, compilation):
    """Decide whether each candidate has an end-to-end vector math binding."""
    inventory = module.get("resource_model", {})
    headers, platform = inventory.get("software_headers", {}), inventory.get(
        "platform", {}
    )
    functions = module["functions"]
    defined = {f["name"] for f in functions}
    external_math = {
        decl["name"]
        for decl in module.get("externs", ())
        if decl["name"] not in defined
        and decl.get("provider") == "libc"
        and decl.get("header") == "math.h"
    }
    profiles, catalogs, decisions = {}, {}, {}
    for fn, scheduled in zip(functions, scheduled_members, strict=True):
        original = source_members.get(fn["name"], {})
        contract = fn.get("source_numeric_contract")
        key = digest(json_bytes(contract)) if contract is not None else None
        contract = contract if isinstance(contract, dict) else {}
        if key is not None and key not in catalogs:
            catalogs[key] = _source_math_declarations(contract, headers)
        catalog = catalogs.get(key, {})
        sites = []
        for path, node, symbol in _math_operations(
            fn, {"pow", *catalog}, external_math
        ):
            arguments = [
                arg.get("type") if isinstance(arg, dict) else None
                for arg in node.get("args", [])
            ]
            signature = node.get("source_intrinsic_contract", {})
            signature = signature if isinstance(signature, dict) else {}
            same_signature = (
                signature.get("semantics_version") == 1
                and signature.get("normalized_name") == symbol
                and signature.get("result_type") == node.get("type") is not None
                and signature.get("argument_types") == arguments
                and None not in arguments
            )
            candidates = catalog.get(symbol, [])
            missing = [
                "source_math_callsite_binding",
                "region_vectorization_proof",
                "caller_errno_observation_proof",
                "caller_fenv_observation_proof",
                "rounding_and_trap_contract",
                "target_vector_library_abi_qualification",
            ]
            if key is None:
                missing.append("source_numeric_contract")
            if not contract.get("compiler"):
                missing.append("source_compiler_identity")
            if not same_signature:
                missing.append("source_intrinsic_signature")
            if not candidates:
                missing.append("source_math_declaration")
            elif not any(c["evidence_verified"] for c in candidates):
                missing.append("verified_source_math_header")
            if any(
                c["declaration"].get("target_condition") is not None for c in candidates
            ):
                missing.append("source_declaration_target_condition")
            if not node.get("site"):
                missing.append("source_site")
            if not original:
                missing.append("source_function_member")
            sites.append(
                dict(
                    site=node.get("site"),
                    operation_path=path,
                    operation=node["op"],
                    interface=symbol,
                    argument_types=arguments,
                    result_type=node.get("type"),
                    source_intrinsic_signature_verified=same_signature,
                    selected="preserve_operation",
                    vector_eligibility="not_qualified",
                    obligations=missing,
                )
            )
        if not sites:
            continue
        if key is not None:
            profile = profiles.setdefault(
                key,
                {
                    k: contract.get(k)
                    for k in (
                        "compiler",
                        "source",
                        "source_flags",
                        "policy",
                        "caller_observations",
                        "runtime_fp_environment",
                    )
                },
            )
            declarations = profile.setdefault("declarations", {})
            for site in sites:
                symbol = site["interface"]
                if symbol in catalog:
                    declarations[symbol] = catalog[symbol]
        decisions[fn["name"]] = dict(
            exec_function_sha256=original.get("sha256"),
            scheduled_function_sha256=scheduled["sha256"],
            source_contract_sha256=key,
            sites=sites,
        )
    if not decisions:
        return None
    qualification = compilation["qualification"]
    return dict(
        semantics_version=1,
        scope="scheduled_operations",
        coverage="selected executable operands in function bodies/initializers; real-power and compiler-declared interface spellings",
        source_contracts=profiles,
        functions=decisions,
        target=dict(
            compiler=qualification.get("compiler"),
            compiler_id=qualification.get("compiler_id"),
            compiler_version=qualification.get("compiler_version"),
            compilation_sha256=digest(json_bytes(compilation)),
            captured_platform=dict(
                architecture=platform.get("architecture"),
                isa_profiles=platform.get("isa_profiles"),
            ),
            vector_library=None,
            vector_abi=None,
            qualified_isa=None,
            qualification="no_registered_vector_math_binding",
        ),
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

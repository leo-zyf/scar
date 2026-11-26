"""B1 target-compiler capabilities and whole-image compilation decisions."""

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from tempfile import TemporaryDirectory

from scar.contract import (
    digest,
    external_math_declaration,
    floating_environment_declaration,
    json_bytes,
    validate_access_contract,
    validate_numeric_contract,
)
from scar.host import cpu_availability
from .runtime import ImageLinkage


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


_NUMERIC_OPTIONS = {
    "may_omit_math_errno": "no-math-errno",
    "may_assume_default_rounding": "no-rounding-math",
}
_STRICT_FP = ["-fno-fast-math", "-frounding-math", "-ffp-contract=off"]


def _numeric_probe_observations(output):
    """Read the GNU optimizer's independent records for the fixed probe definitions."""
    parts = re.split(r"^;; Function (\w+) \(", output, flags=re.MULTILINE)
    bodies = dict(zip(parts[1::2], parts[2::2], strict=True))
    expected = {
        "scar_strict_before": [],
        "scar_local_errno": ["no-math-errno"],
        "scar_local_rounding": ["no-rounding-math"],
        "scar_local_both": ["no-math-errno", "no-rounding-math"],
        "scar_strict_after": [],
        "scar_strict_rounding": [],
    }
    for name, options in expected.items():
        body = bodies.get(name, "")
        observed = re.findall(r'"(no-math-errno|no-rounding-math)"', body)
        if not body or observed != options:
            raise ValueError("GNU optimizer did not confirm isolated function options")

    def errno_path(name):
        return re.search(r"\b(?:__builtin_)?sqrt\s*\(", bodies[name]) is not None

    if not (
        errno_path("scar_strict_before")
        and errno_path("scar_strict_after")
        and re.search(r"= 1(?:\.0+)?e\+0 \+ ", bodies["scar_strict_rounding"])
        and re.search(r"return 1(?:\.0+)?e\+0;", bodies["scar_local_rounding"])
    ):
        raise ValueError("GNU probe did not distinguish local and strict FP behavior")
    return dict(
        options_per_probe_function=expected,
        strict_neighbors_retain_sqrt_errno_path=True,
        strict_neighbor_retains_dynamic_rounding=True,
        local_rounding_constant_folded=True,
        local_sqrt_retains_errno_path=errno_path("scar_local_errno"),
    )


def _numeric_feature(compilation, cache):
    identity = compilation["qualification"]
    if not identity.get("compiler_id"):
        identity = _ipo_feature(cache)
    if identity.get("compiler_id") != "GNU":
        return dict(supported=False, reason="GNU C++ compiler has not been identified")
    source = Path(__file__).with_name("numeric_probe.cpp")
    try:
        args = shlex.split(os.environ.get("CXX") or "c++")
        compiler = shutil.which(args[0]) if args else None
        if not compiler or _file(compiler) != identity.get("compiler"):
            return dict(
                supported=False, reason="target compiler invocation is unverified"
            )
        target_flags = shlex.split(os.environ.get("CXXFLAGS", ""))
        probe_sha = digest(source.read_bytes())
    except (OSError, ValueError) as exc:
        return dict(supported=False, reason=str(exc))
    flags = [
        *args[1:],
        *target_flags,
        "-O3",
        "-DNDEBUG",
        "-std=c++20",
        "-Werror",
        *_STRICT_FP,
        "-fno-lto",
    ]
    result = dict(
        supported=False,
        compiler_id="GNU",
        compiler=identity["compiler"],
        compiler_version=identity["compiler_version"],
        probe_sha256=probe_sha,
        arguments=flags,
        baseline_fp=_STRICT_FP,
        evidence="gnu_optimized_tree_compile_only",
        application_executed=False,
    )
    key = ("function_numeric", digest(json_bytes(result)))
    if key not in cache:
        with TemporaryDirectory(prefix="scar-numeric-options-") as scratch:
            try:
                process = subprocess.run(
                    [
                        compiler,
                        *flags,
                        "-fdump-tree-optimized=probe.tree",
                        "-c",
                        str(source),
                        "-o",
                        "probe.o",
                    ],
                    cwd=scratch,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if process.returncode:
                    raise ValueError(
                        "GNU numeric option probe failed: " + process.stderr[-1000:]
                    )
                tree = (Path(scratch) / "probe.tree").read_bytes()
                observations = _numeric_probe_observations(tree.decode())
                result.update(
                    supported=True, observations=observations, tree_sha256=digest(tree)
                )
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                result["reason"] = str(exc)
        cache[key] = result
    return cache[key]


def numeric_permissions(
    module, source_members, scheduled_members, compilation, *, cache=None
):
    """Transfer source compilation permissions at each emitted definition boundary."""
    functions = {}
    for fn, scheduled in zip(module["functions"], scheduled_members, strict=True):
        contract = validate_numeric_contract(fn)
        if contract is None:
            continue
        row = dict(
            exec_function_sha256=source_members[fn["name"]]["sha256"],
            scheduled_function_sha256=scheduled["sha256"],
            contract_sha256=digest(json_bytes(fn["numeric_contract"])),
            options=[],
        )
        if not compilation.get("floating_environment") and not any(
            fn.get("method", {}).get(k) for k in ("defaulted", "deleted", "implicit")
        ):
            row["options"] = [
                option
                for key, option in _NUMERIC_OPTIONS.items()
                if fn["numeric_contract"]["permissions"][key]
            ]
        functions[fn["name"]] = row
    if not functions:
        return None
    qualification = (
        _numeric_feature(compilation, {} if cache is None else cache)
        if any(row["options"] for row in functions.values())
        else dict(
            supported=False,
            reason="no emitted function grants local floating-point options",
        )
    )
    if not qualification["supported"]:
        for row in functions.values():
            row["options"] = []
    return dict(
        semantics_version=1,
        scope="function_definitions",
        feature="gnu_function_numeric_options",
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
            dict(
                declaration=row,
                evidence_verified=verified,
                target_applicability=_source_math_target(contract, row),
            )
        )
    return declarations


def _source_math_target(contract, declaration):
    """Consume the source compiler's ABI predicate and its captured evidence."""
    condition = declaration.get("target_condition")
    if condition is None:
        return True
    mode = contract.get("target_mode", {})
    proof = mode.get("evidence", {})
    expected = dict(
        source_contract_field="target_mode",
        rule="gnu_x86_multilib_abi_name",
        compiler_frontend_sha256=contract.get("compiler", {})
        .get("frontend", {})
        .get("sha256"),
        option_query_stdout_sha256=contract.get("option_query", {}).get(
            "stdout_sha256"
        ),
    )
    if (
        condition.get("evaluation") != "derived"
        or type(condition.get("value")) is not bool
        or mode.get("semantics_version") != 1
        or not mode.get("multilib_abi")
        or proof.get("kind") != "derived"
        or any(not expected[key] for key in expected)
        or any(
            proof.get(key) != value
            for key, value in expected.items()
            if key != "source_contract_field"
        )
        or condition.get("evidence") != expected
        or condition["value"] != (condition.get("selector") == mode["multilib_abi"])
    ):
        return None
    return condition["value"]


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
        decl["name"]: interface
        for decl in module.get("externs", ())
        if decl["name"] not in defined
        and (interface := external_math_declaration(decl)) is not None
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
                and signature.get(
                    "normalized_argument_types", signature.get("argument_types")
                )
                == arguments
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
                missing.append(
                    "source_vector_math_declaration"
                    if node["op"] == "call"
                    else "source_math_declaration"
                )
            elif not any(c["evidence_verified"] for c in candidates):
                missing.append("verified_source_math_header")
            applicable = [
                c
                for c in candidates
                if c["evidence_verified"] and c["target_applicability"] is True
            ]
            if candidates and not applicable:
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
                    **(
                        {"external_declaration": external_math[symbol]}
                        if node["op"] == "call"
                        else {}
                    ),
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
                        "target_mode",
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
        coverage="selected executable operands in function bodies/initializers; resolved external math calls, real-power and compiler-declared interfaces",
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


def compilation(module, *, cache=None, linkage=None):
    """Keep a complete image visible across target translation-unit boundaries.

    Compiler optimization is delegated to the qualified toolchain. This policy
    does not relax floating-point behavior or claim a measured runtime saving.
    """
    functions = {function["name"] for function in module["functions"]}
    calls = sum(
        field == "body" and node["op"] == "call" and node.get("callee") in functions
        for _, field, node in (linkage or ImageLinkage(module)).sites
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
    interfaces = sorted(
        d["name"]
        for d in module.get("externs", ())
        if floating_environment_declaration(d)
    )
    if interfaces:
        # GNU's general rounding option alone does not keep FP operations
        # on the correct side of an explicit change in rounding mode.
        reason = (
            "observable floating environment requires preserving operation placement"
        )
        decision.update(floating_environment="observable", reason=reason)
        decision["evidence"]["floating_environment_interfaces"] = interfaces
        decision["qualification"]["reason"] = reason
        return decision
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

"""B1 typed arbitrary-integer operations and their optional GMP CPU SDK."""

import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sysconfig
from tempfile import TemporaryDirectory

from scar.contract import digest, json_bytes
from .runtime import QualificationFailure, qualification_check

PROFILE = "signed-arbitrary-precision-v1"
_COMPARISONS = {"eq", "ne", "lt", "le", "gt", "ge"}
_SIGNATURES = {
    "from_f64": (("f64",), "integer"),
    "from_i64": (("i64",), "integer"),
    "decimal_literal": (("text",), "integer"),
    **{
        name: (("integer", "integer"), "integer")
        for name in ("add", "sub", "mul", "floordiv", "mod")
    },
    **{name: (("integer", "integer"), "bool") for name in _COMPARISONS},
    "negative": (("integer",), "integer"),
    "absolute": (("integer",), "integer"),
    "to_binary64": (("integer",), "f64"),
    "decimal": (("integer",), "text"),
    "shape_extent": (("integer",), "i64"),
    "sequence_index": (("integer",), "i64"),
    "array_index_width": (("integer",), "i64"),
}
_LICENSES = {"COPYRIGHT", "COPYING.LGPLv3", "COPYING.GPLv2", "COPYING.GPLv3"}


def _use(node):
    operation, args = node.get("operation"), node.get("args")
    signature = _SIGNATURES.get(operation) if isinstance(operation, str) else None
    if (
        node.get("op") != "integer_op"
        or node.get("operator_family") != "integer"
        or type(node.get("semantics_version")) is not int
        or node["semantics_version"] != 1
        or signature is None
        or not isinstance(args, list)
        or tuple(arg.get("type") if isinstance(arg, dict) else None for arg in args)
        != signature[0]
        or node.get("type") != signature[1]
        or not isinstance(node.get("effects"), list)
        or set(node.get("effects", ()))
        != ({"VAL", "MEM"} if signature[1] in {"integer", "text"} else {"VAL"})
        or node.get("may_raise") is not (operation not in _COMPARISONS)
        or node.get("reads") != []
        or node.get("writes") != []
    ):
        raise QualificationFailure(
            "semantic_contract",
            "integer.signature",
            "integer operation requires its exact typed arguments, immutable effects and exception contract",
            operation=operation,
            site=node.get("site"),
            expected=signature,
        )
    exception_keys = {
        "from_f64": {"value_error", "overflow_error"},
        "to_binary64": {"overflow_error"},
        "floordiv": {"zero_division_error"},
        "mod": {"zero_division_error"},
        "shape_extent": {"value_error"},
        "sequence_index": {"index_error"},
        "array_index_width": {"index_error", "overflow_error"},
    }
    messages = node.get("exception_messages", {})
    if (
        not isinstance(messages, dict)
        or set(messages) != exception_keys.get(operation, set())
        or any(
            not isinstance(message, str) or not message for message in messages.values()
        )
    ):
        raise QualificationFailure(
            "semantic_contract",
            "integer.exception_contract",
            "integer operation requires messages for each possible source exception",
            operation=operation,
            site=node.get("site"),
        )
    if operation == "decimal_literal" and (
        args[0].get("op") != "literal"
        or not isinstance(args[0].get("value"), str)
        or re.fullmatch(r"-?(?:0|[1-9][0-9]*)", args[0]["value"], re.ASCII) is None
    ):
        raise QualificationFailure(
            "semantic_contract",
            "integer.decimal_literal",
            "integer literals require a canonical ASCII decimal string",
            site=node.get("site"),
        )
    if operation == "decimal" and (
        type(node.get("maximum_digits")) is not int
        or not 1 <= node["maximum_digits"] <= 640
    ):
        raise QualificationFailure(
            "semantic_contract",
            "integer.decimal_digit_limit",
            "integer decimal output requires a proven 1..640 digit bound",
            site=node.get("site"),
            maximum_digits=node.get("maximum_digits"),
        )
    return operation


def _files():
    override = os.environ.get("SCAR_INTEGER_SDK")
    local = Path.home() / ".local/share/scar/sdk/gmp-6.3.0"
    if override or local.is_dir():
        prefix = (
            Path(override).expanduser().resolve(strict=True)
            if override
            else local.resolve(strict=True)
        )
        return {
            name: prefix / name
            for name in ("include/gmp.h", "include/gmpxx.h", "lib/libgmp.a", *_LICENSES)
        }
    multiarch = sysconfig.get_config_var("MULTIARCH")
    includes = [
        Path("/usr/include") / str(multiarch) / "gmp.h",
        Path("/usr/include/gmp.h"),
    ]
    libraries = [
        Path("/usr/lib") / str(multiarch) / "libgmp.a",
        Path("/usr/lib/libgmp.a"),
        Path("/usr/lib64/libgmp.a"),
    ]
    include = next((p for p in includes if p.is_file()), None)
    archive = next((p for p in libraries if p.is_file()), None)
    if include is None or archive is None:
        raise ValueError(
            "GMP development SDK is unavailable; install libgmp-dev or set SCAR_INTEGER_SDK"
        )
    return {
        "include/gmp.h": include,
        "include/gmpxx.h": Path("/usr/include/gmpxx.h"),
        "lib/libgmp.a": archive,
        "COPYRIGHT": Path("/usr/share/doc/libgmp-dev/copyright"),
        "COPYING.LGPLv3": Path("/usr/share/common-licenses/LGPL-3"),
        "COPYING.GPLv2": Path("/usr/share/common-licenses/GPL-2"),
        "COPYING.GPLv3": Path("/usr/share/common-licenses/GPL-3"),
    }


def _sdk(cache):
    files = {}
    for name, path in _files().items():
        path = path.resolve(strict=True)
        data = path.read_bytes()
        files[name] = dict(path=str(path), sha256=digest(data), size=len(data))
    text = Path(files["include/gmp.h"]["path"]).read_text()
    version = []
    for macro in (
        "__GNU_MP_VERSION",
        "__GNU_MP_VERSION_MINOR",
        "__GNU_MP_VERSION_PATCHLEVEL",
    ):
        match = re.search(
            r"^#define\s+" + macro + r"\s+([0-9]+)\s*$", text, re.MULTILINE
        )
        if match is None:
            raise ValueError("GMP public header lacks its release identity")
        version.append(int(match[1]))
    if version[0] != 6:
        raise ValueError("the registered arbitrary-integer provider requires GMP 6")
    version = ".".join(map(str, version))
    arguments = shlex.split(os.environ.get("CXX") or "c++")
    compiler = shutil.which(arguments[0]) if arguments else None
    if compiler is None:
        raise ValueError("GMP qualification requires a C++20 compiler")
    compiler = str(Path(compiler).resolve(strict=True))
    source = Path(__file__).with_name("integer_probe.cpp")
    compiler_bytes, source_bytes = Path(compiler).read_bytes(), source.read_bytes()
    identity = dict(
        provider="gmp-cpu",
        semantics_version=1,
        device="cpu",
        profile=PROFILE,
        version=version,
        files={
            name: {k: row[k] for k in ("sha256", "size")} for name, row in files.items()
        },
    )
    sdk_sha = digest(json_bytes(identity))
    key = digest(
        json_bytes(
            dict(
                sdk=sdk_sha,
                compiler=digest(compiler_bytes),
                arguments=arguments[1:],
                probe=digest(source_bytes),
            )
        )
    )
    if key not in cache:
        with TemporaryDirectory(prefix="scar-gmp-qualification-") as temporary:
            executable = Path(temporary) / "probe"
            command = [
                compiler,
                *arguments[1:],
                "-std=c++20",
                "-Werror",
                "-pedantic-errors",
                "-I" + str(Path(files["include/gmp.h"]["path"]).parent),
                str(source),
                files["lib/libgmp.a"]["path"],
                "-o",
                str(executable),
            ]
            built = subprocess.run(command, capture_output=True, text=True, timeout=60)
            if built.returncode:
                raise ValueError(
                    "GMP static SDK failed C++20 link qualification: "
                    + built.stderr[-3000:]
                )
            observed = subprocess.run(
                [executable], capture_output=True, text=True, timeout=10
            )
            fields = observed.stdout.split()
            if (
                observed.returncode
                or len(fields) != 3
                or fields[0] != version
                or not all(v.isdecimal() for v in fields[1:])
            ):
                raise ValueError("GMP static SDK failed version/ABI/API self-check")
            cache[key] = dict(
                compiler=dict(
                    path=compiler,
                    sha256=digest(compiler_bytes),
                    size=len(compiler_bytes),
                ),
                compiler_arguments=arguments[1:],
                cxx_standard=20,
                evidence="fixed_probe_compile_link_and_execute",
                application_executed=False,
                probe_sha256=digest(source_bytes),
                abi=dict(
                    architecture=platform.machine(),
                    limb_bits=int(fields[1]),
                    pointer_bits=int(fields[2]),
                    link="static",
                    api="gmp-mpz-c",
                ),
                semantic_scope="SDK identity, ABI and declared GMP primitives; not application validation or integer-to-binary64 rounding proof",
            )
    qualification = cache[key]
    return dict(
        identity,
        files=files,
        sdk_sha256=sdk_sha,
        abi=qualification["abi"],
        qualification=qualification,
    )


def qualify(requirement, nodes, *, cache=None):
    if (
        not isinstance(requirement, dict)
        or requirement.keys() - {"semantics_version", "operations"}
        or type(requirement.get("semantics_version")) is not int
        or requirement["semantics_version"] != 1
        or not isinstance(requirement.get("operations"), list)
    ):
        raise QualificationFailure(
            "semantic_contract",
            "integer.requirement",
            "integer requires versioned operations or a typed integer interface",
        )
    declared = requirement["operations"]
    if any(not isinstance(name, str) or name not in _SIGNATURES for name in declared):
        raise QualificationFailure(
            "implementation_profile",
            "integer.registered_operations",
            "unregistered arbitrary-integer operation",
            operations=declared,
        )
    operations = {_use(node) for node in nodes}
    if not operations <= set(declared):
        raise QualificationFailure(
            "semantic_contract",
            "integer.undeclared_operations",
            "integer occurrence is missing its declared operation",
            missing=sorted(operations - set(declared)),
        )
    with qualification_check("target_platform", "integer.gmp_sdk", provider="gmp-cpu"):
        binding = _sdk(cache if cache is not None else {})
    return dict(
        binding,
        operations=sorted(operations),
        selection="fixed exact integer provider; no performance optimum claimed",
    )

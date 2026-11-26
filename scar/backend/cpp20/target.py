"""B2a: lower a sealed scheduled program into a typed C++20 target package."""

import json
from pathlib import Path
import re

from scar.contract import (
    PHYSICAL,
    digest,
    document,
    external_math_declaration,
    file_receipt,
    json_bytes,
    member,
    open_product,
    publish,
    read_image_product,
    read_member,
    validate_images,
    validate_numeric_contract,
    write_json,
)
from .receipts import (
    SDK_HEADERS,
    gmp_files,
    is_digest,
    pocketfft_files,
    validate_mpi_sdk,
    xtensor_headers,
)

_XTENSOR_NOTICE = b"""BSD 3-Clause License

xtensor
Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht
Copyright (c) 2016, QuantStack

xtl
Copyright (c) 2017, Sylvain Corlay, Johan Mabille and Wolf Vollprecht
Copyright (c) 2017, QuantStack
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


def _publish_blob(destination, group, data, sha, suffix=""):
    """Seal the bytes already fingerprinted by this producer."""
    directory = destination / group
    if not directory.resolve().is_relative_to(destination):
        raise ValueError(f"target member directory escapes its package: {group}")
    relative = f"{group}/{sha}{suffix}"
    publish(destination / relative, data)
    return dict(path=relative, sha256=sha, size=len(data))


def _seal_xtensor(destination, binding):
    # Consume exactly the installation selected by B1. Its location is not
    # needed by B2b or by the delivered project's native build.
    if binding.get("provider") != "xtensor-cpu" or not isinstance(
        binding.get("prefix"), str
    ):
        raise ValueError("missing scheduled native operator SDK identity")
    prefix = Path(binding["prefix"]).resolve(strict=True)
    raw = (prefix / "scar-sdk.json").read_bytes()
    receipt_sha = digest(raw)
    if receipt_sha != binding.get("receipt_sha256"):
        raise ValueError("installed operator SDK differs from its scheduled receipt")
    receipt = json.loads(raw)
    if receipt.get("packages") != binding.get("packages") or len(
        receipt.get("files", {})
    ) != binding.get("validated_files"):
        raise ValueError(
            "scheduled operator SDK package contract differs from its receipt"
        )

    headers = {}
    for name, sha in xtensor_headers(receipt).items():
        path = (prefix / name).resolve(strict=True)
        if not path.is_relative_to(prefix):
            raise ValueError(f"operator SDK header escapes its selected prefix: {name}")
        data = path.read_bytes()
        if digest(data) != sha:
            raise ValueError(f"operator SDK header changed since scheduling: {name}")
        headers[name] = _publish_blob(destination, "operator-sdk", data, sha)
    return dict(
        provider="xtensor-cpu",
        receipt=_publish_blob(destination, "operator-sdk", raw, receipt_sha),
        headers=headers,
        notice=_publish_blob(
            destination, "operator-sdk", _XTENSOR_NOTICE, digest(_XTENSOR_NOTICE)
        ),
    )


def _seal_sdk_file(destination, source, expected):
    data = source.read_bytes()
    if len(data) != expected["size"] or digest(data) != expected["sha256"]:
        raise ValueError(f"SDK member changed since physical qualification: {source}")
    return _publish_blob(destination, "operator-sdk", data, expected["sha256"])


def _seal_pocketfft(destination, binding):
    files = pocketfft_files(binding)
    if not isinstance(binding.get("prefix"), str):
        raise ValueError("PocketFFT schedule lacks its selected installation")
    prefix = Path(binding["prefix"]).resolve(strict=True)
    sealed = {}
    for name, expected in files.items():
        path = (prefix / name).resolve(strict=True)
        if not path.is_relative_to(prefix):
            raise ValueError("PocketFFT source escapes its selected installation")
        sealed[name] = _seal_sdk_file(destination, path, expected)
    return {
        **{
            key: binding[key]
            for key in (
                "provider",
                "semantics_version",
                "device",
                "revision",
                "profile",
            )
        },
        "files": sealed,
    }


def _seal_gmp(destination, binding):
    files = gmp_files(binding)
    sealed = {}
    for name, expected in files.items():
        path = Path(expected.get("path", ""))
        if not path.is_absolute():
            raise ValueError("GMP schedule lacks the selected absolute SDK member")
        sealed[name] = _seal_sdk_file(destination, path, expected)
    return {
        **{
            key: binding[key]
            for key in (
                "provider",
                "semantics_version",
                "device",
                "profile",
                "version",
                "sdk_sha256",
                "abi",
            )
        },
        "files": sealed,
    }


def _compilation_options(module, physical):
    """Project a sealed link-image decision onto the C++ target build contract."""
    from .model import build_options

    seal = "compilation_binding_sha256"
    if seal not in module and "compilation" not in physical:
        return build_options()
    decision = physical.get("compilation")
    if not isinstance(decision, dict) or module.get(seal) != digest(
        json_bytes(decision)
    ):
        raise ValueError("missing or mismatched physical compilation binding")
    qualification = decision.get("qualification")
    evidence = decision.get("evidence")
    if (
        type(decision.get("semantics_version")) is not int
        or decision["semantics_version"] != 1
        or decision.get("scope") != "link_image"
        or not isinstance(decision.get("reason"), str)
        or not decision["reason"].strip()
        or not isinstance(qualification, dict)
        or type(qualification.get("supported")) is not bool
        or not isinstance(evidence, dict)
    ):
        raise ValueError("invalid physical compilation binding")
    options = build_options(decision.get("mode"), decision.get("workers"))
    if decision["mode"] == "gnu_lto":
        if not qualification["supported"] or qualification.get("compiler_id") != "GNU":
            raise ValueError("GNU LTO requires a qualified GNU toolchain")
        budget = evidence.get("worker_budget")
        if type(budget) is not int or decision["workers"] > budget:
            raise ValueError("GNU LTO workers exceed the sealed compilation budget")
        for tool in ("compiler", "cmake"):
            record = qualification.get(tool)
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("path"), str)
                or not record["path"]
                or type(record.get("size")) is not int
                or record["size"] <= 0
                or not isinstance(record.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
            ):
                raise ValueError(f"GNU LTO lacks a sealed {tool} toolchain receipt")
    return options


def _math_operation(function, path):
    if (
        not isinstance(path, list)
        or not path
        or path[0] not in ("body", "initializers")
    ):
        raise ValueError("invalid scheduled math operation path")
    node = function
    for key in path:
        if type(key) is int and isinstance(node, list) and 0 <= key < len(node):
            node = node[key]
        elif isinstance(key, str) and isinstance(node, dict) and key in node:
            node = node[key]
        else:
            raise ValueError("invalid scheduled math operation path")
    if not isinstance(node, dict):
        raise ValueError("expected an operation at the scheduled math path")
    return node


def _numeric_decisions(module, receipts, physical):
    from .model import NUMERIC_FUNCTION_OPTIONS

    functions = {}
    for fn, receipt in zip(module["functions"], receipts, strict=True):
        contract = validate_numeric_contract(fn)
        if contract is not None:
            functions[fn["name"]] = (fn, receipt["sha256"], contract)
    key = "numeric_permissions_binding_sha256"
    decision = physical.get("numeric_permissions")
    if not functions and decision is None and key not in module:
        return None, {}
    if not isinstance(decision, dict) or module.get(key) != digest(
        json_bytes(decision)
    ):
        raise ValueError("missing or mismatched function numeric permission binding")
    qualification = decision.get("qualification")
    rows = decision.get("functions")
    if (
        type(decision.get("semantics_version")) is not int
        or decision["semantics_version"] != 1
        or decision.get("scope") != "function_definitions"
        or decision.get("feature") != "gnu_function_numeric_options"
        or not isinstance(rows, dict)
        or not rows
        or rows.keys() != functions.keys()
        or not isinstance(qualification, dict)
        or type(qualification.get("supported")) is not bool
    ):
        raise ValueError("invalid function numeric permission binding")
    supported = qualification["supported"]
    compiler_sha = None
    if supported:
        compiler = qualification.get("compiler") or {}
        checks = qualification.get("observations") or {}
        if (
            qualification.get("compiler_id") != "GNU"
            or not isinstance(compiler, dict)
            or not compiler.get("path")
            or type(compiler.get("size")) is not int
            or compiler["size"] < 1
            or not is_digest(compiler.get("sha256"))
            or not is_digest(qualification.get("probe_sha256"))
            or not is_digest(qualification.get("tree_sha256"))
            or qualification.get("evidence") != "gnu_optimized_tree_compile_only"
            or qualification.get("application_executed") is not False
            or qualification.get("baseline_fp")
            != ["-fno-fast-math", "-frounding-math", "-ffp-contract=off"]
            or not isinstance(checks, dict)
            or not isinstance(checks.get("options_per_probe_function"), dict)
            or not all(
                checks.get(k) is True
                for k in (
                    "strict_neighbors_retain_sqrt_errno_path",
                    "strict_neighbor_retains_dynamic_rounding",
                    "local_rounding_constant_folded",
                )
            )
        ):
            raise ValueError(
                "function numeric permissions require a qualified GNU compiler"
            )
        compiler_sha = compiler["sha256"]
        image_compiler = (
            physical.get("compilation", {}).get("qualification", {}).get("compiler")
        )
        if image_compiler is not None and image_compiler != compiler:
            raise ValueError("function numeric compiler differs from image compiler")
    elif (
        not isinstance(qualification.get("reason"), str)
        or not qualification["reason"].strip()
    ):
        raise ValueError("strict numeric fallback requires a qualification reason")
    options, audit = {}, {}
    for name, (fn, member_sha, contract) in functions.items():
        row = rows[name]
        expected = [
            option
            for claim, option in NUMERIC_FUNCTION_OPTIONS.items()
            if supported
            and contract["permissions"][claim]
            and not any(
                fn.get("method", {}).get(k)
                for k in ("defaulted", "deleted", "implicit")
            )
        ]
        if (
            not isinstance(row, dict)
            or row.get("scheduled_function_sha256") != member_sha
            or not is_digest(row.get("exec_function_sha256"))
            or row.get("contract_sha256") != digest(json_bytes(contract))
            or row.get("options") != expected
            or (
                expected
                and expected not in checks["options_per_probe_function"].values()
            )
        ):
            raise ValueError(
                "numeric options differ from the scheduled function or its permissions"
            )
        audit[name] = dict(scheduled_function_sha256=member_sha, options=expected)
        if expected:
            options[name] = expected
    return (
        dict(
            semantics_version=1,
            scope="function_definitions",
            binding_sha256=module[key],
            compiler_sha256=compiler_sha if options else None,
            functions=audit,
        ),
        options,
    )


def _math_decisions(module, receipts, physical):
    seal = "math_interface_binding_sha256"
    if seal not in module and "math_interfaces" not in physical:
        return None, set()
    decision = physical.get("math_interfaces")
    if not isinstance(decision, dict) or module.get(seal) != digest(
        json_bytes(decision)
    ):
        raise ValueError("missing or mismatched physical math interface binding")
    owners = decision.get("functions")
    target = decision.get("target")
    if (
        type(decision.get("semantics_version")) is not int
        or decision["semantics_version"] != 1
        or decision.get("scope") != "scheduled_operations"
        or not isinstance(owners, dict)
        or not owners
        or not isinstance(target, dict)
        or target.get("compilation_sha256")
        != digest(json_bytes(physical.get("compilation")))
    ):
        raise ValueError(
            "invalid scheduled math interface scope or compilation identity"
        )
    functions = {
        fn["name"]: (fn, receipt["sha256"])
        for fn, receipt in zip(module["functions"], receipts, strict=True)
    }
    if len(functions) != len(receipts):
        raise ValueError("ambiguous scheduled math function name")
    externs = {
        d["name"]: d for d in module.get("externs", ()) if d["name"] not in functions
    }
    interfaces, pending = {}, set()
    for name, row in owners.items():
        if name not in functions or not isinstance(row, dict):
            raise ValueError("unknown scheduled math function")
        fn, member_sha = functions[name]
        contract = fn.get("source_numeric_contract")
        source_sha = digest(json_bytes(contract)) if contract is not None else None
        if (
            row.get("scheduled_function_sha256") != member_sha
            or row.get("source_contract_sha256") != source_sha
        ):
            raise ValueError("scheduled math function member or contract mismatch")
        sites = row.get("sites")
        if not isinstance(sites, list) or not sites:
            raise ValueError("scheduled math function needs a nonempty list of sites")
        for site in sites:
            if not isinstance(site, dict) or (
                site.get("selected"),
                site.get("vector_eligibility"),
            ) != ("preserve_operation", "not_qualified"):
                raise ValueError("unsupported scheduled math interface decision")
            node = _math_operation(fn, site.get("operation_path"))
            op = node.get("op")
            symbol = node.get("callee") if op == "call" else node.get("name")
            external = externs.get(symbol, {}) if isinstance(symbol, str) else {}
            if "external_declaration" in site:
                declaration = external_math_declaration(external)
                if (
                    op != "call"
                    or declaration is None
                    or json_bytes(declaration)
                    != json_bytes(site["external_declaration"])
                ):
                    raise ValueError(
                        "scheduled math external declaration does not match"
                    )
            elif op == "call" and (
                external.get("provider"),
                external.get("header"),
            ) != ("libc", "math.h"):
                # Original v1 decisions only identified direct calls through math.h.
                raise ValueError(
                    "scheduled math site lacks a matching source declaration"
                )
            args = node.get("args")
            if (
                not isinstance(symbol, str)
                or not symbol
                or op not in {"intrinsic", "call", "library_call"}
                or op == "library_call"
                and (node.get("provider") != "cxx_standard" or "receiver" in node)
                or not isinstance(args, list)
                or not all(isinstance(a, dict) for a in args)
                or (op, symbol) != (site.get("operation"), site.get("interface"))
                or "site" not in site
                or node.get("site") != site["site"]
                or "result_type" not in site
                or node.get("type") != site["result_type"]
                or [a.get("type") for a in args] != site.get("argument_types")
            ):
                raise ValueError(
                    "scheduled math site does not match operation, interface or types"
                )
            if id(node) in pending:
                raise ValueError("duplicate scheduled math operation path")
            pending.add(id(node))
            interfaces[symbol] = interfaces.get(symbol, 0) + 1
    summary = dict(
        semantics_version=1,
        scope="scheduled_operations",
        binding_sha256=module[seal],
        selected="preserve_operation",
        vector_eligibility="not_qualified",
        functions=len(owners),
        sites=len(pending),
        interfaces=interfaces,
    )
    return summary, pending


def _lower_image(path, revision, scheduled, destination):
    from .lowering import Lowerer
    from .model import encode_unit

    if scheduled.get("status") != "READY" or scheduled.get("kind") == "image_bundle":
        raise ValueError("expected a ready single-image physical schedule")
    physical = read_member(path.parent, scheduled["physical"])
    if (
        physical.get("schema"),
        physical.get("schema_version"),
        physical.get("ir_sha256"),
    ) != (
        "scar.physical-plan",
        scheduled["schema_version"],
        scheduled["ir_sha256"],
    ) or physical.get(
        "status", "READY"
    ) != "READY":
        raise ValueError("physical plan does not belong to its scheduled IR package")
    options = _compilation_options(scheduled["module"], physical)
    module = dict(
        scheduled["module"],
        functions=[read_member(path.parent, f) for f in scheduled["functions"]],
    )
    math, pending_math = _math_decisions(module, scheduled["functions"], physical)
    numeric, options_by_function = _numeric_decisions(
        module, scheduled["functions"], physical
    )
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    lowerer = Lowerer(
        module, physical, pending_math=pending_math, numeric_options=options_by_function
    )
    if (
        physical.get("source_import_effects", {}).get("isolated_entry")
        or "isolated_entry_binding_sha256" in module
    ):
        from .imports import isolated_units

        units = isolated_units(path.parent, physical, module)
        lowerer.used_sdk_headers.update({"import.hpp", "project_import.hpp"})
    else:
        units = lowerer.units()
    if pending_math:
        raise ValueError("scheduled math path does not refer to a lowered operation")
    members = [
        dict(output=name, tree=member(destination, "target-units", encode_unit(unit)))
        for name, unit in units.items()
    ]
    support = Path(__file__).with_name("runtime.hpp").read_bytes()
    runtime = _publish_blob(destination, "runtime", support, digest(support), ".hpp")
    headers = []
    lowerer.used_sdk_headers.add("exceptions.hpp")
    if "mapping.hpp" in lowerer.used_sdk_headers:
        lowerer.used_sdk_headers.add("text.hpp")
    if "format.hpp" in lowerer.used_sdk_headers:
        lowerer.used_sdk_headers.add("array.hpp")
    if "array_io.hpp" in lowerer.used_sdk_headers:
        lowerer.used_sdk_headers.update({"array.hpp", "host.hpp", "text.hpp"})
    if "fourier.hpp" in lowerer.used_sdk_headers:
        lowerer.used_sdk_headers.add("array.hpp")
    if "host.hpp" in lowerer.used_sdk_headers:
        lowerer.used_sdk_headers.add("exceptions.hpp")
    for name in sorted(lowerer.used_sdk_headers):
        if name not in SDK_HEADERS:
            raise ValueError(f"unregistered target SDK header {name!r}")
        data = Path(__file__).with_name(name).read_bytes()
        blob = _publish_blob(destination, "runtime", data, digest(data), ".hpp")
        headers.append(dict(output=SDK_HEADERS[name], blob=blob))
    operator_sdks = {}
    mpi = physical.get("operator_backends", {}).get("mpi")
    if mpi is not None or "MPI" in lowerer.libraries.links:
        validate_mpi_sdk(mpi)
        lowerer.libraries.links.add("MPI")
        operator_sdks["MPI"] = member(destination, "operator-sdk", mpi)
    metadata = dict(
        build_options=options,
        openmp=lowerer.openmp,
        runtime_libraries=sorted(lowerer.libraries.links),
        coordinate_maps=lowerer.coordinate_maps,
        record_field_offset_checks=lowerer.layout_scope,
    )
    if math is not None:
        metadata["math_interface_consumption"] = math
    if numeric is not None:
        if options_by_function.keys() != lowerer.attributed_definitions.keys():
            raise ValueError(
                "missing emitted definition for a function numeric permission"
            )
        for name, sha in lowerer.attributed_definitions.items():
            numeric["functions"][name]["target_signature_sha256"] = sha
        metadata["numeric_permission_consumption"] = numeric
    if "XTensor" in lowerer.libraries.links:
        operator_sdks["XTensor"] = _seal_xtensor(
            destination, physical["operator_backends"]["ndarray"]
        )
    if "PocketFFT" in lowerer.libraries.links:
        operator_sdks["PocketFFT"] = _seal_pocketfft(
            destination, physical.get("operator_backends", {}).get("fourier")
        )
    if "GMP" in lowerer.libraries.links:
        operator_sdks["GMP"] = _seal_gmp(
            destination, physical.get("operator_backends", {}).get("integer")
        )
    if operator_sdks:
        metadata["operator_sdks"] = operator_sdks
    product = document(
        "scar.cpp20-target-ir",
        stage="B2a",
        scheduled_ir_sha256=revision,
        units=members,
        ir_sha256=physical.get("ir_sha256"),
        runtime_sha256=runtime["sha256"],
        runtime=runtime,
        sdk_headers=headers,
        **metadata,
    )
    path = destination / "target-ir.json"
    write_json(path, product)
    return path


def _check_launch_targets(folder, plans):
    workers, launches, targets, references = {}, [], {}, []
    for image_id, plan in plans.items():
        _, _, physical = read_image_product(
            folder, plan, "physical", "physical-plan.json", PHYSICAL
        )
        bindings = physical.get("operator_backends", {})
        targets[image_id] = dict(
            image_id=image_id, output=plan["output"], ir_sha256=physical["ir_sha256"]
        )
        process = bindings.get("process", {})
        references.extend(process.get("images", {}).items())
        sdk = bindings.get("mpi")
        if isinstance(sdk, dict) and physical.get("status") == "READY":
            workers[image_id] = (
                plan["output"],
                physical["ir_sha256"],
                digest(json_bytes(sdk)),
            )
        launches.extend(process.get("launches", {}).items())
    for name, reference in references:
        if name not in targets or reference != targets[name]:
            raise ValueError("process launch target is outside the qualified image set")
    for site, launch in launches:
        expected = workers.get(launch["image_id"])
        actual = tuple(
            launch.get(key)
            for key in (
                "worker_output",
                "worker_ir_sha256",
                "worker_mpi_receipt_sha256",
            )
        )
        if actual != expected:
            raise ValueError(
                f"MPI launch {site!r} differs from the sealed worker physical plan"
            )


def lower_scheduled(source, destination):
    """Lower each link image in its own symbol namespace."""
    path, revision, scheduled = open_product(
        source, "scheduled-ir.json", "scar.scheduled-ir"
    )
    if scheduled.get("kind") == "source_frame_library":
        from .imports import lower_library

        return lower_library(path, revision, scheduled, destination)
    if scheduled.get("kind") != "image_bundle":
        return _lower_image(path, revision, scheduled, destination)
    if scheduled.get("status") != "READY":
        raise ValueError("incomplete physical execution schedule")
    images, root = scheduled["images"], scheduled["root_image"]
    validate_images(images, root)
    physical = read_member(path.parent, scheduled["physical"])
    if (
        physical.get("schema") != PHYSICAL
        or physical.get("schema_version") != scheduled["schema_version"]
        or physical.get("ir_sha256") != scheduled["ir_sha256"]
        or physical.get("kind") != "image_bundle"
        or physical.get("status") != "READY"
        or physical.get("root_image") != root
        or physical.get("planning_scope") != "per_image"
    ):
        raise ValueError("physical plan does not belong to the scheduled image bundle")
    validate_images(physical.get("images"), root)
    plans = {plan["id"]: plan for plan in physical["images"]}
    if set(plans) != {image["id"] for image in images}:
        raise ValueError("physical plan image set differs from the schedule")
    _check_launch_targets(path.parent, plans)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    targets = []
    for image in images:
        child, child_hash, data = read_image_product(
            path.parent, image, "scheduled", "scheduled-ir.json", "scar.scheduled-ir"
        )
        if data.get("module", {}).get("entry") != image["entry"]:
            raise ValueError(f"physical image {image['id']} entry mismatch")
        plan = plans[image["id"]]
        if (plan["entry"], plan["output"]) != (image["entry"], image["output"]):
            raise ValueError(f"physical image {image['id']} identity mismatch")
        declared, selected = plan.get("physical", {}), data.get("physical", {})
        if any(declared.get(key) != selected.get(key) for key in ("sha256", "size")):
            raise ValueError(f"physical image {image['id']} receipt mismatch")
        compiled = _lower_image(
            child, child_hash, data, destination / "images" / image["id"]
        )
        targets.append(
            dict(
                id=image["id"],
                entry=image["entry"],
                output=image["output"],
                target=file_receipt(destination, compiled),
            )
        )
    target = destination / "target-ir.json"
    write_json(
        target,
        document(
            "scar.cpp20-target-ir",
            stage="B2a",
            kind="image_bundle",
            root_image=root,
            images=targets,
            ir_sha256=scheduled["ir_sha256"],
            scheduled_ir_sha256=revision,
        ),
    )
    return target

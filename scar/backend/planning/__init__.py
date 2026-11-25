"""B1 selects target resources and seals a relocatable physical execution plan."""

from pathlib import Path

from scar.contract import (
    IR,
    EXECUTION_ANNOTATIONS,
    MAPPING_PROFILE,
    MAPPING_TYPES,
    PHYSICAL,
    array_rank_domain,
    copy_member,
    digest,
    document,
    file_receipt,
    json_bytes,
    mapping_value_type,
    member,
    open_product,
    read_image_product,
    read_member,
    validate_images,
    write_json,
    walk_operations,
    walk_records,
)
from scar.host import array_sdk
from . import fourier, integer, mpi, runtime, toolchain
from .runtime import QualificationFailure, qualification_check

_ARRAY_OPERATIONS = {
    "array_create",
    "array_iota",
    "array_literal",
    "array_from_sequence",
    "array_copy",
    "array_unary",
    "array_view",
    "array_size",
    "array_rank",
    "array_extent",
    "array_reshape",
    "array_broadcast_copy",
    "array_load",
    "array_store",
    "array_binary",
    "array_update",
    "array_format",
}
_ARRAY_OPERATORS = {
    "array_unary": {"f64": {"neg", "pos"}, "c128": {"neg", "pos"}},
    "array_binary": {
        "f64": {"add", "sub", "mul", "div", "pow"},
        "c128": {"add", "sub", "mul"},
    },
}
_ARRAY_OPERATORS["array_update"] = _ARRAY_OPERATORS["array_binary"]

_MAPPING_OPERATIONS = {
    "mapping_make",
    "mapping_put",
    "mapping_merge",
    "mapping_get",
    "mapping_size",
    "mapping_identity",
    "mapping_items",
    "mapping_iterator",
    "mapping_next",
    "mapping_current",
    "structured_pack",
    "structured_is_mapping",
    "structured_as_mapping",
    "structured_truth",
    "structured_compare",
}


def _array_ports(node):
    """Immediate element-bearing operands/results; axis counts are shape metadata."""
    result = node.get("type")
    if result not in ("i64", "text", "unit", None):
        yield result
    for role in ("base", "left", "right", "value"):
        if isinstance(node.get(role), dict):
            if node["op"] == "array_from_sequence" and role == "value":
                continue
            yield node[role].get("type")
    if node["op"] == "array_literal":
        for item in node.get("values", ()):
            if isinstance(item, dict):
                yield item.get("type")


def _observed_operators(module):
    # Collect tagged operations and the type-only dependencies of older
    # packages in one traversal of the sealed IR.
    observed, elements, array_nodes, process_requests = {}, set(), [], []
    fourier_nodes, mapping_nodes, integer_nodes = [], [], []
    defined = {fn["name"] for fn in module.get("functions", ())}
    external = [
        decl for decl in module.get("externs", ()) if decl["name"] not in defined
    ]
    pending = [
        module.get(key, ()) for key in ("functions", "globals", "records", "entry_abi")
    ]
    pending.append(external)
    for node in walk_records(pending, exclude_fields=EXECUTION_ANNOTATIONS):
        op = node.get("op")
        if op is not None:
            family = node.get("operator_family")
            if family is not None:
                if not isinstance(family, str) or not family or not isinstance(op, str):
                    raise ValueError("invalid operator family or operation tag")
                observed.setdefault(family, set()).add(
                    node.get("operation")
                    if family == "integer" and isinstance(node.get("operation"), str)
                    else op
                )
            if op == "integer_op" or family == "integer":
                operation = node.get("operation")
                observed.setdefault("integer", set()).add(
                    operation if isinstance(operation, str) else op
                )
                integer_nodes.append(node)
            if (
                op in _ARRAY_OPERATIONS
                or family == "ndarray"
                or (
                    isinstance(op, str)
                    and op.startswith("array_")
                    and family is None
                    and op not in {"array_source_open", "array_text_read", "array_fft"}
                    and any(
                        isinstance(t, dict) and t.get("kind") == "ndarray"
                        for t in _array_ports(node)
                    )
                )
            ):
                observed.setdefault("ndarray", set()).add(op)
                array_nodes.append(node)
                if op in {"array_binary", "array_update"}:
                    for role in ("left", "right"):
                        ty = node.get(role, {}).get("type")
                        if ty in ("f64", "c128"):
                            elements.add(ty)
            if op == "array_fft" or family == "fourier":
                observed.setdefault("fourier", set()).add(op)
                fourier_nodes.append(node)
            if isinstance(op, str) and (
                family == "builtin_scalar_mapping"
                or op.startswith(("mapping_", "structured_"))
            ):
                observed.setdefault("builtin_scalar_mapping", set()).add(op)
                mapping_nodes.append(node)
            if op in {"array_text_read", "array_source_open"}:
                observed.setdefault(
                    (
                        "text_array_io"
                        if op == "array_text_read"
                        else "text_array_source"
                    ),
                    set(),
                ).add(op)
            if (
                op in ("process_spawn", "process_request")
                or op == "host_call"
                and node.get("interface") in {"process.stdout", "process.wait"}
            ):
                observed.setdefault("process", set()).add(op)
            if op == "process_request":
                process_requests.append(node)
        if node.get("kind") == "ndarray":
            observed.setdefault("ndarray", set())
            element = node.get("element")
            elements.add(element if isinstance(element, str) else "unspecified")
        if node.get("type") == "integer" or node.get("result") == "integer":
            observed.setdefault("integer", set())
        if isinstance(node.get("kind"), str) and node["kind"] in MAPPING_TYPES:
            observed.setdefault("builtin_scalar_mapping", set())
            mapping_nodes.append(node)
        if mpi.dependency(node):
            observed.setdefault("mpi", set())
    return (
        observed,
        elements,
        array_nodes,
        process_requests,
        fourier_nodes,
        mapping_nodes,
        integer_nodes,
    )


def _mapping_binding(requirement, seen, nodes, module):
    if not isinstance(requirement, dict) or requirement.keys() != {
        "profile",
        "operations",
    }:
        raise ValueError("mapping requires a profile and operation list")
    if requirement["profile"] != MAPPING_PROFILE:
        raise QualificationFailure(
            "implementation_profile",
            "mapping.registered_profile",
            "unsupported mapping profile",
            profile=requirement["profile"],
            supported=MAPPING_PROFILE,
        )
    requested = requirement["operations"]
    if (
        not isinstance(requested, list)
        or any(not isinstance(op, str) for op in requested)
        or not seen <= set(requested)
    ):
        raise ValueError("mapping operation declarations must cover observed uses")
    if missing := sorted(set(requested) - _MAPPING_OPERATIONS):
        raise QualificationFailure(
            "implementation_profile",
            "mapping.registered_operations",
            "unregistered mapping operation",
            operations=missing,
        )
    fresh_writes = set()
    for node in nodes:
        if "op" not in node:
            mapping_value_type(node)
            continue
        op = node["op"]
        if (
            node.get("profile") != MAPPING_PROFILE
            or type(node.get("semantics_version")) is not int
            or node["semantics_version"] != 1
            or node.get("operator_family") != "builtin_scalar_mapping"
        ):
            raise ValueError("mapping operation requires the sealed builtin profile")
        if op == "structured_compare":
            ports = [node.get(k, {}).get("type") for k in ("left", "right")]
            structured = {"kind": "structured_value", "profile": MAPPING_PROFILE}
            if node.get("operator") not in ("eq", "ne") or ports not in (
                [structured, "text"],
                ["text", structured],
            ):
                if None in ports or not isinstance(node.get("operator"), str):
                    raise ValueError(
                        "structured comparison requires typed operands and an operator"
                    )
                raise QualificationFailure(
                    "implementation_profile",
                    "mapping.comparison_profile",
                    "only equality between a tagged value and text is qualified",
                    operator=node["operator"],
                    operand_types=ports,
                    site=node.get("site"),
                )
        if op == "mapping_get" and node.get("text_profile") not in module.get(
            "text_profiles", {}
        ):
            raise ValueError(
                "mapping lookup requires its Unicode key diagnostic profile"
            )
        if op in ("mapping_put", "mapping_merge"):
            proof = node.get("acyclicity")
            if (
                isinstance(proof, dict)
                and type(proof.get("semantics_version")) is int
                and proof
                == {"semantics_version": 1, "rule": "unpublished_fresh_builder"}
            ):
                fresh_writes.add(id(node))
                continue
            value = node.get("value") or {}
            tag = value.get("tag")
            if (
                op != "mapping_put"
                or not isinstance(proof, dict)
                or type(proof.get("semantics_version")) is not int
                or proof
                != {"semantics_version": 1, "rule": "nonmapping_scalar", "tag": tag}
                or tag not in ("unit", "bool", "i64", "f64", "text")
                or value.get("op") != "structured_pack"
                or value.get("value", {}).get("type") != tag
            ):
                raise ValueError(
                    "mapping writes require a verified acyclicity certificate"
                )
    if fresh_writes:
        for function in module.get("functions", ()):
            fresh_writes.difference_update(runtime.private_mapping_writes(function))
            if not fresh_writes:
                break
        if fresh_writes:
            raise ValueError(
                "mapping construction destination is not an unpublished fresh builder"
            )
    return {
        "provider": "cxx20-ordered-text-mapping",
        "profile": MAPPING_PROFILE,
        "operations": sorted(seen),
    }


def _ndarray_binding(requirement, observed, elements, nodes):
    if (
        not isinstance(requirement, dict)
        or requirement.keys() - {"device", "element", "elements", "operations"}
        or "element" in requirement
        and "elements" in requirement
        or not isinstance(requirement.get("operations", []), list)
        or any(not isinstance(op, str) for op in requirement.get("operations", ()))
    ):
        raise ValueError("requires CPU arrays and registered array operations")
    if requirement.get("device", "cpu") != "cpu":
        raise QualificationFailure(
            "implementation_profile",
            "ndarray.registered_device",
            "unregistered ndarray device",
            device=requirement["device"],
            supported="cpu",
        )
    declared = requirement.get("elements")
    if "elements" not in requirement:
        declared = (
            [requirement["element"]]
            if "element" in requirement
            else sorted(elements or {"f64"})
        )
    if (
        not isinstance(declared, list)
        or not declared
        or any(not isinstance(t, str) or not t or t == "unspecified" for t in declared)
        or len(set(declared)) != len(declared)
        or not elements.issubset(declared)
    ):
        raise QualificationFailure(
            "semantic_contract",
            "ndarray.element_contract",
            "unsupported or undeclared CPU array element types",
            declared=declared,
            observed=sorted(elements),
            fields=["type.element", "element", "elements"],
        )
    if missing_elements := sorted(set(declared) - {"f64", "c128"}):
        raise QualificationFailure(
            "implementation_profile",
            "ndarray.registered_elements",
            "unsupported CPU array element types",
            elements=missing_elements,
            supported=["c128", "f64"],
        )
    missing = sorted(
        observed.union(requirement.get("operations", ())) - _ARRAY_OPERATIONS
    )
    if missing:
        raise QualificationFailure(
            "implementation_profile",
            "ndarray.registered_operations",
            "unavailable CPU ndarray operations: " + ", ".join(missing),
            operations=missing,
        )
    qualified = {element: set() for element in declared}
    for node in nodes:
        op = node["op"]
        if op == "array_from_sequence":
            source = node.get("value", {}).get("type")
            result = node.get("type")
            if (
                not isinstance(source, dict)
                or source.get("kind") != "sequence_ref"
                or not isinstance(source.get("element"), str)
                or source.get("element") not in {"bool", "i64", "f64", "c128"}
                or not isinstance(result, dict)
                or result.get("rank") != 1
                or result.get("element") not in {"f64", "c128"}
                or source.get("element") == "c128"
                and result.get("element") != "c128"
                or type(node.get("semantics_version")) is not int
                or node.get("semantics_version") != 1
                or node.get("conversion") != "numpy_numeric_sequence"
                or node.get("ownership") != "fresh_independent_storage"
                or node.get("source_read") != "current_elements_in_index_order"
                or set(node.get("effects", ())) != {"MEM", "STATE", "XFER"}
                or node.get("reads") != ["sequence.state"]
                or node.get("writes") != ["new.array_storage"]
                or node.get("may_raise") is not True
            ):
                raise QualificationFailure(
                    "semantic_contract",
                    "ndarray.sequence_materialization",
                    "array_from_sequence requires a closed numeric list and fresh rank-one array storage",
                    operation=op,
                    site=node.get("site"),
                )
        # Associate each use with its own ports. One image can have both
        # complex arithmetic and real-only operations on independent arrays.
        kinds = set()
        for port in _array_ports(node):
            if port is None:
                raise QualificationFailure(
                    "semantic_contract",
                    "ndarray.operand_type",
                    f"{op} requires a typed operand",
                    operation=op,
                    site=node.get("site"),
                    field="type",
                )
            if isinstance(port, dict) and port.get("kind") == "ndarray":
                with qualification_check(
                    "semantic_contract",
                    "ndarray.array_type",
                    operation=op,
                    site=node.get("site"),
                    observed=port,
                ):
                    array_rank_domain(port)
                kinds.add(port["element"])
            elif isinstance(port, str):
                kinds.add(port)
        if len(kinds) != 1 or not kinds.issubset(qualified):
            raise QualificationFailure(
                "semantic_contract",
                "ndarray.operand_element",
                f"{op} requires one declared operand/result element type",
                operation=op,
                site=node.get("site"),
                observed_elements=sorted(kinds),
                declared_elements=declared,
            )
        element = next(iter(kinds))
        if element == "c128" and op in {"array_iota", "array_format"}:
            raise QualificationFailure(
                "implementation_profile",
                "ndarray.operation_element",
                f"{op} is unavailable for c128",
                operation=op,
                element=element,
                site=node.get("site"),
            )
        if op in _ARRAY_OPERATORS:
            operator = node.get("operator")
            if not isinstance(operator, str):
                raise QualificationFailure(
                    "semantic_contract",
                    "ndarray.operator_operand",
                    f"{op} requires an operator",
                    operation=op,
                    site=node.get("site"),
                    field="operator",
                )
            if operator not in _ARRAY_OPERATORS[op][element]:
                raise QualificationFailure(
                    "implementation_profile",
                    "ndarray.operation_element",
                    f"{op}.{operator} is unavailable for {element}",
                    operation=op,
                    element=element,
                    operator=operator,
                    site=node.get("site"),
                )
            op = f"{op}.{operator}"
        qualified[element].add(op)
    if declared == ["c128"] and set(requirement.get("operations", ())) & {
        "array_iota",
        "array_format",
    }:
        raise QualificationFailure(
            "implementation_profile",
            "ndarray.operation_element",
            "c128 requires a registered complex array operation",
            element="c128",
            operations=sorted(
                set(requirement["operations"]) & {"array_iota", "array_format"}
            ),
        )
    with qualification_check(
        "target_platform",
        "ndarray.sdk_identity",
        provider="xtensor-cpu",
        device="cpu",
        packages=["xtensor", "xtl"],
    ):
        sdk = array_sdk(required=True)
    return dict(
        sdk,
        device="cpu",
        elements=sorted(qualified),
        operations_by_element={k: sorted(v) for k, v in qualified.items()},
        selection="fixed semantic provider; no global performance optimum claimed",
    )


def _text_array_binding(requirement, observed, module):
    import ctypes
    import platform

    profile = "numpy-2.5.5-whitespace-f64-v1"
    if (
        not isinstance(requirement, dict)
        or requirement.keys() - {"operations", "profile"}
        or not isinstance(requirement.get("operations", []), list)
        or any(not isinstance(op, str) for op in requirement.get("operations", ()))
    ):
        raise ValueError("requires the registered whitespace binary64 text parser")
    operations = sorted(observed.union(requirement.get("operations", ())))
    if requirement.get("profile", profile) != profile or set(operations) - {
        "array_text_read"
    }:
        raise QualificationFailure(
            "implementation_profile",
            "text_input.registered_profile",
            "requires the registered whitespace binary64 text parser",
            profile=requirement.get("profile", profile),
            operations=operations,
        )
    for node in walk_operations(
        [module.get(key, ()) for key in ("functions", "globals", "records")]
    ):
        if node["op"] != "array_text_read":
            continue
        if node.get("source_origin") != {"kind": "named_text_file"}:
            raise QualificationFailure(
                "semantic_contract",
                "text_input.stream_origin",
                "text input requires a proven borrowed named UTF-8 file stream; unresolved source/diagnostic identity cannot select this provider",
                site=node.get("site"),
                source_origin=node.get("source_origin"),
                required={"kind": "named_text_file"},
            )
        parser = node.get("parser", {})
        if (
            parser.get("profile") != profile
            or parser.get("encoding") != "unicode"
            or parser.get("delimiter") != "whitespace_runs"
            or parser.get("comments") != "#"
            or type(parser.get("ndmin")) is not int
            or parser["ndmin"] != 0
            or node.get("source", {}).get("type")
            != dict(kind="handle", resource="text_stream")
            or set(node.get("effects", ())) != {"IO", "STATE", "VAL", "MEM"}
            or array_rank_domain(node.get("type")) != (0, 1, 2)
            or node["type"]["element"] != "f64"
        ):
            raise ValueError("unregistered text parsing contract")
        location, warning = node.get("location", {}), node.get("warning", {})
        if (
            not isinstance(location.get("diagnostic_file"), str)
            or not location["diagnostic_file"]
            or type(location.get("line")) is not int
            or location["line"] <= 0
            or warning.get("operation") != "loadtxt"
            or not isinstance(warning.get("module"), str)
            or not warning["module"]
            or not isinstance(warning.get("source_line"), str)
            or not warning["source_line"]
        ):
            raise ValueError("text input requires its captured warning site and line")
    if platform.system() != "Linux":
        raise QualificationFailure(
            "target_platform",
            "text_input.platform",
            "the registered text parser requires Linux locale APIs",
            platform=platform.system(),
            required="Linux",
        )
    symbols = ("strtod_l", "newlocale", "freelocale")
    with qualification_check(
        "target_platform",
        "text_input.libc_symbols",
        library="libc",
        symbols=list(symbols),
    ):
        library = ctypes.CDLL(None)
        if missing := [name for name in symbols if not hasattr(library, name)]:
            raise QualificationFailure(
                "target_platform",
                "text_input.libc_symbols",
                "native C-locale number conversion is unavailable",
                library="libc",
                missing=missing,
            )
    return dict(
        provider="scar-cpu-text-input",
        profile=profile,
        semantics_version=1,
        source="borrowed_named_utf8_stream",
        numeric_converter="system-libc:strtod_l:C",
    )


def _process_binding(requirement, observed, module, context):
    import ctypes
    import platform

    if (
        not isinstance(requirement, dict)
        or requirement.keys() - {"operations", "binding"}
        or requirement.get("binding", "declared_images")
        not in ("declared_images", "explicit_targets")
        or not isinstance(requirement.get("operations", []), list)
        or any(not isinstance(op, str) for op in requirement.get("operations", ()))
    ):
        raise ValueError("requires declared image process operations")
    if missing := sorted(
        observed.union(requirement.get("operations", ()))
        - {"process_spawn", "host_call"}
    ):
        raise QualificationFailure(
            "implementation_profile",
            "process.registered_operations",
            "requires declared image process operations",
            operations=missing,
        )
    if not context:
        raise ValueError("process targets require the enclosing sealed image bundle")
    images = context["images"]
    declarations = [
        {k: image[k] for k in ("id", "entry", "output")} for image in images
    ]
    if (
        module.get("execution_images") != declarations
        or module.get("current_image") != context["current_image"]
    ):
        raise ValueError(
            "process image declarations disagree with the enclosing bundle"
        )
    targets = {
        image["id"]: dict(
            image_id=image["id"], output=image["output"], ir_sha256=image["ir_sha256"]
        )
        for image in images
    }
    references, launches = set(), {}
    for node in walk_operations(
        [module.get(k, ()) for k in ("functions", "globals", "records")]
    ):
        if node["op"] == "process_spawn":
            if node.get("image_id") not in targets:
                raise ValueError(
                    "process launch references an undeclared or unsealed image"
                )
            image = node["image_id"]
            if image in context.get("unavailable", ()):
                raise QualificationFailure(
                    "semantic_contract",
                    "process.image_completion",
                    "process launch requires a complete execution IR for its target",
                    image_id=image,
                    site=node.get("site"),
                )
            references.add(image)
            if "launcher" in node:
                qualify_worker = context.get("qualify_mpi")
                if qualify_worker is None:
                    raise ValueError("MPI launch requires the sealed worker IR package")
                _, worker = qualify_worker(image)
                site, binding = mpi.qualify_launcher(
                    node, targets[image], worker, context["launcher_cache"]
                )
                if site in launches and launches[site] != binding:
                    raise ValueError("MPI launch site has competing physical bindings")
                launches[site] = binding
                node["launch_binding"] = site
            elif node.get("shell", False) is not False or "source_command" in node:
                raise QualificationFailure(
                    "external_effects",
                    "process.launcher_binding",
                    "shell image launch requires an explicitly qualified launcher",
                    site=node.get("site"),
                    image_id=image,
                    fields=["launcher", "source_command"],
                )
        elif node.get("operator_family") == "process" and (
            node["op"] != "host_call"
            or node.get("interface") not in {"process.stdout", "process.wait"}
        ):
            raise QualificationFailure(
                "implementation_profile",
                "process.registered_interface",
                "unregistered process interface",
                interface=node.get("interface"),
                operation=node["op"],
            )
    if platform.system() != "Linux" or not Path("/proc/self/exe").is_file():
        raise QualificationFailure(
            "target_platform",
            "process.executable_namespace",
            "image processes require the Linux executable namespace",
            platform=platform.system(),
            required_path="/proc/self/exe",
        )
    symbol = "posix_spawn_file_actions_addclosefrom_np"
    with qualification_check("target_platform", "process.spawn_abi", symbol=symbol):
        if not hasattr(ctypes.CDLL(None), symbol):
            raise ValueError(
                "process deployment lacks the required native spawn file actions"
            )
    return dict(
        provider="posix_spawn",
        target="linux-cxx20",
        binding="declared_images",
        images={name: targets[name] for name in sorted(references)},
        **({"launches": launches} if launches else {}),
        environment_contract="immutable executable generation during execution; inherited cwd, environment and input",
    )


def _qualify(module, image_context=None):
    requested = module.get("operator_requirements", {})
    if not isinstance(requested, dict):
        raise ValueError("operator requirements must be a family-indexed table")
    requested = dict(requested)
    (
        observed,
        array_elements,
        array_nodes,
        process_requests,
        fourier_nodes,
        mapping_nodes,
        integer_nodes,
    ) = _observed_operators(module)
    for family, operators in observed.items():
        if family not in requested:
            requirement = dict(operations=sorted(operators))
            if family == "builtin_scalar_mapping":
                requirement["profile"] = MAPPING_PROFILE
            if family == "integer":
                requirement["semantics_version"] = 1
            if family == "process" and process_requests:
                requirement["binding"] = "explicit_targets"
            if family == "ndarray":
                requirement["device"] = "cpu"
                if len(array_elements) > 1:
                    requirement["elements"] = sorted(array_elements)
                else:
                    requirement["element"] = next(iter(array_elements), "f64")
            requested[family] = requirement
    bindings, issues = {}, []
    process = requested.get("process")
    if (
        not process_requests
        and isinstance(process, dict)
        and isinstance(process.get("operations"), list)
        and "process_request" in process["operations"]
    ):
        # A family declaration alone supplies no executable target either.
        process_requests = [{}]
    if process_requests:
        for request in process_requests:
            issues.append(
                dict(
                    stage="B1",
                    code="missing_process_target_binding",
                    requirement="process",
                    target="cpu-cxx20",
                    request_id=request.get("request_id"),
                    site=request.get("site"),
                    location=request.get("location"),
                    binding_obligations=request.get("binding_obligations", []),
                    qualification=dict(
                        stage="external_effects",
                        rule="process.executable_binding",
                        evidence=dict(
                            request_id=request.get("request_id"),
                            site=request.get("site"),
                            target_binding=request.get("target_binding"),
                            binding_obligations=request.get("binding_obligations", []),
                        ),
                    ),
                    detail="logical process request requires a qualified executable target before physical scheduling",
                )
            )
    for family, requirement in sorted(requested.items()):
        if (process_requests or "text_array_source" in observed) and (
            family
            in {"ndarray", "mpi", "process", "text_array_io", "fourier", "integer"}
        ):
            # Deployment cannot proceed. Report other unregistered families
            # below without probing any physical providers.
            continue
        operators = observed.get(family, set())
        evidence = dict(family=family, observed_operations=sorted(operators))
        if isinstance(requirement, dict):
            evidence.update(
                {
                    k: requirement[k]
                    for k in ("profile", "binding", "api", "device", "operations")
                    if k in requirement
                }
            )
        checks = None
        try:
            with qualification_check(
                "semantic_contract", family + ".ir_contract", **evidence
            ):
                if family == "mpi":
                    resolve_worker = (image_context or {}).get("qualify_mpi")
                    requested[family], bindings[family] = (
                        resolve_worker(image_context["current_image"], module)
                        if resolve_worker
                        else mpi.qualify(
                            module,
                            requirement,
                            explicit=family in module.get("operator_requirements", {}),
                        )
                    )
                elif family == "process":
                    bindings[family] = _process_binding(
                        requirement, operators, module, image_context
                    )
                elif family == "text_array_io":
                    bindings[family] = _text_array_binding(
                        requirement, operators, module
                    )
                elif family == "fourier":
                    bindings[family] = fourier.qualify(
                        requirement, fourier_nodes, requested
                    )
                elif family == "integer":
                    bindings[family] = integer.qualify(
                        requirement,
                        integer_nodes,
                        cache=(image_context or {}).get("compiler_features"),
                    )
                elif family == "builtin_scalar_mapping":
                    bindings[family] = _mapping_binding(
                        requirement, operators, mapping_nodes, module
                    )
                elif family == "ndarray":
                    bindings[family] = _ndarray_binding(
                        requirement, operators, array_elements, array_nodes
                    )
                elif family == "text_array_source":
                    raise QualificationFailure(
                        "implementation_profile",
                        "text_input.source_resolver",
                        "NumPy path resolution includes local/compressed alternatives and URL caching; no qualified native resolver is registered",
                        **evidence,
                    )
                else:
                    if family == "module_registry":
                        checks = runtime.registry_checks(module)
                        if checks["guard_count"] or checks["open_effect_count"]:
                            raise QualificationFailure(
                                "external_effects",
                                "module_registry.observable_semantics",
                                "module registry lookup, source-visible state and continuations require an implementation",
                                **evidence,
                                operator_counts=checks["operator_counts"],
                                required_read_domains=checks["required_read_domains"],
                                checks_field="registry_checks",
                            )
                    raise QualificationFailure(
                        "implementation_profile",
                        "operator_family.registered",
                        "no target implementation is registered for this operator family",
                        **evidence,
                    )
        except QualificationFailure as error:
            issues.append(
                dict(
                    stage="B1",
                    code="missing_operator_binding",
                    requirement=family,
                    target="cpu-cxx20",
                    detail=str(error),
                    qualification=error.qualification,
                    **({"registry_checks": checks} if checks is not None else {}),
                )
            )
    native, runtime_issues = runtime.qualify(module, probe=not issues)
    issues.extend(runtime_issues)
    if native:
        bindings["posix_runtime"] = native
    return requested, bindings, issues


def _load_module(path, ir):
    module = dict(
        ir["module"], functions=[read_member(path.parent, r) for r in ir["functions"]]
    )
    if "source_imports" in module:
        module["source_imports"] = read_member(path.parent, module["source_imports"])
    if "source_obligations" in module:
        module["source_obligations"] = read_member(
            path.parent, module["source_obligations"]
        )
    if "resource_flow" in module.get("resource_model", {}):
        inventory = dict(module["resource_model"])
        inventory["resource_flow"] = read_member(
            path.parent, inventory["resource_flow"]
        )
        module["resource_model"] = inventory
    return module


def _schedule_image(
    path, revision, ir, destination, policy, workers, image_context=None
):
    if ir["status"] == "INCOMPLETE":
        if ir.get("functions"):
            raise ValueError("incomplete execution IR contains executable functions")
        issues = []
        if "unresolved" in ir:
            records = read_member(path.parent, ir["unresolved"])
            if (
                not isinstance(records, list)
                or not records
                or not all(
                    isinstance(record, dict) and isinstance(record.get("code"), str)
                    for record in records
                )
            ):
                raise ValueError("invalid unresolved execution IR diagnostics")
            issues = records
        if not issues:
            # Older IR packages only recorded the completion status.
            issues.append(
                dict(
                    code="incomplete_execution_ir",
                    reported_by="M4",
                    detail="input image was sealed without an executable body",
                )
            )
        fields = dict(
            stage="B1", status="INCOMPLETE", ir_sha256=revision, issues=issues
        )
        return _seal_schedule(
            destination,
            fields,
            document(PHYSICAL, target="cpu-cxx20"),
            dict(entry=ir["module"]["entry"]),
        )
    if "unresolved" in ir:
        raise ValueError("ready execution IR contains unresolved diagnostics")
    module = _load_module(path, ir)
    inventory = dict(module.get("resource_model", {}))
    if "artifact_flow" in inventory:
        inventory["artifact_flow"] = copy_member(
            path.parent, inventory["artifact_flow"], destination, "analysis"
        )
        module["resource_model"] = inventory
    module.pop("image_bindings", None)
    module.pop("process_launch_bindings", None)
    module.pop("posix_runtime_binding_sha256", None)
    module.pop("parameter_access_binding_sha256", None)
    module.pop("compilation_binding_sha256", None)
    module.pop("math_interface_binding_sha256", None)
    module.pop("numeric_permissions_binding_sha256", None)
    module.pop("fourier_binding_sha256", None)
    module.pop("integer_binding_sha256", None)
    module.pop("builtin_scalar_mapping_binding_sha256", None)
    requirements, bindings, issues = _qualify(module, image_context)
    import_effects = runtime.import_effects(module, bindings)
    import_receipt = (
        member(destination, "analysis", module["source_imports"])
        if import_effects is not None
        else None
    )
    if binding := bindings.get("posix_runtime"):
        module["posix_runtime_binding_sha256"] = digest(json_bytes(binding))
    if binding := bindings.get("fourier"):
        module["fourier_binding_sha256"] = digest(json_bytes(binding))
    if binding := bindings.get("integer"):
        module["integer_binding_sha256"] = digest(json_bytes(binding))
    if binding := bindings.get("builtin_scalar_mapping"):
        module["builtin_scalar_mapping_binding_sha256"] = digest(json_bytes(binding))
    if "process" in bindings:
        module["image_bindings"] = bindings["process"]["images"]
        if "launches" in bindings["process"]:
            module["process_launch_bindings"] = {
                site: digest(json_bytes(binding))
                for site, binding in bindings["process"]["launches"].items()
            }
    status = "INCOMPLETE" if issues else "READY"
    fields = dict(stage="B1", status=status, ir_sha256=revision)
    if issues:
        fields["issues"] = issues
        physical = document(PHYSICAL, target="cpu-cxx20")
        scheduled = dict(entry=module["entry"])
        if "artifact_flow" in inventory:
            scheduled["module"] = dict(
                entry=module["entry"],
                resource_model={"artifact_flow": inventory["artifact_flow"]},
            )
    else:
        from .cpu import schedule

        access = toolchain.parameter_access(
            module, cache=(image_context or {}).get("compiler_features")
        )
        if access:
            module["parameter_access_binding_sha256"] = digest(json_bytes(access))
        compilation = toolchain.compilation(
            module, cache=(image_context or {}).get("compiler_features")
        )
        module["compilation_binding_sha256"] = digest(json_bytes(compilation))
        mpi = bindings.get("mpi", {})
        uses_mpi = bool(mpi.get("called_or_possible") or mpi.get("address_taken"))
        selected, physical = schedule(module, policy, 1 if uses_mpi else workers)
        if uses_mpi:
            # No rank placement or guaranteed MPI thread level is sealed yet.
            # The cap applies to SCAR's generated teams, including calibration.
            worker_policy = physical["worker_policy"]
            caps = worker_policy["caps"]
            caps["mpi_intra_rank_limit"] = 1
            if workers is None:
                del caps["requested_worker_limit"]
            else:
                caps["requested_worker_limit"] = workers
            worker_policy["mpi"] = dict(
                scope="scar_generated_intra_rank_workers",
                caller_limit=workers,
                reason="rank_placement_and_provided_thread_level_unverified",
            )
        physical["compilation"] = compilation
        if access:
            physical["parameter_access"] = access
        functions = [
            member(destination, "scheduled-units", fn) for fn in selected["functions"]
        ]
        source_members = {
            fn["name"]: receipt
            for fn, receipt in zip(module["functions"], ir["functions"], strict=True)
        }
        numeric = toolchain.numeric_permissions(
            selected,
            source_members,
            functions,
            compilation,
            cache=(image_context or {}).get("compiler_features"),
        )
        if numeric:
            physical["numeric_permissions"] = numeric
            selected["numeric_permissions_binding_sha256"] = digest(json_bytes(numeric))
        math = toolchain.math_interfaces(
            selected, source_members, functions, compilation
        )
        if math:
            physical["math_interfaces"] = math
            selected["math_interface_binding_sha256"] = digest(json_bytes(math))
        del selected["functions"]
        if import_receipt is not None:
            selected["source_imports"] = import_receipt
        if "source_obligations" in selected:
            selected["source_obligations"] = member(
                destination, "analysis", selected["source_obligations"]
            )
        if "resource_flow" in selected.get("resource_model", {}):
            inventory = dict(selected["resource_model"])
            inventory["resource_flow"] = member(
                destination, "resources", inventory["resource_flow"]
            )
            selected["resource_model"] = inventory
        scheduled = dict(module=selected, functions=functions)
    if native := bindings.get("posix_runtime"):
        scheduled["runtime_boundaries"] = dict(
            provider="posix_runtime",
            qualification=native["qualification"],
            descendant_closure=native["descendant_closure"],
            sites=sorted(native["sites"]),
        )
    if requirements:
        physical["operator_requirements"] = requirements
    if import_effects is not None:
        physical["source_import_effects"] = dict(import_effects, ledger=import_receipt)
    if bindings:
        physical["operator_backends"] = bindings
    inventory = module.get("resource_model")
    if inventory:
        physical["source_resources"] = dict(
            sha256=digest(json_bytes(inventory)),
            count=len(inventory.get("resources", ())),
        )
        if "resource_flow" in inventory:
            physical["resource_flow"] = member(
                destination, "resources", inventory["resource_flow"]
            )
    return _seal_schedule(destination, fields, physical, scheduled)


def _seal_schedule(destination, fields, physical, scheduled):
    physical.update(fields)
    physical_member = member(destination, "scheduled-units", physical)
    write_json(destination / "physical-plan.json", physical)
    package = document(
        "scar.scheduled-ir",
        **fields,
        **scheduled,
        physical=physical_member,
    )
    path = destination / "scheduled-ir.json"
    write_json(path, package)
    return path, fields["status"], fields.get("issues", [])


def schedule_ir(source, destination, *, policy="auto", workers=None):
    if workers is not None and (type(workers) is not int or workers < 1):
        raise ValueError("worker_limit must be a positive integer")
    path, revision, ir = open_product(source, "exec-ir.json", IR)
    if ir.get("status") not in {"READY", "INCOMPLETE"}:
        raise ValueError("invalid execution IR status")
    destination = Path(destination).resolve()
    if ir.get("kind") != "image_bundle":
        scheduled, _, _ = _schedule_image(
            path, revision, ir, destination, policy, workers
        )
        return scheduled

    images, root = ir["images"], ir["root_image"]
    validate_images(images, root)
    audit = (
        {"workflow": copy_member(path.parent, ir["workflow"], destination, "analysis")}
        if "workflow" in ir
        else {}
    )
    inputs = []
    for image in images:
        child = read_image_product(path.parent, image, "ir", "exec-ir.json", IR)
        child_ir = child[2]
        if (
            child_ir.get("status") not in {"READY", "INCOMPLETE"}
            or child_ir.get("kind") == "image_bundle"
            or child_ir.get("module", {}).get("entry") != image["entry"]
        ):
            raise ValueError(
                f"image {image['id']} requires an execution IR for its declared entry"
            )
        inputs.append((image, child))

    unavailable = {
        image["id"] for image, child in inputs if child[2]["status"] == "INCOMPLETE"
    }
    if bool(unavailable) != (ir["status"] == "INCOMPLETE"):
        raise ValueError(
            "bundle completion status disagrees with sealed image statuses"
        )

    scheduled, plans, issues = [], [], []
    catalog = [
        {k: image[k] for k in ("id", "entry", "output")} | {"ir_sha256": child[1]}
        for image, child in inputs
    ]
    child_inputs = {image["id"]: child for image, child in inputs}
    qualified, launcher_cache, compiler_features = {}, {}, {}

    def qualify_mpi(image_id, module=None):
        # A controller may precede its worker in the input manifest. Qualify
        # the worker itself, reusing the result for its own schedule.
        if image_id not in qualified:
            if module is None:
                child_path, _, child_ir = child_inputs[image_id]
                module = _load_module(child_path, child_ir)
            claims = module.get("operator_requirements", {})
            try:
                result = mpi.qualify(
                    module, claims.get("mpi"), explicit="mpi" in claims
                )
                qualified[image_id] = result, None
            except ValueError as exc:
                qualification = (
                    exc.qualification
                    if isinstance(exc, QualificationFailure)
                    else dict(
                        stage="semantic_contract", rule="mpi.ir_contract", evidence={}
                    )
                )
                qualified[image_id] = None, (str(exc), qualification)
        result, error = qualified[image_id]
        if error is not None:
            detail, qualification = error
            raise QualificationFailure(
                qualification["stage"],
                qualification["rule"],
                f"MPI capability of image {image_id}: {detail}",
                **{**qualification["evidence"], "image_id": image_id},
            )
        return result

    for image, child in inputs:
        identity = {key: image[key] for key in ("id", "entry", "output")}
        child_dir = destination / "images" / image["id"]
        child_path, child_status, child_issues = _schedule_image(
            *child,
            child_dir,
            policy,
            workers,
            image_context=dict(
                current_image=image["id"],
                images=catalog,
                unavailable=unavailable,
                qualify_mpi=qualify_mpi,
                launcher_cache=launcher_cache,
                compiler_features=compiler_features,
            ),
        )
        identity["status"] = child_status
        issues.extend(dict(issue, image_id=image["id"]) for issue in child_issues)
        scheduled.append(
            dict(identity, scheduled=file_receipt(destination, child_path))
        )
        plans.append(
            dict(
                identity,
                physical=file_receipt(destination, child_dir / "physical-plan.json"),
            )
        )
    fields = dict(
        kind="image_bundle",
        stage="B1",
        status="INCOMPLETE" if issues else "READY",
        root_image=root,
        ir_sha256=revision,
    )
    if issues:
        fields["issues"] = issues
    physical_path = destination / "physical-plan.json"
    write_json(
        physical_path,
        document(
            PHYSICAL,
            **fields,
            planning_scope="per_image",
            interprocess_schedule="unmodeled",
            images=plans,
        ),
    )
    package = document(
        "scar.scheduled-ir",
        **fields,
        **audit,
        images=scheduled,
        physical=file_receipt(destination, physical_path),
    )
    result = destination / "scheduled-ir.json"
    write_json(result, package)
    return result

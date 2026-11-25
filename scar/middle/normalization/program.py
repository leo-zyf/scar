"""Normalize closed symbol bindings and call contracts without reading graph files."""

from copy import deepcopy
from ..contracts import is_complete, publish_program, require_program
from scar.contract import (
    json_bytes,
    semantic_form as semantics,
    validate_access_contract,
    walk_operations as _operations,
)


def _abi_type(t):
    if not isinstance(t, dict):
        return t
    if t["kind"] in {"const", "volatile"}:
        return _abi_type(t["of"])
    if t["kind"] in {"ptr", "ref"}:
        return {"ptr": _abi_type(t["to"])}
    return t


def _abi(fn):
    return [
        _abi_type(fn["result"]),
        [_abi_type(p["type"]) for p in fn["params"]],
        bool(fn.get("variadic")),
    ]


def _parameter_access(function):
    """Bind source preconditions to the accessed bytes of definition parameters."""
    claims = []
    for param in function["params"]:
        if "access_precondition" not in param:
            continue
        precondition = param.pop("access_precondition")
        if (
            not isinstance(precondition, dict)
            or set(precondition)
            != {"rule", "scope", "paths", "basis", "readonly_when_accessed"}
            or precondition["scope"] != "invocation"
        ):
            raise ValueError(
                f"invalid access precondition on {function['name']}.{param['name']}"
            )
        claims.append(
            {
                "parameter": param["name"],
                "region": "accessed_referent_bytes",
                "rule": precondition["rule"],
                "paths": precondition["paths"],
                "basis": precondition["basis"],
                "readonly_when_accessed": precondition["readonly_when_accessed"],
            }
        )
    if claims:
        function["access_contract"] = {
            "semantics_version": 1,
            "scope": "invocation",
            "parameters": sorted(claims, key=lambda claim: claim["parameter"]),
        }
    validate_access_contract(function)


def _typed_object_use(node, field, method):
    owner = {"kind": "record", "name": method["owner"]}
    if method["kind"] == "constructor":
        return (
            node.get("op") in {"object_construct", "object_new"}
            and field == "symbol"
            and node.get("type") == owner
            and node.get("parameter_types") == [p["type"] for p in method["params"]]
            and isinstance(node.get("args"), list)
            and len(node["args"]) == len(method["params"])
        )
    receiver = node.get("value")
    return (
        node.get("op") == "object_destroy"
        and field == "destructor"
        and node.get("owner") == method["owner"]
        and isinstance(receiver, dict)
        and ("type" not in receiver or receiver["type"] == {"kind": "ptr", "to": owner})
    )


def _object_member_interfaces(module, declarations, definitions):
    """Separate compiler-supplied object operations from callable ABI imports."""
    records = {r["name"]: r for r in module["records"]}
    methods = {}
    for external in module["externs"]:
        method = external.get("method", {})
        name, owner = external["name"], method.get("owner")
        record = records.get(owner)
        if (
            name in definitions
            or name == module["entry"]
            or not record
            or method.get("kind") not in {"constructor", "destructor"}
            or method.get("symbol") != name
            or method.get("defaulted") is not True
            or method.get("deleted") is not False
            or external.get("variadic")
            or external["result"] != "void"
            or method.get("result") != "void"
        ):
            continue
        declared = [m for m in record.get("methods", ()) if m.get("symbol") == name]
        public_type = bool(
            record.get("complete")
            and record.get("provider")
            and record.get("symbol")
            and record["provider"] == external.get("provider")
        )
        if (
            (record.get("opaque") and not public_type)
            or not (method.get("implicit") is True or declared or public_type)
            or any(m != method for m in declared)
        ):
            continue
        expected = [
            {"kind": "ptr", "to": {"kind": "record", "name": owner}},
            *[p["type"] for p in method["params"]],
        ]
        if any(
            signature["result"] != "void"
            or signature.get("variadic")
            or [p["type"] for p in signature["params"]] != expected
            for signature in (external, *declarations.get(name, ()))
        ):
            continue
        methods[name] = method
    if not methods:
        return []

    # Inspect every executable occurrence, including indirect target lists,
    # initializers and nested operations. A record's own method declaration
    # describes its object semantics and is not a callable use.
    declared_owners = {
        id(method): record["name"]
        for record in records.values()
        for method in record.get("methods", ())
    }
    uses, other = {}, set()
    pending = [*module["functions"], *module["globals"], *module["records"]]
    while pending:
        node = pending.pop()
        if isinstance(node, list):
            for value in node:
                if isinstance(value, str) and value in methods:
                    other.add(value)
                elif isinstance(value, (dict, list)):
                    pending.append(value)
            continue
        op = node.get("op")
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                pending.append(value)
                continue
            if not isinstance(value, str) or value not in methods:
                continue
            method = methods[value]
            if key == "symbol" and declared_owners.get(id(node)) == method["owner"]:
                continue
            if key == "value" and op in {"literal", "string"}:
                continue
            if not _typed_object_use(node, key, method):
                other.add(value)
                continue
            counts = uses.setdefault(value, {})
            counts[op] = counts.get(op, 0) + 1

    closed = uses.keys() - other
    module["externs"] = [
        decl for decl in module["externs"] if decl["name"] not in closed
    ]
    changes = []
    for name in sorted(closed):
        declarations.pop(name, None)
        method = methods[name]
        changes.append(
            dict(
                code="language_object_special_member",
                symbol=name,
                owner=method["owner"],
                member_kind=method["kind"],
                uses=uses[name],
                proof="consistent special-member declarations and exclusively typed object lifetime operations; construction and destruction remain in the executable graph",
            )
        )
    return changes


def _discarded_scalar_result(
    symbol, declarations, definition, functions, globals_, architecture
):
    if architecture != "x86_64" or definition["result"] != "void":
        return None
    integers = {prefix + str(bits) for prefix in ("i", "u") for bits in (8, 16, 32, 64)}
    expected = _abi(definition)
    for signature in declarations:
        actual = _abi(signature)
        if (
            not isinstance(signature["result"], str)
            or signature["result"] not in integers | {"void"}
            or actual[1:] != expected[1:]
        ):
            return None
    calls = 0
    invalid = False

    def visit(value, parent=None, field=None):
        nonlocal calls, invalid
        if isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            if value.get("op") == "function_ref" and value["name"] == symbol:
                invalid = True
            if value.get("op") == "call" and value["callee"] == symbol:
                calls += 1
                if not parent or parent.get("op") != "expr" or field != "value":
                    invalid = True
            for key, child in value.items():
                visit(child, value, key)

    visit([list(functions.values()), list(globals_.values())])
    if invalid or not calls:
        return None
    return dict(
        code="discarded_scalar_result_abi",
        symbol=symbol,
        classification="baseline_defect",
        calls=calls,
        proof="direct calls discard every result; no address escape; parameter ABI agrees; target x86_64 scalar return has no hidden result argument",
        lowering="use the defined void result and preserve arguments and effects",
    )


def _unqualified(spec):
    while isinstance(spec, dict) and spec.get("kind") in {"const", "volatile"}:
        spec = spec["of"]
    return spec


def _same_referent(source, target):
    """Qualification conversions preserve type, indirections and array bounds."""
    source, target = _unqualified(source), _unqualified(target)
    if source == target:
        return True
    if not isinstance(source, dict) or not isinstance(target, dict):
        return False
    kind = source.get("kind")
    if kind != target.get("kind"):
        return False
    if kind == "ptr":
        return _same_referent(source["to"], target["to"])
    if kind == "array" and source["shape"] == target["shape"]:
        return _same_referent(source["of"], target["of"])
    return False


def _address_argument(value, source, dest):
    """Express the existing address binding; produce no arithmetic or data copy."""
    left, right = _unqualified(source), _unqualified(dest)
    if left == right or not isinstance(left, dict) or not isinstance(right, dict):
        return value
    kinds = left.get("kind"), right.get("kind")
    if not all(k in {"ref", "ptr"} for k in kinds):
        return value
    if not _same_referent(left["to"], right["to"]):
        raise ValueError(
            "callsite and target refer to different element types or bounds"
        )
    if kinds == ("ref", "ptr"):
        value = dict(op="unary", operator="addr", value=value)
    if left["to"] != right["to"]:
        kind = "ref" if kinds == ("ref", "ref") else "ptr"
        null_pointer = (
            kind == "ptr"
            and value.get("op") == "literal"
            and value["value"] in (None, 0)
        )
        value = {
            "op": "cast",
            "mode": "static" if null_pointer else "qualification",
            "conversion": "argument_qualification",
            "source_type": {"kind": kind, "to": left["to"]},
            "type": {"kind": kind, "to": right["to"]},
            "value": value,
        }
    if kinds == ("ptr", "ref"):
        value = dict(op="unary", operator="deref", value=value)
    return value


def _bind_call_arguments(functions, externs, globals_, records):
    """Make per-call reference/pointer ABI bindings explicit in executable IR."""
    declarations = {**externs, **functions}
    changes, issues = [], []
    for node in _operations(
        [list(functions.values()), list(globals_.values()), records]
    ):
        if node["op"] == "cast" and node.get("conversion") == "legacy_string_pointer":
            changes.append(
                dict(
                    code="explicit_legacy_string_binding",
                    source_type=node["source_type"],
                    target_type=node["type"],
                    location=node.get("location"),
                    proof="compiler-admitted literal binding; preserve static read-only storage with an explicit qualification conversion",
                )
            )
        if node["op"] != "call":
            continue
        target = declarations.get(node["callee"])
        if target is None:
            continue
        expected, actual = len(target["params"]), len(node["args"])
        if actual < expected or actual != expected and not target.get("variadic"):
            issues.append(
                dict(
                    code="argument_count",
                    symbol=node["callee"],
                    detail="call arguments do not satisfy the closed parameter contract",
                    expected=expected,
                    actual=actual,
                    variadic=bool(target.get("variadic")),
                    location=node.get("location"),
                )
            )
            continue
        if "parameter_types" not in node:
            continue
        for index, (source, param) in enumerate(
            zip(node["parameter_types"], target["params"])
        ):
            dest = param["type"]
            argument = node["args"][index]
            try:
                bound = _address_argument(argument, source, dest)
            except ValueError as error:
                issues.append(
                    dict(
                        code="argument_referent_type",
                        symbol=node["callee"],
                        parameter=index,
                        source_type=source,
                        target_type=dest,
                        location=node.get("location"),
                        detail=str(error),
                    )
                )
                continue
            if bound is argument:
                continue
            node["args"][index] = bound
            node["parameter_types"][index] = dest
            changes.append(
                dict(
                    code="explicit_argument_binding",
                    symbol=node["callee"],
                    parameter=index,
                    source_type=source,
                    target_type=dest,
                    source_basis=node.get("parameter_type_basis", "callsite_signature"),
                    proof="the callsite and target bind the same referent type and address; adapt only passing and cv qualification",
                )
            )
    return changes, issues


def _state_contract(function, evidence):
    """Describe frame-owned scalars and all visible ways their identity can escape."""
    scalars, escaped, declared = {}, set(), set()

    def referenced_locals(value):
        return {
            node["name"]
            for node in _operations(value)
            if node["op"] == "var" and node.get("scope") != "global"
        }

    def storage_locals(value):
        """Separate addressed storage from values used to locate that storage."""
        if not isinstance(value, dict):
            return referenced_locals(value)
        if value.get("op") in {"index", "member"}:
            return storage_locals(value["base"])
        if value.get("op") == "unary" and value.get("operator") == "addr":
            return storage_locals(value["value"])
        # Unknown lvalue protocols retain the previous conservative boundary.
        return referenced_locals(value)

    def register(declaration, initialized):
        name, spec = declaration["name"], declaration["type"]
        if (
            name in declared
            or declaration.get("storage") not in (None, "auto")
            or declaration.get("cell")
        ):
            escaped.add(name)
        declared.add(name)
        while isinstance(spec, dict) and spec.get("kind") in {"const", "volatile"}:
            if spec["kind"] == "volatile":
                escaped.add(name)
            spec = spec["of"]
        if isinstance(spec, str) and spec not in {"void"}:
            scalars[name] = dict(type=spec, initialized=initialized)
        elif isinstance(spec, dict) and spec.get("kind") in {"ref", "rref"}:
            escaped.update(storage_locals(declaration.get("init")))

    for param in function["params"]:
        register(param, True)
    calls = {"call", "indirect_call", "library_construct", "library_call"}
    for node in _operations([function.get("initializers"), function["body"]]):
        op = node["op"]
        if op == "decl":
            register(node, "init" in node)
        if op in {"map", "each", "buffer", "runtime_array"}:
            escaped.add(node["index"] if op == "map" else node["name"])
        if op == "unary" and node["operator"] == "addr":
            escaped.update(storage_locals(node["value"]))
        if op in calls or op.startswith("object_"):
            args = node.get("args", ())
            passing = node.get("parameter_types")
            if passing is None and op == "indirect_call":
                passing = node.get("signature", {}).get("params")
            modes = node.get("parameter_modes")
            for index, arg in enumerate(args):
                formal = (
                    passing[index]
                    if passing is not None and index < len(passing)
                    else None
                )
                while isinstance(formal, dict) and formal.get("kind") in {
                    "const",
                    "volatile",
                }:
                    formal = formal["of"]
                mode = (
                    modes[index] if modes is not None and index < len(modes) else None
                )
                if mode == "reference" or (
                    mode is None
                    and (
                        formal is None
                        or isinstance(formal, dict)
                        and formal.get("kind") in {"ref", "rref"}
                    )
                ):
                    escaped.update(storage_locals(arg))
    for state in evidence.get("states", {}).values():
        if state.get("kind") != "global" and state.get("address_taken"):
            escaped.add(state["name"])
    return dict(
        private_scalars=sorted(scalars.keys() - escaped),
        scalars=scalars,
        escaped=sorted(escaped),
        evidence=evidence.get("graph_id"),
        basis="frame ownership, reference binding and visible argument passing",
    )


def _derive_artifacts(module, *, phase):
    """Derive evidence only after the contracts named by this input phase."""
    from .artifacts import analyze as analyze_artifacts
    from .buffers import analyze_buffers
    from scar.contract import validate_artifact_flow

    complete = phase == "canonical"
    basis = dict(
        phase=phase,
        complete=complete,
        evidence_scope=(
            "post_interface_and_state_contracts_pre_logical_rewrites"
            if phase != "protocol_prefix"
            else "qualified_closed_functions_and_protocol_prefixes_pre_interface_contracts"
        ),
        completed_passes=[
            "protocol_normalization",
            "numeric_contract_validation",
            "parameter_access",
            *(
                [
                    "object_interfaces",
                    "abi_declarations",
                    "call_arguments",
                    "mpi_effects",
                    "frame_state",
                ]
                if phase != "protocol_prefix"
                else []
            ),
        ],
    )
    analysis = analyze_artifacts(module)
    buffers = analyze_buffers(module, analysis["graph"])
    flow = validate_artifact_flow(buffers.pop("graph"))
    flow.setdefault("coverage", {})["derivation_basis"] = basis
    analysis.update(graph=flow, basis=basis, buffers={**buffers, "basis": basis})
    module["artifact_analysis"] = analysis
    if flow["nodes"]:
        module.setdefault("resource_model", {})["artifact_flow"] = flow


def normalize(bound):
    from .protocols import normalize_protocols
    from .mpi import qualify as qualify_mpi
    from scar.contract import validate_numeric_contract

    require_program(bound, "M1")
    module = deepcopy(bound)
    module, protocol_changes, issues = normalize_protocols(module)
    for function in module["functions"]:
        validate_numeric_contract(function)
        _parameter_access(function)
    if issues:
        # An open object protocol is not a native ABI. In particular, Python
        # keyword/default binding cannot be checked by counting C arguments.
        module["normalizations"] = protocol_changes
        module["baseline_defects"] = []
        _derive_artifacts(module, phase="protocol_prefix")
        return publish_program(module, "M2", complete=False), issues
    declarations = module.pop("declarations")
    environment = module.pop("abi_environment")
    functions = {f["name"]: f for f in module["functions"]}
    globals_ = {g["name"]: g for g in module["globals"]}
    object_changes = _object_member_interfaces(module, declarations, functions)
    externs = {e["name"]: e for e in module["externs"]}
    defects = []
    for name, variants in declarations.items():
        if len({json_bytes(_abi(v)) for v in variants}) < 2:
            continue
        witness = (
            _discarded_scalar_result(
                name,
                variants,
                functions[name],
                functions,
                globals_,
                environment.get("architecture"),
            )
            if name in functions
            else None
        )
        if witness:
            defects.append(witness)
        else:
            issues.append(
                dict(
                    code="incompatible_abi",
                    symbol=name,
                    detail="parameter or result types differ across declarations",
                )
            )
    module["baseline_defects"] = defects
    call_changes, call_issues = _bind_call_arguments(
        functions, externs, globals_, module["records"]
    )
    module["normalizations"] = protocol_changes + object_changes + call_changes
    issues.extend(call_issues)
    mpi_changes, mpi_issues = qualify_mpi(module)
    module["normalizations"].extend(mpi_changes)
    issues.extend(mpi_issues)
    for function in module["functions"]:
        evidence = module.get("state_evidence", {}).get(function["name"], {})
        function["state_contract"] = _state_contract(function, evidence)
    complete = is_complete(bound) and not issues
    _derive_artifacts(module, phase="canonical" if complete else "interface_prefix")
    return publish_program(module, "M2", complete=complete), issues

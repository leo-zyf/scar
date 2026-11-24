"""Normalize closed symbol bindings and call contracts without reading graph files."""

from copy import deepcopy
from scar.contract import (
    json_bytes,
    semantic_form as semantics,
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
            if not isinstance(source, dict) or not isinstance(dest, dict):
                continue
            if source.get("kind") == "ref" and dest.get("kind") == "ptr":
                node["args"][index] = dict(
                    op="unary", operator="addr", value=node["args"][index]
                )
            elif source.get("kind") == "ptr" and dest.get("kind") == "ref":
                node["args"][index] = dict(
                    op="unary", operator="deref", value=node["args"][index]
                )
            else:
                continue
            changes.append(
                dict(
                    code="explicit_argument_binding",
                    symbol=node["callee"],
                    parameter=index,
                    source_type=source,
                    target_type=dest,
                    proof="the callsite and definition share an ABI address; preserve the referred storage",
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
            escaped.update(referenced_locals(declaration.get("init")))

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
            escaped.update(referenced_locals(node["value"]))
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
                    escaped.update(referenced_locals(arg))
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


def normalize(bound):
    from .protocols import normalize_protocols
    from .mpi import qualify as qualify_mpi

    module = deepcopy(bound)
    module, protocol_changes, issues = normalize_protocols(module)
    if issues:
        # An open object protocol is not a native ABI. In particular, Python
        # keyword/default binding cannot be checked by counting C arguments.
        module["normalizations"] = protocol_changes
        module["baseline_defects"] = []
        return module, issues
    declarations = module.pop("declarations")
    environment = module.pop("abi_environment")
    functions = {f["name"]: f for f in module["functions"]}
    globals_ = {g["name"]: g for g in module["globals"]}
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
    module["normalizations"] = protocol_changes + call_changes
    issues.extend(call_issues)
    mpi_changes, mpi_issues = qualify_mpi(module)
    module["normalizations"].extend(mpi_changes)
    issues.extend(mpi_issues)
    for function in module["functions"]:
        evidence = module.get("state_evidence", {}).get(function["name"], {})
        function["state_contract"] = _state_contract(function, evidence)
    return module, issues

"""Versioned data contracts between the four independent middle-end passes.

Passes own their outputs; consumers must not modify a received product. These
checks validate the handoff shape, not the legality of an optimization. The
producer still supplies the semantic proofs and explicit unresolved findings.
"""

from scar.contract import document

_SCHEMA = "scar.middle-program"
_PHASES = {"M1", "M2", "M3", "M4"}


def require_program(module, phase):
    """Reject a missing, misordered or structurally invalid pass input."""
    if phase not in _PHASES or not isinstance(module, dict):
        raise ValueError("expected a middle-end program product")
    header = module.get("layer_contract")
    if (
        not isinstance(header, dict)
        or set(header) != {"schema", "schema_version", "phase", "completion"}
        or header["schema"] != _SCHEMA
        or type(header["schema_version"]) is not int
        or header["schema_version"] != 1
        or header["phase"] != phase
        or header["completion"] not in {"complete", "partial"}
    ):
        raise ValueError(f"expected {_SCHEMA} v1 {phase} product")
    for field in ("functions", "globals", "externs", "records"):
        records = module.get(field)
        if not isinstance(records, list) or any(
            not isinstance(record, dict) for record in records
        ):
            raise ValueError(f"{phase} requires a {field} record table")
    if not isinstance(module.get("entry"), str) or not module["entry"]:
        raise ValueError(f"{phase} requires a program entry symbol")
    names = set()
    for function in module["functions"]:
        name = function.get("name")
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(function.get("params"), list)
            or not isinstance(function.get("body"), list)
            or "result" not in function
        ):
            raise ValueError(f"{phase} has a malformed or repeated function record")
        names.add(name)
    if header["completion"] == "complete" and module["entry"] not in names:
        raise ValueError(f"{phase} has no function for its entry symbol")
    if phase == "M1" and (
        not isinstance(module.get("declarations"), dict)
        or not isinstance(module.get("abi_environment"), dict)
    ):
        raise ValueError("M1 requires declarations and the captured ABI environment")
    if phase != "M1" and header["completion"] == "complete":
        if "declarations" in module or "abi_environment" in module:
            raise ValueError(f"{phase} still contains unnormalized ABI bindings")
        if any(
            fn.get("execution_kind") == "python_protocol" for fn in module["functions"]
        ):
            raise ValueError(f"{phase} cannot claim closed source protocols")
    return module


def is_complete(module):
    return module["layer_contract"]["completion"] == "complete"


def publish_program(module, phase, *, complete):
    """Publish explicit phase metadata without hashing or copying the whole graph."""
    if type(complete) is not bool:
        raise ValueError("program completion must be explicit")
    result = dict(
        module,
        layer_contract=document(
            _SCHEMA,
            phase=phase,
            completion="complete" if complete else "partial",
        ),
    )
    return require_program(result, phase)

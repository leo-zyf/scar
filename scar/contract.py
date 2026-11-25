"""Versioned, relocatable stage products and their shared wire primitives."""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

GRAPH = "scar.program-graph"
PLAN = "scar.logical-plan"
IR = "scar.execution-ir"
PHYSICAL = "scar.physical-plan"
ARTIFACT_FLOW = "scar.artifact-flow"
RESOURCE_FLOW = "scar.resource-flow"
WORKFLOW = "scar.workflow-graph"
VERSION = 1


def validate_python_entry_declaration(declaration):
    """Validate a requested launch mode, without granting its startup premises."""
    if (
        not isinstance(declaration, dict)
        or set(declaration) - {"kind", "interpreter", "flags"}
        or not isinstance(declaration.get("kind"), str)
        or declaration.get("kind")
        not in {"standalone_script", "embedded", "unspecified"}
    ):
        raise ValueError("invalid Python entry environment declaration")
    if "interpreter" in declaration and (
        not isinstance(declaration["interpreter"], str)
        or not Path(declaration["interpreter"]).is_absolute()
        or "\0" in declaration["interpreter"]
    ):
        raise ValueError("Python entry interpreter needs an absolute path")
    flags = declaration.get("flags", [])
    if not isinstance(flags, list) or any(
        not isinstance(flag, str) or not flag or "\0" in flag for flag in flags
    ):
        raise ValueError("Python entry flags must be argument strings")


def validate_python_entry_environment(record):
    """Check observed compiler startup facts without treating them as app state."""
    if (
        not isinstance(record, dict)
        or type(record.get("semantics_version")) is not int
        or record["semantics_version"] != 1
        or not isinstance(record.get("entries"), list)
        or not record["entries"]
        or not isinstance(record.get("analysis"), dict)
    ):
        raise ValueError("invalid Python entry environment facts")
    identities = set()
    for entry in record["entries"]:
        if not isinstance(entry, dict) or set(entry) != {"image", "declaration"}:
            raise ValueError("invalid Python entry environment binding")
        identity = entry["image"]
        if identity is not None and (not isinstance(identity, str) or not identity):
            raise ValueError("invalid Python entry image identity")
        if identity in identities:
            raise ValueError("duplicate Python entry environment binding")
        identities.add(identity)
        validate_python_entry_declaration(entry["declaration"])
    analysis = record["analysis"]
    if (
        analysis.get("scope") != "analysis_process_snapshot"
        or analysis.get("confidence") != "observed"
        or analysis.get("audit_hooks") != "unknown"
        or analysis.get("deployment_equivalence") != "unproven"
        or analysis.get("application_initial_cache") != "unproven"
    ):
        raise ValueError("Python startup observation cannot grant application premises")
    interpreter = analysis.get("interpreter")
    if not isinstance(interpreter, dict) or any(
        not isinstance(interpreter.get(field), str) or not interpreter[field]
        for field in (
            "executable",
            "implementation",
            "version",
            "prefix",
            "base_prefix",
        )
    ):
        raise ValueError("Python startup needs its observed interpreter identity")
    image = interpreter.get("image")
    if (
        not Path(interpreter["executable"]).is_absolute()
        or not isinstance(image, dict)
        or not isinstance(image.get("path"), str)
        or not Path(image["path"]).is_absolute()
        or not isinstance(image.get("sha256"), str)
        or re.fullmatch("[0-9a-f]{64}", image["sha256"]) is None
        or type(image.get("size")) is not int
        or image["size"] < 1
    ):
        raise ValueError("Python startup interpreter image receipt is malformed")
    for field in ("search_path", "meta_path"):
        if not isinstance(analysis.get(field), list) or any(
            not isinstance(item, str) for item in analysis[field]
        ):
            raise ValueError(f"invalid observed Python {field}")
    flags = analysis.get("flags")
    if not isinstance(flags, dict) or any(
        not isinstance(key, str) or type(value) not in {bool, int, type(None)}
        for key, value in flags.items()
    ):
        raise ValueError("invalid observed Python process flags")
    dispatch = analysis.get("import_callable")
    if (
        not isinstance(dispatch, dict)
        or not isinstance(dispatch.get("kind"), str)
        or dispatch.get("kind") not in {"builtin_import", "other", "missing"}
        or (
            dispatch.get("type") is not None
            if dispatch.get("kind") == "missing"
            else not isinstance(dispatch.get("type"), str) or not dispatch["type"]
        )
    ):
        raise ValueError("invalid observed Python import callable")
    cache = analysis.get("selected_module_cache")
    if not isinstance(cache, dict) or any(
        not isinstance(name, str)
        or not name
        or not isinstance(state, str)
        or state not in {"present_module", "present_none", "present_other", "absent"}
        for name, state in cache.items()
    ):
        raise ValueError("invalid selected Python module-cache observations")


def validate_numeric_contract(function):
    """Validate portable permissions against sealed source policy evidence."""
    contract = function.get("numeric_contract")
    if contract is None:
        return None
    if not isinstance(contract, dict):
        raise ValueError("invalid numerical function contract")
    source = function.get("source_numeric_contract")
    permissions = contract.get("permissions")
    options = {
        "may_omit_math_errno": "math_errno",
        "may_assume_default_rounding": "rounding_math",
    }
    if (
        type(contract.get("semantics_version")) is not int
        or contract["semantics_version"] != 1
        or contract.get("scope") != "function"
        or not isinstance(source, dict)
        or type(source.get("semantics_version")) is not int
        or source["semantics_version"] != 1
        or source.get("scope") != "compilation_unit"
        or not isinstance(source.get("policy"), dict)
        or contract.get("source_contract_sha256") != digest(json_bytes(source))
        or not isinstance(permissions, dict)
        or permissions.keys() != options.keys()
        or any(type(value) is not bool for value in permissions.values())
    ):
        raise ValueError("invalid or detached numerical function contract")
    policy = source.get("policy", {})
    if any(
        allowed and policy.get(options[key]) is not False
        for key, allowed in permissions.items()
    ):
        raise ValueError("numerical permission exceeds captured source policy")
    return contract


def validate_source_literal(candidate):
    """Check a source fact's type and lineage, without approving its runtime binding."""
    literal = candidate.get("literal")
    if candidate.get("kind") != "literal" or not isinstance(literal, dict):
        raise ValueError("expected a typed source literal candidate")
    types = {
        "unit": ("builtins.NoneType", type(None)),
        "bool": ("builtins.bool", bool),
        "i64": ("builtins.int", int),
        "f64": ("builtins.float", float),
        "text": ("builtins.str", str),
        "python_int": ("builtins.int", int),
        "python_float": ("builtins.float", float),
    }
    kind = literal.get("type")
    if kind not in types or candidate.get("source_type") != types[kind][0]:
        raise ValueError("source literal and exact Python type disagree")
    if "value" in literal:
        value = literal["value"]
        if type(value) is not types[kind][1]:
            raise ValueError("source literal value has the wrong type")
    elif kind in {"python_int", "python_float"} and isinstance(
        literal.get("value_hex"), str
    ):
        try:
            value = (
                int(literal["value_hex"], 16)
                if kind == "python_int"
                else float.fromhex(literal["value_hex"])
            )
        except ValueError as error:
            raise ValueError("invalid source numeric literal encoding") from error
    else:
        raise ValueError("source literal lacks a lossless value")
    if (
        kind == "i64"
        and not -(1 << 63) <= value < 1 << 63
        or kind == "f64"
        and not math.isfinite(value)
        or kind == "python_int"
        and (
            -(1 << 63) <= value < 1 << 63
            or candidate.get("representation") != "requires_integer_range_contract"
        )
        or kind == "python_float"
        and (
            math.isfinite(value)
            or candidate.get("representation") != "requires_nonfinite_float_contract"
        )
    ):
        raise ValueError("source literal representation contradicts its range")
    for key in ("provider_id", "sha256"):
        if not isinstance(candidate.get(key), str) or not re.fullmatch(
            r"[0-9a-f]{64}", candidate[key]
        ):
            raise ValueError("source literal lacks a captured identity")
    if (
        any(
            type(candidate.get(key)) is not int or candidate[key] < 1
            for key in ("line", "end_line", "column", "end_column")
        )
        or candidate["end_line"] < candidate["line"]
    ):
        raise ValueError("source literal lacks a valid source span")
    route = candidate.get("via")
    if (
        not isinstance(route, list)
        or not route
        or not isinstance(route[-1], dict)
        or route[-1].get("kind") != "literal"
    ):
        raise ValueError("source literal lacks its producer lineage")
    leaf = route[-1]
    if any(
        leaf.get(key) != candidate.get(key)
        for key in ("module", "name", "sha256", "member", "provider_id", "line")
    ):
        raise ValueError("source literal and its producer lineage disagree")
    return literal


def python_provider_id(record):
    """Name resolved source identity independently of requested exports."""
    if record.get("status") not in {"verified_distribution", "verified_stdlib"}:
        return None
    fields = (
        "module",
        "status",
        "origin",
        "sha256",
        "hash_basis",
        "distribution",
        "version",
        "member",
        "loader",
        "interpreter",
        "platform",
        "canonical_module",
    )
    return digest(json_bytes({key: record[key] for key in fields if key in record}))


@dataclass(frozen=True)
class Resource:
    id: str
    kind: str
    capacity: int | None
    attributes: dict


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: str
    value: str | None


def document(schema, **fields):
    return dict(schema=schema, schema_version=VERSION, **fields)


_WIRE_TAG = "$scar.wire.v1"
_SURROGATE = re.compile(r"[\ud800-\udfff]")


def json_value(value):
    """Escape surrogate code points and reserved mapping keys into ordinary JSON."""
    if isinstance(value, str) and _SURROGATE.search(value):
        return {_WIRE_TAG: ["text", value.encode("utf-8", "surrogatepass").hex()]}
    if isinstance(value, dict):
        if _WIRE_TAG in value or any(
            isinstance(key, str) and _SURROGATE.search(key) for key in value
        ):
            pairs = [[json_value(key), json_value(value[key])] for key in sorted(value)]
            return {_WIRE_TAG: ["map", pairs]}
        result = value
        for key, child in value.items():
            encoded = json_value(child)
            if encoded is not child:
                if result is value:
                    result = value.copy()
                result[key] = encoded
        return result
    if isinstance(value, (list, tuple)):
        result = value
        for index, child in enumerate(value):
            encoded = json_value(child)
            if encoded is not child:
                if result is value:
                    result = list(value)
                result[index] = encoded
        return result
    return value


def _json_text(value):
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False
    )


def json_bytes(value):
    """Encode a canonical SCAR product while preserving each text code point."""
    text = _json_text(value)
    # ASCII JSON exposes every possible surrogate and every reserved key here.
    if "\\ud" in text or _WIRE_TAG in text:
        encoded = json_value(value)
        if encoded is not value:
            text = _json_text(encoded)
    return text.encode("ascii") + b"\n"


def _wire_object(pairs):
    fields = dict(pairs)
    if _WIRE_TAG not in fields:
        return fields
    payload = fields[_WIRE_TAG]
    if len(pairs) != 1 or not isinstance(payload, list) or len(payload) != 2:
        raise ValueError("invalid SCAR wire envelope")
    kind, content = payload
    if kind == "text" and isinstance(content, str):
        try:
            raw = bytes.fromhex(content)
            value = raw.decode("utf-8", "surrogatepass")
        except ValueError as error:
            raise ValueError("invalid SCAR wire text encoding") from error
        if raw.hex() != content or not _SURROGATE.search(value):
            raise ValueError("noncanonical SCAR wire text")
        return value
    if kind == "map" and isinstance(content, list):
        result = {}
        previous = None
        needs_escape = False
        for pair in content:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("invalid SCAR wire map entry")
            key, value = pair
            if not isinstance(key, str) or (previous is not None and key <= previous):
                raise ValueError("SCAR wire map keys must be unique and sorted")
            needs_escape = (
                needs_escape or key == _WIRE_TAG or bool(_SURROGATE.search(key))
            )
            result[key] = value
            previous = key
        if not needs_escape:
            raise ValueError("redundant SCAR wire map envelope")
        return result
    raise ValueError("invalid SCAR wire envelope payload")


def json_loads(data):
    """Decode ordinary JSON and validated SCAR text and mapping envelopes."""
    if isinstance(data, str):
        marked = _WIRE_TAG in data or "\\u00" in data
    else:
        # Escaped ASCII key characters and UTF-16/32 inputs need a decoded-key check.
        marked = b"$scar.wire.v1" in data or b"\\u00" in data or b"\x00" in data
    if marked:
        return json.loads(data, object_pairs_hook=_wire_object)
    return json.loads(data)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def byte_value_id(sha256):
    """Path-independent byte-content identity shared by stage products."""
    if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("byte values require a SHA-256 content receipt")
    return "bytes:sha256:" + sha256


def array_rank_domain(spec):
    """Validate an array wire type; None denotes an unbounded runtime rank."""
    if (
        not isinstance(spec, dict)
        or set(spec) != {"kind", "element", "rank"}
        or spec["kind"] != "ndarray"
        or not isinstance(spec["element"], str)
        or not spec["element"]
    ):
        raise ValueError("invalid array type record")
    rank = spec["rank"]
    if type(rank) is int and rank >= 0:
        return (rank,)
    if isinstance(rank, dict) and set(rank) == {"values"}:
        domain = rank["values"]
        if (
            isinstance(domain, list)
            and domain
            and all(type(n) is int and n >= 0 for n in domain)
            and domain == sorted(set(domain))
        ):
            return tuple(domain)
    if (
        isinstance(rank, dict)
        and set(rank) == {"minimum"}
        and type(rank["minimum"]) is int
        and rank["minimum"] == 0
    ):
        return None
    raise ValueError("invalid array rank domain")


def product_items(spec):
    """Validate an immutable managed product; members are values, not raw ABI refs."""
    scalars = {
        "unit",
        "text",
        "bool",
        "char",
        "i8",
        "u8",
        "i16",
        "u16",
        "i32",
        "u32",
        "i64",
        "u64",
        "f32",
        "f64",
        "f80",
        "c64",
        "c128",
    }

    def member(value):
        if isinstance(value, str):
            return value in scalars
        if not isinstance(value, dict):
            return False
        kind = value.get("kind")
        if kind in MAPPING_TYPES:
            mapping_value_type(value)
            return True
        if kind == "product_ref":
            product_items(value)
            return True
        if kind == "sequence_ref":
            sequence_element(value)
            return True
        if kind == "ndarray":
            array_rank_domain(value)
            return value["element"] in scalars - {"unit", "text"}
        if kind in {"handle", "object_ref"}:
            field = "resource" if kind == "handle" else "record"
            return (
                set(value) == {"kind", field}
                and isinstance(value[field], str)
                and bool(value[field])
            )
        return False

    if (
        not isinstance(spec, dict)
        or set(spec) != {"kind", "items"}
        or spec["kind"] != "product_ref"
        or not isinstance(spec["items"], list)
        or any(not member(item) for item in spec["items"])
    ):
        raise ValueError("product requires fixed ordered managed value types")
    return tuple(spec["items"])


MAPPING_PROFILE = "builtin-scalar-map-v1"
MAPPING_TYPES = frozenset(
    {"mapping_ref", "structured_value", "mapping_items_ref", "mapping_iterator_ref"}
)


def mapping_value_type(spec):
    """A tagged scalar/map value is distinct from an identity-bearing mapping.

    This profile has text keys and preserves unit, bool, i64, f64 and text
    alternatives alongside shared nested mappings. It grants no custom-key,
    cyclic-object or scalar-identity protocol.
    """
    if (
        not isinstance(spec, dict)
        or set(spec) != {"kind", "profile"}
        or not isinstance(spec.get("kind"), str)
        or spec["kind"] not in MAPPING_TYPES
        or spec.get("profile") != MAPPING_PROFILE
    ):
        raise ValueError("invalid structured mapping value type")
    return spec["kind"]


def sequence_element(spec):
    """Validate a mutable, homogeneous managed sequence with a concrete element."""
    if (
        not isinstance(spec, dict)
        or set(spec) != {"kind", "element"}
        or spec["kind"] != "sequence_ref"
    ):
        raise ValueError("sequence requires one concrete managed element type")
    product_items(dict(kind="product_ref", items=[spec["element"]]))
    return spec["element"]


def validate_access_contract(function):
    """Validate invocation-scoped, parameter-based exclusive modified bytes.

    This is not pairwise disjointness of pointer values or read-only storage.
    Read-only arguments may overlap. If bytes accessed through a root are also
    modified by any means, all accesses must be based on that root. The extra
    readonly_when_accessed clause prevents deriving C99 const-restrict's
    stronger promise merely from an ordinary pointer-to-const type.
    A loop-local independence flag cannot establish this stronger contract.
    """
    contract = function.get("access_contract")
    if contract is None:
        return ()
    if (
        not isinstance(contract, dict)
        or set(contract) != {"semantics_version", "scope", "parameters"}
        or type(contract["semantics_version"]) is not int
        or contract["semantics_version"] != 1
        or contract["scope"] != "invocation"
        or not isinstance(contract["parameters"], list)
        or not contract["parameters"]
    ):
        raise ValueError("invalid invocation access contract")
    params = {p["name"]: p for p in function["params"]}
    if len(params) != len(function["params"]):
        raise ValueError("access contract requires unique parameter bindings")
    roots = []
    for claim in contract["parameters"]:
        if (
            not isinstance(claim, dict)
            or set(claim)
            != {
                "parameter",
                "rule",
                "region",
                "paths",
                "basis",
                "readonly_when_accessed",
            }
            or not isinstance(claim["parameter"], str)
            or claim["parameter"] not in params
            or claim["parameter"] in roots
            or claim["rule"] != "exclusive_modified_bytes"
            or claim["region"] != "accessed_referent_bytes"
            or claim["paths"] != "based_on_parameter"
            or type(claim["readonly_when_accessed"]) is not bool
        ):
            raise ValueError("invalid parameter access claim")
        basis = claim["basis"]
        if (
            not isinstance(basis, dict)
            or set(basis) != {"kind", "rule_id", "evidence"}
            or basis["kind"] != "source_precondition"
            or not isinstance(basis["rule_id"], str)
            or not basis["rule_id"]
            or not isinstance(basis["evidence"], dict)
            or not basis["evidence"]
        ):
            raise ValueError("access claim requires its source precondition evidence")
        spec = params[claim["parameter"]]["type"]
        while isinstance(spec, dict) and spec.get("kind") in {"const", "volatile"}:
            spec = spec["of"]
        if not isinstance(spec, dict) or spec.get("kind") != "ptr":
            raise ValueError("parameter access claim requires a pointer root")
        referent = spec["to"]
        const_referent = False
        while isinstance(referent, dict) and referent.get("kind") in {
            "const",
            "volatile",
        }:
            const_referent |= referent["kind"] == "const"
            referent = referent["of"]
        if isinstance(referent, dict) and referent.get("kind") == "function":
            raise ValueError("parameter access claim requires object storage")
        if const_referent and not claim["readonly_when_accessed"]:
            raise ValueError(
                "const pointer access requires a readonly source precondition"
            )
        roots.append(claim["parameter"])
    if roots != sorted(roots):
        raise ValueError("parameter access claims must be sorted by root")
    return tuple(roots)


def validate_resource_flow(flow, resources=None):
    """Validate symbolic copies and materializations, never physical IO claims."""
    if (flow.get("schema"), flow.get("schema_version")) != (RESOURCE_FLOW, VERSION):
        raise ValueError("unsupported resource flow contract")
    resource_ids = flow.get("resources")
    if (
        not isinstance(resource_ids, list)
        or any(not isinstance(ref, str) or not ref for ref in resource_ids)
        or len(set(resource_ids)) != len(resource_ids)
    ):
        raise ValueError("resource flow requires unique inventory references")
    known_resources = set(resource_ids)
    resource_types = (
        {record["id"]: record["kind"] for record in resources}
        if resources is not None
        else None
    )
    if resource_types is not None and not known_resources <= resource_types.keys():
        raise ValueError("resource flow references an absent inventory resource")

    def port(value):
        return (
            isinstance(value, dict)
            and isinstance(value.get("owner"), str)
            and bool(value["owner"])
            and isinstance(value.get("path"), list)
            and all(
                isinstance(key, str) or type(key) is int and key >= 0
                for key in value["path"]
            )
        )

    nodes = flow.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("resource flow requires typed node records")
    records = {}
    for node in nodes:
        ref = node.get("id")
        if (
            not isinstance(ref, str)
            or not ref
            or ref in records
            or node.get("kind") not in {"COPY", "MATERIALIZATION"}
        ):
            raise ValueError("invalid or duplicate resource flow node")
        records[ref] = node
    copies = {ref: node for ref, node in records.items() if node["kind"] == "COPY"}
    materializations = {
        ref: node for ref, node in records.items() if node["kind"] == "MATERIALIZATION"
    }
    for ref, node in copies.items():
        dispatch = node.get("dispatch", {})
        if (
            not port(node)
            or node.get("source") not in materializations
            or node.get("target") not in materializations
            or node["source"] == node["target"]
            or node.get("blocking") is not True
            or node.get("value_relation")
            not in {"equal_bytes", "broadcast", "tree_copy"}
            or node.get("destination_storage")
            not in {"fresh", "existing", "may_create", "fresh_tree"}
            or not isinstance(node.get("opcode"), str)
            or not node["opcode"]
            or not isinstance(node.get("scope"), list)
            or not isinstance(node.get("effects"), list)
            or any(not isinstance(effect, str) for effect in node["effects"])
            or any(not isinstance(item, str) for item in node.get("conditions", []))
            or dispatch.get("kind") != "host_thread"
            or dispatch.get("namespace") != "source_inventory"
            or dispatch.get("binding") != "caller_thread"
            or not isinstance(dispatch.get("resource_candidates"), list)
            or not set(dispatch["resource_candidates"]) <= known_resources
        ):
            raise ValueError("invalid copy operation contract")
        if resource_types is not None and any(
            resource_types[resource] != "CPU_POOL"
            for resource in dispatch["resource_candidates"]
        ):
            raise ValueError("host dispatch must reference process CPU pools")
        count = node.get("byte_count", {})
        if count.get("kind") == "constant":
            valid_count = type(count.get("value")) is int and count["value"] >= 0
        else:
            valid_count = count.get("kind") in {
                "operand_nbytes",
                "operand_count",
                "result_nbytes",
                "tree_bytes",
            } and port(count)
        if not valid_count:
            raise ValueError("copy requires a logical byte count or symbolic extent")
        for frame in node["scope"]:
            if (
                not isinstance(frame, dict)
                or not isinstance(frame.get("kind"), str)
                or not port({"owner": node["owner"], "path": frame.get("path")})
            ):
                raise ValueError("copy requires structural control scope references")
    for ref, node in materializations.items():
        region, location = node.get("region", {}), node.get("location", {})
        if (
            not isinstance(node.get("owner"), str)
            or not node["owner"]
            or node.get("operand") is not None
            and not port(node["operand"])
            or node.get("domain") not in {"host_memory", "filesystem", "device_memory"}
            or region.get("dtype") is not None
            and not isinstance(region["dtype"], str)
        ):
            raise ValueError("materialization requires a typed region and operand")
        for key in ("shape", "strides_bytes"):
            value = region.get(key)
            if value is not None and (
                not isinstance(value, list)
                or any(
                    item is not None
                    and (type(item) is not int or key == "shape" and item < 0)
                    for item in value
                )
            ):
                raise ValueError("invalid materialization shape or byte strides")
        offset = region.get("offset_bytes")
        storage = region.get("storage", {})
        if (
            offset is not None
            and (type(offset) is not int or offset < 0)
            or storage.get("kind") not in {"operand_storage", "fresh_allocation"}
            or storage.get("operation") not in copies
            or location.get("kind") not in {"resource_domain", "filesystem_path"}
            or location.get("namespace") != "source_inventory"
            or location.get("binding")
            not in {"runtime_allocator", "execution_namespace"}
            or not isinstance(location.get("resource_candidates"), list)
            or not set(location["resource_candidates"]) <= known_resources
        ):
            raise ValueError("invalid materialization storage or location domain")
        if storage.get("origin") is not None and not port(storage["origin"]):
            raise ValueError("storage origin must reference a canonical value port")
        if resource_types is not None and any(
            resource_types[resource]
            not in {
                "host_memory": {"DRAM", "NUMA_MEMORY"},
                "device_memory": {"DEVICE_MEMORY", "HBM"},
                "filesystem": {"STORAGE_MOUNT"},
            }[node["domain"]]
            for resource in location["resource_candidates"]
        ):
            raise ValueError("materialization candidates disagree with storage domain")
        logical, ready = node.get("logical_value", {}), node.get("ready", {})
        if not (
            logical.get("kind") == "operand_version"
            and port(logical)
            or logical.get("kind") in {"same_as", "broadcast_of", "tree_copy_of"}
            and logical.get("materialization") in materializations
            and logical["materialization"] != ref
        ):
            raise ValueError("materialization requires its logical value provenance")
        if not (
            ready.get("kind") == "input_ready"
            and port(ready)
            or ready.get("kind") == "operation_complete"
            and ready.get("operation") in copies
        ):
            raise ValueError("materialization requires its readiness relation")
    edges = flow.get("edges")
    if not isinstance(edges, list):
        raise ValueError("resource flow requires relationship records")
    relationships = set()
    for edge in edges:
        source, target, kind = edge.get("source"), edge.get("target"), edge.get("kind")
        if not (
            kind == "READS"
            and source in materializations
            and target in copies
            or kind == "PRODUCES"
            and source in copies
            and target in materializations
            or kind == "MAY_RESIDE_ON"
            and source in materializations
            and target in known_resources
            or kind == "MAY_DISPATCH_ON"
            and source in copies
            and target in known_resources
        ):
            raise ValueError("invalid resource flow relationship")
        relationships.add((kind, source, target))
    for ref, node in copies.items():
        if (
            not {
                ("READS", node["source"], ref),
                ("PRODUCES", ref, node["target"]),
            }
            <= relationships
        ):
            raise ValueError("copy is missing its input/output data relationships")
        if any(
            ("MAY_DISPATCH_ON", ref, resource) not in relationships
            for resource in node["dispatch"]["resource_candidates"]
        ):
            raise ValueError("copy is missing its controller resource relationships")
        target = materializations[node["target"]]
        relation = {
            "equal_bytes": "same_as",
            "broadcast": "broadcast_of",
            "tree_copy": "tree_copy_of",
        }[node["value_relation"]]
        if target["logical_value"] != dict(
            kind=relation, materialization=node["source"]
        ) or target["ready"] != dict(kind="operation_complete", operation=ref):
            raise ValueError("copy result provenance or readiness disagrees")
    coverage = flow.get("coverage", {})
    if (
        coverage.get("kind") not in {"canonical", "partial"}
        or type(coverage.get("functions")) is not int
        or coverage["functions"] < 0
        or coverage.get("copies") != len(copies)
    ):
        raise ValueError("resource flow requires its scoped coverage")
    return flow


def validate_artifact_flow(flow):
    """Check typed artifact references without promoting snapshots to history."""
    if (flow.get("schema"), flow.get("schema_version")) != (ARTIFACT_FLOW, VERSION):
        raise ValueError("unsupported artifact flow contract")
    kinds = {
        "FILE_LOCATOR",
        "BYTE_VALUE",
        "FILE_VERSION",
        "FILE_VERSION_REQUIREMENT",
        "SOURCE_CAPTURE",
        "SOURCE_OPERATION",
        "BUILD_OPERATION",
        "EXECUTION_IMAGE",
        "FILE_VERSION_TEMPLATE",
        "CANONICAL_OPERATION",
        "BUFFER_LOCATOR",
        "BUFFER_VERSION",
        "BYTE_EXPRESSION",
        "COMMAND_REQUEST",
        "PROCESS_INTENT",
        "PROCESS_TEMPLATE",
        "DIRECTORY_TEMPLATE",
    }
    records = {}
    for node in flow["nodes"]:
        identity = node.get("id")
        if (
            not isinstance(identity, str)
            or not identity
            or identity in records
            or node.get("kind") not in kinds
        ):
            raise ValueError("invalid or duplicate artifact node")
        records[identity] = node
        if node["kind"] == "BYTE_VALUE" and identity != byte_value_id(node["sha256"]):
            raise ValueError("byte value identity disagrees with its receipt")
        if node["kind"] == "BYTE_VALUE" and "literal_text" in node:
            if (
                not isinstance(node["literal_text"], str)
                or digest(node["literal_text"].encode("utf-8")) != node["sha256"]
            ):
                raise ValueError(
                    "literal artifact text disagrees with its content identity"
                )
    version_kinds = {"FILE_VERSION", "FILE_VERSION_TEMPLATE"}

    def conditional_candidates(rows):
        if not isinstance(rows, list):
            raise ValueError("conditional file candidates require a list")
        seen = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "version",
                "locator",
                "invalidated_by",
                "requires",
            }:
                raise ValueError("invalid conditional file candidate record")
            version, locator = row["version"], row["locator"]
            if (
                not isinstance(version, str)
                or not isinstance(locator, str)
                or version in seen
                or records.get(version, {}).get("kind") not in version_kinds
                or records[version].get("locator") != locator
                or records.get(locator, {}).get("kind") != "FILE_LOCATOR"
            ):
                raise ValueError(
                    "conditional candidate requires its original file version and locator"
                )
            seen.add(version)
            barriers = row["invalidated_by"]
            if (
                not isinstance(barriers, list)
                or any(not isinstance(ref, str) for ref in barriers)
                or len(barriers) != len(set(barriers))
                or any(
                    records.get(ref, {}).get("kind") != "CANONICAL_OPERATION"
                    or records[ref].get("operation") != "file_state_barrier"
                    for ref in barriers
                )
                or not isinstance(row["requires"], list)
                or not row["requires"]
                or any(
                    not isinstance(requirement, str) or not requirement
                    for requirement in row["requires"]
                )
            ):
                raise ValueError(
                    "conditional candidate requires barriers and unresolved obligations"
                )

    for node in records.values():
        if "conditional_candidates" in node:
            conditional_candidates(node["conditional_candidates"])
        if (
            node["kind"] == "CANONICAL_OPERATION"
            and node.get("operation") == "file_state_barrier"
        ):
            if (
                type(node.get("filesystem_epoch")) is not int
                or node["filesystem_epoch"] < 0
                or any(
                    type(node.get(key)) is not bool
                    for key in ("changes_cwd", "changes_namespace")
                )
                or node.get("instance_scope") != "per_normal_dynamic_effect"
                or not isinstance(node.get("reason"), str)
                or not node["reason"]
                or (
                    node.get("origin") is not None
                    and records.get(node["origin"], {}).get("kind")
                    not in {"SOURCE_OPERATION", "CANONICAL_OPERATION"}
                )
            ):
                raise ValueError(
                    "file state barrier requires its ordered source effect"
                )
        if node["kind"] == "DIRECTORY_TEMPLATE":
            if node.get("form") not in {
                "entry",
                "filesystem_root",
                "created",
                "existing",
                "phi",
                "symbolic_parent",
            }:
                raise ValueError("directory template requires its identity form")
            for field, expected in (
                ("parent", "DIRECTORY_TEMPLATE"),
                ("locator", "FILE_LOCATOR"),
            ):
                if (
                    node.get(field) is not None
                    and records.get(node[field], {}).get("kind") != expected
                ):
                    raise ValueError(f"directory template requires its {field} node")
            if node.get("producer") is not None and records.get(
                node["producer"], {}
            ).get("kind") not in {"SOURCE_OPERATION", "CANONICAL_OPERATION"}:
                raise ValueError("directory template producer must be an operation")
            incoming = node.get("incoming_directories", [])
            if not isinstance(incoming, list) or any(
                records.get(ref, {}).get("kind") != "DIRECTORY_TEMPLATE"
                for ref in incoming
            ):
                raise ValueError("directory phi references an absent directory")
        if node["kind"] in {
            "FILE_VERSION",
            "FILE_VERSION_REQUIREMENT",
            "FILE_VERSION_TEMPLATE",
        }:
            if records.get(node.get("locator"), {}).get("kind") != "FILE_LOCATOR":
                raise ValueError("file version requires its locator node")
            value = node.get("value")
            if value is not None and records.get(value, {}).get("kind") != "BYTE_VALUE":
                raise ValueError("file version references an absent byte value")
            if node.get("form") == "phi":
                incoming = node.get("incoming_versions")
                if (
                    not isinstance(incoming, list)
                    or len(incoming) < 2
                    or any(
                        records.get(ref, {}).get("kind")
                        not in {"FILE_VERSION", "FILE_VERSION_TEMPLATE"}
                        for ref in incoming
                    )
                ):
                    raise ValueError("file phi requires its incoming version nodes")
        if node["kind"] == "BUFFER_VERSION":
            if records.get(node.get("locator"), {}).get("kind") != "BUFFER_LOCATOR":
                raise ValueError("buffer version requires its storage locator")
            if (
                node.get("expression") is not None
                and records.get(node["expression"], {}).get("kind") != "BYTE_EXPRESSION"
            ):
                raise ValueError("buffer version references an absent byte expression")
        if node["kind"] == "COMMAND_REQUEST":
            if (
                node.get("source_version") is not None
                and records.get(node["source_version"], {}).get("kind")
                != "BUFFER_VERSION"
            ):
                raise ValueError("command request references an absent buffer version")
            if (
                node.get("source_value") is not None
                and records.get(node["source_value"], {}).get("kind") != "BYTE_VALUE"
            ):
                raise ValueError("command request references an absent byte value")
        if node["kind"] == "PROCESS_TEMPLATE":
            if (
                records.get(node.get("producer"), {}).get("kind")
                not in {"SOURCE_OPERATION", "CANONICAL_OPERATION"}
                or node.get("instance_scope") != "per_successful_dynamic_launch"
                or node.get("descendant_effects") != "unresolved"
                or not isinstance(node.get("symbol"), str)
                or not isinstance(node.get("routes"), dict)
            ):
                raise ValueError(
                    "process template requires its logical launch contract"
                )
    for relation in flow["edges"]:
        if relation["source"] not in records or relation["target"] not in records:
            raise ValueError("artifact edge references an absent node")
        if not isinstance(relation.get("kind"), str) or not relation["kind"]:
            raise ValueError("artifact edge requires its relationship kind")
        kind = relation["kind"]
        if kind in {
            "MAY_SATISFY_FILE_VERSION",
            "CONDITIONALLY_AVAILABLE_BEFORE_LAUNCH",
            "CONDITIONAL_VERSION_AT_NORMAL_JOIN",
            "SUPERSEDES_CONDITIONAL_FILE_VERSION",
        }:
            source, target = (records[relation[key]] for key in ("source", "target"))
            expected = (
                {"FILE_VERSION_REQUIREMENT"}
                if kind == "MAY_SATISFY_FILE_VERSION"
                else (
                    version_kinds
                    if kind == "SUPERSEDES_CONDITIONAL_FILE_VERSION"
                    else {"SOURCE_OPERATION", "CANONICAL_OPERATION"}
                )
            )
            if (
                source["kind"] not in version_kinds
                or target["kind"] not in expected
                or relation.get("conditional") is not True
                or (
                    kind != "SUPERSEDES_CONDITIONAL_FILE_VERSION"
                    and relation.get("establishes_current_version") is not False
                )
                or (
                    kind == "CONDITIONALLY_AVAILABLE_BEFORE_LAUNCH"
                    and relation.get("establishes_child_read") is not False
                )
            ):
                raise ValueError(
                    "conditional file relation cannot establish a current version or a child read"
                )
            for ref in relation.get("invalidated_by", ()):
                if records.get(ref, {}).get("operation") != "file_state_barrier":
                    raise ValueError(
                        "conditional file relation references an absent barrier"
                    )
        if kind == "HAS_FILE_STATE_BARRIER":
            barrier = records[relation["target"]]
            if (
                barrier.get("operation") != "file_state_barrier"
                or barrier.get("origin") != relation["source"]
            ):
                raise ValueError(
                    "file state barrier is not connected to its actual source effect"
                )
        if relation["kind"] in {"COPIES_FILE_VERSION", "COPIES_REQUIRED_FILE_VERSION"}:
            source, target = (records[relation[key]] for key in ("source", "target"))
            required = relation["kind"] == "COPIES_REQUIRED_FILE_VERSION"
            if (
                source["kind"]
                not in (
                    {"FILE_VERSION_REQUIREMENT"}
                    if required
                    else {"FILE_VERSION", "FILE_VERSION_TEMPLATE"}
                )
                or target["kind"] != "FILE_VERSION_TEMPLATE"
                or target.get("source_requirement" if required else "source_version")
                != source["id"]
                or relation.get("conditional") is not True
                or (
                    "destination_case" in relation
                    and relation["destination_case"]
                    not in {
                        "resolved_target",
                        "destination_is_not_directory",
                        "destination_is_directory",
                    }
                )
            ):
                raise ValueError(
                    "copy relation requires its conditional source version"
                )
        if relation["kind"] == "DIRECTORY_PARENT" and (
            records[relation["source"]]["kind"] != "DIRECTORY_TEMPLATE"
            or records[relation["target"]]["kind"] != "DIRECTORY_TEMPLATE"
            or relation.get("conditional") is not True
        ):
            raise ValueError(
                "directory parent requires conditional directory identities"
            )
        if relation["kind"] in {
            "STARTS_PROCESS",
            "USES_PROCESS_HANDLE",
            "COMPLETES_DIRECT_PROCESS_IF_NORMAL",
        }:
            source, target = (records[relation[key]] for key in ("source", "target"))
            process = target if relation["kind"] == "STARTS_PROCESS" else source
            operation = source if relation["kind"] == "STARTS_PROCESS" else target
            if (
                process["kind"] != "PROCESS_TEMPLATE"
                or operation["kind"] not in {"SOURCE_OPERATION", "CANONICAL_OPERATION"}
                or (
                    relation["kind"] == "COMPLETES_DIRECT_PROCESS_IF_NORMAL"
                    and (
                        relation.get("conditional") is not True
                        or relation.get("descendants_completed") is not False
                    )
                )
            ):
                raise ValueError("process relationship requires its instance template")
        if relation["kind"] == "LOCATES_DIRECTORY" and (
            records[relation["source"]]["kind"] != "FILE_LOCATOR"
            or records[relation["target"]]["kind"] != "DIRECTORY_TEMPLATE"
            or relation.get("conditional") is not True
        ):
            raise ValueError(
                "directory location requires its conditional locator relation"
            )
        if (
            relation["kind"] == "CAPTURED_AT_SAME_LOCATOR"
            and relation.get("establishes_version_equality") is not False
        ):
            raise ValueError("capture snapshots do not prove a historical read version")
    for effect in flow.get("source_effects", ()):
        if records.get(effect["operation"], {}).get("kind") != "SOURCE_OPERATION":
            raise ValueError("artifact effect requires its source operation")
    for environment in flow.get("launch_environments", ()):
        if "conditional_candidates" in environment:
            conditional_candidates(environment["conditional_candidates"])
        if (
            records.get(environment.get("operation"), {}).get("kind")
            not in {"SOURCE_OPERATION", "CANONICAL_OPERATION"}
            or environment.get("target_binding") not in {"declared_image", "unbound"}
            or (
                environment.get("target_binding") == "declared_image"
                and (
                    not isinstance(environment.get("image_id"), str)
                    or not environment["image_id"]
                )
            )
            or environment.get("historical_instance") != "not_established"
            or (
                environment.get("directory") is not None
                and records.get(environment["directory"], {}).get("kind")
                != "DIRECTORY_TEMPLATE"
            )
            or any(
                records.get(version, {}).get("kind")
                not in {"FILE_VERSION", "FILE_VERSION_TEMPLATE"}
                for version in environment.get("visible_versions", ())
            )
        ):
            raise ValueError("launch environment requires conditional operation state")
        process = environment.get("process_template")
        if process is not None and (
            records.get(process, {}).get("kind") != "PROCESS_TEMPLATE"
            or records[process].get("launch_environment") != environment["id"]
        ):
            raise ValueError(
                "launch environment references a different process template"
            )
    processes = {}
    for process in flow.get("source_processes", ()):
        operation = records.get(process.get("operation"), {})
        if (
            operation.get("kind") != "SOURCE_OPERATION"
            or operation.get("operation") != process.get("operation_kind")
            or process.get("operation_kind")
            not in {
                "process_launch",
                "process_wait",
                "process_poll",
                "process_communicate",
                "process_stream",
            }
            or any(
                operation.get(key) != process.get(key)
                for key in ("unit", "graph_id", "port")
            )
            or process.get("execution") != "not_observed"
            or process["operation"] in processes
        ):
            raise ValueError(
                "process source record requires a unique matching operation"
            )
        processes[process["operation"]] = process

        def process_operand(value):
            if isinstance(value, list):
                for item in value:
                    process_operand(item)
            elif isinstance(value, dict) and set(value) == {"name", "value"}:
                if value["name"] is not None and not isinstance(value["name"], str):
                    raise ValueError(
                        "process keyword requires its source argument name"
                    )
                process_operand(value["value"])
            elif isinstance(value, str):
                reference = records.get(value, {})
                if reference.get("kind") != "SOURCE_OPERATION" or any(
                    reference.get(key) != operation.get(key)
                    for key in ("unit", "graph_id")
                ):
                    raise ValueError(
                        "process operand requires a source operation in its graph"
                    )
            else:
                raise ValueError("invalid process operand reference")

        for operand in process.get("operands", {}).values():
            process_operand(operand)
    for relation in flow["edges"]:
        if relation["kind"] == "PROCESS_HANDLE_TEMPLATE" and (
            processes.get(relation["source"], {}).get("operation_kind")
            != "process_launch"
            or relation["target"] not in processes
            or relation.get("conditional") is not True
            or relation.get("dynamic_instance_equality") != "unproven"
        ):
            raise ValueError(
                "source handle templates do not prove dynamic process identity"
            )
    for relation in flow["edges"]:
        source, target = records[relation["source"]], records[relation["target"]]
        if relation["kind"] == "OBSERVED_LAUNCH_INTENT" and (
            source.get("operation") != "process_launch"
            or source["kind"] != "SOURCE_OPERATION"
            or target["kind"] != "PROCESS_INTENT"
            or (target.get("caller") or {}).get("source_code_matches_snapshot")
            is not True
            or relation.get("source_sha256") != target["caller"].get("source_sha256")
        ):
            raise ValueError(
                "source launch attribution requires captured source identity"
            )
        if relation["kind"] == "OBSERVED_PROCESS_START" and (
            source["kind"] != "PROCESS_INTENT"
            or source.get("launch_outcome") != "started"
            or type(source.get("child_pid")) is not int
            or source["child_pid"] <= 0
            or target["kind"] != "BUILD_OPERATION"
            or target.get("ppid") != source.get("parent_pid")
            or target.get("trace_sha256") != source.get("trace_sha256")
        ):
            raise ValueError("observed process start requires an actual child PID")
        if relation["kind"] == "OBSERVED_PROCESS_START":
            mapping = relation.get("instrumentation_mapping")
            if mapping is None:
                if (
                    target["pid"] != source["child_pid"]
                    or relation.get("evidence")
                    != "observed_child_pid_and_process_identity"
                ):
                    raise ValueError("process PID equality is not established")
            else:
                tree = flow.get("process_observations", {}).get("process_tree", {})
                if (
                    relation.get("evidence")
                    != "observed_child_pid_and_kernel_wrapper_mapping"
                    or mapping.get("provider") != "strace-process-v1"
                    or mapping.get("sha256") != tree.get("sha256")
                    or not isinstance(mapping.get("sha256"), str)
                    or not re.fullmatch("[0-9a-f]{64}", mapping["sha256"])
                    or mapping.get("published_pid") != source["child_pid"]
                    or mapping.get("reported_pid") != target["pid"]
                    or mapping.get("wrapper_executable")
                    != target.get("instrumentation_reporter")
                ):
                    raise ValueError(
                        "instrumented PID mapping lacks its kernel evidence"
                    )
        if relation["kind"] == "INTENT_PARENT_INVOCATION" and (
            source["kind"] != "BUILD_OPERATION"
            or target["kind"] != "PROCESS_INTENT"
            or source.get("pid") != target.get("parent_pid")
            or source.get("trace_sha256") != target.get("trace_sha256")
        ):
            raise ValueError("process intent parent disagrees with its capture")
        if relation["kind"] == "REPORTED_PROCESS_PARENT" and (
            source["kind"] != "BUILD_OPERATION"
            or target["kind"] != "BUILD_OPERATION"
            or source.get("pid") != target.get("ppid")
            or source.get("trace_sha256") != target.get("trace_sha256")
        ):
            raise ValueError("observed process ancestry disagrees with its capture")
    for binding in flow.get("process_observations", {}).get("launch_bindings", ()):
        if records.get(binding["intent"], {}).get("kind") != "PROCESS_INTENT":
            raise ValueError("process observation requires its intent record")
        for field, kind in (
            ("source_operation", "SOURCE_OPERATION"),
            ("invocation", "BUILD_OPERATION"),
        ):
            reference = binding.get(field)
            if reference is not None and records.get(reference, {}).get("kind") != kind:
                raise ValueError("process observation references an absent endpoint")
        if binding.get("status") == "BOUND" and (
            binding.get("source_operation") is None or binding.get("invocation") is None
        ):
            raise ValueError("bound launch observation requires both endpoints")
    return flow


def validate_workflow(flow):
    """Check cross-image entry bindings without assuming file-version equality."""
    if (flow.get("schema"), flow.get("schema_version")) != (WORKFLOW, VERSION):
        raise ValueError("unsupported workflow graph contract")
    images = {image["id"] for image in flow["images"]}
    if len(images) != len(flow["images"]) or flow["root_image"] not in images:
        raise ValueError("workflow requires distinct declared images and its root")
    kinds = {"IMAGE", "ENTRY_STATE", "LAUNCH", "WAIT", "FILE_COPY", "FILE_READ"}
    nodes = {node["id"]: node for node in flow["nodes"]}
    if len(nodes) != len(flow["nodes"]) or any(
        node.get("kind") not in kinds or node.get("image_id") not in images
        for node in nodes.values()
    ):
        raise ValueError("workflow nodes require a declared image and explicit kind")
    relationships = {
        "EXPANDS_TO": ({"IMAGE"}, kinds - {"IMAGE"}),
        "LAUNCHES_IMAGE": ({"LAUNCH"}, {"IMAGE"}),
        "BINDS_ENTRY_STATE": ({"LAUNCH"}, {"ENTRY_STATE"}),
        "WAITS_FOR": ({"LAUNCH"}, {"WAIT"}),
    }
    for edge in flow["edges"]:
        types = relationships.get(edge.get("kind"))
        source, target = nodes.get(edge.get("source")), nodes.get(edge.get("target"))
        if (
            types is None
            or source is None
            or target is None
            or source["kind"] not in types[0]
            or target["kind"] not in types[1]
        ):
            raise ValueError("invalid workflow relationship")
        if edge["kind"] == "BINDS_ENTRY_STATE" and (
            edge.get("conditional") is not True
            or edge.get("binding_time") != "process_entry"
            or target.get("state")
            not in {"argv", "cwd", "environment", "filesystem_namespace"}
        ):
            raise ValueError("entry binding requires its launch-time conditions")
        if edge["kind"] == "WAITS_FOR" and (
            edge.get("conditional") is not True
            or edge.get("descendants_completed") is not False
        ):
            raise ValueError("a wait only closes the direct process")
    if flow.get("file_version_equivalence_established") is not False:
        raise ValueError("entry bindings do not prove child file-version equality")
    return flow


def validate_text_profiles(profiles):
    """Validate versioned Unicode classification data at a stage boundary."""
    if not isinstance(profiles, dict):
        raise ValueError("text profiles must be an identity-indexed table")
    for identity, profile in profiles.items():
        if not isinstance(profile, dict) or set(profile) != {
            "unicode_version",
            "interpreter",
            "nonprintable",
        }:
            raise ValueError("invalid Unicode profile fields")
        if not all(
            isinstance(profile[key], str) and profile[key]
            for key in ("unicode_version", "interpreter")
        ):
            raise ValueError("Unicode profile requires source version identities")
        ranges = profile["nonprintable"]
        if not isinstance(ranges, list):
            raise ValueError("Unicode ranges must be closed integer intervals")
        previous = -1
        for interval in ranges:
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError("invalid Unicode interval")
            low, high = interval
            if (
                type(low) is not int
                or type(high) is not int
                or not previous < low <= high <= 0x10FFFF
            ):
                raise ValueError("Unicode intervals must be ordered and disjoint")
            previous = high
        if identity != digest(json_bytes(profile)):
            raise ValueError("Unicode profile identity does not match its content")
    return profiles


def validate_state_flow(graph):
    """Check namespace/reference closure, without claiming effect completeness."""
    states, flow = graph["states"], graph["state_flow"]
    count = len(graph["nodes"])
    definitions = {(domain, "entry") for domain in states}
    references = []

    def port(value):
        if type(value) is not int or not 0 <= value < count:
            raise ValueError("state flow references an undefined operation port")

    def state(domain):
        if not isinstance(domain, str) or domain not in states:
            raise ValueError(f"state flow references an undefined state: {domain}")

    def token(domain, value):
        state(domain)
        if not isinstance(value, str) or not value:
            raise ValueError("state flow requires a nonempty epoch identity")
        return domain, value

    for record in flow["versions"]:
        port(record["node"])
        if record["kind"] not in {
            "ACCESS",
            "OPAQUE",
            "MERGE",
            "LOOP_ENTRY",
            "LOOP_CARRY",
        }:
            raise ValueError("unknown state flow record type")
        for domain in (*record.get("read", ()), *record.get("may_write", ())):
            state(domain)
        for domain, value in record.get("outgoing", {}).items():
            definitions.add(token(domain, value))
        for field in ("output", "header"):
            if field in record:
                definitions.add(token(record["domain"], record[field]))
        references.extend(token(d, v) for d, v in record.get("incoming", {}).items())
        references.extend(token(record["domain"], v) for v in record.get("inputs", ()))
        references.extend(
            token(record["domain"], record[k])
            for k in ("initial", "next", "exit")
            if record.get(k) is not None
        )
    if any(reference not in definitions for reference in references):
        raise ValueError("state flow references an undefined epoch in its state domain")
    for edge in flow.get("transfers", ()):
        port(edge["node"])
        if edge["target"] is not None:
            port(edge["target"])
    for field in ("unexpanded_regions", "unresolved_protocols", "unstructured_control"):
        for node in flow.get("coverage", {}).get(field, ()):
            port(node)
    for domain, definition in flow.get("introduced_states", {}).items():
        state(domain)
        if states[domain] != definition:
            raise ValueError("analysis state differs from its published definition")
    return flow


def semantic_form(value):
    """Compare operation products independently of diagnostic annotations."""
    if isinstance(value, dict):
        return {
            k: semantic_form(v)
            for k, v in value.items()
            if k not in {"location", "site", "origin_region"}
        }
    if isinstance(value, list):
        return [semantic_form(v) for v in value]
    return value


_CONTROL_REGIONS = {
    "for": ("init", "body"),
    "if": ("then", "else"),
    "try": ("body", "handlers", "else", "finally"),
    **dict.fromkeys(
        ("block", "do_while", "each", "map", "resource_scope", "switch", "while"),
        ("body",),
    ),
}


def control_regions(statement):
    """Yield structured statement-list slots, never expression operands or alternatives."""
    for field in _CONTROL_REGIONS.get(statement["op"], ()):
        if field == "handlers":
            for handler in statement.get(field, ()):
                yield handler, "body", handler.get("binding")
        elif field in statement:
            yield statement, field, None


EXECUTION_ANNOTATIONS = frozenset(
    {
        "source_evidence",
        "source_provider",
        "source_runtime",
        "python_scope",
        "source_numeric_contract",
        "numeric_contract",
        "state_contract",
        "iteration_layout",
        "checks",
        "artifact_effect",
    }
)


def validate_source_coverage(record):
    """Check a bounded source-coverage claim, independently of its parser."""
    if (
        not isinstance(record, dict)
        or set(record)
        != {"semantics_version", "language", "scope", "complete", "records"}
        or type(record["semantics_version"]) is not int
        or record["semantics_version"] != 1
        or record["language"] not in {"python", "c", "c++", "fortran"}
        or not isinstance(record["scope"], str)
        or not record["scope"]
        or type(record["complete"]) is not bool
        or not isinstance(record["records"], list)
    ):
        raise ValueError("invalid source coverage contract")
    incomplete = False
    for item in record["records"]:
        if (
            not isinstance(item, dict)
            or not {"kind", "role", "owner", "location", "status"} <= item.keys()
            or not isinstance(item["kind"], str)
            or not item["kind"]
            or item["role"] not in {"program_unit", "specification"}
            or not isinstance(item["owner"], list)
            or any(not isinstance(name, str) or not name for name in item["owner"])
            or not isinstance(item["location"], dict)
            or not isinstance(item["location"].get("file"), str)
            or not item["location"]["file"]
            or (
                item["location"].get("line") is not None
                and (
                    type(item["location"]["line"]) is not int
                    or item["location"]["line"] < 1
                )
            )
            or item["status"] not in {"modeled", "transparent", "incomplete"}
        ):
            raise ValueError("invalid source coverage record")
        if item["status"] == "incomplete":
            incomplete = True
            issue = item.get("diagnostic")
            if (
                not isinstance(issue, dict)
                or not isinstance(issue.get("code"), str)
                or not issue["code"]
                or not isinstance(issue.get("detail"), str)
                or not issue["detail"]
            ):
                raise ValueError("unclosed source construct requires a diagnostic")
    if record["complete"] == incomplete:
        raise ValueError("source coverage completion contradicts its records")
    return record


def walk_records(value, *, exclude_fields=()):
    """Visit wire records, with explicit exclusions for non-executable evidence."""
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, (tuple, list)):
            pending.extend(item)
        elif isinstance(item, dict):
            yield item
            pending.extend(v for k, v in item.items() if k not in exclude_fields)


def walk_operations(value, *, exclude_fields=()):
    """Iterate neutral operation records without interpreting their semantics."""
    for item in walk_records(value, exclude_fields=exclude_fields):
        if "op" in item:
            yield item


def publish(path, data):
    """Publish a complete file; readers see either revision during a rerun."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as scratch:
        name = Path(scratch.name)
        try:
            scratch.write(data)
            scratch.flush()
            scratch.close()
            os.replace(name, path)
        finally:
            name.unlink(missing_ok=True)


def write_json(path, value):
    publish(path, json_bytes(value))


def member(folder, group, value):
    data = json_bytes(value)
    sha = digest(data)
    relative = f"{group}/{sha}.json"
    path = folder / relative
    if not path.exists():
        publish(path, data)
    return dict(path=relative, sha256=sha, size=len(data))


def read_blob(folder, receipt):
    """Read one sealed byte member without imposing a serialization format."""
    folder = Path(folder)
    relative = Path(receipt["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact member must be relative to its package")
    path = (folder / relative).resolve(strict=True)
    if not path.is_relative_to(folder.resolve()):
        raise ValueError("artifact member escapes its package")
    data = path.read_bytes()
    if len(data) != receipt["size"] or digest(data) != receipt["sha256"]:
        raise ValueError(f"artifact checksum mismatch for {relative}")
    return data


def read_member(folder, receipt):
    return json_loads(read_blob(folder, receipt))


def copy_member(source_folder, receipt, destination, group):
    """Publish a sealed JSON member in another package without decoding its data."""
    if (
        not isinstance(receipt, dict)
        or not {"path", "sha256", "size"} <= receipt.keys()
    ):
        raise ValueError("expected a sealed member receipt")
    group = Path(group)
    if group.is_absolute() or not group.parts or ".." in group.parts:
        raise ValueError("artifact group must stay within its destination package")
    data = read_blob(source_folder, receipt)
    sha = digest(data)
    destination = Path(destination).resolve()
    relative = group / (sha + ".json")
    path = (destination / relative).resolve()
    if not path.is_relative_to(destination):
        raise ValueError("artifact destination escapes its package")
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("existing sealed member has different bytes")
    else:
        publish(path, data)
    return dict(path=relative.as_posix(), sha256=sha, size=len(data))


def file_receipt(folder, path):
    """Reference an already published product inside a relocatable package."""
    folder, path = Path(folder).resolve(), Path(path).resolve(strict=True)
    if not path.is_relative_to(folder) or not path.is_file():
        raise ValueError("artifact product must be a file inside its package")
    data = path.read_bytes()
    return dict(
        path=path.relative_to(folder).as_posix(), sha256=digest(data), size=len(data)
    )


def validate_images(images, root_image, *, unit_count=None):
    """Validate image namespaces; compilation-unit indices belong to the graph."""
    if not isinstance(images, list) or not images:
        raise ValueError("expected a nonempty program image table")
    identifiers, outputs = set(), set()
    for image in images:
        if not isinstance(image, dict):
            raise ValueError("each program image must be a record")
        for key in ("id", "output"):
            if not isinstance(image.get(key), str) or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_-]*", image[key]
            ):
                raise ValueError(f"invalid program image {key}")
        if image["id"] in identifiers or image["output"] in outputs:
            raise ValueError("program image ids and executable names must be unique")
        if not isinstance(image.get("entry"), str) or not image["entry"]:
            raise ValueError("each program image needs an entry symbol")
        identifiers.add(image["id"])
        outputs.add(image["output"])
        if unit_count is not None:
            units = image.get("units")
            if (
                not isinstance(units, list)
                or not units
                or any(
                    type(index) is not int or not 0 <= index < unit_count
                    for index in units
                )
                or len(set(units)) != len(units)
            ):
                raise ValueError("invalid program image compilation-unit indices")
    if not isinstance(root_image, str) or root_image not in identifiers:
        raise ValueError("root_image must identify a declared program image")


def read_image_product(folder, image, key, filename, schema):
    """Open a child image product, checking its receipt and its layer contract."""
    receipt = image[key]
    product = read_member(Path(folder), receipt)
    if Path(receipt["path"]).name != filename or (
        product.get("schema"),
        product.get("schema_version"),
    ) != (schema, VERSION):
        raise ValueError(f"image {image['id']} must contain {schema} v{VERSION}")
    return Path(folder) / receipt["path"], receipt["sha256"], product


def open_product(path, filename, schema):
    path = Path(path)
    path = path / filename if path.is_dir() else path
    data = path.read_bytes()
    product = json_loads(data)
    if (product.get("schema"), product.get("schema_version")) != (schema, VERSION):
        raise ValueError(f"{path} must contain {schema} v{VERSION}")
    return path, digest(data), product

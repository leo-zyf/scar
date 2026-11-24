"""Versioned, relocatable stage products and their shared wire primitives."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

GRAPH = "scar.program-graph"
PLAN = "scar.logical-plan"
IR = "scar.execution-ir"
PHYSICAL = "scar.physical-plan"
ARTIFACT_FLOW = "scar.artifact-flow"
VERSION = 1


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


def json_bytes(value):
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def digest(data):
    return hashlib.sha256(data).hexdigest()


def byte_value_id(sha256):
    """Path-independent byte-content identity shared by stage products."""
    if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("byte values require a SHA-256 content receipt")
    return "bytes:sha256:" + sha256


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
    for node in records.values():
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
    for relation in flow["edges"]:
        if relation["source"] not in records or relation["target"] not in records:
            raise ValueError("artifact edge references an absent node")
        if not isinstance(relation.get("kind"), str) or not relation["kind"]:
            raise ValueError("artifact edge requires its relationship kind")
        if (
            relation["kind"] == "CAPTURED_AT_SAME_LOCATOR"
            and relation.get("establishes_version_equality") is not False
        ):
            raise ValueError("capture snapshots do not prove a historical read version")
    for effect in flow.get("source_effects", ()):
        if records.get(effect["operation"], {}).get("kind") != "SOURCE_OPERATION":
            raise ValueError("artifact effect requires its source operation")
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


def walk_operations(value):
    """Iterate neutral operation records without interpreting their semantics."""
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, dict):
            if "op" in item:
                yield item
            pending.extend(item.values())


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
    return json.loads(read_blob(folder, receipt))


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
    product = json.loads(data)
    if (product.get("schema"), product.get("schema_version")) != (schema, VERSION):
        raise ValueError(f"{path} must contain {schema} v{VERSION}")
    return path, digest(data), product

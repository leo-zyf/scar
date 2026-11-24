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
        "BUFFER_LOCATOR",
        "BUFFER_VERSION",
        "BYTE_EXPRESSION",
        "COMMAND_REQUEST",
        "PROCESS_INTENT",
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

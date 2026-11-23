"""Versioned, relocatable stage products and their shared wire primitives."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

GRAPH = "scar.program-graph"
PLAN = "scar.logical-plan"
IR = "scar.execution-ir"
PHYSICAL = "scar.physical-plan"
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


def read_member(folder, receipt):
    relative = Path(receipt["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact member must be relative to its package")
    path = (folder / relative).resolve(strict=True)
    if not path.is_relative_to(folder.resolve()):
        raise ValueError("artifact member escapes its package")
    data = path.read_bytes()
    if len(data) != receipt["size"] or digest(data) != receipt["sha256"]:
        raise ValueError(f"artifact checksum mismatch for {relative}")
    return json.loads(data)


def open_product(path, filename, schema):
    path = Path(path)
    path = path / filename if path.is_dir() else path
    data = path.read_bytes()
    product = json.loads(data)
    if (product.get("schema"), product.get("schema_version")) != (schema, VERSION):
        raise ValueError(f"{path} must contain {schema} v{VERSION}")
    return path, digest(data), product

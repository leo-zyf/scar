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


def semantic_form(value):
    """Compare operation products independently of diagnostic annotations."""
    if isinstance(value, dict):
        return {
            k: semantic_form(v)
            for k, v in value.items()
            if k not in {"location", "site"}
        }
    if isinstance(value, list):
        return [semantic_form(v) for v in value]
    return value


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

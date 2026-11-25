"""F0 resolves and locks declared inputs without consulting a language parser."""

from dataclasses import dataclass
import json
from pathlib import Path

from scar.contract import (
    digest,
    json_bytes,
    validate_images,
    validate_python_entry_declaration,
)


@dataclass(frozen=True)
class SourceRequest:
    manifest: Path
    root: Path
    entry: str
    profiles: tuple
    source_hashes: dict
    python_buffers: dict
    project_entry: Path | None
    entry_abi: dict | None
    observation: dict | None
    images: list[dict] | None = None
    root_image: str | None = None
    build_trace: dict | None = None
    python_entries: tuple = ()


def _capture_bytes(manifest, receipt, label):
    if not isinstance(receipt, dict) or not isinstance(receipt.get("path"), str):
        raise ValueError(label + " requires its captured file path")
    data = (manifest.parent / receipt["path"]).resolve(strict=True).read_bytes()
    if receipt.get("sha256", digest(data)) != digest(data):
        raise ValueError(label + " lock changed")
    return data


def with_entry_abi(profile, entry_abi):
    if (
        entry_abi
        and entry_abi.get("kind") == "binary_f64_arrays"
        and profile["language"].lower() == "python"
    ):
        return {**profile, "abi": {**profile.get("abi", {}), **entry_abi}}
    return profile


def _profiles(units, default_language):
    if not isinstance(units, list) or not units:
        raise ValueError("expected a nonempty list of explicit compilation profiles")
    for unit in units:
        if not isinstance(unit, dict) or not isinstance(unit.get("path"), str):
            raise ValueError("each compilation unit needs a path and language")
        profile = {"language": default_language, **unit}
        if not isinstance(profile["language"], str) or profile["language"].lower() in {
            "",
            "mixed",
        }:
            raise ValueError("each compilation unit needs its source language")
        yield profile


def _image_profiles(spec):
    """Lock explicit profiles and leave script import discovery to F1."""
    root_image, images = spec.get("root_image"), spec["images"]
    validate_images(images, root_image, unit_count=None)
    if any(k in spec for k in ("units", "source", "observe")):
        raise ValueError("declare units or a source script separately for each image")
    table, indices, membership = [], {}, []
    for image in images:
        if "observe" in image or "source_root" in image:
            raise ValueError(
                "images share source_root and cannot declare a per-image observation"
            )
        identity = {
            key: image[key]
            for key in ("id", "entry", "output", "entry_abi")
            if key in image
        }
        if "source" in image:
            script = image["source"]
            if (
                "units" in image
                or not isinstance(script, str)
                or Path(script).suffix.lower() != ".py"
                or image["entry"] != "__main__"
                or image.get("entry_abi")
            ):
                raise ValueError(
                    "a script image takes a .py source and the __main__ entry; "
                    "use explicit units for custom entry ABIs"
                )
            membership.append({**identity, "source": script})
            continue
        profiles = [
            with_entry_abi(p, image.get("entry_abi"))
            for p in _profiles(image.get("units"), spec.get("language"))
        ]
        python_profiles = [p for p in profiles if p["language"].lower() == "python"]
        module_paths = {
            p["abi"].get("module"): p["path"]
            for p in python_profiles
            if p.get("abi", {}).get("kind") == "python_module"
        }
        for p in python_profiles:
            abi = p.get("abi", {})
            if abi.get("kind") == "python_module":
                project = abi.get("project", {})
                if abi.get("module") not in project or any(
                    module_paths.get(module) != row.get("path")
                    for module, row in project.items()
                ):
                    raise ValueError(
                        "a Python script image must include each module declared in its project catalog"
                    )
        if (
            image["entry"] == "__main__"
            and len(python_profiles) == len(profiles)
            and not any(
                (
                    p.get("abi", {}).get("kind") == "python_module"
                    and p["abi"].get("main")
                )
                or "__main__" in p.get("abi", {}).get("exports", {}).values()
                for p in python_profiles
            )
        ):
            raise ValueError(
                "a Python script image requires explicit Python module profiles for its entry and imported project modules"
            )
        image_units = []
        for p in profiles:
            key = json_bytes(p)
            if key not in indices:
                indices[key] = len(table)
                table.append(p)
            image_units.append(indices[key])
        if len(image_units) != len(set(image_units)):
            raise ValueError("duplicate compilation profile within one image")
        membership.append({**identity, "units": image_units})
    root = next(i for i in membership if i["id"] == root_image)
    return tuple(table), membership, root


def _python_entries(spec, profiles, images):
    if images is None:
        candidates = [(None, spec, profiles)]
    else:
        if "entry_environment" in spec:
            raise ValueError("declare entry_environment on the individual Python image")
        candidates = [
            (image["id"], source, [profiles[i] for i in image.get("units", ())])
            for image, source in zip(images, spec["images"])
        ]
    entries = []
    for image, source, units in candidates:
        python = any(unit["language"].lower() == "python" for unit in units) or (
            "source" in source and Path(source["source"]).suffix.lower() == ".py"
        )
        if not python:
            if "entry_environment" in source:
                raise ValueError("entry_environment requires a Python source or unit")
            continue
        declaration = source.get("entry_environment", {"kind": "unspecified"})
        validate_python_entry_declaration(declaration)
        entries.append(dict(image=image, declaration=declaration))
    return tuple(entries)


def read_request(path: Path) -> SourceRequest:
    manifest = Path(path).resolve(strict=True)
    project_entry = manifest if manifest.suffix.lower() == ".py" else None
    if project_entry:
        root = manifest.parent
        spec = dict(
            entry="__main__", units=[dict(path=manifest.name, language="python")]
        )
    else:
        spec = json.loads(manifest.read_bytes())
        root = (manifest.parent / spec.get("source_root", ".")).resolve(strict=True)
        if "source" in spec and "units" not in spec and "images" not in spec:
            if (
                spec.get("entry", "__main__") == "__main__"
                and Path(spec["source"]).suffix.lower() == ".py"
                and not any(k in spec for k in ("abi", "observe", "entry_abi"))
            ):
                spec.setdefault("entry", "__main__")
                project_entry = root / spec["source"]
            spec["units"] = [
                dict(path=spec["source"], language="python", abi=spec.get("abi", {}))
            ]
    images, build_trace = None, None
    if "images" in spec:
        profiles, images, entry = _image_profiles(spec)
    elif "build_trace" in spec:
        profiles, entry = (), {"entry": spec.get("entry", "main")}
    else:
        profiles = tuple(_profiles(spec.get("units"), spec.get("language")))
        if len({json_bytes(p) for p in profiles}) != len(profiles):
            raise ValueError("expected distinct compilation profiles")
        entry = spec
    if "build_trace" in spec:
        if "root_image" in spec and images is None:
            raise ValueError("root_image requires an explicit image catalog")
        if any(
            key in spec
            for key in (
                "units",
                "source",
                "observe",
                "entry_abi",
            )
        ):
            raise ValueError(
                "declare controller units within images beside a build trace"
            )
        trace = spec["build_trace"]
        if not isinstance(trace, dict) or trace.get("provider") != "bear-3-intercept":
            raise ValueError("expected the bear-3-intercept build trace provider")
        raw = _capture_bytes(manifest, trace, "build trace")
        targets = spec.get("targets")
        if targets is not None and (
            not isinstance(targets, list)
            or not targets
            or not all(isinstance(target, str) and target for target in targets)
            or len(targets) != len(set(targets))
        ):
            raise ValueError("build targets must be distinct nonempty output paths")
        build_trace = dict(provider=trace["provider"], raw=raw, targets=targets)
        for field in ("process_observation", "process_tree"):
            if field in trace:
                if field == "process_tree" and (
                    not isinstance(trace[field], dict)
                    or trace[field].get("provider") != "strace-process-v1"
                ):
                    raise ValueError(
                        "expected the strace-process-v1 process tree provider"
                    )
                build_trace[field] = _capture_bytes(manifest, trace[field], field)
    locks = spec.get("source_lock", {}).get("files", {})
    python_names = {p["path"] for p in profiles if p["language"].lower() == "python"}
    scripts = [image["source"] for image in images or () if "source" in image]
    python_names.update(scripts)
    names = {}
    for name in dict.fromkeys([*locks, *(p["path"] for p in profiles), *scripts]):
        source = (root / name).resolve(strict=True)
        if Path(name).is_absolute() or not source.is_relative_to(root):
            raise ValueError(f"input must be inside source_root: {name}")
        names[name] = source.relative_to(root).as_posix()
    hashes, sizes, buffers = {}, {}, {}
    python_files = {names[n] for n in python_names}
    for name, relative in names.items():
        if relative not in hashes:
            raw = (root / relative).read_bytes()
            hashes[relative], sizes[relative] = digest(raw), len(raw)
            if relative in python_files or relative.endswith(".py"):
                buffers[relative] = raw
        if name in locks:
            lock = locks[name]
            if (
                hashes[relative] != lock["sha256"]
                or sizes[relative] != lock["size_bytes"]
            ):
                raise ValueError(f"source lock changed: {name}")
        if name in python_names:
            # A manifest may spell a unit through a local symlink.
            buffers[name] = buffers[relative]
    observation = (
        {"entry": spec["entry"], "source": spec["source"], "observe": spec["observe"]}
        if "observe" in spec
        else None
    )
    return SourceRequest(
        manifest,
        root,
        entry["entry"],
        profiles,
        hashes,
        buffers,
        project_entry,
        entry.get("entry_abi"),
        observation,
        images,
        spec.get("root_image"),
        build_trace,
        _python_entries(spec, profiles, images),
    )

"""F0 resolves and locks declared inputs without consulting a language parser."""

from dataclasses import dataclass
import json
from pathlib import Path

from scar.contract import digest, json_bytes


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
        if "source" in spec and "units" not in spec:
            spec["units"] = [
                dict(path=spec["source"], language="python", abi=spec.get("abi", {}))
            ]
    profiles = tuple({"language": spec.get("language"), **p} for p in spec["units"])
    if not profiles or len({json_bytes(p) for p in profiles}) != len(profiles):
        raise ValueError("expected distinct compilation profiles")
    if any(not p["language"] or p["language"] == "mixed" for p in profiles):
        raise ValueError("each compilation unit needs its source language")
    locks = spec.get("source_lock", {}).get("files", {})
    python_names = {p["path"] for p in profiles if p["language"].lower() == "python"}
    names = {}
    for name in dict.fromkeys([*locks, *(p["path"] for p in profiles)]):
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
        spec["entry"],
        profiles,
        hashes,
        buffers,
        project_entry,
        spec.get("entry_abi"),
        observation,
    )

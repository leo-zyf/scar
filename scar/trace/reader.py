"""Reader."""

from __future__ import annotations

import json
import gzip
from pathlib import Path
from scar.ir import Effect, Event, ExecutionGraph, Knowledge


# Open retained trace artifacts as streams, including lossless gzip archives.
def resolve(path: str | Path, *, default_name: str | None = None) -> Path:
    path = Path(path)
    if path.is_dir() and default_name is not None:
        path = path / default_name
    if not path.exists() and path.suffix != ".gz":
        archive = Path(str(path) + ".gz")
        if archive.is_file():
            return archive
    return path


def open_binary(path: str | Path):
    path = resolve(path)
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def open_text(path: str | Path):
    path = resolve(path)
    return (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz"
            else path.open("r", encoding="utf-8"))


# Read JSONL traces back into the IR.
def load(path: str | Path) -> ExecutionGraph:
    path = Path(path)
    graph = ExecutionGraph()
    events_file = resolve(path, default_name="events.jsonl")
    with open_text(events_file) as stream:
        for line in stream:
            if not line.strip():
                continue
            raw = json.loads(line)
            effect_raw = raw.pop("effect", {})
            for key in ("rng_effect", "may_raise", "external_effect", "ordering_effect"):
                value = effect_raw.get(key, "UNKNOWN")
                try:
                    effect_raw[key] = Knowledge(value)
                except ValueError:
                    effect_raw[key] = Knowledge.UNKNOWN
            raw.pop("index", None)
            graph.add(Event(effect=Effect(**effect_raw), **raw))
    return graph

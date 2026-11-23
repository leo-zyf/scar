"""Build a source-backed program graph from an explicitly selected input."""

import json
from pathlib import Path

from scar.contract import Graph, artifact, read_artifact, write_graph

from .hardware import snapshot


def build(source: Path, out: Path, *, jobs: int = 1) -> Graph:
    if type(jobs) is not int or jobs < 1:
        raise ValueError("frontend jobs must be a positive integer")
    source = Path(source).resolve(strict=True)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "graph.json").exists():
        raise FileExistsError(out / "graph.json")
    evidence_dir = out / "frontend"
    evidence_dir.mkdir()
    request = target = None
    if source.suffix != ".py":
        request = artifact(source)
        target = json.loads(read_artifact(request))
        if not isinstance(target, dict):
            raise ValueError("frontend request must be a JSON object")
    if target is None or target.get("schema") == "scar.python-entry":
        from .python import build_python

        target, regions, operations, edges, evidence = build_python(source, evidence_dir, target=target,
                                                                  manifest=request)
    else:
        from .native import build_native

        target, regions, operations, edges, evidence = build_native(target, request, evidence_dir, jobs=jobs)
    resources, resource_edges, hardware = snapshot(Path(target["source_root"]))
    graph = Graph(target, regions, operations, (*edges, *resource_edges), resources,
                  {**evidence, "hardware": hardware})
    write_graph(out / "graph.json", graph)
    return graph

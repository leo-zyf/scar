"""Build a source-backed program graph from an explicitly selected input."""

from dataclasses import asdict, replace
import json
from pathlib import Path

from scar.contract import Edge, Graph, artifact, read_artifact, verify_artifact, write_graph

from .hardware import snapshot


def _mixed_graph(target, request, out, jobs, bundle):
    from .clang import scalar_declarations
    from .native import build_native
    from .python import build_python

    python = [row for row in target["units"] if row.get("language") == "python"]
    names = {row["path"] for row in python}
    files = target["source_lock"]["files"]
    native = dict(target, units=[row for row in target["units"] if row.get("language") != "python"],
                  source_lock={"files": {path: entry for path, entry in files.items() if path not in names}})
    folder = out / "native"
    folder.mkdir()
    graph = build_native(native, request, folder, jobs=jobs, bundle=bundle)
    native, evidence = graph.target, graph.evidence
    regions, operations, edges = graph.regions, graph.operations, graph.edges
    region_kinds = {region.id: region.kind for region in regions}
    parents = {child: region.id for region in regions for child in region.children}
    native_functions = {}
    for op in operations:
        symbol = op.attributes.get("symbol")
        if symbol is not None:
            region = op.region
            while region_kinds[region] != "FUNCTION":
                region = parents[region]
            native_functions.setdefault(symbol, set()).add(region)

    symbols = {symbol for unit in python for role in ("exports", "imports")
               for symbol in unit["abi"].get(role, {}).values()}
    declarations = scalar_declarations(native, evidence["compiler_types"], symbols)
    exported, functions = {}, []
    regions, operations, edges = list(regions), list(operations), list(edges)
    for index, unit in enumerate(python):
        name = unit["path"]
        folder = out / f"python-{index}"
        folder.mkdir()
        spec = dict(schema_version=1, source_root=native["source_root"], source=name,
                    source_lock={"files": {name: files[name]}}, scalar_abi=unit["abi"])
        unit_graph = build_python(Path(request.path), folder, target=spec, declarations=declarations)
        regions.extend(unit_graph.regions)
        operations.extend(unit_graph.operations)
        edges.extend(unit_graph.edges)
        proof = unit_graph.evidence
        evidence["locked_sources"].update(proof["locked_sources"])
        evidence.setdefault("python_materials", {})[name] = proof["source_material"]
        for function in unit_graph.program.scalars:
            symbol = function["name"]
            if symbol in exported or symbol in native_functions:
                raise ValueError(f"multiple definitions for mixed-language symbol {symbol}")
            exported[symbol] = function["region"]
            functions.append(function)
        imports = unit["abi"].get("imports", {})
        for op in unit_graph.operations:
            name = op.attributes.get("callee_syntax")
            if op.kind == "CALL" and name in imports:
                symbol = imports[name]
                destinations = native_functions.get(symbol, ())
                if len(destinations) != 1:
                    raise ValueError(f"import {symbol} requires one native provider")
                edges.append(Edge(op.id, next(iter(destinations)), "CALL", symbol))

    for row in evidence["native_calls"]:
        if row["resolution"] == "EXTERNAL" and row["symbol"] in exported:
            row["target"] = exported[row["symbol"]]
            row["resolution"] = "MIXED_ABI_DEFINITION"
            edges.append(Edge(row["operation"], row["target"], "CALL", row["symbol"]))
    native.update(units=target["units"], source_lock=target["source_lock"])
    return replace(graph, regions=tuple(regions), operations=tuple(operations), edges=tuple(edges),
                   program=replace(graph.program, scalars=tuple(functions)))


def _observe(graph, out):
    from .observe import observe_python

    target = graph.target
    path = Path(target["source_root"]) / target["units"][0]["path"]
    observation, links = observe_python(path, target["entry"], target["observe"], out, graph.regions,
                                        graph.operations)
    source = graph.evidence["locked_sources"][target["units"][0]["path"]]
    verify_artifact(source)
    if graph.evidence["input"] != source:
        verify_artifact(graph.evidence["input"])
    coverage = dict(graph.evidence["coverage"], runtime="BOUNDED_CALLS")
    return replace(graph, edges=(*graph.edges, *links),
                   evidence=dict(graph.evidence, coverage=coverage, observation=asdict(observation)))


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

        graph = build_python(source, evidence_dir, target=target, manifest=request)
        if "observe" in graph.target:
            graph = _observe(graph, evidence_dir)
    elif any(unit.get("language") == "python" for unit in target.get("units", ())):
        graph = _mixed_graph(target, request, evidence_dir, jobs, out)
    else:
        from .native import build_native

        graph = build_native(target, request, evidence_dir, jobs=jobs, bundle=out)
    resources, resource_edges, hardware = snapshot(Path(graph.target["source_root"]))
    graph = replace(graph, edges=(*graph.edges, *resource_edges), resources=resources,
                    evidence=dict(graph.evidence, hardware=hardware))
    write_graph(out / "graph.json", graph)
    return graph

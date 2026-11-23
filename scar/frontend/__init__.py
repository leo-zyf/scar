"""Build a source-backed program graph from an explicitly selected input."""

import json
from pathlib import Path

from scar.contract import Edge, Graph, artifact, read_artifact, write_graph

from .hardware import snapshot


def _mixed_graph(target, request, out, jobs):
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
    native, regions, operations, edges, evidence = build_native(native, request, folder, jobs=jobs)
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
        _, rs, ops, links, proof = build_python(Path(request.path), folder, target=spec, declarations=declarations)
        regions.extend(rs)
        operations.extend(ops)
        edges.extend(links)
        evidence["locked_sources"].update(proof["locked_sources"])
        evidence.setdefault("python_materials", {})[name] = proof["source_material"]
        for function in proof["scalar_functions"]:
            symbol = function["name"]
            if symbol in exported or symbol in native_functions:
                raise ValueError(f"multiple definitions for mixed-language symbol {symbol}")
            exported[symbol] = function["region"]
            functions.append(function)
        imports = unit["abi"].get("imports", {})
        for op in ops:
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
    evidence["scalar_functions"] = functions
    native.update(units=target["units"], source_lock=target["source_lock"])
    return native, tuple(regions), tuple(operations), tuple(edges), evidence


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
    elif any(unit.get("language") == "python" for unit in target.get("units", ())):
        target, regions, operations, edges, evidence = _mixed_graph(target, request, evidence_dir, jobs)
    else:
        from .native import build_native

        target, regions, operations, edges, evidence = build_native(target, request, evidence_dir, jobs=jobs)
    resources, resource_edges, hardware = snapshot(Path(target["source_root"]))
    graph = Graph(target, regions, operations, (*edges, *resource_edges), resources,
                  {**evidence, "hardware": hardware})
    write_graph(out / "graph.json", graph)
    return graph

"""Build an indexed, source-independent program graph from compilation units."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path

from scar.contract import GRAPH, digest, document, json_bytes, member, write_json
from .graph import function_graph
from .hardware import snapshot


def _unit(job):
    root, profile, revision, out = job
    path = root / profile["path"]
    language = profile["language"].lower()
    flags = profile.get("flags", [])
    if language in {"c", "c++", "cpp", "cxx"}:
        from .c_family import translate
    elif language in {"fortran", "f77", "f90"}:
        from .fortran import translate
    elif language == "python":
        from .python import translate
    else:
        raise ValueError(f"unsupported source language {language}")
    try:
        if language == "python":
            unit = translate(path, flags, root=root, abi=profile.get("abi", {}))
        elif language in {"c", "c++", "cpp", "cxx"}:
            unit = translate(
                path, flags, root=root, language="c" if language == "c" else "c++"
            )
        else:
            unit = translate(path, flags, root=root)
    except ValueError as error:
        unit = dict(
            functions=[],
            diagnostics=[
                dict(
                    code="unit_analysis",
                    file=profile["path"],
                    line=None,
                    detail=str(error),
                )
            ],
        )
    functions = [function_graph(f, revision) for f in unit.get("functions", [])]
    initializers = function_graph(
        dict(
            name="@initializers",
            params=[],
            result="void",
            body=[{"op": "decl", **g} for g in unit.get("globals", [])],
        ),
        revision,
    )
    payload = {k: unit.get(k, []) for k in ("externs", "records", "templates", "enums")}
    for record in payload["records"]:
        fields = [f for f in record.get("fields", ()) if "init" in f]
        if fields:
            record["initializer_graph"] = function_graph(
                dict(
                    name="@field_initializers/" + record["name"],
                    result="void",
                    params=[
                        dict(
                            name="this",
                            type=dict(
                                kind="ptr", to=dict(kind="record", name=record["name"])
                            ),
                        )
                    ],
                    body=[
                        dict(
                            op="decl",
                            name=f["name"],
                            type=f["type"],
                            init=f.pop("init"),
                        )
                        for f in fields
                    ],
                ),
                revision,
            )
    payload["global_graph"] = initializers
    payload["functions"] = functions
    receipt = member(out, "units", payload)
    return dict(
        path=profile["path"],
        language=language,
        flags=flags,
        revision=revision,
        graph=receipt,
        initializer_calls=initializers["calls"],
        initializer_objects=initializers["objects"],
        record_initializers={
            r["name"]: r["initializer_graph"]["calls"]
            for r in payload["records"]
            if "initializer_graph" in r
        },
        functions=[
            dict(
                id=f["id"],
                name=f["signature"]["name"],
                defined=f["defined"],
                calls=f["calls"],
                objects=f["objects"],
                dispatch=f["dispatch"],
                nodes=len(f["nodes"]),
            )
            for f in functions
        ],
        diagnostics=unit.get("diagnostics", []),
    )


def _input(manifest):
    if manifest.suffix.lower() == ".py":
        from .python import project_units

        return manifest.parent, {
            "entry": "__main__",
            "units": project_units(manifest, manifest.parent),
        }
    spec = json.loads(manifest.read_bytes())
    root = (manifest.parent / spec.get("source_root", ".")).resolve(strict=True)
    if "source" in spec and "units" not in spec:
        spec["units"] = [
            dict(path=spec["source"], language="python", abi=spec.get("abi", {}))
        ]
    return root, spec


def build(manifest, out, *, jobs=1):
    manifest, out = Path(manifest).resolve(strict=True), Path(out).resolve()
    root, spec = _input(manifest)
    locks = spec.get("source_lock", {}).get("files", {})
    sources = {}
    for relative in dict.fromkeys([*locks, *(u["path"] for u in spec["units"])]):
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or Path(relative).is_absolute():
            raise ValueError(f"input must be inside source_root: {relative}")
        raw = path.read_bytes()
        sources[relative] = digest(raw)
        if relative in locks:
            lock = locks[relative]
            if sources[relative] != lock["sha256"] or len(raw) != lock["size_bytes"]:
                raise ValueError(f"source lock changed: {relative}")
    snapshot_id = digest(json_bytes(sources))
    profiles = [{"language": spec.get("language"), **p} for p in spec["units"]]
    if (
        not profiles
        or jobs < 1
        or len({json_bytes(p) for p in profiles}) != len(profiles)
    ):
        raise ValueError(
            "expected distinct compilation profiles and a positive worker count"
        )
    if any(not p["language"] or p["language"] == "mixed" for p in profiles):
        raise ValueError("each compilation unit needs its source language")
    evidence = None
    entry_abi = spec.get("entry_abi")
    if "observe" in spec:
        from .observation import array_abi, observe

        evidence = observe(manifest, root, spec)
        entry_abi = array_abi(spec["entry"], evidence)
    if entry_abi and entry_abi["kind"] == "binary_f64_arrays":
        for p in profiles:
            if p["language"] == "python":
                p["abi"] = {**p.get("abi", {}), **entry_abi}
    out.mkdir(parents=True, exist_ok=True)
    work = [(root, p, digest(json_bytes([snapshot_id, p])), out) for p in profiles]
    if jobs == 1 or len(work) == 1:
        units = list(map(_unit, work))
    else:
        with ProcessPoolExecutor(max_workers=min(jobs, len(work))) as pool:
            units = list(pool.map(_unit, work, chunksize=1))
    resources, topology, platform = snapshot(root)
    diagnostics = [issue for unit in units for issue in unit["diagnostics"]]
    counts = Counter(d["code"] for d in diagnostics)
    totals = dict(
        units=len(units),
        definitions=sum(f["defined"] for u in units for f in u["functions"]),
        operations=sum(f["nodes"] for u in units for f in u["functions"]),
        diagnostic_sites=len(diagnostics),
        resources=len(resources),
    )
    graph = document(
        GRAPH,
        entry=spec["entry"],
        entry_abi=entry_abi,
        observation=evidence,
        units=units,
        source_revision=snapshot_id,
        resources=[asdict(r) for r in resources],
        platform=platform,
        topology=[
            dict(source=e.source, target=e.target, kind=e.value) for e in topology
        ],
        contract={"arithmetic": "preserve", "externally_visible_effects": "preserve"},
        coverage=dict(totals=totals, diagnostic_categories=dict(counts.most_common())),
    )
    write_json(out / "graph.json", graph)
    return dict(graph=str(out / "graph.json"), **totals)

"""Build an indexed, source-independent program graph from compilation units."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from scar.contract import GRAPH, digest, document, json_bytes, member, write_json
from scar.host import cpu_availability


def _unit(job):
    from .graph import function_graph
    from .state_flow import state_flow

    root, profile, revision, out, source_hashes, parsed, source_bytes = job
    path = root / profile["path"]
    language = profile["language"].lower()
    flags = profile.get("flags", [])
    if language in {"c", "c++", "cpp", "cxx"}:
        from .languages.c_family import parse_unit, normalize_unit
    elif language in {"fortran", "f77", "f90"}:
        from .languages.fortran import parse_unit, normalize_unit
    elif language == "python":
        from .languages.python import parse_unit, normalize_unit
    else:
        raise ValueError(f"unsupported source language {language}")
    try:
        if language == "python" and parsed is None:
            parsed = parse_unit(
                path,
                flags,
                root=root,
                abi=profile.get("abi", {}),
                source_bytes=source_bytes,
                source_hashes=source_hashes,
            )
        elif language in {"c", "c++", "cpp", "cxx"}:
            parsed = parse_unit(
                path,
                flags,
                root=root,
                language="c" if language == "c" else "c++",
                source_hashes=source_hashes,
            )
        elif parsed is None:
            parsed = parse_unit(path, flags, root=root)
        unit = normalize_unit(parsed)
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
    captured = dict(getattr(parsed, "source_digests", {}))
    source_facts = dict(
        files=captured,
        compiler=getattr(parsed, "compiler_id", None),
        providers=getattr(parsed, "providers", {}),
    )
    revision = digest(json_bytes([revision, source_facts]))
    functions = [function_graph(f, revision) for f in unit.get("functions", [])]
    initializers = function_graph(
        dict(
            name="@initializers",
            params=[],
            result="void",
            body=[
                {"op": "decl", **g, "scope": "global"} for g in unit.get("globals", [])
            ],
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
    # F4 consumes the complete F3 graph without accessing language parsers.
    state_graphs = [initializers, *functions]
    state_graphs.extend(
        r["initializer_graph"] for r in payload["records"] if "initializer_graph" in r
    )
    for graph in state_graphs:
        graph["state_flow"] = state_flow(
            graph["nodes"], graph["entry"], graph["accesses"]
        )
    receipt = member(out, "units", payload)
    return dict(
        path=profile["path"],
        language=language,
        flags=flags,
        revision=revision,
        source_facts=source_facts,
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
                source_defined=f["source_defined"],
                calls=f["calls"],
                objects=f["objects"],
                dispatch=f["dispatch"],
                process_operations=f["process_operations"],
                nodes=len(f["nodes"]),
            )
            for f in functions
        ],
        diagnostics=unit.get("diagnostics", []),
    )


def build(manifest, out, *, jobs=1):
    from .hardware import snapshot
    from .input import read_request, with_entry_abi

    if jobs < 1:
        raise ValueError("expected a positive worker count")
    request, out = read_request(manifest), Path(out).resolve()
    root = request.root
    sources = dict(request.source_hashes)
    profiles = request.profiles
    if request.project_entry:
        from .languages.python import capture_project

        capture = capture_project(
            request.project_entry,
            root,
            buffers=request.python_buffers,
            source_hashes=request.source_hashes,
        )
        profiles, parsed_units = capture.profiles, capture.units
        sources.update(capture.source_digests)
    else:
        parsed_units = (None,) * len(profiles)
    snapshot_id = digest(json_bytes(sources))
    evidence = None
    entry_abi = request.entry_abi
    if request.observation is not None:
        from .observation import array_abi, observe

        evidence = observe(request.manifest, root, request.observation)
        entry_abi = array_abi(request.entry, evidence)
    if request.images is None and entry_abi:
        profiles = tuple(with_entry_abi(p, entry_abi) for p in profiles)
    out.mkdir(parents=True, exist_ok=True)
    work = [
        (
            root,
            p,
            digest(json_bytes([snapshot_id, p])),
            out,
            sources,
            parsed,
            request.python_buffers.get(p["path"]) if parsed is None else None,
        )
        for p, parsed in zip(profiles, parsed_units)
    ]
    workers = min(jobs, len(work), cpu_availability().worker_budget)
    if workers == 1:
        units = list(map(_unit, work))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            units = list(pool.map(_unit, work, chunksize=1))
    for unit in units:
        for name, revision in unit["source_facts"]["files"].items():
            if sources.setdefault(name, revision) != revision:
                raise ValueError(
                    f"compilation units observed different source revisions: {name}"
                )
    resources, topology, platform = snapshot(root)
    diagnostics = [issue for unit in units for issue in unit["diagnostics"]]
    counts = Counter(d["code"] for d in diagnostics)
    totals = dict(
        units=len(units),
        definitions=sum(f["source_defined"] for u in units for f in u["functions"]),
        qualified_definitions=sum(f["defined"] for u in units for f in u["functions"]),
        operations=sum(f["nodes"] for u in units for f in u["functions"]),
        diagnostic_sites=len(diagnostics),
        resources=len(resources),
    )
    graph = document(
        GRAPH,
        entry=request.entry,
        entry_abi=entry_abi,
        observation=evidence,
        units=units,
        source_revision=digest(json_bytes(sources)),
        resources=[asdict(r) for r in resources],
        platform=platform,
        topology=[
            dict(source=e.source, target=e.target, kind=e.value) for e in topology
        ],
        contract={"arithmetic": "preserve", "externally_visible_effects": "preserve"},
        coverage=dict(totals=totals, diagnostic_categories=dict(counts.most_common())),
        **(
            {"images": request.images, "root_image": request.root_image}
            if request.images is not None
            else {}
        ),
    )
    write_json(out / "graph.json", graph)
    return dict(graph=str(out / "graph.json"), workers=workers, **totals)

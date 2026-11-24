"""Build an indexed, source-independent program graph from compilation units."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from scar.contract import (
    GRAPH,
    digest,
    document,
    json_bytes,
    member,
    validate_images,
    validate_state_flow,
    write_json,
)
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
        runtime=getattr(parsed, "runtime_facts", {}),
    )
    if text_profiles := getattr(parsed, "text_profiles", None):
        source_facts["text_profiles"] = text_profiles
    if external_headers := getattr(parsed, "external_headers", None):
        source_facts["external_headers"] = dict(external_headers)
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
        for identity, state in graph["state_flow"]["introduced_states"].items():
            if graph["states"].setdefault(identity, state) != state:
                raise ValueError(
                    f"state analysis introduced a conflicting domain: {identity}"
                )
        validate_state_flow(graph)
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


def _capture_inputs(request):
    """Expand each script into its own module catalog and intern full unit profiles."""
    sources = dict(request.source_hashes)
    profiles = list(request.profiles)
    parsed_units = [None] * len(profiles)
    images = request.images
    if request.project_entry or any("source" in image for image in images or ()):
        from .languages.python.facts import ProjectCapture

        scripts = ProjectCapture(
            request.root,
            buffers=request.python_buffers,
            source_hashes=request.source_hashes,
        )
        if request.project_entry:
            project = scripts.capture(request.project_entry)
            sources.update(project.source_digests)
            return project.profiles, project.units, images, sources, None
        indices = {json_bytes(profile): i for i, profile in enumerate(profiles)}
        images = []
        for image in request.images:
            if "source" not in image:
                images.append(image)
                continue
            project = scripts.capture(image["source"])
            sources.update(project.source_digests)
            units = []
            for profile, parsed in zip(project.profiles, project.units):
                key = json_bytes(profile)
                if key not in indices:
                    indices[key] = len(profiles)
                    profiles.append(profile)
                    parsed_units.append(parsed)
                else:
                    parsed_units[indices[key]] = parsed
                units.append(indices[key])
            images.append(
                {k: v for k, v in image.items() if k != "source"} | {"units": units}
            )
    if images is not None:
        validate_images(images, request.root_image, unit_count=len(profiles))
    return profiles, parsed_units, images, sources, None


def _capture_build(request):
    """Refine a locked build trace into explicit translation-unit tasks."""
    from .builds.facts import capture_build
    from .builds.normalize import normalize_build

    trace = request.build_trace
    facts = capture_build(trace["raw"], root=request.root, provider=trace["provider"])
    capture = normalize_build(facts, targets=trace["targets"], entry=request.entry)
    sources = dict(request.source_hashes)
    for name, revision in capture["source_hashes"].items():
        if sources.setdefault(name, revision) != revision:
            raise ValueError(
                f"build capture observed a different source revision: {name}"
            )
    profiles, images = capture["profiles"], capture["images"]
    if images:
        validate_images(images, capture["root_image"], unit_count=len(profiles))
    return profiles, [None] * len(profiles), images or None, sources, capture


def build(manifest, out, *, jobs=1):
    from .hardware import snapshot
    from .input import read_request, with_entry_abi

    if jobs < 1:
        raise ValueError("expected a positive worker count")
    request, out = read_request(manifest), Path(out).resolve()
    root = request.root
    profiles, parsed_units, images, sources, captured_build = (
        _capture_build(request) if request.build_trace else _capture_inputs(request)
    )
    root_image = captured_build["root_image"] if captured_build else request.root_image
    snapshot_id = digest(json_bytes(sources))
    evidence = None
    entry_abi = request.entry_abi
    if request.observation is not None:
        from .observation import array_abi, observe

        evidence = observe(request.manifest, root, request.observation)
        entry_abi = array_abi(request.entry, evidence)
    if images is None and entry_abi:
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
    if workers <= 1:
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
    resource_records = [asdict(resource) for resource in resources]
    if captured_build:
        resource_records.extend(captured_build["software_resources"])
    build_graph = captured_build["build_graph"] if captured_build else None
    diagnostics = [issue for unit in units for issue in unit["diagnostics"]]
    if build_graph is not None:
        diagnostics.extend(build_graph["diagnostics"])
    counts = Counter(d["code"] for d in diagnostics)
    totals = dict(
        units=len(units),
        definitions=sum(f["source_defined"] for u in units for f in u["functions"]),
        qualified_definitions=sum(f["defined"] for u in units for f in u["functions"]),
        operations=sum(f["nodes"] for u in units for f in u["functions"]),
        diagnostic_sites=len(diagnostics),
        resources=len(resource_records),
    )
    graph = document(
        GRAPH,
        entry=request.entry,
        entry_abi=entry_abi,
        observation=evidence,
        units=units,
        source_revision=digest(json_bytes(sources)),
        resources=resource_records,
        platform=platform,
        topology=[
            dict(source=e.source, target=e.target, kind=e.value) for e in topology
        ],
        contract={"arithmetic": "preserve", "externally_visible_effects": "preserve"},
        coverage=dict(totals=totals, diagnostic_categories=dict(counts.most_common())),
        **({"images": images, "root_image": root_image} if images is not None else {}),
        **(
            {
                "build": dict(
                    graph=member(out, "build", build_graph),
                    provider=request.build_trace["provider"],
                    trace_sha256=digest(request.build_trace["raw"]),
                    diagnostics=build_graph["diagnostics"],
                )
            }
            if build_graph is not None
            else {}
        ),
    )
    write_json(out / "graph.json", graph)
    return dict(graph=str(out / "graph.json"), workers=workers, **totals)

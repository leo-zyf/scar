"""Compose F0–F5 products into an indexed, source-independent program graph."""

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
    validate_python_entry_environment,
    validate_state_flow,
    write_json,
)
from scar.host import cpu_availability


def _unit(job):
    from .artifacts.graph import unit_operations
    from .graph import unit_graph
    from .state_flow import state_flow

    index, root, profile, revision, out, source_hashes, parsed, source_bytes = job
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
    if source_origins := getattr(parsed, "source_origins", None):
        source_facts["source_origins"] = source_origins
    if numeric := getattr(parsed, "source_numeric_contract", None):
        source_facts["source_numeric_contract"] = numeric
    if text_profiles := getattr(parsed, "text_profiles", None):
        source_facts["text_profiles"] = text_profiles
    if external_headers := getattr(parsed, "external_headers", None):
        source_facts["external_headers"] = dict(external_headers)
    if source_compiler := profile.get("source_compiler"):
        source_facts["source_compiler"] = source_compiler
    identity_facts = source_facts
    numeric_semantics = getattr(parsed, "numeric_semantics_sha256", None)
    if numeric_semantics is not None:
        identity_facts = {**source_facts, "source_numeric_contract": numeric_semantics}
    revision = digest(json_bytes([revision, identity_facts]))
    payload = unit_graph(unit, revision, numeric_semantics=numeric_semantics)
    functions, initializers = payload["functions"], payload["global_graph"]
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
    unit_index = dict(
        path=profile["path"],
        language=language,
        flags=flags,
        compilation_profile_sha256=digest(json_bytes(profile)),
        revision=revision,
        source_facts=source_facts,
        **(
            {"numeric_semantics_sha256": numeric_semantics} if numeric_semantics else {}
        ),
        **(
            {"source_coverage": unit["source_coverage"]}
            if "source_coverage" in unit
            else {}
        ),
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
    return unit_index, unit_operations(index, payload)


def build(manifest, out, *, jobs=1):
    from .capture import capture_project
    from .artifacts.graph import join_artifacts
    from .hardware import snapshot
    from .input import read_request, with_entry_abi

    if jobs < 1:
        raise ValueError("expected a positive worker count")
    request, out = read_request(manifest), Path(out).resolve()
    root = request.root
    captured = capture_project(request)
    profiles, parsed_units = captured.profiles, captured.parsed_units
    images, sources = captured.images, dict(captured.source_hashes)
    captured_build, root_image = captured.build, captured.root_image
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
            index,
            root,
            p,
            digest(json_bytes([snapshot_id, p])),
            out,
            sources,
            parsed,
            request.python_buffers.get(p["path"]) if parsed is None else None,
        )
        for index, (p, parsed) in enumerate(zip(profiles, parsed_units, strict=True))
    ]
    workers = min(jobs, len(work), cpu_availability().worker_budget)
    if workers <= 1:
        results = list(map(_unit, work))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_unit, work, chunksize=1))
    units = [unit for unit, _ in results]
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
    source_facts = {}
    if captured.python_environment is not None:
        environment = captured.python_environment.for_providers(
            name
            for unit in units
            if unit["language"] == "python"
            for name in unit["source_facts"]["providers"]
        )
        validate_python_entry_environment(environment)
        source_facts["python_entry_environment"] = environment
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
        **({"source_facts": source_facts} if source_facts else {}),
        artifact_flow=member(
            out,
            "artifacts",
            join_artifacts(
                units,
                (projection for _, projection in results),
                build_graph,
                images,
                root,
            ),
        ),
        **({"images": images, "root_image": root_image} if images is not None else {}),
        **(
            {
                "build": dict(
                    graph=member(out, "build", build_graph),
                    provider=request.build_trace["provider"],
                    trace_sha256=digest(request.build_trace["raw"]),
                    diagnostics=build_graph["diagnostics"],
                    bindings=captured.build_bindings,
                )
            }
            if build_graph is not None
            else {}
        ),
    )
    write_json(out / "graph.json", graph)
    return dict(graph=str(out / "graph.json"), workers=workers, **totals)

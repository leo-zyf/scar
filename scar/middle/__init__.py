"""Publish logical rewrites and a closed execution IR using only the graph package."""

from collections import Counter
from pathlib import Path

from scar.contract import (
    GRAPH,
    IR,
    PLAN,
    document,
    file_receipt,
    open_product,
    member,
    validate_images,
    write_json,
)


def _plan_image(graph, folder, revision, out, *, shape, trip_count):
    from .binding import bind
    from .normalization import normalize
    from .rewrites import rewrite
    from .execution import prepare, seal

    abi = graph.get("entry_abi")
    if shape is not None or trip_count is not None:
        if not abi or abi["kind"] != "binary_f64_arrays":
            raise ValueError(
                "array specialization requires a declared binary array entry"
            )
        if shape is not None and (
            not isinstance(shape, list)
            or not shape
            or not all(type(n) is int and n >= 0 for n in shape)
        ):
            raise ValueError("expected nonnegative array dimensions")
        abi = {
            **abi,
            "shape": shape if shape is not None else abi["shape"],
            "count": trip_count if trip_count is not None else abi["count"],
        }
    if abi is not None:
        graph = dict(graph, entry_abi=abi)
    module, issues, count = bind(graph, folder)
    binding_issues = len(issues)
    module, normalized_issues = normalize(module)
    issues.extend(normalized_issues)
    status = "INCOMPLETE" if issues else "READY"
    module, changes = rewrite(module)
    module, lowering_facts = prepare(module, executable=not issues)
    ir_path = seal(
        module,
        out,
        status=status,
        graph_sha256=revision,
        contract=graph["contract"],
        issues=[
            {**issue, "reported_by": "M1" if index < binding_issues else "M2"}
            for index, issue in enumerate(issues)
        ],
    )
    import_analysis = {}
    if "python_entry_environment" in module:
        import_analysis["python_entry_environment"] = member(
            out, "analysis", module["python_entry_environment"]
        )
    sealed_fields = module.keys() & {"source_imports", "source_obligations"}
    if sealed_fields:
        _, _, product = open_product(ir_path, "exec-ir.json", IR)
        import_analysis.update({key: product["module"][key] for key in sealed_fields})
    artifact_analysis = module.get("artifact_analysis")
    if artifact_analysis is not None:
        artifact_analysis = {
            **artifact_analysis,
            "graph": member(out, "analysis", artifact_analysis["graph"]),
        }
    resource_analysis = module.get("resource_analysis")
    if resource_analysis is not None:
        resource_analysis = {
            **resource_analysis,
            "graph": member(out, "resources", resource_analysis["graph"]),
        }
    protocol_analysis = module.get("protocol_analysis")
    if protocol_analysis is not None:
        protocol_analysis = dict(protocol_analysis)
        if "closed_fragments" in protocol_analysis:
            protocol_analysis["closed_fragments"] = [
                {**fragment, "function": member(out, "analysis", fragment["function"])}
                for fragment in protocol_analysis["closed_fragments"]
            ]
        if "prefix_fragments" in protocol_analysis:
            protocol_analysis["prefix_fragments"] = [
                {
                    key: value
                    for key, value in fragment.items()
                    if key not in {"body", "frontier"}
                }
                | {"fragment": member(out, "analysis", fragment)}
                for fragment in protocol_analysis["prefix_fragments"]
            ]
    report = document(
        PLAN,
        status=status,
        graph_sha256=revision,
        entry=graph["entry"],
        reachable_symbols=count,
        changes=changes,
        issues=issues,
        baseline_defects=module.get("baseline_defects", []),
        normalizations=module.get("normalizations", []),
        **import_analysis,
        **(
            {"artifact_analysis": artifact_analysis}
            if artifact_analysis is not None
            else {}
        ),
        **(
            {"resource_analysis": resource_analysis}
            if resource_analysis is not None
            else {}
        ),
        **(
            {"protocol_analysis": protocol_analysis}
            if protocol_analysis is not None
            else {}
        ),
        layers=[
            dict(
                layer="M1_BIND",
                status="INCOMPLETE" if binding_issues else "READY",
                issues=binding_issues,
            ),
            dict(
                layer="M2_NORMALIZE",
                status="INCOMPLETE" if issues else "READY",
                issues=len(normalized_issues),
            ),
            dict(
                layer="M3_REWRITE",
                enabled=True,
                executed=not issues,
                changes=len(changes),
            ),
            dict(layer="M4_EXECUTION", status=status, **lowering_facts),
        ],
        issue_categories=dict(Counter(d["code"] for d in issues).most_common()),
    )
    write_json(out / "plan.json", report)
    return report, ir_path


def plan(graph_path, out, *, shape=None, trip_count=None):
    path, revision, graph = open_product(graph_path, "graph.json", GRAPH)
    out = Path(out).resolve()
    options = dict(shape=shape, trip_count=trip_count)
    if "images" not in graph:
        report, ir_path = _plan_image(graph, path.parent, revision, out, **options)
        return dict(
            status=report["status"],
            plan=str(out / "plan.json"),
            ir=str(ir_path),
            reachable_symbols=report["reachable_symbols"],
            rewrites=len(report["changes"]),
            blockers=len(report["issues"]),
        )

    images, root = graph["images"], graph["root_image"]
    validate_images(images, root, unit_count=len(graph["units"]))
    plans, products = [], []
    for image in images:
        identity = {key: image[key] for key in ("id", "entry", "output")}
        view = dict(
            graph,
            current_image=image["id"],
            entry=image["entry"],
            entry_abi=image.get("entry_abi"),
            units=[graph["units"][index] for index in image["units"]],
        )
        child_dir = out / "images" / image["id"]
        report, child_ir = _plan_image(
            view,
            path.parent,
            revision,
            child_dir,
            shape=shape if image["id"] == root else None,
            trip_count=trip_count if image["id"] == root else None,
        )
        products.append(dict(identity, ir=file_receipt(out, child_ir)))
        plans.append(
            dict(
                identity,
                status=report["status"],
                reachable_symbols=report["reachable_symbols"],
                issues=report["issues"],
                changes=report["changes"],
                plan=file_receipt(out, child_dir / "plan.json"),
            )
        )
    status = "READY" if all(p["status"] == "READY" for p in plans) else "INCOMPLETE"
    from .workflow import link_images

    workflow = member(out, "analysis", link_images(out, products, root))
    fields = dict(
        kind="image_bundle",
        root_image=root,
        graph_sha256=revision,
        status=status,
        workflow=workflow,
    )
    ir_path = out / "exec-ir.json"
    write_json(
        ir_path,
        document(
            IR,
            **fields,
            contract=graph["contract"],
            logical_optimization=True,
            images=products,
        ),
    )
    total = dict(
        reachable_symbols=sum(p["reachable_symbols"] for p in plans),
        blockers=sum(len(p["issues"]) for p in plans),
        rewrites=sum(len(p["changes"]) for p in plans),
    )
    write_json(out / "plan.json", document(PLAN, **fields, **total, images=plans))
    return dict(
        status=status,
        plan=str(out / "plan.json"),
        ir=str(ir_path),
        images=len(plans),
        **total,
    )

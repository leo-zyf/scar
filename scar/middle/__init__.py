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
    validate_images,
    write_json,
)


def _plan_image(graph, folder, revision, out, *, optimize_graph, shape, trip_count):
    from .binding import bind
    from .normalization import normalize
    from .rewrites import optimize
    from .execution import prepare, seal

    bound, issues, count = bind(graph, folder)
    binding_issues = len(issues)
    module, normalized_issues = normalize(bound)
    issues.extend(normalized_issues)
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
    if abi:
        module["entry_abi"] = abi
    status = "INCOMPLETE" if issues else "READY"
    changes = []
    lowering_facts = {}
    if not issues:
        for fn in module["functions"]:
            if optimize_graph:
                changes.extend(optimize(fn))
        module, lowering_facts = prepare(module)
    ir_path = seal(
        module,
        out,
        status=status,
        graph_sha256=revision,
        contract=graph["contract"],
        logical_optimization=optimize_graph,
    )
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
        layers=[
            dict(layer="M1_BIND", issues=binding_issues),
            dict(layer="M2_NORMALIZE", issues=len(normalized_issues)),
            dict(layer="M3_REWRITE", enabled=optimize_graph, changes=len(changes)),
            dict(layer="M4_EXECUTION", status=status, **lowering_facts),
        ],
        issue_categories=dict(Counter(d["code"] for d in issues).most_common()),
    )
    write_json(out / "plan.json", report)
    return report, ir_path


def plan(graph_path, out, *, optimize_graph=True, shape=None, trip_count=None):
    path, revision, graph = open_product(graph_path, "graph.json", GRAPH)
    out = Path(out).resolve()
    options = dict(optimize_graph=optimize_graph, shape=shape, trip_count=trip_count)
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
            optimize_graph=optimize_graph,
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
    fields = dict(
        kind="image_bundle", root_image=root, graph_sha256=revision, status=status
    )
    ir_path = out / "exec-ir.json"
    write_json(
        ir_path,
        document(
            IR,
            **fields,
            contract=graph["contract"],
            logical_optimization=optimize_graph,
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

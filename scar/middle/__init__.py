"""Publish logical rewrites and a closed execution IR using only the graph package."""

from collections import Counter
from pathlib import Path

from scar.contract import GRAPH, PLAN, document, open_product, write_json


def plan(graph_path, out, *, optimize_graph=True, shape=None, trip_count=None):
    from .binding import bind
    from .normalization import normalize
    from .rewrites import optimize
    from .execution import prepare, seal

    path, revision, graph = open_product(graph_path, "graph.json", GRAPH)
    out = Path(out).resolve()
    bound, issues, count = bind(graph, path.parent)
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
    return dict(
        status=status,
        plan=str(out / "plan.json"),
        ir=str(ir_path),
        reachable_symbols=count,
        rewrites=len(changes),
        blockers=len(issues),
    )

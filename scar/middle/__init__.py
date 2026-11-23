"""Publish logical rewrites and a closed execution IR using only the graph package."""

from collections import Counter
from pathlib import Path

from scar.contract import GRAPH, IR, PLAN, document, member, open_product, write_json
from .link import link
from .rewrites import optimize


def plan(graph_path, out, *, optimize_graph=True, shape=None, trip_count=None):
    path, revision, graph = open_product(graph_path, "graph.json", GRAPH)
    out = Path(out).resolve()
    module, issues, count = link(graph, path.parent)
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
    fragments = []
    if not issues:
        for fn in module.pop("functions"):
            if optimize_graph:
                changes.extend(optimize(fn))
            fragments.append(member(out, "functions", fn))
    ir = document(
        IR,
        status=status,
        graph_sha256=revision,
        contract=graph["contract"],
        module=module if not issues else {"entry": graph["entry"]},
        functions=fragments,
        logical_optimization=optimize_graph,
    )
    write_json(out / "exec-ir.json", ir)
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
        issue_categories=dict(Counter(d["code"] for d in issues).most_common()),
    )
    write_json(out / "plan.json", report)
    return dict(
        status=status,
        plan=str(out / "plan.json"),
        ir=str(out / "exec-ir.json"),
        reachable_symbols=count,
        rewrites=len(changes),
        blockers=len(issues),
    )

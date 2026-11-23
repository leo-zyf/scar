"""Layer contracts keep binding, normalization, rewrites and sealing independent."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import scar
from scar.contract import document, GRAPH, member, write_json
from scar.frontend.graph import function_graph

_LAYERS = {
    "binding": "scar.middle.binding",
    "normalization": "scar.middle.normalization",
    "rewrites": "scar.middle.rewrites",
    "execution": "scar.middle.execution",
}


def _layer(name, code, data=None):
    blocked = [module for key, module in _LAYERS.items() if key != name]
    blocked.extend(["scar.frontend", "scar.backend"])
    script = """import sys,json
class BlockOtherLayers:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise ImportError('cross-layer import: ' + fullname)
blocked = json.loads(sys.argv[1])
sys.meta_path.insert(0,BlockOtherLayers())
data=json.loads(sys.stdin.read())
""" + code
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, json.dumps(blocked)],
        input=json.dumps(data),
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(scar.__file__).resolve().parents[1]),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _canonical_function(function):
    module = _layer(
        "normalization",
        """from scar.middle.normalization import normalize
module,issues=normalize(data)
assert not issues
print(json.dumps(module))
""",
        dict(
            functions=[function],
            globals=[],
            records=[],
            externs=[],
            declarations={},
            abi_environment=dict(architecture="x86_64"),
        ),
    )
    return module["functions"][0]


def test_middle_layers_consume_only_preceding_products(tmp_path):
    ptr = {"kind": "ptr", "to": "i64"}
    ref = {"kind": "ref", "to": "i64"}
    main = dict(
        name="main",
        result="i32",
        params=[],
        body=[
            dict(
                op="decl",
                name="x",
                type="i64",
                init=dict(op="literal", type="i64", value=5),
            ),
            dict(
                op="expr",
                value=dict(
                    op="call",
                    callee="update",
                    args=[dict(op="var", name="x")],
                    parameter_types=[ref],
                ),
            ),
            dict(op="return", value=dict(op="literal", type="i32", value=0)),
        ],
    )
    update = dict(
        name="update",
        result="void",
        params=[dict(name="value", type=ptr)],
        body=[dict(op="return")],
    )
    functions = [function_graph(f, "revision") for f in (main, update)]
    globals_ = function_graph(
        dict(name="@initializers", result="void", params=[], body=[]), "revision"
    )
    shard = member(
        tmp_path,
        "units",
        dict(
            functions=functions,
            global_graph=globals_,
            records=[],
            externs=[
                dict(name="update", result="i32", params=[dict(name="value", type=ref)])
            ],
        ),
    )
    graph = document(
        GRAPH,
        entry="main",
        platform={"architecture": "x86_64"},
        resources=[],
        topology=[],
        units=[
            dict(
                graph=shard,
                initializer_calls=[],
                diagnostics=[],
                functions=[
                    dict(
                        name=f["signature"]["name"],
                        calls=f["calls"],
                        objects=f["objects"],
                    )
                    for f in functions
                ],
            )
        ],
    )
    graph_file = tmp_path / "graph.json"
    write_json(graph_file, graph)
    bound = _layer(
        "binding",
        """from scar.middle.binding import bind
from pathlib import Path
module,issues,count=bind(data['graph'],Path(data['folder']))
assert not issues and count==2
print(json.dumps(module))
""",
        dict(graph=graph, folder=str(tmp_path)),
    )
    assert "declarations" in bound and "normalizations" not in bound
    assert bound["state_evidence"]["main"]["graph_id"] == functions[0]["id"]
    graph_file.unlink()
    normalized = _layer(
        "normalization",
        """from scar.middle.normalization import normalize
module,issues=normalize(data)
assert not issues
print(json.dumps(module))
""",
        bound,
    )
    assert normalized["baseline_defects"][0]["calls"] == 1
    call = normalized["functions"][0]["body"][1]["value"]
    assert call["args"][0]["op"] == "unary" and call["args"][0]["operator"] == "addr"
    assert "declarations" not in normalized and "declarations" in bound
    coordinate = dict(
        name="coordinate",
        result="void",
        params=[dict(name="n", type="i64"), dict(name="out", type=ptr)],
        body=[
            dict(
                op="map",
                index="i",
                extent=dict(op="var", name="n"),
                independent=True,
                site="map",
                body=[
                    dict(
                        op="assign",
                        target=dict(
                            op="index",
                            base=dict(op="var", name="out"),
                            indices=[dict(op="var", name="i")],
                        ),
                        value=dict(
                            op="binary",
                            operator="mod",
                            left=dict(op="var", name="i"),
                            right=dict(op="var", name="n"),
                            site="mod",
                        ),
                    )
                ],
            )
        ],
    )
    coordinate = _canonical_function(coordinate)
    changed = _layer(
        "rewrites",
        """from scar.middle.rewrites import optimize
changes=optimize(data)
print(json.dumps({'function':data,'changes':changes}))
""",
        coordinate,
    )
    assert changed["changes"][0]["rule"] == "bounded_integer_coordinate"
    assert changed["function"]["body"][0]["body"][0]["value"]["op"] == "var"
    unsafe = deepcopy(coordinate)
    unsafe["body"][0]["body"].insert(
        0,
        dict(
            op="assign",
            target=dict(op="var", name="n"),
            value=dict(op="literal", type="i64", value=1),
        ),
    )
    rejected = _layer(
        "rewrites",
        """from scar.middle.rewrites import optimize
changes=optimize(data)
print(json.dumps({'function':data,'changes':changes}))
""",
        unsafe,
    )
    assert not any(
        c["rule"] == "bounded_integer_coordinate" for c in rejected["changes"]
    )
    normalized["functions"].append(changed["function"])
    sealed = _layer(
        "execution",
        """from pathlib import Path
from scar.middle.execution import prepare,seal
module,facts=prepare(data['module'])
path=seal(module,Path(data['out']),status='READY',graph_sha256='reference',contract={},logical_optimization=True)
print(json.dumps({'path':str(path),'facts':facts}))
""",
        dict(module=normalized, out=str(tmp_path / "sealed")),
    )
    product = json.loads(Path(sealed["path"]).read_bytes())
    assert product["status"] == "READY" and len(product["functions"]) == 3


def test_graph_construction_and_state_analysis_are_separate():
    fn = dict(
        name="count",
        result="void",
        params=[dict(name="n", type="i64")],
        body=[
            dict(
                op="decl",
                name="value",
                type="i64",
                init=dict(op="literal", type="i64", value=0),
            ),
            dict(
                op="map",
                index="i",
                extent=dict(op="var", name="n"),
                body=[
                    dict(
                        op="assign",
                        target=dict(op="var", name="value"),
                        value=dict(
                            op="binary",
                            operator="add",
                            left=dict(op="var", name="value"),
                            right=dict(op="literal", type="i64", value=1),
                        ),
                    )
                ],
            ),
        ],
    )
    graph = function_graph(fn, "revision")
    assert "state_flow" not in graph
    from scar.frontend.state_flow import state_flow

    result = state_flow(graph["nodes"], graph["entry"], graph["accesses"])
    assert result["status"] == "structured"
    assert any(version["kind"] == "LOOP_CARRY" for version in result["versions"])
    assert "state_flow" not in graph


def test_rewrites_use_binding_ownership_for_implicit_borrows():
    def var(name):
        return dict(op="var", name=name)

    ref = dict(kind="ref", to="i64")
    coordinate = dict(
        name="coordinate",
        result="void",
        params=[dict(name="out", type=dict(kind="ptr", to="i64"))],
        body=[
            dict(
                op="decl",
                name="n",
                type="i64",
                init=dict(op="literal", type="i64", value=6),
            ),
            dict(op="decl", name="saved", type="i64", init=var("n")),
            dict(
                op="map",
                index="i",
                extent=var("saved"),
                independent=True,
                site="map",
                body=[
                    dict(
                        op="assign",
                        target=dict(op="index", base=var("out"), indices=[var("i")]),
                        value=dict(
                            op="binary", operator="mod", left=var("i"), right=var("n")
                        ),
                    ),
                ],
            ),
        ],
    )
    borrowings = [
        [
            dict(op="decl", name="r", type=ref, init=var("n")),
            dict(
                op="assign",
                target=var("r"),
                value=dict(op="literal", type="i64", value=1),
            ),
        ],
        [
            dict(
                op="expr",
                value=dict(
                    op="object_construct",
                    record="Holder",
                    args=[var("n")],
                    parameter_types=[ref],
                ),
            )
        ],
        [
            dict(
                op="expr",
                value=dict(
                    op="library_call", args=[var("n")], parameter_modes=["reference"]
                ),
            )
        ],
        [
            dict(
                op="expr",
                value=dict(
                    op="indirect_call", args=[var("n")], signature=dict(params=[ref])
                ),
            )
        ],
        [dict(op="expr", value=dict(op="call", callee="open", args=[var("n")]))],
    ]
    for borrow in [[], *borrowings]:
        fn = deepcopy(coordinate)
        fn["body"][2:2] = borrow
        fn = _canonical_function(fn)
        assert ("n" in fn["state_contract"]["private_scalars"]) == (not borrow)
        result = _layer(
            "rewrites",
            """from scar.middle.rewrites import optimize
print(json.dumps(optimize(data)))
""",
            fn,
        )
        assert any(c["rule"] == "bounded_integer_coordinate" for c in result) == (
            not borrow
        )

    scoped = deepcopy(coordinate)
    scoped["body"][-1]["body"][0]["value"]["right"]["scope"] = "global"
    result = _layer(
        "rewrites",
        """from scar.middle.rewrites import optimize
print(json.dumps(optimize(data)))
""",
        _canonical_function(scoped),
    )
    assert not any(c["rule"] == "bounded_integer_coordinate" for c in result)

    multidimensional = deepcopy(coordinate)
    loop = multidimensional["body"][-1]
    loop["extent"] = dict(
        op="binary", operator="mul", left=var("n"), right=var("saved")
    )
    other = deepcopy(loop)
    other["site"] = "map2"
    multidimensional["body"].append(other)
    changed = _layer(
        "rewrites",
        """from scar.middle.rewrites import optimize
changes=optimize(data)
print(json.dumps({'function':data,'changes':changes}))
""",
        _canonical_function(multidimensional),
    )
    assert any(c["rule"] == "common_iteration_extent" for c in changed["changes"])
    function = _layer(
        "execution",
        """from scar.middle.execution import annotate_iteration_layouts
assert annotate_iteration_layouts(data)==2
print(json.dumps(data))
""",
        changed["function"],
    )
    for loop in function["body"][-2:]:
        assert loop["iteration_layout"]["axes"] == [var("n"), var("saved")]


def test_reference_binding_and_opaque_call_use_the_referred_state():
    from scar.frontend.state_flow import state_flow

    def var(name):
        return dict(op="var", name=name)

    ref = dict(kind="ref", to="i64")
    number = dict(op="literal", type="i64", value=9)
    fn = dict(
        name="inspect",
        result="i64",
        params=[],
        body=[
            dict(op="decl", name="n", type="i64", init=number),
            dict(op="decl", name="r", type=ref, init=var("n")),
            dict(op="assign", target=var("r"), value=number),
            dict(
                op="expr",
                value=dict(
                    op="call", callee="retain", args=[var("r")], parameter_types=[ref]
                ),
            ),
            dict(
                op="expr",
                value=dict(op="call", callee="later", args=[], parameter_types=[]),
            ),
            dict(op="return", value=var("n")),
        ],
    )
    graph = function_graph(fn, "revision")
    slots = {state["name"]: slot for slot, state in graph["states"].items()}
    target, reference = slots["n"], slots["r"]
    assert graph["states"][reference]["alias_to"] == target
    assert graph["states"][target]["address_taken"]
    assert any(
        a["domain"] == target and a["mode"] == "write" and a.get("binding") == reference
        for a in graph["accesses"]
    )
    records = state_flow(graph["nodes"], graph["entry"], graph["accesses"])["versions"]
    versions = {r["node"]: r for r in records}
    binding, mutation, call, later, read = [
        versions[h["port"]] for h in graph["entry"][1:]
    ]
    assert reference in binding["outgoing"] and target not in binding["outgoing"]
    assert mutation["incoming"][reference] == binding["outgoing"][reference]
    assert call["incoming"][target] == mutation["outgoing"][target]
    assert read["incoming"][target] == call["outgoing"][target]
    assert read["incoming"]["memory"] == later["outgoing"]["memory"]

"""Layer contracts keep binding, normalization, rewrites and sealing independent."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
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


def test_execution_scopes_automatic_for_initializers_before_sealing():
    index = dict(op="var", name="i", type="i64")

    def literal(n):
        return dict(op="literal", type="i64", value=n)

    function = dict(
        name="main",
        result="i32",
        params=[],
        body=[
            dict(
                op="for",
                init=[
                    dict(op="decl", name="i", type="i64", init=literal(0)),
                    dict(
                        op="runtime_array",
                        name="scratch",
                        element="f64",
                        extent=literal(2),
                        storage="scoped",
                        initialization="uninitialized",
                    ),
                ],
                condition=dict(
                    op="binary",
                    operator="lt",
                    type="bool",
                    left=index,
                    right=literal(2),
                ),
                step=dict(
                    op="update",
                    operator="add",
                    target=index,
                    value=literal(1),
                    type="i64",
                ),
                body=[dict(op="continue")],
            ),
            dict(op="return", value=dict(op="literal", type="i32", value=0)),
        ],
    )
    result = _layer(
        "execution",
        """from scar.middle.execution import prepare
module,facts=prepare(data)
print(json.dumps(dict(module=module,facts=facts)))
""",
        dict(functions=[function], globals=[]),
    )
    block = result["module"]["functions"][0]["body"][0]
    assert block["op"] == "block"
    initial, array, loop = block["body"]
    assert initial["name"] == "i" and array["name"] == "scratch"
    assert array["scope_control"] == "lexical_continuation"
    assert loop["op"] == "for" and loop["init"] == []
    assert loop["condition"] == function["body"][0]["condition"]
    assert loop["step"] == function["body"][0]["step"]
    assert result["facts"]["scoped_lifetimes"] == 1


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


def test_mpi_contracts_preserve_lifetimes_without_guessing_user_symbols():
    names = [
        "MPI_Isend",
        "MPI_Irecv",
        "MPI_Send_init",
        "MPI_Request_free",
        "MPI_Ibarrier",
        "MPI_Finalize",
    ]
    body = [dict(op="call", callee=name, args=[]) for name in names]
    module = dict(
        functions=[dict(name="worker", body=body)],
        globals=[],
        records=[],
        externs=[dict(name=name, provider="mpi") for name in names],
        operator_requirements=dict(ndarray=dict(operations=[])),
    )
    code = """from scar.middle.normalization.mpi import qualify
changes,issues=qualify(data)
print(json.dumps(dict(module=data,changes=changes,issues=issues)))
"""
    result = _layer("normalization", code, module)
    assert not result["issues"]
    qualified = result["module"]
    assert (
        qualified["operator_requirements"]["ndarray"]
        == module["operator_requirements"]["ndarray"]
    )
    contracts = {
        call["callee"]: call["mpi_contract"]["interfaces"][call["callee"]]
        for call in qualified["functions"][0]["body"]
    }
    assert contracts["MPI_Isend"]["buffer_access"]["mode"] == "read"
    assert contracts["MPI_Irecv"]["buffer_access"]["mode"] == "write"
    assert contracts["MPI_Irecv"]["buffer_lifetime"] == "call_until_request_completion"
    assert contracts["MPI_Send_init"]["communication_started_by_call"] is False
    assert contracts["MPI_Request_free"]["operation_completion"] == "not_implied"
    assert contracts["MPI_Ibarrier"]["synchronization"] == "on_request_completion"
    assert (
        contracts["MPI_Finalize"]["satisfies_pending_completion_obligations"] is False
    )
    assert all(
        not call["mpi_contract"]["reorderable"]
        for call in qualified["functions"][0]["body"]
    )

    shadowed = dict(
        functions=[
            dict(name="MPI_Finalize", body=[]),
            dict(name="caller", body=[dict(op="call", callee="MPI_Finalize", args=[])]),
        ],
        externs=[dict(name="MPI_Finalize", provider="mpi")],
        source_evidence=dict(historical=dict(provider="mpi")),
    )
    result = _layer("normalization", code, shadowed)
    assert not result["issues"] and "mpi" not in result["module"].get(
        "operator_requirements", {}
    )
    assert "operator_family" not in result["module"]["functions"][1]["body"][0]
    forged = deepcopy(module)
    forged["functions"][0]["body"][0]["mpi_interface"] = "MPI_Finalize"
    assert _layer("normalization", code, forged)["issues"]


def test_constant_file_proof_preserves_control_and_handle_boundaries():
    def literal(value):
        return dict(op="literal", type="text", value=value)

    handle = dict(op="var", name="file")
    opened = dict(
        op="host_call",
        interface="stream.open",
        args=[literal("out.h"), literal("w"), literal("utf-8"), literal("universal")],
    )
    start = dict(op="decl", name="file", init=opened)
    write = dict(
        op="expr",
        value=dict(
            op="host_call",
            interface="stream.write",
            args=[handle, literal("constant\n")],
        ),
    )
    close = dict(
        op="expr", value=dict(op="host_call", interface="stream.close", args=[handle])
    )
    code = "from scar.middle.normalization.artifacts import analyze\nprint(json.dumps(analyze(data)))"

    def analysis(body, **extra):
        module = dict(functions=[dict(name="writer", body=body)], **extra)
        return _layer("normalization", code, module)

    positive = analysis([start, write, close])
    assert len(positive["constant_materializations"]) == 1
    assert (
        positive["constant_materializations"][0]["file_content_status"]
        == "conditional_on_storage_and_writer_contracts"
    )
    copied = dict(
        op="expr",
        value=dict(
            op="host_call",
            interface="fs.copy_file",
            args=[literal("out.h"), literal("copy.h")],
        ),
    )
    read_handle = dict(op="var", name="reader")
    read = [
        dict(
            op="decl",
            name="reader",
            init={**opened, "args": [literal("copy.h"), *opened["args"][1:]]},
        ),
        dict(
            op="expr",
            value=dict(op="host_call", interface="stream.read", args=[read_handle]),
        ),
    ]
    read[0]["init"]["args"][1] = literal("r")
    copied_flow = analysis([start, write, close, copied, *read])["graph"]
    nodes = {node["id"]: node for node in copied_flow["nodes"]}
    (edge,) = [
        edge for edge in copied_flow["edges"] if edge["kind"] == "COPIES_FILE_VERSION"
    ]
    source, target = (nodes[edge[key]] for key in ("source", "target"))
    assert source["value"] == target["value"] and source["locator"] != target["locator"]
    assert target["requires"] and target["copy_contract"]["metadata"] == "content_only"
    assert {
        edge["role"]
        for edge in copied_flow["edges"]
        if edge["kind"] == "REQUIRES_FILE_VERSION"
    } == {"copy_source", "open_read", "stream_read"}
    for boundary in (
        dict(op="call", callee="unknown", args=[]),
        dict(op="process_request", request_id="unbound"),
    ):
        invalidated = analysis(
            [start, write, close, dict(op="expr", value=boundary), copied]
        )["graph"]
        assert any(
            edge["kind"] == "MAY_INVALIDATE_FILE_VERSION"
            for edge in invalidated["edges"]
        )
        assert not any(
            edge["kind"] == "COPIES_FILE_VERSION" for edge in invalidated["edges"]
        )
    copy2 = deepcopy(copied)
    copy2["value"]["interface"] = "fs.copy2"
    assert any(
        item["field"] == "copy_destination_kind"
        for item in analysis([start, write, close, copy2])["boundaries"]
    )
    parameters = [dict(name=name, type="text") for name in ("source", "destination")]
    source, destination = [dict(op="var", **item) for item in parameters]
    helper_copy, helper_read = deepcopy(copied), deepcopy(read)
    helper_copy["value"]["args"] = [source, destination]
    helper_read[0]["init"]["args"][0] = destination
    helper = dict(
        name="copy_read",
        params=parameters,
        result="text",
        body=[
            helper_copy,
            *helper_read,
            dict(
                op="expr",
                value=dict(
                    op="host_call", interface="stream.close", args=[read_handle]
                ),
            ),
            dict(op="return", value=destination),
        ],
    )
    caller_read = deepcopy(read)
    caller_read[0]["init"]["args"][0] = dict(op="var", name="returned", type="text")
    caller = dict(
        name="writer",
        body=[
            start,
            write,
            close,
            dict(
                op="decl",
                name="returned",
                type="text",
                init=dict(
                    op="call",
                    callee="copy_read",
                    site="writer/copy",
                    args=[literal("out.h"), literal("copy.h")],
                ),
            ),
            *caller_read,
        ],
    )
    module = dict(functions=[caller, helper])
    cross_call = _layer("normalization", code, module)
    (copy_edge,) = [
        edge
        for edge in cross_call["graph"]["edges"]
        if edge["kind"] == "COPIES_FILE_VERSION"
    ]
    reads = [
        edge
        for edge in cross_call["graph"]["edges"]
        if edge.get("role") == "stream_read"
    ]
    assert len(reads) == 2 and all(
        edge["source"] == copy_edge["target"] for edge in reads
    )
    assert cross_call["file_call_contexts"] == 1
    assert all(
        item["normal_path_closed"] for item in cross_call["file_state_transitions"]
    )
    module["functions"][1]["body"].insert(
        1, dict(op="expr", value=dict(op="call", callee="unknown", args=[]))
    )
    barrier = _layer("normalization", code, module)
    assert not any(
        edge.get("role") == "stream_read" for edge in barrier["graph"]["edges"]
    )
    cases = [
        [
            start,
            write,
            dict(op="expr", value=dict(op="call", callee="unknown", args=[handle])),
            close,
        ],
        [start, write, dict(op="decl", name="other", init=opened), close],
        [
            start,
            dict(op="while", condition=dict(op="var", name="condition"), body=[write]),
            close,
        ],
        [start, write, dict(op="block", body=[dict(op="return")]), close],
        [
            start,
            write,
            dict(
                op="decl",
                name="child",
                init=dict(op="process_spawn", image_id="worker", argv=[]),
            ),
            close,
        ],
        [
            dict(
                op="decl",
                name="child",
                init=dict(op="process_spawn", image_id="worker", argv=[]),
            ),
            start,
            write,
            close,
        ],
        [
            start,
            write,
            dict(
                op="expr",
                value=dict(
                    op="text_print",
                    args=[literal("other")],
                    sep=literal(" "),
                    end=literal("\n"),
                ),
            ),
            close,
        ],
        [
            dict(op="decl", name="g", scope="global", init=literal("old")),
            dict(op="expr", value=dict(op="call", callee="changes_global", args=[])),
            start,
            dict(
                op="expr",
                value=dict(
                    op="host_call",
                    interface="stream.write",
                    args=[handle, dict(op="var", name="g", scope="global")],
                    argument_facts=dict(args=[None, dict(type="text", exact="old")]),
                ),
            ),
            close,
        ],
    ]
    for body in cases:
        assert not analysis(body)["constant_materializations"]
    prefix = dict(
        symbol="partial",
        completion="prefix",
        body=[start],
        frontier=dict(
            operator="if",
            regions=[dict(path=["then"], completion="complete", body=[write, close])],
        ),
    )
    negative = _layer(
        "normalization",
        code,
        dict(functions=[], protocol_analysis=dict(prefix_fragments=[prefix])),
    )
    assert not negative["constant_materializations"]


def test_scalar_file_facts_respect_destination_types_and_persistent_state():
    code = """from scar.middle.normalization.values import result_summaries, apply_result, _UNKNOWN
from scar.middle.normalization.artifacts import analyze
functions={f['name']:f for f in data['functions']}
summaries=result_summaries(functions)
results={name:apply_result(summary,[],summaries) for name,summary in summaries.items()}
assert results['exact'] == 3
assert all(results[name] is _UNKNOWN for name in ('narrow','static','recursive'))
assert not analyze(data)['constant_materializations']
print(json.dumps({'checked':list(results)}))
"""
    value = dict(op="var", name="result", type="i32")
    narrow = dict(
        name="narrow",
        result="i32",
        params=[],
        body=[
            dict(op="decl", name="result", type="i32"),
            dict(
                op="assign",
                target=value,
                value=dict(op="literal", type="f64", value=3.75),
            ),
            dict(op="return", value=value),
        ],
    )
    exact = dict(
        name="exact",
        result="i32",
        params=[],
        body=[dict(op="return", value=dict(op="literal", type="i64", value=3))],
    )
    persistent = dict(
        name="static",
        result="i32",
        params=[],
        body=[
            dict(
                op="decl",
                name="result",
                type="i32",
                storage="static",
                init=dict(op="literal", type="i32", value=3),
            ),
            dict(op="return", value=value),
        ],
    )
    recursive = dict(
        name="recursive",
        result="i32",
        params=[],
        body=[
            dict(
                op="return",
                value=dict(op="call", callee="recursive", args=[], type="i32"),
            )
        ],
    )
    stream = dict(
        op="var", name="file", type=dict(kind="handle", resource="text_stream")
    )

    def text(value):
        return dict(op="literal", type="text", value=value)

    writer = dict(
        name="writer",
        result="void",
        params=[],
        body=[
            dict(
                op="decl",
                name="file",
                type=stream["type"],
                init=dict(
                    op="host_call",
                    interface="stream.open",
                    args=[text("out.txt"), text("w"), text("utf-8"), text("universal")],
                ),
            ),
            *deepcopy(narrow["body"][:-1]),
            dict(
                op="expr",
                value=dict(
                    op="text_print",
                    args=[value],
                    stream=stream,
                    sep=text(" "),
                    end=text("\n"),
                    scalar_format=dict(integer_base=10),
                ),
            ),
            dict(
                op="expr",
                value=dict(op="host_call", interface="stream.close", args=[stream]),
            ),
        ],
    )
    counter = dict(op="var", name="counter", type="i64")
    writer_static = deepcopy(writer)
    writer_static["name"] = "writer_static"
    writer_static["body"] = [
        dict(
            op="decl",
            name="counter",
            type="i64",
            storage="static",
            init=dict(op="literal", type="i64", value=1),
        ),
        deepcopy(writer["body"][0]),
        dict(
            op="expr",
            value=dict(
                op="text_print",
                args=[counter],
                stream=stream,
                sep=text(" "),
                end=text("\n"),
                scalar_format=dict(integer_base=10),
            ),
        ),
        deepcopy(writer["body"][-1]),
        dict(
            op="assign", target=counter, value=dict(op="literal", type="i64", value=2)
        ),
    ]
    result = _layer(
        "normalization",
        code,
        dict(functions=[narrow, exact, persistent, recursive, writer, writer_static]),
    )
    assert set(result["checked"]) == {
        "narrow",
        "exact",
        "static",
        "recursive",
        "writer",
        "writer_static",
    }


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

    missing_argument = deepcopy(bound)
    missing_argument["functions"][0]["body"][1]["value"]["args"].clear()
    issues = _layer(
        "normalization",
        "from scar.middle.normalization import normalize\nprint(json.dumps(normalize(data)[1]))",
        missing_argument,
    )
    arity = next(issue for issue in issues if issue["code"] == "argument_count")
    assert (arity["expected"], arity["actual"], arity["variadic"]) == (1, 0, False)


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


def test_local_opaque_regions_preserve_state_connections_and_loop_structure():
    from scar.frontend.state_flow import state_flow

    def var(name):
        return dict(op="var", name=name)

    def number(value):
        return dict(op="literal", type="i64", value=value)

    fn = dict(
        name="observed",
        result="i64",
        params=[dict(name="source", type="i64"), dict(name="stop", type="bool")],
        body=[
            dict(op="decl", name="stable", type="i64", init=number(5), site="stable"),
            dict(op="decl", name="total", type="i64", init=number(0)),
            dict(op="decl", name="item", type="i64", init=number(-1)),
            {
                "op": "try",
                "site": "protected",
                "body": [
                    dict(op="assign", target=var("total"), value=number(2)),
                ],
                "handlers": [
                    dict(
                        op="exception_handler",
                        binding=None,
                        body=[
                            dict(op="assign", target=var("total"), value=number(99)),
                        ],
                        cleanup=[],
                    )
                ],
                "finally": [],
            },
            dict(op="expr", value=var("total"), site="after_protected"),
            dict(op="expr", value=var("stable"), site="stable_read"),
            dict(
                op="for",
                protocol="iterator",
                range=var("source"),
                target=var("item"),
                site="loop",
                body=[
                    dict(
                        op="assign",
                        target=var("total"),
                        value=var("item"),
                        site="body_write",
                    ),
                    dict(
                        op="with_context",
                        resources=[],
                        body=[
                            {
                                "op": "if",
                                "condition": var("stop"),
                                "then": [dict(op="break")],
                                "else": [],
                            },
                        ],
                        effects=["STATE"],
                        state_reads=["process.cwd"],
                        state_writes=["process.cwd"],
                    ),
                ],
            ),
            dict(op="return", value=var("total"), site="result"),
        ],
    )
    graph = function_graph(fn, "partial-regions")
    flow = state_flow(graph["nodes"], graph["entry"], graph["accesses"])
    sites = {n["site"]: i for i, n in enumerate(graph["nodes"]) if "site" in n}
    slots = {s["name"]: key for key, s in graph["states"].items()}

    def record(site, kind=None):
        return next(
            r
            for r in flow["versions"]
            if r["node"] == sites[site] and (kind is None or r["kind"] == kind)
        )

    protected = record("protected", "OPAQUE")
    assert (
        record("after_protected")["incoming"][slots["total"]]
        == protected["outgoing"][slots["total"]]
    )
    assert (
        record("stable_read")["incoming"][slots["stable"]]
        == record("stable")["outgoing"][slots["stable"]]
    )
    assert slots["stable"] not in protected["outgoing"]
    assert {
        r["phase"]
        for r in flow["versions"]
        if r["node"] == sites["loop"] and r["kind"] == "OPAQUE"
    } == {"iterator_setup", "iterator_advance"}
    assert any(
        r["kind"] == "LOOP_CARRY" and r["domain"] == slots["total"]
        for r in flow["versions"]
    )
    from scar.contract import validate_state_flow

    states = graph["states"] | flow["introduced_states"]
    sealed = dict(graph, states=states, state_flow=flow)
    validate_state_flow(sealed)
    damaged = deepcopy(sealed)
    damaged["state_flow"]["versions"][0]["incoming"]["missing"] = "entry"
    with pytest.raises(ValueError, match="undefined state"):
        validate_state_flow(damaged)
    assert any(
        t.get("possible") and t["target"] == sites["loop"] and t["phase"] == "exit"
        for t in flow["transfers"]
    )
    assert len([n for n in graph["nodes"] if n["op"] == "for"]) == 1
    assert flow["coverage"]["complete"] is False and flow["versions"]
    assert states["external/process.cwd"]["kind"] == "external_state"
    assert all(
        set(r.get("incoming", {})) | set(r.get("outgoing", {})) <= states.keys()
        for r in flow["versions"]
    )

    cell = dict(op="var", name="shared", cell=dict(owner="outer", name="shared"))
    fn["body"] = [
        dict(op="expr", value=cell, site="before"),
        dict(
            op="expr",
            value=dict(op="call", callee="callback", args=[], parameter_types=[]),
            site="call",
        ),
        dict(op="return", value=cell, site="after"),
    ]
    graph = function_graph(fn, "cell-sharing")
    flow = state_flow(graph["nodes"], graph["entry"], graph["accesses"])
    records = {r["node"]: r for r in flow["versions"]}
    before, call, after = [records[h["port"]] for h in graph["entry"]]
    assert before["incoming"]["memory"] != after["incoming"]["memory"]
    assert after["incoming"]["memory"] == call["outgoing"]["memory"]
    fn["body"] = [dict(op="return", value=cell)]
    graph = function_graph(fn, "read-only-cell")
    flow = state_flow(graph["nodes"], graph["entry"], graph["accesses"])
    assert flow["introduced_states"]["memory"]["kind"] == "shared_memory"
    assert "memory" not in graph["states"]


def test_coordinate_rewrite_invalidates_condition_writes_before_branch():
    def var(name):
        return dict(op="var", name=name)

    def number(value):
        return dict(op="literal", type="i64", value=value)

    coordinate = dict(op="binary", operator="mod", left=var("i"), right=var("n"))
    function = dict(
        name="fill",
        params=[dict(name="a", type=dict(kind="ptr", to="i64"))],
        result="void",
        body=[
            dict(op="decl", name="n", type="i64", init=number(4)),
            dict(op="decl", name="saved", type="i64", init=var("n")),
            {
                "op": "if",
                "condition": dict(op="unary", operator="predec", value=var("n")),
                "then": [
                    dict(
                        op="map",
                        site="fill/map",
                        index="i",
                        extent=var("saved"),
                        independent=True,
                        body=[
                            dict(
                                op="assign",
                                target=dict(
                                    op="index", base=var("a"), indices=[var("i")]
                                ),
                                value=coordinate,
                            )
                        ],
                    )
                ],
                "else": [],
            },
        ],
    )
    function = _canonical_function(function)
    for mutating in (True, False):
        incoming = deepcopy(function)
        if not mutating:
            incoming["body"][-1]["condition"] = dict(
                op="literal", type="bool", value=True
            )
        output = _layer(
            "rewrites",
            """from scar.middle.rewrites import optimize
changes=optimize(data)
print(json.dumps(dict(function=data,changes=changes)))
""",
            incoming,
        )
        rules = {c["rule"] for c in output["changes"]}
        assert ("bounded_integer_coordinate" in rules) is not mutating
        value = output["function"]["body"][-1]["then"][0]["body"][0]["value"]
        n = 3 if mutating else 4
        actual = [i if value["op"] == "var" else i % n for i in range(4)]
        assert actual == ([0, 1, 2, 0] if mutating else [0, 1, 2, 3])

    incoming = deepcopy(function)
    loop = incoming["body"].pop()["then"][0]
    incoming["body"] += [
        dict(op="label", name="again"),
        loop,
        {
            "op": "if",
            "condition": dict(
                op="binary", operator="eq", left=var("n"), right=number(4)
            ),
            "then": [
                dict(op="assign", target=var("n"), value=number(3)),
                dict(op="goto", name="again"),
            ],
            "else": [],
        },
    ]
    output = _layer(
        "rewrites",
        """from scar.middle.rewrites import optimize
changes=optimize(data)
print(json.dumps(dict(function=data,changes=changes)))
""",
        incoming,
    )
    # The second visit has n=3 although the source-order initialization used 4.
    assert not any(c["rule"] == "bounded_integer_coordinate" for c in output["changes"])


def test_iteration_bindings_and_reachable_loop_predecessors():
    from scar.frontend.state_flow import state_flow

    def var(name):
        return dict(op="var", name=name)

    def value(n):
        return dict(op="literal", type="i64", value=n)

    def write(site, rhs):
        return dict(op="assign", site=site, target=var("x"), value=rhs)

    def branch(condition, *body):
        return {"op": "if", "condition": var(condition), "then": list(body), "else": []}

    fn = dict(
        name="advance",
        result="i64",
        params=[dict(name=n, type="bool") for n in ("again", "skip", "stop", "done")],
        body=[
            dict(op="decl", site="outer", type="i64", name="i", init=value(12)),
            dict(op="decl", type="i64", name="x", init=value(0)),
            dict(
                op="map",
                site="map",
                index="i",
                extent=var("i"),
                body=[write("element", var("i"))],
            ),
            dict(
                op="for",
                site="loop",
                init=[write("start", value(2))],
                condition=var("again"),
                step=dict(op="unary", operator="preinc", value=var("i")),
                body=[
                    dict(
                        op="decl",
                        site="scratch",
                        name="scratch",
                        type="i64",
                        init=value(8),
                    ),
                    branch(
                        "skip",
                        write("skipped", value(4)),
                        dict(op="continue"),
                        write("unreachable", value(999)),
                    ),
                    branch("stop", write("stopped", value(6)), dict(op="break")),
                    branch(
                        "done",
                        write("returned", value(7)),
                        dict(op="return", value=var("x")),
                    ),
                    write("normal", var("scratch")),
                ],
            ),
            dict(op="return", site="result", value=var("x")),
        ],
    )
    graph = function_graph(fn, "audit")
    flow = state_flow(graph["nodes"], graph["entry"], graph["accesses"])
    assert flow["status"] == "structured"
    nodes = {s["site"]: i for i, s in enumerate(graph["nodes"]) if "site" in s}

    def slot(name, kind):
        return next(
            d
            for d, s in graph["states"].items()
            if s["name"] == name and s["kind"] == kind
        )

    records = flow["versions"]

    def access(site):
        return next(
            r for r in records if r["kind"] == "ACCESS" and r["node"] == nodes[site]
        )

    outer, element, state = (
        slot("i", "local"),
        slot("i", "iteration"),
        slot("x", "local"),
    )
    extent = graph["nodes"][nodes["map"]]["extent"]["port"]
    assert next(r for r in records if r["node"] == extent)["incoming"] == {
        outer: access("outer")["outgoing"][outer]
    }
    assert access("element")["incoming"][element] == access("map")["outgoing"][element]
    assert access("map")["phase"] == "iteration"
    assert not any(r["node"] == nodes["unreachable"] for r in records)

    loop = nodes["loop"]
    carry = next(
        r
        for r in records
        if r["kind"] == "LOOP_CARRY" and r["node"] == loop and r["domain"] == state
    )
    joins = {
        r["phase"]: r
        for r in records
        if r["kind"] == "MERGE" and r["node"] == loop and r["domain"] == state
    }
    assert carry["initial"] == access("start")["outgoing"][state]
    assert carry["next"] == joins["latch"]["output"]
    assert set(joins["latch"]["inputs"]) == {
        access(n)["outgoing"][state] for n in ("skipped", "normal")
    }
    assert set(joins["exit"]["inputs"]) == {
        carry["header"],
        access("stopped")["outgoing"][state],
    }
    assert access("result")["incoming"][state] == joins["exit"]["output"]
    scratch = slot("scratch", "local")
    assert (
        access("normal")["incoming"][scratch] == access("scratch")["outgoing"][scratch]
    )
    assert not any(
        r["kind"] == "LOOP_CARRY" and r["domain"] == scratch for r in records
    )
    assert {
        (graph["nodes"][t["node"]]["op"], t["phase"])
        for t in flow["transfers"]
        if t["target"] == loop
    } == {("break", "exit"), ("continue", "step")}

    # An open iterator can mutate arbitrary state in its unobserved next() call.
    fn["body"] = [
        dict(op="for", protocol="iterator", range=var("i"), target=var("x"), body=[])
    ]
    graph = function_graph(fn, "audit")
    assert (
        state_flow(graph["nodes"], graph["entry"], graph["accesses"])["status"]
        == "conservative_iterator_protocol"
    )

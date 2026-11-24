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
    product_type = dict(kind="product_ref", items=["text", "i64"])
    product = dict(op="var", name="paths", type=product_type)

    def packed(path, count):
        return dict(
            op="product_make",
            type=product_type,
            items=[literal(path), dict(op="literal", type="i64", value=count)],
            identity=dict(kind="fresh"),
            effects=["MEM", "STATE"],
            may_raise=True,
        )

    product_copy = deepcopy(copied)
    product_copy["value"]["args"][0] = dict(
        op="product_get", type="text", base=product, index=0, effects=["STATE"]
    )
    for alternate, expected in (("out.h", 1), ("other.h", 0)):
        paths = analysis(
            [
                start,
                write,
                close,
                dict(op="decl", name="paths", type=product_type),
                dict(
                    op="if",
                    condition=dict(op="var", name="choice", type="bool"),
                    then=[dict(op="assign", target=product, value=packed("out.h", 1))],
                    **{
                        "else": [
                            dict(
                                op="assign", target=product, value=packed(alternate, 2)
                            )
                        ]
                    },
                ),
                product_copy,
            ]
        )["graph"]
        assert (
            sum(edge["kind"] == "COPIES_FILE_VERSION" for edge in paths["edges"])
            == expected
        )
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
    alternatives = analysis([start, write, close, copy2, *read])["graph"]
    versions = {node["id"]: node for node in alternatives["nodes"]}
    copies = [
        edge for edge in alternatives["edges"] if edge["kind"] == "COPIES_FILE_VERSION"
    ]
    assert {edge["destination_case"] for edge in copies} == {
        "destination_is_not_directory",
        "destination_is_directory",
    }
    assert {
        versions[versions[edge["target"]]["locator"]]["path"] for edge in copies
    } == {"./copy.h", "./copy.h/out.h"}
    assert not any(
        edge.get("role") == "open_read" for edge in alternatives["edges"]
    ), "an unselected destination arm must not become the current file version"
    unknown_source = analysis([copy2])["graph"]
    assert (
        sum(
            edge["kind"] == "COPIES_REQUIRED_FILE_VERSION"
            for edge in unknown_source["edges"]
        )
        == 2
    )
    assert not any(
        edge["kind"] == "COPIES_FILE_VERSION" for edge in unknown_source["edges"]
    )
    unknown_paths = deepcopy(copy2)
    unknown_paths["value"]["args"] = [
        dict(op="var", name=name, type="text") for name in ("source", "target")
    ]
    dynamic = analysis([unknown_paths])["graph"]
    locators = [node for node in dynamic["nodes"] if node["kind"] == "FILE_LOCATOR"]
    unresolved = [node for node in locators if node.get("operand")]
    assert len(unresolved) == 3 and len({node["id"] for node in unresolved}) == 3
    launch = dict(
        op="expr",
        value=dict(
            op="process_spawn",
            image_id="worker",
            argv=[literal("./worker")],
            cwd="inherit",
        ),
    )
    handoff = analysis([start, write, close, copied, launch, *read])["graph"]
    (environment,) = handoff["launch_environments"]
    assert environment["image_id"] == "worker"
    assert environment["argument_ports"] == [["argv", 0]]
    assert len(environment["visible_versions"]) == 2
    assert environment["historical_instance"] == "not_established"
    assert environment["directory"] is not None
    assert (
        sum(
            edge["kind"] == "AVAILABLE_BEFORE_LAUNCH_IF_NORMAL"
            for edge in handoff["edges"]
        )
        == 2
    )
    assert not any(
        edge.get("role") == "open_read" for edge in handoff["edges"]
    ), "spawn still invalidates unqualified filesystem state after recording its input"
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
    # Both DFS branches belong to the same recursive component; visiting
    # only back-edge stack entries can miss a branch through a finished node.
    recursive = dict(
        functions=[
            dict(
                name=name,
                params=[],
                result="unit",
                body=[
                    dict(op="expr", value=dict(op="call", callee=target, args=[]))
                    for target in targets
                ],
            )
            for name, targets in (("a", ("b", "c")), ("b", ("a",)), ("c", ("b",)))
        ]
    )
    assert (
        _layer("normalization", code, recursive)["file_effect_summary"][
            "recursive_functions"
        ]
        == 3
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


def test_projection_import_keeps_verified_class_and_shared_registry_effects():
    root = "/captured/site-packages/"

    def provider(module, member, checksum):
        return dict(
            status="verified_distribution",
            distribution="matplotlib",
            version="3.11.2",
            module=module,
            member=member,
            origin=root + member,
            sha256=checksum,
            proof_scope="installed_file_with_digest",
        )

    package = provider(
        "matplotlib",
        "matplotlib/__init__.py",
        "37abb78e74e9e0085a4e306a919606ffa8ff3c263def80ae2f4d938cfa79cd32",
    )
    pyplot = provider(
        "matplotlib.pyplot",
        "matplotlib/pyplot.py",
        "2c27b5739ad10034f22866cc7647ce968c56f9625cf0f7a288b800e94917ea51",
    )
    projection = provider(
        "mpl_toolkits.mplot3d",
        "mpl_toolkits/mplot3d/__init__.py",
        "7c7f4774c7c532f8db216ab2da08d09e6da6dda7b50af2e2b87e8bc0a1e8d242",
    )
    selected = provider(
        "mpl_toolkits.mplot3d.axes3d",
        "mpl_toolkits/mplot3d/axes3d.py",
        "6823b0e69fe54255ee82de392aeefac5e2813f23204eec8b1ad540a0357aee35",
    )
    selected.update(
        name="Axes3D",
        conditions=[],
        decorators=[],
        via=[
            dict(
                kind=kind,
                module=source["module"],
                member=source["member"],
                sha256=source["sha256"],
                name="Axes3D",
            )
            for kind, source in (("from", projection), ("definition", selected))
        ],
    )
    projection["exports"] = dict(
        Axes3D=dict(status="source_candidates", unresolved=[], candidates=[selected])
    )
    first = dict(
        op="python_import",
        module="matplotlib.pyplot",
        source_provider=package,
        import_resolution=pyplot,
        provider="python_package",
        request=dict(form="import", level=0, module="matplotlib.pyplot"),
        alias="plt",
        binding=dict(op="var", name="py:test:plt", scope="global"),
        effects=["CTRL", "STATE", "IO"],
    )
    imported = dict(
        op="python_import",
        module="mpl_toolkits.mplot3d",
        member="Axes3D",
        source_provider=dict(
            module="mpl_toolkits",
            status="namespace",
            loader="namespace",
            proof_scope="ordered_search_locations",
            locations=[root + "mpl_toolkits"],
        ),
        import_resolution=projection,
        provider="python_package",
        request=dict(
            form="from", level=0, module="mpl_toolkits.mplot3d", fromlist=["Axes3D"]
        ),
        binding=dict(op="var", name="py:test:Axes3D", scope="global"),
        effects=["CTRL", "STATE", "IO"],
        site="source/axes-import",
    )
    alias = deepcopy(imported)
    alias["binding"]["name"] = "py:test:Alias"
    alias["site"] = "source/alias-import"
    namespace, source, route = (deepcopy(imported) for _ in range(3))
    namespace["source_provider"]["locations"] = ["/other/mpl_toolkits"]
    source["import_resolution"]["exports"]["Axes3D"]["candidates"][0]["sha256"] = (
        "0" * 64
    )
    route["import_resolution"]["exports"]["Axes3D"]["candidates"][0]["via"][0][
        "module"
    ] = "unrelated"
    results = _layer(
        "normalization",
        """from scar.middle.normalization.protocols import _Project,_Frame,_Unclosed
from scar.contract import walk_operations
results=[]
for imports in data:
    project=_Project({})
    frame=_Frame(project,dict(name='entry',params=[],body=[],python_scope=dict(kind='module',module='test')),())
    steps=[]
    issue=None
    try:
        for node in imports:
            frame.imported(node,steps)
    except _Unclosed as error:
        issue=error.issue['code']
    results.append(dict(issue=issue,operations=list(walk_operations(steps)),
        bindings={name:dict(kind=value.kind,value=value.value) for name,value in project.names.items()},
        requirement=project.graphics.requirement()))
print(json.dumps(results))
""",
        [
            [first, imported, alias],
            [imported],
            [first, namespace],
            [first, source],
            [first, route],
        ],
    )
    positive = results[0]
    assert positive["issue"] is None
    imports = [
        node
        for node in positive["operations"]
        if node["op"] == "graphics_projection_import"
    ]
    assert (
        len(imports) == 2
        and sum(node["op"] == "graphics_context" for node in positive["operations"])
        == 1
    )
    assert imports[0]["context"] == imports[1]["context"]
    assert {node["site"] for node in imports} == {
        "source/axes-import",
        "source/alias-import",
    }
    for node in imports:
        assert node["may_raise"] and set(node["effects"]) == {"IO", "MEM", "STATE"}
        assert {
            "module.registry",
            "graphics.projection_registry",
            "graphics.class.shared_state",
        } <= set(node["writes"])
        assert node["initialization"] == "import_once"
        assert node["projection_registration"] == "preserve_source_import_closure"
        assert node["import_binding"]["initial_registry"] == "pristine_or_verified"
    assert (
        positive["bindings"]["py:test:Axes3D"] == positive["bindings"]["py:test:Alias"]
    )
    assert positive["bindings"]["py:test:Axes3D"]["kind"] == "graphics_class"
    assert positive["requirement"]["profile"] == "pyplot-3.11.2"
    assert "graphics_projection_import" in positive["requirement"]["operations"]
    assert [item["issue"] for item in results[1:]] == [
        "python.graphics_context",
        "python.graphics_provider",
        "python.graphics_import_binding",
        "python.graphics_import_binding",
    ]


def test_immutable_products_preserve_types_identity_and_activation_boundaries():
    result = _layer(
        "normalization",
        """from scar.middle.normalization import arrays, products, values
from scar.middle.normalization.protocols import _Project, _Frame, _Unclosed
from scar.contract import walk_operations

def var(name):
    return dict(op='var',name=name)
def literal(value):
    return dict(op='literal',value=value)
def function(name,params=(),body=()):
    return dict(name=name,execution_kind='python_protocol',
        python_scope=dict(kind='function',module='test'),
        call_protocol=dict(kind='regular'),
        params=[dict(name=p,binding='positional') for p in params],body=list(body))
def aggregate(items,identity=None,kind='tuple'):
    return dict(op='aggregate',kind=kind,items=items,
        identity=identity or dict(kind='fresh'))

pack=function('pack',('a','b'),[
    dict(op='return',value=aggregate([var('a'),var('b')]))])
mutation=function('mutate',('pair',),[
    dict(op='assign',target=dict(op='index',base=dict(op='index',base=var('pair'),
        indices=[literal(0)]),indices=[literal(0)]),value=literal(9.0))])
caller=function('caller')
project=_Project({f['name']:f for f in (pack,mutation,caller)})
array=values.array_kind(1)
closed=project.compile('pack',(array,'numpy_f64'))
kind=project.result_kinds['pack']
assert values.product_members(kind)==(array,'numpy_f64')
assert closed['result']==dict(kind='product_ref',items=[values.wire(array),'f64'])
assert any(op['op']=='product_make' for op in walk_operations(closed['body']))
assert project.result_facts['pack']['product_members']==[{},{}]
assert arrays.may_write_external(project.compile('mutate',(kind,)),project.closed)

frame=_Frame(project,caller,())
facts=project.array_state.create((2,),[1.,2.])
frame.env['a']=values.Value(array,None,facts).at('input')
steps=[]
pair=arrays.capture_sequence(frame,aggregate([var('a'),var('a')]),steps)
members=products.elements(frame,{},pair,steps)
assert members[0].facts['array_allocation']==members[1].facts['array_allocation']
project.array_state.invalidate(members[0])
assert facts['array_allocation'] not in project.array_state.contents
returned=products.returned_facts(pair)
assert returned['product_members']==[dict(shape=(2,)),dict(shape=(2,))]
assert 'array_allocation' not in repr(returned) and 'literal_region' not in repr(returned)
frame.env['pair']=pair
made=sum(op['op']=='product_make' for op in walk_operations(steps))
forward=frame.call(dict(op='python_invoke',target=dict(op='python_builtin',name='tuple'),
    args=[var('pair')],keywords=[]),steps,False)
assert forward.kind==pair.kind
assert sum(op['op']=='product_make' for op in walk_operations(steps))==made
assert frame.truth(pair,{}).exact is True
length=frame.call(dict(op='python_invoke',target=dict(op='python_builtin',name='len'),
    args=[var('pair')],keywords=[]),steps,False)
assert length.exact==2
assert products.get(frame,{},pair,-1,steps).kind==array
for _ in range(2):
    called=frame.invoke_project({},'pack',[members[0],values.Value('numpy_f64',
        dict(op='literal',type='f64',value=4.0),dict(exact=4.0))],{},steps,False)
    assert called.facts['product_members']==[{},{}]

pool=dict(kind='constant_pool',pool='test-code',slot=1)
constant=arrays.capture_sequence(frame,aggregate([
    dict(op='unary',operator='usub',value=literal(1)),
    dict(op='binary',operator='add',left=literal(1),right=literal(1))],pool),steps)
assert constant.facts['constant_expr']['op']=='product_make'
assert all(item['op']=='literal' for item in constant.facts['constant_expr']['items'])
assert [item['value'] for item in constant.facts['constant_expr']['items']]==[-1,2]
shape=arrays.capture_sequence(frame,aggregate([literal(2),literal(3)]),steps)
zeros=arrays.construct(frame,{},values.Value('array_constructor','zeros'),[shape],{},steps,False)
assert zeros.facts['shape']==(2,3)
index_pair=arrays.capture_sequence(frame,aggregate([literal(1),literal(7.0)]),steps)
frame.env['target']=values.Value(array,None,dict(shape=(2,))).at('target')
first=len(steps)
frame.store(dict(op='unpack',items=[var('index'),dict(op='index',base=var('target'),
    indices=[var('index')])]),index_pair,steps)
ordered=steps[first:]
assert next(i for i,s in enumerate(ordered) if s['op']=='assign' and s['target']['name']=='index') < next(i for i,s in enumerate(ordered) if s['op']=='array_store')
unresolved=arrays.capture_sequence(frame,aggregate([var('a')],dict(kind='unresolved')),steps)
mutable=arrays.capture_sequence(frame,aggregate([var('a')],kind='list'),steps)
assert unresolved.kind=='tuple' and mutable.kind=='list'
for item in (unresolved,mutable):
    try:
        frame.scalar(item,{})
    except _Unclosed as error:
        assert error.issue['code']=='python.value_contract'
    else:
        raise AssertionError('an unresolved sequence escaped as a runtime product')
frame.env['local']=values.literal(1).at('local')
try:
    arrays.capture_sequence(frame,aggregate([var('local')],pool),steps)
except _Unclosed as error:
    assert error.issue['code']=='python.product_constant_pool'
else:
    raise AssertionError('a local exact value entered the source constant pool')
assert values.product_kind(('f64',))!=values.product_kind(('numpy_f64',))
assert values.wire(values.product_kind(('f64',)))==values.wire(values.product_kind(('numpy_f64',)))
different=function('different',('flag','number'),[dict(op='if',condition=var('flag'),
    then=[dict(op='return',value=aggregate([var('number')]))],
    **{'else':[dict(op='return',value=aggregate([literal(1.0)]))]})])
project.functions['different']=different
try:
    project.compile('different',('bool','numpy_f64'))
except _Unclosed as error:
    assert error.issue['code']=='python.result_contract'
else:
    raise AssertionError('wire-equal but logically different product returns merged')
print(json.dumps(dict(result=closed['result'])))
""",
    )
    assert result["result"]["kind"] == "product_ref"


def test_resource_flow_projects_copies_without_unrolling_or_placement_claims():
    array = dict(kind="ndarray", element="f64", rank=2)

    def integer(value):
        return dict(op="literal", type="i64", value=value)

    def variable(name, kind=array):
        return dict(op="var", name=name, type=kind)

    def create():
        return dict(op="array_create", type=array, shape=[integer(2), integer(3)])

    def copy(value, site):
        return dict(
            op="expr",
            value=dict(op="array_copy", type=value["type"], value=value, site=site),
        )

    body = [
        dict(op="decl", name="base", type=array, init=create()),
        dict(op="decl", name="alias", type=array, init=variable("base")),
        copy(variable("alias"), "known"),
        copy(
            dict(
                op="array_view",
                type={**array, "rank": 1},
                base=variable("base"),
                indices=[integer(0)],
            ),
            "view",
        ),
        copy(
            dict(
                op="array_reshape",
                type=array,
                value=variable("base"),
                shape=[integer(3), integer(2)],
                layout="C",
            ),
            "reshape",
        ),
        copy(variable("dynamic"), "dynamic"),
        dict(
            op="expr",
            value=dict(
                op="array_broadcast_copy",
                type=array,
                value=variable("base"),
                shape=[variable("n", "i64"), integer(3)],
                site="broadcast",
            ),
        ),
        dict(
            op="expr",
            value=dict(
                op="host_call",
                interface="fs.copy2",
                args=[
                    dict(op="literal", type="text", value=value)
                    for value in ("source", "unknown-destination")
                ],
                site="file",
            ),
        ),
        dict(
            op="expr",
            value=dict(
                op="call",
                callee="memcpy",
                args=[
                    variable("dst", "opaque"),
                    variable("src", "opaque"),
                    integer(16),
                ],
                site="native",
            ),
        ),
        dict(
            op="if",
            condition=variable("flag", "bool"),
            then=[
                dict(op="assign", target=variable("base"), value=variable("dynamic"))
            ],
            **{"else": []},
        ),
        copy(variable("base"), "joined"),
        dict(op="assign", target=variable("base"), value=create()),
        dict(
            op="while",
            condition=variable("flag", "bool"),
            body=[
                copy(variable("base"), "iteration"),
                dict(op="assign", target=variable("base"), value=variable("dynamic")),
            ],
        ),
        copy(variable("base"), "after-loop"),
    ]
    body[2]["value"]["source_evidence"] = dict(
        op="array_copy", type=array, value=variable("base"), site="historical"
    )
    inventory = [
        dict(id=identity, kind=kind)
        for identity, kind in (
            ("pool", "CPU_POOL"),
            ("cpu", "CPU"),
            ("dram", "DRAM"),
            ("mount", "STORAGE_MOUNT"),
            ("gpu", "GPU"),
            ("hbm", "DEVICE_MEMORY"),
        )
    ]
    module = dict(
        functions=[dict(name="main", body=body)],
        externs=[dict(name="memcpy", provider="libc")],
        resource_model=dict(resources=inventory),
    )
    shadowed = deepcopy(module)
    shadowed["functions"].append(dict(name="memcpy", body=[]))
    partial = dict(
        functions=[
            dict(
                name="unclosed",
                execution_kind="python_protocol",
                body=[copy(variable("dynamic"), "must-not-project")],
            )
        ],
        protocol_analysis=dict(
            closed_fragments=[
                dict(
                    symbol="closed",
                    completion="complete",
                    function=dict(name="closed", body=[copy(create(), "closed")]),
                )
            ],
            prefix_fragments=[
                dict(
                    symbol="prefix",
                    body=[copy(variable("dynamic"), "prefix")],
                    frontier=dict(body=[copy(variable("dynamic"), "not-a-prefix")]),
                )
            ],
        ),
        resource_model=dict(resources=inventory),
    )
    flows = _layer(
        "execution",
        """from scar.middle.resource_flow import analyze
from scar.contract import validate_resource_flow
flows=[analyze(data[0]),analyze(data[1]),analyze(data[2],partial=True)]
for flow,module in zip(flows,data):
    validate_resource_flow(flow,module['resource_model']['resources'])
assert flows[0] == analyze(data[0])
print(json.dumps(flows))
""",
        [module, shadowed, partial],
    )
    flow = flows[0]
    nodes = {node["id"]: node for node in flow["nodes"]}
    copies = {node["site"]: node for node in nodes.values() if node["kind"] == "COPY"}
    assert flow["coverage"] == dict(kind="canonical", functions=1, copies=10)
    assert copies["known"]["byte_count"] == dict(kind="constant", value=48)
    assert copies["view"]["byte_count"] == dict(kind="constant", value=24)
    assert copies["reshape"]["byte_count"] == dict(kind="constant", value=48)
    assert copies["native"]["destination_storage"] == "existing"
    assert copies["native"]["byte_count"] == dict(kind="constant", value=16)
    assert all(
        copies[site]["byte_count"]["kind"] == "operand_nbytes"
        for site in ("dynamic", "joined", "iteration", "after-loop")
    )
    assert copies["broadcast"]["byte_count"]["kind"] == "result_nbytes"
    assert copies["iteration"]["scope"][0]["kind"] == "while"
    assert (
        len([node for node in nodes.values() if node.get("site") == "iteration"]) == 1
    )
    source, destination = (
        nodes[copies["known"][role]] for role in ("source", "target")
    )
    assert source["region"]["storage"]["kind"] == "operand_storage"
    assert destination["region"]["storage"]["kind"] == "fresh_allocation"
    assert destination["logical_value"] == dict(
        kind="same_as", materialization=source["id"]
    )
    assert destination["ready"]["kind"] == "operation_complete"
    assert destination["location"]["resource_candidates"] == ["dram"]
    assert source["region"]["storage"]["origin"] == dict(
        owner="main", path=["body", 0, "init"]
    )
    assert all(
        nodes[copies[site]["source"]]["region"]["storage"]["origin"]
        == source["region"]["storage"]["origin"]
        for site in ("view", "reshape")
    )
    assert destination["region"]["storage"]["origin"] != (
        source["region"]["storage"]["origin"]
    )
    assert copies["known"]["dispatch"]["resource_candidates"] == ["pool"]
    file = nodes[copies["file"]["target"]]
    assert file["operand"] is None and file["region"]["path_resolution"]
    assert file["location"]["resource_candidates"] == ["mount"]
    assert not any(node.get("site") == "native" for node in flows[1]["nodes"])
    assert flows[2]["coverage"] == dict(kind="partial", functions=2, copies=2)
    assert {node["site"] for node in flows[2]["nodes"] if node["kind"] == "COPY"} == {
        "closed",
        "prefix",
    }


def test_resource_flow_preserves_local_product_fields_without_call_aliases():
    array = dict(kind="ndarray", element="f64", rank=1)
    product = dict(kind="product_ref", items=[array])

    def variable(name, kind=product):
        return dict(op="var", name=name, type=kind)

    def create():
        return dict(
            op="array_create",
            type=array,
            shape=[dict(op="literal", type="i64", value=2)],
        )

    def make(value):
        return dict(
            op="product_make",
            type=dict(kind="product_ref", items=[value["type"]]),
            items=[value],
            identity=dict(kind="fresh"),
            effects=["MEM", "STATE"],
        )

    def get(value):
        return dict(
            op="product_get",
            type=value["type"]["items"][0],
            base=value,
            index=0,
            effects=["STATE"],
        )

    def copy(value, site):
        return dict(
            op="expr", value=dict(op="array_copy", type=array, value=value, site=site)
        )

    alias, unknown = variable("alias"), variable("unknown")
    body = [
        copy(get(unknown), "parameter"),
        dict(op="decl", name="a", type=array, init=create()),
        copy(variable("a", array), "direct"),
        dict(op="decl", name="p", type=product, init=make(variable("a", array))),
        dict(op="decl", name="alias", type=product, init=variable("p")),
        dict(
            op="decl",
            name="nested",
            type=dict(kind="product_ref", items=[product]),
            init=make(alias),
        ),
        copy(
            get(get(variable("nested", dict(kind="product_ref", items=[product])))),
            "nested",
        ),
        copy(get(alias), "field"),
        dict(
            op="if",
            condition=variable("flag", "bool"),
            then=[dict(op="assign", target=alias, value=make(create()))],
            **{"else": []},
        ),
        copy(get(alias), "branch"),
        dict(op="assign", target=alias, value=unknown),
        copy(get(alias), "rebound"),
        dict(op="assign", target=alias, value=variable("p")),
        dict(
            op="while",
            condition=variable("flag", "bool"),
            body=[
                copy(get(alias), "loop"),
                dict(op="assign", target=alias, value=unknown),
            ],
        ),
        copy(get(alias), "after-loop"),
        dict(
            op="decl",
            name="returned",
            type=product,
            init=dict(op="call", callee="unknown", args=[], type=product),
        ),
        copy(get(variable("returned")), "call"),
    ]
    flow = _layer(
        "execution",
        """from scar.middle.resource_flow import analyze
from scar.contract import validate_resource_flow
flow=analyze(data)
validate_resource_flow(flow)
print(json.dumps(flow))
""",
        dict(
            functions=[
                dict(
                    name="main", params=[dict(name="unknown", type=product)], body=body
                )
            ]
        ),
    )
    nodes = {node["id"]: node for node in flow["nodes"]}
    copies = {node["site"]: node for node in nodes.values() if node["kind"] == "COPY"}
    origin = dict(owner="main", path=["body", 1, "init"])
    for site in ("direct", "nested", "field", "branch"):
        assert copies[site]["byte_count"] == dict(kind="constant", value=16)
        source = nodes[copies[site]["source"]]
        assert source["region"]["storage"]["origin"] == (
            None if site == "branch" else origin
        )
        assert "_members" not in source["region"]
    for site in ("parameter", "rebound", "loop", "after-loop", "call"):
        assert copies[site]["byte_count"]["kind"] == "operand_nbytes"
        assert nodes[copies[site]["source"]]["region"]["storage"]["origin"] is None
    assert len(copies) == 9 and copies["loop"]["scope"][0]["kind"] == "while"


def test_file_versions_preserve_stable_aliases_and_closed_writer_returns():
    def literal(value, kind="text"):
        return dict(op="literal", type=kind, value=value)

    def variable(name, scope=None, kind="text"):
        return dict(op="var", name=name, scope=scope, type=kind)

    def host(interface, *args, **extra):
        return dict(op="host_call", interface=interface, args=list(args), **extra)

    def effect(value):
        return dict(op="expr", value=value)

    stream_type = dict(kind="handle", resource="text_stream")
    stream = variable("stream", kind=stream_type)

    def writer(path, fact=None, closed=True):
        opened = host(
            "stream.open",
            path,
            literal("w"),
            literal("utf-8"),
            literal("universal"),
        )
        if fact is not None:
            opened["argument_facts"] = dict(args=[dict(type="text", exact=fact)])
        return [
            dict(op="decl", name="stream", type=stream_type, init=opened),
            effect(host("stream.write", stream, literal("constant\n"))),
            *([effect(host("stream.close", stream))] if closed else []),
        ]

    code = """from scar.middle.normalization.artifacts import analyze
from scar.contract import validate_artifact_flow
results=[analyze(module) for module in data]
for result in results:
    validate_artifact_flow(result['graph'])
print(json.dumps(results))
"""
    aliases = [
        dict(op="decl", name="first", type="text", init=variable("left", "global")),
        dict(op="decl", name="spare", type="text", init=variable("right", "global")),
        dict(op="decl", name="unrelated", type="i64", init=literal(0, "i64")),
        dict(op="decl", name="alias", type="text"),
        dict(op="assign", target=variable("alias"), value=variable("first")),
    ]
    no_file = dict(
        name="numeric",
        params=[],
        result="unit",
        body=[dict(op="while", condition=variable("again", kind="bool"), body=[])],
    )

    def alias_module(barrier, rebound=()):
        return dict(
            protocol_analysis=dict(single_bind_globals=["left", "right"]),
            functions=[
                dict(
                    name="main",
                    params=[],
                    result="unit",
                    body=[
                        *aliases,
                        effect(dict(op="call", callee=barrier, args=[])),
                        *rebound,
                        *writer(variable("alias"), "left.txt"),
                        *writer(variable("spare"), "right.txt"),
                    ],
                ),
                no_file,
            ],
        )

    rebound = dict(
        op="assign", target=variable("alias"), value=variable("mutable", "global")
    )
    modules = [alias_module(name) for name in ("numeric", "opaque")]
    modules += [
        alias_module("numeric", [rebound]),
        alias_module(
            "numeric",
            [
                dict(
                    op="if",
                    condition=variable("flag", kind="bool"),
                    then=[rebound],
                    **{"else": []},
                )
            ],
        ),
        alias_module(
            "numeric",
            [
                dict(
                    op="while", condition=variable("again", kind="bool"), body=[rebound]
                )
            ],
        ),
    ]

    def returned_writer(closed):
        return dict(
            functions=[
                dict(
                    name="main",
                    params=[],
                    result="unit",
                    body=[
                        effect(
                            dict(
                                op="call", callee="writer", site="main/writer", args=[]
                            )
                        ),
                        effect(
                            host("fs.copy_file", literal("source"), literal("copy"))
                        ),
                    ],
                ),
                dict(
                    name="writer",
                    params=[],
                    result=stream_type,
                    body=[
                        *writer(literal("source"), closed=closed),
                        dict(op="return", value=stream),
                    ],
                ),
            ]
        )

    results = _layer(
        "normalization", code, [*modules, returned_writer(True), returned_writer(False)]
    )
    for index, result in enumerate(results[: len(modules)]):
        names = {
            node["name"]
            for node in result["graph"]["nodes"]
            if node["kind"] == "FILE_LOCATOR"
        }
        assert "right.txt" in names
        assert ("left.txt" in names) == (index < 2)
    closed, opened = results[-2:]
    assert any(
        edge["kind"] == "COPIES_FILE_VERSION" for edge in closed["graph"]["edges"]
    )
    assert all(row["normal_path_closed"] for row in closed["file_state_transitions"])
    assert not any(
        edge["kind"] == "COPIES_FILE_VERSION" for edge in opened["graph"]["edges"]
    )
    assert any(
        not row["normal_path_closed"] for row in opened["file_state_transitions"]
    )


def test_directory_targets_and_file_phis_follow_summarized_loop_exit():
    def literal(value):
        return dict(
            op="literal",
            type={str: "text", int: "i64", type(None): "unit"}[type(value)],
            value=value,
        )

    def host(interface, *args):
        return dict(op="host_call", interface=interface, args=list(args))

    def effect(interface, *args):
        return dict(op="expr", value=host(interface, *args))

    stream = dict(op="var", name="stream")

    def writer(path, contents):
        return [
            dict(
                op="decl",
                name="stream",
                init=host(
                    "stream.open",
                    literal(path),
                    literal("w"),
                    literal("utf-8"),
                    literal("universal"),
                ),
            ),
            effect("stream.write", stream, literal(contents)),
            effect("stream.close", stream),
        ]

    reader = dict(op="var", name="reader")
    module = dict(
        functions=[
            dict(
                name="main",
                result="unit",
                params=[dict(name=name, type="bool") for name in ("again", "flag")],
                body=[
                    dict(
                        op="while",
                        condition=dict(op="var", name="again", type="bool"),
                        body=[*writer("loop-only", "iteration"), dict(op="break")],
                    ),
                    effect("fs.mkdir", literal("out"), literal(0o777), literal(None)),
                    {
                        "op": "if",
                        "condition": dict(op="var", name="flag", type="bool"),
                        "then": writer("input", "left"),
                        "else": writer("input", "right"),
                    },
                    effect("fs.copy2", literal("input"), literal("out")),
                    effect("process.chdir", literal("out")),
                    dict(
                        op="decl",
                        name="reader",
                        init=host(
                            "stream.open",
                            literal("input"),
                            literal("r"),
                            literal("utf-8"),
                            literal("universal"),
                        ),
                    ),
                    effect("stream.read", reader, literal(-1)),
                    effect("stream.close", reader),
                ],
            )
        ]
    )
    result = _layer(
        "normalization",
        """from scar.middle.normalization.artifacts import analyze
from scar.contract import validate_artifact_flow
result=analyze(data)
validate_artifact_flow(result['graph'])
print(json.dumps(result))
""",
        module,
    )
    graph = result["graph"]
    nodes = {node["id"]: node for node in graph["nodes"]}
    (directory,) = [
        node
        for node in nodes.values()
        if node["kind"] == "DIRECTORY_TEMPLATE" and node["form"] == "created"
    ]
    assert directory["parent"] is not None
    (copy,) = [edge for edge in graph["edges"] if edge["kind"] == "COPIES_FILE_VERSION"]
    source, target = nodes[copy["source"]], nodes[copy["target"]]
    assert source["form"] == "phi" and len(set(source["incoming_versions"])) == 2
    assert source["value"] is None and target["value"] is None
    copied_locator = nodes[target["locator"]]
    assert copied_locator["directory"] == directory["id"]
    assert copied_locator["name"] == "input"
    (read,) = [edge for edge in graph["edges"] if edge.get("role") == "stream_read"]
    assert read["source"] == target["id"]
    assert any(
        row.get("iterations") == "not_unrolled"
        and row["condition"] == "structured_loop_normal_exit"
        for row in result["boundaries"]
    )
    assert sum(node.get("operation") == "while" for node in nodes.values()) == 1
    assert not any(node.get("literal_text") == "iteration" for node in nodes.values())


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


def test_native_stream_input_ports_preserve_conditional_buffer_versions():
    from scar.contract import ARTIFACT_FLOW

    stream = dict(
        kind="library", provider="cxx_standard", name="basic_ifstream", args=["char"]
    )
    pointer = dict(kind="ptr", to="char")

    def var(name):
        return dict(op="var", name=name)

    def scalar(n):
        return dict(op="literal", type="i64", value=n)

    def member(site, name, args, interface, api, roles, defaults, **contract):
        constructor = name is None
        return dict(
            op="library_construct" if constructor else "library_call",
            **(
                {"type": stream}
                if constructor
                else {"name": name, "receiver": var("alias")}
            ),
            site=site,
            provider="cxx_standard",
            args=args,
            artifact_effect=dict(
                semantics_version=1,
                interface=interface,
                operands=roles,
                defaults=defaults,
                contract=dict(
                    semantics_version=1,
                    source_profile="cxx_standard_ifstream_char_v1",
                    source_interface=api,
                    handle="result" if constructor else "receiver",
                    **contract,
                ),
                obligations=["existing_stream_locale_and_state"],
            ),
        )

    opened = member(
        "opened",
        None,
        [var("path")],
        "stream.open",
        "std::basic_ifstream::basic_ifstream",
        {"path": ["args", 0]},
        {"mode": {"kind": "ios_openmode", "flag": "in"}},
        result="stream_object",
        success_condition="filebuf_open_succeeds",
        effective_mode="mode_or_in",
        filesystem_effect="depends_on_effective_mode",
    )

    def extract(site, count, delim=None):
        args = [dict(op="array_decay", value=var("line")), scalar(count)]
        roles = dict(stream=["receiver"], dst=["args", 0], count=["args", 1])
        if delim is not None:
            roles["delimiter"] = ["args", 2]
            args.append(scalar(delim))
        return member(
            site,
            "getline",
            args,
            "stream.getline_into",
            "std::basic_istream::getline",
            roles,
            {} if delim is not None else {"delimiter": {"kind": "widen_newline"}},
            result="receiver_reference",
            delimiter="extract_without_storing",
            count_unit="char_elements_including_terminator_space",
            success_condition="input_sentry_and_extraction_state",
        )

    close = member(
        "closed",
        "close",
        [],
        "stream.close",
        "std::basic_ifstream::close",
        {"stream": ["receiver"]},
        {},
        result="void",
        success_condition="filebuf_close_succeeds",
    )
    forged = dict(extract("forged", 8), provider="other_library")

    def addr(value):
        return dict(op="unary", operator="addr", value=value)

    def deref(value):
        return dict(op="unary", operator="deref", value=value)

    def read_at(site, receiver):
        return dict(op="expr", value=dict(extract(site, 8), receiver=receiver))

    stream_pointer = dict(kind="ptr", to=stream)
    indirect = deref(deref(var("pp")))
    body = [
        dict(
            op="decl", name="line", type={"kind": "array", "of": "char", "shape": [8]}
        ),
        dict(op="decl", name="source", type=stream, init=opened),
        dict(
            op="decl",
            name="alias",
            type=dict(kind="ref", to=stream),
            init=var("source"),
        ),
        *[
            dict(op="expr", value=v)
            for v in (
                extract("first", 8),
                extract("second", 0, 59),
                close,
                extract("after", 8),
                forged,
            )
        ],
        dict(op="decl", name="a", type=stream, init=dict(opened, site="open_a")),
        dict(op="decl", name="b", type=stream, init=dict(opened, site="open_b")),
        dict(op="decl", name="p", type=stream_pointer, init=addr(var("a"))),
        read_at("one_pointer", deref(var("p"))),
        dict(
            op="decl",
            name="pp",
            type=dict(kind="ptr", to=stream_pointer),
            init=addr(var("p")),
        ),
        dict(op="assign", target=var("p"), value=addr(var("b"))),
        read_at("two_pointers", indirect),
        dict(op="expr", value=dict(close, site="unknown_close", receiver=indirect)),
        read_at("after_unknown_close", var("a")),
    ]
    flow = _layer(
        "normalization",
        """from scar.contract import validate_artifact_flow
from scar.middle.normalization.buffers import analyze_buffers
graph=analyze_buffers(data['module'],data['flow'])['graph']
validate_artifact_flow(graph)
print(json.dumps(graph))
""",
        dict(
            module=dict(
                functions=[
                    dict(
                        name="read_records",
                        params=[dict(name="path", type=pointer)],
                        body=body,
                    )
                ]
            ),
            flow=document(ARTIFACT_FLOW, nodes=[], edges=[]),
        ),
    )
    nodes = {r["id"]: r for r in flow["nodes"]}
    by_site = {
        r["site"]: r["id"]
        for r in nodes.values()
        if r["kind"] == "CANONICAL_OPERATION" and r["site"]
    }
    edges = flow["edges"]
    expr = {
        nodes[r["operation"]]["site"]: r
        for r in nodes.values()
        if r["kind"] == "BYTE_EXPRESSION"
    }
    assert set(expr) == {
        "first",
        "second",
        "after",
        "one_pointer",
        "two_pointers",
        "after_unknown_close",
    }
    assert {expr[site]["stream_object"] for site in ("first", "second", "after")} == {
        by_site["opened"]
    }
    assert expr["one_pointer"]["stream_object"] == by_site["open_a"]
    assert expr["two_pointers"]["stream_object"] is None
    assert expr["after_unknown_close"]["stream_object"] == by_site["open_a"]
    assert expr["first"]["defaults"]["delimiter"] == {"kind": "widen_newline"}
    assert expr["second"]["integer_inputs"] == {"count": 0, "delimiter": 59}
    assert expr["first"]["obligations"] == ["existing_stream_locale_and_state"]
    first, second, after = [
        next(
            nodes[e["source"]]
            for e in edges
            if e["kind"] == "SUPPLIES_STREAM_BYTES"
            and e["target"] == expr[site]["id"]
            and e["conditional"]
        )
        for site in ("first", "second", "after")
    ]
    locator = nodes[first["locator"]]
    assert locator["path"] is None and locator["operand"] == ["args", 0]
    assert nodes[locator["operand_port"]]["operation"] == "var"
    assert nodes[locator["operand_value"]]["kind"] == "BUFFER_VERSION"
    assert first["id"] != second["id"] and second["locator"] == first["locator"]
    assert first["value"] is None and first["historical_instance"] == "not_established"
    assert (
        after["locator"] != first["locator"] and nodes[after["locator"]]["path"] is None
    )
    assert any(
        e["kind"] == "MAY_CLOSE_FILE"
        and e["source"] == by_site["closed"]
        and e["target"] == first["locator"]
        for e in edges
    )
    for site in ("two_pointers", "after_unknown_close"):
        requirements = [
            nodes[e["source"]]
            for e in edges
            if e["kind"] == "SUPPLIES_STREAM_BYTES" and e["target"] == expr[site]["id"]
        ]
        assert requirements and all(
            nodes[r["locator"]]["identity_scope"] == "unresolved_stream_association"
            for r in requirements
        )
    writes = [r for r in nodes.values() if r.get("form") == "conditional_stream_write"]
    assert {r["expression"] for r in writes} == {
        r["id"] for s, r in expr.items() if s != "second"
    }
    assert all(nodes[r["locator"]]["name"] == "line" for r in writes)
    assert not any(
        e["kind"] == "HAS_C_STRING_PAYLOAD" and e["source"] in {r["id"] for r in writes}
        for e in edges
    )

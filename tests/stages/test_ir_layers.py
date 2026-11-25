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


def test_binding_rejects_module_code_from_another_source_revision(tmp_path):
    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle.binding import bind

    source = tmp_path / "entry.py"
    source.write_text("name = 'sample'\nprint(name)\n")
    folder = tmp_path / "graph"
    build(source, folder)
    graph = json.loads((folder / "graph.json").read_bytes())
    unit = graph["units"][0]
    fragment = read_member(folder, unit["graph"])
    scope = next(
        fn["signature"]["python_scope"]
        for fn in fragment["functions"]
        if fn["signature"].get("python_scope", {}).get("source_code")
    )
    scope["source_code"]["source_sha256"] = "0" * 64
    unit["graph"] = member(folder, "units", fragment)
    source.unlink()
    with pytest.raises(ValueError, match="captured compilation unit"):
        bind(graph, folder)


def test_binding_checks_import_result_against_its_source_publication(tmp_path):
    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle.binding import bind

    source = tmp_path / "entry.py"
    source.write_text("import math as library\nlibrary.sin(1)\n")
    folder = tmp_path / "graph"
    build(source, folder)
    graph = json.loads((folder / "graph.json").read_bytes())
    unit = next(u for u in graph["units"] if u["path"] == "entry.py")
    fragment = read_member(folder, unit["graph"])
    declaration = next(
        node
        for fn in fragment["functions"]
        for node in fn["nodes"]
        if node["op"] == "python_import" and node["module"] == "math"
    )
    source.unlink()
    bind(graph, folder)
    expression = next(
        node["source_expression"]
        for fn in fragment["functions"]
        for node in fn["nodes"]
        if "source_expression" in node
    )
    first, last = expression["instructions"][0], expression["instructions"][-1]
    offset = first["offset"]
    first["offset"] = last["offset"]
    unit["graph"] = member(folder, "units", fragment)
    with pytest.raises(ValueError, match="source instruction position"):
        bind(graph, folder)
    first["offset"] = offset
    declaration["source_import_store"]["store"]["name"] = "math"
    unit["graph"] = member(folder, "units", fragment)
    with pytest.raises(ValueError, match="source import operands"):
        bind(graph, folder)


def test_binding_matches_stack_consumers_to_sealed_graph_ports(tmp_path):
    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle.binding import bind

    entry = tmp_path / "entry.py"
    entry.write_text(
        'label = "left" + "right"\n'
        "pool = [1, True, 'tail', None]\n"
        "value = factory(label)\n"
        "del value, label\n"
        "raise failure() from cause\n"
    )
    folder = tmp_path / "graph"
    build(entry, folder)
    graph = json.loads((folder / "graph.json").read_bytes())
    unit = graph["units"][0]
    payload = read_member(folder, unit["graph"])
    nodes = next(
        fn["nodes"]
        for fn in payload["functions"]
        if fn["signature"].get("python_scope", {}).get("kind") == "module"
    )
    entry.unlink()
    bind(graph, folder)
    folded = next(n for n in nodes if "source_constant_store" in n)
    root = nodes[folded["value"]["port"]]
    leaf = nodes[root["left"]["port"]]
    rows = {
        n["op"]: n["source_expression"]["instructions"]
        for n in nodes
        if "source_expression" in n
    }
    display = next(
        n for n in nodes if n.get("op") == "aggregate" and n.get("kind") == "list"
    )
    items = display["items"]
    for record, field, changed, error in (
        (leaf, "value", "changed", "folded constant"),
        (display, "items", [items[1], items[0], *items[2:]], "ordered display items"),
        (rows["assign"][-1], "name", "label", "expression store"),
        (rows["unbind"][0], "name", "label", "ordered targets"),
        (rows["raise"][-1], "oparg", 1, "raise terminator"),
    ):
        original = record[field]
        record[field] = changed
        unit["graph"] = member(folder, "units", payload)
        with pytest.raises(ValueError, match=error):
            bind(graph, folder)
        record[field] = original


def test_binding_checks_branch_successors_and_binary_dispatch(tmp_path):
    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle.binding import bind

    source = tmp_path / "entry.py"
    source.write_text("if not pick():\n    first()\nvalue = left + right\nlater()\n")
    folder = tmp_path / "graph"
    build(source, folder)
    index = json.loads((folder / "graph.json").read_bytes())
    unit = index["units"][0]
    fragment = read_member(folder, unit["graph"])
    nodes = next(
        f["nodes"]
        for f in fragment["functions"]
        if f["signature"].get("python_scope", {}).get("kind") == "module"
    )
    branch = next(n["source_expression"] for n in nodes if n["op"] == "if")
    assignment = next(n["source_expression"] for n in nodes if n["op"] == "assign")
    binary = next(i for i in assignment["instructions"] if i["opcode"] == "BINARY_OP")
    source.unlink()
    bind(index, folder)
    for obj, key, changed, message in (
        (
            branch["instructions"][-1],
            "target",
            branch["branch"]["then"]["offset"],
            "jump target",
        ),
        (branch["branch"]["else"], "statement_position", [4, 4, 0, 7], "continuation"),
        (binary, "oparg", 10, "binary instruction"),
        (binary, "operator", "sub", "binary instruction"),
    ):
        original = obj[key]
        obj[key] = changed
        unit["graph"] = member(folder, "units", fragment)
        with pytest.raises(ValueError, match=message):
            bind(index, folder)
        obj[key] = original


def test_function_construction_uses_the_declared_child_code(tmp_path):
    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle.binding import bind

    source = tmp_path / "entry.py"
    source.write_text(
        "def first(arg):\n    return arg\ndef second():\n    return 2\ndef evaluated(x=first(1)):\n    return x\n"
    )
    folder = tmp_path / "graph"
    build(source, folder)
    index = json.loads((folder / "graph.json").read_bytes())
    unit = index["units"][0]
    payload = read_member(folder, unit["graph"])
    module = next(
        f
        for f in payload["functions"]
        if f["signature"].get("python_scope", {}).get("kind") == "module"
    )
    first, second, evaluated = [
        n for n in module["nodes"] if n["op"] == "python_define"
    ]
    assert "source_definition" not in evaluated
    assert first["source_definition"]["child"]["name"] == "first"
    assert second["source_definition"]["child"]["name"] == "second"
    source.unlink()
    bind(index, folder)
    first["source_definition"]["child"] = second["source_definition"]["child"]
    unit["graph"] = member(folder, "units", payload)
    with pytest.raises(ValueError, match="child code"):
        bind(index, folder)


def test_loader_owns_result_separately_from_conditional_source_body(tmp_path):
    from scar.frontend import build
    from scar.middle.binding import bind
    from scar.middle.normalization.host_protocols import ImportOccurrences

    entry = tmp_path / "entry.py"
    entry.write_text("import helper as loaded\n")
    child = tmp_path / "helper.py"
    child.write_text("tag = 'source'\n")
    folder = tmp_path / "graph"
    build(entry, folder)
    graph = json.loads((folder / "graph.json").read_bytes())
    entry.unlink()
    child.unlink()
    module, issues, _ = bind(graph, folder)
    assert not issues
    functions = {f["name"]: f for f in module["functions"]}
    node = functions["__main__"]["body"][0]
    ports = node["loader_ports"]
    receipt = node["source_import_store"]
    assert ports["site"] == node["site"]
    assert ports["dispatch"] == {
        "instruction": receipt["import_name"]["offset"],
        "result": "import_result",
    }
    assert ports["binding"] == {
        "instruction": receipt["store"]["offset"],
        "input": "import_result",
        "on_import_exception": "skip",
    }
    body = ports["body"]
    assert body["selection"] == "each_actual_matching_code_activation"
    assert body["completion"] == "return_to_loader_or_propagate"
    (candidate,) = body["candidates"]
    assert candidate["initializer"] == node["initializers"][0]
    child_code = functions[candidate["initializer"]]["python_scope"]["source_code"]
    assert candidate["code_sha256"] == child_code["code_sha256"]
    assert candidate["source_key"] == child_code["source_key"]
    ledger = ImportOccurrences(functions).get("__main__", node)
    assert ledger["loader_ports"] == ports and ledger["activation"] == "unvisited"


def test_binding_rejects_exception_edge_to_a_different_handler(tmp_path):
    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle.binding import bind

    source = tmp_path / "entry.py"
    source.write_text(
        "try:\n    action()\nexcept ValueError as err:\n    first(err)\nexcept TypeError:\n    raise\n"
    )
    folder = tmp_path / "graph"
    build(source, folder)
    graph = json.loads((folder / "graph.json").read_bytes())
    unit = graph["units"][0]
    payload = read_member(folder, unit["graph"])
    source.unlink()
    bind(graph, folder)
    proof = next(
        n["source_exception_handler"]
        for fn in payload["functions"]
        for n in fn["nodes"]
        if n.get("source_exception_handler", {}).get("arm_index") == 0
    )
    proof["branch"]["target"] = proof["body_offset"]
    unit["graph"] = member(folder, "units", payload)
    with pytest.raises(ValueError, match="dispatch edges"):
        bind(graph, folder)


def test_middle_products_reject_out_of_order_passes_and_keep_inputs_owned(tmp_path):
    canonical = dict(
        entry="main",
        functions=[
            dict(
                name="main",
                params=[],
                result="i32",
                body=[
                    dict(
                        op="return",
                        value=dict(
                            op="binary",
                            operator="add",
                            type="i32",
                            left=dict(op="literal", type="i32", value=3),
                            right=dict(op="literal", type="i32", value=4),
                        ),
                    )
                ],
                state_contract=dict(private_scalars=[], scalars={}),
            )
        ],
        globals=[],
        externs=[],
        records=[],
    )
    rewritten = _layer(
        "rewrites",
        """from copy import deepcopy
from scar.middle.rewrites import rewrite
source=product(data,'M2')
before=deepcopy(source)
try: rewrite(data)
except ValueError: pass
else: raise AssertionError('unlabelled program accepted')
wrong=product(dict(data,declarations={},abi_environment={}),'M1')
try: rewrite(wrong)
except ValueError: pass
else: raise AssertionError('M1 bypassed M2')
output,changes=rewrite(source)
assert source==before and output['functions'][0] is not source['functions'][0]
assert output['functions'][0]['state_contract'] is not source['functions'][0]['state_contract']
assert output['functions'][0]['body'][0]['value']['value']==7
assert changes[0]['rule']=='constant_fold'
partial=product(data,'M2',complete=False)
skipped,changes=rewrite(partial)
assert not changes and skipped['layer_contract']['completion']=='partial'
assert skipped['functions']==data['functions']
print(json.dumps(output))
""",
        canonical,
    )
    result = _layer(
        "execution",
        """from pathlib import Path
from scar.middle.execution import prepare,seal
source=data['module']
try: prepare(product(source,'M2'))
except ValueError: pass
else: raise AssertionError('M2 bypassed M3')
try: prepare(product(source,'M3',complete=False))
except ValueError: pass
else: raise AssertionError('partial program prepared for execution')
module,facts=prepare(source)
assert source['layer_contract']['phase']=='M3'
assert module['layer_contract']['phase']=='M4'
try: seal(source,Path(data['out']),status='READY',graph_sha256=None,contract={})
except ValueError: pass
else: raise AssertionError('unprepared M3 product sealed')
path=seal(module,Path(data['out']),status='READY',graph_sha256=None,contract={})
print(json.dumps(dict(path=str(path))))
""",
        dict(module=rewritten, out=str(tmp_path / "execution")),
    )
    sealed = json.loads(Path(result["path"]).read_bytes())
    assert sealed["module"]["layer_contract"]["phase"] == "M4"
    assert sealed["status"] == "READY"


def test_entry_environment_stays_observed_and_import_reads_keep_each_binding(tmp_path):
    import shutil

    from scar.contract import read_member, walk_operations
    from scar.frontend import build

    source = tmp_path / "entry.py"
    source.write_text(
        "import os\nalias = os\nimport time\n"
        "print(os.getcwd())\nprint(alias.path.exists('.'))\n"
    )
    request = tmp_path / "request.json"
    write_json(
        request,
        dict(
            source_root=".",
            root_image="driver",
            images=[
                dict(
                    id="driver",
                    entry="__main__",
                    output="driver",
                    source=source.name,
                    entry_environment=dict(
                        kind="standalone_script",
                        interpreter=sys.executable,
                        flags=["-B"],
                    ),
                )
            ],
        ),
    )
    folder = tmp_path / "graph"
    build(request, folder)
    graph = json.loads((folder / "graph.json").read_bytes())
    capture = graph["source_facts"]["python_entry_environment"]
    graph["current_image"] = "driver"
    source.unlink()
    bound = _layer(
        "binding",
        """from pathlib import Path
from scar.middle.binding import bind
module,issues,count=bind(data['graph'],Path(data['folder']))
assert not issues
assert module['python_entry_environment']['analysis']['confidence']=='observed'
bad=json.loads(json.dumps(data['graph']))
bad['source_facts']['python_entry_environment']['analysis']['application_initial_cache']='proven'
try: bind(bad,Path(data['folder']))
except ValueError: pass
else: raise AssertionError('compiler cache granted application initial state')
print(json.dumps(module))
""",
        dict(graph=graph, folder=str(folder)),
    )
    assert bound["python_entry_environment"]["analysis"] == capture["analysis"]
    shutil.rmtree(folder)
    normalized = _layer(
        "normalization",
        """from scar.middle.normalization import normalize
module,issues=normalize(data)
assert not issues
print(json.dumps(module))
""",
        bound,
    )
    imports = normalized["source_imports"]
    assert imports["entry_environment"] == bound["python_entry_environment"]
    assert (
        imports["entry_environment"]["analysis"]["deployment_equivalence"] == "unproven"
    )
    guards = [
        node
        for function in normalized["functions"]
        for node in walk_operations(function)
        if node["op"] == "module_import_preserves" and node["module"] == "time"
    ]
    assert len(guards) == 1
    guard = guards[0]
    assert guard["occurrence_effect_coverage"] == "MODELED"
    producers = [
        n
        for fn in normalized["functions"]
        for n in walk_operations(fn["body"])
        if n.get("source_import_occurrence") == guard["occurrence"]
    ]
    assert len(producers) == 1
    assert producers[0]["op"] == "module_namespace_initialize"
    assert producers[0]["state_capture"]["occurrence"] == guard["occurrence"]
    requirement = guard["checks"]["__main__"]["namespace"]
    assert {read["member"] for read in requirement["binding_reads"]} == {"os", "alias"}
    uses = {
        read["member"]: read["operand_use"] for read in requirement["binding_reads"]
    }
    assert [edge["name"] for edge in uses["os"]["attributes"]] == ["getcwd"]
    assert [edge["name"] for edge in uses["alias"]["attributes"]] == ["path", "exists"]
    assert all(use["consumer"]["role"] == "call_target" for use in uses.values())
    assert all(
        use["consumer"]["site"] != use["attributes"][-1]["site"]
        for use in uses.values()
    )
    assert requirement["value_state_projection"] == "unproven"
    assert (
        requirement["requires"] == "same_namespace_bindings_and_reachable_value_state"
    )
    # A launch declaration plus a compiler-process observation still grants no
    # native import-effect supplier. This must remain an actionable B1 refusal.
    plan = tmp_path / "plan"
    _layer(
        "execution",
        """from pathlib import Path
from scar.middle.execution import prepare,seal
module,_=prepare(product(data['module'],'M3'))
seal(module,Path(data['out']),status='READY',graph_sha256='captured',contract={})
print(json.dumps({}))
""",
        dict(module=normalized, out=str(plan)),
    )
    ir = json.loads((plan / "exec-ir.json").read_bytes())
    assert read_member(plan, ir["module"]["source_imports"]) == imports
    from scar.backend.planning import schedule_ir

    schedule_ir(plan, tmp_path / "schedule", policy="serial")
    physical = json.loads((tmp_path / "schedule/physical-plan.json").read_bytes())
    assert physical["status"] == "INCOMPLETE"
    assert any(
        issue["requirement"] == "module_registry" for issue in physical["issues"]
    )


def test_external_source_modules_keep_lookup_order_and_offline_stage_products(tmp_path):
    from base64 import urlsafe_b64encode
    from hashlib import sha256
    import shutil

    from scar.contract import read_member

    source = tmp_path / "source"
    site = source / ".sdk/site-packages"
    package = site / "graph_source_fixture"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        'print("package-init")\nknown = 17\n'
        'def __getattr__(name):\n    print("lookup", name)\n    return 23\n'
    )
    (package / "known.py").write_text('print("unexpected-child")\n')
    metadata = site / "graph_source_fixture-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: graph-source-fixture\nVersion: 1.0\n"
    )
    (metadata / "top_level.txt").write_text("graph_source_fixture\n")
    records = []
    for path in sorted(site.rglob("*")):
        if path.is_file():
            raw = path.read_bytes()
            digest = urlsafe_b64encode(sha256(raw).digest()).decode().rstrip("=")
            records.append(f"{path.relative_to(site)},sha256={digest},{len(raw)}")
    (metadata / "RECORD").write_text("\n".join(records) + "\n")
    main = source / "main.py"
    main.write_text(
        "import graph_source_fixture as p\nimport graph_source_fixture as again\n"
        "print(p.known)\nprint(p.missing)\n"
        "from graph_source_fixture import known\nprint(known)\n"
    )
    guarded = source / "guarded.py"
    guarded.write_text("from graph_source_fixture import missing\nprint(missing)\n")
    ordinary = source / "ordinary.py"
    ordinary.write_text(main.read_text().partition("from graph_source_fixture")[0])
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join((str(site), str(Path(scar.__file__).parents[1]))),
        PYTHONDONTWRITEBYTECODE="1",
    )
    expected = subprocess.run(
        [sys.executable, "-B", main],
        env=env,
        capture_output=True,
        check=True,
    ).stdout
    assert expected == b"package-init\n17\nlookup missing\n23\n17\n"
    from_expected = subprocess.run(
        [sys.executable, "-B", guarded],
        env=env,
        capture_output=True,
        check=True,
    ).stdout
    assert from_expected == b"package-init\nlookup missing\nlookup missing\n23\n"
    ordinary_output = subprocess.check_output([sys.executable, "-B", ordinary], env=env)
    assert ordinary_output == b"package-init\n17\nlookup missing\n23\n"
    command = """
import sys
phase, *args = sys.argv[1:]
forbidden = tuple('scar.' + p for p in ('frontend', 'middle', 'backend') if p != phase)
class Boundary:
    def find_spec(self, name, path=None, target=None):
        if any(name == p or name.startswith(p + '.') for p in forbidden):
            raise RuntimeError('stage import violation ' + name)
sys.meta_path.insert(0, Boundary())
from scar.cli import main
raise SystemExit(main(args))
"""

    def run(phase, *args, returncode=0):
        p = subprocess.run(
            [sys.executable, "-B", "-c", command, phase, *map(str, args)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=90,
        )
        assert p.returncode == returncode, p.stderr.decode(errors="replace")
        return json.loads(p.stdout)

    graph = tmp_path / "graph"
    negative = tmp_path / "from-graph"
    bound = tmp_path / "bound-graph"
    run("frontend", "build", "--entry", main, "--out", bound)
    run("frontend", "build", "--entry", ordinary, "--out", graph)
    run("frontend", "build", "--entry", guarded, "--out", negative)
    program = json.loads((graph / "graph.json").read_bytes())
    external = [u for u in program["units"] if u["path"].startswith("@python/")]
    assert external
    flow = read_member(graph, program["artifact_flow"])
    assert any(
        n.get("kind") == "FILE_LOCATOR"
        and n.get("path") == str(package / "__init__.py")
        for n in flow["nodes"]
    )
    assert not any(
        n.get("kind") == "FILE_LOCATOR" and "/@python/" in n.get("path", "")
        for n in flow["nodes"]
    )
    shutil.rmtree(source)
    plan, delivery = tmp_path / "plan", tmp_path / "delivery"
    assert run("middle", "plan", "--graph", graph, "--out", plan)["status"] == "READY"
    ir = json.loads((plan / "exec-ir.json").read_bytes())
    imports = read_member(plan, ir["module"]["source_imports"])
    assert imports["scope"] == "source_import_occurrences"
    assert any(row["activation"] == "visited" for row in imports["occurrences"])
    assert (
        json.loads((plan / "plan.json").read_bytes())["source_imports"]
        == ir["module"]["source_imports"]
    )
    functions = [read_member(plan, ref) for ref in ir["functions"]]
    assert any("__getattr__" in f["name"] for f in functions)
    assert not any("graph_source_fixture.known:<init>" in f["name"] for f in functions)
    shutil.rmtree(graph)
    result = run("backend", "compile", "--ir", plan, "--out", delivery)
    physical = json.loads((delivery / "physical-plan.json").read_bytes())
    audit = physical["source_import_effects"]
    assert audit["coverage"] == "INCOMPLETE"
    assert audit["activation_counts"]["visited"] > 0
    assert read_member(delivery, audit["ledger"]) == imports
    shutil.rmtree(plan)
    actual = subprocess.run([result["executable"]], capture_output=True, check=True)
    assert actual.stdout == ordinary_output and not actual.stderr
    bound_plan = tmp_path / "bound-plan"
    run("middle", "plan", "--graph", bound, "--out", bound_plan, returncode=2)
    [issue] = json.loads((bound_plan / "plan.json").read_bytes())["issues"]
    assert issue["code"] == "python.module_attribute_state"
    evidence = issue["source_evidence"]
    assert evidence["member"] == "known" and evidence["graph_binding"] == "bound"
    assert (
        evidence["child_candidate"]["initializer"]
        == "py:graph_source_fixture.known:<init>"
    )
    rejected = tmp_path / "from-plan"
    assert (
        run("middle", "plan", "--graph", negative, "--out", rejected, returncode=2)[
            "status"
        ]
        == "INCOMPLETE"
    )
    report = json.loads((rejected / "plan.json").read_bytes())
    assert any(i["code"] == "python.import_from_hook" for i in report["issues"])
    partial = json.loads((rejected / "exec-ir.json").read_bytes())
    assert not partial["functions"]
    assert read_member(rejected, partial["module"]["source_imports"])["occurrences"]
    assert report["source_imports"] == partial["module"]["source_imports"]


def test_fromlist_keeps_bound_children_and_import_from_callbacks(tmp_path):
    import shutil

    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle import plan

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("token = 11\nimport pkg.reader\n")
    (package / "token.py").write_text("print('unexpected child initializer')\n")
    (package / "reader.py").write_text(
        "from . import token as first, token as second\nprint(first, second)\n"
    )
    entry = tmp_path / "entry.py"
    entry.write_text("import pkg\n")
    expected = subprocess.check_output([sys.executable, "-B", entry])
    assert expected == b"11 11\n"
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text("""import builtins, os, sys
original = builtins.__import__
class Returned:
    count = 0
    @property
    def token(self):
        self.count += 1
        return 90 + self.count
def observed(name, globals=None, locals=None, fromlist=(), level=0):
    if name == '' and level == 1 and fromlist == ('token','token') and globals.get('__name__') == 'pkg.reader':
        package = sys.modules['pkg']
        if os.environ['IMPORT_CASE'] == 'delete':
            del package.token
        elif os.environ['IMPORT_CASE'] == 'rebind':
            package.token = 83
        else:
            original(name, globals, locals, fromlist, level)
            return Returned()
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = observed
""")

    def run_with_hook(case):
        return subprocess.check_output(
            [sys.executable, "-B", entry],
            env=dict(os.environ, PYTHONPATH=str(hooks), IMPORT_CASE=case),
        )

    assert run_with_hook("rebind") == b"83 83\n"
    assert run_with_hook("delete").startswith(
        b"unexpected child initializer\n<module 'pkg.token'"
    )
    assert run_with_hook("replace_result") == b"91 92\n"
    graph, ir = tmp_path / "graph", tmp_path / "ir"
    build(entry, graph)
    shutil.rmtree(package)
    entry.unlink()
    assert plan(graph, ir)["status"] == "INCOMPLETE"
    report = json.loads((ir / "plan.json").read_bytes())
    [issue] = report["issues"]
    assert issue["code"] == "python.module_attribute_state"
    evidence = issue["source_evidence"]
    ledger = read_member(ir, report["source_imports"])
    row = next(r for r in ledger["occurrences"] if r["id"] == evidence["occurrence"])
    assert evidence["import_effects"] == row["import_effects"]
    assert evidence["import_effects"]["effect_coverage"] == "UNPROVEN"
    assert not evidence["import_effects"]["producer_sites"]
    assert evidence["bindings"] == row["bindings"]
    assert [b["name"] for b in evidence["bindings"]] == [
        "py:pkg.reader:first",
        "py:pkg.reader:second",
    ]
    assert [
        (c["binding_index"], c["member"], c["graph_binding"])
        for c in evidence["ordered_checks"]
    ] == [(0, "token", "bound"), (1, "token", "bound")]
    assert all(
        c["child_candidate"]["initializer"] == "py:pkg.token:<init>"
        for c in evidence["ordered_checks"]
    )
    assert not any(
        e["target"] == "py:pkg.token:<init>"
        for row in report["normalizations"]
        for e in row.get("removed_edges", ())
    )
    assert not json.loads((ir / "exec-ir.json").read_bytes())["functions"]


@pytest.mark.parametrize("hook", [False, True])
def test_fromlist_keeps_current_import_callback_and_hook_outcomes(tmp_path, hook):
    from scar.frontend import build
    from scar.middle import plan

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text(
        "first = 3\n"
        + ("def __getattr__(name):\n    return 'package hook'\n" if hook else "")
        + "from . import first as known, child\nprint(known, child)\n"
    )
    (package / "child.py").write_text("print('child initializer')\n")
    entry = tmp_path / "entry.py"
    entry.write_text("import pkg\n")
    normal = subprocess.check_output([sys.executable, "-B", entry])
    if hook:
        assert normal == b"3 package hook\n"
    else:
        assert normal.startswith(b"child initializer\n3 <module")
    injected = """import builtins, runpy, sys
original = builtins.__import__
def tracked(name, globals=None, locals=None, fromlist=(), level=0):
    if name == '' and level == 1 and fromlist == ('first','child') and globals.get('__name__') == 'pkg':
        sys.modules['pkg'].child = 'injected'
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = tracked
runpy.run_path('entry.py', run_name='__main__')
"""
    assert (
        subprocess.check_output([sys.executable, "-B", "-c", injected], cwd=tmp_path)
        == b"3 injected\n"
    )
    graph, ir = tmp_path / "graph", tmp_path / "ir"
    build(entry, graph)
    assert plan(graph, ir)["status"] == "INCOMPLETE"
    (issue,) = json.loads((ir / "plan.json").read_bytes())["issues"]
    assert issue["code"] == "python.module_attribute_state"
    evidence = issue["source_evidence"]
    assert evidence["phase"] == "during_IMPORT_NAME_before_IMPORT_FROM"
    assert evidence["member"] == "first" and evidence["binding_index"] == 0
    first, child = evidence["ordered_checks"]
    assert first["member"] == "first" and first["graph_binding"] == "bound"
    assert child["member"] == "child" and child["graph_binding"] == "no_graph_binding"
    assert evidence["registry_state"] == "initializing"
    assert child["child_candidate"]["initializer"] == "py:pkg.child:<init>"
    assert evidence["actual_lookup"] == "unproven_after_current_import_callbacks"
    assert set(evidence["outcomes"]) == {
        "present",
        "attribute_error",
        "other_exception",
    }
    assert evidence["hook_binding"] == ("bound" if hook else "no_graph_binding")


def test_package_fromlist_publication_and_deletion_retain_order_and_callbacks(tmp_path):
    import shutil

    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle import plan

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from . import child as first, child as second\ndel first, second\n"
        "from other.library import action as first, action as second\nfirst(second)\nboundary = 0 or 1\n"
    )
    (package / "child.py").write_text("print('child')\n")
    other = tmp_path / "other"
    other.mkdir()
    (other / "__init__.py").write_text("print('parent')\n")
    (other / "library.py").write_text(
        "print('library')\ndef action(arg):\n    print('source function')\n"
    )
    entry = tmp_path / "main.py"
    entry.write_text("import pkg\n")
    assert (
        subprocess.check_output([sys.executable, "-B", entry])
        == b"child\nparent\nlibrary\nsource function\n"
    )
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text("""import builtins, os
original = builtins.__import__
def importer(name, globals=None, locals=None, fromlist=(), level=0):
    if os.environ.get('FOREIGN_BINDINGS'):
        if name == 'other.library' and fromlist == ('action','action'):
            class Returned:
                action = 43
                def __del__(self):
                    globals['first'] = lambda x: print('changed', x)
            return Returned()
    elif level == 1 and fromlist == ('child', 'child') and globals['__name__'] == 'pkg':
        print('import once')
        class Value:
            def __del__(self):
                if 'second' in globals:
                    print('delete second during release')
                    del globals['second']
        class Result:
            @property
            def child(self):
                print('lookup', 'first' in globals)
                return Value()
            def __del__(self):
                print('release result')
        return Result()
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = importer
""")
    observed = subprocess.run(
        [sys.executable, "-B", entry],
        capture_output=True,
        env=dict(os.environ, PYTHONPATH=str(hooks)),
    )
    assert (
        observed.returncode == 1
        and b"NameError: name 'second' is not defined" in observed.stderr
    )
    assert (
        observed.stdout
        == b"import once\nlookup False\nlookup True\nrelease result\ndelete second during release\n"
    )
    assert (
        subprocess.check_output(
            [sys.executable, "-B", entry],
            env=dict(os.environ, PYTHONPATH=str(hooks), FOREIGN_BINDINGS="1"),
        )
        == b"child\nchanged 43\n"
    )

    graph, ir = tmp_path / "graph", tmp_path / "ir"
    build(entry, graph)
    shutil.rmtree(package)
    shutil.rmtree(other)
    entry.unlink()
    assert plan(graph, ir)["status"] == "INCOMPLETE"
    report = json.loads((ir / "plan.json").read_bytes())
    [issue] = report["issues"]
    assert (
        issue["code"] == "python.global_store_after_callback"
        and issue["location"]["line"] == 5
    )
    analysis = report["protocol_analysis"]
    fragment = next(
        f for f in analysis["prefix_fragments"] if f["symbol"] == "py:pkg:<init>"
    )
    prefix = read_member(ir, fragment["fragment"])
    publish, delete, foreign, use = [s["value"] for s in prefix["body"]]
    assert (
        use["op"] == "module_object_expression"
        and use["expression"]["op"] == "source_object_discard"
    )
    call = use["expression"]["value"]
    assert call["op"] == "source_object_call"
    assert [
        operand["instruction"]["name"] for operand in [call["target"], *call["args"]]
    ] == ["first", "second"]
    assert (
        foreign["op"] == "module_fromlist_publish" and foreign["location"]["line"] == 3
    )
    assert (
        foreign["resolved_module"] == "other.library"
        and foreign["frame_globals"] == publish["frame_globals"]
    )
    assert foreign["initializers"] == ["py:other:<init>", "py:other.library:<init>"]
    assert foreign["request"]["fromlist"] == ["action", "action"]
    assert [
        (b["binding_index"], b["name"], b["slot"]) for b in foreign["bindings"]
    ] == [(0, "first", "py:pkg:first"), (1, "second", "py:pkg:second")]
    previous = call["target"]["source_evidence"]["prior_publication"]
    assert (
        previous["occurrence"] == foreign["occurrence"]
        and previous["binding_index"] == 0
    )
    assert previous["source_evidence"] == foreign["bindings"][0]["source_evidence"]
    candidate = previous["source_evidence"]["bound_value"]
    assert (
        candidate["class"] == "function"
        and candidate["symbol"] == "py:other.library:action"
    )
    assert candidate["guards"]
    assert [publish["op"], delete["op"]] == [
        "module_fromlist_publish",
        "module_global_delete",
    ]
    assert [p["location"]["line"] for p in (publish, delete)] == [1, 2]
    assert all(
        p["operator_family"] == "module_registry"
        and p["may_raise"]
        and p["external_effect"] == "UNKNOWN"
        for p in (publish, delete)
    )
    assert (
        publish["frame_globals"]
        == delete["frame_globals"]
        == dict(
            owner="py:pkg:<init>", module="pkg", identity="current_activation_globals"
        )
    )
    assert publish["request"]["fromlist"] == ["child", "child"]
    assert all(b["namespace"] == "frame.locals" for b in publish["bindings"])
    assert (
        [b["name"] for b in publish["bindings"]]
        == delete["names"]
        == ["first", "second"]
    )
    assert all(
        c["child_candidate"]["initializer"] == "py:pkg.child:<init>"
        and c["when"]["kind"] == "package_attribute_missing"
        for c in publish["fromlist_checks"]
    )
    assert publish["initializers"] == ["py:pkg:<init>"]
    imports = read_member(ir, report["source_imports"])["occurrences"]
    occurrence = next(row for row in imports if row["id"] == publish["occurrence"])
    [producer] = occurrence["producer_sites"]
    assert producer["opcode"] == publish["op"] and producer["site"] == publish["site"]
    assert occurrence["effect_coverage"] == "MODELED"
    assert not any(
        edge["target"] == "py:pkg.child:<init>"
        for normalization in report["normalizations"]
        for edge in normalization.get("removed_edges", ())
    )


def test_workflow_binds_entry_state_without_inventing_child_file_versions(tmp_path):
    process = dict(kind="handle", resource="process")

    def variable(name):
        return dict(op="var", name=name, type=process)

    def literal(text):
        return dict(op="literal", type="text", value=text)

    def launch(name):
        return dict(
            op="decl",
            name=name,
            type=process,
            init=dict(
                op="process_spawn",
                image_id="worker",
                argv=[literal("./worker")],
                cwd="inherit",
                stdin="inherit",
                stdout="pipe",
                stderr="inherit",
            ),
        )

    def wait(name):
        return dict(
            op="expr",
            value=dict(op="host_call", interface="process.wait", args=[variable(name)]),
        )

    body = [
        dict(
            op="expr",
            value=dict(
                op="host_call",
                interface="fs.copy_file",
                args=[literal("source"), literal("input.par")],
            ),
        ),
        launch("p"),
        dict(op="decl", name="alias", type=process, init=variable("p")),
        launch("q"),
        wait("alias"),
        wait("q"),
    ]
    body[3]["init"]["launcher"] = dict(
        kind="mpi", ranks=dict(op="literal", type="i64", value=2), semantics_version=1
    )
    module = dict(
        entry="main",
        current_image="driver",
        functions=[dict(name="main", params=[], result="unit", body=body)],
    )
    flow = _layer(
        "normalization",
        "from scar.middle.normalization.artifacts import analyze\nprint(json.dumps(analyze(data)['graph']))",
        module,
    )
    code = """from pathlib import Path
from scar.contract import ARTIFACT_FLOW, document, file_receipt
from scar.middle.execution import seal
from scar.middle.workflow import link_images
out=Path(data['out'])
consumer=document(ARTIFACT_FLOW,nodes=[
    dict(id='read-op',kind='CANONICAL_OPERATION',operation='library_call'),
    dict(id='input',kind='FILE_LOCATOR',path=None,cwd='process_cwd_at_open'),
    dict(id='read',kind='FILE_VERSION_REQUIREMENT',locator='input',value=None,
         consumer='read-op',role='runtime_stream_read',requires=['file_bytes_consulted']),
],edges=[])
images=[]
for name,flow in [('driver',data['flow']),('worker',consumer)]:
    module=dict(entry='main',functions=[],resource_model=dict(artifact_flow=flow))
    path=seal(product(module,'M4',complete=False),out/name,status='INCOMPLETE',graph_sha256='source',
              contract={})
    images.append(dict(id=name,entry='main',output=name,ir=file_receipt(out,path)))
relocated=out.with_name('relocated-ir')
out.rename(relocated)
print(json.dumps(link_images(relocated,images,'driver')))
"""
    workflow = _layer("execution", code, dict(out=str(tmp_path / "ir"), flow=flow))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    launches = [node for node in nodes.values() if node["kind"] == "LAUNCH"]
    assert len(launches) == 2
    assert {node["child_image"] for node in launches} == {"worker"}
    assert {node["replication"]["kind"] for node in launches} == {
        "single_process",
        "mpi",
    }
    bindings = [e for e in workflow["edges"] if e["kind"] == "BINDS_ENTRY_STATE"]
    assert len(bindings) == 8
    assert all(e["binding_time"] == "process_entry" for e in bindings)
    waits = [e for e in workflow["edges"] if e["kind"] == "WAITS_FOR"]
    assert len(waits) == 2 and all(e["descendants_completed"] is False for e in waits)
    (read,) = [node for node in nodes.values() if node["kind"] == "FILE_READ"]
    assert read["path"] is None
    assert any(node["kind"] == "FILE_COPY" for node in nodes.values())
    assert workflow["file_version_equivalence_established"] is False
    assert not any(
        nodes[e["source"]]["kind"] == "FILE_COPY"
        and nodes[e["target"]]["kind"] == "FILE_READ"
        for e in workflow["edges"]
    )


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
from scar.middle.contracts import publish_program
def product(module, phase, complete=True):
    module=dict(module)
    for field in ('functions','globals','externs','records'):
        module.setdefault(field,[])
    if 'entry' not in module and module['functions']:
        module['entry']=module['functions'][0]['name']
    return publish_program(module,phase,complete=complete)
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


def test_source_literal_exports_require_actual_lookup_and_source_continuation():
    from scar.contract import python_provider_id

    source = dict(
        module="time",
        status="verified_stdlib",
        platform="linux",
        origin="/sealed/time.py",
        member="time.py",
        sha256="a" * 64,
        loader="path",
        proof_scope="resolved_module",
        hash_basis="provider_file",
        interpreter="cpython 3.12.3",
    )
    source["provider_id"] = python_provider_id(source)
    candidate = dict(
        kind="literal",
        source_type="builtins.str",
        literal=dict(type="text", value="source-value"),
        module="time",
        name="stamp",
        member="time.py",
        origin=source["origin"],
        sha256=source["sha256"],
        provider_id=source["provider_id"],
        line=5,
        end_line=5,
        column=1,
        end_column=21,
        conditions=[],
        via=[
            dict(
                kind="literal",
                module="time",
                name="stamp",
                member="time.py",
                line=5,
                sha256=source["sha256"],
                provider_id=source["provider_id"],
            )
        ],
    )
    source["exports"] = dict(
        stamp=dict(status="source_candidates", candidates=[candidate], unresolved=[])
    )
    result = _layer(
        "normalization",
        """from copy import deepcopy
from scar.middle.normalization.protocols import _Project,_Frame,_Unclosed
from scar.contract import walk_operations
def setup(source):
    def imported(member,index):
        return dict(op='python_import',module='time',member=member,
            source_provider=source,import_resolution=source,provider='python_package',
            request=dict(form='from',level=0,module='time',fromlist=['sleep','stamp'],binding_index=index),
            binding=dict(op='var',name='py:test:'+member,scope='global'),
            effects=['CTRL','STATE','IO'],site='source/from/'+str(index),location=dict(file='entry.py',line=1))
    imports=[imported('sleep',0),imported('stamp',1)]
    entry=dict(name='entry',params=[],body=imports,python_scope=dict(kind='module',module='test',main=True,initializer='entry',provider_sources={'time':source}))
    project=_Project({'entry':entry});project.active.append('entry')
    return project,_Frame(project,entry,()),imports
project,frame,imports=setup(data)
steps=[]
frame.imported(imports[0],steps)
assert steps[0]['op']=='block'
assert steps[0]['body'][0]['value']['op']=='module_namespace_initialize'
frame.imported(imports[1],steps)
operations=list(walk_operations(steps))
initializers=[n for n in operations if n['op']=='module_namespace_initialize']
assert len(initializers)==1 and steps[0]['body'][0]['value'] is initializers[0]
assert initializers[0]['request']['fromlist']==['sleep','stamp']
assert initializers[0]['initialization']['fromlist'].startswith('perform_complete')
assert project.names['py:test:sleep'].kind=='host_callable'
bound=project.names['py:test:stamp']
assert bound.exact=='source-value' and bound.value['op']=='var'
guard=next(n for n in operations if n['op']=='module_export_matches')
value=next(n for n in operations if n['op']=='module_export_value')
continuation=next(n for n in operations if n['op']=='module_export_continuation_required')
assert guard['capture']==value['capture']==continuation['capture']
assert guard['checks'] and guard['external_effect']=='UNKNOWN'
assert guard['instance_scope']=='per_dynamic_lookup_in_current_activation'
assert continuation['noreturn'] and 'not_raise_or_repeat_lookup' in continuation['semantics']
assert value['projection'].startswith('actual_captured_object')
assert 'capture' not in bound.facts['source_export_binding']
row=next(iter(project.source_imports.rows.values()))
assert len(row['bindings'])==2 and row['effect_coverage']=='MODELED'
assert initializers[0]['namespace_binding']['bindings']==row['bindings']
assert len(row['producer_sites'])==1
assert {'module_export_matches','module_export_value','module_export_continuation_required'} <= set(project.operator_requirements()['module_registry']['operations'])
try:
    frame.evaluate(dict(op='binary',operator='is',left=bound.value,right=bound.value),steps)
except _Unclosed as error: identity_gap=error.issue['code']
else: raise AssertionError('source scalar identity was folded by value')
bad=deepcopy(data);bad['exports']['stamp']['candidates'][0]['literal']=dict(type='python_int',value=1<<100)
bad['exports']['stamp']['candidates'][0]['source_type']='builtins.int'
bad['exports']['stamp']['candidates'][0]['representation']='requires_integer_range_contract'
project,frame,imports=setup(bad)
try:
    frame.imported(imports[0],[]);frame.imported(imports[1],[])
except _Unclosed as error: representation_gap=error.issue['code']
else: raise AssertionError('large source integer was silently narrowed')
generic=deepcopy(data);generic.update(module='source_library',status='verified_distribution',distribution='source-library',version='1.0',origin='/sealed/source_library.py',member='source_library.py')
from scar.contract import python_provider_id
generic['provider_id']=python_provider_id(generic)
leaf=generic['exports']['stamp']['candidates'][0]
leaf.update(module=generic['module'],origin=generic['origin'],member=generic['member'],provider_id=generic['provider_id'])
leaf['via']=[dict(kind='literal',module=generic['module'],name='stamp',member=generic['member'],line=5,sha256=generic['sha256'],provider_id=generic['provider_id'])]
node=dict(op='python_import',module='source_library',provider='python_package',source_provider=generic,import_resolution=generic,request=dict(form='import',module='source_library',level=0),alias='lib',binding=dict(op='var',name='py:test:lib',scope='global'),site='source/import',effects=['UNKNOWN'])
reader=dict(name='reader',params=[],call_protocol=dict(kind='regular'),python_scope=dict(kind='function',module='test',initializer='entry'),body=[dict(op='return',value=dict(op='member',base=dict(op='var',name='py:test:alias',scope='global'),field='stamp',site='source/lookup'))])
entry=dict(name='entry',params=[],body=[node],python_scope=dict(kind='module',module='test',main=True,initializer='entry',provider_sources={'source_library':generic}))
project=_Project({'entry':entry,'reader':reader});project.active.append('entry');frame=_Frame(project,entry,());steps=[]
frame.imported(node,steps)
frame.store(dict(op='var',name='py:test:alias',scope='global'),project.names['py:test:lib'],steps)
first=project.compile('reader',());second=project.compile('reader',())
assert first is second and project.result_facts['reader']['exact']=='source-value'
assert 'source_export_binding' not in project.result_facts['reader']
lookup=next(n for n in walk_operations(first) if n['op']=='module_export_matches')
assert lookup['lookup']=='attribute' and lookup['namespace_binding']['occurrence']==next(iter(project.source_imports.rows))
assert lookup['instance_scope']=='per_dynamic_lookup_in_current_activation'
print(json.dumps(dict(identity_gap=identity_gap,representation_gap=representation_gap,lookup=lookup['op'])))
""",
        source,
    )
    assert result["representation_gap"] == "python.source_export_representation"
    assert result["identity_gap"] == "python.arithmetic_contract"
    assert result["lookup"] == "module_export_matches"


def test_module_import_state_keeps_normal_exceptional_and_parent_lifetimes_separate():
    result = _layer(
        "normalization",
        """from scar.middle.normalization.protocols import _Project,_Frame,_Unclosed
from scar.middle.normalization.values import Value
from scar.contract import walk_operations
def variable(name): return dict(op='var',name=name,scope='global')
def module(name,symbol,body):
    return dict(name=symbol,params=[],body=body,python_scope=dict(kind='module',module=name,main=False,initializer=symbol))
child=module('package.child','child_init',[dict(op='assign',target=variable('py:package.child:value'),value=dict(op='literal',type='i64',value=7))])
inside=dict(op='python_import',module='package.child',provider='project',member='value',initializers=['package_init','child_init'],request=dict(form='from',module='package.child',level=0,fromlist=['value']),binding=variable('py:package:captured'),effects=['CTRL','STATE','IO'],site='package/child')
parent=module('package','package_init',[inside])
entry=module('test','entry',[])
imported=dict(op='python_import',module='package',provider='project',initializers=['package_init'],request=dict(form='import',module='package',level=0),binding=variable('py:test:package'),effects=['CTRL','STATE','IO'],site='test/import')
project=_Project({'entry':entry,'package_init':parent,'child_init':child});project.active.append('entry');frame=_Frame(project,entry,())
start=frame.snapshot();steps=[]
protected=dict(op='try',body=[imported],handlers=[dict(op='exception_handler',test=dict(op='python_builtin',name='ImportError'),name=None,binding=None,body=[dict(op='raise',has_exception=False,has_cause=False)],cleanup=[])],**{'else':[],'finally':[]})
assert frame.sequence([protected],steps)
assert not frame.dynamic and not frame.control_regions
assert project.imports.registry_entry('package')['state']==project.imports.registry_entry('package.child')['state']=='ready'
calls=[n['callee'] for n in walk_operations(project.closed['package_init']) if n['op']=='call']
assert calls==['child_init'],calls
assert [n['callee'] for n in walk_operations(steps) if n['op']=='call']==['package_init']
after=frame.snapshot()
assert 'package' not in start[2]['registry']
assert project.names['py:package:captured'].facts['exact']==7
exceptional=frame.exceptional_snapshot(start,after,[imported])
assert exceptional[2]['registry']['package']['state']=='unresolved'
assert 'initializing' in exceptional[2]['registry']['package']['possible_states']
frame.merge([after,exceptional]);assert project.imports.registry_entry('package')['state']=='unresolved'
joined=frame.snapshot()
projected=frame.project_final_state(after,joined,joined)
assert projected[2]['registry']['package']['state']=='ready'
frame.restore(joined);project.unknown_registry('entry',{'site':'finally/callback'})
projected=frame.project_final_state(after,joined,frame.snapshot())
assert projected[2]['namespace_writes']['package']['state']=='unresolved'
frame.restore(after);assert project.imports.registry_entry('package')['state']=='ready'
clone=frame.copy_state(after);clone[2]['registry'].pop('package')
assert after[2]['registry']['package']['state']=='ready'
frame.restore(after)
frame.env['condition']=Value('bool',dict(op='var',name='condition',type='bool'))
branch=dict(op='if',condition=dict(op='var',name='condition'),then=[imported],**{'else':[]})
try: frame.conditional(branch,[])
except _Unclosed as error: issue=error.issue['code']
else: raise AssertionError('runtime branch asserted module initialization')
assert issue=='python.conditional_import'
assert not frame.dynamic and not frame.control_regions
frame.dynamic=1;frame.enter_control('loop',{'site':'loop'})
try:
    try: frame.imported(imported,[])
    except _Unclosed as error: assert error.issue['code']=='python.conditional_import'
    else: raise AssertionError('loop import acquired an unconditional proof')
finally: frame.leave_control();frame.dynamic=0
frame.restore(after)
before=project.imports.snapshot()
comparison=dict(op='module_import_preserves',type='bool',operator_family='module_registry',occurrence='foreign/import',direct_registry_keys=['foreign'],conditional_registry_keys=[],checks={})
project.unknown_registry('entry',{'site':'foreign/import'})
guarded=[]
project.scoped_registry_effect('entry',{'site':'foreign/import'},comparison,guarded,before=before)
assert project.imports.registry_entry('package')['state']=='unresolved'
assert project.registry_entry('package',{'site':'consumer'})['state']=='ready'
assert guarded[0]['op']=='if'
assert guarded[0]['else'][0]['value']['op']=='module_import_continuation_required'
assert comparison['checks']['package']['registry']['requires']=='same_module_instance_and_complete'
project.imports.binding_written('py:package:fresh')
project.observe_import('package','namespace',{'site':'fresh-read'},'fresh')
assert 'namespace' not in comparison['checks']['package']
project.observe_import('package','namespace',{'site':'prior-read'},'captured')
assert comparison['checks']['package']['namespace']['first_member']=='captured'
project.unknown_registry('entry',{'site':'unrelated/callback'})
assert project.registry_entry('package',{'site':'later'})['state']=='unresolved'
print(json.dumps({'normal':'ready','exceptional':'unresolved','child_calls':calls,'branch_issue':issue}))
""",
    )
    assert result == dict(
        normal="ready",
        exceptional="unresolved",
        child_calls=["child_init"],
        branch_issue="python.conditional_import",
    )


def test_binding_preserves_unit_scoped_provider_and_export_evidence(
    tmp_path, monkeypatch
):
    from base64 import urlsafe_b64encode
    from hashlib import sha256

    from scar.contract import walk_operations
    from scar.frontend import build

    source, installed, graph = (tmp_path / n for n in ("source", "installed", "graph"))
    source.mkdir()
    package = installed / "provider_sample"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from importlib import import_module\n"
        "def __getattr__(name):\n"
        "    if name == 'ops': return import_module(__name__ + '.ops')\n"
        "    raise AttributeError(name)\n"
    )
    (package / "ops.py").write_text("from .kernel import apply, reduce\n")
    kernel = package / "kernel.py"
    kernel.write_text(
        "def apply(x): return x\ndef reduce(x): return x\nprint('kernel-init')\nmarker = 42\n"
    )
    kernel_hash = sha256(kernel.read_bytes()).hexdigest()
    metadata = installed / "provider_sample-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: provider-sample\nVersion: 1.0\n"
    )
    rows = []
    for path in sorted(package.glob("*.py")):
        raw = path.read_bytes()
        checksum = urlsafe_b64encode(sha256(raw).digest()).decode().rstrip("=")
        rows.append(f"{path.relative_to(installed)},sha256={checksum},{len(raw)}\n")
    (metadata / "RECORD").write_text("".join(rows))
    monkeypatch.syspath_prepend(str(installed))
    entry = source / "entry.py"
    entry.write_text(
        "import provider_sample\nimport sibling\nprovider_sample.ops.apply(1)\nsibling.run()\n"
    )
    (source / "sibling.py").write_text(
        "import provider_sample\ndef run(): return provider_sample.ops.reduce(2)\n"
    )
    build(entry, graph)
    cached = source / "cached.py"
    cached.write_text(
        "import time\nimport importlib as lib\nfrom importlib import import_module as lookup, import_module as second\n"
        "import provider_sample.kernel\n"
        "def read(): return lookup('provider_sample.kernel').marker\n"
        "import math\n"
        "before = read()\n"
        "print(read(), lib.import_module('.kernel', package='provider_sample').marker, before)\n"
        "import provider_sample.kernel as repeated\n"
        "last = repeated.marker\n"
    )
    cached_graph = tmp_path / "cached-graph"
    build(cached, cached_graph)
    reference = subprocess.run(
        [sys.executable, "-B", str(cached)],
        env={**os.environ, "PYTHONPATH": str(installed)},
        check=True,
        capture_output=True,
    )
    assert reference.stdout == b"kernel-init\n42 42 42\n"
    source.rename(tmp_path / "hidden-source")
    installed.rename(tmp_path / "hidden-provider")
    script = """from pathlib import Path
from scar.contract import GRAPH, open_product
from scar.middle.binding import bind
path, _, graph = open_product(Path(data), "graph.json", GRAPH)
program, issues, _ = bind(graph, path.parent)
assert not issues
print(json.dumps(program))
"""
    bound = _layer("binding", script, str(graph))
    functions = {fn["name"]: fn for fn in bound["functions"]}
    root = functions["__main__"]["python_scope"]["provider_sources"]
    caller = functions["py:sibling:run"]["python_scope"]
    assert "provider_sources" not in caller
    child = functions[caller["initializer"]]["python_scope"]["provider_sources"]
    for name in ("provider_sample", "provider_sample.ops"):
        assert root[name]["provider_id"] == child[name]["provider_id"]
    assert (
        root["provider_sample"]["provider_id"]
        != root["provider_sample.ops"]["provider_id"]
    )
    assert set(root["provider_sample.ops"]["exports"]) == {"apply"}
    assert set(child["provider_sample.ops"]["exports"]) == {"reduce"}

    issue = _layer(
        "normalization",
        """from scar.middle.normalization.protocols import normalize_protocols
_, _, (issue,) = normalize_protocols(data)
print(json.dumps(issue))
""",
        bound,
    )
    assert issue["code"] == "python.import_module_cache"
    assert issue["source_evidence"]["module"] == "provider_sample.ops"
    assert issue["source_evidence"]["registry_state"] == "no_scheduled_initialization"
    export = root["provider_sample.ops"]["exports"]["apply"]
    assert not export["unresolved"]
    (definition,) = export["candidates"]
    assert (definition["module"], definition["line"], definition["sha256"]) == (
        "provider_sample.kernel",
        1,
        kernel_hash,
    )

    bound_cached = _layer("binding", script, str(cached_graph))
    result, decisions, issues = _layer(
        "normalization",
        "from scar.middle.normalization.protocols import normalize_protocols\nprint(json.dumps(normalize_protocols(data)))",
        bound_cached,
    )
    assert not issues and result["protocol_analysis"]["status"] == "READY"
    requirement = result["operator_requirements"]["module_registry"]
    assert set(requirement["operations"]) == {
        "module_namespace_initialize",
        "module_import_preserves",
        "module_import_continuation_required",
    }
    assert (
        requirement["entry_contract"]["initial_cache"] == "unobserved_by_static_capture"
    )
    ops = list(walk_operations(result["functions"]))
    init = "py:provider_sample.kernel:<init>"
    assert sum(n.get("callee") == init for n in ops) == 1
    namespace = next(n for n in ops if n["op"] == "module_namespace_initialize")
    assert namespace["may_raise"] and "process.import_hooks" in namespace["reads"]
    guarded = next(
        n
        for n in ops
        if n["op"] == "if"
        and n["condition"]["op"] == "module_import_preserves"
        and n["condition"]["module"] == "math"
    )
    transition = guarded["condition"]
    rows = result["source_imports"]["occurrences"]
    fromlist = next(r for r in rows if len(r["bindings"]) == 2)
    assert [b["name"].rsplit(":", 1)[-1] for b in fromlist["bindings"]] == [
        "lookup",
        "second",
    ]
    assert (
        fromlist["effect_coverage"] == "MODELED"
        and len(fromlist["producer_sites"]) == 1
    )
    assert sum(op.get("source_import_occurrence") == fromlist["id"] for op in ops) == 1
    unused = next(r for r in rows if r["request"]["module"] == "time")
    [loader] = [op for op in ops if op.get("source_import_occurrence") == unused["id"]]
    [site] = unused["producer_sites"]
    assert unused["effect_coverage"] == "MODELED" and unused["obligations"]
    assert site["opcode"] == loader["op"] == "module_namespace_initialize"
    assert site["site"] == loader["site"] == unused["site"]
    assert loader["namespace_binding"]["retains"] == "actual_IMPORT_NAME_result"
    assert loader["namespace_binding"]["bindings"] == unused["bindings"]
    assert loader["external_effect"] == "UNKNOWN" and loader["may_raise"]
    occurrence = next(r for r in rows if r["id"] == transition["occurrence"])
    assert transition["writes"] == [] and transition["reads"] == [
        "occurrence.before",
        "occurrence.after",
    ]
    assert transition["producer_sites"] == occurrence["producer_sites"]
    assert transition["type"] == "bool" and transition["direct_registry_keys"] == [
        "math"
    ]
    checked = transition["checks"]["provider_sample.kernel"]
    assert (
        checked["registry"]["site"] and checked["namespace"]["first_member"] == "marker"
    )
    assert guarded["else"][0]["value"]["noreturn"]
    hits = [d for d in decisions if d["code"] == "module_registry_hit"]
    assert {d["owner"] for d in hits} == {"py:cached:read", "__main__"}
    assert all(
        d["initialized_by"]["initializer"] == init and d["initialized_by"]["site"]
        for d in hits
    )

    # The same compiled bodies cease to prove a cache hit when a later
    # effectful namespace import can mutate the registry through hooks.
    body = next(f["body"] for f in bound_cached["functions"] if f["name"] == "__main__")
    again = next(
        n
        for n in body
        if n.get("op") == "python_import" and n["request"]["form"] == "from"
    )
    target = next(n for n in body if n.get("module") == "provider_sample.kernel")

    def after_registry_write(position, *, repeat_target=False, expect_issue=True):
        body[position:position] = [again, target] if repeat_target else [again]
        try:
            return _layer(
                "normalization",
                "from scar.middle.normalization.protocols import normalize_protocols\n"
                + (
                    "_,_,(issue,) = normalize_protocols(data)\nprint(json.dumps(issue))"
                    if expect_issue
                    else "print(json.dumps(normalize_protocols(data)))"
                ),
                bound_cached,
            )
        finally:
            del body[position : position + (2 if repeat_target else 1)]

    issue = after_registry_write(body.index(target) + 1)
    repeated_import = next(
        n
        for n in body[body.index(target) + 1 :]
        if n.get("op") == "python_import" and n.get("module") == target["module"]
    )
    assert issue["code"] == "python.import_initializer_state"
    assert issue["source_evidence"]["module"] == "provider_sample"
    assert issue["source_evidence"]["initializer"] == target["initializers"][0]
    assert issue["source_evidence"]["registry_state"] == "unresolved"
    assert issue["site"] == repeated_import["site"]
    assert issue["source_evidence"]["last_change"]["site"] == again["site"]
    ordinary = after_registry_write(body.index(target) + 1, repeat_target=True)
    assert ordinary["code"] == "python.import_initializer_state"
    assert ordinary["source_evidence"]["module"] == "provider_sample"
    assert ordinary["source_evidence"]["initializer"] == target["initializers"][0]
    assert ordinary["source_evidence"]["registry_state"] == "unresolved"
    assert ordinary["site"] == target["site"]
    assert ordinary["source_evidence"]["last_change"]["site"] == again["site"]
    before = next(n for n in body if n["op"] == "assign")
    cached_call = after_registry_write(body.index(before) + 1)
    assert cached_call["code"] == "python.import_module_cache"
    assert cached_call["source_evidence"]["module"] == target["module"]
    assert cached_call["source_evidence"]["required"]["initializer"] == init
    assert cached_call["source_evidence"]["registry_state"] == "unresolved"
    assert cached_call["source_evidence"]["last_change"]["site"] == again["site"]
    assert cached_call["symbol"] == before["value"]["target"]["name"]
    assert cached_call["site"] == next(
        op["site"]
        for op in walk_operations(body[body.index(before) + 1])
        if op["op"] == "call"
        and op["target"].get("name") == before["value"]["target"]["name"]
    )
    old_alias, _, alias_issues = after_registry_write(len(body) - 1, expect_issue=False)
    assert not alias_issues
    # The retained alias is usable only through the explicit normal comparison
    # edge. Its failed edge still requires a source-equivalent continuation.
    alias_read = body[-1]["value"]
    alias_guard = next(
        n
        for n in walk_operations(old_alias["functions"])
        if n["op"] == "if"
        and n["condition"]["op"] == "module_import_preserves"
        and n["condition"]["module"] == "importlib"
        and n["condition"]["site"] == again["site"]
        and target["module"] in n["condition"]["checks"]
        and n["condition"]["checks"][target["module"]]["namespace"]["first_member"]
        == alias_read["field"]
    )
    namespace_check = alias_guard["condition"]["checks"][target["module"]]
    assert namespace_check["before"]["initializer"] == init
    assert namespace_check["namespace"]["requires"] == (
        "same_namespace_bindings_and_reachable_value_state"
    )
    assert any(
        read["member"] == alias_read["field"]
        and read["site"] == alias_read["site"]
        and read["consumer"] == "__main__"
        for read in namespace_check["namespace"]["binding_reads"]
    )
    root_check = alias_guard["condition"]["checks"]["__main__"]["namespace"]
    assert root_check["first_member"] == alias_read["base"]["name"].rsplit(":", 1)[-1]
    assert root_check["lexical_namespace"] == "cached"
    failure = alias_guard["else"][0]["value"]
    assert failure["op"] == "module_import_continuation_required"
    assert failure["operator_family"] == "module_registry"
    assert failure["site"] == again["site"]
    assert failure["noreturn"] and set(failure["effects"]) == {"CTRL", "OPAQUE"}
    assert failure["semantics"] == (
        "transfer_to_source_equivalent_continuation_or_prove_edge_unreachable_before_delivery"
    )


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
module,facts=prepare(product(data,'M3'))
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
module,issues=normalize(product(data,'M1'))
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


def test_invocation_access_uses_source_claims_and_allows_shared_reads():
    ptr = {"kind": "ptr", "to": {"kind": "const", "of": "f64"}}
    mutable = {"kind": "ptr", "to": "f64"}
    proof = {
        "rule": "exclusive_modified_bytes",
        "scope": "invocation",
        "paths": "based_on_parameter",
        "readonly_when_accessed": True,
        "basis": {
            "kind": "source_precondition",
            "rule_id": "qualified_source_pointer",
            "evidence": {"location": {"file": "unit", "line": 1}},
        },
    }
    readers = {
        "name": "readers",
        "result": "void",
        "params": [
            {"name": name, "type": ptr, "access_precondition": proof}
            for name in ("right", "left")
        ],
        "body": [{"op": "return"}],
    }
    value = {"op": "var", "name": "shared"}
    call = {"op": "call", "callee": "readers", "args": [value, value]}
    bridge = {
        "op": "call",
        "callee": "forward",
        "args": [value],
        "parameter_types": [ptr],
    }
    forward = {
        "name": "forward",
        "result": "void",
        "params": [{"name": "data", "type": mutable}],
        "body": [{"op": "return"}],
    }
    caller = {
        "name": "caller",
        "result": "void",
        "params": [{"name": "shared", "type": ptr, "elementwise_noalias": True}],
        "body": [{"op": "expr", "value": call}, {"op": "expr", "value": bridge}],
    }
    output = _layer(
        "normalization",
        """from scar.middle.normalization import normalize
module, issues = normalize(product(data,'M1'))
assert not issues
data["functions"][1]["body"][1]["value"]["parameter_types"][0] = {"kind": "ptr", "to": "f32"}
assert any(i["code"] == "argument_referent_type" for i in normalize(product(data,'M1'))[1])
print(json.dumps(module["functions"]))
""",
        {
            "functions": [readers, caller, forward],
            "globals": [],
            "records": [],
            "externs": [],
            "declarations": {},
            "abi_environment": {"architecture": "x86_64"},
        },
    )
    functions = {fn["name"]: fn for fn in output}
    contract = functions["readers"]["access_contract"]
    assert contract == {
        "semantics_version": 1,
        "scope": "invocation",
        "parameters": [
            {
                "parameter": name,
                "rule": proof["rule"],
                "region": "accessed_referent_bytes",
                "paths": proof["paths"],
                "readonly_when_accessed": True,
                "basis": proof["basis"],
            }
            for name in ("left", "right")
        ],
    }
    assert all("access_precondition" not in p for p in functions["readers"]["params"])
    assert "access_contract" not in functions["caller"]
    first, second = functions["caller"]["body"]
    assert first["value"] == call
    cast = second["value"]["args"][0]
    assert second["value"]["parameter_types"] == [mutable]
    assert cast["mode"] == "qualification" and cast["source_type"] == ptr
    assert cast["type"] == mutable and cast["value"] == value


def test_materialization_forwarding_retains_mask_order_and_requires_ownership():
    def var(name):
        return dict(op="var", name=name)

    def cell(name):
        return dict(op="index", base=var(name), indices=[var("i")])

    def assign(target, value):
        return dict(op="assign", target=target, value=value)

    def mapping(body, site):
        return dict(
            op="map",
            index="i",
            extent=var("n"),
            independent=False,
            body=body,
            site=site,
        )

    def selected(statement):
        return dict(op="if", condition=cell("mask"), then=[statement], **{"else": []})

    def buffer(name, dtype):
        return dict(
            op="buffer",
            name=name,
            type=dtype,
            extent=var("n"),
            storage="compiler_temporary",
        )

    read = dict(op="unary", operator="deref", value=var("limit"))
    mask = mapping(
        [
            assign(
                cell("mask"),
                dict(op="binary", operator="lt", left=cell("values"), right=read),
            )
        ],
        "mask-stage",
    )
    function = _canonical_function(
        dict(
            name="clamp",
            result="void",
            params=[
                dict(name="n", type="i64"),
                dict(
                    name="values",
                    type=dict(kind="ptr", to="f64"),
                    elementwise_noalias=True,
                ),
                dict(
                    name="limit", type=dict(kind="ptr", to=dict(kind="const", of="f64"))
                ),
            ],
            body=[
                dict(
                    op="block",
                    body=[
                        buffer("mask", "bool"),
                        mask,
                        buffer("scratch", "f64"),
                        mapping(
                            [selected(assign(cell("scratch"), read))], "copy-stage"
                        ),
                        mapping(
                            [selected(assign(cell("values"), cell("scratch")))],
                            "commit-stage",
                        ),
                    ],
                )
            ],
        )
    )

    def rewrite(fn):
        return _layer(
            "rewrites",
            "from scar.middle.rewrites import optimize\nchanges=optimize(data)\nprint(json.dumps(dict(function=data,changes=changes)))",
            fn,
        )

    result = rewrite(function)
    (change,) = result["changes"]
    assert change["rule"] == "pointwise_materialization_forwarding"
    body = result["function"]["body"][0]["body"]
    assert len(body) == 3 and body[1] == mask
    assert body[2]["body"][0]["then"][0]["value"] == read
    assert body[2]["independent"] is True
    # The load stays conditional: zero extents and false masks do not dereference.
    assert body[2]["body"][0]["condition"] == cell("mask")

    cases = []
    missing_alias_proof = deepcopy(function)
    missing_alias_proof["params"][1]["elementwise_noalias"] = False
    cases.append(missing_alias_proof)
    for kind in (
        "arithmetic",
        "escape",
        "different-mask",
        "shifted-index",
        "allocation",
        "conversion",
        "effect",
        "mask-escape",
        "mask-write",
        "nondominating-mask",
        "pointer-shadow",
        "pointer-rebind",
        "self-broadcast",
    ):
        unsafe = deepcopy(function)
        steps = unsafe["body"][0]["body"]
        source = steps[3]["body"][0]["then"][0]
        if kind == "arithmetic":
            source["value"] = dict(
                op="binary",
                operator="add",
                left=read,
                right=dict(op="literal", type="f64", value=1.0),
            )
        elif kind == "escape":
            steps.append(
                dict(
                    op="expr",
                    value=dict(op="call", callee="opaque", args=[var("scratch")]),
                )
            )
        elif kind == "different-mask":
            steps[4]["body"][0]["condition"] = dict(
                op="unary", operator="not", value=cell("mask")
            )
        elif kind == "shifted-index":
            steps[4]["body"][0]["then"][0]["value"]["indices"] = [
                dict(op="literal", type="i64", value=0)
            ]
        elif kind == "allocation":
            del steps[2]["storage"]
        elif kind == "conversion":
            steps[2]["type"] = "f32"
        elif kind == "effect":
            steps[3]["body"][0]["then"].insert(
                0, dict(op="expr", value=dict(op="call", callee="opaque", args=[]))
            )
        elif kind == "mask-escape":
            steps.append(
                dict(
                    op="expr",
                    value=dict(
                        op="call",
                        callee="opaque",
                        args=[dict(op="unary", operator="addr", value=cell("mask"))],
                    ),
                )
            )
        elif kind == "mask-write":
            steps.append(
                assign(cell("mask"), dict(op="literal", type="bool", value=True))
            )
        elif kind == "nondominating-mask":
            steps.append(steps.pop(1))
        elif kind in {"pointer-shadow", "pointer-rebind"}:
            unsafe["params"].append(dict(name="input", type=dict(kind="ptr", to="f64")))
            source["value"] = cell("input")
            address = dict(
                op="unary",
                operator="addr",
                value=dict(
                    op="index",
                    base=var("input"),
                    indices=[dict(op="literal", type="i64", value=1)],
                ),
            )
            rebound = (
                dict(
                    op="decl",
                    name="values",
                    type=dict(kind="ptr", to="f64"),
                    init=address,
                )
                if kind == "pointer-shadow"
                else assign(var("values"), address)
            )
            steps.insert(0, rebound)
        else:
            source["value"] = dict(op="unary", operator="deref", value=var("values"))
        cases.append(unsafe)
    assert all(not rewrite(fn)["changes"] for fn in cases)


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
    process_type = dict(kind="handle", resource="process")
    pipe_type = dict(kind="handle", resource="text_stream")
    child = dict(op="var", name="child", type=process_type)
    alias = dict(op="var", name="alias", type=process_type)
    pipe = dict(op="var", name="pipe", type=pipe_type)
    spawned = {
        **launch["value"],
        "type": process_type,
        "stdout": "pipe",
        "stderr": "stdout",
        "stdin": "inherit",
    }
    waited = dict(
        op="expr", value=dict(op="host_call", interface="process.wait", args=[alias])
    )
    started = [
        dict(
            op="expr",
            value=dict(op="host_call", interface="fs.mkdir", args=[literal("work")]),
        ),
        dict(
            op="expr",
            value=dict(
                op="host_call", interface="process.chdir", args=[literal("work")]
            ),
        ),
        start,
        write,
        close,
        dict(op="decl", name="child", type=process_type, init=spawned),
        dict(op="decl", name="alias", type=process_type, init=child),
    ]
    next_child = dict(
        op="decl", name="second", type=process_type, init=deepcopy(spawned)
    )
    lifecycle = analysis(
        [
            *started,
            dict(
                op="decl",
                name="pipe",
                type=pipe_type,
                init=dict(op="host_call", interface="process.stdout", args=[alias]),
            ),
            dict(
                op="expr",
                value=dict(
                    op="host_call",
                    interface="stream.readline",
                    args=[pipe, dict(op="literal", type="i64", value=-1)],
                ),
            ),
            waited,
            copied,
            next_child,
        ]
    )["graph"]
    launches = lifecycle["launch_environments"]
    assert len(launches) == 2 and launches[0]["directory"] == launches[1]["directory"]
    assert (
        not launches[1]["prior_live_process_effects"]
        and launches[1]["prior_external_process_effects"]
    )
    templates = {
        node["id"]: node
        for node in lifecycle["nodes"]
        if node["kind"] == "PROCESS_TEMPLATE"
    }
    completed = [
        edge
        for edge in lifecycle["edges"]
        if edge["kind"] == "COMPLETES_DIRECT_PROCESS_IF_NORMAL"
    ]
    assert len(templates) == 2 and len(completed) == 1
    assert (
        completed[0]["source"] == launches[0]["process_template"]
        and not completed[0]["descendants_completed"]
    )
    assert {
        edge.get("role")
        for edge in lifecycle["edges"]
        if edge["kind"] == "USES_PROCESS_HANDLE"
        and edge["source"] == completed[0]["source"]
    } >= {"alias", "stdout", "pipe_read", "wait"}
    assert not any(edge["kind"] == "COPIES_FILE_VERSION" for edge in lifecycle["edges"])
    for control in (
        dict(
            op="if",
            condition=dict(op="var", name="flag", type="bool"),
            then=[waited],
            **{"else": []},
        ),
        dict(
            op="while",
            condition=dict(op="var", name="flag", type="bool"),
            body=[waited],
        ),
    ):
        conditional = analysis([*started, control, next_child])["graph"]
        before, after = conditional["launch_environments"]
        assert (
            after["prior_live_process_effects"]
            and after["directory"] == before["directory"]
        )
    rebound = analysis(
        [
            *started,
            dict(
                op="assign",
                target=alias,
                value=dict(op="var", name="unknown", type=process_type),
            ),
            waited,
        ]
    )["graph"]
    assert not any(
        edge["kind"] == "COMPLETES_DIRECT_PROCESS_IF_NORMAL"
        for edge in rebound["edges"]
    )
    spawn_handle = dict(op="decl", name="child", type=process_type, init=spawned)
    process_alias = dict(op="decl", name="alias", type=process_type, init=child)
    other = dict(op="var", name="other", type=pipe_type)
    reader_open = deepcopy(opened)
    reader_open["args"][:2] = [literal("other"), literal("r")]
    scoped = analysis(
        [
            start,
            dict(
                op="resource_scope",
                resource=handle,
                body=[
                    spawn_handle,
                    process_alias,
                    dict(
                        op="if",
                        condition=dict(op="var", name="flag", type="bool"),
                        then=[],
                        **{"else": []},
                    ),
                    dict(op="decl", name="other", type=pipe_type, init=reader_open),
                    dict(op="assign", target=handle, value=other),
                ],
            ),
            waited,
            close,
        ]
    )
    scoped_nodes = {node["id"]: node for node in scoped["graph"]["nodes"]}
    (scope_close,) = [
        node
        for node in scoped_nodes.values()
        if node.get("operation") == "resource_scope_close"
    ]
    assert {
        "resource_scope_enter_succeeded",
        "body_normal_completion",
        "stream_close_succeeded",
    } <= set(scope_close["requires"])
    assert any(
        node["kind"] == "FILE_VERSION_TEMPLATE"
        and node["completion"] == scope_close["id"]
        for node in scoped_nodes.values()
    )
    assert (
        sum(
            edge["kind"] == "COMPLETES_DIRECT_PROCESS_IF_NORMAL"
            for edge in scoped["graph"]["edges"]
        )
        == 1
    )
    pipe_open = dict(
        op="decl",
        name="pipe",
        type=pipe_type,
        init=dict(op="host_call", interface="process.stdout", args=[alias]),
    )
    pipe_scope = dict(op="resource_scope", resource=pipe, body=[])
    pipe_flow = analysis([spawn_handle, process_alias, pipe_open, pipe_scope, waited])[
        "graph"
    ]
    assert any(
        edge["kind"] == "USES_PROCESS_HANDLE" and edge.get("role") == "pipe_close"
        for edge in pipe_flow["edges"]
    )
    assert (
        sum(
            edge["kind"] == "COMPLETES_DIRECT_PROCESS_IF_NORMAL"
            for edge in pipe_flow["edges"]
        )
        == 1
    )
    datasource = dict(op="var", name="datasource", type=pipe_type)
    datasource_alias = dict(op="var", name="datasource_alias", type=pipe_type)
    source_open = dict(
        op="array_source_open",
        type=pipe_type,
        source=literal("input.txt"),
        resolution=dict(
            profile="numpy-2.5.5-datasource-v1",
            mode="rt",
            encoding="source_default",
            errors="strict",
            compressed_fallback=True,
            url_cache=True,
        ),
        effects=["IO", "STATE", "MEM", "XFER"],
        writes=["filesystem", "network", "new.stream"],
    )
    datasource_scope = dict(
        op="resource_scope",
        resource=datasource_alias,
        body=[
            dict(
                op="if",
                condition=dict(op="var", name="flag", type="bool"),
                then=[],
                **{"else": []},
            ),
            dict(op="expr", value=dict(op="array_text_read", source=datasource)),
        ],
    )
    datasource_flow = analysis(
        [
            spawn_handle,
            process_alias,
            dict(op="decl", name="datasource", type=pipe_type, init=source_open),
            dict(op="decl", name="datasource_alias", type=pipe_type, init=datasource),
            datasource_scope,
            waited,
        ]
    )
    assert not any(
        row.get("field") == "resource_scope_exit"
        for row in datasource_flow["boundaries"]
    )
    assert any(
        row.get("field") == "file_read_version"
        and row.get("reason") == "array_reader_source_version_unproven"
        for row in datasource_flow["boundaries"]
    )
    assert not datasource_flow["constant_materializations"]
    assert not datasource_flow["file_templates"]
    datasource_edges = datasource_flow["graph"]["edges"]
    assert all(
        sum(edge["kind"] == kind for edge in datasource_edges) == 1
        for kind in (
            "CAPTURES_RESOURCE_IF_NORMAL",
            "RESOURCE_SCOPE_CLOSE_IF_NORMAL",
            "COMPLETES_DIRECT_PROCESS_IF_NORMAL",
        )
    )
    untyped_open = {**source_open, "type": "text"}
    untyped_source = analysis(
        [
            dict(op="decl", name="datasource", type=pipe_type, init=untyped_open),
            dict(op="decl", name="datasource_alias", type=pipe_type, init=datasource),
            datasource_scope,
        ]
    )
    assert any(
        row.get("field") == "resource_scope_exit"
        and row.get("reason") == "captured_text_stream_identity_unproven"
        for row in untyped_source["boundaries"]
    )
    for abrupt in (dict(op="return"), dict(op="throw"), dict(op="unsupported")):
        interrupted = analysis(
            [
                spawn_handle,
                process_alias,
                pipe_open,
                {**pipe_scope, "body": [abrupt]},
                waited,
            ]
        )
        assert not any(
            edge["kind"]
            in {"RESOURCE_SCOPE_CLOSE_IF_NORMAL", "COMPLETES_DIRECT_PROCESS_IF_NORMAL"}
            for edge in interrupted["graph"]["edges"]
        )
        assert any(
            row.get("field") == "resource_scope_exit"
            and row.get("reason") == "non_normal_cleanup_unmodeled"
            for row in interrupted["boundaries"]
        )
    loop_scope = analysis(
        [
            start,
            write,
            dict(
                op="while",
                condition=dict(op="var", name="flag", type="bool"),
                body=[dict(op="resource_scope", resource=handle, body=[])],
            ),
            close,
        ]
    )
    assert not loop_scope["constant_materializations"]
    assert any(
        row.get("field") == "loop_effect_summary" and row["file_havoc"]
        for row in loop_scope["boundaries"]
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


def test_mutable_sequence_constraints_preserve_construction_and_live_iteration():
    result = _layer(
        "normalization",
        """from copy import deepcopy
from scar.middle.normalization import arrays,sequences,values
from scar.middle.normalization.protocols import normalize_protocols,_Project,_Frame,_Unclosed
from scar.contract import walk_operations
def var(name,global_=False):
    return dict(op='var',name='py:test:'+name if global_ else name,**({'scope':'global'} if global_ else {}))
def lit(value):
    return dict(op='literal',value=value)
def call(target,*args):
    return dict(op='python_invoke',target=target,args=list(args),keywords=[])
def assign(name,value):
    return dict(op='assign',target=var(name,True),value=value)
def aggregate(items,site):
    return dict(op='aggregate',kind='list',items=items,site=site)
def fn(name,params,body,module=False):
    return dict(name=name,execution_kind='python_protocol',params=[dict(name=p,binding='positional') for p in params],
        python_scope=dict(kind='module' if module else 'function',module='test',main=module),
        call_protocol=dict(kind='regular'),body=body)
fill=fn('fill',['destination'],[
    dict(op='expr',value=call(dict(op='member',base=var('destination'),field='append'),lit(4))),
    dict(op='return',value=var('destination'))])
entry=fn('entry',[],[
    dict(op='python_define',symbol='fill',target=var('fill',True)),
    assign('items',aggregate([],'empty-construction')),
    assign('alias',var('items',True)),
    assign('returned',call(var('fill',True),var('alias',True))),
    assign('same',dict(op='compare_chain',left=var('items',True),operators=['is'],operands=[var('returned',True)])),
    assign('ordered',call(dict(op='python_builtin',name='sorted'),var('items',True))),
    assign('pending',aggregate([lit(1),lit(2)],'initial-pending')),
    dict(op='for',protocol='iterator',range=var('pending',True),target=var('item',True),body=[
        dict(op='expr',value=call(dict(op='member',base=var('pending',True),field='append'),var('item',True))),
        assign('pending',aggregate([lit(9)],'loop-rebinding')),dict(op='continue')],**{'else':[]}),
],True)
module=dict(entry='entry',functions=[entry,fill],globals=[],externs=[],records=[],declarations={})
ready,changes,issues=normalize_protocols(deepcopy(module))
assert not issues
assert 'm2_typevar' not in json.dumps(ready) and 'sequence_var:' not in json.dumps(ready)
closed={f['name']:f for f in ready['functions']}
seq=dict(kind='sequence_ref',element='i64')
assert closed['fill']['params'][0]['type']==closed['fill']['result']==seq
ops=list(walk_operations(closed['entry']['body']))
assert sum(n['op']=='sequence_make' and n.get('site')=='empty-construction' for n in ops)==1
loop=next(n for n in ops if n['op']=='for')
condition=loop['condition']['right']
assert condition['op']=='sequence_size' and condition['base']['name']!='py:test:pending'
assert loop['step']['op']=='update' and loop['step']['operator']=='add'
assert any(n['op']=='continue' for n in walk_operations(loop['body']))
selected=next(n for n in walk_operations(loop['body']) if n['op']=='sequence_get')
assert selected['base']==condition['base'] and selected['index']==loop['step']['target']
assert not any(n['op']=='sequence_make' and n.get('site')=='empty-construction' for n in walk_operations(loop['body']))
assert any(n['op']=='sequence_identity' for n in ops)
append=next(n for n in walk_operations(closed['fill']['body']) if n['op']=='sequence_append')
assert append['type']=='unit' and append['base']['type']==seq
assert not ready['protocol_analysis'].get('result_facts')
empty=deepcopy(module);empty['functions']=[fn('entry',[],[assign('items',aggregate([],'untyped'))],True)]
incomplete,_,issues=normalize_protocols(empty)
assert issues and issues[0]['code']=='python.sequence_element_type'
assert 'm2_typevar' not in json.dumps(incomplete.get('protocol_analysis',{}))
bad=deepcopy(module)
bad['functions'][0]['body'].append(dict(op='expr',value=call(dict(op='member',base=var('items',True),field='append'),lit('different'))))
incomplete,_,issues=normalize_protocols(bad)
assert issues and issues[0]['code']=='python.sequence_element_type'
assert 'm2_typevar' not in json.dumps(incomplete.get('protocol_analysis',{}))

shape=aggregate([lit(2),lit(3)],'shape')
reshape=fn('reshape',['array'],[dict(op='return',value=call(dict(op='member',base=var('array'),field='reshape'),shape))])
project=_Project({'reshape':reshape})
project.providers['numpy']={'version':'2.5.5'}
assert id(shape) not in project.sequences.static_lists
array=values.array_kind(1)
compiled=project.compile('reshape',(array,))
assert id(shape) in project.sequences.static_lists
assert not any(n['op']=='sequence_make' for n in walk_operations(compiled['body']))
fake=_Project({'reshape':reshape})
_Frame(fake,reshape,(values.object_kind('User'),))
assert id(shape) not in fake.sequences.static_lists

# Source lists can remain static only when every escaping use is a closed
# immediate consumer. Nested constructors and aliases follow the same graph.
np_import=dict(op='python_import',module='numpy',request={'form':'import'},
    binding=var('np',True),source_provider=dict(module='numpy',status='verified_distribution',
    distribution='numpy',version='2.5.5',sha256='a'*64,origin='/sealed/numpy/__init__.py'),effects=['CTRL','STATE','IO'])
sp_import=dict(op='python_import',module='subprocess',request={'form':'import'},
    binding=var('sp',True),source_provider=dict(module='subprocess',status='verified_stdlib',
    sha256='b'*64,origin='/sealed/subprocess.py'),effects=['CTRL','STATE','IO'])
inner=aggregate([lit(1),lit(2)],'inner');nested=aggregate([inner],'nested')
display=aggregate([lit('name'),lit(2)],'display')
argv=aggregate([lit('./worker')],'argv')
consumers=fn('entry',[],[np_import,sp_import,assign('data',nested),assign('alias',var('data',True)),
    dict(op='expr',value=call(dict(op='member',base=var('np',True),field='array'),var('alias',True))),
    dict(op='expr',value=call(dict(op='python_builtin',name='print'),display)),
    dict(op='expr',value=dict(op='process_launch',target=dict(op='member',base=var('sp',True),field='Popen'),args=[argv],keywords=[]))],True)
consumer_project=_Project({'entry':consumers})
assert {id(display),id(argv)} <= consumer_project.sequences.static_lists
assert not {id(inner),id(nested)} & consumer_project.sequences.static_lists
mutating=deepcopy(consumers)
mutating['body'].append(dict(op='expr',value=call(dict(op='member',base=var('alias',True),field='append'),lit(3))))
mutating_project=_Project({'entry':mutating})
mutating_list=mutating['body'][2]['value']
assert id(mutating_list) not in mutating_project.sequences.static_lists
shadowed=fn('entry',[],[assign('print',lit(1)),dict(op='expr',value=call(dict(op='python_builtin',name='print'),display))],True)
assert id(display) not in _Project({'entry':shadowed}).sequences.static_lists

# Read-only constant projections carry exact scalar facts. Projecting a
# container and then mutating or observing identity must taint its base too.
configuration=aggregate([lit(2),lit(4)],'configuration')
selector=lambda base,index=lit(1):dict(op='index',base=base,indices=[index])
selected=fn('entry',[],[assign('configuration',configuration),
    assign('selected',selector(var('configuration',True))),
    assign('ratio',dict(op='binary',operator='div',left=lit(1.0),right=var('selected',True))),
    assign('length',call(dict(op='python_builtin',name='len'),var('configuration',True))),
    dict(op='if',condition=var('configuration',True),then=[],**{'else':[]})],True)
selected_module=dict(entry='entry',functions=[selected],globals=[],externs=[],records=[],declarations={})
proven,_,issues=normalize_protocols(selected_module)
assert not issues
selected_ops=list(walk_operations(proven['functions'][0]['body']))
assert any(n['op']=='sequence_make' for n in selected_ops)
assert any(n['op']=='sequence_get' and n.get('may_raise') is True for n in selected_ops)
assert any(n['op']=='assign' and n['target'].get('name')=='py:test:selected' and n['value'].get('op')=='var' for n in selected_ops)
assert any(n['op']=='binary' and n['operator']=='div' for n in walk_operations(proven['functions'][0]['body']))
for consumer in (
    lambda:call(dict(op='member',base=var('inner',True),field='append'),lit(3)),
    lambda:dict(op='compare_chain',left=var('inner',True),operators=['is'],operands=[var('inner',True)]),
):
    child=aggregate([lit(1)],'child');outer=aggregate([child],'outer')
    projected=fn('entry',[],[assign('outer',outer),assign('inner',selector(var('outer',True),lit(0))),
        dict(op='expr',value=consumer())],True)
    domain=_Project({'entry':projected}).sequences
    assert not {id(child),id(outer)} & domain.static_lists
# A dynamic index still selects the runtime list path; no contents are cached.
dynamic=aggregate([lit(1),lit(2)],'dynamic-index')
dynamic_fn=fn('entry',['index'],[assign('data',dynamic),
    dict(op='expr',value=call(dict(op='python_builtin',name='print'),selector(var('data',True),var('index'))))],True)
assert id(dynamic) not in _Project({'entry':dynamic_fn}).sequences.static_lists
# Imported module fields share their declared global slot in the use graph.
settings=fn('settings',[],[dict(op='assign',target=dict(op='var',name='py:settings:configuration',scope='global'),value=configuration)],True)
settings['python_scope'].update(module='settings',main=False)
project_import=dict(op='python_import',provider='project',module='settings',request={'form':'import'},
    binding=var('settings',True),source_provider=dict(status='project',module='settings'),
    import_resolution=dict(status='project',module='settings'),bound_value=dict(**{'class':'module'},module='settings'))
imported=fn('entry',[],[project_import,assign('local',dict(op='member',base=var('settings',True),field='configuration')),
    assign('ratio',dict(op='binary',operator='div',left=lit(1.0),right=selector(var('local',True))))],True)
assert id(configuration) not in _Project({'entry':imported,'settings':settings}).sequences.static_lists
imported['body'].append(dict(op='expr',value=call(dict(op='member',base=var('local',True),field='append'),lit(9))))
assert id(configuration) not in _Project({'entry':imported,'settings':settings}).sequences.static_lists
stored=aggregate([lit(8.0),lit(9.0)],'slice-rhs')
store=fn('store',['array'],[dict(op='assign',target=dict(op='index',base=var('array'),indices=[lit(0)]),value=stored)])
store_project=_Project({'store':store})
compiled=store_project.compile('store',(values.array_kind(2),))
assert id(stored) in store_project.sequences.static_lists
assert not any(n['op']=='sequence_make' for n in walk_operations(compiled['body']))
# F3's lexical declaration is not a write of an unknown value. An actual
# unknown rebinding, in contrast, invalidates whole-function array qualification.
row_values=aggregate([lit(0.0),dict(op='index',base=var('local_array'),indices=[lit(0),lit(0)])],'mixed-row')
writer=fn('writer',[],[
    dict(op='decl',name='local_array',initial_binding='unbound',type={'kind':'python_object'}),
    dict(op='assign',target=var('local_array'),value=call(dict(op='member',base=var('np',True),field='zeros'),
        dict(op='aggregate',kind='tuple',items=[lit(2),lit(2)]))),
    dict(op='assign',target=dict(op='index',base=var('local_array'),indices=[lit(0)]),value=row_values)])
writer_entry=fn('entry',[],[np_import,dict(op='python_define',symbol='writer',target=var('writer',True)),
    dict(op='expr',value=call(var('writer',True)))],True)
writer_module=dict(entry='entry',functions=[writer_entry,writer],globals=[],externs=[],records=[],declarations={})
closed,_,issues=normalize_protocols(deepcopy(writer_module))
assert not issues and not any(n['op']=='sequence_make' for f in closed['functions'] for n in walk_operations(f['body']))
assert id(row_values) in _Project({'entry':writer_entry,'writer':writer}).sequences.static_lists
rebound=deepcopy(writer)
rebound['body'].insert(2,dict(op='assign',target=var('local_array'),value=call(var('unknown_factory'))))
rebound_values=rebound['body'][-1]['value']
assert id(rebound_values) not in _Project({'entry':writer_entry,'writer':rebound}).sequences.static_lists

frame=_Frame(_Project({}),fn('local',[],[]),())
steps=[]
source=sequences.produced(frame,{},'text',dict(op='fs_glob',type=dict(kind='sequence_ref',element='text')),steps,text_order=True)
alias=source.at('alias')
method=sequences.member(frame,{},alias,'append')
unknown=values.Value('text',dict(op='var',name='unknown',type='text'))
returned=sequences.invoke(frame,{},method,[unknown],{},steps,False)
assert returned.kind=='unit' and returned.exact is None
try:
    sequences.invoke(frame,{},values.Value('builtin','sorted'),[source],{},steps,False)
except _Unclosed as error:
    assert error.issue['code']=='python.sequence_sort'
else:
    raise AssertionError('an alias mutation retained an unproved text ordering domain')
assert not sequences.text_order_atom(unknown)
assert not sequences.text_order_atom(values.literal('\\udcc3\\udca9'))
print(json.dumps({'ready':ready['protocol_analysis']['status'],'signatures':len(ready['protocol_analysis']['closed'])}))
""",
    )
    assert result == dict(ready="READY", signatures=2)


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
from scar.contract import python_provider_id,walk_operations
results=[]
for imports in data:
    sources={}
    for node in imports:
        for source in (node['source_provider'],node['import_resolution']):
            if identity:=python_provider_id(source):
                sources[source['module']]={**source,'provider_id':identity}
    entry=dict(name='entry',params=[],body=imports,python_scope=dict(kind='module',module='test',initializer='entry',provider_sources=sources))
    project=_Project({'entry':entry})
    project.active.append('entry')
    project.imports.bind_registry('held',dict(state='ready',initializer='held:init',owner='entry'))
    frame=_Frame(project,entry,())
    steps=[]
    issue=None
    try:
        for node in imports:
            frame.imported(node,steps)
        project.observe_import('held','registry',{'site':'read-after-initializer'})
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
    producers = {
        n["source_import_occurrence"]: n
        for n in positive["operations"]
        if "source_import_occurrence" in n
    }
    checks = [n for n in positive["operations"] if n["op"] == "module_import_preserves"]
    assert len(checks) == 3 and len(producers) == 3
    for check in checks:
        producer = producers[check["occurrence"]]
        assert (
            check["writes"] == []
            and check["producer_sites"][0]["opcode"] == producer["op"]
        )
        assert check["checks"]["held"]["registry"]["site"] == "read-after-initializer"
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


def test_finite_enums_preserve_nominal_identity_order_aliases_and_guarded_projection():
    result = _layer(
        "normalization",
        """
from scar.contract import python_provider_id,walk_operations
from scar.middle.normalization import values
from scar.middle.normalization.protocols import _Project,_Frame,_Unclosed

provider=dict(module='enum', status='verified_stdlib', interpreter='cpython 3.12.3',platform='linux',
              loader='path',hash_basis='provider_file',origin='/captured/enum.py',
              sha256='c8ead615c159598370295649eb296819ad4b40d50b200c4fec2d4269bf7af9ae')
provider['provider_id']=python_provider_id(provider)
scope=dict(kind='module', module='colors', initializer='main',provider_sources={'enum':provider})
def var(name): return dict(op='var',name=name,scope='global')
def member(base,name): return dict(op='member',base=base,field=name,site=name)
def assign(name,v,i): return dict(op='assign',target=dict(op='var',name=name),value=values.literal(v).value,
                                   site='assign-'+str(i),location=dict(file='colors.py',line=i))
def definition(symbol,body):
    return dict(op='python_define_class',site='define-'+symbol,symbol=symbol,target=var(symbol),
                body_symbol=body,bases=[var('Base')],source_name=symbol,qualname=symbol,location=dict(file='colors.py',line=3),
                creation_protocol=dict(bases='resolve_supplied',metaclass='from_bases',namespace='prepare_then_execute',
                                       implicit_bindings={'__qualname__':{'source':'lexical_qualname'}}))
imp=dict(op='python_import',module='enum',member='Enum',binding=var('Base'),site='import',
         request=dict(form='from',module='enum',level=0,fromlist=['Enum']),effects=['STATE','IO'],
         source_provider={k:v for k,v in provider.items() if k!='provider_id'},
         import_resolution={k:v for k,v in provider.items() if k!='provider_id'})
body=[assign('first',True,4),assign('alias',1,5),assign('also',1.0,6),assign('word','word',7)]
n1=definition('Channel','body1'); n2=definition('Color','body2')
entry=dict(name='main',params=[],python_scope=scope,body=[imp,n1,n2])
def cls(name,owner,statements):
    return dict(name=name,params=[],owner_class=owner,python_scope=dict(kind='class',module='colors',initializer='main'),body=statements)
pr=_Project({'main':entry, 'body1':cls('body1','Channel',body), 'body2':cls('body2','Color',body)})
frame=_Frame(pr,entry,())
steps=[]
frame.imported(imp,steps)
assert pr.names['Base'].kind=='enum_base'
for d in (n1,n2): frame.store(d['target'],pr.objects.define(frame,d,steps),steps)
first=frame.evaluate(member(var('Channel'),'first'),steps)
alias=frame.evaluate(member(var('Channel'),'alias'),steps)
other=frame.evaluate(member(var('Color'),'first'),steps)
assert first.kind==alias.kind==other.kind=='enum_member' and first.value==alias.value=='first'
assert first.facts['identity']==alias.facts['identity']!=other.facts['identity']
frame.env['chosen']=alias
ref=dict(op='var',name='chosen')
name=frame.evaluate(member(ref,'name'),steps)
value=frame.evaluate(member(ref,'value'),steps)
assert name.kind=='text' and name.exact=='first' and value.kind=='bool' and value.exact is True
def rejected(code,f):
    try: f()
    except _Unclosed as error: assert error.issue['code']=='python.'+code,error.issue
    else: raise AssertionError(code)
rejected('value_contract',lambda:frame.scalar(alias,{}))
rejected('enum_member_attribute',lambda:frame.evaluate(member(ref,'__dict__'),steps))
for count,bad in enumerate([dict(op='python_define',target={'name':'__new__'}),dict(op='assign',target={'op':'var','name':'automatic'},value={'op':'call','callee':'auto'})]):
    d=definition('Bad'+str(count),'b'+str(count))
    pr.functions[d['body_symbol']]=cls(d['body_symbol'],d['symbol'],[bad])
    rejected('enum_namespace' if count==0 else 'enum_member_value',lambda:pr.objects.define(frame,d,steps))
d=definition('Meta','b3'); d['keywords']=[dict(name='metaclass',value=var('meta'))]
rejected('class_protocol',lambda:pr.objects.define(frame,d,steps))
alt={**imp,'site':'shadow','source_provider':dict(provider,status='project'),'import_resolution':dict(provider,status='project')}
rejected('import_identity',lambda:frame.imported(alt,steps))
ops=[n for step in steps for n in walk_operations(step)]
classes=[n for n in ops if n['op']=='enum_define']
assert len(classes)==2 and classes[0]['definition_capture']!=classes[1]['definition_capture']
assert [(r['name'],r['canonical'],r['ordinal'],r['value']['type'],r['location']['line']) for r in classes[0]['members']]==[
    ('first','first',0,'bool',4),('alias','first',0,'i64',5),('also','first',0,'f64',6),('word','word',1,'text',7)]
assert sum(n['op']=='module_namespace_initialize' for n in ops)==1
assert any(n['op']=='module_attribute_resolve' and n['candidate']['kind']=='source_class' for n in ops)
assert all(n['member_capture']==alias.facts['member_capture'] for n in ops if n['op']=='enum_project')
assert {'enum_define','enum_member_read','enum_projection_matches','enum_project','enum_source_continuation_required'}<=set(pr.operator_requirements()['nominal_enum']['operations'])
print(json.dumps(True))
""",
    )
    assert result


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
from scar.middle.normalization.effects import call_writes
assert call_writes(project.compile('mutate',(kind,)),project.closed)['arrays'] is None

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
assert unresolved.kind=='tuple' and values.is_sequence(mutable.kind)
assert frame.project.sequences.resolve(frame.scalar(mutable,{}))['type']['kind']=='sequence_ref'
for item in (unresolved,):
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


def test_complex_array_elements_survive_aliases_and_refuse_unqualified_operations():
    result = _layer(
        "normalization",
        """
from scar.middle.normalization import arrays, expressions, text, values
from scar.middle.normalization.protocols import _Frame, _Project, _Unclosed
from scar.contract import walk_operations

frame = _Frame(_Project({}), dict(name='field', body=[], params=[], python_scope=dict(kind='function',module='m')), ())
frame.project.providers['numpy'] = {'version':'2.5.5'}
steps=[]
def index(name, *indices):
    return dict(op='index', base=dict(op='var',name=name), indices=[dict(op='literal',value=i) for i in indices])
def invoke(target, *args, **kw):
    return arrays.construct(frame, {}, target, list(args), kw, steps, False)
def refuse(field, operation):
    try: operation()
    except _Unclosed as error: assert error.issue['code'] == 'python.' + field, error.issue
    else: raise AssertionError('expected rejection: ' + field)

dtype=arrays.provider_member(frame,{},'complex128',steps)
constructor=values.Value('array_constructor','zeros')
z=invoke(constructor, values.Value('tuple',(values.literal(2),values.literal(3))), dtype=dtype)
frame.env['z']=z
row=arrays.index(frame,index('z',1),steps)
frame.env['r']=row
cp=invoke(arrays.member(frame,{},row,'copy',steps))
reshaped=invoke(arrays.member(frame,{},row,'reshape',steps),values.literal(3),values.literal(1))
assert row.facts['array_allocation'] == reshaped.facts['array_allocation'] == z.facts['array_allocation']
assert cp.facts['array_allocation'] != row.facts['array_allocation']
scalar=values.Value('python_complex',dict(op='literal',type='c128',value=[2.0,-4.0]))
arrays.index(frame,index('r',0),steps,rhs=scalar)
read=arrays.index(frame,index('z',1,0),steps)
assert read.kind == 'numpy_c128' and values.wire(read.kind)=='c128' and values.scalar_evidence(read) is None
def arithmetic(operation, lhs, rhs):
    return expressions.apply_binary(frame,{},operation,lhs,rhs,steps)
for op in ('add','sub','mult'):
    assert arithmetic(op,row,read).kind == values.array_kind(1,'c128')
arrays.update(frame,{'operator':'add'},row,cp,steps)
assert arrays.scalar_unary(frame,{'operator':'usub'},row,steps,False).kind == row.kind
literal=invoke(values.Value('array_constructor','array'), values.Value('tuple',(scalar,values.literal(1.0))),dtype=dtype)
assert literal.kind == row.kind
ops=list(walk_operations(steps))
assert any(n['op']=='array_load' and n['type']=='c128' for n in ops)
assert any(n['op']=='array_store' and n['value']==scalar.value for n in ops)
assert any(n['op']=='array_literal' and n['values'][1]['op']=='cast' and n['values'][1]['type']=='c128' for n in ops)
assert all(n['type']['element']=='c128' for n in ops if n['op'] in {'array_copy','array_reshape','array_view','array_binary','array_unary','array_create','array_literal'})
domain=values.join_arrays((row.kind, z.kind))
assert values.rank_domain(domain)==(1,2) and values.from_wire(values.wire(domain)) == domain
assert not values.storage_compatible(domain,values.array_kind(None,'f64'))
assert values.from_wire(values.wire(values.array_kind(None,'c128'))) == values.array_kind(None,'c128')
refuse('array_comparison', lambda:arithmetic('eq',read,read))
refuse('array_operator', lambda:arithmetic('pow',row,cp))
refuse('array_operator', lambda:arithmetic('div',row,cp))
refuse('array_dtype', lambda:arithmetic('add',row,values.literal(2.0)))
refuse('array_dtype', lambda:invoke(values.Value('array_constructor','array'),row,dtype=values.Value('dtype','f64')))
refuse('array_format_dtype', lambda:text.array_text(frame,{},row,steps))
print(json.dumps(True))
""",
    )
    assert result is True


def test_fourier_contract_requires_sealed_exports_and_preserves_lazy_lookup():
    result = _layer(
        "normalization",
        """
from scar.contract import python_provider_id, walk_operations
from scar.middle.normalization import arrays, values
from scar.middle.normalization.protocols import _Frame, _Project, _Unclosed

def source(module, member, sha):
    record=dict(module=module,status="verified_distribution",distribution="numpy",version="2.5.5",
                origin="/sealed/"+member, member=member,sha256=sha,proof_scope="installed_file_with_digest")
    record["provider_id"]=python_provider_id(record)
    return record
root=source("numpy","numpy/__init__.py","09295a80660f17925ae23765ce8cbd7ff7ceae968d5f2f89349f1cb74c0b9e11")
child=source("numpy.fft","numpy/fft/__init__.py","251bd35d9a814b98e076ab2e0da2a5a7eabbfb11427a7e873ae127d5a26137c0")
leaf=source("numpy.fft._pocketfft","numpy/fft/_pocketfft.py","c809e89c05cf7b97d75abd73fdda8454403fe67772d4e435aa35bc4ce723eeb6")
child["exports"]={}
for name,line in (("fft",121),("ifft",220)):
    via=[dict(kind=kind,name=name,**{k:record[k] for k in ("module","member","provider_id","sha256")})
         for kind,record in (("from",child),("definition",leaf))]
    child["exports"][name]=dict(status="source_candidates",unresolved=[],candidates=[
        dict(leaf,name=name,line=line,kind="definition",via=via)])
scope=dict(kind="module",module="experiment",initializer="entry",
           provider_sources={"numpy":root,"numpy.fft":child})
node=dict(op="python_import",module="numpy",request=dict(form="import",module="numpy"),site="load",
          source_provider=root,effects=["IO","STATE"],binding=dict(op="var",name="np"))
fn=dict(name="entry",params=[],body=[node],python_scope=scope)
project=_Project({"entry":fn})
frame=_Frame(project,fn,())
steps=[]
row=project.source_imports.get("entry",node)
project.source_imports.begin(row,node,steps)
parent=arrays.import_provider(frame,node,steps)
namespace=arrays.provider_member(frame,{"site":"module-member"},"fft",steps,parent)
args=values.Value(values.array_kind(1,"c128"),dict(op="var",name="samples",type=values.wire(values.array_kind(1,"c128"))),project.array_state.create((7,)))
keywords={"norm":values.literal("ortho")}
def reject(code,fn):
    try: fn()
    except _Unclosed as e: assert e.issue["code"]=="python."+code, e.issue
    else: raise AssertionError(code)
for name,axis,direction in (("fft",-1,"forward"),("ifft",0,"inverse")):
    target=arrays.provider_member(frame,{"site":"attr-"+name},name,steps,namespace)
    options=dict(keywords,axis=values.literal(axis))
    out=arrays.construct(frame,{"site":"call-"+name},target,[args],options,steps,False)
    assert out.kind==args.kind and out.facts["shape"]==(7,)
    assert out.facts["array_allocation"]!=args.facts["array_allocation"]
    op=next(x for x in walk_operations(steps[-1]) if x["op"]=="array_fft")
    assert (op["direction"],op["axis"],op["normalization"],op["type"])==(direction,0,"ortho",values.wire(args.kind))
    assert op["source_binding"]==target.facts["binding"] and op["source_profile"]["definition"]["sha256"]==leaf["sha256"]
    assert op["warning"]["location"]=={"diagnostic_file":leaf["origin"],"line":101} and "IO" in op["effects"]
    assert op["empty_axis"]["exception"]=="ValueError"
reject("fourier_arguments",lambda:arrays.construct(frame,{},target,[args],dict(keywords,n=values.literal(7)),steps,False))
reject("fourier_input",lambda:arrays.construct(frame,{},target,[values.Value(values.array_kind(1),args.value)],keywords,steps,False))
reject("fourier_axis",lambda:arrays.construct(frame,{},target,[args],dict(keywords,axis=values.literal(1)),steps,False))
reject("fourier_normalization",lambda:arrays.construct(frame,{},target,[args],{},steps,False))
child["exports"]["ifft"]["candidates"][0]["sha256"]="0"*64
reject("fourier_export",lambda:arrays.provider_member(frame,{},'ifft',steps,namespace))
rows=[operation for step in steps for operation in walk_operations(step)]
assert sum(n["op"]=="module_namespace_initialize" for n in rows)==1
lookups=[n for n in rows if n["op"]=="module_attribute_resolve"]
assert [n["member"] for n in lookups]==["fft","fft","ifft"]
assert lookups[1]["namespace_binding"]["capture"]==lookups[0]["capture"]
assert [n["capture"] for n in rows if n["op"]=="module_callable_matches"]==[n["capture"] for n in lookups[1:]]
assert sum(n["op"]=="module_export_continuation_required" for n in rows)==5
assert project.operator_requirements()["fourier"]["required_provider"]["revision"]=="33ae5dc94c9cdc7f1c78346504a85de87cadaa12"
print(json.dumps(True))
""",
    )
    assert result


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
        {
            "op": "module_object_for",
            "evaluation": [copy(create(), "source-iterable")],
            "body": [copy(create(), "source-body")],
            "else": [copy(create(), "source-exhaustion")],
        },
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
    assert flow["coverage"] == dict(kind="canonical", functions=1, copies=13)
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
    assert copies["source-iterable"]["scope"] == []
    assert [s["kind"] for s in copies["source-body"]["scope"]] == ["module_object_for"]
    assert [s["kind"] for s in copies["source-exhaustion"]["scope"]] == ["exhaustion"]
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
        dict(op="assign", target=variable("a", array), value=create()),
        dict(op="module_namespace_initialize", writes=["UNKNOWN"]),
        copy(variable("a", array), "after-open-write"),
        dict(op="assign", target=variable("a", array), value=create()),
        dict(
            op="while",
            condition=variable("flag", "bool"),
            body=[
                copy(variable("a", array), "preceding-next-iteration"),
                dict(op="module_global_delete", effects=["STATE", "OPAQUE"]),
            ],
        ),
        dict(op="assign", target=variable("a", array), value=create()),
        dict(
            op="decl",
            name="formed",
            type=dict(kind="product_ref", items=[array, "unit"]),
            init=dict(
                op="product_make",
                type=dict(kind="product_ref", items=[array, "unit"]),
                items=[
                    variable("a", array),
                    dict(
                        op="module_fromlist_publish",
                        type="unit",
                        external_effect="UNKNOWN",
                    ),
                ],
            ),
        ),
        copy(
            get(variable("formed", dict(kind="product_ref", items=[array, "unit"]))),
            "during-build",
        ),
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
    for site in (
        "parameter",
        "rebound",
        "loop",
        "after-loop",
        "call",
        "after-open-write",
        "preceding-next-iteration",
        "during-build",
    ):
        assert copies[site]["byte_count"]["kind"] == "operand_nbytes"
        assert nodes[copies[site]["source"]]["region"]["storage"]["origin"] is None
    assert len(copies) == 12 and copies["loop"]["scope"][0]["kind"] == "while"


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
    from scar.contract import digest, json_bytes

    numeric = dict(
        semantics_version=1,
        scope="compilation_unit",
        source=dict(path="/removed/unit.f90", sha256="a" * 64),
        source_flags=["-O3"],
        policy=dict(math_errno=False, trapping_math=True),
        source_math_interfaces=[
            dict(
                interface="pow",
                evidence=dict(file="/removed/math.h", sha256="b" * 64, line=7),
            )
        ],
    )
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
    for function in (main, update):
        function["source_numeric_contract"] = deepcopy(numeric)
        function["numeric_contract"] = dict(
            semantics_version=1,
            scope="function",
            source_contract_sha256=digest(json_bytes(numeric)),
            permissions=dict(
                may_omit_math_errno=True,
                may_assume_default_rounding=False,
            ),
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
                path="unit.f90",
                flags=["-O3"],
                source_facts=dict(
                    files={"unit.f90": "a" * 64},
                    source_numeric_contract=numeric,
                    external_headers={
                        "/removed/math.h": dict(sha256="b" * 64, size=10)
                    },
                ),
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
    assert all(f["source_numeric_contract"] == numeric for f in bound["functions"])
    assert all(
        f["numeric_contract"] == main["numeric_contract"] for f in bound["functions"]
    )
    _layer(
        "binding",
        """from scar.middle.binding import bind
from scar.contract import read_member, member
from pathlib import Path
folder=Path(data['folder'])
for field in ('source', 'interface', 'function', 'permission'):
    graph=json.loads(json.dumps(data['graph']))
    unit=graph['units'][0]
    if field=='source': unit['source_facts']['files']['unit.f90']='c'*64
    elif field=='interface': unit['source_facts']['external_headers']['/removed/math.h']['sha256']='c'*64
    elif field=='function':
        payload=read_member(folder,unit['graph'])
        payload['functions'][0]['signature']['source_numeric_contract']['policy']['math_errno']=True
        unit['graph']=member(folder,'units',payload)
    else:
        payload=read_member(folder,unit['graph'])
        payload['functions'][0]['signature']['numeric_contract']['permissions']['may_assume_default_rounding']=True
        unit['graph']=member(folder,'units',payload)
    try: bind(graph,folder)
    except ValueError: pass
    else: raise AssertionError('inconsistent numerical '+field+' evidence accepted')
print(json.dumps({'rejected':4}))
""",
        dict(graph=graph, folder=str(tmp_path)),
    )
    graph_file.unlink()
    normalized = _layer(
        "normalization",
        """from scar.middle.normalization import normalize
module,issues=normalize(product(data,'M1'))
assert not issues
print(json.dumps(module))
""",
        bound,
    )
    assert normalized["baseline_defects"][0]["calls"] == 1
    call = normalized["functions"][0]["body"][1]["value"]
    assert call["args"][0]["op"] == "unary" and call["args"][0]["operator"] == "addr"
    assert "declarations" not in normalized and "declarations" in bound
    assert all(
        f["numeric_contract"] == main["numeric_contract"]
        for f in normalized["functions"]
    )
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
module,facts=prepare(product(data['module'],'M3'))
path=seal(module,Path(data['out']),status='READY',graph_sha256='reference',contract={})
print(json.dumps({'path':str(path),'facts':facts}))
""",
        dict(module=normalized, out=str(tmp_path / "sealed")),
    )
    product = json.loads(Path(sealed["path"]).read_bytes())
    assert product["status"] == "READY" and len(product["functions"]) == 3
    assert (
        sum(
            "source_numeric_contract"
            in json.loads((tmp_path / "sealed" / receipt["path"]).read_bytes())
            for receipt in product["functions"]
        )
        == 2
    )
    assert (
        sum(
            json.loads((tmp_path / "sealed" / receipt["path"]).read_bytes()).get(
                "numeric_contract"
            )
            == main["numeric_contract"]
            for receipt in product["functions"]
        )
        == 2
    )

    missing_argument = deepcopy(bound)
    missing_argument["functions"][0]["body"][1]["value"]["args"].clear()
    issues = _layer(
        "normalization",
        "from scar.middle.normalization import normalize\nprint(json.dumps(normalize(product(data,'M1'))[1]))",
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
        [
            dict(
                op="expr",
                value=dict(
                    op="call",
                    callee="borrow",
                    args=[dict(op="unary", operator="addr", value=var("n"))],
                    parameter_types=[dict(kind="ptr", to="i64")],
                ),
            )
        ],
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

    element = dict(op="index", base=var("out"), indices=[var("n")])
    for borrow in (
        dict(
            op="expr",
            value=dict(
                op="call",
                callee="borrow",
                args=[dict(op="unary", operator="addr", value=element)],
                parameter_types=[dict(kind="ptr", to="i64")],
            ),
        ),
        dict(op="decl", name="r", type=ref, init=element),
        dict(
            op="expr",
            value=dict(
                op="library_call", args=[element], parameter_modes=["reference"]
            ),
        ),
    ):
        indexed = deepcopy(coordinate)
        indexed["body"][-1]["extent"] = var("n")
        indexed["body"][2:2] = [borrow]
        indexed = _canonical_function(indexed)
        assert "n" in indexed["state_contract"]["private_scalars"]
        assert "out" in indexed["state_contract"]["escaped"]
        changed = _layer(
            "rewrites",
            "from scar.middle.rewrites import optimize\nprint(json.dumps(optimize(data)))",
            indexed,
        )
        assert any(c["rule"] == "bounded_integer_coordinate" for c in changed)

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


def test_file_candidates_preserve_provenance_without_reviving_current_versions():
    def literal(value):
        return dict(op="literal", type="text", value=value)

    def host(interface, *args):
        return dict(op="host_call", interface=interface, args=list(args))

    def effect(interface, *args):
        return dict(op="expr", value=host(interface, *args))

    stream = dict(op="var", name="stream")

    def writer(path, value, mode="w"):
        return [
            dict(
                op="decl",
                name="stream",
                init=host(
                    "stream.open",
                    literal(path),
                    literal(mode),
                    literal("utf-8"),
                    literal("universal"),
                ),
            ),
            effect("stream.write", stream, literal(value)),
            effect("stream.close", stream),
        ]

    console = dict(
        op="expr",
        value=dict(
            op="text_print",
            args=[literal("log")],
            sep=literal(" "),
            end=literal("\n"),
        ),
    )
    request = dict(
        op="expr",
        value=dict(
            op="process_request",
            command=dict(form="argv", args=[]),
            cwd="inherit",
            stdin="inherit",
            stdout="inherit",
            stderr="inherit",
        ),
    )
    flag = dict(op="var", name="flag", type="bool")
    result = _layer(
        "normalization",
        """from scar.middle.normalization.artifacts import analyze
from scar.contract import validate_artifact_flow
result = analyze(data)
validate_artifact_flow(result['graph'])
print(json.dumps(result))
""",
        dict(
            functions=[
                dict(
                    name="known",
                    body=[
                        *writer("stable", "initial"),
                        *writer("fresh", "fresh", "x"),
                        request,
                    ],
                ),
                dict(
                    name="conditional",
                    params=[dict(name="flag", type="bool")],
                    body=[
                        *writer("stable", "initial"),
                        console,
                        effect("fs.copy_file", literal("stable"), literal("copied")),
                        dict(
                            op="if",
                            condition=flag,
                            then=writer("stable", "branch overwrite"),
                            **{"else": []},
                        ),
                        dict(op="while", condition=flag, body=[console]),
                        request,
                    ],
                ),
                dict(
                    name="overwritten",
                    body=[
                        *writer("stable", "initial"),
                        console,
                        *writer("stable", "replacement"),
                        request,
                    ],
                ),
            ]
        ),
    )
    graph = result["graph"]
    records = {node["id"]: node for node in graph["nodes"]}
    launches = {row["symbol"]: row for row in graph["launch_environments"]}

    def version(symbol, text):
        return next(
            row["version"]
            for row in result["constant_materializations"]
            if row["symbol"] == symbol and records[row["value"]]["literal_text"] == text
        )

    known = launches["known"]
    assert version("known", "initial") in known["visible_versions"]
    assert version("known", "fresh") in known["visible_versions"]
    assert known["conditional_candidates"] == []
    conditional = launches["conditional"]
    initial = version("conditional", "initial")
    assert initial not in conditional["visible_versions"]
    (candidate,) = [
        row
        for row in conditional["conditional_candidates"]
        if row["version"] == initial
    ]
    assert candidate["invalidated_by"] and candidate["requires"]
    assert all(
        records[event]["operation"] == "file_state_barrier"
        for event in candidate["invalidated_by"]
    )
    assert "candidate_available_on_selected_normal_predecessor" in candidate["requires"]
    assert any(
        edge["kind"] == "MAY_SATISFY_FILE_VERSION"
        and edge["source"] == initial
        and records[edge["target"]]["role"] == "runtime_copy_source"
        and edge["establishes_current_version"] is False
        for edge in graph["edges"]
    )
    assert any(
        edge["kind"] == "CONDITIONALLY_AVAILABLE_BEFORE_LAUNCH"
        and edge["source"] == initial
        and edge["target"] == conditional["operation"]
        and edge["establishes_child_read"] is False
        for edge in graph["edges"]
    )
    assert (
        sum(
            node.get("operation") == "while" and node.get("symbol") == "conditional"
            for node in records.values()
        )
        == 1
    )
    overwritten = launches["overwritten"]
    old = version("overwritten", "initial")
    assert old not in overwritten["visible_versions"]
    assert old not in {row["version"] for row in overwritten["conditional_candidates"]}
    assert version("overwritten", "replacement") in overwritten["visible_versions"]
    assert any(
        edge["kind"] == "INVALIDATES_FILE_VERSION" and edge["target"] == old
        for edge in graph["edges"]
    )


def test_file_derivations_consume_final_interfaces_and_identify_partial_prefixes():
    pointer = dict(kind="ptr", to=dict(kind="const", of="char"))
    reference = dict(kind="ref", to=dict(kind="const", of="char"))

    def literal(text):
        return dict(op="literal", type="text", value=text)

    def host(interface, *args):
        return dict(op="host_call", interface=interface, args=list(args))

    system = dict(
        op="call",
        callee="system",
        site="system-call",
        type="i32",
        args=[dict(op="var", name="command", type=reference)],
        parameter_types=[reference],
        artifact_effect=dict(
            semantics_version=1,
            interface="process.system",
            operands=dict(command=["args", 0]),
            defaults={},
            obligations=[],
            contract=dict(
                semantics_version=1,
                source_interface="system",
                count_unit="bytes",
                null_command="query_command_processor_availability",
                command_read="through_first_nul_when_nonnull",
                process_effect="external_command_processor_when_nonnull",
                return_value="implementation_status_not_direct_exit_code",
            ),
        ),
    )
    stream = dict(op="var", name="stream")
    writer = dict(
        name="writer",
        result="void",
        params=[],
        body=[
            dict(
                op="decl",
                name="stream",
                type=dict(kind="handle", resource="text_stream"),
                init=host(
                    "stream.open",
                    literal("header"),
                    literal("w"),
                    literal("utf-8"),
                    literal("universal"),
                ),
            ),
            dict(op="expr", value=host("stream.write", stream, literal("constant"))),
            dict(op="expr", value=host("stream.close", stream)),
        ],
    )
    main = dict(
        name="main",
        result="void",
        params=[dict(name="command", type=reference)],
        body=[dict(op="expr", value=system)],
    )
    module = dict(
        entry="main",
        functions=[main, writer],
        globals=[],
        records=[],
        declarations={},
        abi_environment=dict(architecture="x86_64"),
        externs=[
            dict(
                name="system",
                result="i32",
                params=[dict(name="command", type=pointer)],
                provider="libc",
                header="stdlib.h",
                variadic=False,
            )
        ],
    )
    result = _layer(
        "normalization",
        """from copy import deepcopy
from scar.middle.contracts import publish_program, require_program
from scar.middle.normalization import normalize
bound = publish_program(data, 'M1', complete=True)
module, issues = normalize(bound)
assert not issues
require_program(module, 'M2')
assert bound['functions'][0]['body'][0]['value']['parameter_types'] != module['functions'][0]['body'][0]['value']['parameter_types']
assert 'runtime_boundary' not in bound['functions'][0]['body'][0]['value']
upstream_partial, upstream_issues = normalize(publish_program(deepcopy(bound), 'M1', complete=False))
assert not upstream_issues
partial = deepcopy(bound)
partial['entry'] = 'entry'
partial['functions'].append(dict(
    name='entry', result='void', params=[], execution_kind='python_protocol',
    python_scope=dict(kind='module', module='fixture', main=True),
    body=[dict(op='expr', value=dict(op='call', target=dict(op='python_builtin', name='print'), args=[dict(op='literal', type='text', value='prefix')], keywords=[])),
          dict(op='expr', value=dict(op='var', name='not_bound'))],
))
partial = publish_program(partial, 'M1', complete=True)
opened, unresolved = normalize(partial)
assert unresolved
require_program(opened, 'M2')
print(json.dumps(dict(complete=module, partial=opened, upstream_partial=upstream_partial)))
""",
        module,
    )
    complete, partial = result["complete"], result["partial"]
    upstream_partial = result["upstream_partial"]
    assert upstream_partial["layer_contract"]["completion"] == "partial"
    assert upstream_partial["artifact_analysis"]["basis"]["phase"] == "interface_prefix"
    assert not upstream_partial["artifact_analysis"]["basis"]["complete"]
    call = complete["functions"][0]["body"][0]["value"]
    assert call["args"][0]["op"] == "unary" and call["args"][0]["operator"] == "addr"
    assert call["parameter_types"] == [pointer]
    assert call["runtime_boundary"]["profile"] == "posix_libc_system_v1"
    assert complete["artifact_analysis"]["buffers"]["coverage"]["qualified_calls"] == 1
    assert "command" in complete["functions"][0]["state_contract"]["escaped"]
    complete_basis = complete["artifact_analysis"]["basis"]
    assert complete_basis["phase"] == "canonical" and complete_basis["complete"]
    assert {
        "object_interfaces",
        "abi_declarations",
        "call_arguments",
        "mpi_effects",
        "frame_state",
    } <= set(complete_basis["completed_passes"])
    assert complete_basis == complete["artifact_analysis"]["buffers"]["basis"]
    assert (
        complete_basis
        == complete["resource_model"]["artifact_flow"]["coverage"]["derivation_basis"]
    )
    partial_basis = partial["artifact_analysis"]["basis"]
    assert partial_basis["phase"] == "protocol_prefix" and not partial_basis["complete"]
    assert partial["layer_contract"]["completion"] == "partial"
    assert "call_arguments" not in partial_basis["completed_passes"]
    assert partial["protocol_analysis"]["prefix_fragments"]
    assert any(
        row["symbol"] == "writer"
        for row in partial["artifact_analysis"]["constant_materializations"]
    )


def test_import_state_snapshots_share_guards_and_keep_epochs_monotone():
    from scar.middle.normalization.import_state import ImportState, namespace_slot

    state = ImportState(("driver", "package"), main_namespace="driver")
    assert state.registry_key("driver") == "__main__"
    assert state.namespace_owner("__main__") == "driver"
    proof = dict(state="ready", initializer="package:init")
    state.bind_registry("package", proof)
    state.bind_registry("__main__", dict(state="initializing", initializer="entry"))
    slot = namespace_slot("package", "value")
    state.binding_written(slot)
    marker, operation = dict(op="block", body=[]), dict(op="module_import_preserves")
    state.bind_windows(
        "package",
        [(marker, operation, proof, state.snapshot()["binding_epoch"])],
        proof,
    )
    before = state.snapshot()
    copied = state.copy_snapshot(before)
    copied["registry"].clear()
    copied["import_windows"]["package"].clear()
    assert state.registry_entry("package") == proof
    assert state.windows("package")[0][0] is marker
    state.binding_written(slot)
    later_epoch = state.binding_version(slot)
    state.restore(before)
    assert state.binding_version(slot) < later_epoch
    state.binding_written(slot)
    assert state.binding_version(slot) > later_epoch

    # The state tables fork independently; the deferred source guard stays one
    # operation rather than being copied into competing producers.
    marker["op"] = "if"
    assert state.windows("package")[0][0]["op"] == "if"
    merged = state.join_snapshots([before, state.copy_snapshot(before)])
    assert len(merged["import_windows"]["package"]) == 1
    assert merged["import_windows"]["package"][0][1] is operation
    changed = state.invalidate("entry", dict(site="opaque"))
    after = state.snapshot()
    assert state.registry_entry("package")["state"] == "unresolved"
    assert state.namespace_change("__main__") is changed
    assert state.windows("package") == ()
    assert before["registry"]["package"] == proof
    normal = state.join_snapshots([before, after])
    assert normal["registry"]["package"]["state"] == "unresolved"
    assert not normal["import_windows"]
    state.bind_windows("package", [(marker, operation, proof, later_epoch)], proof)
    guarded = state.snapshot()
    for paths in ([guarded, guarded], [before, guarded]):
        joined = state.join_snapshots(paths)
        assert joined["registry"]["package"]["state"] == "unresolved"
        assert joined["import_windows"]["package"][0][1] is operation
    assert not state.join_snapshots([guarded, after])["import_windows"]
    replacement = dict(state="ready", initializer="replacement:init")
    state.bind_windows(
        "package", [(marker, operation, replacement, later_epoch)], replacement
    )
    mismatch = state.snapshot()
    assert not state.join_snapshots([guarded, mismatch])["import_windows"]
    exceptional = state.exceptional_prefix(before, before, may_write=True)
    assert set(exceptional["registry"]["package"]["possible_states"]) == {
        "absent",
        "initializing",
        "ready",
        "unresolved",
    }
    assert not exceptional["import_windows"]
    assert (
        state.exceptional_prefix(before, before, may_write=False)["registry"]
        == before["registry"]
    )
    state.bind_registry("missing", dict(state="absent"))
    latest = state.invalidate("entry", dict(site="next_opaque"))
    assert state.registry_entry("missing")["state"] == "unresolved"
    assert state.registry_entry("package")["epoch"] > changed["epoch"]
    assert state.registry_entry("package")["site"] == "next_opaque"
    assert state.registry_entry("package") is latest


def test_call_writes_follow_actual_ports_and_open_callbacks():
    from scar.middle.normalization import effects, values
    from scar.middle.normalization.protocols import _Frame, _Project

    def var(name):
        return dict(op="var", name=name)

    def fn(name, params, body):
        return dict(
            name=name,
            params=[dict(name=p, binding="positional") for p in params],
            body=body,
            python_scope=dict(kind="function", module="m"),
            call_protocol=dict(kind="regular"),
        )

    # Two nested calls may touch the second actual via a local view and append
    # to the third actual. Neither touches the first argument's allocation.
    leaf = fn(
        "leaf",
        ["a", "s"],
        [
            dict(op="array_store", base=var("a")),
            dict(op="sequence_append", base=var("s")),
        ],
    )
    outer = fn(
        "outer",
        ["first", "second", "list"],
        [
            dict(op="decl", name="row", init=dict(op="array_view", base=var("second"))),
            dict(op="call", callee="leaf", args=[var("row"), var("list")]),
        ],
    )
    table = {"leaf": leaf, "outer": outer}
    assert effects.call_writes(outer, table) == {"arrays": {1}, "sequences": {2}}
    first = values.array_kind(1)
    kinds = (first, first, values.sequence_kind("f64"))
    caller = fn("caller", [], [])
    project = _Project(
        {"caller": caller, "outer": fn("outer", ["first", "second", "list"], [])}
    )
    # Use a sealed signature and the actual M2 call handler, so the check
    # includes caller-side argument substitution and subsequent cached calls.
    outer["params"] = [
        dict(p, type=values.wire(k)) for p, k in zip(outer["params"], kinds)
    ]
    outer["result"] = "unit"
    project.closed.update(table)
    project.types["outer"] = kinds
    project.result_kinds["outer"] = "unit"
    project.result_facts["outer"] = {}
    frame = _Frame(project, caller, ())
    bound = [
        values.Value(first, None, project.array_state.create((1,), [i])).at(name)
        for i, name in enumerate(("x", "y"))
    ]
    seq = values.Value(kinds[2], None, project.sequences.register(True)).at("s")
    frame.invoke_project({}, "outer", [*bound, seq], {}, [], True)
    assert (
        project.array_state.region(bound[0]) is not None
        and project.array_state.region(bound[1]) is None
    )
    assert not project.sequences.safe(seq)
    # Deferred effects can become visible on an already compiled body. No
    # earlier readonly/parameter-only decision may survive as a cached proof.
    for annotation in (
        {"external_effect": "UNKNOWN"},
        {"writes": ["UNKNOWN"]},
        {"effects": ["OPAQUE"]},
        {"pending_import_effect": True},
    ):
        leaf["body"].append(dict(op="external_action", **annotation))
        assert effects.call_writes(outer, table) == {"arrays": None, "sequences": None}
        frame.invoke_project({}, "outer", [*bound, seq], {}, [], True)
        assert project.array_state.region(bound[0]) is None
        leaf["body"].pop()
    leaf["body"] = [
        dict(op="decl", name="tmp", init=dict(op="array_create")),
        dict(op="array_store", base=var("tmp")),
    ]
    assert effects.call_writes(outer, table) == {"arrays": set(), "sequences": set()}


def test_captured_effects_retain_conditional_state_checks_without_reexecution():
    from scar.middle.normalization.protocols import _Project, _Frame

    function = dict(
        name="entry",
        params=[],
        body=[],
        result="void",
        python_scope=dict(kind="module", module="entry", main=True),
    )
    project = _Project({"entry": function})
    project.active.append("entry")
    frame = _Frame(project, function, ())
    proof = dict(state="ready", initializer="package:init")
    project.imports.bind_registry("package", proof)
    project.array_state.contents[1] = []
    project.sequences.contents[1] = ()
    steps = []
    result = frame.effect(
        dict(
            op="captured_action",
            capture="lookup/1",
            writes=["import.external_effects"],
            effects=["STATE", "OPAQUE"],
            external_effect="UNKNOWN",
        ),
        "bool",
        steps,
    )
    assert result.kind == "bool"
    assert not project.array_state.contents and not project.sequences.contents
    assert steps[0]["init"]["op"] == "captured_action"
    assert steps[0]["init"]["registry_state_capture"]["identity"] == "lookup/1"
    assert project.imports.registry_entry("package")["state"] == "unresolved"
    assert len(project.imports.windows("package")) == 1
    assert project.registry_entry("package", {}) == proof
    assert steps[1]["condition"]["op"] == "module_state_preserves"
    fallback = steps[1]["else"][0]["value"]
    assert fallback["op"] == "state_effect_continuation_required"
    assert fallback["producer_capture"] == "lookup/1"
    assert (
        fallback["resume"]
        == "continue_after_original_action_with_captured_result_and_actual_post_state"
    )
    frame.effect(
        dict(op="opaque_action", writes=["import.external_effects"]), "unit", [], True
    )
    assert not project.imports.windows("package")
    assert project.imports.registry_entry("package")["state"] == "unresolved"


def test_import_effect_summaries_keep_named_ports_and_unknown_callbacks():
    from scar.middle.normalization.host_protocols import ImportOccurrences

    node = dict(
        op="python_import",
        site="entry/import",
        module="toolkit.worker",
        request=dict(
            form="from",
            module="toolkit.worker",
            level=0,
            fromlist=["answer", "child", "answer", "*"],
            binding_index=0,
        ),
        binding=dict(name="answer"),
        member="answer",
        effects=["CTRL", "STATE", "IO"],
    )
    ledger = ImportOccurrences({})
    row = ledger.get("entry", node)
    effect = row["import_effects"]
    assert effect["direct_registry_keys"] == ["toolkit", "toolkit.worker"]
    assert effect["conditional_registry_keys"] == [
        "toolkit.worker.answer",
        "toolkit.worker.child",
    ]
    assert effect["effect_coverage"] == "UNPROVEN"
    producer = dict(op="module_namespace_initialize", operator_family="module_registry")
    ledger.producer(row, producer, "source_import")
    assert producer["import_effects"] is effect
    assert producer["source_import_occurrence"] == row["id"]
    assert effect["effect_coverage"] == "MODELED"
    assert effect["external_effect"] == effect["other_registry_writes"] == "UNKNOWN"
    assert effect["namespace_writes"]["other"] == "UNKNOWN"
    assert effect["producer_sites"][0]["opcode"] == producer["op"]


def test_captured_effects_chain_across_module_completion_and_normal_branches():
    from scar.middle.normalization.protocols import _Project, _Frame

    function = dict(
        name="entry",
        params=[],
        body=[],
        result="unit",
        python_scope=dict(kind="module", module="entry"),
    )
    project = _Project({"entry": function})
    project.active.append("entry")
    frame = _Frame(project, function, ())
    state = project.imports
    initializing = dict(state="initializing", initializer="entry")
    ready = dict(initializing, state="ready")
    state.bind_registry("entry", initializing)

    def action(capture):
        steps = []
        frame.effect(
            dict(
                op="captured_action",
                capture=capture,
                writes=["import.external_effects"],
            ),
            "unit",
            steps,
            True,
        )
        return steps

    early = action("initializer")
    state.bind_registry("entry", ready)
    first = action("first")
    start = state.snapshot()
    branch = action("branch")
    fork = state.snapshot()
    state.restore(state.join_snapshots([start, fork]))
    last = action("last")
    assert state.registry_entry("entry")["state"] == "unresolved"
    assert project.registry_entry("entry", {})["state"] == "unresolved"
    project.observe_import("entry", "namespace", {}, "worker")
    for steps, expected in [
        (early, initializing),
        (first, ready),
        (branch, ready),
        (last, ready),
    ]:
        producer, guard = steps
        checks = guard["condition"]["checks"]["entry"]
        assert checks["before"] == expected
        assert checks["namespace"]["first_member"] == "worker"
        assert "registry" not in checks
        fallback = guard["else"][0]["value"]
        assert fallback["producer_capture"] == producer["value"]["capture"]
        assert fallback["op"] == "state_effect_continuation_required"
    # An uncaptured callback or an exceptional prefix on just one branch
    # invalidates the shared continuation, regardless of predecessor order.
    state.restore(start)
    frame.effect(dict(op="opaque_action", writes=["UNKNOWN"]), "unit", [], True)
    opaque = state.snapshot()
    exceptional = state.exceptional_prefix(start, fork, may_write=True)
    for unknown in (opaque, exceptional):
        for paths in ([unknown, fork], [fork, unknown]):
            state.restore(state.join_snapshots(paths))
            action("later")
            assert not state.windows("entry")
            assert project.registry_entry("entry", {})["state"] == "unresolved"


def test_repeated_import_checks_publication_across_initializer_and_audit_hook(tmp_path):
    from scar.contract import walk_operations
    from scar.frontend import build
    from scar.middle.binding import bind
    from scar.middle.normalization.protocols import normalize_protocols

    entry = tmp_path / "entry.py"
    (tmp_path / "dep.py").write_text("import math\ncount = 12\n")
    entry.write_text(
        "import subprocess\nimport dep\n"
        f"child = subprocess.Popen([{sys.executable!r}, '-c', 'pass'])\n"
        "child.wait()\nimport dep as again\nprint(again.count)\n"
    )
    assert subprocess.check_output([sys.executable, "-B", entry]) == b"12\n"
    hooked = """import runpy, sys, types
def replace(event, args):
    if event == 'subprocess.Popen':
        dep = types.ModuleType('dep')
        dep.count = 73
        sys.modules['dep'] = dep
sys.addaudithook(replace)
runpy.run_path('entry.py', run_name='__main__')
"""
    assert (
        subprocess.check_output([sys.executable, "-B", "-c", hooked], cwd=tmp_path)
        == b"73\n"
    )
    graph = tmp_path / "graph"
    build(entry, graph)
    module, issues, _ = bind(json.loads((graph / "graph.json").read_bytes()), graph)
    assert not issues
    normal, _, issues = normalize_protocols(module)
    assert not issues
    operations = list(walk_operations(normal["functions"]))
    (producer,) = [x for x in operations if x.get("callee") == "py:dep:<init>"]
    receipt = producer["module_initialization"]
    assert receipt["publication"]["occurrence"] == producer["source_import_occurrence"]
    guards = {
        n["condition"]["op"]: n
        for n in operations
        if n["op"] == "if" and "dep" in n["condition"].get("checks", {})
    }
    early = guards["module_import_preserves"]
    launch = guards["module_state_preserves"]
    early_check = early["condition"]["checks"]["dep"]
    launch_check = launch["condition"]["checks"]["dep"]
    assert early_check["before"]["publication"] == receipt["publication"]
    assert (
        early_check["registry"]["requires"] == "same_module_instance_and_initializing"
    )
    assert launch_check["before"]["completed_from"] == early_check["before"]
    assert launch_check["registry"]["requires"] == "same_module_instance_and_complete"
    assert early["else"][0]["value"]["op"] == "module_import_continuation_required"
    fallback = launch["else"][0]["value"]
    assert fallback["op"] == "state_effect_continuation_required"
    assert fallback["producer_capture"] == launch["condition"]["producer_capture"]


def test_completion_cannot_link_a_replaced_publication_or_replayed_initializer():
    from scar.middle.normalization.import_state import ImportState

    state = ImportState(("dep",))

    def start():
        return state.begin_initializer("dep", "dep_init", "entry", "import-site", {})

    opening = start()
    windows = [({}, {"checks": {}}, opening, 0)]
    state.bind_windows("dep", windows, opening)
    replacement = start()
    call = dict(callee="dep_init", source_import_occurrence="import-site")
    state.complete_initializer("dep", opening, call, analyzed_body=True)
    assert "module_initialization" not in call
    assert not state.covers_registry_lookup("dep", state.registry_entry("dep"), windows)
    state.bind_registry("dep", replacement)
    state.complete_initializer("dep", replacement, call, analyzed_body=False)
    assert "module_initialization" not in call
    state.bind_registry("dep", replacement)
    state.complete_initializer("dep", replacement, call, analyzed_body=True)
    assert not state.covers_registry_lookup("dep", state.registry_entry("dep"), windows)


@pytest.mark.parametrize(
    ("updates", "bounded"),
    [
        ("if count % 2:\n    count += 1", True),
        (f"if count % 2:\n    count = {10**640}", False),
        ("for i in range(3):\n    count = count * count", False),
    ],
)
def test_decimal_global_uses_ranges_from_every_initializer_store(
    tmp_path, updates, bounded
):
    from scar.contract import walk_operations
    from scar.frontend import build
    from scar.middle.binding import bind
    from scar.middle.normalization.protocols import normalize_protocols

    (tmp_path / "settings.py").write_text(
        "def truncate(value):\n    return int(value)\ncount = truncate(1e300)\n"
        + updates
        + "\ndef show():\n    return str(count)\n"
    )
    entry = tmp_path / "entry.py"
    entry.write_text("import settings\nprint(settings.show())\n")
    graph = tmp_path / "graph"
    build(entry, graph)
    module, issues, _ = bind(json.loads((graph / "graph.json").read_bytes()), graph)
    assert not issues
    normalized, _, issues = normalize_protocols(module)
    if bounded:
        assert not issues
        (display,) = [
            op
            for op in walk_operations(normalized["functions"])
            if op.get("operation") == "decimal"
        ]
        assert display["maximum_digits"] == 309
    else:
        assert [issue["code"] for issue in issues] == ["python.integer_text_limit"]


def test_exact_text_methods_preserve_binding_unicode_and_fresh_containers():
    from types import SimpleNamespace
    import unicodedata

    from scar.middle.normalization import sequences, text, values

    profile = dict(
        interpreter="cpython 3.12.3", unicode_version=unicodedata.unidata_version
    )
    domain, steps = sequences.Domains({}, {}), []

    def fail(node, field, detail):
        raise ValueError(field)

    def effect(op, kind, steps, *, facts):
        steps.append(op)
        return values.Value(kind, None, facts).at(f"v{len(steps)}")

    frame = SimpleNamespace(
        name="f",
        scope={"text_profile": "source"},
        fail=fail,
        effect=effect,
        project=SimpleNamespace(
            text_profiles={"source": profile}, decisions=[], sequences=domain
        ),
    )

    def invoke(s, method, *args, **kwargs):
        node = {"site": "test"}
        target = text.member(frame, node, values.literal(s), method)
        return text.invoke(
            frame,
            node,
            target,
            [values.literal(x) for x in args],
            {k: values.literal(v) for k, v in kwargs.items()},
            steps,
        )

    first = invoke("a:b:c", "split", sep=":", maxsplit=1)
    second = invoke("a:b:c", "split", sep=":", maxsplit=1)
    assert first.value["name"] != second.value["name"]
    assert [op["op"] for op in steps] == ["sequence_make", "sequence_make"]
    assert [item["value"] for item in steps[0]["items"]] == ["a", "b:c"]
    assert domain.resolve_kind(
        invoke("", "split").kind, strict=True
    ) == values.sequence_kind("text")
    partition = invoke("a:b:c", "rpartition", ":")
    assert values.product_members(partition.kind) == ("text", "text", "text")
    assert steps[-1]["identity"] == {"kind": "fresh"} and [
        r["value"] for r in steps[-1]["items"]
    ] == ["a:b", ":", "c"]
    assert (
        invoke("Straße", "casefold").exact == "strasse"
        and invoke("λ", "isalpha").exact is True
    )
    with pytest.raises(ValueError, match="text_intrinsic_arguments"):
        invoke("abc", "replace", "a", "x", count=1)
    profile["interpreter"] = "cpython 3.13.0"
    assert invoke("aa", "replace", "a", "x", count=1).exact == "xa"
    profile["unicode_version"] = "unknown"
    with pytest.raises(ValueError, match="text_case_profile"):
        invoke("\u2003", "strip")
    assert invoke("λxλ", "rfind", "λ").exact == 2
    with pytest.raises(ValueError, match="text_intrinsic_arguments"):
        invoke("abc", "split", "")


def test_mapping_domain_keeps_tags_shared_handles_and_live_iteration():
    from scar.contract import walk_operations
    from scar.middle.normalization import mappings, values

    class Project:
        text_profiles = {"unicode": {}}

        def __init__(self):
            self.operations = set()

        def require_operator(self, family, opcode, **contract):
            assert family == "builtin_scalar_mapping"
            assert contract == {"profile": "builtin-scalar-map-v1"}
            self.operations.add(opcode)

    class Frame:
        def __init__(self):
            self.project, self.scope = Project(), {"text_profile": "unicode"}
            self.environment, self.events, self.ordinal = {}, [], 0

        def fail(self, node, field, message):
            raise ValueError(field + ": " + message)

        def effect(self, operation, kind, steps, discard=False, *, facts=None):
            self.events.append(operation["op"])
            if discard:
                steps.append(dict(op="expr", value=operation))
                return values.literal(None)
            self.ordinal += 1
            name = f"temporary{self.ordinal}"
            steps.append(
                dict(op="decl", name=name, type=values.wire(kind), init=operation)
            )
            return values.Value(kind, None, facts or {}).at(name)

        def evaluate(self, node, steps):
            if node["op"] == "probe":
                self.events.append(node["name"])
                return values.literal(node["value"])
            if node["op"] == "var":
                return self.environment[node["name"]]
            if node["op"] == "mapping":
                return mappings.literal(self, node, steps)
            assert node["op"] == "literal"
            return values.literal(node["value"])

    def source(entries):
        return dict(
            op="mapping",
            entries=entries,
            construction=dict(
                status="captured",
                steps=[
                    dict(
                        kind="build_pairs",
                        first=0,
                        count=len(entries),
                        constant_keys=False,
                    )
                ],
            ),
        )

    frame, steps = Frame(), []
    child = mappings.literal(frame, source([]), steps)
    frame.environment["shared"] = child
    members = [None, True, 7, -0.0, "text"]
    entries = [
        dict(
            kind="pair",
            key=dict(op="probe", name=f"key{i}", value=f"k{i}"),
            value=dict(op="probe", name=f"value{i}", value=value),
        )
        for i, value in enumerate(members)
    ]
    entries.extend(
        dict(
            kind="pair",
            key=dict(op="literal", value=name),
            value=dict(op="var", name="shared"),
        )
        for name in ("left", "right")
    )
    entries.append(
        dict(
            kind="pair",
            key=dict(op="literal", value="k0"),
            value=dict(op="literal", value=False),
        )
    )
    outer = mappings.literal(frame, source(entries), steps)
    operations = list(walk_operations(steps))
    packs = [op for op in operations if op["op"] == "structured_pack"]
    assert {op["tag"] for op in packs} == {
        "unit",
        "bool",
        "i64",
        "f64",
        "text",
        "mapping_ref",
    }
    assert sum(op["op"] == "mapping_make" for op in operations) == 2
    assert len(
        next(op for op in operations if op["op"] == "mapping_make" and op["entries"])[
            "entries"
        ]
    ) == len(entries)
    assert all(op["value"] == child.value for op in packs if op["tag"] == "mapping_ref")
    assert [event for event in frame.events if event.startswith(("key", "value"))] == [
        name for i in range(len(members)) for name in (f"key{i}", f"value{i}")
    ]
    assert frame.events.index("value4") < len(frame.events) - 1
    assert outer.kind == "mapping_ref" and outer.exact is values._UNKNOWN
    alias = outer.at("alias")
    first = mappings.get(frame, {}, alias, values.literal("k2"), steps)
    mappings.put(frame, {}, alias, values.literal("k2"), values.literal(9), steps)
    second = mappings.get(frame, {}, alias, values.literal("k2"), steps)
    assert first.kind == second.kind == "structured_value"
    assert (
        first.value != second.value and first.exact is second.exact is values._UNKNOWN
    )
    method = mappings.member(frame, {}, alias, "items", steps)
    view = mappings.invoke(frame, {}, method, [], {}, steps)
    loop, item, _, _, _, _ = mappings.iteration(frame, {}, view, steps)
    assert loop["condition"]["op"] == "mapping_next"
    assert loop["condition"]["may_raise"] and loop["step"] is None
    assert item.value["op"] == "mapping_current"
    assert values.product_members(item.kind) == ("text", "structured_value")
    assert item.value["base"] == loop["condition"]["base"]
    origin = dict(owner="walk", parameter="data")
    formal = values.Value("structured_value", None, {"mapping_origin": origin}).at(
        "data"
    )
    checked = mappings.as_mapping(frame, {}, formal, steps, access="items")
    assert mappings.pack(frame, {}, checked).facts == {"mapping_origin": origin}
    method = mappings.member(frame, {}, checked, "items", steps)
    rooted_view = mappings.invoke(frame, {}, method, [], {}, steps)
    _, descendant, _, _, _, _ = mappings.iteration(frame, {}, rooted_view, steps)
    assert descendant.facts["product_members"] == [
        {},
        {"mapping_descendant_of": origin},
    ]
    nested_view = values.Value(
        "mapping_items_ref", rooted_view.value, {"mapping_descendant_of": origin}
    )
    _, nested, _, _, _, _ = mappings.iteration(frame, {}, nested_view, steps)
    assert nested.facts == descendant.facts
    for kind in (
        "mapping_ref",
        "structured_value",
        "mapping_items_ref",
        "mapping_iterator_ref",
    ):
        assert values.runtime(kind) and values.from_wire(values.wire(kind)) == kind
    assert mappings.pack(frame, {}, first) is first
    assert (
        mappings.compare(
            frame, {"operator": "eq"}, first, values.literal("x"), steps
        ).kind
        == "bool"
    )
    with pytest.raises(ValueError, match="mapping_value"):
        mappings.pack(frame, {}, values.Value("numpy_f64", values.literal(1.0).value))
    with pytest.raises(ValueError, match="mapping_key"):
        mappings.get(frame, {}, alias, values.literal(1), steps)
    with pytest.raises(ValueError, match="mapping_acyclicity"):
        mappings.put(frame, {}, alias, values.literal("cycle"), alias, steps)
    with pytest.raises(ValueError, match="mapping_compare"):
        mappings.compare(frame, {"operator": "eq"}, alias, alias, steps)
    with pytest.raises(ValueError, match="mapping_insertion_schedule"):
        mappings.literal(frame, dict(op="mapping", entries=[]), steps)


def test_empty_counted_regions_keep_iterator_effects_and_skip_back_edge_facts():
    result = _layer(
        "normalization",
        """from scar.middle.normalization import arrays,sequences,values
from scar.middle.normalization.protocols import _Project,_Frame,_Unclosed
from scar.contract import walk_operations
def var(name): return dict(op='var',name=name)
def lit(value): return dict(op='literal',value=value)
def call(target,*args): return dict(op='python_invoke',target=target,args=list(args),keywords=[])
def assign(name,value): return dict(op='assign',target=var(name),value=value)
def fixture():
    fn=dict(name='caller',params=[],python_scope=dict(kind='function',module='test'),body=[])
    project=_Project({'caller':fn})
    frame=_Frame(project,fn,())
    facts=project.array_state.create((1,),[3.])
    frame.env['array']=values.Value(values.array_kind(1),None,facts).at('array')
    steps=[]
    frame.env['items']=sequences.capture(frame,{},[values.literal(4)],steps).at('items')
    frame.env['target']=values.literal(7).at('target')
    frame.env['seen']=values.literal(-1).at('seen')
    return project,frame,steps
def stores():
    return [dict(op='assign',target=dict(op='index',base=var('array'),indices=[lit(0)]),value=lit(9.)),
            dict(op='expr',value=call(dict(op='member',base=var('items'),field='append'),lit(5)))]
def loop(domain,body,otherwise=()):
    return dict(op='for',protocol='iterator',range=domain,target=var('target'),body=body,**{'else':list(otherwise)})
for arguments in [(0,),(3,3),(0,4,-1)]:
    project,frame,steps=fixture()
    args=[lit(value) for value in arguments]
    args[0]=dict(op='assignment_expression',target=var('seen'),value=args[0])
    domain=call(dict(op='python_builtin',name='range'),*args)
    body=stores()+[dict(op='expr',value=call(var('unreachable_callback')),site='dead-body')]
    source=loop(domain,body,[assign('else_ran',lit(True))])
    assert frame.loop(source,steps)
    assert frame.env['seen'].exact==arguments[0] and frame.env['target'].exact==7
    assert frame.env['else_ran'].exact is True
    assert project.array_state.region(frame.env['array'])==((1,),(3.,))
    assert project.sequences.contents[frame.env['items'].facts['sequence_state']][0].exact==4
    output=next(n for n in steps if n['op']=='for')
    assert not output['body']
    assert ('caller','dead-body') in project.skipped_sites
    assert next(i for i,n in enumerate(steps) if n['op']=='assign' and n['target']['name']=='seen') < steps.index(output)
    assert steps.index(output) < next(i for i,n in enumerate(steps) if n['op']=='assign' and n['target']['name']=='else_ran')

# A zero-trip body does not bind a fresh target, but its else suite still writes.
project,frame,steps=fixture()
del frame.env['target']
assert frame.loop(loop(call(dict(op='python_builtin',name='range'),lit(0)),stores(),stores()),steps)
assert 'target' not in frame.env and not project.array_state.contents
assert not project.sequences.contents
assert any(n['op']=='array_store' for n in steps)

# An iterator error is not hidden by discarding its unreachable body.
project,frame,steps=fixture()
try: frame.loop(loop(call(dict(op='python_builtin',name='range'),lit(0),lit(0),lit(0)),stores()),steps)
except _Unclosed as error: assert error.issue['code']=='python.range_step'
else: raise AssertionError('invalid iterator was skipped')

# Potentially nonempty ranges still invalidate repeated writes.
project,frame,steps=fixture()
frame.env['stop']=values.Value('i64',None,dict(range=(-1,1))).at('stop')
assert frame.loop(loop(call(dict(op='python_builtin',name='range'),var('stop')),stores()),steps)
assert not project.array_state.contents and not project.sequences.contents

# Empty comprehensions evaluate the outer iterator and allocate the result,
# without binding the induction variable or touching filters/inner iterators.
project,frame,steps=fixture()
outer_stop=dict(op='index',base=dict(op='aggregate',kind='tuple',identity=dict(kind='fresh'),
    items=[call(dict(op='python_builtin',name='print'),lit('iterator')),lit(0)]),indices=[lit(1)])
source=dict(op='sequence_comprehension',kind='list',scope=dict(id='comp'),
    generators=[dict(target=var('index'),iterable=call(dict(op='python_builtin',name='range'),outer_stop),
                     filters=[call(var('unreachable_filter'))]),
                dict(target=var('inner'),iterable=call(var('unreachable_iterator')),filters=[])],
    element=call(var('unreachable_element')),site='empty-comprehension')
result=frame.comprehension(source,steps)
assert 'index' not in frame.env and 'inner' not in frame.env
assert project.array_state.region(frame.env['array'])==((1,),(3.,))
assert project.sequences.contents[frame.env['items'].facts['sequence_state']][0].exact==4
assert project.sequences.contents[result.facts['sequence_state']]==()
operations=list(walk_operations(steps))
assert sum(n['op']=='text_print' for n in operations)==1
assert next(i for i,n in enumerate(steps) if any(op['op']=='text_print' for op in walk_operations(n))) < next(i for i,n in enumerate(steps) if any(op['op']=='sequence_make' and op.get('site')=='empty-comprehension' for op in walk_operations(n)))
assert not next(n for n in steps if n['op']=='for')['body']
# A later real append supplies the element constraint, never a virtual body run.
sequences.invoke(frame,{},values.Value('sequence_method',(result,'append')),[values.literal(2)],{},steps,True)
assert project.sequences.element(result.kind)=='i64'
print(json.dumps(True))
""",
    )
    assert result


def test_function_defaults_capture_once_and_keep_managed_identity():
    from types import SimpleNamespace

    from scar.middle.normalization import objects, values

    parameters = [
        dict(
            name="data", binding="posonly", default_from=dict(field="defaults", index=0)
        ),
        dict(
            name="receiver",
            binding="positional",
            default_from=dict(field="defaults", index=1),
        ),
        dict(name="required", binding="keywordonly", required=True),
        dict(
            name="mode",
            binding="keywordonly",
            default_from=dict(field="keyword_defaults", index=1),
        ),
    ]
    function = dict(
        name="defined",
        params=parameters,
        body=[],
        python_scope=dict(kind="function", module="module"),
        call_protocol=dict(kind="regular"),
    )
    source_values = {
        "mapping": values.Value(
            "mapping_ref", dict(op="var", name="source_map"), {"mapping_builder": True}
        ),
        "object": values.Value(
            values.object_kind("Thing"), dict(op="var", name="source_object")
        ),
        "mode": values.literal("stdout"),
    }

    class Frame:
        initializing, dynamic, module = True, 0, "module"

        def __init__(self):
            self.project = SimpleNamespace(
                functions={},
                function_definitions={
                    "defined": {
                        key: value for key, value in function.items() if key != "body"
                    }
                },
                function_defaults={},
                globals={},
            )
            self.evaluated = []

        def evaluate(self, node, steps):
            self.evaluated.append(node["name"])
            return source_values[node["name"]]

        def fail(self, node, field, message):
            raise ValueError(field + ": " + message)

    frame, steps = Frame(), []
    definition = dict(
        op="python_define",
        symbol="defined",
        target=dict(op="var", name="defined", scope="global"),
        defaults=[dict(name="mapping"), dict(name="object")],
        keyword_defaults=[None, dict(name="mode")],
    )
    result = objects.bind_function_definition(frame, definition, steps)
    assert result == values.Value("function", "defined")
    assert frame.evaluated == ["mapping", "object", "mode"]
    captured = frame.project.function_defaults["defined"]
    assert set(captured) == {"data", "receiver", "mode"}
    assert captured["data"].kind == "mapping_ref"
    assert captured["receiver"].kind == values.object_kind("Thing")
    assert captured["data"].facts == {}  # Publication ends the private builder proof.
    assert captured["mode"].exact == "stdout"
    assert all(value.value["scope"] == "global" for value in captured.values())
    assert [step["value"] for step in steps] == [
        source_values[name].value for name in frame.evaluated
    ]
    assert frame.project.globals == {
        value.value["name"]: value.kind for value in captured.values()
    }
    source_values["mapping"] = values.Value(
        "mapping_ref", dict(op="var", name="new_map")
    )
    assert captured["data"].value["name"] == steps[0]["target"]["name"]
    with pytest.raises(ValueError, match="function_redefinition"):
        objects.bind_function_definition(frame, definition, steps)
    assert frame.evaluated == ["mapping", "object", "mode"] and len(steps) == 3
    for unsupported in ("annotations", "decorators", "type_params"):
        with pytest.raises(ValueError, match="function_creation"):
            objects.bind_function_definition(
                frame, {**definition, unsupported: [{}]}, []
            )
    frame.dynamic = 1
    with pytest.raises(ValueError, match="function_creation"):
        objects.bind_function_definition(frame, definition, [])


def test_sequence_literal_initializers_follow_aliases_and_invalidation():
    from types import SimpleNamespace

    from scar.middle.normalization import sequences, values

    domain, steps = sequences.Domains({}, {}), []

    def effect(operation, kind, body, discard=False, *, facts=None):
        body.append(operation)
        if discard:
            return values.literal(None)
        return values.Value(kind, None, facts or {}).at(f"value{len(body)}")

    def fail(node, field, detail):
        raise ValueError(field)

    frame = SimpleNamespace(
        name="entry",
        project=SimpleNamespace(sequences=domain),
        effect=effect,
        scalar=lambda value, node: value.value,
        fail=fail,
    )

    def capture(items):
        return sequences.capture(frame, {"site": "literal"}, items, steps)

    def get(base, index):
        result = sequences.get(frame, {}, base, values.literal(index))
        assert result.value["op"] == "sequence_get"
        assert result.value["may_raise"] is True
        return result

    def append(base, value):
        return sequences.invoke(
            frame,
            {},
            values.Value("sequence_method", (base, "append")),
            [values.literal(value)],
            {},
            steps,
            False,
        )

    first = capture([values.literal("a"), values.literal("b")])
    alias = first.at("alias")
    second = capture([values.literal("a"), values.literal("b")])
    assert first.value != second.value
    assert first.facts["sequence_state"] != second.facts["sequence_state"]
    assert get(alias, 0).exact == "a" and get(alias, -1).exact == "b"
    assert get(alias, 2).exact is values._UNKNOWN
    assert append(alias, "c").exact is None
    assert get(first, -1).exact is values._UNKNOWN
    assert get(second, -1).exact == "b"
    assert domain.safe(first) and domain.safe(second)

    unknown = values.Value(second.kind, None).at("unknown_receiver")
    append(unknown, "d")
    assert get(second, 0).exact is values._UNKNOWN and not domain.contents
    fresh = capture([values.literal(7)])
    assert get(fresh, 0).exact == 7
    domain.invalidate_contents()
    assert get(fresh, 0).exact is values._UNKNOWN
    ordered = capture([values.literal("x")])
    domain.invalidate_contents()
    assert domain.safe(ordered) and get(ordered, 0).exact is values._UNKNOWN
    fresh = capture([values.literal(8)])
    domain.invalidate()
    assert get(fresh, 0).exact is values._UNKNOWN
    bounded = capture([values.literal(i) for i in range(65)])
    assert bounded.facts["sequence_state"] not in domain.contents
    dynamic = capture([values.Value("text", None).at("runtime_text")])
    metadata = domain.contents[dynamic.facts["sequence_state"]]
    assert len(metadata) == 1 and metadata[0].kind == "text"
    assert metadata[0].value is None and metadata[0].facts == {}
    assert get(dynamic, 0).exact is values._UNKNOWN
    assert sum(operation["op"] == "sequence_make" for operation in steps) == 7


def test_finite_scalar_selections_propagate_ranges_without_replacing_winners():
    import math
    import operator
    import struct
    from types import SimpleNamespace

    from scar.middle.normalization import scalars, values

    steps, decisions = [], []

    def effect(operation, kind, body, discard=False, *, facts=None):
        name = f"selected{len(body)}"
        body.append(dict(op="decl", name=name, init=operation))
        return values.Value(kind, None, facts or {}).at(name)

    def fail(node, field, detail):
        raise ValueError(field)

    frame = SimpleNamespace(
        name="entry",
        project=SimpleNamespace(decisions=decisions),
        effect=effect,
        fail=fail,
    )

    def variable(name, kind, bounds=None):
        return values.Value(
            kind,
            dict(op="var", name=name, type=values.wire(kind)),
            {} if bounds is None else {"range": bounds},
        )

    def select(name, args):
        return scalars.extreme(frame, {}, name, args, {}, steps)

    inputs = [
        variable("a", "f64", (-10.0, 10.0)),
        variable("b", "numpy_f64", (2.0, 4.0)),
        variable("c", "f64", (-2.0, 6.0)),
    ]
    high, low = select("max", inputs), select("min", inputs)
    assert high.kind == low.kind == "binary64_choice"
    assert high.facts == {"range": (2.0, 10.0)}
    assert low.facts == {"range": (-10.0, 4.0)}
    assert scalars.construct(frame, {}, "int", [high], {}).facts == {"range": (2, 10)}
    difference = scalars.binary(
        frame,
        {},
        "sub",
        scalars.construct(frame, {}, "float", [high], {}),
        scalars.construct(frame, {}, "float", [low], {}),
    )
    assert difference.facts == {"range": (-2.0, 20.0)}
    assert scalars.construct(frame, {}, "int", [difference], {}).facts == {
        "range": (-2, 20)
    }
    assert all(step["init"]["op"] == "select" for step in steps)
    assert all(step["init"]["condition"]["operator"] in {"gt", "lt"} for step in steps)

    integers = [variable("i", "i64", (-100, 100)), variable("j", "i64", (3, 4))]
    assert select("min", integers).facts == {"range": (-100, 4)}
    assert select("max", integers).facts == {"range": (3, 100)}

    # A finite clamp endpoint cannot make an unbounded, NaN or infinite second
    # argument finite; first-winner unordered comparisons remain real selects.
    for unknown in (
        variable("unbounded", "f64"),
        variable("maybe_array_scalar", "binary64_choice"),
        values.literal(math.nan),
        values.literal(math.inf),
    ):
        result = select("max", [values.literal(0.0), unknown])
        assert result.facts == {}
        frame.project.require_operator = lambda family, operation, **contract: None
        frame.project.text_profiles = {}
        frame.scope = {}
        widened = scalars.construct(
            frame,
            {"source_runtime": {"interpreter": "cpython 3.12.3"}},
            "int",
            [result],
            {},
        )
        assert widened.kind == "integer" and widened.value["operation"] == "from_f64"
        assert widened.value["may_raise"] is True
        assert widened.value["exception_messages"] == {
            "value_error": "cannot convert float NaN to integer",
            "overflow_error": "cannot convert float infinity to integer",
        }

    start = len(steps)
    zeros = [
        variable("left", "f64", (-0.0, 0.0)),
        variable("right", "numpy_f64", (-0.0, 0.0)),
    ]
    first = select("max", zeros)
    assert first.kind == "binary64_choice" and first.exact is values._UNKNOWN

    def evaluate(node, environment):
        if node["op"] == "var":
            return environment[node["name"]]
        if node["op"] == "binary":
            return {"gt": operator.gt, "lt": operator.lt}[node["operator"]](
                evaluate(node["left"], environment),
                evaluate(node["right"], environment),
            )
        assert node["op"] == "select"
        side = "then" if evaluate(node["condition"], environment) else "else"
        return evaluate(node[side], environment)

    for left, right in ((0.0, -0.0), (-0.0, 0.0)):
        environment = dict(left=left, right=right)
        for step in steps[start:]:
            environment[step["name"]] = evaluate(step["init"], environment)
        assert struct.pack("!d", environment[first.value["name"]]) == struct.pack(
            "!d", max(left, right)
        )


def test_unbounded_integer_contracts_keep_ranges_effects_and_conversion_errors():
    import math
    from types import SimpleNamespace

    from scar.middle.normalization import (
        expressions,
        integers,
        products,
        protocols,
        scalars,
        values,
    )

    requirements, steps = [], []

    def require(family, operation, **contract):
        requirements.append((family, operation, contract))

    def effect(operation, kind, body, discard=False, *, facts=None):
        body.append(operation)
        return values.Value(kind, None, facts or {}).at(f"captured{len(body)}")

    def fail(node, field, detail):
        raise ValueError(field)

    frame = SimpleNamespace(
        name="entry",
        scope={},
        project=SimpleNamespace(require_operator=require, text_profiles={}),
        effect=effect,
        fail=fail,
    )
    node = {"source_runtime": {"interpreter": "cpython 3.12.3"}, "site": "conversion"}
    incoming = values.Value("f64", dict(op="var", name="input", type="f64"))
    converted = scalars.construct(frame, node, "int", [incoming], {})
    maximum = ((1 << 53) - 1) << 971
    assert converted.kind == values.wire(converted.kind) == "integer"
    assert converted.bounds == (-maximum, maximum) and maximum.bit_length() == 1024
    assert len(str(maximum)) == 309 and converted.exact is values._UNKNOWN
    assert converted.value["operation"] == "from_f64"
    assert converted.value["may_raise"] is True
    assert converted.value["exception_messages"] == {
        "value_error": "cannot convert float NaN to integer",
        "overflow_error": "cannot convert float infinity to integer",
    }
    captured = scalars.construct(frame, node, "int", [incoming], {}, steps=steps)
    assert captured.value["op"] == "var" and steps[-1]["operation"] == "from_f64"
    count = len(steps)
    assert (
        scalars.construct(frame, node, "int", [captured], {}, steps=steps) == captured
    )
    assert len(steps) == count
    bounded = values.Value("f64", incoming.value, {"range": (-7.9, 8.9)})
    narrow = scalars.construct(frame, node, "int", [bounded], {})
    assert narrow.kind == "i64" and narrow.bounds == (-7, 8)
    for source in (math.nan, math.inf, -math.inf, float(1 << 63)):
        result = scalars.construct(frame, node, "int", [values.literal(source)], {})
        assert result.kind == "integer" and result.value["may_raise"]
    unbounded = values.Value(
        "integer", dict(op="var", name="unbounded", type="integer")
    )
    assert unbounded.bounds is None and scalars._binary64_bounds(unbounded) is None

    positive = integers.literal(frame, node, 1 << 100, steps=steps)
    negative = integers.literal(frame, node, -(1 << 100), steps=steps)
    assert steps[-2]["args"][0]["value"] == str(1 << 100)
    assert steps[-1]["args"][0]["value"] == str(-(1 << 100))
    assert scalars.unary(frame, {"operator": "usub"}, positive).exact == -(1 << 100)
    assert scalars.construct(frame, {}, "abs", [negative], {}).exact == 1 << 100
    added = scalars.binary(frame, node, "add", positive, values.literal(3), steps)
    assert added.kind == "integer" and added.exact == (1 << 100) + 3
    doubled = scalars.binary(frame, node, "mult", converted, values.literal(2), steps)
    assert doubled.bounds == (-maximum * 2, maximum * 2)
    quotient = scalars.binary(frame, node, "floordiv", converted, unbounded, steps)
    assert quotient.bounds == (-maximum, maximum)
    modulo = scalars.binary(frame, node, "mod", unbounded, values.literal(-7), steps)
    assert modulo.bounds == (-6, 0) and steps[-1]["may_raise"]
    assert steps[-1]["exception_messages"] == {
        "zero_division_error": "integer modulo by zero"
    }
    zero = scalars.binary(frame, node, "floordiv", positive, values.literal(0), steps)
    assert zero.kind == "integer" and steps[-1]["may_raise"]
    assert steps[-1]["exception_messages"] == {
        "zero_division_error": "integer division or modulo by zero"
    }
    newer = {"source_runtime": {"interpreter": "cpython 3.14.0"}}
    scalars.binary(frame, newer, "mod", positive, values.literal(0), steps)
    assert steps[-1]["exception_messages"] == {
        "zero_division_error": "division by zero"
    }
    before = len(steps)
    comparison = scalars.comparison(frame, node, "gt", converted, positive, steps)
    assert comparison.kind == "bool" and comparison.value["op"] == "var"
    assert [operation["operation"] for operation in steps[before:]] == [
        "from_f64",
        "gt",
    ]
    assert steps[-1]["may_raise"] is False
    assert all(arg["op"] == "var" for arg in steps[-1]["args"])

    for integer in ((1 << 53) + 1, -((1 << 53) + 3)):
        wide = integers.literal(frame, node, integer)
        rounded = scalars.construct(frame, node, "float", [wide], {})
        assert rounded.kind == "f64" and rounded.exact == float(integer)
        assert rounded.value["operation"] == "to_binary64"
    overflow = scalars.construct(
        frame, node, "float", [integers.literal(frame, node, 1 << 1100)], {}
    )
    assert overflow.exact is values._UNKNOWN and overflow.value["may_raise"]
    assert overflow.value["exception_messages"] == {
        "overflow_error": "int too large to convert to float"
    }
    mixed = scalars.binary(frame, node, "add", positive, values.literal(1.0), steps)
    assert mixed.kind == "f64" and steps[-1]["operation"] == "to_binary64"
    with pytest.raises(ValueError, match="integer_operand"):
        scalars.binary(frame, node, "lt", positive, values.literal(1.0), steps)
    with pytest.raises(ValueError, match="integer_source_version"):
        scalars.construct(frame, {}, "int", [incoming], {})
    assert all(
        family == "integer" and contract == {"semantics_version": 1}
        for family, _, contract in requirements
    )
    assert {operation for _, operation, _ in requirements} >= {
        "from_f64",
        "decimal_literal",
        "from_i64",
        "add",
        "mul",
        "floordiv",
        "mod",
        "gt",
        "negative",
        "absolute",
        "to_binary64",
    }
    bounded = values.Value("integer", unbounded.value, {"range": (-10, 20)})
    other = values.Value("integer", unbounded.value, {"range": (30, 40)})
    assert expressions._join_facts([bounded, other]) == {"range": (-10, 40)}
    assert expressions._join_facts([bounded, unbounded]) == {}
    joined = protocols._join([{"number": bounded}, {"number": unbounded}])
    assert joined["number"].bounds is None
    assert products.returned_facts(bounded) == {"range": (-10, 20)}
    returns = SimpleNamespace(returns=set(), return_facts=None)
    protocols._Frame.record_return(returns, bounded)
    protocols._Frame.record_return(returns, other)
    assert returns.return_facts == {"range": (-10, 40)}
    protocols._Frame.record_return(returns, unbounded)
    assert returns.return_facts == {}


def test_private_pointwise_snapshots_require_real_non_escape_and_injective_indices():
    def var(name):
        return dict(op="var", name=name)

    def literal(value):
        return dict(op="literal", type="i64", value=value)

    def binary(op, left, right):
        return dict(op="binary", operator=op, left=left, right=right)

    # The Fortran-style address is mechanically shown to be i, not selected
    # by a variable name or supplied human-readable proof string.
    index = binary(
        "sub",
        dict(
            op="cast",
            type="i64",
            value=binary("add", literal(1), binary("mul", literal(1), var("i"))),
        ),
        literal(1),
    )

    def cell(name, at=None):
        return dict(
            op="index", base=var(name), indices=[deepcopy(index if at is None else at)]
        )

    def mapping(target, value):
        return dict(
            op="map",
            index="i",
            extent=var("n"),
            independent=True,
            body=[dict(op="assign", target=target, value=value)],
        )

    destination = dict(op="buffer", name="local", type="f64", extent=var("n"))
    temporary = dict(
        op="buffer",
        name="snapshot",
        type="f64",
        extent=var("n"),
        storage="compiler_temporary",
    )
    rhs = binary(
        "div",
        binary(
            "sub", cell("input", binary("add", var("i"), literal(1))), cell("input")
        ),
        cell("local"),
    )
    source = mapping(cell("snapshot", var("i")), rhs)
    copy = mapping(cell("local"), cell("snapshot", var("i")))
    function = _canonical_function(
        dict(
            name="update",
            params=[
                dict(name="n", type="i64"),
                dict(name="input", type=dict(kind="ptr", to="f64")),
            ],
            result="void",
            body=[destination, dict(op="block", body=[temporary, source, copy])],
        )
    )

    def rewrite(fn):
        return _layer(
            "rewrites",
            "from scar.middle.rewrites import optimize\nchanges=optimize(data)\nprint(json.dumps(dict(function=data,changes=changes)))",
            fn,
        )

    result = rewrite(function)
    edits = [
        c
        for c in result["changes"]
        if c["rule"] == "pointwise_materialization_forwarding"
    ]
    assert len(edits) == 1 and edits[0]["storage_proof"]["index_coefficient"] == 1
    body = result["function"]["body"][1]["body"]
    assert len(body) == 1 and body[0]["body"][0]["target"] == cell("local")
    assert body[0]["body"][0]["value"] == rhs
    for mutation in (
        "shifted-self",
        "address-escape",
        "raw-escape",
        "reference-escape",
        "global-homonym",
        "another-consumer",
        "not-private",
        "not-temporary",
        "noninjective",
        "different-element",
        "opaque",
    ):
        bad = deepcopy(function)
        steps = bad["body"][1]["body"]
        if mutation == "shifted-self":
            steps[1]["body"][0]["value"]["right"] = cell(
                "local", binary("add", var("i"), literal(1))
            )
        elif mutation == "address-escape":
            bad["body"].append(
                dict(
                    op="expr",
                    value=dict(
                        op="call",
                        callee="opaque",
                        args=[dict(op="unary", operator="addr", value=cell("local"))],
                    ),
                )
            )
        elif mutation == "raw-escape":
            bad["body"].append(
                dict(
                    op="expr",
                    value=dict(op="call", callee="opaque", args=[var("local")]),
                )
            )
        elif mutation == "reference-escape":
            bad["body"].append(
                dict(
                    op="expr",
                    value=dict(
                        op="call",
                        callee="save_ref",
                        parameter_modes=["reference"],
                        args=[cell("local", literal(0))],
                    ),
                )
            )
        elif mutation == "global-homonym":
            foreign = cell("local", literal(0))
            foreign["base"]["scope"] = "global"
            bad["body"].append(
                dict(
                    op="expr",
                    value=dict(
                        op="call",
                        callee="save_ref",
                        args=[var("local"), foreign],
                    ),
                )
            )
        elif mutation == "another-consumer":
            steps.append(dict(op="expr", value=cell("snapshot", literal(0))))
        elif mutation == "not-private":
            bad["body"] = bad["body"][1:]
            bad["params"].append(dict(name="local", type=dict(kind="ptr", to="f64")))
        elif mutation == "not-temporary":
            steps[0].pop("storage")
        elif mutation == "noninjective":
            steps[2]["body"][0]["target"] = cell("local", literal(0))
        elif mutation == "different-element":
            steps[0]["type"] = "f32"
        else:
            steps[1]["body"][0]["value"] = dict(op="call", callee="unknown", args=[])
        assert not any(
            c["rule"] == "pointwise_materialization_forwarding"
            for c in rewrite(bad)["changes"]
        ), mutation


def test_counted_loop_only_drops_remainders_with_a_stable_induction_domain():
    from scar.middle.rewrites import optimize

    def var(name):
        return {"op": "var", "name": name}

    def constant(n):
        return {"op": "literal", "type": "i64", "value": n}

    coordinate = dict(op="binary", operator="mod", left=var("i"), right=var("n"))
    loop = dict(
        op="for",
        init=[dict(op="decl", name="i", type="i64", init=constant(0))],
        condition=dict(op="binary", operator="lt", left=var("i"), right=var("n")),
        step=dict(op="unary", operator="preinc", value=var("i")),
        body=[dict(op="assign", target=var("sum"), value=coordinate)],
    )
    function = _canonical_function(
        dict(
            name="reduce",
            params=[dict(name="n", type="i64")],
            result="i64",
            body=[
                dict(op="decl", name="sum", type="i64", init=constant(0)),
                loop,
                dict(op="return", value=var("sum")),
            ],
        )
    )

    def reduced(fn):
        changes = optimize(fn)
        return any(change["rule"] == "bounded_integer_coordinate" for change in changes)

    assert reduced(deepcopy(function))
    for protected in ("i", "n"):
        mutated = deepcopy(function)
        mutated["body"][1]["body"].insert(
            0, dict(op="assign", target=var(protected), value=constant(-1))
        )
        assert not reduced(mutated)
        escaped = deepcopy(function)
        escaped["state_contract"]["private_scalars"].remove(protected)
        assert not reduced(escaped)
    negative = deepcopy(function)
    negative["body"][1]["init"][0]["init"] = constant(-1)
    assert not reduced(negative)
    backward = deepcopy(function)
    backward["body"][1]["step"]["operator"] = "predec"
    assert not reduced(backward)
    narrowed = deepcopy(function)
    narrowed["body"][1]["body"][0]["value"]["type"] = "i32"
    assert not reduced(narrowed)
    shadow = deepcopy(function)
    shadow["body"][1]["body"].insert(
        0, dict(op="decl", name="n", type="i64", init=constant(1))
    )
    assert not reduced(shadow)


def test_math_operand_sharing_preserves_effectful_calls_and_control():
    from scar.middle.contracts import publish_program
    from scar.middle.rewrites import rewrite

    arg = dict(
        op="cast",
        mode="static",
        type="f64",
        value=dict(
            op="binary",
            operator="band",
            left=dict(op="var", name="x"),
            right=dict(op="literal", type="u32", value=31),
        ),
    )

    def call(symbol, site, operand=arg):
        return dict(
            op="expr",
            value=dict(
                op="call",
                callee=symbol,
                site=site,
                args=[deepcopy(operand)],
                parameter_types=["f64"],
            ),
        )

    original = [call("cos", "first"), call("sin", "second")]

    def module(body, domain="u32"):
        fn = dict(
            name="run",
            params=[dict(name="x", type=domain)],
            result="void",
            body=body,
            state_contract=dict(
                private_scalars=["x"], scalars={"x": dict(type=domain)}, escaped=[]
            ),
        )
        return publish_program(
            dict(
                entry="run",
                functions=[fn],
                globals=[],
                records=[],
                externs=[
                    dict(
                        name=n,
                        provider="libm",
                        params=[dict(name="v", type="f64")],
                        result="f64",
                    )
                    for n in ("cos", "sin")
                ],
            ),
            "M2",
            complete=True,
        )

    result, changes = rewrite(module(deepcopy(original)))
    fn = result["functions"][0]
    decl, first, second = fn["body"]
    assert decl["init"] == arg and decl["type"] == "f64"
    assert [v["value"]["callee"] for v in (first, second)] == ["cos", "sin"]
    assert (
        first["value"]["args"]
        == second["value"]["args"]
        == [dict(op="var", name=decl["name"], type="f64")]
    )
    assert [c["rule"] for c in changes] == ["share_exact_math_operand"]
    assert (
        fn["math_operand_analysis"]["result_policy"]
        == "preserve_each_call_at_original_control_point"
    )
    x = dict(op="var", name="x")
    step = dict(op="expr", value=dict(op="unary", operator="postinc", value=x))
    for body, domain in [
        ([original[0], step, original[1]], "u32"),
        (
            [
                dict(
                    op="if",
                    condition=dict(op="literal", type="bool", value=True),
                    then=[original[0]],
                    **{"else": [original[1]]},
                )
            ],
            "u32",
        ),
        (
            [
                call("cos", "c", dict(op="cast", mode="static", type="f64", value=x)),
                call("sin", "d", dict(op="cast", mode="static", type="f64", value=x)),
            ],
            "u64",
        ),
        (
            [
                call("cos", "c", dict(op="binary", operator="mul", left=x, right=x)),
                call("sin", "d", dict(op="binary", operator="mul", left=x, right=x)),
            ],
            "f64",
        ),
    ]:
        blocked, changes = rewrite(module(body, domain))
        assert not changes and blocked["functions"][0]["body"] == body
    escaped = module(deepcopy(original))
    escaped["functions"][0]["state_contract"]["escaped"] = ["x"]
    assert not rewrite(escaped)[1]
    loop = dict(
        op="while",
        site="loop",
        condition=dict(op="literal", type="bool", value=False),
        body=[*deepcopy(original), step],
    )
    fn = rewrite(module([loop]))[0]["functions"][0]
    assert fn["body"][0]["op"] == "while" and fn["body"][0]["body"][0]["op"] == "decl"
    facts = fn["math_operand_analysis"]["sites"]
    assert facts[0]["operands"][0]["loops"] == [
        dict(site="loop", syntactically_written_reads=["x"], other_private_reads=[])
    ]


def test_folded_extent_preserves_axis_order_for_coordinate_traversal():
    from scar.middle.execution import annotate_iteration_layouts
    from scar.middle.rewrites import optimize

    def lit(n):
        return dict(op="literal", type="i64", value=n)

    def binary(op, lhs, rhs):
        return dict(op="binary", operator=op, left=lhs, right=rhs)

    pos = dict(op="var", name="i")
    # Non-square first-axis-fastest layout, with a non-unit memory stride.
    offset = binary(
        "add",
        binary("mul", lit(3), binary("mod", pos, lit(2))),
        binary("mod", binary("div", pos, lit(2)), lit(5)),
    )
    fn = {
        "params": [],
        "body": [
            dict(
                op="map",
                index="i",
                extent=binary("mul", lit(2), lit(5)),
                independent=False,
                site="store",
                body=[
                    dict(
                        op="assign",
                        target=dict(
                            op="index",
                            base=dict(op="var", name="out"),
                            indices=[offset],
                        ),
                        value=lit(3),
                    )
                ],
            )
        ],
    }
    for a, b in ((0, 5), (-2, -5), (1 << 62, 4)):
        changed = deepcopy(fn)
        changed["body"][0]["extent"] = binary("mul", lit(a), lit(b))
        optimize(changed)
        assert "extent_factors" not in changed["body"][0]
    optimize(fn)
    loop = fn["body"][0]
    assert loop["extent"] == lit(10) and loop["extent_factors"] == [2, 5]
    source_body = deepcopy(loop["body"])
    assert annotate_iteration_layouts(fn) == 1
    traversal = loop["iteration_layout"]
    assert loop["body"] == source_body
    assert traversal["order"] == "first_axis_fastest" and traversal["axes"] == [
        lit(2),
        lit(5),
    ]
    assert [row["axis"] for row in traversal["coordinates"]] == [0, 1]
    loop["extent_factors"] = [1, 11]
    with pytest.raises(ValueError, match="extent factorization"):
        annotate_iteration_layouts(fn)


@pytest.mark.parametrize(
    ("command", "operation"),
    [("unbound-command", "process_request"), ("./child", "process_spawn")],
)
def test_process_return_checks_survive_unchanged_loop_paths_and_drop_on_throw(
    command, operation
):
    from scar.middle.normalization import processes, values
    from scar.middle.normalization.protocols import _Frame, _Project

    function = dict(
        name="owner",
        params=[],
        body=[],
        result="void",
        python_scope=dict(kind="module", module="owner", main=True),
    )
    project = _Project({"owner": function})
    project.execution_images = [dict(id="child", output="child")]
    project.active.append("owner")
    frame = _Frame(project, function, ())
    imports = project.imports
    initializing = dict(state="initializing", initializer="owner", site="load")
    imports.bind_registry("__main__", initializing)
    steps = []
    earlier = dict(
        op="module_import_preserves",
        checks={},
        occurrence="entry/import",
        direct_registry_keys=[],
        conditional_registry_keys=[],
    )
    project.scoped_registry_effect("owner", {}, earlier, steps)
    imports.bind_registry("__main__", dict(initializing, state="ready"))
    before = imports.snapshot()
    result = processes.invoke(
        frame,
        dict(site="entry/launch"),
        values.Value("host_callable", "subprocess.Popen"),
        [values.literal(command)],
        {},
        steps,
        False,
    )
    assert processes.is_process(result.kind)
    request = steps[1]["init"]
    assert request["op"] == operation and request["external_effect"] == "UNKNOWN"
    after = imports.snapshot()
    exceptional = imports.exceptional_prefix(before, after, may_write=True)
    assert not exceptional["import_windows"]
    assert not imports.join_snapshots([after, exceptional])["import_windows"]
    imports.restore(imports.join_snapshots([after, imports.copy_snapshot(after)]))
    assert project.registry_entry("__main__", {})["state"] == "unresolved"
    project.observe_import("owner", "namespace", {}, "provider")
    for guard, expected in ((steps[0], "initializing"), (steps[2], "ready")):
        check = guard["condition"]["checks"]["__main__"]
        assert check["before"]["state"] == expected
        assert check["namespace"]["binding_reads"][0]["member"] == "provider"
        assert check["namespace"]["value_state_projection"] == "unproven"
    fallback = steps[2]["else"][0]["value"]
    assert fallback["op"] == "state_effect_continuation_required"
    assert fallback["producer_capture"] == request["registry_state_capture"]["identity"]
    assert (
        fallback["resume"]
        == "continue_after_original_action_with_captured_result_and_actual_post_state"
    )


def test_m4_proves_only_function_local_scalar_indexed_scratch():
    from scar.middle.contracts import publish_program
    from scar.middle.execution import prepare

    def var(name):
        return dict(op="var", name=name)

    extent = dict(op="literal", type="i64", value=4)
    slot = dict(op="index", base=var("work"), indices=[var("i")])
    buffer = dict(op="buffer", name="work", type="f64", extent=extent)
    loop = dict(
        op="map",
        index="i",
        extent=extent,
        independent=True,
        body=[
            dict(
                op="assign",
                target=slot,
                value=dict(op="literal", type="f64", value=2.0),
            ),
            dict(op="assign", target=var("seen"), value=slot),
        ],
    )

    def proof(
        *extra, decl=buffer, globals_=(), params=(), other_functions=(), externs=()
    ):
        fn = dict(
            name="entry",
            params=list(params),
            result="void",
            body=[dict(op="decl", name="seen", type="f64"), decl, loop, *extra],
        )
        module = dict(
            entry="entry",
            functions=[fn, *other_functions],
            externs=list(externs),
            records=[],
            globals=list(globals_),
        )
        prepared, _ = prepare(publish_program(module, "M3", complete=True))
        return prepared["functions"][0]["body"][1].get("private_scratch")

    assert proof() == "indexed_scalar_access_only"
    forged = dict(buffer, private_scratch="indexed_scalar_access_only")
    for value in (
        var("work"),
        dict(op="unary", operator="addr", value=slot),
        dict(op="call", callee="retain", args=[slot], parameter_modes=["reference"]),
        dict(op="cast", type=dict(kind="ref", to="f64"), value=slot),
        dict(
            op="call", callee="retain", args=[dict(op="cast", type="f64", value=slot)]
        ),
    ):
        assert proof(dict(op="expr", value=value), decl=forged) is None
    assert (
        proof(dict(op="block", body=[dict(op="decl", name="work", type="i64")])) is None
    )
    assert proof(globals_=[dict(name="work", type="i64")]) is None
    assert proof(params=[dict(name="work", type="i64")]) is None
    assert proof(decl=dict(buffer, allocator="custom")) is None
    assert proof(decl=dict(buffer, storage="static")) is None
    assert (
        proof(
            decl=dict(
                op="decl",
                name="work",
                type=dict(kind="ptr", to="f64"),
                init=dict(op="allocate", type="f64", extent=extent),
            )
        )
        is None
    )
    assert (
        proof(other_functions=[dict(name="_Znam", params=[], result="void", body=[])])
        is None
    )
    assert proof(externs=[dict(name="mallinfo2", params=[], result="i64")]) is None
    catcher = dict(
        op="try", body=[], handlers=[dict(categories=["memory_error"], body=[])]
    )
    assert (
        proof(
            other_functions=[
                dict(name="caller", params=[], result="void", body=[catcher])
            ]
        )
        is None
    )


def test_open_module_literal_store_uses_active_code_and_source_mapping(tmp_path):
    from types import SimpleNamespace

    from scar.backend import compile_ir
    from scar.contract import IR, read_member
    from scar.frontend import build
    from scar.middle import plan

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    source = pkg / "__init__.py"
    source.write_text(
        "from . import child as disposable\ndel disposable\n"
        "message = 'same string !'\nalternate = 'same string !'\n"
        "last = 0 or 123\ndef declaration():\n    global alternate\n"
    )
    (pkg / "child.py").write_text("pass\n")
    entry = tmp_path / "main.py"
    entry.write_text("import pkg\n")

    # STORE_NAME can publish through a separate locals mapping and release
    # an old value that overwrites the newly published constant. STORE_GLOBAL
    # must still obtain the original object from the current code's co_consts.
    code = compile(source.read_bytes(), str(source), "exec")
    constant = next(v for v in code.co_consts if v == "same string !")
    globals_ = {"__builtins__": {"__import__": lambda *a: SimpleNamespace(child=7)}}
    locals_, observations = {}, []

    class Previous:
        def __del__(self):
            observations.append(locals_["message"] is constant)
            locals_["message"] = "changed during release"

    locals_["message"] = Previous()
    exec(code, globals_, locals_)
    assert observations == [True] and locals_["message"] == "changed during release"
    assert globals_["alternate"] is constant and "alternate" not in locals_

    class Reject(dict):
        def __setitem__(self, name, value):
            if name == "message":
                raise LookupError(name)
            super().__setitem__(name, value)

    del globals_["alternate"]
    with pytest.raises(LookupError, match="message"):
        exec(code, globals_, Reject())
    assert "alternate" not in globals_

    graph = tmp_path / "graph"
    build(entry, graph)
    source.unlink()
    index = json.loads((graph / "graph.json").read_bytes())
    unit = next(row for row in index["units"] if row["path"] == "pkg/__init__.py")
    payload = read_member(graph, unit["graph"])
    initializer = next(
        f for f in payload["functions"] if f["signature"]["name"] == "py:pkg:<init>"
    )
    nodes = initializer["nodes"]
    [first, second, _] = [n for n in nodes if n["op"] == "assign"]
    ports = [deepcopy(n["source_constant_store"]) for n in (first, second)]
    assert ports[0]["load"]["index"] == ports[1]["load"]["index"]

    def prefix(path, expected_line):
        folder = tmp_path / ("plan-" + path.stem)
        result = plan(path, folder)
        assert result["status"] == "INCOMPLETE"
        report = json.loads((folder / "plan.json").read_bytes())
        [issue] = report["issues"]
        assert (issue["code"], issue["location"]["line"]) == (
            "python.global_store_after_callback",
            expected_line,
        )
        row = next(
            r
            for r in report["protocol_analysis"]["prefix_fragments"]
            if r["symbol"] == "py:pkg:<init>"
        )
        return [s["value"] for s in read_member(folder, row["fragment"])["body"]]

    actual = prefix(graph / "graph.json", 5)
    assert [n["op"] for n in actual] == [
        "module_fromlist_publish",
        "module_global_delete",
        "module_global_store",
        "module_global_store",
    ]
    stores = actual[-2:]
    assert [s["store_opcode"] for s in stores] == ["STORE_NAME", "STORE_GLOBAL"]
    assert [s["namespace"] for s in stores] == ["frame.locals", "frame.globals"]
    assert [s["source_constant_store"] for s in stores] == ports
    for store in stores:
        assert store["frame_globals"]["owner"] == "py:pkg:<init>"
        assert store["namespace"] in store["writes"] and "frame.code" in store["reads"]
        assert store["external_effect"] == "UNKNOWN" and store["may_raise"]
        assert "same_object" in store["constant"] and "reentry" in store["release"]
        assert "source_equivalent" in store["on_code_mismatch"]

    # Replay a prior schema and a mismatched code receipt at the same source site.
    for label in ("old", "foreign"):
        if label == "old":
            del first["source_constant_store"]
        else:
            first["source_constant_store"] = deepcopy(ports[0])
            first["source_constant_store"]["code"]["code_sha256"] = "0" * 64
        unit["graph"] = member(graph, "units", payload)
        path = graph / (label + ".json")
        write_json(path, index)
        if label == "foreign":
            with pytest.raises(
                ValueError, match="source constant store disagrees with its module"
            ):
                plan(path, tmp_path / "plan-foreign")
        else:
            assert [n["op"] for n in prefix(path, 3)] == [
                "module_fromlist_publish",
                "module_global_delete",
            ]

    # A READY replay through B1 still needs a provider if family metadata is absent.
    untagged = dict(stores[0])
    del untagged["operator_family"]
    fn = dict(
        name="main", result="void", params=[], body=[dict(op="expr", value=untagged)]
    )
    sealed = tmp_path / "sealed"
    write_json(
        sealed / "exec-ir.json",
        document(
            IR,
            status="READY",
            module={"entry": "main"},
            functions=[member(sealed, "functions", fn)],
        ),
    )
    result = compile_ir(sealed, tmp_path / "physical", policy="serial")
    [failure] = result["issues"]
    assert (result["status"], failure["code"], failure["requirement"]) == (
        "INCOMPLETE",
        "missing_operator_binding",
        "module_registry",
    )
    assert failure["registry_checks"]["operator_counts"] == {"module_global_store": 1}


def test_open_object_expression_connects_live_callees_arguments_comparisons_and_truth(
    tmp_path,
):
    import operator
    from types import SimpleNamespace

    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle import plan

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    source = pkg / "__init__.py"
    source.write_text(
        "from . import child as gone\ndel gone\n"
        "if outer(inner(arg)) < right(bound) or outer(arg) >= right(limit):\n    pass\n"
        "boundary = 0 or 1\ndef declaration():\n    global limit\n"
    )
    (pkg / "child.py").write_text("pass\n")
    entry = tmp_path / "entry.py"
    entry.write_text("import pkg\n")
    code = compile(source.read_bytes(), str(source), "exec")
    graph, out = tmp_path / "graph", tmp_path / "out"
    build(entry, graph)
    source.unlink()

    def lower(index, out):
        assert plan(index, out)["status"] == "INCOMPLETE"
        report = json.loads((out / "plan.json").read_bytes())
        [issue] = report["issues"]
        row = next(
            r
            for r in report["protocol_analysis"]["prefix_fragments"]
            if r["symbol"] == "py:pkg:<init>"
        )
        return issue, read_member(out, row["fragment"])

    issue, fragment = lower(graph, out)
    assert (issue["code"], issue["location"]["line"]) == (
        "python.global_store_after_callback",
        5,
    )
    evaluation = fragment["body"][2]["init"]
    assert (evaluation["op"], evaluation["type"]) == (
        "module_object_expression",
        "bool",
    )
    assert evaluation["external_effect"] == "UNKNOWN" and evaluation["may_raise"]
    assert "source_equivalent" in evaluation["on_code_mismatch"]
    expr = evaluation["expression"]
    assert expr["op"] == "select" and expr["then"] == {
        "op": "literal",
        "type": "bool",
        "value": True,
    }
    left = expr["condition"]["value"]
    right = expr["else"]["value"]
    assert left["operator"] == "lt" and right["operator"] == "gte"
    assert right["right"]["args"][0]["lookup_order"] == [
        "frame.globals",
        "frame.builtins",
    ]
    assert left["left"]["args"][0]["target"]["lookup_order"] == [
        "frame.locals",
        "frame.globals",
        "frame.builtins",
    ]

    def execute(mode, interpreted):
        trace = []
        local = {"arg": "argument", "limit": "wrong locals limit"}
        builtins_ = {
            "__import__": lambda *a: SimpleNamespace(child=0),
            "bound": "builtin bound",
        }
        global_ = {"__builtins__": builtins_, "limit": "global limit"}
        scopes = {
            "frame.locals": local,
            "frame.globals": global_,
            "frame.builtins": builtins_,
        }

        class Result:
            def __lt__(self, rhs):
                trace.append("compare lt")
                return Decision()

            def __ge__(self, rhs):
                trace.append("compare ge")
                return Decision()

        class Decision:
            def __bool__(self):
                trace.append("truth")
                return mode == "stop"

        class Callable:
            def __init__(self, label):
                self.label = label

            def __call__(self, arg):
                trace.append(self.label + "(" + arg + ")")
                return Result()

            def __del__(self):
                if self.label == "old":
                    trace.append("release old")

        def inner(arg):
            trace.append("inner(" + arg + ")")
            local["outer"] = Callable("new")
            if mode == "raise":
                raise LookupError("inner")
            return "inner result"

        local.update(outer=Callable("old"), inner=inner)
        global_["right"] = Callable("right")

        def value(n):
            op = n["op"]
            if op == "module_name_load":
                name = n["instruction"]["name"]
                for domain in n["lookup_order"]:
                    if name in scopes[domain]:
                        return scopes[domain][name]
                raise NameError(name)
            if op == "source_object_call":
                return value(n["target"])(*[value(arg) for arg in n["args"]])
            if op == "source_object_compare":
                return {"lt": operator.lt, "gte": operator.ge}[n["operator"]](
                    value(n["left"]), value(n["right"])
                )
            if op == "source_object_truth":
                return bool(value(n["value"]))
            if op == "select":
                return value(n["then"] if value(n["condition"]) else n["else"])
            assert op == "literal"
            return n["value"]

        try:
            if interpreted:
                value(expr)
            else:
                exec(code, global_, local)
        except LookupError as error:
            trace.append("raised " + str(error))
        return trace

    prefix = [
        "inner(argument)",
        "old(inner result)",
        "release old",
        "right(builtin bound)",
        "compare lt",
        "truth",
    ]
    for mode, expected in (
        ("stop", prefix),
        (
            "continue",
            prefix + ["new(argument)", "right(global limit)", "compare ge", "truth"],
        ),
        ("raise", ["inner(argument)", "release old", "raised inner"]),
    ):
        assert execute(mode, False) == expected
        assert execute(mode, True) == expected

    index = json.loads((graph / "graph.json").read_bytes())
    unit = next(row for row in index["units"] if row["path"] == "pkg/__init__.py")
    original = read_member(graph, unit["graph"])
    for variant in ("foreign_code", "missing_target", "spread", "keywords"):
        payload = deepcopy(original)
        mod = next(
            fn
            for fn in payload["functions"]
            if fn["signature"]["name"] == "py:pkg:<init>"
        )
        nodes = mod["nodes"]
        call = next(
            n
            for n in nodes
            if n.get("op") in {"call", "python_invoke"} and n["location"]["line"] == 3
        )
        operand = nodes[call["target"]["port"]]
        if variant == "foreign_code":
            operand["source_name_load"]["code_sha256"] = "0" * 64
        elif variant == "missing_target":
            del call["target"]
        elif variant == "spread":
            call["args"] = [{"port": len(nodes)}]
            nodes.append(
                dict(
                    op="starred",
                    value={"port": nodes.index(operand)},
                    location=operand["location"],
                )
            )
        else:
            call["keywords"] = [{"name": None, "value": call["args"][0]}]
        unit["graph"] = member(graph, "units", payload)
        path = graph / (variant + ".json")
        write_json(path, index)
        issue, fragment = lower(path, tmp_path / ("plan-" + variant))
        assert issue["location"]["line"] == 3
        if variant == "missing_target":
            assert issue["code"] == "python.call_target"
        assert all(
            s.get("init", {}).get("op") != "module_object_expression"
            for s in fragment["body"]
        )


def test_conditional_plain_import_consumes_actual_result_at_its_source_store(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from scar.contract import read_member, walk_operations
    from scar.frontend import build
    from scar.middle import plan

    package = tmp_path / "package"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text(
        "from . import child as gone\ndel gone\n"
        "try:\n    if enabled:\n        import offered\n        seen(offered)\n"
        "except:\n    audit(offered)\n    raise\nfinally:\n    pass\n"
        "audit(offered)\nimport cleanup\n"
        "done(cleanup)\nboundary = 0 or 1\ndef declaration():\n    global cleanup\n"
    )
    for module in (
        package / "child.py",
        tmp_path / "offered.py",
        tmp_path / "cleanup.py",
    ):
        module.write_text("pass\n")
    entry = tmp_path / "main.py"
    entry.write_text("import package\n")
    code = compile(source.read_bytes(), str(source), "exec")
    graph, out = tmp_path / "graph", tmp_path / "out"
    build(entry, graph)
    source.unlink()

    assert plan(graph, out)["status"] == "INCOMPLETE"
    report = json.loads((out / "plan.json").read_bytes())
    [issue] = report["issues"]
    assert (issue["code"], issue["location"]["line"]) == (
        "python.global_store_after_callback",
        15,
    )
    fragment = next(
        r
        for r in report["protocol_analysis"]["prefix_fragments"]
        if r["symbol"] == "py:package:<init>"
    )
    body = read_member(out, fragment["fragment"])["body"]
    protected = next(s for s in body if s["op"] == "try")
    branch = next(s for s in protected["body"] if s["op"] == "if")
    assert not branch["else"]
    imports = [s for s in walk_operations(body) if s["op"] == "module_import_publish"]
    offered, cleanup = sorted(imports, key=lambda n: n["location"]["line"])
    assert branch["then"][0]["value"] == offered and body[-2]["value"] == cleanup
    assert len({s["source_import_occurrence"] for s in imports}) == 2
    occurrences = {
        r["id"]: r for r in read_member(out, report["source_imports"])["occurrences"]
    }
    for operation, mapping in zip(
        (offered, cleanup), ("frame.locals", "frame.globals")
    ):
        [producer] = occurrences[operation["occurrence"]]["producer_sites"]
        assert (producer["site"], producer["role"], producer["opcode"]) == (
            operation["site"],
            "source_import",
            "module_import_publish",
        )
        binding = operation["binding"]
        assert binding["namespace"] == mapping and mapping in operation["writes"]
        assert binding["value"] == {"result": operation["import_call"]["result"]}
        assert operation["occurrence"] == operation["source_import_occurrence"]
        assert operation["external_effect"] == "UNKNOWN" and operation["may_raise"]
        assert "source_equivalent" in operation["on_code_mismatch"]

    def argument(statement):
        return statement["value"]["expression"]["value"]["args"][0]

    assert (
        argument(branch["then"][1])["source_evidence"]["prior_publication"][
            "occurrence"
        ]
        == offered["occurrence"]
    )
    assert "source_evidence" not in argument(protected["handlers"][0]["body"][0])
    assert "source_evidence" not in argument(body[-3])
    assert (
        argument(body[-1])["source_evidence"]["prior_publication"]["occurrence"]
        == cleanup["occurrence"]
    )

    def execute(mode, lowered):
        trace = []
        result, changed = object(), object()

        class Locals(dict):
            def __setitem__(self, name, value):
                if name == "offered" and value is result and mode == "store_error":
                    raise LookupError("store")
                super().__setitem__(name, value)

        local = Locals(enabled=mode != "skip", offered="previous")
        global_ = {}

        def observe(name, value):
            trace.append((name, "returned" if value is result else value))

        def importer(name, globals_, locals_, fromlist, level):
            assert globals_ is global_ and locals_ is local
            if level:
                return SimpleNamespace(child=None)
            trace.append(("import", name))
            assert fromlist is None and type(level) is int and level == 0
            monkeypatch.setitem(sys.modules, name, changed)
            if name == "offered" and mode == "import_error":
                raise LookupError("import")
            return result

        global_["__builtins__"] = {"__import__": importer}
        local.update(
            {
                name: lambda v, n=name: observe(n, v)
                for name in ("seen", "audit", "done")
            }
        )
        if mode == "reentry":

            class Previous:
                def __del__(self):
                    assert local["offered"] is result
                    trace.append(("release", "previous"))
                    local["offered"] = "replaced"

            local["offered"] = Previous()

        def publish(n):
            call, store = n["import_call"], n["binding"]
            frame = {
                "locals": local,
                "globals": global_,
                "builtins": global_["__builtins__"],
            }

            def operand(spec):
                if spec.get("opcode") == "LOAD_SMALL_INT":
                    return spec["oparg"]
                if "table" in spec:
                    return getattr(code, spec["table"])[spec["index"]]
                return frame[spec["frame"]]

            dispatch = call["importer"]
            callback = frame[dispatch["mapping"].split(".")[1]][dispatch["key"]]
            outputs = {
                call["result"]: callback(*(operand(a) for a in call["arguments"]))
            }
            frame[store["namespace"].split(".")[1]][operand(store["key"])] = outputs[
                store["value"]["result"]
            ]

        try:
            if lowered:
                try:
                    if local["enabled"]:
                        publish(offered)
                        local["seen"](local["offered"])
                except BaseException:
                    local["audit"](local["offered"])
                    raise
                local["audit"](local["offered"])
                publish(cleanup)
                local["done"](global_["cleanup"])
            else:
                exec(code, global_, local)
        except LookupError as error:
            trace.append(("raised", str(error)))
        return trace

    for mode in ("skip", "success", "import_error", "store_error", "reentry"):
        actual = execute(mode, False)
        assert execute(mode, True) == actual
        assert actual.count(("import", "offered")) == int(mode != "skip")
        assert actual.count(("import", "cleanup")) == int(not mode.endswith("_error"))
        if mode.endswith("_error"):
            assert actual[-1] == ("raised", mode.split("_")[0])
        if mode == "reentry":
            assert ("seen", "replaced") in actual

    index = json.loads((graph / "graph.json").read_bytes())
    unit = next(r for r in index["units"] if r["path"] == "package/__init__.py")
    graph_unit = read_member(graph, unit["graph"])
    decl = next(
        n
        for fn in graph_unit["functions"]
        for n in fn["nodes"]
        if n["op"] == "python_import" and n["module"] == "offered"
    )
    del decl["source_import_store"]
    unit["graph"] = member(graph, "units", graph_unit)
    old = graph / "old.json"
    write_json(old, index)
    prior = tmp_path / "prior"
    assert plan(old, prior)["status"] == "INCOMPLETE"
    [failure] = json.loads((prior / "plan.json").read_bytes())["issues"]
    assert (failure["code"], failure["location"]["line"]) == (
        "python.import_publication_evidence",
        5,
    )


def test_open_source_call_preserves_descriptor_format_and_keyword_order(tmp_path):
    from types import SimpleNamespace

    from scar.contract import read_member
    from scar.frontend import build
    from scar.frontend.languages.python.environment import code_digest
    from scar.middle import plan

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    source = pkg / "__init__.py"
    source.write_text(
        "from . import child as gone\ndel gone\n"
        "receiver.dispatch(f'<{value!r:{width}}/{other!a}/{value!s}>', "
        "UserWarning, tag=keyword(), literal='distinct token!')\n"
        "receiver.method(receiver.member)\nboundary = 0 or 1\n"
    )
    (pkg / "child.py").write_text("pass\n")
    entry = tmp_path / "main.py"
    entry.write_text("import pkg\n")
    code = compile(source.read_bytes(), str(source), "exec")
    graph, out = tmp_path / "graph", tmp_path / "out"
    build(entry, graph)
    source.unlink()

    def plan_case(path, folder):
        assert plan(path, folder)["status"] == "INCOMPLETE"
        report = json.loads((folder / "plan.json").read_bytes())
        [failure] = report["issues"]
        ref = next(
            r
            for r in report["protocol_analysis"]["prefix_fragments"]
            if r["symbol"] == "py:pkg:<init>"
        )
        return failure, read_member(folder, ref["fragment"])["body"]

    issue, body = plan_case(graph, out)
    assert (issue["location"]["line"], issue["code"]) == (
        5,
        "python.global_store_after_callback",
    )
    regions = [s["value"] for s in body[2:]]
    for region in regions:
        assert region["expression"]["op"] == "source_object_region"
        assert "source_equivalent" in region["on_code_mismatch"]
        assert region["external_effect"] == "UNKNOWN" and region["may_raise"]
        witness = region["source_expression"]
        assert witness["code_sha256"] == region["source_code"]["code_sha256"]
        assert [s["offset"] for s in region["expression"]["steps"]] == [
            i["offset"] for i in witness["instructions"] if i["opcode"] != "PUSH_NULL"
        ]

    def run(mode, execute_region):
        events = []
        category = object()
        active = code
        token_text = (
            "different source token" if mode == "different_code" else "distinct token!"
        )
        if mode == "different_code":
            active = code.replace(
                co_consts=tuple(
                    token_text if v == "distinct token!" else v for v in code.co_consts
                )
            )
        matches = code_digest(active) == regions[0]["source_code"]["code_sha256"]
        assert matches == (mode != "different_code")
        token = next(v for v in active.co_consts if v == token_text)

        class Field:
            def __repr__(self):
                events.append("repr")
                return "é"

            def __str__(self):
                events.append("str")
                return "text"

        class Width:
            def __format__(self, spec):
                events.append("spec")
                if mode == "format_error":
                    raise LookupError("spec")
                return "3"

        class Receiver:
            @property
            def dispatch(self):
                events.append("descriptor")
                if mode == "lookup_error":
                    raise LookupError("descriptor")
                target = self

                def call(text, kind, **kwargs):
                    assert target is receiver and kind is category
                    assert (
                        list(kwargs) == ["tag", "literal"]
                        and kwargs["literal"] is token
                    )
                    events.append(("called", text, kwargs["tag"]))

                return call

            @property
            def member(self):
                events.append("member")
                return self

            def method(self, item):
                assert self is item is receiver
                events.append("method")

        receiver = Receiver()

        def keyword():
            events.append("keyword")
            if mode == "keyword_error":
                raise LookupError("keyword")
            return 9

        local = {
            "receiver": receiver,
            "value": Field(),
            "width": Width(),
            "other": Field(),
            "keyword": keyword,
        }
        builtins_ = {
            "__import__": lambda *a: SimpleNamespace(child=0),
            "UserWarning": category,
        }
        global_ = {"__builtins__": builtins_}
        scopes = {
            "frame.locals": local,
            "frame.globals": global_,
            "frame.builtins": builtins_,
        }
        converters = (lambda v: v, str, repr, ascii)

        def interpret(region):
            stack = {}

            def consume(ref):
                return stack.pop(ref["from_offset"])

            for s in region["expression"]["steps"]:
                op = s["op"]
                if op == "module_name_load":
                    name = s["instruction"]["name"]
                    value = next(
                        scopes[k][name] for k in s["lookup_order"] if name in scopes[k]
                    )
                elif op == "source_object_constant":
                    value = (
                        code.co_consts[s["operand"]]
                        if s["source"] == "current_frame_co_consts"
                        else s["operand"]
                    )
                elif op == "source_object_attribute":
                    value = getattr(
                        consume(s["base"]), active.co_names[s["name_index"]]
                    )
                elif op == "source_object_convert":
                    value = converters[s["conversion"]](consume(s["value"]))
                elif op == "source_object_format":
                    spec = consume(s["spec"]) if s["spec"] is not None else ""
                    value = format(
                        converters[s["conversion"]](consume(s["value"])), spec
                    )
                elif op == "source_object_build_string":
                    value = "".join(consume(item) for item in s["items"])
                elif op == "source_object_call":
                    names = consume(s["keyword_names"]) if s["keyword_names"] else ()
                    value = consume(s["target"])(
                        *[consume(a) for a in s["args"]],
                        **{
                            name: consume(v)
                            for name, v in zip(names, s["keyword_values"], strict=True)
                        },
                    )
                else:
                    assert op == "source_object_discard"
                    consume(s["value"])
                    continue
                stack[s["offset"]] = value
                del value
            assert not stack

        try:
            if execute_region and matches:
                for region in regions:
                    interpret(region)
            else:
                exec(active, global_, local)
        except LookupError as e:
            events.append(("raised", str(e)))
        return events

    for mode in (
        "success",
        "different_code",
        "lookup_error",
        "format_error",
        "keyword_error",
    ):
        expected = run(mode, False)
        assert run(mode, True) == expected
        if mode in {"success", "different_code"}:
            assert expected[-2:] == ["member", "method"]
            assert (
                expected.index("spec") < expected.index("repr")
                if sys.version_info < (3, 13)
                else expected.index("repr") < expected.index("spec")
            )
        else:
            assert expected[-1][0] == "raised" and "method" not in expected

    index = json.loads((graph / "graph.json").read_bytes())
    unit = next(u for u in index["units"] if u["path"] == "pkg/__init__.py")
    data = read_member(graph, unit["graph"])
    fn = next(f for f in data["functions"] if f["signature"]["name"] == "py:pkg:<init>")
    call = next(
        n
        for n in fn["nodes"]
        if n.get("source_expression") and n["location"]["line"] == 3
    )
    call["keywords"][0]["name"] = "other"
    unit["graph"] = member(graph, "units", data)
    where = graph / "mismatch.json"
    write_json(where, index)
    miss, prefix = plan_case(where, tmp_path / "mismatch-plan")
    assert (
        miss["code"] == "python.source_expression_operands"
        and miss["location"]["line"] == 3
    )
    assert len(prefix) == 2


def test_open_named_exception_arms_dispatch_and_release_the_actual_exception(tmp_path):
    from types import SimpleNamespace

    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle import plan

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    source = pkg / "__init__.py"
    source.write_text(
        "from . import child as unused\ndel unused\n"
        "try:\n    operation()\nexcept First:\n    first()\n"
        "except Second as caught:\n    selected(caught)\n"
        "try:\n    other()\nexcept Second as pending:\n    boundary = 0 or 37\n"
        "tail = 1\n"
    )
    (pkg / "child.py").write_text("pass\n")
    entry = tmp_path / "main.py"
    entry.write_text("import pkg\n")
    code = compile(source.read_bytes(), str(source), "exec")
    graph, out = tmp_path / "graph", tmp_path / "out"
    build(entry, graph)
    source.unlink()
    assert plan(graph, out)["status"] == "INCOMPLETE"
    report = json.loads((out / "plan.json").read_bytes())
    [issue] = report["issues"]
    assert (issue["code"], issue["location"]["line"]) == (
        "python.global_store_after_callback",
        12,
    )
    row = next(
        r
        for r in report["protocol_analysis"]["prefix_fragments"]
        if r["symbol"] == "py:pkg:<init>"
    )
    fragment = read_member(out, row["fragment"])
    [closed] = [s for s in fragment["body"] if s["op"] == "try"]
    arms = closed["handlers"]
    assert not arms[0]["on_match"] and not arms[0]["cleanup"]
    for index, arm in enumerate(arms):
        match = arm["match"]
        assert match["op"] == "source_exception_match" and match["may_raise"]
        assert (
            match["external_effect"] == "UNKNOWN"
            and match["matcher"]["op"] == "module_name_load"
        )
        assert match["source_exception_handler"]["arm_index"] == index
        assert "source_exception_dispatch" in match["on_code_mismatch"]
    bind = arms[1]["on_match"][0]["value"]
    clear = arms[1]["cleanup"][0]["value"]
    assert (
        bind["op"] == "source_exception_bind"
        and bind["value"] == arms[1]["match"]["exception"]
    )
    assert (
        clear["op"] == "source_exception_clear"
        and clear["exceptional_cleanup"][-1]["opcode"] == "RERAISE"
    )
    assert [i["opcode"] for i in clear["normal_exit"]][-3:] == [
        "LOAD_CONST",
        "STORE_NAME",
        "DELETE_NAME",
    ]

    frontier = fragment["frontier"]
    assert frontier["operator"] == "try" and not frontier.get("evaluation")
    partial = next(
        r for r in frontier["regions"] if r["path"] == ["handlers", 0, "body"]
    )
    assert (
        partial["completion"] == "prefix"
        and partial["frontier"]["operator"] == "assign"
    )
    assert (
        partial["handler"]["match"]["source_exception_handler"]["load"]["name"]
        == "Second"
    )
    assert partial["handler"]["on_match"][0]["value"]["name"] == "pending"
    assert partial["handler"]["cleanup"][0]["value"]["name"] == "pending"

    def execute(mode, lowered):
        log = []
        original = ValueError("original")

        class Local(dict):
            def __getitem__(self, key):
                if key in ("First", "Second"):
                    log.append(("lookup", key, sys.exception() is original))
                    if key == "First":
                        if mode == "lookup_error":
                            raise RuntimeError("lookup")
                        # A failed match can replace the next handler's target.
                        self["Second"] = (
                            7 if mode == "invalid" else (ValueError, KeyError)
                        )
                return super().__getitem__(key)

            def __setitem__(self, key, value):
                if key == "caught":
                    log.append(("store", value is original, value is None))
                    if (
                        mode == "bind_error"
                        and value is original
                        or mode == "cleanup_error"
                        and value is None
                    ):
                        raise RuntimeError("store")
                super().__setitem__(key, value)
                if key == "caught" and value is original and mode == "reentrant_store":
                    super().__setitem__(key, "replacement")

            def __delitem__(self, key):
                if key == "caught":
                    log.append(("delete", self[key] is None))
                super().__delitem__(key)

        loc = Local(First=ValueError if mode == "first" else KeyError, Second=object())
        global_ = {
            "__builtins__": {"__import__": lambda *args: SimpleNamespace(child=0)}
        }

        def operation():
            log.append("protected")
            if mode != "normal":
                if mode == "miss":
                    raise ArithmeticError("miss")
                raise original

        def selected(e):
            log.append(("selected", e is original, sys.exception() is original))
            if mode == "body_error":
                raise RuntimeError("body")

        loc.update(
            operation=operation,
            first=lambda: log.append("first"),
            selected=selected,
            other=lambda: None,
        )

        def lookup(arm):
            load = arm["match"]["matcher"]
            assert load["lookup_order"] == [
                "frame.locals",
                "frame.globals",
                "frame.builtins",
            ]
            return loc[load["instruction"]["name"]]

        def cleanup(arm, exit_kind):
            step = arm["cleanup"][0]["value"]
            mapping = loc if step["namespace"] == "frame.locals" else global_
            for instruction in step[exit_kind]:
                if instruction["opcode"] == "STORE_NAME":
                    mapping[instruction["name"]] = None
                elif instruction["opcode"] == "DELETE_NAME":
                    del mapping[instruction["name"]]

        try:
            if lowered:
                finished = None
                try:
                    operation()
                except lookup(arms[0]):
                    loc["first"]()
                except lookup(arms[1]):
                    alias = arms[1]["on_match"][0]["value"]
                    loc[alias["name"]] = sys.exception()
                    try:
                        loc["selected"](loc[alias["name"]])
                    except BaseException:
                        cleanup(arms[1], "exceptional_cleanup")
                        raise
                    finished = arms[1]
                if finished is not None:
                    cleanup(finished, "normal_exit")
            else:
                exec(code, global_, loc)
        except BaseException as raised:
            log.append(
                (
                    "raised",
                    type(raised),
                    raised is original,
                    raised.__context__ is original,
                )
            )
        return log, "caught" in loc

    for mode in (
        "normal",
        "first",
        "second",
        "miss",
        "invalid",
        "lookup_error",
        "bind_error",
        "body_error",
        "cleanup_error",
        "reentrant_store",
    ):
        seen = execute(mode, False)
        assert execute(mode, True) == seen
        assert seen[0].count("protected") == 1
        if mode == "normal":
            assert seen == (["protected"], False)
        if mode == "invalid":
            assert seen[0][-1] == ("raised", TypeError, False, True)

    index = json.loads((graph / "graph.json").read_bytes())
    unit = next(u for u in index["units"] if u["path"] == "pkg/__init__.py")
    definition = read_member(graph, unit["graph"])
    arm = next(
        n
        for fn in definition["functions"]
        for n in fn["nodes"]
        if n["op"] == "exception_handler" and n["location"]["line"] == 5
    )
    del arm["source_exception_handler"]
    unit["graph"] = member(graph, "units", definition)
    old = graph / "old.json"
    write_json(old, index)
    prior = tmp_path / "old-plan"
    assert plan(old, prior)["status"] == "INCOMPLETE"
    [gap] = json.loads((prior / "plan.json").read_bytes())["issues"]
    assert gap["location"]["line"] == 5 and gap["code"] == "python.name_binding"


def test_open_module_raise_consumes_folded_constant_and_actual_cause(tmp_path):
    from types import SimpleNamespace

    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle import plan

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    source = pkg / "__init__.py"
    source.write_text(
        "from . import child as unused\ndel unused\n"
        "try:\n    run()\nexcept Match as caught:\n"
        "    message = 'owned ' + 'constant!'\n"
        "    raise make(message) from cause(caught)\n"
        "try:\n    run()\nexcept Match:\n    raise replacement\n"
        "try:\n    run()\nexcept Match:\n    raise\n"
        "tail = 0 or 37\n"
    )
    (pkg / "child.py").write_text("pass\n")
    entry = tmp_path / "main.py"
    entry.write_text("import pkg\n")
    code = compile(source.read_bytes(), str(source), "exec")
    graph, out = tmp_path / "graph", tmp_path / "plan"
    build(entry, graph)
    source.unlink()
    assert plan(graph, out)["status"] == "INCOMPLETE"
    report = json.loads((out / "plan.json").read_bytes())
    [issue] = report["issues"]
    assert (issue["code"], issue["location"]["line"]) == (
        "python.global_store_after_callback",
        16,
    )
    row = next(
        r
        for r in report["protocol_analysis"]["prefix_fragments"]
        if r["symbol"] == "py:pkg:<init>"
    )
    blocks = [n for n in read_member(out, row["fragment"])["body"] if n["op"] == "try"]
    assert len(blocks) == 3
    arms = [block["handlers"][0] for block in blocks]
    store = arms[0]["body"][0]["value"]
    port = store["source_constant_store"]
    assert store["op"] == "module_global_store" and port["rhs_sha256"]
    constant = code.co_consts[port["load"]["index"]]
    assert constant == "owned constant!"
    for arity, arm in zip((2, 1, 0), arms):
        action = arm["body"][-1]["value"]
        assert action["op"] == "source_exception_raise" and action["noreturn"]
        assert action["external_effect"] == "UNKNOWN" and action["may_raise"]
        assert (
            action["instruction"]
            == arm["match"]["source_exception_handler"]["body_raise"]
        )
        assert action["instruction"]["oparg"] == arity
        assert (action["cause"] is not None) == (arity == 2)
        assert (action["exception"] is not None) == (arity != 0)
        assert "before_first" in action["on_code_mismatch"]

    def run(mode, use_ir):
        seen = []
        original = LookupError("original")
        new = ValueError("new")
        replacement = RuntimeError("replacement")
        marker = object()
        loc = dict(Match=LookupError)

        def protect():
            seen.append("run")
            if seen.count("run") == {"single": 2, "bare": 3, "normal": 0}.get(mode, 1):
                raise original

        def make(value):
            seen.append(("make", value is constant, sys.exception() is original))
            if mode == "factory_failure":
                raise new
            loc["caught"] = marker
            return {"invalid_exception": 7, "exception_class": ValueError}.get(
                mode, new
            )

        def cause(value):
            seen.append(("cause", value is marker))
            if mode == "cause_failure":
                raise replacement
            return {"invalid_cause": 7, "none": None, "cause_class": RuntimeError}.get(
                mode, replacement
            )

        loc.update(run=protect, make=make, cause=cause, replacement=replacement)
        globals_ = {
            "__builtins__": {"__import__": lambda *args: SimpleNamespace(child=0)}
        }

        def dispatch(transfer):
            stack = {}

            def operand(ref):
                return stack[ref["from_offset"]]

            for instruction in transfer["evaluation"]:
                if instruction["op"] == "module_name_load":
                    result = loc[instruction["instruction"]["name"]]
                else:
                    assert instruction["op"] == "source_object_call"
                    result = operand(instruction["target"])(
                        *[operand(a) for a in instruction["args"]]
                    )
                stack[instruction["offset"]] = result
            if transfer["exception"] is None:
                raise
            if transfer["cause"] is None:
                raise operand(transfer["exception"])
            raise operand(transfer["exception"]) from operand(transfer["cause"])

        try:
            if use_ir:
                for arm in arms:
                    try:
                        protect()
                    except loc[arm["match"]["matcher"]["instruction"]["name"]] as error:
                        transfer = arm["body"][-1]["value"]
                        if not arm["on_match"]:
                            dispatch(transfer)
                        alias = arm["on_match"][0]["value"]["name"]
                        loc[alias] = error
                        try:
                            loc[port["store"]["name"]] = constant
                            dispatch(transfer)
                        finally:
                            loc[alias] = None
                            del loc[alias]
            else:
                exec(code, globals_, loc)
        except BaseException as raised:
            seen.append(
                (
                    type(raised),
                    str(raised),
                    raised is new,
                    type(raised.__cause__),
                    raised.__cause__ is replacement,
                    raised.__context__ is original,
                    raised.__suppress_context__,
                )
            )
        return seen, "caught" in loc

    for mode in (
        "ordinary",
        "exception_class",
        "invalid_exception",
        "invalid_cause",
        "none",
        "cause_class",
        "factory_failure",
        "cause_failure",
    ):
        actual = run(mode, False)
        assert run(mode, True) == actual
        assert actual[0][1] == ("make", True, True)
        assert (("cause", True) in actual[0]) == (mode != "factory_failure")
        assert not actual[1]

    for mode, error in (("single", RuntimeError), ("bare", LookupError)):
        actual = run(mode, False)
        assert run(mode, True) == actual and actual[0][-1][0] is error
    assert run("normal", True) == run("normal", False) == (["run"] * 3, False)

    index = json.loads((graph / "graph.json").read_bytes())
    unit = next(u for u in index["units"] if u["path"] == "pkg/__init__.py")
    definition = read_member(graph, unit["graph"])
    raised = next(
        n
        for f in definition["functions"]
        for n in f["nodes"]
        if n["op"] == "raise" and n["has_cause"]
    )
    del raised["source_expression"]
    unit["graph"] = member(graph, "units", definition)
    old = graph / "old.json"
    write_json(old, index)
    prior = tmp_path / "old-plan"
    assert plan(old, prior)["status"] == "INCOMPLETE"
    [gap] = json.loads((prior / "plan.json").read_bytes())["issues"]
    assert (gap["code"], gap["location"]["line"]) == ("python.source_raise_evidence", 7)


def test_open_module_store_and_delete_use_actual_operands_and_mappings(tmp_path):
    from types import SimpleNamespace

    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle import plan

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    source = pkg / "__init__.py"
    source.write_text(
        "from . import child as gone\ndel gone\n"
        "first = receiver.build(arg)\nother = first\n"
        "del first, other, first\n"
        "boundary = 0 or 1\ndef declare():\n    global other\n"
    )
    (pkg / "child.py").write_text("pass\n")
    entry = tmp_path / "main.py"
    entry.write_text("import pkg\n")
    code = compile(source.read_bytes(), str(source), "exec")
    graph, out = tmp_path / "graph", tmp_path / "plan"
    build(entry, graph)
    source.unlink()
    assert plan(graph, out)["status"] == "INCOMPLETE"
    report = json.loads((out / "plan.json").read_bytes())
    [issue] = report["issues"]
    assert (issue["code"], issue["location"]["line"]) == (
        "python.global_store_after_callback",
        6,
    )
    prefix = next(
        p
        for p in report["protocol_analysis"]["prefix_fragments"]
        if p["symbol"] == "py:pkg:<init>"
    )
    first, second, deletion = [
        s["value"] for s in read_member(out, prefix["fragment"])["body"][2:]
    ]
    assert [s["store_opcode"] for s in (first, second)] == [
        "STORE_NAME",
        "STORE_GLOBAL",
    ]
    assert [s["op"] for s in first["evaluation"]] == [
        "module_name_load",
        "source_object_attribute",
        "module_name_load",
        "source_object_call",
    ]
    assert all(
        s["value"] == {"from_offset": s["evaluation"][-1]["offset"]}
        for s in (first, second)
    )
    assert [(t["instruction"]["opcode"], t["name"]) for t in deletion["targets"]] == [
        ("DELETE_NAME", "first"),
        ("DELETE_GLOBAL", "other"),
        ("DELETE_NAME", "first"),
    ]
    assert deletion["evaluation"] == [] and "completed_effects" in deletion["unwind"]
    unit = next(
        u
        for u in json.loads((graph / "graph.json").read_bytes())["units"]
        if u["path"] == "pkg/__init__.py"
    )
    function_id, port = deletion["site"].split("/")
    function = next(
        f
        for f in read_member(graph, unit["graph"])["functions"]
        if f["id"] == function_id
    )
    assert (
        function["nodes"][int(port)]["source_expression"]
        == deletion["source_expression"]
    )

    def run(mode, from_ir):
        log = []
        token, rebound = object(), object()
        globals_ = {"__builtins__": {"__import__": lambda *a: SimpleNamespace(child=0)}}

        class Receiver:
            @property
            def build(self):
                log.append("descriptor")

                def invoke(arg):
                    log.append(("call", arg is token))
                    return arg

                return invoke

        class Locals(dict):
            def __setitem__(self, k, v):
                if k == "first":
                    log.append(("store", v is token))
                    if mode == "rejected_store":
                        raise LookupError("reject")
                super().__setitem__(k, v)

            def __delitem__(self, k):
                if k == "first":
                    log.append(
                        ("delete", self.get(k) is token, globals_.get("other") is token)
                    )
                    if self.get(k) is token:
                        super().__delitem__(k)
                        if mode == "reenter":
                            globals_.pop("other")
                        elif mode == "replace":
                            super().__setitem__("first", rebound)
                        return
                super().__delitem__(k)

        loc = Locals(receiver=Receiver(), arg=token)
        try:
            if from_ir:
                mappings = {"frame.locals": loc, "frame.globals": globals_}
                for store in (first, second):
                    stack = {}
                    for op in store["evaluation"]:
                        if op["op"] == "module_name_load":
                            v = loc[op["instruction"]["name"]]
                        elif op["op"] == "source_object_attribute":
                            v = getattr(stack[op["base"]["from_offset"]], op["name"])
                        else:
                            v = stack[op["target"]["from_offset"]](
                                *[stack[a["from_offset"]] for a in op["args"]]
                            )
                        stack[op["offset"]] = v
                    mappings[store["namespace"]][store["name"]] = stack[
                        store["value"]["from_offset"]
                    ]
                for t in deletion["targets"]:
                    try:
                        del mappings[t["namespace"]][t["name"]]
                    except KeyError:
                        raise NameError(t["name"])
            else:
                exec(code, globals_, loc)
        except (LookupError, NameError) as error:
            log.append(type(error).__name__)
        return log, "first" in loc, "other" in globals_

    for mode in ("missing", "replace", "reenter", "rejected_store"):
        actual = run(mode, False)
        assert run(mode, True) == actual
        assert actual[0][:3] == ["descriptor", ("call", True), ("store", True)]
    assert run("reenter", False)[0][-2:] == [("delete", True, True), "NameError"]
    assert run("rejected_store", False)[0][-1] == "LookupError"


def test_open_sequence_displays_keep_allocation_order_and_tuple_identity(tmp_path):
    from types import SimpleNamespace

    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle import plan

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    source = pkg / "__init__.py"
    source.write_text(
        "from . import child as gone\ndel gone\n"
        "empty = []\nfixed = (1, True, 'member!')\nfirst = [1, True, 'member!']\n"
        "second = [1, True, 'member!']\npair = (argument, tap())\nsmall = [tap(), argument]\n"
        "large = (" + ", ".join(["argument"] * 32) + ")\nboundary = 0 or 1\n"
    )
    code = compile(source.read_bytes(), str(source), "exec")
    (pkg / "child.py").write_text("pass\n")
    entry = tmp_path / "main.py"
    entry.write_text("import pkg\n")
    graph, out = tmp_path / "graph", tmp_path / "plan"
    build(entry, graph)
    source.unlink()
    assert plan(graph, out)["status"] == "INCOMPLETE"
    report = json.loads((out / "plan.json").read_bytes())
    [issue] = report["issues"]
    assert (issue["location"]["line"], issue["code"]) == (
        10,
        "python.global_store_after_callback",
    )
    ref = next(
        r
        for r in report["protocol_analysis"]["prefix_fragments"]
        if r["symbol"] == "py:pkg:<init>"
    )
    stores = {
        s["value"]["name"]: s["value"]
        for s in read_member(out, ref["fragment"])["body"]
        if s.get("value", {}).get("op") == "module_global_store"
    }
    assert list(stores) == [
        "empty",
        "fixed",
        "first",
        "second",
        "pair",
        "small",
        "large",
    ]
    for store in stores.values():
        receipt = [
            i
            for i in store["source_expression"]["instructions"]
            if i["opcode"] != "PUSH_NULL"
        ]
        assert [s["offset"] for s in store["evaluation"]] == [
            ins["offset"] for ins in receipt[:-1]
        ]
        assert store["target"]["instruction"] == receipt[-1]
    [pool] = stores["fixed"]["evaluation"]
    constant = code.co_consts[pool["operand"]]
    assert pool["op"] == "source_object_constant" and type(constant) is tuple
    for name in ("first", "second"):
        alloc, items, fill = stores[name]["evaluation"]
        assert (
            alloc["op"] == "source_object_build_sequence"
            and alloc["opcode"] == "BUILD_LIST"
            and alloc["items"] == []
        )
        assert (
            items["op"] == "source_object_constant"
            and items["operand"] == pool["operand"]
        )
        assert (
            fill["op"] == "source_object_list_update"
            and fill["opcode"] == "LIST_EXTEND"
            and fill["may_raise"]
        )
        assert (
            fill["target"] == stores[name]["value"] == {"from_offset": alloc["offset"]}
        )
        assert fill["value"] == {"from_offset": items["offset"]}
    for name, opcode in (("pair", "BUILD_TUPLE"), ("small", "BUILD_LIST")):
        steps = stores[name]["evaluation"]
        items = steps[-1]["items"]
        assert steps[-1]["opcode"] == opcode and [
            k["from_offset"] for k in items
        ] == sorted(k["from_offset"] for k in items)
        assert steps[-1]["may_raise"] and stores[name]["value"] == {
            "from_offset": steps[-1]["offset"]
        }
    large = stores["large"]["evaluation"]
    assert (
        large[0]["opcode"] == "BUILD_LIST"
        and large[-1]["intrinsic"] == "INTRINSIC_LIST_TO_TUPLE"
    )
    appends = [s for s in large if s.get("opcode") == "LIST_APPEND"]
    assert len(appends) == 32 and all(
        a["target"] == large[-1]["source_list"] == {"from_offset": large[0]["offset"]}
        for a in appends
    )

    arg = object()
    for fail in (False, True):
        calls = []

        def tap():
            calls.append(True)
            if fail:
                raise LookupError("operand")
            return arg

        env = dict(
            argument=arg,
            tap=tap,
            __builtins__={"__import__": lambda *a: SimpleNamespace(child=0)},
        )
        if fail:
            with pytest.raises(LookupError, match="operand"):
                exec(code, env)
            assert "pair" not in env and "small" not in env and calls == [True]
        else:
            exec(code, env)
            assert (
                env["pair"] == (arg, arg)
                and env["small"] == [arg, arg]
                and len(env["large"]) == 32
            )
        assert env["fixed"] is constant and env["first"] is not env["second"]
        assert all(
            a is b is c for a, b, c in zip(constant, env["first"], env["second"])
        )

    index = json.loads((graph / "graph.json").read_bytes())
    unit = next(u for u in index["units"] if u["path"] == "pkg/__init__.py")
    fn = read_member(graph, unit["graph"])
    first = next(
        n
        for f in fn["functions"]
        for n in f["nodes"]
        if n["op"] == "assign" and n["location"]["line"] == 5
    )
    fill = next(
        i
        for i in first["source_expression"]["instructions"]
        if i["opcode"] == "LIST_EXTEND"
    )
    fill["opcode"] = "LIST_APPEND"
    unit["graph"] = member(graph, "units", fn)
    altered = graph / "altered.json"
    write_json(altered, index)
    out = tmp_path / "altered"
    assert plan(altered, out)["status"] == "INCOMPLETE"
    [issue] = json.loads((out / "plan.json").read_bytes())["issues"]
    assert (issue["code"], issue["location"]["line"]) == (
        "python.source_expression_operands",
        5,
    )


def test_open_condition_consumes_one_actual_truth_and_routes_its_result(tmp_path):
    from types import SimpleNamespace

    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle import plan

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    source = pkg / "__init__.py"
    source.write_text(
        "from . import child as gone\ndel gone\n"
        "if not receiver.method(argument()):\n    left()\nelse:\n    right()\n"
        "if receiver.method(argument()):\n    yes()\nboundary = 0 or 1\n"
    )
    code = compile(source.read_bytes(), str(source), "exec")
    (pkg / "child.py").write_text("pass\n")
    entry = tmp_path / "main.py"
    entry.write_text("import pkg\n")
    graph, out = tmp_path / "graph", tmp_path / "plan"
    build(entry, graph)
    source.unlink()
    assert plan(graph, out)["status"] == "INCOMPLETE"
    report = json.loads((out / "plan.json").read_bytes())
    [gap] = report["issues"]
    assert (gap["location"]["line"], gap["code"]) == (
        9,
        "python.global_store_after_callback",
    )
    row = next(
        r
        for r in report["protocol_analysis"]["prefix_fragments"]
        if r["symbol"] == "py:pkg:<init>"
    )
    body = read_member(out, row["fragment"])["body"]
    evaluations = [s for s in body if s["op"] == "decl"]
    branches = [s for s in body if s["op"] == "if"]
    assert len(evaluations) == len(branches) == 2
    for decl, branch, negated in zip(evaluations, branches, (True, False)):
        action = decl["init"]
        region = action["expression"]
        steps = region["steps"]
        assert (
            action["op"] == "module_object_expression"
            and action["external_effect"] == "UNKNOWN"
            and action["may_raise"]
        )
        assert branch["condition"]["name"] == decl["name"] and region["type"] == "bool"
        assert [s["op"] for s in steps] == [
            "module_name_load",
            "source_object_attribute",
            "module_name_load",
            "source_object_call",
            "source_object_call",
            "source_object_truth",
        ]
        arg, call, truth = steps[-3:]
        assert call["args"] == [{"from_offset": arg["offset"]}] and truth["value"] == {
            "from_offset": call["offset"]
        }
        assert (
            region["branch"]["input"]
            == region["result"]["value"]
            == {"from_offset": truth["offset"]}
        )
        assert region["result"]["equals"] is not negated
        jump = region["branch"]["instruction"]
        assert jump["opcode"] == (
            "POP_JUMP_IF_TRUE" if negated else "POP_JUMP_IF_FALSE"
        )
        assert jump["target"] == region["branch"]["taken"]["offset"]
        assert truth["opcode"] == (
            "TO_BOOL" if sys.version_info >= (3, 13) else jump["opcode"]
        )
        assert truth["offset"] <= jump["offset"]
        assert "before_first_operand" in action["on_code_mismatch"]

    for mode in ("true", "false", "error", "argument_error"):
        events = []

        class Decision:
            def __bool__(self):
                events.append("truth")
                env["left"] = lambda: events.append("rebound left")
                if mode == "error":
                    raise RuntimeError("truth")
                return mode == "true"

        class Receiver:
            @property
            def method(self):
                events.append("descriptor")

                def invoke(arg):
                    events.append("call")
                    return Decision()

                return invoke

        def argument():
            events.append("argument")
            if mode == "argument_error":
                raise RuntimeError("argument")

        env = dict(
            __builtins__={"__import__": lambda *a: SimpleNamespace(child=0)},
            receiver=Receiver(),
            argument=argument,
            right=lambda: events.append("right"),
            yes=lambda: events.append("yes"),
        )
        trace = ["descriptor", "argument", "call", "truth"]
        if mode in ("error", "argument_error"):
            with pytest.raises(RuntimeError):
                exec(code, env)
            assert events == (trace if mode == "error" else trace[:2])
        else:
            exec(code, env)
            assert events == trace + (
                ["right"] + trace + ["yes"]
                if mode == "true"
                else ["rebound left"] + trace
            )


def test_open_binary_uses_original_operands_and_transfers_the_slot_result(tmp_path):
    from types import SimpleNamespace

    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle import plan

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    source = pkg / "__init__.py"
    source.write_text(
        "from . import child as gone\ndel gone\n"
        "result = left() + right()\nboundary = 0 or 1\n"
    )
    code = compile(source.read_bytes(), str(source), "exec")
    (pkg / "child.py").write_text("pass\n")
    entry = tmp_path / "entry.py"
    entry.write_text("import pkg\n")
    graph, out = tmp_path / "graph", tmp_path / "plan"
    build(entry, graph)
    source.unlink()
    assert plan(graph, out)["status"] == "INCOMPLETE"
    report = json.loads((out / "plan.json").read_bytes())
    [gap] = report["issues"]
    assert (gap["location"]["line"], gap["code"]) == (
        4,
        "python.global_store_after_callback",
    )
    row = next(
        r
        for r in report["protocol_analysis"]["prefix_fragments"]
        if r["symbol"] == "py:pkg:<init>"
    )
    store = read_member(out, row["fragment"])["body"][-1]["value"]
    steps = store["evaluation"]
    assert [s["op"] for s in steps] == [
        "module_name_load",
        "source_object_call",
        "module_name_load",
        "source_object_call",
        "source_object_binary",
    ]
    lhs, rhs, binary = steps[1], steps[3], steps[4]
    assert binary["left"] == {"from_offset": lhs["offset"]} and binary["right"] == {
        "from_offset": rhs["offset"]
    }
    assert (
        store["value"] == {"from_offset": binary["offset"]}
        and store["may_raise"]
        and binary["may_raise"]
    )
    source_op = store["source_expression"]["instructions"][-2]
    assert (
        source_op["opcode"] == "BINARY_OP" and source_op["offset"] == binary["offset"]
    )
    assert (
        source_op["oparg"] == binary["oparg"] == 0
        and source_op["operator"] == binary["operator"] == "add"
    )

    for mode in (
        "subclass",
        "fallback",
        "unsupported",
        "right_failure",
        "slot_failure",
    ):
        events = []
        token = object()

        class L:
            def __add__(self, r):
                events.append("left slot")
                return token if mode == "fallback" else NotImplemented

        class R(L):
            def __radd__(self, l):
                events.append("right slot")
                if mode == "slot_failure":
                    raise LookupError("slot")
                return token if mode == "subclass" else NotImplemented

        def left():
            events.append("left operand")
            return L()

        def right():
            events.append("right operand")
            if mode == "right_failure":
                raise LookupError("right")
            return R()

        env = dict(
            left=left,
            right=right,
            __builtins__={"__import__": lambda *a: SimpleNamespace(child=0)},
        )
        if mode in {"subclass", "fallback"}:
            exec(code, env)
            assert env["result"] is token
        else:
            with pytest.raises(TypeError if mode == "unsupported" else LookupError):
                exec(code, env)
            assert "result" not in env
        suffix = [] if mode == "right_failure" else ["right slot"]
        if mode in {"fallback", "unsupported"}:
            suffix.append("left slot")
        assert events == ["left operand", "right operand", *suffix]


def test_open_function_definition_publishes_fresh_object_with_current_builtins(
    tmp_path,
):
    from types import ModuleType, SimpleNamespace

    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle import plan

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    source = pkg / "__init__.py"
    source.write_text(
        "from . import child as gone\ndel gone\nchange()\n"
        "def first():\n    invoked()\n    return token\n"
        "def other():\n    return token\n"
        "def restricted(value=option()):\n    return value\n"
        "def declaration():\n    global other\n"
    )
    code = compile(source.read_bytes(), str(source), "exec")
    (pkg / "child.py").write_text("pass\n")
    entry = tmp_path / "entry.py"
    entry.write_text("import pkg\n")
    graph, out = tmp_path / "graph", tmp_path / "plan"
    build(entry, graph)
    source.unlink()
    assert plan(graph, out)["status"] == "INCOMPLETE"
    report = json.loads((out / "plan.json").read_bytes())
    [gap] = report["issues"]
    assert (gap["location"]["line"], gap["code"]) == (
        9,
        "python.source_definition_evidence",
    )
    ref = next(
        r
        for r in report["protocol_analysis"]["prefix_fragments"]
        if r["symbol"] == "py:pkg:<init>"
    )
    defs = [
        s["value"]
        for s in read_member(out, ref["fragment"])["body"]
        if "source_definition" in s.get("value", {})
    ]
    assert [d["store_opcode"] for d in defs] == ["STORE_NAME", "STORE_GLOBAL"]
    assert [d["name"] for d in defs] == ["first", "other"]
    for d in defs:
        load, make = d["evaluation"]
        port = d["source_definition"]
        assert d["op"] == "module_global_store" and d["external_effect"] == "UNKNOWN"
        assert (
            load["op"] == "source_object_constant"
            and load["source"] == "current_frame_co_consts"
        )
        assert (
            load["expected_code"] == port["child"]
            and code.co_consts[load["operand"]].co_name == d["name"]
        )
        assert make["op"] == "source_object_make_function" and make["code"] == {
            "from_offset": load["offset"]
        }
        assert (
            make["globals"] == "current_frame_globals"
            and make["builtins"]["absent"] == "current_frame_builtins"
        )
        assert make["may_raise"] and d["value"] == {"from_offset": make["offset"]}
        assert [
            load["offset"],
            make["offset"],
            d["target"]["instruction"]["offset"],
        ] == [i["offset"] for i in port["instructions"]]

    instances = []
    for mode in (
        "replace",
        "module",
        "absent",
        "sentinel",
        "store_failure",
        "reentrant_store",
    ):
        previous = object()
        replacement = object()
        before = {"token": previous, "__import__": lambda *a: SimpleNamespace(child=0)}
        after = {"token": replacement}
        module = ModuleType("replacement")
        module.token = replacement
        glob = {"__builtins__": before}
        made = []
        events = []

        class Local(dict):
            def __setitem__(self, key, value):
                if key == "first":
                    made.append(value)
                    if mode == "store_failure":
                        raise LookupError("store")
                super().__setitem__(key, value)

        loc = Local()

        class Old:
            def __del__(self):
                if mode == "store_failure":
                    return
                events.append(loc["first"] is made[-1])
                if mode == "reentrant_store":
                    loc["first"] = replacement

        dict.__setitem__(loc, "first", Old())

        def change():
            if mode == "absent":
                del glob["__builtins__"]
            else:
                glob["__builtins__"] = {"module": module, "sentinel": 17}.get(
                    mode, after
                )

        loc.update(change=change, option=lambda: None)
        glob["invoked"] = lambda: events.append("body")
        if mode == "store_failure":
            with pytest.raises(LookupError, match="store"):
                exec(code, glob, loc)
            assert "other" not in glob and not events
            del loc["first"]
        else:
            exec(code, glob, loc)
            fn = made[0]
            assert (
                events == [True] and fn is not glob["other"] and fn.__globals__ is glob
            )
            expected = {
                "module": module.__dict__,
                "absent": before,
                "sentinel": 17,
            }.get(mode, after)
            assert (
                fn.__builtins__ is expected and glob["other"].__builtins__ is expected
            )
            assert fn.__code__ is code.co_consts[defs[0]["evaluation"][0]["operand"]]
            assert all(fn is not prior for prior in instances)
            instances.append(fn)
            assert loc["first"] is (replacement if mode == "reentrant_store" else fn)


def test_initializer_return_does_not_prove_caller_import_binding(tmp_path, monkeypatch):
    import builtins
    from types import ModuleType
    from unittest.mock import patch

    from scar.frontend import build
    from scar.middle import plan

    name = "identity_probe_package"
    package = tmp_path / name
    package.mkdir()
    source = package / "__init__.py"
    source.write_text(
        "from . import child as unused\ndel unused\n"
        "tag = 'original'\non_import_body()\n"
    )
    (package / "child.py").write_text("pass\n")
    entry = tmp_path / "main.py"
    entry.write_text(f"import {name}\n")
    code = compile(entry.read_bytes(), str(entry), "exec")
    monkeypatch.syspath_prepend(str(tmp_path))

    for mode in ("keep", "replace", "delete", "raise", "hook_return"):
        seen = []
        replacement = ModuleType(name)
        old = object()

        def on_import_body():
            seen.append(sys.modules[name])
            if mode == "replace":
                sys.modules[name] = replacement
            elif mode == "delete":
                del sys.modules[name]
            elif mode == "raise":
                raise LookupError("body")

        def import_hook(*args):
            result = builtins.__import__(*args)
            return replacement if mode == "hook_return" else result

        monkeypatch.setattr(builtins, "on_import_body", on_import_body, raising=False)
        frame = {
            "__builtins__": dict(vars(builtins), __import__=import_hook),
            name: old,
        }
        with patch.dict(sys.modules):
            if mode in {"delete", "raise"}:
                with pytest.raises(
                    KeyError if mode == "delete" else LookupError
                ) as exc:
                    exec(code, frame)
                assert exc.value.args == (name if mode == "delete" else "body",)
                assert frame[name] is old and name not in sys.modules
            else:
                exec(code, frame)
                assert frame[name] is (seen[0] if mode == "keep" else replacement)
                assert sys.modules[name] is (
                    replacement if mode == "replace" else seen[0]
                )
            assert len(seen) == 1 and seen[0].tag == "original"

    graph, out = tmp_path / "graph", tmp_path / "plan"
    build(entry, graph)
    source.unlink()
    assert plan(graph, out)["status"] == "INCOMPLETE"
    from scar.contract import read_member

    report = json.loads((out / "plan.json").read_bytes())
    [gap] = report["issues"]
    assert (
        gap["code"] == "python.loader_activation_contract"
        and gap["location"]["line"] == 1
    )
    analysis = report["protocol_analysis"]
    [conditional] = analysis["conditional_loader_fragments"]
    [caller] = analysis["closed_fragments"]
    assert caller["symbol"] == "__main__" and conditional["symbol"] != caller["symbol"]
    assert conditional["completion"] == "conditional_source_body"
    [published] = read_member(out, caller["function"])["body"]
    action = published["value"]
    assert action["op"] == "module_import_publish"
    assert action["import_call"]["result"] == action["binding"]["value"]["result"]
    assert action["import_call"]["body"] == conditional["loader_ports"]["body"]
    assert action["site"] == conditional["site"] == gap["site"]
    ir = json.loads((out / "exec-ir.json").read_bytes())
    ledger = read_member(out, ir["module"]["source_imports"])
    current, child = ledger["occurrences"]
    assert (current["activation"], current["effect_coverage"]) == ("visited", "MODELED")
    assert [p["role"] for p in current["producer_sites"]] == ["source_import"]
    assert (
        child["owner"] == conditional["symbol"] and child["activation"] == "unvisited"
    )
    assert not child["producer_sites"]
    assert not ir["functions"]
    evidence = conditional["rejected_assumption"]
    assert evidence["publication"]["publication"]["module"] == name
    assert evidence["publication"]["state"] == "initializing"
    state = evidence["registry_after_body"]
    assert state["state"] == "unresolved" and state["location"]["line"] == 4
    assert state == evidence["last_namespace_change"]
    assert evidence["open_namespace"]["site"] == state["site"]
    assert evidence["comparison_windows"] == []
    assert evidence["producer_role"] == "module_body"


def test_nested_loader_evidence_keeps_conditional_children_out_of_the_active_ledger(
    tmp_path,
):
    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle import plan

    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / "__init__.py").write_text(
        "from . import child as unused\ndel unused\non_import_body()\n"
    )
    (inner / "child.py").write_text("pass\n")
    (tmp_path / "outer.py").write_text("import inner\n")
    entry = tmp_path / "main.py"
    entry.write_text("import outer\n")
    graph, out = tmp_path / "graph", tmp_path / "plan"
    build(entry, graph)
    plan(graph, out)
    report = json.loads((out / "plan.json").read_bytes())
    outer, child = report["protocol_analysis"]["conditional_loader_fragments"]
    assert outer["owner"] == "__main__"
    assert outer["symbol"] == child["owner"] == "py:outer:<init>"
    assert outer["nested_imports"] == [{"owner": child["owner"], "site": child["site"]}]
    assert child["symbol"] == "py:inner:<init>" and not child["nested_imports"]
    ledger = read_member(out, report["source_imports"])["occurrences"]
    rows = {(row["owner"], row["site"]): row for row in ledger}
    for candidate, active in ((outer, True), (child, False)):
        row = rows[candidate["owner"], candidate["site"]]
        assert (row["activation"] == "visited") is active
        assert bool(row["producer_sites"]) is active


def test_plain_and_projected_alias_store_actual_objects_in_source_order(
    tmp_path, monkeypatch
):
    from types import ModuleType

    from scar.frontend import build
    from scar.middle.binding import bind
    from scar.middle.normalization.namespace_operations import publish_plain_import
    from scar.middle.normalization.protocols import _Frame, _Project

    source = tmp_path / "main.py"
    source.write_text(
        "import tree as first\nimport tree.branch\nimport tree.branch.leaf as named\n"
    )
    code = compile(source.read_bytes(), str(source), "exec")
    branch = tmp_path / "tree" / "branch"
    branch.mkdir(parents=True)
    for f in (
        branch.parent / "__init__.py",
        branch / "__init__.py",
        branch / "leaf.py",
    ):
        f.write_text("label = 'candidate'\n")
    graph = tmp_path / "graph"
    build(source, graph)
    source.unlink()
    bound, issues, _ = bind(json.loads((graph / "graph.json").read_bytes()), graph)
    assert not issues
    functions = {f["name"]: f for f in bound["functions"]}
    project = _Project(functions)
    entry = functions[bound["entry"]]
    frame = _Frame(project, entry, ())
    steps = []
    for node in entry["body"]:
        if node["op"] == "python_import":
            row = project.source_imports.get(frame.name, node)
            assert not publish_plain_import(frame, node, row, steps, source_code={})
            if "source_import_projection" in node:
                legacy = {
                    k: v for k, v in node.items() if k != "source_import_projection"
                }
                assert not publish_plain_import(frame, legacy, row, steps)
            assert row["producer_sites"] == []
            assert publish_plain_import(
                frame, node, row, steps, source_code=frame.scope["source_code"]
            )
            assert [p["role"] for p in row["producer_sites"]] == ["source_import"]
    plain, dotted, aliased = [s["value"] for s in steps]
    for op, name in ((plain, "first"), (dotted, "tree"), (aliased, "named")):
        assert op["op"] == "module_import_publish" and op["binding"]["name"] == name
        assert op["may_raise"] and op["external_effect"] == "UNKNOWN"
    imported = {"result": "import_result"}
    assert plain["binding"]["value"] == dotted["binding"]["value"] == imported
    project_member, swap, release, project_leaf = aliased["projection"]["steps"]
    assert [
        s["instruction"]["opcode"]
        for s in (project_member, swap, release, project_leaf)
    ] == ["IMPORT_FROM", "SWAP", "POP_TOP", "IMPORT_FROM"]
    assert project_member["receiver"] == release["release"] == imported
    assert project_leaf["receiver"] == project_member["result"]
    assert aliased["binding"]["value"] == project_leaf["result"]
    cleanup = aliased["projection"]["after_store"]
    assert cleanup["release"] == project_leaf["receiver"]
    assert (
        cleanup["instruction"]["offset"]
        > aliased["source_import_projection"]["store"]["offset"]
    )
    assert [s["instruction"]["name"] for s in (project_member, project_leaf)] == [
        "branch",
        "leaf",
    ]
    assert (
        project.imports.publications[aliased["binding"]["slot"]]["occurrence"]
        == aliased["occurrence"]
    )

    final, previous = object(), object()
    monkeypatch.setitem(sys.modules, "offered.branch.leaf", final)
    monkeypatch.setitem(sys.modules, "tree.branch.leaf", previous)
    for mode in ("success", "fallback", "lookup_error", "store_error", "reenter"):
        events = []

        class Part(ModuleType):
            def __getattr__(self, key):
                events.append(("lookup", key))
                if key == "branch":
                    return Part("offered.branch")
                if mode == "fallback":
                    raise AttributeError(key)
                if mode == "lookup_error":
                    raise LookupError("lookup")
                assert key == "leaf"
                return final

            def __del__(self):
                events.append(("release", self.__name__))
                if self.__name__ == "offered.branch" and mode == "reenter":
                    events.append(("stored_on_release", local["named"] is final))
                    local["named"] = previous

        class Locals(dict):
            def __setitem__(self, key, value):
                if key == "named" and value is final:
                    events.append(("store", key))
                    if mode == "store_error":
                        super().__setitem__(key, value)
                        raise LookupError("store")
                super().__setitem__(key, value)

        first, top = object(), object()

        def importer(name, gl, loc, fromlist, level):
            events.append(("import", name))
            assert gl is glob and loc is local and fromlist is None and level == 0
            if name in {"tree", "tree.branch"}:
                return first if name == "tree" else top
            return Part("offered")

        local = Locals(named=previous)
        glob = {"__builtins__": {"__import__": importer}}
        if mode.endswith("_error"):
            with pytest.raises(LookupError, match=mode.split("_")[0]):
                exec(code, glob, local)
        else:
            exec(code, glob, local)
        assert local["first"] is first and local["tree"] is top
        assert local["named"] is (
            previous if mode in {"lookup_error", "reenter"} else final
        )
        assert events == [
            ("import", "tree"),
            ("import", "tree.branch"),
            ("import", "tree.branch.leaf"),
            ("lookup", "branch"),
            ("release", "offered"),
            ("lookup", "leaf"),
            *([] if mode == "lookup_error" else [("store", "named")]),
            ("release", "offered.branch"),
            *([("stored_on_release", True)] if mode == "reenter" else []),
        ]


def test_named_fromlist_retains_receiver_and_interleaves_each_mapping_store(tmp_path):
    from scar.frontend import build
    from scar.middle.binding import bind
    from scar.middle.normalization.namespace_operations import publish_plain_import
    from scar.middle.normalization.protocols import _Frame, _Project

    entry = tmp_path / "main.py"
    entry.write_text(
        "import math\nglobal first\nfrom math import sin as first, sin as second, cos as third\n"
    )
    code = compile(entry.read_bytes(), str(entry), "exec")
    graph = tmp_path / "graph"
    build(entry, graph)
    entry.unlink()
    bound, issues, _ = bind(json.loads((graph / "graph.json").read_bytes()), graph)
    assert not issues
    functions = {f["name"]: f for f in bound["functions"]}
    project = _Project(functions)
    fn = functions[bound["entry"]]
    frame = _Frame(project, fn, ())
    project.active.append(frame.name)
    first, *members = [n for n in fn["body"] if n["op"] == "python_import"]
    bootstrap = project.source_imports.get(frame.name, first)
    assert publish_plain_import(
        frame, first, bootstrap, [], source_code=frame.scope["source_code"]
    )
    steps = []
    assert frame.sequence(members, steps)
    [statement] = steps
    action = statement["value"]
    assert action["op"] == "module_fromlist_publish" and action["may_raise"]
    assert action["source_fromlist"] == members[0]["source_fromlist"]
    assert (
        action["source_code"] == frame.scope["source_code"]
        and "source_equivalent" in action["on_code_mismatch"]
    )
    result = {"result": action["import_call"]["result"]}
    stores = action["bindings"]
    assert [b["read"]["receiver"] for b in stores] == [result] * 3
    assert [b["value"] for b in stores] == [b["read"]["result"] for b in stores]
    assert [b["read"]["instruction"]["name"] for b in stores] == ["sin", "sin", "cos"]
    assert [b["namespace"] for b in stores] == [
        "frame.globals",
        "frame.locals",
        "frame.locals",
    ]
    assert [b["site"] for b in stores] == members[0]["fromlist_binding_sites"]
    assert action["after_store"]["release"] == result
    offsets = [
        offset
        for b in stores
        for offset in (b["read"]["instruction"]["offset"], b["offset"])
    ] + [action["after_store"]["instruction"]["offset"]]
    assert offsets == sorted(set(offsets))
    [producer] = project.source_imports.get(frame.name, members[0])["producer_sites"]
    assert producer["role"] == "source_import" and producer["opcode"] == action["op"]
    assert code.co_consts[action["import_call"]["arguments"][3]["index"]] == (
        "sin",
        "sin",
        "cos",
    )

    for mode in ("success", "read_error", "store_error", "import_error"):
        events = []

        class Returned:
            def __getattr__(self, name):
                assert glob.get("first") == (None if not events.count("sin") else 1)
                events.append(name)
                if name == "cos" and mode == "read_error":
                    raise LookupError("read")
                return events.count(name)

        class Locals(dict):
            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                if key == "second":
                    events.append("store")
                    if mode == "store_error":
                        raise LookupError("store")

        def importer(name, gl, loc, fromlist, level):
            assert name == "math" and gl is glob and loc is local and level == 0
            if fromlist is not None:
                assert (
                    fromlist
                    is code.co_consts[action["import_call"]["arguments"][3]["index"]]
                )
                events.append("loader")
                if mode == "import_error":
                    raise LookupError("import")
            return Returned()

        glob, local = {"__builtins__": {"__import__": importer}}, Locals()
        if mode == "success":
            exec(code, glob, local)
        else:
            with pytest.raises(LookupError, match=mode.split("_")[0]):
                exec(code, glob, local)
        expected = ["loader"]
        if mode != "import_error":
            expected += ["sin", "sin", "store"]
            assert glob["first"] == 1 and local["second"] == 2
            if mode != "store_error":
                expected.append("cos")
        else:
            assert "first" not in glob and "second" not in local
        assert ("third" in local) == (mode == "success")
        assert events == expected


def test_module_for_uses_actual_callable_iterator_mapping_and_exhaustion(tmp_path):
    from scar.contract import walk_operations
    from scar.frontend import build
    from scar.middle.binding import bind
    from scar.middle.normalization.namespace_operations import publish_plain_import
    from scar.middle.normalization.protocols import _Frame, _Project, _Unclosed

    def prepare(name, body, *, prefix=""):
        source = tmp_path / f"{name}.py"
        source.write_text(prefix + "import math as item\n" + body)
        code = compile(source.read_bytes(), str(source), "exec")
        folder = tmp_path / name
        build(source, folder)
        source.unlink()
        module, issues, _ = bind(
            json.loads((folder / "graph.json").read_bytes()), folder
        )
        assert not issues
        functions = {fn["name"]: fn for fn in module["functions"]}
        entry = functions[module["entry"]]
        project = _Project(functions)
        frame = _Frame(project, entry, ())
        project.active.append(frame.name)
        nodes = entry["body"]
        imp = next(n for n in nodes if n["op"] == "python_import")
        assert publish_plain_import(
            frame,
            imp,
            project.source_imports.get(frame.name, imp),
            [],
            source_code=frame.scope["source_code"],
        )
        return code, frame, nodes[nodes.index(imp) + 1 :]

    code, frame, nodes = prepare(
        "main",
        "for item in range(settings.count):\n    consume(item)\nelse:\n    ended()\nafter(item)\n",
    )
    body = []
    assert frame.sequence(nodes, body)
    loop, after = body
    assert loop["op"] == "module_object_for" and loop["external_effect"] == "UNKNOWN"
    assert (
        loop["source_code"] == frame.scope["source_code"]
        and "source_equivalent" in loop["on_code_mismatch"]
    )
    evaluation = loop["evaluation"]
    assert [e["op"] for e in evaluation] == [
        "module_name_load",
        "module_name_load",
        "source_object_attribute",
        "source_object_call",
    ]
    assert evaluation[0]["instruction"]["name"] == "range"
    assert loop["acquire"]["iterable"] == {"from_offset": evaluation[-1]["offset"]}
    assert loop["advance"]["iterator"] == loop["acquire"]["result"]
    assert (
        loop["target"]["value"] == loop["advance"]["item"]
        and loop["target"]["namespace"] == "frame.locals"
    )
    assert (
        loop["edges"]["backedge"]["target"] == loop["advance"]["instruction"]["offset"]
    )
    assert (
        loop["advance"]["instruction"]["target"]
        == loop["edges"]["cleanup"][0]["offset"]
    )
    assert len(loop["body"]) == len(loop["else"]) == 1
    assert loop["edges"]["exhausted"]["kind"] == "suite"
    [read] = [
        n
        for n in walk_operations(after)
        if n["op"] == "module_name_load" and n["instruction"]["name"] == "item"
    ]
    assert "source_evidence" not in read

    _, partial, tail = prepare(
        "partial",
        "for item_slot in range(settings.count):\n    consume(item_slot)\n    boundary = 0 or 37\nafter()\n",
        prefix="global item_slot\n",
    )
    with pytest.raises(_Unclosed) as failure:
        partial.sequence(tail, [])
    assert failure.value.issue["code"] == "python.global_store_after_callback"
    frontier = partial.prefix_fragment()["frontier"]
    assert frontier["iteration"]["target"]["namespace"] == "frame.globals"
    [arm] = frontier["regions"]
    assert arm["path"] == ["body"] and len(arm["body"]) == 1
    assert arm["frontier"]["location"]["line"] == 5

    _, unproved, tail = prepare(
        "early_break", "for item in range(settings.count):\n    break\nafter()\n"
    )
    original = next(n for n in tail if n["op"] == "for")
    assert "source_expression" not in original
    with pytest.raises(_Unclosed) as gap:
        unproved.sequence(tail, [])
    assert gap.value.issue["code"] == "python.builtin_after_callback"

    old, yielded, changed = object(), object(), object()
    for mode in (
        "empty",
        "normal",
        "acquire_error",
        "advance_error",
        "store_error",
        "body_stop",
        "release_reentry",
    ):
        events = []

        class Locals(dict):
            def __setitem__(self, name, value):
                super().__setitem__(name, value)
                if name == "item" and value is yielded:
                    events.append("store")
                    if mode == "store_error":
                        raise LookupError("store")

        class Settings:
            @property
            def count(self):
                events.append("argument")
                local["range"] = lambda arg: pytest.fail("reloaded target")
                return old

        class Iterator:
            used = False

            def __next__(self):
                events.append("next")
                if mode == "advance_error":
                    raise RuntimeError("next")
                if self.used or mode == "empty":
                    raise StopIteration
                self.used = True
                return yielded

            def __del__(self):
                events.append("release")
                if mode == "release_reentry":
                    local["item"] = changed

        class Iterable:
            def __iter__(self):
                events.append("iter")
                if mode == "acquire_error":
                    raise StopIteration("acquire")
                return Iterator()

        def make(arg):
            assert arg is old
            events.append("call")
            return Iterable()

        def consume(value):
            assert value is yielded
            events.append("body")
            if mode == "body_stop":
                raise StopIteration("body")

        local = Locals(
            range=make,
            settings=Settings(),
            consume=consume,
            ended=lambda: events.append("else"),
            after=lambda v: events.append(("after", v)),
        )
        env = {"__builtins__": {"__import__": lambda *args: old}}
        if mode in {"empty", "normal", "release_reentry"}:
            exec(code, env, local)
            expected = (
                old
                if mode == "empty"
                else changed if mode == "release_reentry" else yielded
            )
            assert events[-3:] == ["release", "else", ("after", expected)]
        else:
            with pytest.raises(
                {
                    "acquire_error": StopIteration,
                    "advance_error": RuntimeError,
                    "store_error": LookupError,
                    "body_stop": StopIteration,
                }[mode]
            ):
                exec(code, env, local)
            assert "else" not in events and not any(
                isinstance(e, tuple) for e in events
            )
        assert events[:3] == ["argument", "call", "iter"] and events.count("call") == 1
        assert events.count("store") == int(
            mode not in {"empty", "acquire_error", "advance_error"}
        )
        assert events.count("body") == int(
            mode in {"normal", "body_stop", "release_reentry"}
        )


def test_imported_function_candidate_uses_actual_selected_callable_and_frame(tmp_path):
    from types import FunctionType

    from scar.contract import walk_operations
    from scar.frontend import build
    from scar.middle.binding import bind
    from scar.middle.normalization import namespace_operations, protocols

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "child.py").write_text("pass\n")
    (package / "__init__.py").write_text(
        "from . import child as old\ndel old\n"
        "def f(value):\n    return value\n"
        "on_import()\n"
    )
    entry = tmp_path / "entry.py"
    entry.write_text(
        "import pkg as receiver\nfrom pkg import f as chosen\n"
        "receiver.f(arg())\nchosen(arg())\nunrelated.f(arg())\n"
    )
    graph = tmp_path / "graph"
    build(entry, graph)
    bound, issues, _ = bind(json.loads((graph / "graph.json").read_bytes()), graph)
    assert not issues
    functions = {
        f["name"]: f
        for f in bound["functions"]
        if f.get("execution_kind") == "python_protocol"
    }
    project, issue = protocols._compile_protocol_entry(bound, functions)
    assert not issue
    conditional = protocols._conditional_loader_fragments(project)
    assert namespace_operations.link_conditional_function_calls(
        project, conditional
    ) == {"py:pkg:f"}
    calls = [
        op
        for op in walk_operations(project.closed[bound["entry"]]["body"])
        if op["op"] == "source_object_call"
    ]
    edges = {
        c["location"]["line"]: c["conditional_activation"]
        for c in calls
        if "conditional_activation" in c
    }
    assert set(edges) == {3, 4}
    property_edge, named_edge = edges[3], edges[4]
    assert (
        property_edge["origin"]["attribute_site"]
        and "read_offset" not in property_edge["origin"]
    )
    assert named_edge["origin"]["read_offset"] < named_edge["origin"]["store_offset"]
    assert property_edge["origin"]["occurrence"] != named_edge["origin"]["occurrence"]
    assert property_edge["candidates"] == named_edge["candidates"]
    [creation] = property_edge["candidates"]
    assert (
        creation["symbol"] == "py:pkg:f"
        and creation["make_offset"] < creation["store_offset"]
    )
    assert (
        creation["code"]["name"] == "f" and "bound_arguments" in property_edge["inputs"]
    )

    def original(value):
        events.append(("original", value))

    def other(value):
        events.append(("other", value))

    for mode in ("same", "clone", "code_replaced", "argument_raises"):
        events = []
        # Both functions capture the same cell, so CPython permits replacing
        # __code__ while evaluating the argument to the already-loaded function.
        witness = FunctionType(
            original.__code__, globals(), closure=original.__closure__
        )
        clone = FunctionType(
            original.__code__, dict(globals()), closure=original.__closure__
        )
        selected = clone if mode == "clone" else witness

        class Receiver:
            @property
            def f(self):
                events.append("lookup")
                return selected

        def arg():
            events.append("argument")
            if mode == "argument_raises":
                raise KeyError("argument")
            if mode == "code_replaced":
                witness.__code__ = other.__code__
            return 4

        if mode == "argument_raises":
            with pytest.raises(KeyError, match="argument"):
                Receiver().f(arg())
            assert events == ["lookup", "argument"]
        else:
            Receiver().f(arg())
            assert (selected is witness and selected.__code__ is original.__code__) is (
                mode == "same"
            )
            assert events == [
                "lookup",
                "argument",
                ("other" if mode == "code_replaced" else "original", 4),
            ]


def test_conditional_module_rebuilds_from_unknown_bindings_and_owns_its_imports(
    tmp_path,
):
    from copy import deepcopy

    from scar.contract import walk_operations
    from scar.frontend import build
    from scar.middle.binding import bind
    from scar.middle.normalization.protocols import normalize_protocols

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "child.py").write_text("pass\n")
    (package / "__init__.py").write_text(
        '"""module documentation"""\n'
        "import math\n"
        "from . import child as unused\ndel unused\n"
        "observed = __name__\n"
        "def callback():\n    return observed\n"
        "touch()\n"
    )
    entry = tmp_path / "entry.py"
    entry.write_text("import pkg\n")
    graph = tmp_path / "graph"
    build(entry, graph)
    bound, issues, _ = bind(json.loads((graph / "graph.json").read_bytes()), graph)
    assert not issues
    legacy = deepcopy(bound)
    module, _, [gap] = normalize_protocols(bound)
    assert gap["code"] == "python.loader_activation_contract"
    [candidate] = module["protocol_analysis"]["conditional_loader_fragments"]
    handoff = candidate["activation_handoff"]
    assert candidate["completion"] == "conditional_source_body"
    assert handoff["status"] == "INCOMPLETE" and "source_issue" not in handoff
    symbol = candidate["symbol"]
    source = next(f for f in bound["functions"] if f["name"] == symbol)
    assert handoff["entry"]["code"] == source["python_scope"]["source_code"]
    ops = list(walk_operations(candidate["function"]["body"]))
    assert ops and {
        o["location"]["line"] for o in ops if o["op"] == "module_import_publish"
    } == {2}
    assert any(
        o["op"] == "module_name_load" and o["location"]["line"] == 5 for o in ops
    )
    assert any(
        o["op"] == "module_global_store" and o["location"]["line"] == 1 for o in ops
    )
    [created] = handoff["function_activations"]
    assert (
        created["code"]["name"] == "callback"
        and len(created["code"]["code_sha256"]) == 64
    )
    unresolved = {d["kind"]: d for d in handoff["unresolved"]}
    assert (
        unresolved["published_function_activation"]["when"]
        == "if_actual_created_function_is_called"
    )
    [child] = unresolved["importer_selected_fromlist_activations"]["checks"]
    assert child["child_candidate"]["module"] == "pkg.child"
    rows = candidate["source_imports"]["occurrences"]
    assert {row["location"]["line"] for row in rows} == {2, 3}
    assert all(
        row["owner"] == symbol
        and row["producer_sites"]
        and row["activation"] == "visited"
        for row in rows
    )
    assert all(
        not row["producer_sites"]
        for row in module["source_imports"]["occurrences"]
        if row["owner"] == symbol
    )
    assert (
        "module_import_publish"
        in candidate["operator_requirements"]["module_registry"]["operations"]
    )
    fn = next(f for f in legacy["functions"] if f["name"] == symbol)
    first = fn["body"][0]
    assert first["op"] == "assign" and first["target"]["name"].endswith(":__doc__")
    fn["body"][0] = {key: first[key] for key in ("site", "location", "value")} | {
        "op": "expr"
    }
    module, _, [gap] = normalize_protocols(legacy)
    [candidate] = module["protocol_analysis"]["conditional_loader_fragments"]
    assert (
        gap["code"] == "python.module_docstring_store" and gap["location"]["line"] == 1
    )
    assert candidate["completion"] == "conditional_source_prefix"
    assert candidate["activation_handoff"]["source_issue"] == gap
    assert not candidate["function"]["body"]
    assert not any(
        row["producer_sites"] for row in candidate["source_imports"]["occurrences"]
    )


def test_conditional_callee_reads_bound_slots_and_returns_through_original_call(
    tmp_path,
):
    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle import plan

    package = tmp_path / "lib"
    package.mkdir()
    (package / "child.py").write_text("pass\n")
    source = package / "__init__.py"
    source.write_text(
        "from . import child as old\ndel old\n"
        "def transform(item):\n"
        "    result = api.apply(item, 'v')\n"
        "    return result\n"
        "def label():\n    return 'v'\n"
        "callback()\n"
    )
    entry = tmp_path / "run.py"
    entry.write_text("import lib\nlib.transform(get_item())\nlib.label()\n")
    graph, output = tmp_path / "graph", tmp_path / "plan"
    build(entry, graph)
    source.unlink()
    entry.unlink()
    plan(graph / "graph.json", output)
    report = json.loads((output / "plan.json").read_bytes())
    fragments = {
        f["symbol"]: f
        for f in report["protocol_analysis"]["conditional_function_fragments"]
    }
    fragment = fragments["py:lib:transform"]
    label = fragments["py:lib:label"]
    assert label["completion"] == "conditional_source_body"
    label_body = read_member(output, label["function"])
    label_result = label_body["body"][0]["steps"]
    slot = (
        label_result[0].get("operand")
        if len(label_result) == 2
        else label_result[0]["value"]["index"]
    )
    assert label_body["constants"][str(slot)] == dict(kind="str", value="v")
    assert fragment["completion"] == "conditional_source_body"
    assert fragment["entry"]["parameters"] == [
        dict(name="item", binding="positional", slot=0)
    ]
    function = read_member(output, fragment["function"])
    assign, exit_ = function["body"]
    load, attribute, argument, constant, call, store = assign["steps"]
    assert load["lookup_order"] == ["frame.globals", "frame.builtins"]
    assert attribute["base"] == {"from_offset": load["offset"]}
    assert argument["op"] == "frame_local_load" and argument["slot"] == 0
    assert (
        function["constants"][str(constant["operand"])]
        == label_body["constants"][str(slot)]
    )
    assert (
        fragment["entry"]["code"]["code_sha256"]
        != label["entry"]["code"]["code_sha256"]
    )
    assert call["target"] == {"from_offset": attribute["offset"]}
    assert call["args"] == [{"from_offset": a["offset"]} for a in (argument, constant)]
    assert store["op"] == "frame_local_store" and store["slot"] == 1
    assert store["value"] == {"from_offset": call["offset"]}
    reread, returned = exit_["steps"]
    assert reread["op"] == "frame_local_load" and reread["slot"] == store["slot"]
    assert returned["op"] == "frame_return"
    assert returned["value"] == {"from_offset": reread["offset"]}
    assert report["status"] == "INCOMPLETE"
    assert not json.loads((output / "exec-ir.json").read_bytes())["functions"]


def test_source_dispatch_replays_closed_graph_imports_together_and_retains_source_gap(
    tmp_path, monkeypatch
):
    from copy import deepcopy
    from scar.contract import EXECUTION_ANNOTATIONS, walk_operations
    from scar.frontend import build
    from scar.middle.binding import bind
    from scar.middle.normalization import protocols

    for lib in ("first", "second"):
        (tmp_path / (lib + ".py")).write_text(f"def read():\n    return '{lib}'\n")
    entry = tmp_path / "main.py"
    entry.write_text(
        "import first\nimport second\nprint(first.read())\nprint(second.read())\n"
    )
    graph = tmp_path / "graph"
    build(entry, graph)
    bound, issues, _ = bind(json.loads((graph / "graph.json").read_bytes()), graph)
    assert not issues
    legacy = deepcopy(bound)
    attempts = []
    attempt = protocols._attempt_protocol_entry

    def counted(*args, **kwargs):
        project, issue = attempt(*args, **kwargs)
        attempts.append(
            (kwargs.get("open_entry", False), args[3], project.source_index)
        )
        return project, issue

    monkeypatch.setattr(protocols, "_attempt_protocol_entry", counted)
    module, _, issues = protocols.normalize_protocols(bound)
    assert not issues
    assert [a[0] for a in attempts] == [False, True] and attempts[1][1] is attempts[0][
        2
    ]
    assert all(
        {p["role"] for p in row["producer_sites"]} == {"module_body"}
        for row in module["source_imports"]["occurrences"]
    )
    dispatch = module["source_dispatch"]
    assert (
        dispatch["completion"] == "modeled_source_entry" and "first_gap" not in dispatch
    )
    assert {(x["owner"], x["site"]) for x in dispatch["required_imports"]} == {
        (r["owner"], r["site"]) for r in module["source_imports"]["occurrences"]
    }
    assert {x["symbol"] for x in dispatch["loaders"]} == {
        "py:first:<init>",
        "py:second:<init>",
    }
    [root] = dispatch["functions"]
    assert root["body"]["name"] == root["symbol"] == dispatch["entry"]
    ops = list(
        walk_operations(root["body"]["body"], exclude_fields=EXECUTION_ANNOTATIONS)
    )
    assert sum(op["op"] == "module_import_publish" for op in ops) == 2
    candidates = {
        c["symbol"]
        for op in ops
        for c in op.get("conditional_activation", {}).get("candidates", ())
    }
    assert (
        candidates
        == {f["symbol"] for f in dispatch["callees"]}
        == {"py:first:read", "py:second:read"}
    )
    assert all(
        f["completion"] == "conditional_source_body" for f in dispatch["callees"]
    )
    assert (
        "preserve_source_exceptions_and_unwind_at_each_operation"
        in dispatch["provider_obligations"]
    )
    fn = next(f for f in legacy["functions"] if f["name"] == "py:second:<init>")
    creation = next(n for n in fn["body"] if n["op"] == "python_define")
    del creation["source_definition"]
    module, _, issues = protocols.normalize_protocols(legacy)
    assert not issues
    alternative = module["source_dispatch"]
    assert alternative["completion"] == "source_prefix"
    assert alternative["first_gap"]["symbol"] == fn["name"]
    assert alternative["first_gap"]["location"]["line"] == 1
    loader = next(x for x in alternative["loaders"] if x["symbol"] == fn["name"])
    assert (
        loader["function"]["completion"] == "prefix" and not loader["function"]["body"]
    )

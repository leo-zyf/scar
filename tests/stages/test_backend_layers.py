"""Each backend pass must replay its sealed input with adjacent implementations absent."""

from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

import scar
from scar.contract import IR, document, file_receipt, member, write_json

_RUN_LAYER = """
import importlib, sys
from pathlib import Path

layer, source, output, *settings = sys.argv[1:]
implementations = {
    "B1": ("scar.backend.planning", "schedule_ir", ("scar.backend.cpp20", "scar.backend.build")),
    "B2a": ("scar.backend.cpp20", "lower_scheduled",
            ("scar.backend.planning", "scar.backend.cpp20.project", "scar.backend.cpp20.printer", "scar.backend.build")),
    "B2b": ("scar.backend.cpp20", "print_target",
            ("scar.backend.planning", "scar.backend.cpp20.lowering", "scar.backend.cpp20.objects",
             "scar.backend.cpp20.libraries", "scar.backend.cpp20.target", "scar.backend.build")),
    "B3": ("scar.backend.build", "build", ("scar.backend.planning", "scar.backend.cpp20")),
}
module, entry, forbidden = implementations[layer]
forbidden += ("scar.frontend", "scar.middle")
class Boundary:
    def find_spec(self, name, path=None, target=None):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden):
            raise RuntimeError("layer import violation " + name)
sys.meta_path.insert(0, Boundary())
implementation = importlib.import_module(module)
if layer == "B2a" and settings:
    from scar.backend.cpp20.lowering import Lowerer
    Lowerer.unit_weight_limit = int(settings[0])
if layer == "B2b":
    # A target package carries the exact SDK revision. Printing must also work
    # without consulting the SDK of the currently installed compiler.
    importlib.import_module("scar.backend.cpp20.project").__file__ = str(Path(output) / "absent-sdk" / "project.py")
run = getattr(implementation, entry)
if layer == "B3":
    run(source)
elif layer == "B1":
    from unittest.mock import patch
    from scar.backend.planning import cpu
    with patch.object(cpu, "_probe", wraps=cpu._probe) as probe:
        run(source, output, policy=settings[0] if settings else "serial",
            workers=int(settings[1]) if len(settings) > 1 else None)
    print(probe.call_count)
else:
    run(source, output)
"""


def test_native_execution_calls_require_bindings_without_blocking_local_names():
    from scar.backend.planning import _qualify, toolchain
    from scar.backend.planning.mpi.sdk import _interfaces

    historical = dict(
        functions=[dict(name="MPI_Finalize", body=[])],
        externs=[dict(name="MPI_Finalize", provider="mpi")],
        source_evidence=dict(
            old_call=dict(op="call", operator_family="mpi", mpi_interface="MPI_Init"),
            old_type=dict(kind="ndarray", element="f32", rank=1),
        ),
    )
    requirements, bindings, issues = _qualify(historical)
    assert requirements == bindings == {} and not issues

    declaration = dict(name="system", provider="libc", header="stdlib.h")
    module = dict(
        functions=[
            dict(
                name="main",
                body=[dict(op="call", callee="system", args=[], site="launch")],
            )
        ],
        externs=[declaration],
    )
    _, _, issues = _qualify(module)
    assert issues[0]["code"] == "missing_execution_boundary_binding"
    module["functions"].append(dict(name="system", body=[]))
    assert _qualify(module)[2] == []
    module["functions"].pop()
    module["functions"][0]["body"] = [dict(op="function_ref", name="system")]
    assert _qualify(module)[2] == []
    module["functions"][0]["body"].append(
        dict(op="indirect_call", targets=["system"], args=[])
    )
    assert _qualify(module)[2][0]["code"] == "missing_execution_boundary_binding"

    # Program/proof separation must also hold inside an actual function.
    ghost = dict(op="call", callee="system", args=[], site="quoted")
    module["functions"][0]["body"] = [dict(op="return", source_evidence=ghost)]
    assert _qualify(module)[2] == []
    module["functions"][0]["initializers"] = [ghost]
    assert _qualify(module)[2][0]["site"] == "quoted"

    mpi = dict(name="MPI_Finalize", result="i32", params=[], provider="mpi")
    direct = dict(
        op="call",
        callee="MPI_Finalize",
        operator_family="mpi",
        mpi_interface="MPI_Finalize",
    )
    indirect = dict(op="indirect_call", targets=["MPI_Finalize"])
    reference = dict(op="function_ref", name="MPI_Finalize")
    module = dict(
        externs=[mpi],
        functions=[
            dict(
                name="main",
                body=[
                    dict(
                        op="return",
                        source_evidence=[
                            direct,
                            indirect,
                            reference,
                            dict(op="call", callee="main"),
                        ],
                    )
                ],
            )
        ],
    )
    expected, uses, _ = _interfaces(module)
    assert (
        expected["operations"]
        == uses["called_or_possible"]
        == uses["address_taken"]
        == []
    )
    assert toolchain.compilation(module)["evidence"]["direct_internal_calls"] == 0
    module["functions"][0]["initializers"] = [direct]
    module["globals"] = [dict(init=reference)]
    module["records"] = [dict(fields=[dict(default=indirect)])]
    expected, uses, _ = _interfaces(module)
    assert expected["operations"] == ["call", "indirect_call"]
    assert uses["called_or_possible"] == uses["address_taken"] == ["MPI_Finalize"]
    direct["mpi_interface"] = "MPI_Init"
    with pytest.raises(ValueError, match="annotation disagrees"):
        _interfaces(module)


def test_system_runtime_binding_preserves_wait_status_and_keeps_descendants_open(
    tmp_path,
):
    from copy import deepcopy

    from scar.backend import compile_ir
    from scar.backend.cpp20.lowering import Lowerer
    from scar.backend.planning import _qualify
    from scar.contract import read_member, walk_operations
    from scar.frontend import build
    from scar.middle import plan

    source = tmp_path / "program.c"
    source.write_text(
        "#include <stdio.h>\n#include <stdlib.h>\n"
        'int main(void) { int status = system("exit 7"); '
        'printf("%d\\n", status); return 0; }\n'
    )
    baseline = tmp_path / "baseline"
    subprocess.run(["cc", "-std=c99", source, "-o", baseline], check=True)
    reference = subprocess.run([baseline], check=True, capture_output=True)
    wait_status = int(reference.stdout)
    assert os.WIFEXITED(wait_status) and os.WEXITSTATUS(wait_status) == 7
    manifest = tmp_path / "request.json"
    write_json(
        manifest,
        dict(
            source_root=str(tmp_path),
            entry="main",
            units=[dict(path=source.name, language="c", flags=["-std=c99"])],
        ),
    )
    graph, logical, delivery = (tmp_path / n for n in ("graph", "plan", "delivery"))
    build(manifest, graph)
    source.unlink()
    assert plan(graph, logical)["status"] == "READY"
    result = compile_ir(logical, delivery, policy="serial")
    assert result["status"] == "COMPILED", result
    actual = subprocess.run([result["executable"]], check=True, capture_output=True)
    assert (actual.stdout, actual.stderr) == (reference.stdout, reference.stderr)

    scheduled = json.loads((delivery / "scheduled-ir.json").read_bytes())
    physical = read_member(delivery, scheduled["physical"])
    binding = physical["operator_backends"]["posix_runtime"]
    (site,) = binding["sites"]
    proof = binding["sites"][site]
    assert (
        binding["qualification"] == "api_bound"
        and proof["descendant_closure"] == "open"
    )
    assert scheduled["runtime_boundaries"]["sites"] == [site]
    assert proof["expanded_argv"] == proof["tools"] == proof["cost"] == "unknown"
    assert binding["open_effects"] == dict(
        stage="external_effects",
        rule="posix.preserve_unknown_descendants",
        evidence=dict(
            external_effect="UNKNOWN",
            unknown_fields=["expanded_argv", "tools", "cost"],
            sites=[site],
        ),
    )
    evidence = binding["deployment"]
    assert evidence["probe"]["calls_system"] is False and evidence["libc"]["sha256"]
    assert evidence["source_header"]["sha256"] in {
        h["sha256"] for h in evidence["headers"]
    }
    assert evidence["shell"]["observation"] in {"present", "absent", "unobserved"}

    module = dict(
        scheduled["module"],
        functions=[read_member(delivery, f) for f in scheduled["functions"]],
    )

    def call(data):
        return next(
            n for n in walk_operations(data["functions"]) if n.get("callee") == "system"
        )

    node = call(module)
    assert node["external_effect"] == "UNKNOWN" and "ORDER" in node["effects"]
    assert node["runtime_boundary"]["command_request"] == proof["command_request"]
    assert any(
        "::system(" in path.read_text()
        for path in (delivery / "project/src").glob("*.cpp")
    )
    for missing in ("runtime_boundary", "artifact_effect", "writes"):
        bad = deepcopy(module)
        del call(bad)[missing]
        assert _qualify(bad)[2][0]["code"] == "missing_execution_boundary_binding"
    bad = deepcopy(module)
    call(bad)["runtime_boundary"]["command_request"] = "unrelated"
    assert _qualify(bad)[2][0]["code"] == "missing_execution_boundary_binding"
    call(bad)["args"] = [dict(op="literal", type="i32", value=0)]
    with pytest.raises(ValueError, match="qualified physical site"):
        Lowerer(bad, physical).expression(call(bad))


def test_unavailable_operator_bindings_stop_before_target_layers(tmp_path):
    def execution(folder, *, family=None, tagged=False, source_open=False):
        function = dict(
            name="main",
            params=[],
            result="i32",
            body=[dict(op="return", value=dict(op="literal", type="i32", value=0))],
        )
        if tagged:
            function["body"].insert(
                0,
                dict(
                    op="expr",
                    value=dict(
                        op="future_operator",
                        operator_family="future",
                        type="unit",
                        effects=["STATE"],
                    ),
                ),
            )
        if source_open:
            function["body"].insert(
                0,
                dict(
                    op="expr",
                    value=dict(
                        op="array_source_open",
                        type=dict(kind="handle", resource="text_stream"),
                        source=dict(op="literal", type="text", value="input.txt"),
                        resolution=dict(profile="numpy-2.5.5-datasource-v1"),
                        effects=["IO", "STATE", "MEM", "XFER"],
                    ),
                ),
            )
        module = dict(entry="main", externs=[], globals=[], records=[])
        if family:
            module["operator_requirements"] = {family: dict(profile="future.v1")}
        path = folder / "exec-ir.json"
        write_json(
            path,
            document(
                IR,
                status="READY",
                module=module,
                functions=[member(folder, "functions", function)],
            ),
        )
        return path

    required = execution(tmp_path / "required", family="future")
    tagged = execution(tmp_path / "tagged", tagged=True)
    unbound_source = execution(tmp_path / "unbound-source", source_open=True)
    bundle = tmp_path / "bundle"
    plain_image = execution(bundle / "images/driver")
    required_image = execution(bundle / "images/solver", family="future")
    write_json(
        bundle / "exec-ir.json",
        document(
            IR,
            status="READY",
            kind="image_bundle",
            root_image="driver",
            images=[
                dict(
                    id=identity,
                    entry="main",
                    output=identity,
                    ir=file_receipt(bundle, source),
                )
                for identity, source in (
                    ("driver", plain_image),
                    ("solver", required_image),
                )
            ],
        ),
    )
    command = """
import sys
class Boundary:
    def find_spec(self, name, path=None, target=None):
        if any(name == p or name.startswith(p + '.') for p in (
            'scar.frontend', 'scar.middle', 'scar.backend.cpp20', 'scar.backend.build')):
            raise RuntimeError('unavailable target layer was entered: ' + name)
sys.meta_path.insert(0, Boundary())
from scar.cli import main
raise SystemExit(main(['compile', '--ir', sys.argv[1], '--out', sys.argv[2], '--physical', 'serial']))
"""
    for index, source in enumerate(
        (required, tagged, bundle / "exec-ir.json", unbound_source)
    ):
        delivery = tmp_path / f"delivery-{index}"
        result = subprocess.run(
            [sys.executable, "-B", "-c", command, str(source), str(delivery)],
            env=dict(
                os.environ,
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONPATH=str(Path(scar.__file__).resolve().parents[1]),
            ),
            capture_output=True,
            timeout=60,
        )
        assert result.returncode == 2, result.stderr.decode()
        report = json.loads(result.stdout)
        assert report["status"] == "INCOMPLETE"
        assert all(i["code"] == "missing_operator_binding" for i in report["issues"])
        assert {i["requirement"] for i in report["issues"]} == {
            "text_array_source" if index == 3 else "future"
        }
        assert not (delivery / "project").exists()
        assert not (delivery / "target-ir.json").exists()
        physical = json.loads((delivery / "physical-plan.json").read_bytes())
        scheduled = json.loads((delivery / "scheduled-ir.json").read_bytes())
        assert physical["status"] == scheduled["status"] == "INCOMPLETE"
        if index == 2:
            assert {i["image_id"] for i in report["issues"]} == {"solver"}

    # When the application entry is blocked, a different qualified program
    # in the same IR is still deliverable through the ordinary compile command.
    from scar.backend import compile_ir

    package = json.loads((bundle / "exec-ir.json").read_bytes())
    package["root_image"] = "solver"
    write_json(bundle / "exec-ir.json", package)
    delivery = tmp_path / "components-release"
    report = compile_ir(bundle, delivery, policy="serial")
    assert report["status"] == "INCOMPLETE" and "executable" not in report
    component = report["component_delivery"]
    assert set(component["executables"]) == {"driver"}
    binary = component["executables"]["driver"]
    published = file_receipt(delivery, delivery / binary["path"])
    assert (published["sha256"], published["size"]) == (
        binary["sha256"],
        binary["size"],
    )
    subprocess.run([delivery / binary["path"]], check=True)
    assert not (delivery / "project").exists()


def test_component_schedule_excludes_dependents_of_incomplete_images(tmp_path):
    from scar.backend.planning import _closed_component_schedule
    from scar.contract import read_member

    edges = {
        "root": [],
        "broken": ["root"],
        "caller": ["broken"],
        "a": ["b"],
        "b": ["a"],
        "alone": [],
    }
    rows = []
    for name, dependencies in edges.items():
        plan = {
            "operator_backends": {"process": {"images": dict.fromkeys(dependencies)}}
        }
        rows.append(
            dict(
                id=name,
                entry="main",
                output=name,
                status="INCOMPLETE" if name == "root" else "READY",
                physical=member(tmp_path, "units", plan),
            )
        )
    receipt = _closed_component_schedule(
        tmp_path, "root", "source-revision", rows, rows
    )
    projected = read_member(tmp_path, receipt)
    assert [row["id"] for row in projected["images"]] == ["a", "b", "alone"]
    assert projected["application_root"] == "root"


def test_file_graph_and_workflow_members_survive_schedule_relocation(
    tmp_path, monkeypatch
):
    from scar.backend.planning import schedule_ir
    from scar.contract import ARTIFACT_FLOW, json_bytes, read_blob, read_member
    from scar.middle.execution import seal
    from scar.middle.contracts import publish_program

    source, schedule = tmp_path / "execution", tmp_path / "schedule"
    images, expected = [], {}
    for image in ("controller", "solver"):
        folder = source / "images" / image
        flow = document(
            ARTIFACT_FLOW, nodes=[dict(kind="FILE_LOCATOR", id=image)], edges=[]
        )
        function = dict(
            name="main",
            params=[],
            result="i32",
            body=[dict(op="return", value=dict(op="literal", type="i32", value=0))],
        )
        module = dict(
            entry="main",
            functions=[function],
            globals=[],
            externs=[],
            records=[],
            resource_model={"artifact_flow": flow},
        )
        module = publish_program(module, "M4", complete=True)
        args = dict(graph_sha256="f" * 64, contract={})
        ir = seal(module, folder, status="READY", **args)
        receipt = member(folder, "analysis", flow)
        assert (
            json.loads(ir.read_bytes())["module"]["resource_model"]["artifact_flow"]
            == receipt
        )
        images.append(
            dict(id=image, entry="main", output=image, ir=file_receipt(source, ir))
        )
        expected[image] = json_bytes(flow)

    incomplete = seal(module, tmp_path / "partial", status="INCOMPLETE", **args)
    partial = json.loads(incomplete.read_bytes())
    assert partial["functions"] == [] and set(partial["module"]) == {
        "entry",
        "resource_model",
    }
    assert (
        read_blob(
            incomplete.parent, partial["module"]["resource_model"]["artifact_flow"]
        )
        == expected["solver"]
    )
    workflow = document(
        "scar.workflow-graph",
        root_image="controller",
        images=["controller", "solver"],
        nodes=[],
        edges=[],
    )
    workflow_receipt = member(source, "analysis", workflow)
    write_json(
        source / "exec-ir.json",
        document(
            IR,
            kind="image_bundle",
            status="READY",
            root_image="controller",
            images=images,
            workflow=workflow_receipt,
        ),
    )
    moved = tmp_path / "moved-execution"
    source.rename(moved)
    workflow_file = moved / workflow_receipt["path"]
    workflow_bytes = workflow_file.read_bytes()
    workflow_file.write_bytes(b" " + workflow_bytes[1:])
    with pytest.raises(ValueError, match="checksum mismatch"):
        schedule_ir(moved, tmp_path / "corrupt-input", policy="serial")
    assert not (tmp_path / "corrupt-input" / "scheduled-ir.json").exists()
    workflow_file.write_bytes(workflow_bytes)
    loads = json.loads
    audit_members = {*expected.values(), json_bytes(workflow)}

    def reject_audit_decode(data, *a, **kw):
        raw = data.encode() if isinstance(data, str) else data
        assert raw not in audit_members, "B1 decoded an audit-only graph"
        return loads(data, *a, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(json, "loads", reject_audit_decode)
        schedule_ir(moved / "exec-ir.json", schedule, policy="serial")
    moved.rename(tmp_path / "hidden-execution")
    delivery = tmp_path / "moved-schedule"
    schedule.rename(delivery)
    sealed = json.loads((delivery / "scheduled-ir.json").read_bytes())
    assert sealed["status"] == "READY" and read_blob(
        delivery, sealed["workflow"]
    ) == json_bytes(workflow)
    for image in sealed["images"]:
        child = read_member(delivery, image["scheduled"])
        child_folder = (delivery / image["scheduled"]["path"]).parent
        assert (
            read_blob(child_folder, child["module"]["resource_model"]["artifact_flow"])
            == expected[image["id"]]
        )


def test_binary_array_adapter_matches_the_declared_entry():
    from scar.backend.cpp20.lowering import Lowerer

    read = {"kind": "ptr", "to": {"kind": "const", "of": "f64"}}
    write = {"kind": "ptr", "to": "f64"}
    params = [
        dict(name="size", type="i64"),
        dict(name="count", type="i64"),
        dict(name="input", type=read),
        dict(name="output", type=write),
    ]
    fn = dict(name="evaluate", result="void", params=params, body=[])
    abi = dict(kind="binary_f64_arrays", shape=[5], count=2, inputs=1)
    module = dict(entry="evaluate", functions=[fn], entry_abi=abi)
    wrapper = Lowerer(module, {}).entry_wrapper()
    abi["entry"] = "evaluate"
    assert Lowerer(module, {}).entry_wrapper() == wrapper
    abi["entry"] = "different"
    with pytest.raises(
        ValueError, match="array ABI names a different executable entry"
    ):
        Lowerer(module, {}).entry_wrapper()


def test_backend_products_replay_and_reject_changed_bytes(tmp_path):
    from dataclasses import replace

    from scar.backend.cpp20.model import (
        Attribute,
        Decl,
        Expr,
        Stmt,
        Type,
        Unit,
        call,
        encode_unit,
        named,
        to_data,
        unit_from_data,
    )
    from scar.contract import read_member

    for declaration, expected in (
        ({"node": "Decl", "name": "f"}, "missing required field in Decl"),
        ({"node": "Decl", "kind": "function", "typo": 0}, "unknown fields in Decl"),
    ):
        with pytest.raises(ValueError, match=expected):
            unit_from_data({"node": "Unit", "declarations": [declaration]})

    def run(layer, source, out, expected=None):
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(Path(scar.__file__).resolve().parents[1]),
        )
        if layer in {"B2b", "B3"}:
            env.update(SCAR_CPU_SDK="/absent-sdk", CMAKE_PREFIX_PATH="/absent-sdk")
        process = subprocess.run(
            [sys.executable, "-B", "-c", _RUN_LAYER, layer, str(source), str(out)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=60,
        )
        error = process.stderr.decode()
        if expected is None:
            assert process.returncode == 0, error
        else:
            assert process.returncode != 0 and expected in error, error
            assert "layer import violation" not in error

    incoming = tmp_path / "execution"
    fn = dict(
        name="main",
        params=[],
        result="i32",
        body=[
            dict(
                op="decl",
                name="probe",
                type=dict(kind="ndarray", element="f64", rank=1),
                init=dict(
                    op="array_literal",
                    type=dict(kind="ndarray", element="f64", rank=1),
                    shape=[dict(op="literal", type="i64", value=1)],
                    values=[dict(op="literal", type="f64", value=2.5)],
                    effects=["MEM", "VAL"],
                ),
            ),
            dict(
                op="expr",
                value=dict(
                    op="call",
                    callee="puts",
                    args=[dict(op="string", value="sealed project")],
                ),
            ),
            dict(
                op="expr",
                value=dict(
                    op="call", callee="puts", args=[dict(op="string", value="ordinary")]
                ),
            ),
            dict(op="return", value=dict(op="literal", type="i32", value=0)),
        ],
    )
    declaration = dict(
        name="puts",
        provider="libc",
        header="stdio.h",
        result="i32",
        params=[
            dict(name="text", type=dict(kind="ptr", to=dict(kind="const", of="char")))
        ],
    )
    ir = incoming / "exec-ir.json"
    write_json(
        ir,
        document(
            IR,
            status="READY",
            module=dict(entry="main", externs=[declaration]),
            functions=[member(incoming, "functions", fn)],
        ),
    )
    b1, b2a, b2b = (
        tmp_path / name for name in ("schedule", "target", "source-project")
    )
    run("B1", ir, b1)
    incoming.rename(tmp_path / "hidden-physical-input")
    run("B2a", b1 / "scheduled-ir.json", b2a)
    b1.rename(tmp_path / "hidden-schedule")
    target = json.loads((b2a / "target-ir.json").read_bytes())
    # Exercise both target spellings of the same sealed libc binding.
    alias = Decl(
        "function",
        name="line_writer",
        type=named("int"),
        linkage="C",
        params=(
            Decl(
                "param", type=Type("ptr", element=Type("const", element=named("char")))
            ),
        ),
        qualifiers=("noexcept(false)",),
        asm_label="puts",
    )
    for invalid in (
        replace(alias, asm_label='puts");'),
        replace(alias, asm_label="std::puts"),
        replace(alias, linkage=None),
        replace(alias, body=()),
        replace(alias, definition="delete"),
        replace(alias, storage=("static",)),
        replace(alias, qualifiers=("const",)),
        replace(
            alias,
            attributes=(
                Attribute("__gnu__::__alias__", (Expr("string", value="local"),)),
            ),
        ),
        Decl("variable", name="data", type=named("int"), asm_label="puts"),
        Decl("record", name="Holder", members=(alias,)),
        Decl(
            "function",
            name="outer",
            type=named("void"),
            body=(Stmt("declaration", decl=alias),),
        ),
    ):
        with pytest.raises(ValueError):
            encode_unit(Unit(declarations=(invalid,)))

    text = Expr("string", value="sealed project")
    old_call, new_call = (
        to_data(call(symbol, text)) for symbol in ("::puts", alias.name)
    )

    def bind_call(node):
        if node == old_call:
            return new_call
        if isinstance(node, dict):
            return {key: bind_call(value) for key, value in node.items()}
        if isinstance(node, list):
            return [bind_call(value) for value in node]
        return node

    for part in target["units"]:
        unit = unit_from_data(bind_call(read_member(b2a, part["tree"])))
        if part["output"] == "include/scar/program.hpp":
            unit = replace(unit, declarations=(*unit.declarations, alias))
        part["tree"] = member(b2a, "target-units", encode_unit(unit))
    sealed = b2a / "alias-target.json"
    write_json(sealed, target)
    run("B2b", sealed, b2b)
    cpp = b2b / "project"
    header = "".join((cpp / "include/scar/program.hpp").read_text().split())
    assert 'noexcept(false)asm("puts");' in header
    code = (cpp / "src/program.cpp").read_text()
    assert 'line_writer("sealed project")' in code and '::puts("ordinary")' in code
    assert target["operator_sdks"]["XTensor"]["provider"] == "xtensor-cpu"
    cmake = (b2b / "project/CMakeLists.txt").read_text()
    assert "vendor/xtensor-cpu/include" in cmake
    assert "SCAR_CPU_SDK" not in cmake and "find_package(xtensor" not in cmake
    source_tree = b2a / target["units"][0]["tree"]["path"]

    wrong_arity = Expr("binary", operator="+", args=(Expr("literal", value=1),))
    invalid = Unit(
        declarations=(
            Decl(
                "function",
                name="main",
                type=named("int"),
                body=(Stmt("return", value=wrong_arity),),
            ),
        )
    )
    expected = "Expr.binary.args: invalid arity"
    with pytest.raises(ValueError, match=r"Expr\.binary\.args: invalid arity"):
        encode_unit(invalid)
    # A valid checksum must not allow malformed target syntax across the B2 boundary.
    rejected = dict(
        target["units"][0], tree=member(b2a, "target-units", to_data(invalid))
    )
    corrupt = b2a / "invalid-target.json"
    write_json(corrupt, dict(target, units=[rejected, *target["units"][1:]]))
    run("B2b", corrupt, tmp_path / "invalid-tree", expected)
    assert not (tmp_path / "invalid-tree/project").exists()
    source_tree.write_bytes(source_tree.read_bytes() + b" ")
    run("B2b", sealed, tmp_path / "changed-tree", "checksum mismatch")
    assert not (tmp_path / "changed-tree/project").exists()
    b2a.rename(tmp_path / "hidden-target")
    run("B3", b2b / "cpp20.json", b2b)
    native = b2b / "project/bin/scar_entry"
    assert subprocess.check_output([native]) == b"sealed project\nordinary\n"
    success = json.loads((b2b / "build.json").read_bytes())
    assert success["stage"] == "B3" and success["warnings"] == 0
    assert 1 <= success["build_workers"] <= success["execution_host"]["worker_budget"]
    generated = b2b / "project/src/program.cpp"
    generated.write_bytes(generated.read_bytes() + b"\n")
    run("B3", b2b / "cpp20.json", b2b, "project member changed")


def test_cpu_budget_is_shared_with_graph_planner_and_formatter(tmp_path, monkeypatch):
    from math import isclose

    import scar.host as host
    from scar.backend.cpp20.project import _format_workers
    from scar.backend.planning.cpu import schedule
    from scar.frontend.hardware import snapshot

    monkeypatch.setattr(host.os, "sched_getaffinity", lambda _: set(range(8)))
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("OMP_THREAD_LIMIT", "8")
    mount = tmp_path / "cpu space"
    escaped = str(mount).replace(" ", r"\040")
    cases = [
        (
            "0::/tenant/job/leaf\n",
            f"31 20 0:27 /tenant {escaped} rw - cgroup2 cgroup rw\n",
            {
                mount / "job/leaf/cpu.max": "350000 100000",
                mount / "job/cpu.max": "250000 100000",
                mount / "cpu.max": "max 100000",
            },
            2.5,
            2,
        ),
        (
            "7:cpu,cpuacct:/job/leaf\n",
            f"31 20 0:27 / {escaped} rw - cgroup cgroup rw,cpu,cpuacct\n",
            {
                mount / "job/leaf/cpu.cfs_quota_us": "-1",
                mount / "job/leaf/cpu.cfs_period_us": "100000",
                mount / "job/cpu.cfs_quota_us": "399999",
                mount / "job/cpu.cfs_period_us": "100000",
            },
            3.99999,
            3,
        ),
    ]
    for groups, mounts, limits, capacity, budget in cases:
        files = {
            Path("/proc/self/cgroup"): groups,
            Path("/proc/self/mountinfo"): mounts,
            **limits,
        }
        monkeypatch.setattr(host, "_read", lambda path: files.get(Path(path)))
        observed = host.cpu_availability()
        assert not observed.issues and observed.worker_budget == budget
        assert isclose(observed.as_record()["capacity_upper_bound_cores"], capacity)
        resources, edges, platform = snapshot(tmp_path)
        (pool,) = [r for r in resources if r.kind == "CPU_POOL"]
        assert pool.capacity == budget
        assert {
            e.target
            for e in edges
            if e.source == pool.id and e.value == "CPU_POOL_MEMBER"
        } == {f"cpu:{i}" for i in range(8)}
        assert platform["cpu_availability"] == observed.as_record()
        _, plan = schedule(dict(functions=[]), "auto", worker_limit=5)
        assert plan["deployment_cpu"] == observed.as_record()
        assert plan["worker_limit"] == budget and plan["workers"] == 1
        assert _format_workers(4, 36) == budget
    files[Path("/proc/self/mountinfo")] = "unreadable mount record"
    unknown = host.cpu_availability()
    assert unknown.issues and unknown.worker_budget == 1
    assert _format_workers(4, 36) == 1


def test_cpu_schedules_nested_control_regions_without_emitting_both_choices():
    from scar.backend.planning.cpu import schedule
    from scar.contract import walk_operations

    extent = dict(op="literal", type="i64", value=4)
    stream = dict(kind="handle", resource="text_stream")

    def scratch(name):
        return dict(
            op="runtime_array",
            name=name,
            element="f64",
            extent=extent,
            storage="scoped",
            initialization="uninitialized",
            scope_control="lexical_continuation",
        )

    def fill(site, value):
        target = dict(
            op="index",
            base=dict(op="var", name="caught_array"),
            indices=[dict(op="var", name="i")],
        )
        return dict(
            op="map",
            index="i",
            site=site,
            independent=True,
            extent=extent,
            body=[
                dict(
                    op="assign",
                    target=target,
                    value=dict(op="literal", type="f64", value=value),
                )
            ],
        )

    first, second, fused = fill("first", 1.0), fill("second", 2.0), fill("fused", 1.0)
    fused["body"] += second["body"]
    alternative = dict(
        op="schedule_choice",
        site="alternative",
        reference=[first, second],
        fused=fused,
        guard="masked_floating_exceptions",
    )
    step = dict(op="unary", operator="preinc", value=dict(op="var", name="j"))
    loop = dict(
        op="for",
        init=[
            dict(op="block", body=[scratch("setup")]),
            dict(op="decl", name="j", type="i64", init=extent),
        ],
        condition=dict(op="literal", type="bool", value=False),
        step=step,
        body=[],
    )
    protected = {
        "op": "try",
        "body": [
            dict(
                op="resource_scope",
                resource=dict(op="var", name="stream", type=stream),
                body=[scratch("inside_resource")],
            )
        ],
        "handlers": [
            dict(
                categories=["exception"],
                binding=None,
                body=[scratch("caught_array"), alternative],
            )
        ],
        "else": [scratch("success"), loop],
        "finally": [scratch("cleanup")],
    }
    fn = dict(
        name="work",
        params=[dict(name="stream", type=stream)],
        result="void",
        body=[protected],
    )
    selected, physical = schedule(dict(functions=[fn]), "serial")
    nodes = list(walk_operations(selected["functions"][0]["body"]))
    arrays = {
        n["name"]: n.get("allocation") for n in nodes if n["op"] == "runtime_array"
    }
    assert arrays == dict.fromkeys(
        ("inside_resource", "caught_array", "success", "setup", "cleanup"),
        "scoped_stack",
    )
    assert {a["name"] for a in physical["allocations"]} == set(arrays)
    assert next(n for n in nodes if n["op"] == "for")["step"] == step
    assert [c["chosen"] for c in physical["choices"]] == ["reference"]
    assert (
        {n["site"] for n in nodes if n["op"] == "map"}
        == set(physical["regions"])
        == {"first", "second"}
    )
    assert all(
        r["work"]["priced"] and r["work"]["bytes_per_element"] == 8
        for r in physical["regions"].values()
    )


def test_cpu_costs_price_the_traversal_actually_emitted():
    from math import isclose
    from statistics import median

    from scar.backend.planning.cpu import schedule

    def var(name):
        return dict(op="var", name=name)

    def binary(operator, left, right):
        return dict(op="binary", operator=operator, left=left, right=right)

    def cell(name, position):
        return dict(op="index", base=var(name), indices=[position])

    i, x, y = map(var, ("i", "x", "y"))
    column = binary("mod", i, x)
    row = binary("mod", binary("div", i, x), y)
    offset = binary(
        "add", column, binary("mul", dict(op="literal", type="i64", value=7), row)
    )
    value = binary("div", cell("input", offset), var("scale"))
    prefix = [0, "value", "left", "indices", 0]
    layout = dict(
        kind="mixed_radix",
        order="first_axis_fastest",
        axes=[x, y],
        fallback="linear",
        equalities=[],
        coordinates=[
            dict(axis=0, path=[*prefix, "left"]),
            dict(axis=1, path=[*prefix, "right", "right"]),
        ],
    )
    loop = dict(
        op="map",
        site="view",
        independent=True,
        index="i",
        extent=binary("mul", x, y),
        body=[dict(op="assign", target=cell("output", i), value=value)],
        iteration_layout=layout,
    )
    params = [dict(name=n, type="i64") for n in ("x", "y")]
    params += [
        dict(name=n, type=dict(kind="ptr", to="f64")) for n in ("input", "output")
    ]
    params.append(dict(name="scale", type="f64"))
    opaque = dict(
        op="map",
        site="external",
        independent=True,
        index="i",
        extent=x,
        body=[dict(op="expr", value=dict(op="call", callee="putchar", args=[i]))],
    )
    module = dict(
        functions=[dict(name="view", params=params, result="void", body=[loop, opaque])]
    )
    _, physical = schedule(module, "auto", worker_limit=2)

    region = physical["regions"]["view"]
    assert region["work"]["operations"] == {
        "arith_i64": 2,
        "div_f64": 1,
        "mod_i64": 1,
        "divmod_i64": 1,
    }
    assert region["coordinate_work"]["operations"] == {"arith_i64": 2, "div_f64": 1}
    assert (
        region["work"]["bytes_per_element"]
        == region["coordinate_work"]["bytes_per_element"]
        == 16
    )
    assert physical["regions"]["external"]["widths"] == [
        dict(minimum_elements=0, workers=1)
    ]
    assert physical["regions"]["external"]["fallback"]

    samples = physical["calibration"]["samples"]
    candidates = {(p["workers"], p["layout"]): p for p in region["candidate_costs"]}
    assert set(candidates) == {(w, "linear") for w in samples} | {(1, "mixed_radix")}
    for (workers, traversal), prediction in candidates.items():
        estimates = []
        for sample in samples[workers]:
            compute = 2 * sample["arith_i64"] + sample["div_f64"]
            if traversal == "linear":
                compute += sample["mod_i64"] + sample["divmod_i64"]
            else:
                compute += region["carry_steps_per_element_bound"] * sample["carry_i64"]
            estimates.append(max(16 * sample["byte"], compute))
        assert isclose(
            prediction["ns_per_element"], (median if workers == 1 else max)(estimates)
        )


def test_copy_costs_share_one_calibration_and_preserve_runtime_binding(tmp_path):
    from scar.middle.execution import prepare, seal
    from scar.middle.contracts import publish_program

    array = dict(kind="ndarray", element="f64", rank=1)
    ptr = dict(kind="ptr", to="u8")
    count = dict(op="var", name="n", type="i64")
    create = dict(op="array_create", type=array, shape=[count], fill="zeros")
    fixed = dict(
        op="call",
        callee="memcpy",
        args=[dict(op="var", name=n, type=ptr) for n in ("dst", "src")]
        + [dict(op="literal", type="i64", value=4096)],
    )
    dynamic = dict(op="array_copy", type=array, value=create)
    broadcast = dict(op="array_broadcast_copy", type=array, value=create, shape=[count])
    filesystem = dict(
        op="host_call",
        interface="fs.copy2",
        args=[dict(op="literal", type="text", value=p) for p in ("in", "out")],
    )
    loop = dict(
        op="map", site="map", independent=True, index="i", extent=count, body=[]
    )
    resources = [
        dict(id="workers", kind="CPU_POOL"),
        dict(id="numa:0", kind="NUMA_MEMORY"),
        dict(id="mount:7", kind="STORAGE_MOUNT", attrs=dict(filesystem_type="tmpfs")),
        dict(id="disk:0", kind="NVME"),
    ]

    def execution(name, body):
        fn = dict(
            name="run",
            params=[
                dict(name="n", type="i64"),
                *(dict(name=n, type=ptr) for n in ("src", "dst")),
            ],
            result="void",
            body=body,
        )
        module = dict(
            entry="run",
            functions=[fn],
            globals=[],
            externs=[dict(name="memcpy", provider="libc")],
            records=[],
            resource_model=dict(resources=resources),
        )
        prepared, _ = prepare(publish_program(module, "M3", complete=True))
        return seal(
            prepared,
            tmp_path / name,
            status="READY",
            graph_sha256="0" * 64,
            contract={},
        )

    def measure(source, policy):
        output = source.parent / policy
        calls = subprocess.check_output(
            [sys.executable, "-B", "-c", _RUN_LAYER, "B1", source, output, policy, "2"],
            env=dict(
                os.environ, PYTHONPATH=str(Path(scar.__file__).resolve().parents[1])
            ),
            cwd=tmp_path,
            timeout=60,
        )
        return int(calls), json.loads((output / "physical-plan.json").read_bytes())

    source = execution(
        "all",
        [
            *(
                dict(op="expr", value=node)
                for node in (fixed, dynamic, broadcast, filesystem)
            ),
            loop,
        ],
    )
    calls, physical = measure(source, "auto")
    measured = physical["calibration"]
    metric = "memcpy_payload_byte"
    assert calls == 1 and set(measured["metrics"]) == {"team", "byte", metric}
    assert measured["metric_widths"][metric] == [1]
    assert measured["sampled_widths"] == list(range(1, physical["worker_limit"] + 1))
    assert {c["workers"] for c in physical["regions"]["map"]["candidate_costs"]} == set(
        measured["sampled_widths"]
    )
    plan = physical["movement_plan"]
    coefficient = plan["coefficients"][metric]
    assert coefficient["samples"] == [row[metric] for row in measured["samples"]["1"]]
    assert (
        len(coefficient["samples"]) == 3
        and 0
        < coefficient["minimum"]
        <= coefficient["median"]
        <= coefficient["maximum"]
    )
    fixed, dynamic, broadcast, filesystem_copy = plan["copies"]
    assert all(
        c["dispatch"]["resource_candidates"] == ["workers"] for c in plan["copies"]
    )
    assert fixed["dispatch"]["binding"] == "caller_thread"
    assert (
        fixed["cost"]["copy_component"]["estimated_ns"] == 4096 * coefficient["median"]
    )
    assert (
        dynamic["cost"]["copy_component"]["expression"]["left"]["operation"]
        == dynamic["id"]
    )
    assert dynamic["cost"]["copy_component"]["estimated_ns"] is None
    assert all(c["cost"]["total_duration_ns"] is None for c in plan["copies"])
    assert (
        broadcast["cost"]["copy_component"]
        is filesystem_copy["cost"]["copy_component"]
        is None
    )
    assert (
        plan["materializations"][dynamic["target"]]["location"]["binding"]
        == "runtime_allocator"
    )
    location = plan["materializations"][filesystem_copy["target"]]["location"]
    assert (
        location["namespace"] == "source_inventory"
        and location["binding"] == "execution_namespace"
    )
    assert location["resource_candidates"] == ["mount:7"]
    assert (
        filesystem_copy["ordering"]["ready_on_success"]["operation"]
        == filesystem_copy["id"]
    )
    calls, serial = measure(source, "serial")
    assert calls == 1 and serial["worker_limit"] == 1
    assert serial["calibration"]["metrics"] == measured["metrics"]
    files_only = execution("files", [dict(op="expr", value=filesystem)])
    assert measure(files_only, "auto")[0] == 0


def test_product_identity_pool_and_shared_arrays_across_target_units(tmp_path):
    def lit(value, ty):
        return dict(op="literal", type=ty, value=value)

    def var(name, ty):
        return dict(op="var", name=name, type=ty)

    def make(*values, slot=None):
        identity = (
            dict(kind="fresh")
            if slot is None
            else dict(kind="constant_pool", pool="module-code", slot=slot)
        )
        return dict(
            op="product_make",
            type=dict(kind="product_ref", items=[v["type"] for v in values]),
            items=list(values),
            identity=identity,
            effects=["MEM", "STATE"],
            may_raise=True,
        )

    def field(value, i):
        return dict(
            op="product_get",
            type=value["type"]["items"][i],
            base=value,
            index=i,
            effects=["STATE"],
        )

    def identical(left, right, operator="eq"):
        return dict(
            op="product_identity",
            type="bool",
            left=left,
            right=right,
            operator=operator,
            effects=["STATE"],
        )

    def invoke(name, result, *args):
        return dict(op="call", callee=name, type=result, args=list(args))

    def decl(name, init):
        return dict(op="decl", name=name, type=init["type"], init=init)

    def returned(name, value, params=()):
        return dict(
            name=name,
            result=value["type"],
            params=[dict(name=n, type=t) for n, t in params],
            body=[dict(op="return", value=value)],
        )

    arr = dict(kind="ndarray", element="f64", rank=1)
    x = var("x", arr)
    pooled = make(make(slot=0), lit("abc", "text"), slot=1)
    other_slot = {**pooled, "identity": {**pooled["identity"], "slot": 2}}
    p = pooled["type"]
    fields = make(x, x, pooled)
    box = var("box", fields["type"])
    alias = var("alias", fields["type"])
    position = [lit(0, "i64")]
    updated = lit(9.0, "f64")
    load = dict(
        op="array_load",
        type="f64",
        base=field(alias, 1),
        indices=position,
        effects=["STATE"],
    )
    body = [
        decl(
            "x",
            dict(
                op="array_literal",
                type=arr,
                shape=[lit(1, "i64")],
                values=[lit(0.0, "f64")],
                effects=["MEM", "VAL"],
            ),
        ),
        decl("box", fields),
        decl("alias", invoke("same_handle", box["type"], box)),
        dict(
            op="array_store",
            base=field(box, 0),
            indices=position,
            value=updated,
            broadcasting="trailing",
            effects=["STATE"],
        ),
    ]
    numbers = dict(kind="sequence_ref", element="i64")
    strings = dict(kind="sequence_ref", element="text")
    seq, ordered = var("seq", numbers), var("ordered", numbers)

    def sequence(action, result, **fields):
        effects = {
            "make": ["MEM", "STATE"],
            "append": ["MEM", "STATE"],
            "size": ["STATE"],
            "get": ["STATE"],
            "sorted_copy": ["MEM", "STATE", "VAL"],
            "identity": ["STATE"],
        }[action]
        return dict(
            op="sequence_" + action,
            type=result,
            effects=effects,
            **({"may_raise": True} if action not in {"size", "identity"} else {}),
            **fields,
        )

    def get(base, index):
        return sequence(
            "get",
            base["type"]["element"],
            base=base,
            index=index,
            source_index_profile="python_list_v1",
        )

    def eq(left, right):
        return dict(op="binary", operator="eq", left=left, right=right)

    incoming = dict(
        op="host_call",
        interface="console.input",
        args=[lit("", "text"), lit("surrogateescape", "text")],
        type="text",
        effects=["IO", "STATE"],
        semantics_version=1,
    )
    byte, words = var("byte", "text"), var("words", strings)
    body.extend(
        [
            decl(
                "seq", sequence("make", numbers, items=[lit(i, "i64") for i in (3, 1)])
            ),
            decl("shared", invoke("same_sequence", numbers, seq)),
            decl("appended", invoke("append_item", "unit", var("shared", numbers))),
            decl(
                "ordered",
                sequence("sorted_copy", numbers, base=seq, comparison="signed_integer"),
            ),
            decl("byte", incoming),
            decl(
                "words",
                sequence(
                    "sorted_copy",
                    strings,
                    base=sequence(
                        "make",
                        strings,
                        items=[lit("\ue000", "text"), byte, lit("é", "text")],
                    ),
                    comparison="unicode_codepoint",
                ),
            ),
        ]
    )
    checks = [
        identical(box, alias),
        identical(box, fields, "ne"),
        identical(invoke("pool_a", p), invoke("pool_b", p)),
        identical(invoke("pool_a", p), other_slot, "ne"),
        identical(field(box, 2), box, "ne"),
        invoke("is_pooled", "bool", field(box, 2)),
        dict(op="binary", operator="eq", left=load, right=updated),
        sequence(
            "identity", "bool", left=seq, right=var("shared", numbers), operator="eq"
        ),
        sequence("identity", "bool", left=seq, right=ordered, operator="ne"),
        sequence("identity", "bool", left=seq, right=words, operator="ne"),
        eq(var("appended", "unit"), lit(None, "unit")),
        eq(sequence("size", "i64", base=seq), lit(3, "i64")),
        eq(get(seq, lit(0, "i64")), lit(3, "i64")),
        eq(get(seq, lit(-1, "i64")), lit(2, "i64")),
        eq(get(ordered, lit(0, "i64")), lit(1, "i64")),
        eq(get(words, lit(1, "i64")), byte),
        invoke("index_fails", "bool", seq, lit(3, "i64")),
        invoke("index_fails", "bool", seq, lit(-4, "i64")),
    ]
    for code, condition in enumerate(checks, 1):
        body.append(
            dict(
                op="if",
                condition=dict(op="unary", operator="not", value=condition),
                then=[dict(op="return", value=lit(code, "i32"))],
                **{"else": []},
            )
        )
    body.append(dict(op="return", value=lit(0, "i32")))
    functions = [
        dict(name="main", result="i32", params=[], body=body),
        returned("pool_a", dict(pooled, site="first")),
        returned("pool_b", dict(pooled, site="second")),
        returned("same_handle", box, [("box", box["type"])]),
        returned("is_pooled", identical(var("p", p), pooled), [("p", p)]),
        returned("same_sequence", seq, [("seq", numbers)]),
        returned(
            "append_item",
            sequence("append", "unit", base=seq, value=lit(2, "i64")),
            [("seq", numbers)],
        ),
        dict(
            name="index_fails",
            params=[dict(name="seq", type=numbers), dict(name="index", type="i64")],
            result="bool",
            body=[
                dict(
                    op="try",
                    body=[dict(op="expr", value=get(seq, var("index", "i64")))],
                    handlers=[
                        dict(
                            categories=["index_error"],
                            binding=None,
                            body=[dict(op="return", value=lit(True, "bool"))],
                        )
                    ],
                    **{"else": [], "finally": []},
                ),
                dict(op="return", value=lit(False, "bool")),
            ],
        ),
    ]
    source = tmp_path / "exec-ir.json"
    write_json(
        source,
        document(
            IR,
            status="READY",
            module=dict(entry="main", externs=[]),
            functions=[member(tmp_path, "functions", f) for f in functions],
        ),
    )
    for stage, file in (
        ("B1", "exec-ir.json"),
        ("B2a", "scheduled-ir.json"),
        ("B2b", "target-ir.json"),
        ("B3", "cpp20.json"),
    ):
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                _RUN_LAYER,
                stage,
                tmp_path / file,
                tmp_path,
                *(["1"] if stage == "B2a" else []),
            ],
            env=dict(
                os.environ, PYTHONPATH=str(Path(scar.__file__).resolve().parents[1])
            ),
            check=True,
            capture_output=True,
            timeout=60,
        )
    project = tmp_path / "project"
    assert len(list((project / "src").glob("*.cpp"))) >= 3
    assert (
        subprocess.run(
            [project / "bin/scar_entry"], input=b"\xff\n", check=True
        ).returncode
        == 0
    )
    header = (project / "include/scar/program.hpp").read_text()
    assert 'extern "C" bool is_pooled' not in header and "bool is_pooled" in header

    from scar.backend.cpp20.lowering import Lowerer

    lower = Lowerer(dict(functions=[]), {})
    lower.expression(pooled)
    with pytest.raises(ValueError, match="conflicting.*constant pool slot"):
        lower.expression(make(make(slot=0), lit("changed", "text"), slot=1))
    with pytest.raises(ValueError, match="requires literal"):
        lower.expression(make(var("local", "text"), slot=3))
    floats = dict(kind="sequence_ref", element="f64")
    with pytest.raises(ValueError, match="unsupported sorted sequence"):
        lower.expression(
            sequence(
                "sorted_copy",
                floats,
                base=var("floats", floats),
                comparison="signed_integer",
            )
        )


def test_math_interface_declaration_remains_distinct_from_vector_authorization(
    tmp_path,
):
    from copy import deepcopy

    from scar.backend.cpp20 import lower_scheduled
    from scar.backend.planning import schedule_ir
    from scar.contract import digest, json_bytes, read_member

    path, sha = "/absent/selected-math.h", "c" * 64
    captured = dict(path=path, sha256=sha, size=201, evidence="gnu_semantic_preinclude")
    evidence = dict(file=path, sha256=sha, line=17, provider=captured["evidence"])
    interface = dict(
        semantics_version=1,
        interface="pow",
        declaration="gnu_builtin",
        execution="not_observed",
        attributes={"simd": {"branch": "notinbranch"}},
        evidence=evidence,
        target_condition=dict(selector="x86_64", evaluation="unresolved"),
    )
    contract = dict(
        semantics_version=1,
        scope="compilation_unit",
        policy=dict(math_errno=False),
        caller_observations=dict(errno="unproven", fenv="unproven"),
        semantic_capture=dict(returncode=0, inputs=[captured]),
        runtime_fp_environment="unobserved",
        source_math_interfaces=[
            interface,
            dict(interface, evidence=dict(evidence, sha256="d" * 64)),
        ],
    )
    args = [dict(op="literal", type="f64", value=v) for v in (2.0, 3.0)]
    signed = dict(
        normalized_name="pow",
        result_type="f64",
        argument_types=["f64", "f64"],
        semantics_version=1,
    )
    power = dict(
        op="intrinsic",
        name="pow",
        site="graph/2",
        args=args,
        type="f64",
        source_intrinsic_contract=signed,
    )
    integer_power = dict(
        power,
        name="powi",
        site="graph/3",
        args=[args[0], dict(op="literal", type="i32", value=3)],
    )
    untouched = dict(op="call", callee="pow", site="graph/4", args=args)
    local_pow = dict(
        name="pow",
        params=[dict(name=n, type="f64") for n in ("x", "y")],
        result="f64",
        body=[dict(op="return", value=dict(op="var", name="x", type="f64"))],
    )
    body = [dict(op="expr", value=node) for node in (integer_power, power, untouched)]
    ref = dict(
        op="map",
        site="ref",
        index="i",
        independent=True,
        extent=dict(op="literal", type="i64", value=3),
        body=[dict(op="expr", value=dict(power, site="graph/ref"))],
    )
    fused = dict(
        ref,
        site="fused",
        body=[dict(op="expr", value=dict(power, site="graph/unselected"))],
    )
    body.append(
        dict(
            op="schedule_choice",
            site="choice",
            reference=[ref],
            fused=fused,
            guard="masked_floating_exceptions",
            proof=dict(stale=dict(power, site="graph/history")),
        )
    )
    body[0]["validation_evidence"] = dict(power, site="graph/not-executed")
    cosine = dict(
        name="cos",
        provider="libm",
        params=[dict(name="x", type="f64")],
        result="f64",
        location=dict(file="/sdk/bits/mathcalls.h", line=62),
    )
    body.append(
        dict(
            op="expr",
            value=dict(op="call", callee="cos", site="graph/5", args=args[:1]),
        )
    )
    body.append(dict(op="return", value=dict(op="literal", type="i32", value=0)))
    fn = dict(
        name="main",
        result="i32",
        params=[],
        body=body,
        source_numeric_contract=contract,
    )
    origin, destination = tmp_path / "execution", tmp_path / "schedule"
    source = member(origin, "functions", fn)
    inventory = dict(
        software_headers={path: dict(size=201, sha256=sha)},
        platform=dict(architecture="x86_64", isa_profiles={"isa:0": ["sse2"]}),
    )
    write_json(
        origin / "exec-ir.json",
        document(
            IR,
            status="READY",
            functions=[source, member(origin, "functions", local_pow)],
            module=dict(
                entry="main",
                externs=[cosine],
                resource_model=inventory,
                math_interface_binding_sha256="forged",
            ),
        ),
    )
    filename = schedule_ir(origin, destination, policy="serial")
    origin.rename(tmp_path / "hidden-input")
    package = json.loads(filename.read_bytes())
    physical = read_member(destination, package["physical"])
    math = physical["math_interfaces"]
    decision = math["functions"]["main"]
    assert package["module"]["math_interface_binding_sha256"] == digest(
        json_bytes(math)
    )
    assert decision["exec_function_sha256"] == source["sha256"]
    assert decision["scheduled_function_sha256"] == package["functions"][0]["sha256"]
    retained = read_member(destination, package["functions"][0])["body"]
    assert retained[:3] == body[:3] and retained[3]["site"] == "ref"
    assert decision["source_contract_sha256"] == digest(json_bytes(contract))
    profile = math["source_contracts"][decision["source_contract_sha256"]]
    assert profile["caller_observations"] == contract["caller_observations"]
    assert [c["evidence_verified"] for c in profile["declarations"]["pow"]] == [
        True,
        False,
    ]
    site, mapped, external = decision["sites"]
    assert external["site"] == "graph/5" and external["external_declaration"] == dict(
        symbol="cos",
        provider="libm",
        declaration_sha256=digest(json_bytes(cosine)),
        parameter_types=["f64"],
        result="f64",
        variadic=False,
    )
    assert external["selected"] == "preserve_operation"
    assert "source_vector_math_declaration" in external["obligations"]
    forged = deepcopy(physical)
    forged_math = forged["math_interfaces"]
    forged_math["functions"]["main"]["sites"][-1]["external_declaration"][
        "result"
    ] = "f32"
    bad = destination / "forged-abi.json"
    write_json(
        bad,
        dict(
            package,
            module=dict(
                package["module"],
                math_interface_binding_sha256=digest(json_bytes(forged_math)),
            ),
            physical=member(destination, "scheduled-units", forged),
        ),
    )
    with pytest.raises(ValueError, match="external declaration does not match"):
        lower_scheduled(bad, tmp_path / "rejected")
    assert mapped["site"] == "graph/ref" and mapped["operation_path"] == [
        "body",
        3,
        "body",
        0,
        "value",
    ]
    assert site["site"] == "graph/2" and site["operation_path"] == ["body", 1, "value"]
    assert (
        site["source_intrinsic_signature_verified"]
        and site["selected"] == "preserve_operation"
    )
    assert (
        site["vector_eligibility"] == "not_qualified"
        and math["target"]["vector_library"] is None
    )
    assert {
        "source_declaration_target_condition",
        "caller_errno_observation_proof",
        "caller_fenv_observation_proof",
        "target_vector_library_abi_qualification",
    } <= set(site["obligations"])

    other_member = package["functions"][1]["sha256"]
    for owner_changes, site_changes, error in (
        ({}, {}, "missing or mismatched physical math"),
        ({"scheduled_function_sha256": other_member}, {}, "function member"),
        ({}, {"selected": "vector_call"}, "unsupported scheduled math"),
        ({}, {"vector_eligibility": "qualified"}, "unsupported scheduled math"),
        ({}, {"argument_types": ["f64", "i64"]}, "does not match"),
        (
            {},
            {
                "operation_path": ["body", 0, "validation_evidence"],
                "site": "graph/not-executed",
            },
            "does not refer to a lowered operation",
        ),
    ):
        changed = deepcopy(physical)
        binding = changed["math_interfaces"]
        owner = binding["functions"]["main"]
        owner.update(owner_changes)
        owner["sites"][0].update(site_changes)
        expected_sha = (
            digest(json_bytes(binding)) if owner_changes or site_changes else "0" * 64
        )
        changed_package = dict(
            package,
            module=dict(package["module"], math_interface_binding_sha256=expected_sha),
            physical=member(destination, "scheduled-units", changed),
        )
        bad = destination / "invalid.json"
        write_json(bad, changed_package)
        with pytest.raises(ValueError, match=error):
            lower_scheduled(bad, tmp_path / "rejected")
        assert not (tmp_path / "rejected/target-ir.json").exists()

    target, printed = tmp_path / "target", tmp_path / "project"
    for stage, incoming, out in (("B2a", filename, target), ("B2b", target, printed)):
        subprocess.run(
            [sys.executable, "-B", "-c", _RUN_LAYER, stage, str(incoming), str(out)],
            cwd=tmp_path,
            env=dict(
                os.environ, PYTHONPATH=str(Path(scar.__file__).resolve().parents[1])
            ),
            check=True,
            capture_output=True,
            timeout=60,
        )
        if stage == "B2a":
            destination.rename(tmp_path / "hidden-schedule")
    audit = json.loads((target / "target-ir.json").read_bytes())[
        "math_interface_consumption"
    ]
    assert audit == dict(
        semantics_version=1,
        scope="scheduled_operations",
        binding_sha256=package["module"]["math_interface_binding_sha256"],
        selected="preserve_operation",
        vector_eligibility="not_qualified",
        functions=1,
        sites=3,
        interfaces={"pow": 2, "cos": 1},
    )
    assert (
        json.loads((printed / "cpp20.json").read_bytes())["math_interface_consumption"]
        == audit
    )


def test_invocation_access_qualifies_only_sealed_definition_parameters(tmp_path):
    from dataclasses import replace

    from scar.backend.cpp20.lowering import Lowerer
    from scar.backend.cpp20.model import Decl, Type, named, validate_node
    from scar.contract import PHYSICAL, digest, json_bytes

    pointer = dict(kind="ptr", to="i32")
    claim = dict(
        parameter="out",
        rule="exclusive_modified_bytes",
        region="accessed_referent_bytes",
        paths="based_on_parameter",
        readonly_when_accessed=False,
        basis=dict(
            kind="source_precondition", rule_id="test", evidence={"binding": "out"}
        ),
    )
    read_claim = dict(
        claim,
        parameter="input",
        readonly_when_accessed=True,
        basis=dict(claim["basis"], evidence={"binding": "input"}),
    )
    access = dict(
        semantics_version=1, scope="invocation", parameters=[read_claim, claim]
    )
    worker = dict(
        name="store",
        result="void",
        params=[
            dict(name="out", type=dict(kind="const", of=pointer)),
            dict(name="input", type=dict(kind="ptr", to=dict(kind="const", of="i32"))),
            dict(name="hint", type=pointer, elementwise_noalias=True),
        ],
        body=[
            dict(op="block", body=[dict(op="decl", name="out", type="i32")]),
            dict(
                op="assign",
                target=dict(
                    op="unary", operator="deref", value=dict(op="var", name="out")
                ),
                value=dict(
                    op="unary", operator="deref", value=dict(op="var", name="input")
                ),
            ),
        ],
        access_contract=access,
    )
    main = dict(
        name="main",
        params=[],
        result="i32",
        body=[dict(op="return", value=dict(op="literal", type="i32", value=0))],
    )
    roots = dict(
        contract_sha256=digest(json_bytes(access)), parameters=["input", "out"]
    )
    decision = dict(
        semantics_version=1,
        scope="invocation",
        feature="gnu_restrict_parameter",
        qualifier="__restrict__",
        functions={"store": roots},
        qualification=dict(
            supported=True,
            evidence="compile_only",
            application_executed=False,
            cxx_standard=20,
            definition="SCAR_CXX_ACCESS_PROBE=1",
            compiler=dict(path="/offline/compiler", size=1, sha256="1" * 64),
            compiler_arguments=[],
            probe_sha256="2" * 64,
        ),
    )

    def module(binding):
        return dict(
            entry="main", parameter_access_binding_sha256=digest(json_bytes(binding))
        )

    def lower(binding, *, worker_fn=worker, seal=None):
        fields = module(binding) if seal is None else seal
        return Lowerer(
            dict(fields, functions=[main, worker_fn]), dict(parameter_access=binding)
        )

    scheduled = tmp_path / "scheduled"
    revision = "3" * 64
    physical = document(PHYSICAL, ir_sha256=revision, parameter_access=decision)
    write_json(
        scheduled / "scheduled-ir.json",
        document(
            "scar.scheduled-ir",
            status="READY",
            ir_sha256=revision,
            module=module(decision),
            physical=member(scheduled, "physical", physical),
            functions=[member(scheduled, "functions", fn) for fn in (main, worker)],
        ),
    )
    target, result = tmp_path / "target", tmp_path / "result"
    env = dict(os.environ, PYTHONPATH=str(Path(scar.__file__).resolve().parents[1]))
    for stage, source, destination in (
        ("B2a", scheduled, target),
        ("B2b", target, result),
    ):
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                _RUN_LAYER,
                stage,
                str(source),
                str(destination),
            ],
            cwd=tmp_path,
            env=env,
            check=True,
            capture_output=True,
        )
    include = result / "project/include"
    source = result / "project/src/program.cpp"
    body = source.read_text()
    assert body.count("__restrict__") == 2
    assert "__restrict__ v_out" in body and "__restrict__ v_input" in body
    assert "__restrict__" not in (include / "scar/program.hpp").read_text()
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-Werror",
            "-pedantic-errors",
            "-I",
            str(include),
            "-c",
            str(source),
            "-o",
            str(tmp_path / "program.o"),
        ],
        check=True,
        capture_output=True,
    )

    with pytest.raises(ValueError, match="parameter access binding"):
        Lowerer(dict(functions=[worker]), {})
    with pytest.raises(ValueError, match="parameter access binding"):
        lower(decision, seal=dict(parameter_access_binding_sha256="0" * 64))
    for altered in (
        dict(functions={"store": dict(roots, parameters=["hint", "input", "out"])}),
        dict(functions={"store": dict(roots, contract_sha256="0" * 64)}),
        dict(functions={"extra": roots, "store": roots}),
        dict(qualification={"supported": False, "reason": "feature unavailable"}),
        dict(qualification={"supported": True}),
    ):
        with pytest.raises(ValueError, match="parameter access"):
            lower(dict(decision, **altered))
    weak = dict(
        access, parameters=[dict(read_claim, readonly_when_accessed=False), claim]
    )
    weak_decision = dict(
        decision,
        functions={"store": dict(roots, contract_sha256=digest(json_bytes(weak)))},
    )
    with pytest.raises(ValueError, match="const|readonly|read.only"):
        lower(weak_decision, worker_fn=dict(worker, access_contract=weak))
    region_only = {k: v for k, v in worker.items() if k != "access_contract"}
    with pytest.raises(ValueError, match="parameter access binding"):
        lower(decision, worker_fn=region_only)
    unsupported = dict(
        decision,
        qualifier=None,
        qualification=dict(supported=False, reason="compiler unavailable"),
    )
    for emitter in (
        lower(unsupported),
        Lowerer(dict(functions=[main, region_only], entry="main"), {}),
    ):
        units = emitter.units()
        assert all(
            not p.qualifiers
            for u in units.values()
            for d in u.declarations
            for p in d.params
        )

    param = Decl("param", type=named("int"), name="n", qualifiers=("__restrict__",))
    for invalid in (
        param,
        replace(param, name=None, type=None, definition="variadic"),
        replace(param, type=Type("ptr", element=Type("function", result=named("int")))),
        replace(
            param, type=Type("ptr", element=named("int")), qualifiers=("restrict",)
        ),
    ):
        with pytest.raises(ValueError):
            validate_node(invalid, Decl)


def test_link_image_compilation_survives_target_relocation(tmp_path):
    from scar.backend.cpp20 import lower_scheduled, print_target
    from scar.contract import PHYSICAL, digest, json_bytes

    revision = "1" * 64
    tool = dict(path="/qualified/tool", size=1, sha256="2" * 64)
    decision = dict(
        semantics_version=1,
        scope="link_image",
        mode="gnu_lto",
        workers=2,
        evidence=dict(worker_budget=2),
        reason="cross-unit calls",
        qualification=dict(
            supported=True, compiler_id="GNU", compiler=tool, cmake=tool
        ),
    )
    scheduled = tmp_path / "scheduled"
    target, printed = tmp_path / "target", tmp_path / "printed"
    members = []
    for name, value in (
        ("answer", dict(op="literal", type="i32", value=7)),
        ("main", dict(op="call", callee="answer", args=[])),
    ):
        function = dict(
            name=name, params=[], result="i32", body=[dict(op="return", value=value)]
        )
        members.append(member(scheduled, "functions", function))

    def seal(settings, bound):
        module = dict(entry="main")
        physical = document(PHYSICAL, ir_sha256=revision)
        if settings is not None:
            physical["compilation"] = settings
        if bound is not None:
            module["compilation_binding_sha256"] = digest(json_bytes(bound))
        write_json(
            scheduled / "scheduled-ir.json",
            document(
                "scar.scheduled-ir",
                status="READY",
                ir_sha256=revision,
                module=module,
                physical=member(scheduled, "physical", physical),
                functions=members,
            ),
        )

    def isolated(stage, source, out, *args):
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                _RUN_LAYER,
                stage,
                str(source),
                str(out),
                *args,
            ],
            cwd=tmp_path,
            env=dict(
                os.environ, PYTHONPATH=str(Path(scar.__file__).resolve().parents[1])
            ),
            check=True,
            capture_output=True,
            timeout=60,
        )

    for settings, bound, error in (
        (decision, None, "compilation binding"),
        (None, decision, "compilation binding"),
        (dict(decision, workers=1), decision, "compilation binding"),
    ):
        seal(settings, bound)
        with pytest.raises(ValueError, match=error):
            lower_scheduled(scheduled, target)
    for settings, error in (
        (dict(decision, workers=3), "compilation budget"),
        (
            dict(decision, qualification=dict(supported=True, compiler_id="Clang")),
            "GNU",
        ),
    ):
        seal(settings, settings)
        with pytest.raises(ValueError, match=error):
            lower_scheduled(scheduled, target)
    noop = dict(decision, mode="none", workers=1, qualification=dict(supported=False))
    default = dict(semantics_version=1, interprocedural=dict(mode="none", workers=1))
    for settings in (None, noop):
        seal(settings, settings)
        artifact = lower_scheduled(scheduled, tmp_path / "default-target")
        assert json.loads(artifact.read_bytes())["build_options"] == default

    seal(decision, decision)
    isolated("B2a", scheduled, target, "1")
    scheduled.rename(tmp_path / "hidden-schedule")
    target = target.rename(tmp_path / "relocated-target")
    isolated("B2b", target, printed)
    package = json.loads((printed / "cpp20.json").read_bytes())
    assert len(package["sources"]) >= 2
    assert package["build_options"] == dict(
        semantics_version=1, interprocedural=dict(mode="gnu_lto", workers=2)
    )
    old = json.loads((target / "target-ir.json").read_bytes())
    del old["build_options"]
    historical = target / "legacy.json"
    write_json(historical, old)
    legacy = print_target(historical, tmp_path / "legacy-project")
    assert legacy["build_options"] == default
    assert "-flto" not in (Path(legacy["project"]) / "CMakeLists.txt").read_text()
    write_json(
        historical,
        dict(
            old,
            build_options=dict(default, interprocedural=dict(mode="none", workers=2)),
        ),
    )
    with pytest.raises(ValueError, match="build options"):
        print_target(historical, tmp_path / "invalid-project")

    def configure(source, directory, limit, *, check=True):
        return subprocess.run(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(directory),
                "-DCMAKE_CXX_COMPILER=g++",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                f"-DSCAR_COMPILER_WORKER_LIMIT={limit}",
            ],
            check=check,
            capture_output=True,
            text=True,
        )

    assert (
        "warning"
        not in configure(legacy["project"], tmp_path / "legacy-build", 1).stderr.lower()
    )
    invalid = configure(legacy["project"], tmp_path / "legacy-build", 0, check=False)
    assert invalid.returncode != 0 and "must be a positive integer" in invalid.stderr

    images = [
        dict(id=name, output=name, entry="main", target=file_receipt(tmp_path, package))
        for name, package in (
            ("first", target / "target-ir.json"),
            ("second", artifact),
        )
    ]
    bundle = tmp_path / "bundle.json"
    write_json(
        bundle,
        document(
            "scar.cpp20-target-ir",
            kind="image_bundle",
            root_image="first",
            images=images,
            ir_sha256=revision,
        ),
    )
    result = tmp_path / "bundle-project"
    isolated("B2b", bundle, result)
    bundle.unlink()
    target.rename(tmp_path / "hidden-target")
    artifact.parent.rename(tmp_path / "hidden-default-target")
    root = result / "project"
    cmake = (root / "CMakeLists.txt").read_text()
    assert all(f"$<TARGET_OBJECTS:scar_{i['id']}.objects>" in cmake for i in images)
    assert "add_dependencies(scar_first scar.objects.ready)" in cmake
    assert "add_dependencies(scar_second scar_first)" in cmake
    build = tmp_path / "build"
    assert "warning" not in configure(root, build, 1).stderr.lower()
    commands = json.loads((build / "compile_commands.json").read_bytes())
    for command in commands:
        lto = Path(command["file"]).is_relative_to(root / "images/first")
        assert ("-flto=1" in command["command"]) == lto
        assert ("-fno-fat-lto-objects" in command["command"]) == lto
    link = subprocess.run(
        ["cmake", "--build", str(build), "--parallel", "2", "--verbose"],
        check=True,
        capture_output=True,
        text=True,
    )
    invocations = [
        line for line in link.stdout.splitlines() if "g++" in line and " -o " in line
    ]
    assert len(invocations) == len(commands) + 2
    assert all(" -c " in line for line in invocations[:-2])
    for image, line in zip(images, invocations[-2:]):
        output = image["output"]
        assert Path(line.split(" -o ", 1)[1].split()[0]).name == output
        assert ("-flto=1" in line) == (output == "first")
        assert ("-flto-partition=one" in line) == (output == "first")
        assert subprocess.run([build / "images" / image["id"] / output]).returncode == 7


def test_function_numeric_permissions_preserve_strict_neighbors_across_lto(
    tmp_path, monkeypatch
):
    from copy import deepcopy
    from scar.backend.build import build
    from scar.backend.cpp20 import lower_scheduled, print_target
    from scar.backend.cpp20.lowering import Lowerer
    from scar.backend.planning import schedule_ir
    from scar.contract import digest, json_bytes, read_member

    monkeypatch.setenv("CXX", "g++")
    monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", "2")
    # Three syntax nodes in each small sqrt definition: group a permitted and a
    # strict definition together, then start a separate TU for the next neighbor.
    monkeypatch.setattr(Lowerer, "unit_weight_limit", 6)
    source, plan, target, delivery = (
        tmp_path / s for s in ("source", "plan", "target", "delivery")
    )

    def var(name, ty):
        return dict(op="var", name=name, type=ty)

    def integer(value):
        return dict(op="literal", type="i32", value=value)

    def invoke(symbol, result, *args):
        return dict(op="call", callee=symbol, args=list(args), type=result)

    def compare(op, a, b):
        return dict(op="binary", operator=op, type="bool", left=a, right=b)

    def sqrt_fn(name):
        return dict(
            name=name,
            params=[dict(name="x", type="f64")],
            result="f64",
            body=[
                dict(
                    op="return",
                    value=dict(
                        op="intrinsic", name="sqrt", args=[var("x", "f64")], type="f64"
                    ),
                )
            ],
        )

    local = sqrt_fn("permitted")
    local["source_numeric_contract"] = dict(
        semantics_version=1,
        scope="compilation_unit",
        policy=dict(math_errno=False, rounding_math=False),
        caller_observations=dict(errno="unproven", fenv="unproven"),
    )
    local["numeric_contract"] = dict(
        semantics_version=1,
        scope="function",
        source_contract_sha256=digest(json_bytes(local["source_numeric_contract"])),
        permissions=dict(may_omit_math_errno=True, may_assume_default_rounding=True),
    )
    functions = [local, sqrt_fn("strict_same"), sqrt_fn("strict_other")]
    point = dict(kind="ptr", to="i32")
    error = dict(op="unary", operator="deref", value=var("error", point), type="i32")
    main = dict(
        name="main",
        result="i32",
        params=[
            dict(name="argc", type="i32"),
            dict(name="argv", type=dict(kind="ptr", to=dict(kind="ptr", to="char"))),
        ],
        body=[
            dict(
                op="decl",
                name="error",
                type=point,
                init=invoke("__errno_location", point),
            ),
            dict(
                op="decl",
                name="x",
                type="f64",
                init=dict(
                    op="cast",
                    type="f64",
                    value=dict(
                        op="unary", operator="neg", value=var("argc", "i32"), type="i32"
                    ),
                ),
            ),
        ],
    )
    for i, fn in enumerate(functions):
        main["body"].extend(
            [
                dict(op="assign", target=error, value=integer(0)),
                dict(
                    op="decl",
                    name=f"root{i}",
                    type="f64",
                    init=invoke(fn["name"], "f64", var("x", "f64")),
                ),
                dict(
                    op="if",
                    condition=dict(
                        op="unary",
                        operator="not",
                        type="bool",
                        value=dict(
                            op="intrinsic",
                            name="isnan",
                            type="bool",
                            args=[var(f"root{i}", "f64")],
                        ),
                    ),
                    then=[dict(op="return", value=integer(1 + i))],
                    **{"else": []},
                ),
            ]
        )
        if fn is not local:
            main["body"].append(
                dict(
                    op="if",
                    condition=compare("eq", error, integer(0)),
                    then=[dict(op="return", value=integer(4 + i))],
                    **{"else": []},
                )
            )
    main["body"].append(dict(op="return", value=integer(0)))
    write_json(
        source / "exec-ir.json",
        document(
            IR,
            status="READY",
            module=dict(
                entry="main",
                externs=[
                    dict(
                        name="__errno_location",
                        params=[],
                        result=point,
                        provider="libc",
                        header="errno.h",
                    )
                ],
            ),
            functions=[member(source, "functions", f) for f in [main, *functions]],
        ),
    )
    scheduled = schedule_ir(source, plan, policy="serial", workers=1)
    pkg = json.loads(scheduled.read_bytes())
    physical = read_member(plan, pkg["physical"])
    binding = physical["numeric_permissions"]
    assert binding["qualification"]["supported"], binding["qualification"]
    assert binding["functions"].keys() == {"permitted"}
    assert binding["functions"]["permitted"]["options"] == [
        "no-math-errno",
        "no-rounding-math",
    ]
    assert physical["compilation"]["mode"] == "gnu_lto"

    def altered(binding):
        mod = {
            k: v
            for k, v in pkg["module"].items()
            if k != "numeric_permissions_binding_sha256"
        }
        phys = deepcopy(physical)
        if binding is None:
            del phys["numeric_permissions"]
        else:
            phys["numeric_permissions"] = binding
            mod["numeric_permissions_binding_sha256"] = digest(json_bytes(binding))
        path = plan / "altered.json"
        write_json(
            path, dict(pkg, module=mod, physical=member(plan, "scheduled-units", phys))
        )
        return path

    for key, value in (
        ("options", ["fast-math"]),
        ("scheduled_function_sha256", "0" * 64),
    ):
        bad = deepcopy(binding)
        bad["functions"]["permitted"][key] = value
        with pytest.raises(ValueError, match="numeric options"):
            lower_scheduled(altered(bad), tmp_path / "rejected")
    with pytest.raises(ValueError, match="numeric permission"):
        lower_scheduled(altered(None), tmp_path / "rejected")
    fallback = deepcopy(binding)
    fallback.update(
        qualification=dict(supported=False, reason="unavailable compiler feature")
    )
    fallback["functions"]["permitted"]["options"] = []
    fallback_target = lower_scheduled(altered(fallback), tmp_path / "strict")
    assert (
        json.loads(fallback_target.read_bytes())["numeric_permission_consumption"][
            "compiler_sha256"
        ]
        is None
    )

    ir = lower_scheduled(scheduled, target)
    receipt = json.loads(ir.read_bytes())["numeric_permission_consumption"]
    assert (
        receipt["functions"]["permitted"]["scheduled_function_sha256"]
        == binding["functions"]["permitted"]["scheduled_function_sha256"]
    )
    plan.rename(tmp_path / "moved-schedule")
    result = print_target(ir, delivery)
    assert result["numeric_permission_consumption"] == receipt
    tus = [f.read_text() for f in (delivery / "project/src").glob("*.cpp")]
    same = next(tu for tu in tus if "gnu::optimize" in tu)
    assert "strict_same(" in same and "strict_other(" not in same
    assert any("strict_other(" in tu and "gnu::optimize" not in tu for tu in tus)
    assert (
        "gnu::optimize"
        not in (delivery / "project/include/scar/program.hpp").read_text()
    )
    cmake = (delivery / "project/CMakeLists.txt").read_text()
    assert "-fno-fast-math -frounding-math -ffp-contract=off" in cmake
    assert "-fno-math-errno" not in cmake and "-fno-rounding-math" not in cmake
    target.rename(tmp_path / "moved-target")
    executable = build(delivery / "cpp20.json")
    subprocess.run([str(executable)], check=True)


def test_explicit_error_causes_survive_handlers_cleanup_and_exit(tmp_path):
    from scar.backend.cpp20 import lower_scheduled, print_target
    from scar.backend.cpp20.lowering import Lowerer
    from scar.contract import PHYSICAL

    handle = dict(kind="handle", resource="error")
    value = dict(op="var", name="value", type=handle)
    cause = dict(op="var", name="cause", type=handle)
    none = dict(op="literal", type="unit", value=None)
    plain = dict(op="throw", value=value)
    dispatch = dict(plain, cause_mode="explicit", cause=cause)
    clear = dict(plain, cause_mode="explicit", cause=none)
    region = dict(
        op="try",
        body=[dispatch],
        handlers=[
            dict(categories=["base_exception"], binding=None, body=[dict(op="rethrow")])
        ],
        **{"finally": [dict(op="expr", value=none)]},
    )

    def routine(name, body, *params):
        return dict(
            name=name,
            params=[dict(name=p, type=handle) for p in params],
            result="void",
            body=[body],
        )

    main = dict(
        name="main",
        params=[],
        result="i32",
        body=[dict(op="return", value=dict(op="literal", type="i32", value=0))],
    )
    funcs = [
        main,
        routine("with_cause", region, "value", "cause"),
        routine("from_none", clear, "value"),
        routine("again", plain, "value"),
    ]
    physical = document(PHYSICAL, ir_sha256="a" * 64)
    schedule = tmp_path / "scheduled"
    module = dict(entry="main")
    for invalid in (
        dict(plain, cause=cause),
        dict(plain, cause_mode="explicit"),
        dict(dispatch, cause_mode="implicit"),
        dict(dispatch, cause=dict(op="literal", type="text", value="bad")),
    ):
        with pytest.raises(ValueError, match="explicit throw"):
            Lowerer(dict(module, functions=funcs), physical).control.statement(invalid)
    write_json(
        schedule / "scheduled-ir.json",
        document(
            "scar.scheduled-ir",
            status="READY",
            ir_sha256="a" * 64,
            module=module,
            physical=member(schedule, "physical", physical),
            functions=[member(schedule, "functions", f) for f in funcs],
        ),
    )
    target = lower_scheduled(schedule, tmp_path / "target")
    project = Path(print_target(target, tmp_path / "delivery")["project"])
    sources = sorted((project / "src").glob("*.cpp"))
    runtime = tmp_path / "libcause.so"
    args = ["c++", "-std=c++20", "-O2", "-Werror", "-I", str(project / "include")]
    subprocess.run([*args, "-fPIC", "-shared", *sources, "-o", runtime], check=True)
    # A native observer reads the identities and cause metadata after invoking the
    # sealed IR's actual throw/handler/finally paths. The SDK is the packaged copy.
    observer = tmp_path / "observe.cpp"
    observer.write_text(r"""
#include "scar/program.hpp"
#include <cassert>
int main() {
  std::weak_ptr<const scar::error_record> released;
  {
    auto inner = scar::make_error("module_not_found_error", "dependency");
    auto outer = scar::make_error("import_error", std::string("a\0b", 3));
    released = outer;
    assert(scar::matches(inner, "import_error") && !scar::error_suppresses_context(outer));
    try { with_cause(outer, inner); assert(false); }
    catch (const scar::program_error &e) {
      auto seen = scar::capture_error();
      assert(seen == outer && scar::error_cause(seen) == inner);
      assert(scar::error_suppresses_context(seen) && e.message() == std::string("a\0b", 3));
    }
    try { again(outer); assert(false); }
    catch (...) { assert(scar::error_cause(scar::capture_error()) == inner); }
    try { from_none(outer); assert(false); }
    catch (...) {
      auto seen = scar::capture_error();
      assert(seen == outer && scar::error_suppresses_context(seen) && !scar::error_cause(seen));
    }
    try { with_cause(outer, outer); assert(false); }
    catch (...) { assert(scar::error_cause(scar::capture_error()) == outer); }
    auto exit = scar::make_exit_error(19,"exit",std::string(""));
    try { with_cause(exit,inner); assert(false); }
    catch (const scar::exit_signal &ex) {
      assert(ex.status == 19 && ex.exit_message() && ex.exit_message()->empty());
      assert(scar::capture_error() == exit && scar::error_cause(exit) == inner);
    }
    try { throw std::system_error(std::make_error_code(std::errc::permission_denied)); }
    catch (...) {
      auto native = scar::capture_error();
      auto original = std::current_exception();
      try { again(native); assert(false); }
      catch (...) { assert(std::current_exception() == original); }
      try { with_cause(native, inner); assert(false); }
      catch (const std::system_error &ex) {
        assert(ex.code() == std::errc::permission_denied);
        assert(scar::capture_error() == native && scar::error_cause(native) == inner);
      }
    }
  }
  assert(released.expired());
}
""")
    exe = tmp_path / "observer"
    subprocess.run([*args, observer, runtime, "-o", exe], check=True)
    subprocess.run([exe], check=True)


def test_complex128_array_provider_preserves_components_and_storage(tmp_path):
    from scar.backend import compile_ir
    from scar.backend.cpp20.lowering import Lowerer
    from scar.backend.planning import _qualify

    def arr(rank, element="c128"):
        return dict(kind="ndarray", element=element, rank=rank)

    def lit(ty, val):
        return dict(op="literal", type=ty, value=val)

    def dims(*sizes):
        return [lit("i64", n) for n in sizes]

    def cx(real, imag):
        return lit("c128", [real, imag])

    def math_op(operator, left, right, update=False):
        scalar = isinstance(left["type"], str) and isinstance(right["type"], str)
        word = {"add": "add", "sub": "subtract", "mul": "multiply", "div": "divide"}[
            operator
        ]
        return dict(
            op="array_update" if update else "array_binary",
            type="unit" if update else left["type"],
            left=left,
            right=right,
            operator=operator,
            broadcasting="trailing",
            numeric_policy="numpy_default",
            effects=["STATE", "IO"] if update else ["MEM", "VAL", "STATE", "IO"],
            warning=dict(operation=("scalar " if scalar else "") + word),
            location=dict(file="numbers", line=19),
        )

    def load(base, *indices):
        return dict(
            op="array_load",
            type="c128",
            base=base,
            indices=dims(*indices),
            effects=["STATE"],
        )

    body = []

    def bind(name, value):
        body.append(dict(op="decl", name=name, type=value["type"], init=value))
        return dict(op="var", name=name, type=value["type"])

    def check(left, right):
        condition = dict(op="binary", operator="ne", left=left, right=right)
        body.append(
            dict(
                op="if",
                condition=condition,
                then=[dict(op="return", value=lit("i32", len(body)))],
                **{"else": []},
            )
        )

    m = bind(
        "matrix",
        dict(
            op="array_literal",
            type=arr(2),
            shape=dims(2, 2),
            values=[cx(1, 2), cx(3, 4), cx(5, 6), cx(7, 8)],
            effects=["MEM", "VAL"],
        ),
    )
    row = bind(
        "row",
        dict(
            op="array_view",
            type=arr(1),
            base=m,
            indices=dims(-1),
            effects=["STATE", "MEM"],
        ),
    )
    flat = bind(
        "flat",
        dict(
            op="array_reshape",
            type=arr(1),
            value=m,
            shape=dims(4),
            layout="C",
            effects=["STATE", "MEM"],
        ),
    )
    copied = bind(
        "copied",
        dict(op="array_copy", type=arr(1), value=row, effects=["MEM", "STATE", "XFER"]),
    )
    broadcast = bind(
        "expanded",
        dict(
            op="array_broadcast_copy",
            type=arr(2),
            value=row,
            shape=dims(3, 2),
            broadcasting="trailing",
            effects=["MEM", "STATE", "XFER"],
        ),
    )
    body.append(
        dict(
            op="array_store",
            base=row,
            indices=dims(0),
            value=cx(2, -3),
            broadcasting="trailing",
            effects=["STATE"],
        )
    )
    check(load(flat, 2), cx(2, -3))
    check(load(copied, 0), cx(5, 6))
    check(load(broadcast, 2, 0), cx(5, 6))
    added = bind("sum", math_op("add", row, copied))
    diff = bind("difference", math_op("sub", row, copied))
    check(load(added, 0), cx(7, 3))
    check(load(diff, 0), cx(-3, -9))
    body.append(math_op("mul", row, row, update=True))
    check(load(m, 1, 0), cx(-5, -12))
    zeros = bind(
        "zeros",
        dict(
            op="array_create",
            type=arr(1),
            shape=dims(2),
            initialization="zero",
            effects=["MEM", "VAL"],
        ),
    )
    check(load(zeros, 1), cx(0, 0))
    body.append(
        dict(
            op="array_store",
            base=zeros,
            indices=dims(1),
            value=dict(op="cast", type="c128", value=lit("f64", 3.5)),
            broadcasting="trailing",
            effects=["STATE"],
        )
    )
    check(load(zeros, 1), cx(3.5, 0))
    negative = bind(
        "negative",
        dict(
            op="array_unary",
            type=arr(1),
            value=row,
            operator="neg",
            effects=["MEM", "VAL"],
        ),
    )
    check(load(negative, 0), cx(5, 12))
    real = bind(
        "indices",
        dict(
            op="array_iota",
            type=arr(1, "f64"),
            count=lit("i64", 2),
            max_count=1 << 53,
            effects=["MEM", "VAL"],
        ),
    )
    check(
        dict(op="array_size", type="i64", value=real, effects=["STATE"]), lit("i64", 2)
    )
    check(math_op("div", lit("f64", 3.0), lit("f64", 2.0)), lit("f64", 1.5))
    special = bind("special", math_op("mul", cx("nan", "inf"), cx("inf", 0)))
    for component in ("real", "imag"):
        value = dict(op="intrinsic", name=component, type="f64", args=[special])
        check(
            dict(op="intrinsic", name="isnan", type="bool", args=[value]),
            lit("bool", True),
        )
    body.append(dict(op="return", value=lit("i32", 0)))
    fn = dict(name="main", result="i32", params=[], body=body)
    ir = tmp_path / "exec-ir.json"
    write_json(
        ir,
        document(
            IR,
            status="READY",
            module=dict(entry="main"),
            functions=[member(tmp_path, "functions", fn)],
        ),
    )
    release = compile_ir(ir, tmp_path / "delivery", policy="serial")
    assert release["status"] == "COMPILED", release
    physical = json.loads(Path(release["physical_plan"]).read_bytes())
    binding = physical["operator_backends"]["ndarray"]
    assert binding["elements"] == ["c128", "f64"]
    assert "array_iota" in binding["operations_by_element"]["f64"]
    assert "array_update.mul" in binding["operations_by_element"]["c128"]
    run = subprocess.run([release["executable"]], check=True, capture_output=True)
    assert (
        run.stderr
        == b"numbers:19: RuntimeWarning: invalid value encountered in scalar multiply\n"
    )

    lower = Lowerer(dict(functions=[]), physical)
    for bad in (
        dict(math_op("mul", row, row), operator="div"),
        dict(math_op("mul", row, row), operator="pow", power_path="power"),
        dict(math_op("mul", row, row), operator="eq"),
        dict(op="array_conjugate", value=row, type=row["type"], effects=["MEM", "VAL"]),
        dict(
            op="array_format", type="text", value=row, effects=["VAL", "MEM", "STATE"]
        ),
        dict(
            op="array_view",
            type=arr(1, "f64"),
            base=m,
            indices=dims(0),
            effects=["STATE", "MEM"],
        ),
    ):
        _, _, issues = _qualify(
            dict(functions=[dict(name="main", body=[dict(op="expr", value=bad)])])
        )
        assert [issue["code"] for issue in issues] == ["missing_operator_binding"]
        with pytest.raises(ValueError):
            lower.expression(bad)


def test_native_ortho_fourier_binding_survives_sdk_relocation(tmp_path, monkeypatch):
    import shutil

    import numpy as np

    from scar.backend.build import build
    from scar.backend.cpp20 import lower_scheduled, print_target
    from scar.backend.planning import _qualify, schedule_ir

    revision = "33ae5dc94c9cdc7f1c78346504a85de87cadaa12"
    installed = Path(
        os.environ.get("SCAR_FOURIER_SDK")
        or Path.home() / ".local/share/scar/sdk" / ("pocketfft-" + revision)
    )
    prefix = tmp_path / "source-sdk"
    shutil.copytree(installed, prefix)
    monkeypatch.setenv("SCAR_FOURIER_SDK", str(prefix))
    vector = dict(kind="ndarray", element="c128", rank=1)

    def lit(ty, v):
        return dict(op="literal", type=ty, value=v)

    def values(ty, dims, items):
        return dict(
            op="array_literal",
            type=ty,
            shape=[lit("i64", n) for n in dims],
            values=[lit("c128", parts) for parts in items],
            effects=["MEM", "VAL"],
        )

    def fft(source, direction, diagnostic=False):
        operation = dict(
            op="array_fft",
            operator_family="fourier",
            input=source,
            type=vector,
            axis=0,
            direction=direction,
            normalization="ortho",
            required_provider=dict(name="pocketfft", revision=revision),
            effects=["MEM", "VAL"],
            may_raise=True,
        )
        if diagnostic:
            operation.update(
                numeric_policy="numpy_default",
                effects=["MEM", "VAL", "STATE", "IO"],
                warning=dict(
                    operation={"forward": "fft", "inverse": "ifft"}[direction],
                    module="sealed.numeric",
                    location=dict(diagnostic_file="sealed/math", line=101),
                    source_line="transform(data)",
                ),
            )
        return operation

    body = []

    def bind(n, value):
        body.append(dict(op="decl", name=n, type=value["type"], init=value))
        return dict(op="var", name=n, type=value["type"])

    def print_values(value, size):
        for j in range(size):
            z = bind(
                f"z{len(body)}",
                dict(
                    op="array_load",
                    base=value,
                    indices=[lit("i64", j)],
                    type="c128",
                    effects=["STATE"],
                ),
            )
            components = [
                dict(op="intrinsic", name=k, type="f64", args=[z])
                for k in ("real", "imag")
            ]
            body.append(
                dict(
                    op="expr",
                    value=dict(
                        op="call",
                        callee="printf",
                        args=[dict(op="string", value="%a %a\n"), *components],
                        type="i32",
                    ),
                )
            )

    samples = [(1, -2), (-3, 1), (5, 4), (-1, -1), (0.25, 8), (2, -4), (1.25, 0)]
    n = len(samples)
    matrix = bind(
        "matrix", values(dict(vector, rank=2), [2, n], [(0, 0)] * n + samples)
    )
    data = bind(
        "data",
        dict(
            op="array_view",
            base=matrix,
            indices=[lit("i64", 1)],
            type=vector,
            effects=["STATE", "MEM"],
        ),
    )
    for direction in ("forward", "inverse"):
        print_values(bind(direction, fft(data, direction)), n)
    print_values(data, n)
    huge = bind("huge", values(vector, [4], [(float(np.finfo(np.float64).max), 0)] * 4))
    print_values(bind("extreme", fft(huge, "forward", diagnostic=True)), 1)
    empty = values(vector, [0], [])
    error = dict(op="var", name="error", type=dict(kind="handle", resource="error"))
    body.extend(
        [
            dict(op="decl", name="error", type=error["type"]),
            dict(
                op="try",
                body=[dict(op="expr", value=fft(empty, "inverse"))],
                handlers=[
                    dict(
                        categories=["value_error"],
                        binding=error,
                        body=[
                            dict(
                                op="return",
                                value=dict(
                                    op="binary",
                                    operator="ne",
                                    left=dict(
                                        op="host_call",
                                        interface="error.message",
                                        semantics_version=1,
                                        args=[error],
                                        type="text",
                                        effects=["VAL"],
                                    ),
                                    right=lit(
                                        "text",
                                        "Invalid number of FFT data points (0) specified.",
                                    ),
                                ),
                            )
                        ],
                    )
                ],
                **{"else": [], "finally": []},
            ),
            dict(op="return", value=lit("i32", 1)),
        ]
    )
    fn = dict(name="main", params=[], result="i32", body=body)
    printf = dict(
        name="printf",
        provider="libc",
        header="stdio.h",
        result="i32",
        variadic=True,
        params=[
            dict(name="format", type=dict(kind="ptr", to=dict(kind="const", of="char")))
        ],
    )
    ir = tmp_path / "execution/exec-ir.json"
    write_json(
        ir,
        document(
            IR,
            status="READY",
            module=dict(entry="main", externs=[printf]),
            functions=[member(ir.parent, "functions", fn)],
        ),
    )
    valid = fft(data, "forward")

    def issues(node, **fields):
        return _qualify(
            dict(
                functions=[dict(name="main", body=[dict(op="expr", value=node)])],
                **fields,
            )
        )[2]

    for override in (
        dict(type=dict(vector, rank=2)),
        dict(axis=-1),
        dict(normalization="backward"),
        dict(required_provider=dict(name="pocketfft", revision="0" * 40)),
        dict(n=2),
        dict(numeric_policy="numpy_default"),
    ):
        assert issues(dict(valid, **override))[0]["requirement"] == "fourier"
    guarded = dict(
        valid, source_binding=dict(required_family="module_registry", capture="lookup")
    )
    assert issues(guarded)[0]["requirement"] == "fourier"
    assert [
        i["requirement"]
        for i in issues(
            guarded, operator_requirements={"module_registry": {"operations": []}}
        )
    ] == ["module_registry"]
    scheduled = schedule_ir(ir, tmp_path / "schedule", policy="serial")
    physical = json.loads((scheduled.parent / "physical-plan.json").read_bytes())
    selected = physical["operator_backends"]["fourier"]
    assert selected["revision"] == revision and selected["directions"] == [
        "forward",
        "inverse",
    ]
    ir.parent.rename(tmp_path / "hidden-ir")
    target = lower_scheduled(scheduled, tmp_path / "target")
    scheduled.parent.rename(tmp_path / "hidden-schedule")
    prefix.rename(tmp_path / "hidden-sdk")
    assert "missing pinned native Fourier SDK" in issues(valid)[0]["detail"]
    project = print_target(target, tmp_path / "delivery")
    target.parent.rename(tmp_path / "hidden-target")
    executable = build(project["cpp20_manifest"])
    ran = subprocess.run([executable], capture_output=True, check=True)
    result = [
        complex(*map(float.fromhex, row.split()))
        for row in ran.stdout.decode().splitlines()
    ]
    baseline = np.array([complex(*v) for v in samples], dtype=np.complex128)
    assert len(result) == 3 * n + 1
    for start, transform in ((0, np.fft.fft), (n, np.fft.ifft)):
        np.testing.assert_allclose(
            result[start : start + n],
            transform(baseline, norm="ortho"),
            rtol=3e-15,
            atol=3e-15,
        )
    np.testing.assert_array_equal(result[2 * n : 3 * n], baseline)
    assert np.isposinf(result[-1].real) and result[-1].imag == 0
    assert ran.stderr.decode().splitlines() == [
        "sealed/math:101: RuntimeWarning: overflow encountered in fft",
        "  transform(data)",
        "sealed/math:101: RuntimeWarning: invalid value encountered in fft",
        "  transform(data)",
    ]


def test_registry_diagnostic_preserves_observation_scope_and_actual_reads(tmp_path):
    from copy import deepcopy

    from scar.backend.planning import runtime, schedule_ir
    from scar.contract import digest, json_bytes, read_member

    environment = dict(
        semantics_version=1,
        entries=[dict(image="driver", declaration=dict(kind="standalone_script"))],
        analysis=dict(
            scope="analysis_process_snapshot",
            confidence="observed",
            deployment_equivalence="unproven",
            application_initial_cache="unproven",
            audit_hooks="unknown",
            interpreter=dict(
                implementation="cpython",
                version="3.12",
                executable="/capture/python",
                prefix="/capture",
                base_prefix="/capture",
                image=dict(path="/capture/python", size=42, sha256="a" * 64),
            ),
            flags={},
            search_path=[],
            meta_path=[],
            import_callable=dict(
                kind="builtin_import", type="builtin_function_or_method"
            ),
            selected_module_cache=dict(held="present_module", missing="absent"),
        ),
    )
    occurrence = dict(
        id="source/import",
        owner="main",
        site="import-site",
        activation="visited",
        effect_coverage="PARTIAL",
        obligations=["unresolved_initialization"],
        producer_sites=[],
    )
    ledger = dict(
        semantics_version=1,
        scope="source_import_occurrences",
        occurrences=[occurrence],
        entry_environment=environment,
    )
    reads = [
        dict(
            member=f"member{i}",
            analysis_binding_epoch=i,
            consumer="reader",
            site=f"read/{i}",
            location=dict(file="reader.py", line=i + 1),
        )
        for i in range(9)
    ]
    checks = dict(
        held=dict(
            before=dict(state="ready"),
            registry=dict(
                requires="same_module_instance_and_complete", first_consumer="entry"
            ),
            namespace=dict(
                requires="same_namespace_bindings_and_reachable_value_state",
                first_consumer="reader",
                first_member="member0",
                binding_reads=reads,
                value_state_projection="unproven",
            ),
        )
    )
    guard = dict(
        op="module_import_preserves",
        type="bool",
        operator_family="module_registry",
        occurrence=occurrence["id"],
        site=occurrence["site"],
        location=dict(file="driver.py", line=9),
        module="dep",
        checks=checks,
    )
    from scar.backend.planning import _qualify

    historical = dict(
        functions=[
            dict(
                name="main",
                body=[],
                source_evidence=guard,
                state_contract=dict(
                    discarded_type=dict(kind="ndarray", element="unsupported", rank=1)
                ),
            )
        ]
    )
    assert _qualify(historical) == ({}, {}, [])
    duplicate = dict(op="if", condition=guard, then=[], **{"else": []})
    fn = dict(
        name="main",
        params=[],
        result="i32",
        body=[
            dict(
                op="if",
                condition=guard,
                then=[duplicate],
                **{"else": []},
                source_evidence=dict(guard, occurrence="not-executed"),
            )
        ],
    )
    source, out = tmp_path / "sealed", tmp_path / "scheduled"
    write_json(
        source / "exec-ir.json",
        document(
            IR,
            status="READY",
            module=dict(
                entry="main", source_imports=member(source, "analysis", ledger)
            ),
            functions=[member(source, "functions", fn)],
        ),
    )
    schedule = schedule_ir(source, out, policy="serial")
    physical = json.loads((out / "physical-plan.json").read_bytes())
    assert (
        physical["status"]
        == json.loads(schedule.read_bytes())["status"]
        == "INCOMPLETE"
    )
    (issue,) = physical["issues"]
    assert (issue["code"], issue["requirement"]) == (
        "missing_operator_binding",
        "module_registry",
    )
    summary = issue["registry_checks"]
    assert (
        summary["guard_count"],
        summary["occurrence_count"],
        summary["guard_variant_count"],
    ) == (2, 1, 1)
    (row,) = summary["occurrences"]
    assert (row["owner"], row["occurrence"], row["site"], row["location"]) == (
        "main",
        occurrence["id"],
        guard["site"],
        guard["location"],
    )
    assert row["guard_copies"] == 2 and row["checks_sha256"] == digest(
        json_bytes(checks)
    )
    registry, namespace = sorted(row["checks"], key=lambda c: c["domain"], reverse=True)
    assert registry["requires"] == "same_module_instance_and_complete"
    assert namespace["requires"] == "same_namespace_bindings_and_reachable_value_state"
    assert namespace["value_state_projection"] == "unproven"
    limit = summary["limits"]["binding_reads_per_check"]
    assert (
        namespace["binding_reads"] == reads[:limit]
        and row["omitted_binding_reads"] == len(reads) - limit
    )
    audit = physical["source_import_effects"]
    assert read_member(out, audit["ledger"]) == ledger
    observation = audit["entry_environment"]
    assert observation["entry_environment_sha256"] == digest(json_bytes(environment))
    assert (
        observation["scope"] == "analysis_process_snapshot"
        and observation["confidence"] == "observed"
    )
    assert (
        observation["application_initial_cache"]
        == observation["deployment_equivalence"]
        == "unproven"
    )
    assert (
        observation["audit_hooks"] == "unknown"
        and observation["selected_module_cache_count"] == 2
    )
    assert audit["occurrences"][0]["complete_effect_equivalence"] == "unproven"
    # Older IR may contain an import occurrence whose source-visible loader
    # was omitted entirely. An otherwise executable body cannot discharge it.
    legacy = json.loads((source / "exec-ir.json").read_bytes())
    function = {
        **fn,
        "body": [dict(op="return", value=dict(op="literal", type="i32", value=0))],
    }
    legacy["functions"] = [member(source, "functions", function)]
    write_json(source / "exec-ir.json", legacy)
    unqualified = tmp_path / "unused-import"
    schedule_ir(source, unqualified, policy="serial")
    missing = json.loads((unqualified / "physical-plan.json").read_bytes())
    assert missing["status"] == "INCOMPLETE"
    assert [i["code"] for i in missing["issues"]] == ["unqualified_source_import"]
    assert (
        missing["issues"][0]["qualification"]["evidence"]["first_occurrence"]
        == occurrence["id"]
    )

    inactive = deepcopy(ledger)
    inactive["occurrences"][0]["activation"] = "proven_inactive"
    legacy["module"]["source_imports"] = member(source, "analysis", inactive)
    write_json(source / "exec-ir.json", legacy)
    closed = tmp_path / "inactive-import"
    schedule_ir(source, closed, policy="serial")
    assert json.loads((closed / "physical-plan.json").read_bytes())["status"] == "READY"

    corrupt = deepcopy(ledger)
    corrupt["entry_environment"]["analysis"]["application_initial_cache"] = "proven"
    with pytest.raises(ValueError, match="application premises"):
        runtime.import_effects(dict(functions=[], source_imports=corrupt), {})


def test_source_dispatch_seals_import_result_and_conditional_callee_for_b1(tmp_path):
    import shutil

    from scar.backend.planning import schedule_ir
    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle import plan

    entry, leaf = tmp_path / "entry.py", tmp_path / "leaf.py"
    entry.write_text("import leaf\nprint(leaf.fetch())\n")
    leaf.write_text("def fetch():\n    return 'result'\n")
    graph, execution = tmp_path / "graph", tmp_path / "execution"
    build(entry, graph)
    entry.unlink()
    leaf.unlink()
    assert plan(graph, execution)["status"] == "READY"
    shutil.rmtree(graph)
    (execution / "plan.json").unlink()
    path = execution / "exec-ir.json"
    ir = json.loads(path.read_bytes())
    index = read_member(execution, ir["module"]["source_dispatch"])
    (loader,), (callee,) = index["loaders"], index["callees"]
    assert "body" not in callee["function"]
    assert read_member(execution, callee["function"])["symbol"] == callee["symbol"]

    def inspect():
        out = tmp_path / "schedule"
        schedule_ir(execution, out, policy="serial")
        physical = json.loads((out / "physical-plan.json").read_bytes())
        assert physical["status"] == "INCOMPLETE"
        (reason,) = physical["issues"]
        assert reason["code"] == "unqualified_source_import"
        (occurrence,) = physical["source_import_effects"]["occurrences"]
        assert occurrence["complete_effect_equivalence"] == "unproven"
        assert (
            reason["qualification"]["evidence"]["source_dispatch"]
            == occurrence["source_dispatch"]
        )
        component = read_member(out, physical["source_frame_schedule"])
        [region] = read_member(out, component["regions"])
        assert occurrence["source_dispatch"]["import_binding_verified"]
        assert region["resume_offset"] == region["instructions"][-1][1] + 2
        if occurrence["source_dispatch"]["missing"]:
            assert [row[0] for row in region["instructions"]][-2:] == [
                "IMPORT_NAME",
                "STORE_NAME",
            ]
        else:
            assert region["instructions"][-1][0] == "POP_TOP"
        return occurrence["source_dispatch"]

    detail = inspect()
    assert detail["status"] == "modeled_source_path_requires_native_implementation"
    assert detail["loaders"] == [loader["symbol"]]
    assert detail["calls"][0]["callees"] == [callee["symbol"]]

    def module_program():
        out = tmp_path / "schedule"
        physical = json.loads((out / "physical-plan.json").read_bytes())
        schedule = read_member(out, physical["source_frame_schedule"])
        return next(
            b for b in read_member(out, schedule["bodies"]) if b["kind"] == "module"
        )

    assert module_program()["resume_offset"] is None
    exit_ = loader["entry"].pop("frame_exit")
    ir["module"]["source_dispatch"] = member(execution, "analysis", index)
    write_json(path, ir)
    inspect()
    prefix = module_program()
    assert prefix["instructions"][-1][0] == "STORE_NAME"
    assert prefix["resume_offset"] == prefix["instructions"][-1][1] + 2
    loader["entry"]["frame_exit"] = exit_

    (caller,) = index["functions"]
    original = caller["body"]
    body = read_member(execution, original)
    body["body"] = body["body"][:1]
    caller["body"] = member(execution, "functions", body)
    ir["module"]["source_dispatch"] = member(execution, "analysis", index)
    write_json(path, ir)
    assert inspect()["missing"] == ["conditional_callable_body_or_continuation"]

    caller["body"] = original
    index["callees"] = []
    ir["module"]["source_dispatch"] = member(execution, "analysis", index)
    write_json(path, ir)
    missing = inspect()
    assert missing["status"] == "missing_source_dataflow"
    assert missing["missing"] == ["conditional_callable_body_or_continuation"]
    (execution / loader["function"]["path"]).write_bytes(b"{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        inspect()


def test_nested_import_region_requires_a_matching_loader_frame(tmp_path):
    from scar.backend.planning import _load_module
    from scar.backend.planning.runtime import import_effects
    from scar.frontend import build
    from scar.middle import plan

    entry = tmp_path / "entry.py"
    entry.write_text("import outer\nprint(outer.answer())\n")
    (tmp_path / "outer.py").write_text(
        "import inner\ndef answer():\n    return inner.number()\n"
    )
    (tmp_path / "inner.py").write_text("def number():\n    return 42\n")
    graph, execution = tmp_path / "graph", tmp_path / "execution"
    build(entry, graph)
    plan(graph, execution)
    path = execution / "exec-ir.json"
    module = _load_module(path, json.loads(path.read_bytes()))
    dispatch = module["source_dispatch"]
    parent = next(row for row in dispatch["loaders"] if row["owner"] == "__main__")
    nested = next(
        row
        for row in module["source_imports"]["occurrences"]
        if row["owner"] == parent["symbol"]
    )

    # A source-prefix graph can retain the child body while only the outer
    # import is reached. The child's own ledger supplies no execution witness.
    dispatch["required_imports"] = [
        r for r in dispatch["required_imports"] if r["owner"] == "__main__"
    ]
    dispatch["completion"] = "source_prefix"
    nested["activation"] = "unvisited"

    def regions():
        effects = import_effects(module, {})
        occurrence = next(r for r in effects["occurrences"] if r["id"] == nested["id"])
        assert occurrence["activation"] == "unvisited"
        return occurrence.get("source_dispatch"), {
            r["owner"] for r in effects.get("native_frame_regions", ())
        }

    review, owners = regions()
    assert owners == {"__main__", parent["symbol"]}
    assert review["activation_scope"] == "matching_loader_frame_at_site"
    parent["entry"]["code"] = dict(parent["entry"]["code"], code_sha256="0" * 64)
    assert regions() == (None, set())


def test_conditional_frames_require_returns_on_both_arms_and_exhaustion(tmp_path):
    from scar.backend.planning.import_frames import _callee_return
    from scar.backend.planning.runtime import _body_returns_to_caller
    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle import plan

    entry, leaf = tmp_path / "entry.py", tmp_path / "leaf.py"
    entry.write_text("import leaf\nleaf.choose(input())\nleaf.scan(input())\n")
    leaf.write_text(
        "def choose(flag):\n    if flag == 'yes':\n        return 1\n"
        "    else:\n        return 2\n"
        "def scan(items):\n    for item in items:\n        print(item)\n"
        "    else:\n        return 'done'\n"
    )
    graph, execution = tmp_path / "graph", tmp_path / "ir"
    build(entry, graph)
    entry.unlink()
    leaf.unlink()
    plan(graph, execution)
    ir = json.loads((execution / "exec-ir.json").read_bytes())
    index = read_member(execution, ir["module"]["source_dispatch"])
    checked = set()
    for descriptor in index["callees"]:
        fragment = {
            **descriptor,
            "function": read_member(execution, descriptor["function"]),
        }
        [control] = fragment["function"]["body"]
        checked.add(control["op"])
        assert _body_returns_to_caller(fragment)
        assert not _callee_return(fragment)
        for arm in (
            ("then", "else") if control["op"] == "source_frame_if" else ("else",)
        ):
            suite, control[arm] = control[arm], []
            assert not _body_returns_to_caller(fragment)
            control[arm] = suite
    assert checked == {"source_frame_if", "source_frame_for"}


def test_native_import_frame_uses_live_registry_and_dispatches_returned_objects(
    tmp_path,
):
    from scar.backend import compile_ir
    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle import plan

    name, member_name = "source_dep", "describe"
    entry, leaf = tmp_path / "entry.py", tmp_path / f"{name}.py"
    entry.write_text(f"import {name}\nprint({name}.{member_name}())\n")
    leaf.write_text(
        "'module documentation'\ndef describe():\n    return 'value from sealed source'\n"
    )
    graph, execution, delivery = (tmp_path / n for n in ("graph", "ir", "delivery"))
    build(entry, graph)
    assert plan(graph, execution)["status"] == "READY"

    # Each original entry runs in a fresh interpreter. The declaring source
    # files and entry are the same across cold, warm and modified dispatch.
    startup = tmp_path / "startup"
    startup.mkdir()
    (startup / "sitecustomize.py").write_text(
        "import builtins, os, sys, types\n"
        f"name = {name!r}\n"
        "mode = os.environ['MODE']\n"
        "obj = types.ModuleType(name)\n"
        f"setattr(obj, {member_name!r}, lambda: 'replacement')\n"
        "if mode == 'hit': sys.modules[name] = obj\n"
        "if mode == 'hook':\n"
        "    orig = builtins.__import__\n"
        "    builtins.__import__ = lambda n,*a,**k: obj if n == name else orig(n,*a,**k)\n"
        "def audit(event, args):\n"
        "    if event == 'import' and args[0] == name:\n"
        "        if mode == 'deny': raise RuntimeError('denied')\n"
        "        if mode == 'audit_insert': sys.modules[name] = obj\n"
        "sys.addaudithook(audit)\n"
        "if mode in {'replace','failure','recursive'}:\n"
        "    from importlib.machinery import PathFinder\n"
        "    class Loader:\n"
        "        def create_module(self,spec): return None\n"
        "        def exec_module(self,module):\n"
        "            if mode == 'recursive': assert __import__(name) is module\n"
        "            self.original.exec_module(module)\n"
        "            if mode in {'replace','failure'}: sys.modules[name] = obj\n"
        "            if mode == 'failure': raise RuntimeError('exec failure')\n"
        "        def find_spec(self,fullname,path,target=None):\n"
        "            if fullname == name:\n"
        "                spec = PathFinder.find_spec(fullname,path)\n"
        "                self.original, spec.loader = spec.loader, self\n"
        "                return spec\n"
        "    sys.meta_path.insert(0, Loader())\n"
    )
    reference = {}
    for mode in (
        "cold",
        "hit",
        "deny",
        "hook",
        "audit_insert",
        "replace",
        "failure",
        "recursive",
    ):
        run = subprocess.run(
            [sys.executable, "-B", entry],
            capture_output=True,
            env={**os.environ, "PYTHONPATH": str(startup), "MODE": mode},
        )
        reference[mode] = (run.returncode, run.stdout)
    entry.unlink()
    leaf.unlink()
    result = compile_ir(execution, delivery, policy="serial")
    assert result["status"] == "INCOMPLETE" and "executable" not in result
    component = result["source_frame_components"]
    with pytest.raises(ValueError, match="already contains a generated project"):
        compile_ir(execution, delivery, policy="serial")
    ir = json.loads((execution / "exec-ir.json").read_bytes())
    index = read_member(execution, ir["module"]["source_dispatch"])
    (callee,) = index["callees"]
    constants = read_member(execution, callee["function"])["constants"]
    (literal,) = constants.values()
    assert literal["kind"] == "str"
    schedule = read_member(delivery, component["schedule"])
    bodies = read_member(delivery, schedule["bodies"])
    module, function = sorted(bodies, key=lambda b: b["kind"] != "module")
    assert all(body["resume_offset"] is None for body in bodies)
    instructions = module["instructions"]
    assert instructions[0][0] == "LOAD_CONST"
    assert module["constants"][str(instructions[0][2])] == {
        "kind": "str",
        "value": "module documentation",
    }
    assert instructions[1][0] == "STORE_NAME" and instructions[1][3] == "__doc__"
    assert (
        instructions[2][0] == "LOAD_CHILD_CODE"
        and instructions[2][3] == function["code_sha256"]
    )
    assert instructions[3][0] == "MAKE_FUNCTION"
    assert instructions[4][0] == "STORE_NAME" and instructions[4][3] == member_name
    assert any(value["kind"] == "NoneType" for value in module["constants"].values())
    assert function["constants"] == constants
    from scar.backend.planning.import_frames import _module_intervals

    loader = dict(index["loaders"][0])
    loader["function"] = read_member(execution, loader["function"])
    published = loader["function"]["body"][0]["value"]
    published["constant_value"]["value"] += " altered"
    assert all(
        steps[0][1] > instructions[1][1] for steps, _ in _module_intervals(loader, {})
    )
    source_code = read_member(execution, index["functions"][0]["body"])
    code = source_code["body"][0]["value"]["source_code"]["code_sha256"]
    import shutil

    manifest = read_member(delivery, component["manifest"])
    assert manifest["kind"] == "source_frame_library" and manifest["sources"] == []
    target, printed = tmp_path / "fresh-target", tmp_path / "fresh-print"
    shutil.rmtree(graph)
    shutil.rmtree(execution)
    env = {**os.environ, "PYTHONPATH": str(Path(scar.__file__).resolve().parents[1])}
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            _RUN_LAYER,
            "B2a",
            delivery / component["schedule"]["path"],
            target,
        ],
        env=env,
        check=True,
    )
    shutil.rmtree(delivery)
    subprocess.run(
        [sys.executable, "-B", "-c", _RUN_LAYER, "B2b", target, printed],
        env=env,
        check=True,
    )
    shutil.rmtree(target)
    source = Path(__file__).resolve().parents[1] / "fixtures" / "import_frame_host.cpp"
    native = tmp_path / "run"
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-I",
            str(printed / "project"),
            str(source),
            "-o",
            str(native),
        ],
        check=True,
    )

    states = dict(
        cold="1 1 1 1",
        hit="0 0 1 1",
        deny="1 0 0 0",
        hook="0 0 1 0",
        audit_insert="1 0 1 1",
        replace="1 1 1 1",
        failure="1 1 0 0",
        recursive="1 1 1 1",
        mismatch="0 0 0 0",
    )
    reference["mismatch"] = (2, b"")
    for mode, state in states.items():
        run = subprocess.run([native, mode, code, "replacement"], capture_output=True)
        assert (run.returncode, run.stdout) == reference[mode], (mode, run.stderr)
        assert run.stderr.decode().strip() == state


def test_module_subscription_interval_preserves_live_operands_and_failure_pc(tmp_path):
    from copy import deepcopy
    from scar.backend import compile_ir
    from scar.backend.cpp20.imports import _body_tables
    from scar.backend.planning.import_frames import _subscript_store
    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle import plan

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "child.py").write_text("pass\n")
    source = pkg / "__init__.py"
    source.write_text(
        "from . import child as old\ndel old\nbox = [None]\n"
        "box[0] = [1.0 / 4, 0.0, +0.5]\nbox[0] = 'next'\n"
    )
    source_code = compile(source.read_bytes(), str(source), "exec")
    entry = tmp_path / "main.py"
    entry.write_text("import pkg\n")
    graph, middle, physical = (tmp_path / n for n in ("graph", "middle", "physical"))
    build(entry, graph)
    plan(graph, middle)
    ir = json.loads((middle / "exec-ir.json").read_bytes())
    dispatch = read_member(middle, ir["module"]["source_dispatch"])
    loader = next(r for r in dispatch["loaders"] if r["symbol"] == "py:pkg:<init>")
    fn = read_member(middle, loader["function"])
    ops = [
        s["value"]
        for s in fn["body"]
        if s.get("value", {}).get("op") == "module_object_expression"
        and s["value"]["expression"]["steps"][-1]["op"] == "source_object_setitem"
    ]
    assert len(ops) == 2
    damaged = deepcopy(ops[0])
    write = damaged["expression"]["steps"][-1]
    write["value"], write["base"] = write["base"], write["value"]
    assert not _subscript_store(damaged, loader["entry"]["code"])
    source.unlink()
    (pkg / "child.py").unlink()
    entry.unlink()
    result = compile_ir(middle, physical, policy="serial")
    component = result["source_frame_components"]
    schedule = read_member(physical, component["schedule"])
    bodies = read_member(physical, schedule["bodies"])
    seq, scalar = [b for b in bodies if b.get("normal_successor")]
    steps = seq["instructions"]
    assert [s[0] for s in steps] == [
        "BUILD_LIST",
        "LOAD_FRAME_CONST",
        "LIST_EXTEND",
        "LOAD_NAME",
        "LOAD_FRAME_CONST" if sys.version_info.minor == 12 else "LOAD_SMALL_INT",
        "STORE_SUBSCR",
    ]
    assert scalar["instructions"][0][0] == "LOAD_FRAME_CONST"
    for row, op in zip((seq, scalar), ops, strict=True):
        assert row["normal_successor"] == op["source_expression"]["normal_successor"]
        assert row["resume_offset"] == row["normal_successor"]["offset"]
    bad = deepcopy(seq)
    bad["normal_successor"]["cache_offsets"][0] += 1
    with pytest.raises(ValueError, match="subscription continuation"):
        _body_tables([bad], [])
    bad = deepcopy(seq)
    bad["instructions"][0] = ["LOAD_NAME", steps[0][1], steps[3][2], steps[3][3]]
    with pytest.raises(ValueError, match="operand stack"):
        _body_tables([bad], [])
    items = source_code.co_consts[steps[1][2]]
    assert items == (0.25, 0.0, 0.5)

    src = tmp_path / "store.cpp"
    src.write_text(r"""
#define main import_host_main
#include "import_frame_host.cpp"
#undef main
#include <cassert>
struct live : activation {
  handle box=object_value("receiver"), tuple=object_value("tuple"), key=object_value("0"), rhs{};
  text mode{};
  std::vector<unsigned> visited{};
  unsigned error=0, resume=0;
  bool enter_body(const body_program &p) { return &active==&p && values.empty(); }
  bool admit_subscript_region(const body_program &) { return mode!="decline"; }
  bool before(unsigned pc) { visited.push_back(pc); return true; }
  handle build_list(std::span<slot> items) {
    assert(items.empty()); return rhs=object_value("fresh list");
  }
  void extend_list(const handle &list,const handle &values) {
    assert(list==rhs && values==tuple); list->attributes=values->attributes;
  }
  handle load_name(unsigned i) {
    assert(i==active.instructions[3].operand && rhs->attributes==tuple->attributes);
    if(mode=="lookup") throw std::runtime_error("lookup");
    return box;
  }
  handle small_int(unsigned i) { assert(i==0); return key; }
  void store_subscript(const handle &receiver,const handle &index,const handle &value) {
    assert(receiver==box && index==key && value==rhs && value!=tuple);
    if(mode=="before") throw std::runtime_error("before write");
    receiver->attributes[index->text]=value;
    if(mode=="after") throw std::runtime_error("after write");
  }
  void resume_at(unsigned pc) { resume=pc; }
  void unwind(unsigned pc,std::exception_ptr) { error=pc; }
};
int main(int argc,char **argv){
  assert(argc==6);
  body_provider provider{generated::bodies};
  const auto &p=provider.programs[std::stoul(argv[2])];const auto &a=p.instructions;
  live f{activation{p,provider,object_value("module"),object_value("None")}};
  f.mode=argv[1]; f.constants[a[1].operand]=f.tuple;
  if(a[4].operation==body_opcode::constant) f.constants[a[4].operand]=f.key;
  for(int i=3;i<argc;++i) f.tuple->attributes[std::to_string(i-3)]=object_value(argv[i]);
  auto status=provider.enter_at(f,f.mode=="frame" ? "" : p.code_sha256,p.symbol,
                                a.front().offset+(f.mode=="pc" ? 2 : 0));
  if(f.mode=="frame" || f.mode=="pc" || f.mode=="decline"){
    assert(status==outcome::unmatched && f.visited.empty() && !f.rhs);return 0;
  }
  assert(status==outcome::continued_in_host && bool(f.rhs));
  assert(f.box->attributes.contains("0")==(f.mode=="ok" || f.mode=="after"));
  if(f.mode=="ok") assert(f.values.empty() && f.resume==p.resume_offset && !f.error);
  else {
    assert(!f.resume && f.error==f.visited.back());
    assert(f.error==a[f.mode=="lookup" ? 3 : 5].offset);
    assert(f.values.size()==(f.mode=="lookup" ? 1u : 3u));
  }
  for(size_t i=0;i<f.visited.size();++i) assert(f.visited[i]==a[i].offset);
}
""")
    binary = tmp_path / "store"
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-I",
            str(Path(__file__).resolve().parents[1] / "fixtures"),
            "-I",
            str((physical / component["regions"]["path"]).parent),
            str(src),
            "-o",
            str(binary),
        ],
        check=True,
    )
    for mode in ("ok", "before", "after", "lookup", "frame", "pc", "decline"):
        subprocess.run(
            [binary, mode, str(bodies.index(seq)), *map(str, items)], check=True
        )


@pytest.mark.parametrize("dynamic", [False, True], ids=["pool", "runtime"])
def test_native_definition_uses_original_positional_defaults(tmp_path, dynamic):
    from copy import deepcopy

    from scar.backend import compile_ir
    from scar.backend.cpp20.imports import _body_tables
    from scar.backend.planning.import_frames import _module_intervals
    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle import plan

    entry, leaf = tmp_path / "run.py", tmp_path / "properties.py"
    entry.write_text("import properties\nprint(properties.label())\n")
    defaults = "x=seed, y=seed, z=None" if dynamic else "x=5, y=None"
    prefix = "seed = 'initial'\n" if dynamic else ""
    leaf.write_text(f"{prefix}def label({defaults}):\n    return 'value'\n")
    code = compile(leaf.read_bytes(), str(leaf), "exec")
    ns = {}
    exec(code, ns)
    graph, plan_dir, delivery = (tmp_path / n for n in ("graph", "ir", "delivery"))
    build(entry, graph)
    assert plan(graph, plan_dir)["status"] == "READY"
    ir = json.loads((plan_dir / "exec-ir.json").read_bytes())
    index = read_member(plan_dir, ir["module"]["source_dispatch"])
    (loader,) = index["loaders"]
    loader["function"] = read_member(plan_dir, loader["function"])
    def_index = int(dynamic)
    op = loader["function"]["body"][def_index]["value"]
    proof = op["source_definition"]
    if dynamic:
        assert ns["label"].__defaults__ == ("initial", "initial", None)
        seen = []

        class ChangingNames(dict):
            def __getitem__(self, name):
                if name == "seed":
                    seen.append(object())
                    return seen[-1]
                return super().__getitem__(name)

        local = ChangingNames()
        exec(code, {}, local)
        assert local["label"].__defaults__ == (*seen, None) and len(seen) == 2

        class FailingNames(ChangingNames):
            def __getitem__(self, name):
                if name == "seed" and seen:
                    raise RuntimeError("second lookup")
                return super().__getitem__(name)

        seen.clear()
        failing = FailingNames()
        with pytest.raises(RuntimeError, match="second lookup"):
            exec(code, {}, failing)
        assert len(seen) == 1 and dict(failing) == {"seed": "initial"}
        slot = proof["instructions"][2]["index"]
    else:
        slot = proof["positional_defaults"]["index"]
        assert code.co_consts[slot] is ns["label"].__defaults__
        assert code.co_consts[slot] == (5, None)
    bad_loader = deepcopy(loader)
    evaluation = bad_loader["function"]["body"][def_index]["value"]["evaluation"]
    if dynamic:
        evaluation[3]["items"][:2] = evaluation[3]["items"][:2][::-1]
    else:
        evaluation[0]["operand"] += 1
    assert all(
        row[0] != "MAKE_FUNCTION"
        for interval, _ in _module_intervals(bad_loader, {})
        for row in interval
    )

    entry.unlink()
    leaf.unlink()
    result = compile_ir(plan_dir, delivery, policy="serial")
    assert result["status"] == "INCOMPLETE"
    component = result["source_frame_components"]
    schedule = read_member(delivery, component["schedule"])
    (body,) = [
        r for r in read_member(delivery, schedule["bodies"]) if r["kind"] == "module"
    ]
    steps = body["instructions"][2 * int(dynamic) :]
    if dynamic:
        assert [s[0] for s in steps[:4]] == [
            "LOAD_NAME",
            "LOAD_NAME",
            "LOAD_FRAME_CONST",
            "BUILD_TUPLE",
        ]
        assert steps[3][1:] == [proof["positional_defaults"]["build_offset"], 3, ""]
        assert steps[2][2] == slot
    else:
        assert steps[0] == [
            "LOAD_DEFAULT_TUPLE",
            proof["instructions"][0]["offset"],
            slot,
            2,
        ]
    make = next(s for s in steps if s[0] == "MAKE_FUNCTION")
    after_make = steps[steps.index(make) + 1]
    if sys.version_info.minor == 12:
        assert make[2] == 1 and after_make[0] == "STORE_NAME"
        failure_pc = make[1]
    else:
        assert (
            make[2] == 0
            and after_make[0] == "SET_FUNCTION_ATTRIBUTE"
            and after_make[2:] == [1, "defaults"]
        )
        failure_pc = after_make[1]
    bad = deepcopy(body)
    bad["minor"] = 14 if body["minor"] == 12 else 12
    with pytest.raises(ValueError, match="function publication"):
        _body_tables([bad], [])
    later = deepcopy(body)
    shift = steps[-1][1] + 2
    for ins in later["instructions"]:
        ins[1] += shift
    later["entry_offset"] = later["instructions"][0][1]
    _body_tables([body, later], [])
    with pytest.raises(ValueError, match="overlapping"):
        _body_tables([body, later, later], [])
    later["entry_offset"] += 2
    with pytest.raises(ValueError, match="instruction interval"):
        _body_tables([later], [])
    bad = deepcopy(body)
    if dynamic:
        bad["instructions"][5][2] -= 1
        with pytest.raises(ValueError, match="stack for function defaults"):
            _body_tables([bad], [])
    else:
        bad["instructions"][0][2] = steps[1][2]
        with pytest.raises(ValueError, match="constant slot"):
            _body_tables([bad], [])

    # The fixture supplies the active code and pool. The typed program supplies
    # the operand index, attachment opcode, publication target and failure PC.
    src = tmp_path / "defaults.cpp"
    src.write_text(r"""
#define main import_host_main
#include "import_frame_host.cpp"
#undef main
#include <cassert>
struct defaults_frame : activation {
  handle tuple = object_value("tuple"), created{};
  text mode{};
  unsigned error = 0;
  std::vector<handle> seen{};
  handle load_name(unsigned index) {
    assert(index == 0 && module->attributes.contains("seed"));
    if (mode == "fail_lookup" && !seen.empty()) throw std::runtime_error("lookup");
    seen.push_back(object_value("current binding"));
    return seen.back();
  }
  handle build_tuple(std::span<slot> items) {
    assert(items.size() == 3 && seen.size() == 2 && seen[0] != seen[1]);
    assert(items[0] == seen[0] && items[1] == seen[1] && items[2] == none);
    if (mode == "fail_build") throw std::runtime_error("allocation");
    tuple = object_value("tuple");
    for (unsigned i = 0; i < items.size(); ++i)
      tuple->attributes[std::to_string(i)] = *items[i];
    return tuple;
  }
  bool matches_defaults_tuple(const handle &v, unsigned arity) {
    return mode != "wrong_tuple" && v == tuple && arity == 2;
  }
  handle make_function(const handle &code) {
    if (mode == "fail_make") throw std::runtime_error("creation");
    return created = activation::make_function(code);
  }
  void set_function_defaults(const handle &fn, const handle &v) {
    assert(fn == created && v == tuple && !module->attributes.contains("label"));
    if (mode == "fail") throw std::runtime_error("defaults");
    fn->attributes["__defaults__"] = v;
  }
  handle make_function(const handle &code, const handle &v) {
    auto fn = make_function(code);
    set_function_defaults(fn, v);
    return fn;
  }
  void unwind(unsigned pc, std::exception_ptr) { error = pc; }
};
struct without_tuple : defaults_frame {
  handle build_tuple(std::span<slot>) = delete;
};
int main(int argc, char **argv) {
  assert(argc == 4);
  const body_provider provider{generated::bodies};
  const auto &body = provider.programs.front(), &child = provider.programs.back();
  auto module = object_value("module"), none = object_value("None");
  defaults_frame f{activation{body, provider, module, none}};
  f.mode = argv[1];
  const bool dynamic = body.instructions.front().operation == body_opcode::constant;
  f.constants[std::stoul(argv[2])] = dynamic ? none : f.tuple;
  auto result = run_body(f, f.mode == "wrong_frame" ? child : body);
  if (f.mode == "ok") {
    assert(result == outcome::completed && module->attributes.at("label") == f.created);
    assert(f.created->attributes.at("__defaults__") == f.tuple && f.returned == none);
    if (dynamic) {
      std::weak_ptr<value> first = f.seen[0], second = f.seen[1];
      f.seen.clear();
      assert(first.lock() == f.tuple->attributes.at("0") && second.lock() == f.tuple->attributes.at("1"));
    }
  } else {
    if (f.mode == "fail_lookup")
      assert(!f.created && f.values.size() == 1 && f.values[0] == f.seen[0]);
    if (f.mode == "fail_build")
      assert(!f.created && f.values.size() == 3 && f.values[1] == f.seen[1]);
    assert(!module->attributes.contains("label") && !f.returned);
    assert(module->attributes.contains("seed") == (dynamic && f.mode != "wrong_frame"));
    assert(f.error == std::stoul(argv[3]));
    assert(result == (f.mode == "wrong_frame" ? outcome::unmatched : outcome::continued_in_host));
  }
  if (dynamic) {
    without_tuple missing{defaults_frame{activation{body, provider, object_value("empty"), none}}};
    assert(run_body(missing, body) == outcome::unmatched);
    assert(missing.seen.empty() && missing.module->attributes.empty() && missing.values.empty());
  }
  activation legacy{body, provider, object_value("empty"), none};
  assert(run_body(legacy, body) == outcome::unmatched && legacy.values.empty() && legacy.module->attributes.empty());
}
""")
    built = tmp_path / "defaults"
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-I",
            str(Path(__file__).resolve().parents[1] / "fixtures"),
            "-I",
            str((delivery / component["regions"]["path"]).parent),
            str(src),
            "-o",
            str(built),
        ],
        check=True,
    )
    for mode, error_offset in [
        ("ok", 0),
        ("fail", failure_pc),
        ("fail_make", make[1]),
        ("wrong_frame", 0),
    ] + (
        [("fail_lookup", steps[1][1]), ("fail_build", steps[3][1])]
        if dynamic
        else [("wrong_tuple", 0)]
    ):
        subprocess.run([built, mode, str(slot), str(error_offset)], check=True)


def test_ordered_mapping_views_and_fresh_construction_cross_native_layers(tmp_path):
    from scar.backend.planning import _qualify
    from scar.contract import MAPPING_PROFILE, digest, json_bytes

    m, s, view, iterator = [
        dict(kind=k, profile=MAPPING_PROFILE)
        for k in (
            "mapping_ref",
            "structured_value",
            "mapping_items_ref",
            "mapping_iterator_ref",
        )
    ]
    pair = dict(kind="product_ref", items=["text", s])
    profile = dict(
        interpreter="fixture",
        unicode_version="fixture",
        nonprintable=[[0, 31], [127, 160]],
    )
    profile_id = digest(json_bytes(profile))

    def op(name, result, **ports):
        effects = ["STATE"]
        if name in (
            "mapping_make",
            "mapping_put",
            "mapping_merge",
            "mapping_items",
            "mapping_iterator",
            "mapping_next",
        ):
            effects = ["MEM", "STATE"]
        elif name in ("structured_pack", "structured_compare"):
            effects = ["VAL", "STATE"]
        raises = name not in (
            "mapping_current",
            "mapping_size",
            "structured_truth",
            "structured_compare",
            "structured_is_mapping",
            "mapping_identity",
        )
        return dict(
            op=name,
            type=result,
            profile=MAPPING_PROFILE,
            operator_family="builtin_scalar_mapping",
            semantics_version=1,
            effects=effects,
            may_raise=raises,
            **ports,
        )

    def lit(value, t="text"):
        return dict(op="literal", type=t, value=value)

    def var(name, t):
        return dict(op="var", name=name, type=t)

    def decl(name, init):
        return dict(op="decl", name=name, type=init["type"], init=init)

    def expr(value):
        return dict(op="expr", value=value)

    def pack(v):
        return op(
            "structured_pack",
            s,
            tag="mapping_ref" if v["type"] == m else v["type"],
            value=v,
        )

    def make(*entries):
        return op(
            "mapping_make",
            m,
            entries=[dict(key=lit(k), value=pack(v)) for k, v in entries],
        )

    def put(base, key, value):
        return expr(
            op(
                "mapping_put",
                "unit",
                base=base,
                key=lit(key),
                value=pack(value),
                acyclicity=dict(
                    semantics_version=1, rule="nonmapping_scalar", tag=value["type"]
                ),
            )
        )

    def get(base, key):
        return op("mapping_get", s, base=base, key=lit(key), text_profile=profile_id)

    def advance(base):
        return op("mapping_next", "bool", base=base)

    def current(base):
        return op(
            "mapping_current",
            pair,
            base=base,
            requires=["successful_mapping_next_for_this_iterator"],
        )

    def field(base, index):
        return dict(
            op="product_get",
            base=base,
            type=pair["items"][index],
            index=index,
            effects=["STATE"],
        )

    def equals(left, right):
        if left["type"] == s:
            return op(
                "structured_compare", "bool", operator="eq", left=left, right=right
            )
        return dict(op="binary", operator="eq", left=left, right=right, type="bool")

    def check(condition):
        return dict(
            op="if",
            condition=dict(op="unary", operator="not", value=condition),
            then=[dict(op="return", value=lit(1, "i32"))],
            **{"else": []},
        )

    def catches(name, arg, value, category):
        return dict(
            name=name,
            params=[dict(name=arg["name"], type=arg["type"])],
            result="bool",
            body=[
                dict(
                    op="try",
                    body=[expr(value)],
                    handlers=[
                        dict(
                            categories=[category],
                            binding=None,
                            body=[dict(op="return", value=lit(True, "bool"))],
                        )
                    ],
                    **{"else": [], "finally": []},
                ),
                dict(op="return", value=lit(False, "bool")),
            ],
        )

    def caught(name, arg):
        return dict(op="call", callee=name, args=[arg], type="bool")

    d, child = var("d", m), var("child", m)
    cursor = var("cursor", iterator)
    fresh = dict(semantics_version=1, rule="unpublished_fresh_builder")
    body = [
        decl("child", make()),
        decl(
            "d",
            make(("b", lit("first")), ("a", lit("second")), ("b", lit("overwritten"))),
        ),
        dict(
            op="if",
            condition=lit(True, "bool"),
            then=[
                expr(
                    op(
                        "mapping_merge",
                        "unit",
                        base=d,
                        source=make(("nested", child)),
                        acyclicity=fresh,
                    )
                )
            ],
            **{"else": []},
        ),
        decl(
            "alias",
            op("structured_as_mapping", m, value=get(d, "nested"), access="subscript"),
        ),
        put(var("alias", m), "leaf", lit("same")),
        check(
            op(
                "mapping_identity",
                "bool",
                left=child,
                right=var("alias", m),
                operator="is",
            )
        ),
        check(equals(get(child, "leaf"), lit("same"))),
        decl("view", op("mapping_items", view, base=d)),
        put(d, "late", lit("before iteration")),
        decl("cursor", op("mapping_iterator", iterator, value=var("view", view))),
        check(advance(cursor)),
        decl("held", current(cursor)),
        put(d, "b", lit("changed")),
        check(equals(field(var("held", pair), 0), lit("b"))),
        check(equals(field(var("held", pair), 1), lit("overwritten"))),
        put(d, "a", lit("new second")),
        check(advance(cursor)),
        check(equals(field(current(cursor), 0), lit("a"))),
        check(equals(field(current(cursor), 1), lit("new second"))),
        put(d, "insert", lit(9007199254740993, "i64")),
        check(caught("changed_size", cursor)),
        check(caught("missing_key", d)),
        check(op("structured_truth", "bool", value=get(d, "insert"))),
        check(op("structured_is_mapping", "bool", value=get(d, "nested"))),
        check(equals(op("mapping_size", "i64", base=d), lit(5, "i64"))),
        dict(op="return", value=lit(0, "i32")),
    ]
    functions = [
        dict(name="main", params=[], result="i32", body=body),
        catches("changed_size", cursor, advance(cursor), "runtime_error"),
        catches("missing_key", d, get(d, "\n\u00a0"), "key_error"),
    ]
    module = dict(entry="main", externs=[], text_profiles={profile_id: profile})
    forged = op(
        "mapping_put", "unit", base=d, key=lit("self"), value=pack(d), acyclicity=fresh
    )
    for unsafe_body in (
        [decl("d", make()), expr(forged)],
        [
            decl("d", make()),
            decl("alias", d),
            expr(op("mapping_merge", "unit", base=d, source=make(), acyclicity=fresh)),
        ],
    ):
        _, _, issues = _qualify(
            dict(module, functions=[dict(name="main", body=unsafe_body)])
        )
        assert (
            issues[0]["code"] == "missing_operator_binding"
            and "unpublished fresh builder" in issues[0]["detail"]
        )
    source = tmp_path / "exec-ir.json"
    write_json(
        source,
        document(
            IR,
            status="READY",
            module=module,
            functions=[member(tmp_path, "functions", fn) for fn in functions],
        ),
    )
    for stage, filename in [
        ("B1", "exec-ir.json"),
        ("B2a", "scheduled-ir.json"),
        ("B2b", "target-ir.json"),
        ("B3", "cpp20.json"),
    ]:
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                _RUN_LAYER,
                stage,
                tmp_path / filename,
                tmp_path,
                *(["1"] if stage == "B2a" else []),
            ],
            env=dict(
                os.environ, PYTHONPATH=str(Path(scar.__file__).resolve().parents[1])
            ),
            check=True,
            timeout=90,
        )
    subprocess.run([tmp_path / "project/bin/scar_entry"], check=True)
    assert (tmp_path / "project/include/scar/mapping.hpp").is_file()


def test_b1_qualification_distinguishes_contract_profile_platform_and_open_effects(
    tmp_path, monkeypatch
):
    from scar.backend.planning import schedule_ir
    from scar.contract import read_member

    monkeypatch.setenv("SCAR_CPU_SDK", str(tmp_path / "absent"))
    ty = {"kind": "ndarray", "element": "f64", "rank": 1}
    size = dict(
        op="array_size",
        type="i64",
        value=dict(op="var", name="a", type=ty),
        site="owner/7",
    )
    registry = dict(
        op="module_import_preserves",
        occurrence="import/2",
        site="owner/8",
        type="bool",
        operator_family="module_registry",
        checks={
            "dep": {
                "namespace": {
                    "requires": "same_namespace_bindings_and_reachable_value_state",
                    "binding_reads": [{"member": "distance", "site": "owner/9"}],
                }
            }
        },
    )
    continuation = dict(
        op="module_export_continuation_required",
        type="void",
        operator_family="module_registry",
        member="distance",
        site="owner/10",
        reads=["source.continuation"],
        semantics="resume_after_failed_match",
        external_effect="UNKNOWN",
    )
    cases = [
        (
            [
                dict(
                    size,
                    value=dict(
                        size["value"], type={"kind": "ndarray", "element": "f64"}
                    ),
                )
            ],
            "ndarray",
            "semantic_contract",
            "ndarray.array_type",
        ),
        (
            [dict(size, value=dict(size["value"], type=dict(ty, element="f32")))],
            "ndarray",
            "implementation_profile",
            "ndarray.registered_elements",
        ),
        ([size], "ndarray", "target_platform", "ndarray.sdk_identity"),
        (
            [dict(op="example_call", operator_family="unregistered", site="owner/6")],
            "unregistered",
            "implementation_profile",
            "operator_family.registered",
        ),
        (
            [registry, continuation],
            "module_registry",
            "external_effects",
            "module_registry.observable_semantics",
        ),
    ]
    for i, (ops, family, stage, rule) in enumerate(cases):
        root = tmp_path / str(i)
        ir = root / "exec-ir.json"
        fn = dict(
            name="main",
            params=[],
            result="i32",
            body=[dict(op="expr", value=op) for op in ops],
        )
        # An annotation mentioning a provider cannot create a native requirement.
        fn["source_evidence"] = {"op": "omitted", "operator_family": "historical"}
        write_json(
            ir,
            document(
                IR,
                status="READY",
                module={"entry": "main"},
                functions=[member(root, "functions", fn)],
            ),
        )
        out = root / "schedule"
        scheduled = json.loads(schedule_ir(ir, out, policy="serial").read_bytes())
        physical = read_member(out, scheduled["physical"])
        assert scheduled["status"] == physical["status"] == "INCOMPLETE"
        assert scheduled["issues"] == physical["issues"]
        (issue,) = physical["issues"]
        assert (issue["code"], issue["requirement"]) == (
            "missing_operator_binding",
            family,
        )
        decision = issue["qualification"]
        assert (decision["stage"], decision["rule"]) == (stage, rule)
        evidence = decision["evidence"]
        if rule == "ndarray.array_type":
            assert evidence["site"] == "owner/7" and "rank" not in evidence["observed"]
        elif rule == "ndarray.registered_elements":
            assert evidence["elements"] == ["f32"]
        elif rule == "ndarray.sdk_identity":
            assert evidence["provider"] == "xtensor-cpu"
        elif rule == "operator_family.registered":
            assert evidence["observed_operations"] == ["example_call"]
        else:
            checks = issue[evidence["checks_field"]]
            assert checks["occurrences"][0]["checks"][0]["binding_reads"] == [
                {"member": "distance", "site": "owner/9"}
            ]
            assert (
                checks["operator_examples"][0]["semantics"]
                == "resume_after_failed_match"
            )
            assert evidence["required_read_domains"] == {"source.continuation": 1}


def test_registry_requirement_covers_untagged_consumers_and_continuations():
    from scar.backend.planning import _observed_operators, runtime
    from scar.contract import MODULE_REGISTRY_OPERATIONS

    ops = [dict(op=name) for name in sorted(MODULE_REGISTRY_OPERATIONS)]
    module = {"functions": [{"name": "entry", "body": ops, "source_evidence": ops}]}
    assert (
        _observed_operators(module)[0]["module_registry"] == MODULE_REGISTRY_OPERATIONS
    )
    assert runtime.registry_checks(module)["operator_counts"] == {
        n: 1 for n in MODULE_REGISTRY_OPERATIONS
    }


@pytest.mark.parametrize("claimed_binding", [False, True])
def test_b1_checks_fromlist_publication_and_deletion_without_family_tags(
    tmp_path, claimed_binding
):
    from scar.backend import compile_ir
    from scar.contract import read_member

    frame = dict(
        owner="py:pkg:<init>", module="pkg", identity="current_activation_globals"
    )
    effects = dict(
        type="unit", frame_globals=frame, may_raise=True, external_effect="UNKNOWN"
    )
    publish = dict(
        op="module_fromlist_publish",
        occurrence="pkg/1",
        site="pkg/1",
        request=dict(form="from", module="", level=1, fromlist=["child"]),
        bindings=[
            dict(binding_index=0, member="child", name="child", slot="py:pkg:child")
        ],
        reads=["frame.globals", "frame.builtins", "module.registry"],
        **effects,
    )
    delete = dict(
        op="module_global_delete",
        site="pkg/2",
        names=["child"],
        reads=["frame.globals"],
        **effects,
    )
    function = dict(
        name=frame["owner"],
        params=[],
        result="void",
        body=[dict(op="expr", value=publish), dict(op="expr", value=delete)],
        source_evidence=dict(prior=[publish, delete]),
    )
    module = {"entry": frame["owner"]}
    if claimed_binding:
        module["operator_requirements"] = {
            "module_registry": {"binding": "native", "operations": []}
        }
    write_json(
        tmp_path / "exec-ir.json",
        document(
            IR,
            status="READY",
            module=module,
            functions=[member(tmp_path, "functions", function)],
        ),
    )
    out = tmp_path / "delivery"
    result = compile_ir(tmp_path, out, policy="serial")
    scheduled = json.loads(Path(result["scheduled_ir"]).read_bytes())
    physical = read_member(out, scheduled["physical"])
    assert result["status"] == scheduled["status"] == physical["status"] == "INCOMPLETE"
    assert result["issues"] == scheduled["issues"] == physical["issues"]
    [issue] = result["issues"]
    assert (issue["stage"], issue["code"], issue["requirement"]) == (
        "B1",
        "missing_operator_binding",
        "module_registry",
    )
    qualification = issue["qualification"]
    assert qualification["rule"] == "module_registry.observable_semantics"
    assert qualification["evidence"]["observed_operations"] == [
        publish["op"],
        delete["op"],
    ]
    checks = issue["registry_checks"]
    assert checks["operator_counts"] == {publish["op"]: 1, delete["op"]: 1}
    assert {(s["operation"], s["site"]) for s in checks["operator_examples"]} == {
        (publish["op"], publish["site"]),
        (delete["op"], delete["site"]),
    }
    assert not (out / "target-ir.json").exists()


def test_rounded_f64_iota_matches_numpy_allocation_and_index_errors(tmp_path):
    from scar.host import array_sdk

    # Keep even the near-address-space inputs deterministic on overcommitting hosts.
    stops = [
        0,
        1,
        5,
        1.25,
        131006464,
        134152192,
        2**53 + 2,
        2**60 - 128,
        2**60,
        2**63 - 1024,
        2**63,
        2**63 + 2048,
        float("inf"),
    ]
    args = [str(x) for x in stops]
    source = tmp_path / "iota.cpp"
    source.write_text(r"""
#include "array.hpp"
#include <cstdlib>
#include <iostream>
#include <sys/resource.h>

int main(int argc, char **argv) {
  const rlimit limit{256 << 20, 256 << 20};
  if (setrlimit(RLIMIT_AS, &limit)) return 1;
  for (int i = 2; i < argc; ++i) {
    try {
      auto a = scar::array_iota_rounded<double>(std::strtod(argv[i], nullptr),
                                                 argv[1][0] == '1');
      for (std::int64_t j = 0; j < a.size(1); ++j)
        std::cout << a.load(1, {j}) << ' ';
      std::cout << a.load(1, {-1}) << '\n';
    } catch (...) {
      auto error = scar::capture_error();
      std::cout << error->category << ' ' << error->message << '\n';
    }
  }
}
""")
    binary = tmp_path / "iota"
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-Werror",
            "-O2",
            "-I",
            Path(scar.__file__).parent / "backend/cpp20",
            "-I",
            Path(array_sdk(required=True)["prefix"]) / "include",
            source,
            "-o",
            binary,
        ],
        check=True,
    )
    reference = subprocess.check_output(
        [
            sys.executable,
            "-c",
            """
import numpy as np, resource, sys
resource.setrlimit(resource.RLIMIT_AS, (256 << 20, 256 << 20))
for stop in sys.argv[1:]:
    try:
        a = np.arange(float(stop), dtype=np.float64)
        print(*(int(x) for x in a), int(a[-1]))
    except (IndexError, MemoryError, ValueError) as e:
        category = {IndexError: "index_error", MemoryError: "memory_error",
                    ValueError: "value_error"}
        print(next(v for k, v in category.items() if isinstance(e, k)), e)
""",
            *args,
        ],
        text=True,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1"),
    ).splitlines()
    at_limit = reference[stops.index(2**63)]
    empty = "index_error index -1 is out of bounds for axis 0 with size 0"
    rejected = "value_error Maximum allowed size exceeded"
    assert at_limit in (empty, rejected)
    endpoint_empty = str(int(at_limit == empty))
    assert (
        subprocess.check_output([binary, endpoint_empty, *args], text=True).splitlines()
        == reference
    )
    assert (
        subprocess.check_output([binary, "0", str(2**63)], text=True).strip()
        == rejected
    )
    assert (
        subprocess.check_output([binary, "1", str(2**63)], text=True).strip() == empty
    )


def test_integer_array_index_width_preserves_lp64_exception_categories(tmp_path):
    from scar.backend.cpp20.lowering import Lowerer
    from scar.backend.cpp20.printer import _expression
    from scar.backend.planning import integer
    from scar.contract import digest, json_bytes

    argument = dict(op="var", name="index", type="integer")
    messages = dict(index_error="outside index domain", overflow_error="unsigned index")
    node = dict(
        op="integer_op",
        operation="array_index_width",
        operator_family="integer",
        semantics_version=1,
        type="i64",
        args=[argument],
        effects=["VAL"],
        may_raise=True,
        reads=[],
        writes=[],
        exception_messages=messages,
    )
    requirement = dict(semantics_version=1, operations=["array_index_width"])
    for field in messages:
        incomplete = {k: v for k, v in messages.items() if k != field}
        with pytest.raises(
            ValueError, match="messages for each possible source exception"
        ):
            integer.qualify(requirement, [dict(node, exception_messages=incomplete)])
    binding = integer.qualify(requirement, [node])
    module = dict(functions=[], integer_binding_sha256=digest(json_bytes(binding)))
    lower = Lowerer(module, dict(operator_backends=dict(integer=binding)))
    call = _expression(lower.expression(node))
    index = _expression(lower.expression(argument))
    source, binary = tmp_path / "width.cpp", tmp_path / "width"
    source.write_text(f"""
#include "integer.hpp"
#include <iostream>
int main(int argc, char **argv) {{
  for (int i = 1; i < argc; ++i) {{
    try {{
      scar::integers::value {index}(argv[i], 10);
      std::cout << {call} << '\\n';
    }} catch (const scar::program_error &e) {{
      std::cout << e.category() << " " << e.message() << '\\n';
    }}
  }}
}}
""")
    files = binding["files"]
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-Werror",
            "-I",
            Path(scar.__file__).parent / "backend/cpp20",
            "-I",
            Path(files["include/gmp.h"]["path"]).parent,
            source,
            files["lib/libgmp.a"]["path"],
            "-o",
            binary,
        ],
        check=True,
    )
    index_error = "index_error " + messages["index_error"]
    overflow_error = "overflow_error " + messages["overflow_error"]
    cases = {
        -(2**100): index_error,
        -(2**63) - 1: index_error,
        -(2**63): str(-(2**63)),
        -1: "-1",
        0: "0",
        2**63 - 1: str(2**63 - 1),
        2**63: overflow_error,
        2**64 - 1: overflow_error,
        2**64: index_error,
        2**100: index_error,
    }
    received = subprocess.check_output([binary, *map(str, cases)], text=True)
    assert received.splitlines() == list(cases.values())


def test_private_buffer_placement_respects_proof_budget_and_heap_fallback(
    tmp_path, monkeypatch
):
    from scar.backend.planning import cpu
    from scar.backend.cpp20.lowering import Lowerer
    from scar.backend.cpp20.printer import statement

    def buffer(name, proof=True):
        value = dict(
            op="buffer", name=name, type="f64", extent=dict(op="var", name="n")
        )
        if proof:
            value["private_scratch"] = "indexed_scalar_access_only"
        return value

    fn = dict(
        name="worker",
        result="void",
        params=[dict(name="n", type="i64")],
        body=[buffer("x"), buffer("y"), buffer("escaped", False), buffer("z")],
    )
    module = dict(functions=[fn], globals=[], resource_model={"resources": []})
    monkeypatch.setattr(cpu, "_scratch_geometry", lambda: None)
    fallback, _ = cpu.schedule(module, "serial")
    assert all("inline_capacity" not in op for op in fallback["functions"][0]["body"])
    monkeypatch.setattr(cpu, "_scratch_geometry", lambda: (32, 64))
    scheduled, physical = cpu.schedule(module, "serial")
    ops = scheduled["functions"][0]["body"]
    assert [s.get("inline_capacity") for s in ops] == [4, 4, None, None]
    assert {a["name"] for a in physical["allocations"]} == {"x", "y"}
    assert all("inline_capacity" not in op for op in fn["body"])
    owned, owned_plan = cpu.schedule(module, "serial", consume_functions=True)
    assert owned == scheduled and owned_plan == physical
    assert owned["functions"][0] is fn
    owned["resource_model"] = {"path": "sealed-inventory"}
    assert module["resource_model"] == {"resources": []}
    lower = Lowerer(scheduled, physical)
    rendered = "\n".join(statement(s) for s in lower.statement(ops[0]))
    assert "scar::small_buffer" in rendered
    assert "new " in "\n".join(statement(s) for s in lower.statement(ops[2]))
    source = tmp_path / "buffer.cpp"
    source.write_text(
        """
#include "runtime.hpp"
#include <cstdint>
#include <cstdlib>
#include <new>
static int allocations = 0;
void *operator new[](std::size_t n) {
  ++allocations;
  if (auto p=std::malloc(n)) return p;
  throw std::bad_alloc();
}
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void check(std::int64_t v_n) {
"""
        + rendered
        + """
  if (v_n>0) {
    v_x[v_n-1]=2.75;
    if(v_x[v_n-1]!=2.75) std::abort();
  }
}
int main(){
  check(0); check(4);
  if(allocations!=0) return 1;
  check(5);
  if(allocations!=1) return 2;
  try { check(-1); } catch(const std::bad_array_new_length&){ return 0; }
  return 3;
}
"""
    )
    binary = tmp_path / "buffer"
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-Werror",
            "-I",
            str(Path(scar.__file__).parent / "backend/cpp20"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)


def test_project_import_entry_binds_its_domain_and_rolls_back_failed_load(tmp_path):
    from copy import deepcopy

    from scar.backend.cpp20 import lower_scheduled, print_target
    from scar.backend.planning import schedule_ir
    from scar.contract import read_member
    from scar.frontend import build
    from scar.middle import plan

    source = tmp_path / "src"
    source.mkdir()
    (source / "entry.py").write_text("import leaf\nprint(leaf.value())\n")
    (source / "leaf.py").write_text("def value():\n    return 'sealed'\n")
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            dict(
                source_root="src",
                source="entry.py",
                entry_environment=dict(
                    kind="standalone_script", deployment="isolated_project_modules_v1"
                ),
            )
        )
    )
    graph, execution, schedule = (
        tmp_path / k for k in ("graph", "execution", "schedule")
    )
    build(request, graph)
    assert plan(graph, execution)["status"] == "READY"
    package = schedule_ir(execution, schedule, policy="serial")
    sealed = json.loads(package.read_bytes())
    physical = read_member(schedule, sealed["physical"])
    assert sealed["status"] == "READY"
    target = lower_scheduled(package, tmp_path / "target")
    project = print_target(target, tmp_path / "delivery")
    probe = tmp_path / "verify.cpp"
    probe.write_text("""#include "scar/program.hpp"
using namespace scar::imports;
int main() {
  isolated::process domain(generated::modules);
  auto key = generated::modules.front().name;
  auto &region = generated::regions.front();
  try {
    default_absolute(domain, key, region.minor, {});
    return 1;
  } catch (const std::logic_error &) {}
  if (domain.lookup_module(key)) return 2;
  auto cold = default_absolute(domain, key, region.minor, region.bodies);
  auto hit = default_absolute(domain, key, region.minor, {});
  return cold == hit && !hit->initializing && hit->members.contains("value") ? 0 : 3;
}
""")
    binary = tmp_path / "verify"
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-Werror",
            "-I",
            Path(project["project"]) / "include",
            probe,
            "-o",
            binary,
        ],
        check=True,
    )
    subprocess.run([binary], check=True)
    for change in ("missing", "altered"):
        forged = deepcopy(physical)
        effects = forged["source_import_effects"]
        if change == "missing":
            del effects["isolated_entry"]
        else:
            effects["isolated_entry"]["modules"][0]["name"] = "different"
        sealed["physical"] = member(schedule, "scheduled-units", forged)
        write_json(package, sealed)
        with pytest.raises(ValueError, match="sealed execution domain"):
            lower_scheduled(package, tmp_path / change)

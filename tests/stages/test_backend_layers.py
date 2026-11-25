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
            ("scar.backend.planning", "scar.backend.cpp20.printer", "scar.backend.build")),
    "B2b": ("scar.backend.cpp20", "print_target",
            ("scar.backend.planning", "scar.backend.cpp20.lowering", "scar.backend.cpp20.objects",
             "scar.backend.cpp20.libraries", "scar.backend.build")),
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
    implementation.__file__ = str(Path(output) / "absent-sdk" / "__init__.py")
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
    from scar.backend.planning import _qualify

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


def test_file_graph_and_workflow_members_survive_schedule_relocation(
    tmp_path, monkeypatch
):
    from scar.backend.planning import schedule_ir
    from scar.contract import ARTIFACT_FLOW, json_bytes, read_blob, read_member
    from scar.middle.execution import seal

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
            resource_model={"artifact_flow": flow},
        )
        args = dict(graph_sha256="f" * 64, contract={}, logical_optimization=False)
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
    from scar.backend.cpp20 import _format_workers
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
        _, plan = schedule(dict(functions=[]), "serial", worker_limit=5)
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
            resource_model=dict(resources=resources),
        )
        prepared, _ = prepare(module)
        return seal(
            prepared,
            tmp_path / name,
            status="READY",
            graph_sha256="0" * 64,
            contract={},
            logical_optimization={},
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
    assert measure(source, "serial")[0] == 0
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
    site, mapped = decision["sites"]
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
        sites=2,
        interfaces={"pow": 2},
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

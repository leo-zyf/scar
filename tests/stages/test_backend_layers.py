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
    run("B2b", b2a / "target-ir.json", b2b)
    target = json.loads((b2a / "target-ir.json").read_bytes())
    assert target["operator_sdks"]["XTensor"]["provider"] == "xtensor-cpu"
    cmake = (b2b / "project/CMakeLists.txt").read_text()
    assert "vendor/xtensor-cpu/include" in cmake
    assert "SCAR_CPU_SDK" not in cmake and "find_package(xtensor" not in cmake
    source_tree = b2a / target["units"][0]["tree"]["path"]
    from scar.backend.cpp20.model import (
        Decl,
        Expr,
        Stmt,
        Unit,
        encode_unit,
        named,
        to_data,
    )

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
    run("B2b", b2a / "target-ir.json", tmp_path / "changed-tree", "checksum mismatch")
    assert not (tmp_path / "changed-tree/project").exists()
    b2a.rename(tmp_path / "hidden-target")
    run("B3", b2b / "cpp20.json", b2b)
    native = b2b / "project/bin/scar_entry"
    assert subprocess.check_output([native]) == b"sealed project\n"
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
    checks = [
        identical(box, alias),
        identical(box, fields, "ne"),
        identical(invoke("pool_a", p), invoke("pool_b", p)),
        identical(invoke("pool_a", p), other_slot, "ne"),
        identical(field(box, 2), box, "ne"),
        invoke("is_pooled", "bool", field(box, 2)),
        dict(op="binary", operator="eq", left=load, right=updated),
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
    assert subprocess.run([project / "bin/scar_entry"], check=True).returncode == 0
    header = (project / "include/scar/program.hpp").read_text()
    assert 'extern "C" bool is_pooled' not in header and "bool is_pooled" in header

    from scar.backend.cpp20.lowering import Lowerer

    lower = Lowerer(dict(functions=[]), {})
    lower.expression(pooled)
    with pytest.raises(ValueError, match="conflicting.*constant pool slot"):
        lower.expression(make(make(slot=0), lit("changed", "text"), slot=1))
    with pytest.raises(ValueError, match="requires literal"):
        lower.expression(make(var("local", "text"), slot=3))

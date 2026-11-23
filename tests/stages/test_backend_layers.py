"""Each backend pass must replay its sealed input with adjacent implementations absent."""

from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

import scar
from scar.contract import IR, document, member, write_json

_RUN_LAYER = """
import importlib, sys
from pathlib import Path

layer, source, output = sys.argv[1:]
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
run = getattr(importlib.import_module(module), entry)
if layer == "B3":
    run(source)
elif layer == "B1":
    run(source, output, policy="serial")
else:
    run(source, output)
"""


def test_backend_products_replay_and_reject_changed_bytes(tmp_path):
    def run(layer, source, out, expected=None):
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(Path(scar.__file__).resolve().parents[1]),
        )
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

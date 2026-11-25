"""F2 language normalization consumes only the facts captured by F1."""

import ast
import builtins
import json
from pathlib import Path
import shutil
import subprocess
import symtable
import tokenize

from scar.frontend.languages import fortran, python


def test_cpu_pool_affinity_does_not_remove_visible_hardware(tmp_path, monkeypatch):
    import pytest

    from scar import host
    from scar.frontend import hardware

    online = hardware._numbers(hardware._text(Path("/sys/devices/system/cpu/online")))
    if len(online) < 2:
        pytest.skip("requires multiple sysfs-visible CPUs")
    monkeypatch.setattr(hardware, "array_sdk", lambda: None)
    monkeypatch.setattr(host.os, "sched_getaffinity", lambda _: online)
    wide, wide_edges, _ = hardware.snapshot(tmp_path)
    selected = min(online)
    monkeypatch.setattr(host.os, "sched_getaffinity", lambda _: {selected})
    narrow, narrow_edges, platform = hardware.snapshot(tmp_path)
    kinds = {"CPU", "CPU_CORE", "CPU_CACHE", "NUMA_MEMORY", "DRAM"}
    assert [r for r in wide if r.kind in kinds] == [
        r for r in narrow if r.kind in kinds
    ]
    assert len({r.id for r in narrow}) == len(narrow)
    assert {f"cpu:{cpu}" for cpu in online} <= {r.id for r in narrow if r.kind == "CPU"}
    topology = {"THREAD_ON_CORE", "SHARED_CPU_CACHE", "HOST_MEMORY_ACCESS"}
    assert [e for e in wide_edges if e.value in topology] == [
        e for e in narrow_edges if e.value in topology
    ]
    assert [e.target for e in narrow_edges if e.value == "CPU_POOL_MEMBER"] == [
        f"cpu:{selected}"
    ]
    assert platform["affinity_cpu_ids"] == [selected]
    assert platform["cpu_availability"]["worker_budget"] == 1

    def unavailable(_):
        raise OSError("affinity unavailable")

    monkeypatch.setattr(host.os, "sched_getaffinity", unavailable)
    _, unknown_edges, unknown = hardware.snapshot(tmp_path)
    assert unknown["affinity_cpu_ids"] == []
    assert unknown["cpu_availability"]["affinity_evidence"] == "unavailable"
    assert unknown["cpu_availability"]["worker_budget"] == 1
    assert not any(e.value == "CPU_POOL_MEMBER" for e in unknown_edges)


def test_capture_mount_topology_distinguishes_backing_and_visibility(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from scar.frontend import hardware

    bind = tmp_path / "data set"
    remote = bind / "network"
    request = remote / "input"
    request.mkdir(parents=True)
    devices = tmp_path / "dev-block"
    devices.mkdir()
    disk = tmp_path / "block" / "disk"
    partition = disk / "partition1"
    partition.mkdir(parents=True)
    (partition / "partition").write_text("1")
    (disk / "size").write_text("4096")
    (devices / "8:1").symlink_to(partition)
    volume = tmp_path / "block" / "dm-test"
    (volume / "slaves").mkdir(parents=True)
    (volume / "slaves" / partition.name).symlink_to(partition)
    (volume / "slaves" / "same-partition").symlink_to(partition)
    (devices / "253:0").symlink_to(volume)
    namespace = tmp_path / "namespace"
    namespace.symlink_to("mnt:[9101]")
    mountinfo = tmp_path / "mountinfo"
    escaped = str(bind).replace(" ", r"\040")
    mountinfo.write_bytes(
        (
            "10 2 0:1 / / rw - overlay overlay rw\n"
            f"11 10 253:0 /data\\134old {escaped} rw - ext4 /dev/by-label/data rw\n"
            f"12 11 0:22 / {escaped}/network rw shared:8 - nfs4 host:/dir\\040name ro\n"
            "13 10 0:31 / /scratch rw - tmpfs tmpfs rw\n"
            "14 10 0:33 / /dev rw - devtmpfs udev rw\n"
            "15 14 0:35 / /dev/shm rw - tmpfs shm rw\n"
            "16 10 0:33 / /proc rw - proc proc rw\n"
            "17 16 0:37 / /proc/keys ro - tmpfs tmpfs ro\n"
            "18 10 0:38 / /other/control rw - cgroup2 none rw\n"
            "20 10 0:40 / /stack rw - tmpfs tmpfs rw\n"
            "21 20 0:41 / /stack rw - tmpfs tmpfs rw\n"
            "22 20 0:42 / /stack/hidden rw - tmpfs tmpfs rw\n"
            "23 10 0:43 / /uncertain rw - tmpfs tmpfs rw\n"
            "24 10 0:44 / /uncertain rw - tmpfs tmpfs rw\n"
            "25 16 0:45 / /proc/fs/nfsd rw - nfsd nfsd rw\n"
            "26 14 0:46 / /dev/hugepages rw - hugetlbfs hugetlbfs rw\n"
            "27 10 8:1 / /partition rw - ext4 /dev/partition1 rw\n"
        ).encode()
    )
    stat_calls = []

    def statvfs(path):
        stat_calls.append(path)
        return SimpleNamespace(f_blocks=10, f_frsize=4096)

    monkeypatch.setattr(hardware.os, "statvfs", statvfs)
    rows, edges, platform = hardware._storage_mounts(
        request, {}, mountinfo=mountinfo, device_root=devices, namespace_link=namespace
    )
    by_id = {row.id: row for row in rows}
    assert platform["input_mount"] == "mount:12"
    assert platform["mount_namespace"] == "mnt:[9101]"
    assert stat_calls == [request]
    assert by_id["mount:12"].capacity == 40960
    assert all(
        row.capacity is None
        for row in rows
        if row.kind == "STORAGE_MOUNT" and row.id != platform["input_mount"]
    )
    binding, network = (by_id[f"mount:{i}"].attributes for i in (11, 12))
    assert (binding["root"], binding["mountpoint"], binding["major_minor"]) == (
        r"/data\old",
        str(bind),
        "253:0",
    )
    assert network["source"] == "host:/dir name" and network["read_only"]
    assert network["backing_kind"] == "network_filesystem"
    assert by_id["mount:15"].attributes["backing_kind"] == "host_memory"
    assert not {f"mount:{i}" for i in (14, 16, 17, 18, 25)} & by_id.keys()
    assert by_id["mount:26"].attributes["backing_kind"] == "host_memory"
    assert by_id["mount:26"].attributes["page_policy"] == "huge_pages"
    assert {
        i: by_id[f"mount:{i}"].attributes["path_visibility"]
        for i in (20, 21, 22, 23, 24)
    } == {
        20: "covered",
        21: "visible",
        22: "covered",
        23: "unresolved",
        24: "unresolved",
    }
    links = {(e.source, e.target, e.value) for e in edges}
    assert ("mount:11", "block:dm-test", "MOUNT_BACKING_DEVICE") in links
    assert ("mount:27", "block:partition1", "MOUNT_BACKING_DEVICE") in links
    assert ("block:dm-test", "block:partition1", "BLOCK_BACKED_BY") in links
    assert ("block:partition1", "block:disk", "BLOCK_BACKED_BY") in links
    assert len(links) == len(edges)
    assert ("mount:12", "mount:11", "MOUNT_PARENT") in links
    assert all(e.source in by_id and e.target in by_id for e in edges)
    assert not any(
        e.source == "mount:15" and e.value == "MOUNT_BACKING_DEVICE" for e in edges
    )


def test_build_normalization_uses_only_captured_artifact_facts(tmp_path, monkeypatch):
    from scar.frontend.builds.normalize import normalize_build

    root = str(tmp_path)
    source, object_, executable = (
        str(tmp_path / n) for n in ("unit.c", "unit.o", "app")
    )
    compile_args = [
        "g++",
        "-std=c++17",
        "-Iinclude",
        "-B",
        "toolchain",
        "-c",
        "unit.c",
        "-ounit.o",
    ]
    link_args = ["g++", "unit.o", "-lm", "-o", "app"]
    facts = dict(
        root=root,
        provider="bear-3-intercept",
        trace_sha256="a" * 64,
        invocations=[
            dict(
                id=str(i),
                executable="/usr/bin/g++",
                directory=root,
                arguments=args,
                status=0,
                environment={},
            )
            for i, args in enumerate((compile_args, link_args))
        ],
        compilations=[
            dict(
                arguments=compile_args,
                directory=root,
                file="unit.c",
                canonical_file=source,
                output="unit.o",
            )
        ],
        files={
            path: dict(sha256="b" * 64, size=1, kind=kind)
            for path, kind in (
                (source, "source"),
                (object_, "object"),
                (executable, "executable"),
            )
        },
        libraries={executable: dict(needed=["libstdc++.so.6"], observed_at="capture")},
        tools={},
        issues=[],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("F2 build normalization accessed the host")

    for module, name in (
        (builtins, "open"),
        (Path, "open"),
        (subprocess, "Popen"),
        (shutil, "which"),
    ):
        monkeypatch.setattr(module, name, forbidden)
    output = normalize_build(facts, targets=["app"])
    assert not output["build_graph"]["diagnostics"]
    (profile,) = output["profiles"]
    assert profile["language"] == "c++"
    assert "-ounit.o" not in profile["flags"]
    assert str(tmp_path / "toolchain") in profile["flags"]
    assert str(tmp_path / "include") in profile["flags"]
    assert output["source_hashes"] == {"unit.c": "b" * 64}
    resources = output["software_resources"]
    assert {r["kind"] for r in resources} >= {
        "LIBRARY_REQUEST",
        "ELF_LIBRARY_DEPENDENCY",
    }
    graph = output["build_graph"]
    identities = {n["id"] for n in graph["nodes"]} | {r["id"] for r in resources}
    assert all(
        e["source"] in identities and e["target"] in identities for e in graph["edges"]
    )
    link_args.insert(1, "unobserved.o")
    incomplete = normalize_build(facts, targets=["app"])
    assert any(
        d["code"] == "build.missing_object_producer"
        for d in incomplete["build_graph"]["diagnostics"]
    )
    # Equal source flags do not identify the source toolchain. Keep both
    # profiles even when the two object outputs enter one final link.
    other = dict(facts["compilations"][0])
    other["arguments"] = ["/opt/other/g++", *compile_args[1:-1], "-oother.o"]
    other["output"] = "other.o"
    facts["compilations"].append(other)
    facts["invocations"].append(
        dict(
            id="other",
            executable="/opt/other/g++",
            directory=root,
            arguments=other["arguments"],
            status=0,
            environment={},
        )
    )
    link_args.remove("unobserved.o")
    link_args.insert(2, "other.o")
    distinct = normalize_build(facts, targets=["app"])
    assert len(distinct["profiles"]) == 2
    assert distinct["profiles"][0]["flags"] == distinct["profiles"][1]["flags"]
    assert (
        distinct["profiles"][0]["source_compiler"]
        != distinct["profiles"][1]["source_compiler"]
    )


def _offline(monkeypatch, language, facts, *parser_methods):
    def forbidden(*args, **kwargs):
        raise AssertionError("normalization attempted to read source or invoke a tool")

    with monkeypatch.context() as patch:
        for module, name in (
            (builtins, "open"),
            (Path, "open"),
            (tokenize, "open"),
            (subprocess, "Popen"),
            (shutil, "which"),
            *parser_methods,
        ):
            patch.setattr(module, name, forbidden)
        first = language.normalize_unit(facts)
        encoded = json.dumps(first, sort_keys=True, allow_nan=False)
        second = language.normalize_unit(facts)
    assert encoded == json.dumps(second, sort_keys=True, allow_nan=False)
    return first


def test_python_capture_then_offline_normalization(tmp_path, monkeypatch):
    entry = tmp_path / "entry.py"
    entry.write_text(
        "def outer(x):\n"
        "    def inner():\n"
        "        return x\n"
        "    return inner()\n"
        "outer(7)\n"
    )
    (unit,) = python.project_units(entry, tmp_path)
    facts = python.parse_unit(entry, (), root=tmp_path, abi=unit["abi"])
    from scar.contract import validate_text_profiles

    profiles = validate_text_profiles(facts.text_profiles)
    assert len(profiles) == 1
    profile_id, profile = next(iter(profiles.items()))
    assert profile["interpreter"] == facts.compiler_id
    assert profile["nonprintable"][0] == [0, 31]
    outer = facts.scopes.children["outer", 1]
    assert outer.cell == outer.children["inner", 2].free == {"x"}
    entry.unlink()
    output = _offline(
        monkeypatch, python, facts, (ast, "parse"), (symtable, "symtable")
    )
    functions = {f["name"]: f for f in output["functions"]}
    assert set(functions) == {
        "__main__",
        "py:entry:outer",
        "py:entry:outer.<locals>.inner",
    }
    assert all(f["partial_body"] for f in functions.values())
    assert {f["python_scope"]["text_profile"] for f in functions.values()} == {
        profile_id
    }


def test_python_select_comprehension_and_opaque_syntax_offline(tmp_path, monkeypatch):
    from scar.contract import walk_operations
    from scar.frontend.languages.python import normalize

    entry = tmp_path / "entry.py"
    entry.write_text(
        "def left(x): return x\n"
        "def right(x): return x\n"
        "def choose(flag, rows):\n"
        "    handler = rest = left\n"
        "    match rows:\n"
        "        case {'key': item, **rest} if flag:\n"
        "            handler = right\n"
        "            left(item + 1)\n"
        "    handler(0)\n"
        "    rest(0)\n"
        "    pick = (callee := left) if flag else (callee := right)\n"
        "    callee(0)\n"
        "    return [left(row) if flag else right(extra)\n"
        "            for row in rows if row for extra in (row,) if extra]\n"
    )
    (facts,) = python.capture_project(entry, tmp_path).units
    match = next(n for n in ast.walk(facts.tree) if isinstance(n, ast.Match))
    entry.unlink()
    unit = _offline(
        monkeypatch,
        normalize,
        facts,
        (ast, "parse"),
        (symtable, "symtable"),
        (builtins, "compile"),
    )
    fn = next(f for f in unit["functions"] if f["name"] == "py:entry:choose")
    ops = list(walk_operations(fn["partial_body"]))
    opaque = next(n for n in ops if n["op"] == "unsupported")
    assert list(walk_operations(opaque)) == [opaque]
    assert opaque["evaluation"] == {
        "control": "unresolved",
        "field_order": "syntax_only",
    }
    assert opaque["may_raise"] and set(opaque["effect_summary"].values()) == {"UNKNOWN"}
    assert opaque["missing"]["status"] == "unresolved"

    def check_syntax(source, record):
        if isinstance(source, ast.AST):
            assert type(source).__name__ == record["syntax"]
            assert [k for k, _ in record["fields"]] == list(source._fields)
            if hasattr(source, "lineno"):
                assert record["span"] == [
                    source.lineno,
                    source.col_offset + 1,
                    source.end_lineno,
                    source.end_col_offset + 1,
                ]
            else:
                assert record["span"] == "enclosing"
            for key, field in record["fields"]:
                check_syntax(getattr(source, key), field)
        elif isinstance(source, list):
            assert len(source) == len(record)
            for child, field in zip(source, record):
                check_syntax(child, field)
        else:
            assert type(record) is type(source) and source == record

    check_syntax(match, opaque["syntax"])
    invokes = {n["target"]["name"]: n for n in ops if n["op"] == "python_invoke"}
    assert all(
        name in invokes and not invokes[name].get("candidate_callees")
        for name in ("handler", "rest")
    )
    assert {c["symbol"] for c in invokes["callee"]["candidate_callees"]} == {
        "py:entry:left",
        "py:entry:right",
    }
    (comp,) = (n for n in ops if n["op"] == "sequence_comprehension")
    first, second = comp["generators"]
    assert comp["kind"] == "list" and comp["scope"]["bindings"] == ["extra", "row"]
    assert first["iterable"]["name"] == "rows" and not first["asynchronous"]
    assert first["target"]["cell"] == second["iterable"]["items"][0]["cell"]
    assert first["filters"][0]["cell"] == first["target"]["cell"]
    assert second["filters"][0]["cell"] == second["target"]["cell"]
    decision = comp["element"]
    assert decision["op"] == "python_select" and decision["condition"]["name"] == "flag"
    assert decision["evaluation"] == "condition_then_one_arm"
    assert decision["then"]["callee"] == "py:entry:left"
    assert decision["else"]["callee"] == "py:entry:right"
    assert decision["then"]["args"][0]["cell"]["owner"] == comp["scope"]["id"]
    assert decision["else"]["args"][0]["cell"] == second["target"]["cell"]


def test_python_mapping_displays_and_scoped_comprehensions_offline(
    tmp_path, monkeypatch
):
    from types import CodeType

    from scar.contract import walk_operations
    from scar.frontend.graph import function_graph
    from scar.frontend.languages.python import normalize
    from scar.frontend.state_flow import state_flow

    entry = tmp_path / "entry.py"
    entry.write_text(
        "def collect(k, step):\n"
        "    if isinstance(k, dict):\n"
        "        return {k: collect(v, step) for k, v in k.items() if v and collect(v, step) for j in step(k) if j}\n"
        "    return k\n"
        "def display(key, value, extra):\n"
        "    return {key(0):value(0), key(1):value(1), **extra(), key(2):{'nested':[1, None]}}\n"
        "class Local:\n"
        "    rows = [1]\n"
        "    lookup = {x: rows for x in rows}\n"
        "async def later(stream):\n"
        "    return {x: x async for x in stream if (seen := x)}\n"
    )
    (facts,) = python.capture_project(entry, tmp_path).units
    entry.unlink()
    unit = _offline(
        monkeypatch,
        normalize,
        facts,
        (ast, "parse"),
        (symtable, "symtable"),
        (builtins, "compile"),
    )
    functions = {f["name"]: f for f in unit["functions"]}
    collect = functions["py:entry:collect"]
    display, nested = sorted(
        (
            n
            for n in walk_operations(functions["py:entry:display"]["partial_body"])
            if n["op"] == "mapping"
        ),
        key=lambda n: n["location"]["column"],
    )
    assert [e["kind"] for e in display["entries"]] == ["pair", "pair", "unpack", "pair"]
    assert nested["entries"][0]["value"]["kind"] == "list"
    if hasattr(CodeType, "co_positions"):
        assert display["construction"]["status"] == "captured"
        steps = display["construction"]["steps"]
        entries = []
        for step in steps:
            if step["kind"] == "build_pairs":
                entries.extend(range(step["first"], step["first"] + step["count"]))
            elif step["kind"] != "merge_batch":
                entries.append(step["entry"])
        assert entries == list(range(4))
        assert {"kind": "unpack_mapping", "entry": 2} in steps
        # The two side-effecting pairs in a single source BUILD_MAP delay hashing
        # until both operands of the second pair have also been evaluated.
        if any(
            ("BUILD_MAP", 2) in stream for stream in facts.mapping_builders.values()
        ):
            assert steps[0] == dict(
                kind="build_pairs", first=0, count=2, constant_keys=False
            )
    else:
        assert display["construction"]["status"] == "unresolved"
    (comp,) = (
        n
        for n in walk_operations(collect["partial_body"])
        if n["op"] == "mapping_comprehension"
    )
    outer, inner = comp["generators"]
    assert outer["iterable"]["target"]["base"]["name"] == "k"
    assert (
        "cell" not in outer["iterable"]["target"]["base"]
        or outer["iterable"]["target"]["base"]["cell"]["owner"] == collect["name"]
    )
    assert [n["name"] for n in outer["target"]["items"]] == ["k", "v"]
    assert (
        outer["filters"][0]["op"] == "short_circuit"
        and inner["filters"][0]["name"] == "j"
    )
    assert (
        inner["iterable"]["args"][0]["cell"]
        == comp["key"]["cell"]
        == outer["target"]["items"][0]["cell"]
    )
    assert comp["key"]["cell"]["owner"] == comp["scope"]["id"]
    assert comp["value"]["callee"] == collect["name"]
    assert (
        sum(
            n["op"] == "call" and n["callee"] == collect["name"]
            for n in walk_operations(comp)
        )
        == 2
    )
    assert (
        collect["partial_body"][-1]["value"].get("cell", {}).get("owner")
        != comp["scope"]["id"]
    )
    (class_comp,) = (
        n
        for n in walk_operations(
            functions["py:entry:Local.<class_body>"]["partial_body"]
        )
        if n["op"] == "mapping_comprehension"
    )
    assert class_comp["generators"][0]["iterable"]["name"] == "rows"
    assert (
        class_comp["value"]["name"] == "py:entry:rows"
        and class_comp["value"]["scope"] == "global"
    )
    (async_comp,) = (
        n
        for n in walk_operations(functions["py:entry:later"]["partial_body"])
        if n["op"] == "mapping_comprehension"
    )
    assert async_comp["generators"][0]["asynchronous"] is True
    (named,) = async_comp["generators"][0]["filters"]
    assert named["op"] == "assignment_expression"
    assert named["target"]["name"] == "seen"
    assert named["target"].get("cell", {}).get("owner") != async_comp["scope"]["id"]

    graph = function_graph(collect, facts.source_hash)
    port, row = next(
        (p, n)
        for p, n in enumerate(graph["nodes"])
        if n["op"] == "mapping_comprehension"
    )
    target = graph["nodes"][row["generators"][0]["target"]["port"]]
    writes = {a["node"] for a in graph["accesses"] if a["mode"] == "write"}
    assert all(t["port"] in writes for t in target["items"])
    assert collect["name"] in graph["calls"]
    flow = state_flow(graph["nodes"], graph["entry"], graph["accesses"])
    boundary = next(b for b in flow["boundary_details"] if b["node"] == port)
    assert boundary["first_iterable"] == row["generators"][0]["iterable"]["port"]
    assert boundary["iteration_count"] == "zero_or_more"


def test_python_tuple_identity_tracks_compiled_objects_offline(tmp_path, monkeypatch):
    from dataclasses import replace
    import dis
    import sys
    from types import CodeType

    from scar.contract import walk_operations
    from scar.frontend.languages.python import normalize

    entry = tmp_path / "items.py"
    source = (
        "def fresh(x): return x, x\n"
        "def first(): return (1, 7)\n"
        "def second(): return (1, 7)\n"
        "def equal(): return (True, 7)\n"
        "def empty(): return ()\n"
        "def nested(): return (1 + 1, -3, ('q', ()), ('x',) * 2)\n"
        "def child(): return ('q', ())\n"
        "def elided(x, y):\n    a, b = x, y\n    return a\n"
        "def mutable(x): return [x]\n"
    )
    entry.write_text(source)
    facts = python.capture_project(entry, tmp_path).units[0]
    other = tmp_path / "other.py"
    other.write_text(source)
    separate = python.capture_project(other, tmp_path).units[0]
    other.unlink()
    program = compile(source, str(entry), "exec", dont_inherit=True)
    constants = {
        c.co_name: next(v for v in c.co_consts if type(v) is tuple)
        for c in program.co_consts
        if isinstance(c, CodeType)
        and c.co_name in {"first", "second", "equal", "nested", "child"}
    }
    entry.unlink()

    def aggregates(unit):
        result = _offline(
            monkeypatch,
            normalize,
            unit,
            (builtins, "compile"),
            (dis, "get_instructions"),
        )
        by_function = {}
        for fn in result["functions"]:
            for node in walk_operations(fn.get("partial_body", [])):
                if node["op"] == "aggregate":
                    by_function[fn["name"].rsplit(":", 1)[-1]] = node
                    break
        return by_function

    entry_ops = aggregates(facts)
    assert "identity" not in entry_ops["mutable"]
    identity = {
        k: op["identity"] for k, op in entry_ops.items() if op["kind"] == "tuple"
    }
    if sys.implementation.name != "cpython" or not any(
        col is not None
        for _, _, col, _ in getattr(program, "co_positions", lambda: ())()
    ):
        assert all(f["kind"] == "unresolved" for f in identity.values())
        return
    assert identity["fresh"] == {"kind": "fresh"}
    assert identity["elided"]["kind"] == "unresolved"
    assert identity["empty"]["kind"] == "constant_pool"
    assert constants["equal"] == constants["first"]
    assert constants["equal"] is not constants["first"]
    for name, value in constants.items():
        assert identity[name]["kind"] == "constant_pool"
        assert (identity[name] == identity["first"]) == (value is constants["first"])
    nested = entry_ops["nested"]["items"]
    assert nested[2]["identity"]["kind"] == "constant_pool"
    assert (nested[2]["identity"] == identity["child"]) == (
        constants["nested"][2] is constants["child"]
    )
    assert nested[2]["items"][1]["identity"] == identity["empty"]
    assert nested[3]["left"]["identity"]["kind"] == "unresolved"

    imported = aggregates(replace(facts, abi={**facts.abi, "main": False}))
    assert imported["first"]["identity"]["pool"] != identity["first"]["pool"]
    assert imported["empty"]["identity"] == identity["empty"]
    other_ops = aggregates(separate)
    assert other_ops["first"]["identity"]["pool"] != identity["first"]["pool"]
    assert other_ops["empty"]["identity"] == identity["empty"]


def test_module_constant_stores_bind_source_bytecode_and_graph(tmp_path, monkeypatch):
    import dis
    import sys

    from scar.contract import digest, json_bytes, read_member, walk_operations
    from scar.frontend import build
    from scar.frontend.languages.python.environment import code_digest

    entry = tmp_path / "bindings.py"
    source = (
        "global explicit\n"
        "é = 'same'; also = 'same'\n"
        "explicit = ('different')\n"
        "if False:\n    elided = 'no instruction'\n"
        "try:\n    probe()\nfinally:\n    repeated = 'two instructions'\n"
        "object.field = 'attribute'\nleft = right = 'chain'\nnumber = 1\n"
        "def nested():\n    local = 'nested'\n"
    )
    entry.write_text(source)
    code = compile(source, str(entry), "exec", dont_inherit=True)
    instructions = {i.offset: i for i in dis.get_instructions(code)}
    out = tmp_path / "graph"
    build(entry, out)
    graph = json.loads((out / "graph.json").read_bytes())
    unit = read_member(out, graph["units"][0]["graph"])
    (root,) = (fn for fn in unit["functions"] if fn["signature"]["name"] == "__main__")
    rows = [
        n["source_constant_store"]
        for n in root["nodes"]
        if "source_constant_store" in n
    ]
    if sys.implementation.name != "cpython" or not any(
        column is not None
        for _, _, column, _ in getattr(code, "co_positions", lambda: ())()
    ):
        assert not rows and "source_code" not in root["signature"]["python_scope"]
        return
    scope = root["signature"]["python_scope"]
    assert scope["source_code"] == dict(
        kind="module",
        source_key=entry.name,
        source_sha256=digest(source.encode()),
        code_sha256=code_digest(code),
        compiler=f"{sys.implementation.name} {sys.version.split()[0]}",
        optimize=sys.flags.optimize,
    )
    assert [r["store"]["name"] for r in rows] == ["é", "also", "explicit"]
    assert [r["store"]["opcode"] for r in rows] == [
        "STORE_NAME",
        "STORE_NAME",
        "STORE_GLOBAL",
    ]
    assert rows[0]["load"]["index"] == rows[1]["load"]["index"]
    assert rows[0]["load"]["offset"] != rows[1]["load"]["offset"]
    for node in root["nodes"]:
        if "source_constant_store" not in node:
            continue
        fact = node["source_constant_store"]
        assert fact["semantics_version"] == 1 and fact["code"] == scope["source_code"]
        for key, operand in (("load", "value"), ("store", "target")):
            port = fact[key]
            ins = instructions[port["offset"]]
            assert (port["opcode"], port["index"], port["position"]) == (
                ins.opname,
                ins.arg,
                list(ins.positions),
            )
            arg = root["nodes"][node[operand]["port"]]
            loc = arg["location"]
            assert port["position"] == [
                loc["line"],
                loc["end_line"],
                loc["column"] - 1,
                loc["end_column"] - 1,
            ]
            if key == "load":
                assert port["type"] == "str" and code.co_consts[ins.arg] == arg["value"]
                assert port["value_sha256"] == digest(json_bytes(arg["value"]))
            else:
                assert port["name"] == code.co_names[ins.arg]
                assert arg["name"] == "py:bindings:" + port["name"]
    assert all(
        "source_code" not in f["signature"]["python_scope"]
        and all("source_constant_store" not in n for n in f["nodes"])
        for f in unit["functions"]
        if f is not root
    )
    facts = python.capture_project(entry, tmp_path).units[0]
    entry.unlink()
    offline = _offline(
        monkeypatch, python, facts, (builtins, "compile"), (dis, "get_instructions")
    )
    receipts = [
        n["source_constant_store"]
        for n in walk_operations(offline["functions"])
        if "source_constant_store" in n
    ]
    assert sorted(receipts, key=lambda r: r["store"]["offset"]) == rows


def test_compiled_module_fold_and_raise_use_complete_source_operands(tmp_path):
    import dis
    import sys

    from scar.contract import digest, json_bytes, read_member
    from scar.frontend import build

    source = (
        'message = ("first" +\n           "second")\n'
        "factor = -(1.5 + 2.25)\n"
        "effect = touch() + 1\n"
        "choice = 340 if touch() else 340\n"
        "runtime = 1 / 0\n"
        "try:\n    perform()\nexcept Error as cause:\n    raise new(message) from cause\n"
        "if flag:\n    raise single\n"
        "try:\n    perform()\nexcept Other:\n    raise\n"
        "try:\n    perform()\nfinally:\n    repeated = 100 * 200\n"
        "    raise uncertain() from cause\n"
    )
    path = tmp_path / "operations.py"
    path.write_text(source)
    code = compile(source, str(path), "exec", dont_inherit=True)
    out = tmp_path / "graph"
    build(path, out)
    unit = read_member(
        out, json.loads((out / "graph.json").read_bytes())["units"][0]["graph"]
    )
    fn = next(f for f in unit["functions"] if f["signature"]["name"] == "__main__")
    rows = {i.offset: i for i in dis.get_instructions(code)}
    stores = [n for n in fn["nodes"] if "source_constant_store" in n]
    raises = [n for n in fn["nodes"] if "source_expression" in n]
    if sys.implementation.name != "cpython" or not any(
        i.positions.col_offset is not None for i in rows.values()
    ):
        assert not stores and not raises
        return
    assert [n["source_constant_store"]["store"]["name"] for n in stores] == [
        "message",
        "factor",
    ]
    leaves = [
        ["binary", "add", ["literal", "str", "first"], ["literal", "str", "second"]],
        [
            "unary",
            "usub",
            ["binary", "add", ["literal", "float", 1.5], ["literal", "float", 2.25]],
        ],
    ]
    code_sha = fn["signature"]["python_scope"]["source_code"]["code_sha256"]
    for n, shape in zip(stores, leaves, strict=True):
        fact = n["source_constant_store"]
        load, store = fact["load"], fact["store"]
        assert (
            fact["rhs_sha256"] == digest(json_bytes(shape))
            and fact["code"]["code_sha256"] == code_sha
        )
        assert fn["nodes"][n["value"]["port"]]["op"] == shape[0]
        assert load["opcode"] == rows[load["offset"]].opname == "LOAD_CONST"
        value = code.co_consts[load["index"]]
        assert load["type"] == type(value).__name__ and load["value_sha256"] == digest(
            json_bytes(value)
        )
        assert (
            rows[store["offset"]].argval == store["name"]
            and store["opcode"] == "STORE_NAME"
        )
    assert [n["location"]["line"] for n in raises] == [10, 12, 16]
    for n in raises:
        receipt = n["source_expression"]
        native = [
            i for i in rows.values() if i.positions.lineno == n["location"]["line"]
        ]
        last = next(i for i in native if i.opname == "RAISE_VARARGS")
        assert n["op"] == "raise" and receipt["code_sha256"] == code_sha
        operands = receipt["instructions"]
        assert operands[-1]["position"] == receipt["position"] == list(last.positions)
        assert (operands[-1]["opcode"], operands[-1]["oparg"]) == (
            "RAISE_VARARGS",
            int(n["has_exception"]) + int(n["has_cause"]),
        )
        assert [
            (row["opcode"], row["offset"], row["position"]) for row in operands
        ] == [
            (i.opname, i.offset, list(i.positions))
            for i in native
            if i.offset <= last.offset
        ]
    assert [
        r.get("name")
        for r in raises[0]["source_expression"]["instructions"]
        if "name" in r
    ] == ["new", "message", "cause"]


def test_module_assign_and_delete_use_the_actual_named_slots(tmp_path):
    import dis
    import sys

    from scar.contract import read_member
    from scar.frontend import build

    path = tmp_path / "slots.py"
    source = (
        "global saved\n"
        "saved = record.format(value, digits=2)\n"
        "alias = saved\n"
        "method = record.format\n"
        "number = 2\n"
        "literal = 'old'\n"
        "folded = 'o' + 'ld'\n"
        "del saved, alias, alias\n"
        "del method\n"
        "record.attr = value\n"
        "left = right = value\n"
        "del (number, method)\n"
        "del record.attr\n"
        "try:\n    probe()\nfinally:\n    repeated = callback()\n    del repeated\n"
    )
    path.write_text(source)
    program = compile(source, str(path), "exec", dont_inherit=True)
    out = tmp_path / "graph"
    build(path, out)
    unit = read_member(
        out, json.loads((out / "graph.json").read_bytes())["units"][0]["graph"]
    )
    fn = next(f for f in unit["functions"] if f["signature"]["name"] == "__main__")
    actual = {i.offset: i for i in dis.get_instructions(program)}
    operations = [n for n in fn["nodes"] if "source_expression" in n]
    if sys.implementation.name != "cpython" or not any(
        i.positions.col_offset is not None for i in actual.values()
    ):
        assert not operations
        return
    assert [(n["location"]["line"], n["op"]) for n in operations] == [
        (2, "assign"),
        (3, "assign"),
        (4, "assign"),
        (5, "assign"),
        (8, "unbind"),
        (9, "unbind"),
    ]
    assert [
        n["location"]["line"] for n in fn["nodes"] if "source_constant_store" in n
    ] == [6, 7]
    for n in operations:
        receipt = n["source_expression"]
        rows = receipt["instructions"]
        assert (
            receipt["code_sha256"]
            == fn["signature"]["python_scope"]["source_code"]["code_sha256"]
        )
        loc = n["location"]
        assert receipt["position"] == [
            loc["line"],
            loc["end_line"],
            loc["column"] - 1,
            loc["end_column"] - 1,
        ]
        # The final return may share the last binding's line; the source operation
        # itself ends at the captured write or delete.
        native = [
            i
            for i in actual.values()
            if i.positions.lineno == loc["line"] and i.offset <= rows[-1]["offset"]
        ]
        assert [(r["opcode"], r["offset"], r["position"]) for r in rows] == [
            (i.opname, i.offset, list(i.positions)) for i in native
        ]
        targets = [n["target"]] if n["op"] == "assign" else n["targets"]
        for encoded, row in zip(targets, rows[-len(targets) :], strict=True):
            target = fn["nodes"][encoded["port"]]
            name = target["name"].rsplit(":", 1)[1]
            assert row["opcode"] == ("STORE" if n["op"] == "assign" else "DELETE") + (
                "_GLOBAL" if name == "saved" else "_NAME"
            )
            assert (
                row["oparg"] == row["index"]
                and program.co_names[row["index"]] == row["name"] == name
            )
    assert [
        (r["opcode"], r["name"])
        for r in operations[4]["source_expression"]["instructions"]
    ] == [
        ("DELETE_GLOBAL", "saved"),
        ("DELETE_NAME", "alias"),
        ("DELETE_NAME", "alias"),
    ]


def test_module_display_ports_bind_typed_pool_items_and_dynamic_order(tmp_path):
    import dis

    from scar.contract import digest, json_bytes, read_member
    from scar.frontend import build

    path = tmp_path / "display.py"
    text = (
        "empty=[]\nunit=()\npacked=[True,1,1.0,'1',None]\npooled=(True,1,1.0,'1',None)\n"
        "dynamic=[value, build(arg)]\npair=(value, build(arg))\n"
        "large=("
        + ",".join(["value"] * 31)
        + ")\nexpanded=[*values]\nnested=[['inner']]\n"
    )
    path.write_text(text)
    code = compile(text, str(path), "exec", dont_inherit=True)
    out = tmp_path / "graph"
    build(path, out)
    fn = next(
        f
        for f in read_member(
            out, json.loads((out / "graph.json").read_bytes())["units"][0]["graph"]
        )["functions"]
        if f["signature"]["name"] == "__main__"
    )
    native = {i.offset: i for i in dis.get_instructions(code)}
    witnesses = [n for n in fn["nodes"] if "source_expression" in n]
    if not any(i.positions.col_offset is not None for i in native.values()):
        assert not witnesses
        return
    assert [n["location"]["line"] for n in witnesses] == list(range(1, 8))
    for n in witnesses:
        aggregate = fn["nodes"][n["value"]["port"]]
        assert aggregate["op"] == "aggregate"
        rows = n["source_expression"]["instructions"]
        assert [r["opcode"] for r in rows] == [
            i.opname
            for i in native.values()
            if i.positions.lineno == n["location"]["line"]
            and i.offset <= rows[-1]["offset"]
        ]
        for row in rows:
            if "items_sha256" in row:
                items = [fn["nodes"][p["port"]] for p in aggregate["items"]]
                actual = code.co_consts[row["index"]]
                assert type(actual) is tuple and row["item_count"] == len(items) == len(
                    actual
                )
                shape = [["literal", i["type"]["class"], i["value"]] for i in items]
                assert shape == [["literal", type(v).__name__, v] for v in actual]
                assert (
                    row["items_sha256"] == digest(json_bytes(shape))
                    and "names" not in row
                )
    assert [r["opcode"] for r in witnesses[2]["source_expression"]["instructions"]] == [
        "BUILD_LIST",
        "LOAD_CONST",
        "LIST_EXTEND",
        "STORE_NAME",
    ]
    large = witnesses[-1]["source_expression"]["instructions"]
    assert (
        sum(r["opcode"] == "LIST_APPEND" for r in large) == 31
        and large[-2]["intrinsic"] == "INTRINSIC_LIST_TO_TUPLE"
    )


def test_module_if_call_receipt_binds_truth_and_both_successors(tmp_path):
    import dis
    import sys

    from scar.contract import read_member
    from scar.frontend import build

    text = (
        "if choose(arg):\n    left()\nelse:\n    right()\n"
        "if not probe.value(flag):\n    selected()\nnext_step()\n"
        "if choose(arg) or other():\n    compound()\n"
        "try:\n    check()\nfinally:\n    if decide():\n        duplicated()\n"
        "if final():\n    done()\n"
    )
    path = tmp_path / "branch.py"
    path.write_text(text)
    code = compile(text, str(path), "exec", dont_inherit=True)
    out = tmp_path / "graph"
    build(path, out)
    fn = next(
        f
        for f in read_member(
            out, json.loads((out / "graph.json").read_bytes())["units"][0]["graph"]
        )["functions"]
        if f["signature"]["name"] == "__main__"
    )
    native = {i.offset: i for i in dis.get_instructions(code)}
    ordered = list(native)
    found = [n for n in fn["nodes"] if n["op"] == "if" and "source_expression" in n]
    if not any(i.positions.col_offset is not None for i in native.values()):
        assert not found
        return
    assert [n["location"]["line"] for n in found] == [1, 5, 15]
    for n, kind in zip(
        found, ("suite", "continuation", "implicit_return"), strict=True
    ):
        receipt = n["source_expression"]
        rows = receipt["instructions"]
        edges = receipt["branch"]
        jumped = rows[-1]
        j = native[jumped["offset"]]
        negated = n["location"]["line"] == 5
        assert edges["negated"] is negated and edges["else"]["kind"] == kind
        assert (
            jumped["opcode"]
            == j.opname
            == ("POP_JUMP_IF_TRUE" if negated else "POP_JUMP_IF_FALSE")
        )
        assert edges["else"]["offset"] == jumped["target"] == j.argval
        assert edges["operand_position"] == list(j.positions)
        assert [r["opcode"] for r in rows] == [
            native[o].opname
            for o in ordered[
                ordered.index(rows[0]["offset"]) : ordered.index(j.offset) + 1
            ]
        ]
        assert ("TO_BOOL" in [r["opcode"] for r in rows]) == (
            sys.version_info >= (3, 13)
        )
        offset = ordered.index(j.offset) + 1
        if marker := edges["fallthrough_marker"]:
            assert (
                native[ordered[offset]].opname == marker["opcode"] == "NOT_TAKEN"
                and marker["offset"] == ordered[offset]
            )
            offset += 1
        assert edges["then"]["offset"] == ordered[offset]
        for dest in (edges["then"], edges["else"]):
            actual = native[dest["offset"]]
            assert dest["opcode"] == actual.opname and dest["position"] == list(
                actual.positions
            )
        if kind == "suite":
            assert edges["else"]["statement_position"] == [4, 4, 4, 11]
        elif kind == "continuation":
            assert edges["else"]["statement_position"] == [7, 7, 0, 11]
        else:
            assert code.co_consts[edges["else"]["constant_index"]] is None


def test_module_name_loads_follow_actual_callee_and_argument_ports(tmp_path):
    import dis
    import sys
    from collections import Counter

    from scar.contract import read_member
    from scar.frontend import build
    from scar.frontend.languages.python.environment import code_digest

    entry = tmp_path / "lookups.py"
    source = (
        "global action, arg\n"
        "result = action(convert(é), arg, len(é))\n"
        "value += arg\n"
        "if False:\n    omitted\n"
        "try:\n    action\nfinally:\n    repeated\n"
        "def inner(): return action(arg)\n"
    )
    entry.write_text(source)
    code = compile(source, str(entry), "exec", dont_inherit=True)
    out = tmp_path / "graph"
    build(entry, out)
    graph = json.loads((out / "graph.json").read_bytes())
    unit = read_member(out, graph["units"][0]["graph"])
    root = next(f for f in unit["functions"] if f["signature"]["name"] == "__main__")
    nodes = root["nodes"]
    loads = [n["source_name_load"] for n in nodes if "source_name_load" in n]
    if sys.implementation.name != "cpython" or not any(
        c is not None for _, _, c, _ in getattr(code, "co_positions", lambda: ())()
    ):
        assert not loads
        return
    assert root["signature"]["python_scope"]["source_code"][
        "code_sha256"
    ] == code_digest(code)
    instructions = [
        i
        for i in dis.get_instructions(code)
        if i.opname in {"LOAD_NAME", "LOAD_GLOBAL"}
    ]
    counts = Counter(i.positions for i in instructions)
    expected = {i.offset: i for i in instructions if counts[i.positions] == 1}
    assert {r["offset"] for r in loads} == set(expected)
    for r in loads:
        ins = expected[r["offset"]]
        assert (r["opcode"], r["oparg"], r["name"], r["position"]) == (
            ins.opname,
            ins.arg,
            ins.argval,
            list(ins.positions),
        )
        assert code.co_names[r["index"]] == ins.argval and r[
            "code_sha256"
        ] == code_digest(code)
    outer = nodes[next(n for n in nodes if n["op"] == "assign")["value"]["port"]]
    target = nodes[outer["target"]["port"]]
    convert, argument, length = [nodes[port["port"]] for port in outer["args"]]
    ports = [
        target,
        nodes[convert["target"]["port"]],
        nodes[convert["args"][0]["port"]],
        argument,
        nodes[length["target"]["port"]],
        nodes[length["args"][0]["port"]],
    ]
    assert [n["source_name_load"]["name"] for n in ports] == [
        "action",
        "convert",
        "é",
        "arg",
        "len",
        "é",
    ]
    assert [n["source_name_load"]["offset"] for n in ports] == sorted(
        i.offset for i in instructions if i.positions.lineno == 2
    )
    assert ports[4]["op"] == "python_builtin"
    assert all(
        "source_name_load" not in n
        for f in unit["functions"]
        if f is not root
        for n in f["nodes"]
    )


def test_module_import_store_matches_cpython_stack_and_source_span(tmp_path):
    import dis
    import sys

    from scar.contract import read_member
    from scar.frontend import build
    from scar.frontend.languages.python.environment import code_digest

    (tmp_path / "pkg").mkdir()
    for relative in ("pkg/__init__.py", "pkg/leaf.py", "other.py"):
        (tmp_path / relative).write_text("")
    entry = tmp_path / "driver.py"
    source = (
        "global chosen\n"
        "if enabled:\n    import pkg.leaf\n    import other as chosen\n"
        "import pkg.leaf as projected\n"
        "import other, other as second\n"
        "from pkg import leaf\n"
        "if False:\n    import absent\n"
        "try:\n    enabled\nfinally:\n    import other\n"
        "def nested():\n    import other\n"
    )
    entry.write_text(source)
    code = compile(source, str(entry), "exec", dont_inherit=True)
    by_offset = {i.offset: i for i in dis.get_instructions(code)}
    out = tmp_path / "graph"
    build(entry, out)
    graph = json.loads((out / "graph.json").read_bytes())
    unit = read_member(
        out, next(u["graph"] for u in graph["units"] if u["path"] == entry.name)
    )
    root = next(f for f in unit["functions"] if f["signature"]["name"] == "__main__")
    sites = [n for n in root["nodes"] if "source_import_store" in n]
    if sys.implementation.name != "cpython" or not any(
        col is not None for _, _, col, _ in getattr(code, "co_positions", lambda: ())()
    ):
        assert not sites
        return
    assert [(n["location"]["line"], n["module"]) for n in sites] == [
        (3, "pkg.leaf"),
        (4, "other"),
    ]
    code_sha = code_digest(code)
    assert root["signature"]["python_scope"]["source_code"]["code_sha256"] == code_sha
    for node, destination, store_opcode in zip(
        sites, ("pkg", "chosen"), ("STORE_NAME", "STORE_GLOBAL"), strict=True
    ):
        port = node["source_import_store"]
        assert node["op"] == "python_import" and port["code_sha256"] == code_sha
        loc = node["location"]
        assert port["position"] == [
            loc["line"],
            loc["end_line"],
            loc["column"] - 1,
            loc["end_column"] - 1,
        ]
        assert node["request"]["fromlist"] == [] and port["fromlist"]["value"] is None
        level = by_offset[port["level"]["offset"]]
        assert level.opname in {"LOAD_CONST", "LOAD_SMALL_INT"}
        expected = [
            ("level", level.opname, 0),
            ("fromlist", "LOAD_CONST", None),
            ("import_name", "IMPORT_NAME", node["request"]["module"]),
            ("store", store_opcode, destination),
        ]
        for field, opcode, value in expected:
            operand = port[field]
            ins = by_offset[operand["offset"]]
            assert (ins.opname, list(ins.positions)) == (opcode, port["position"])
            assert operand["opcode"] == opcode and ins.argval == value
            if opcode == "LOAD_SMALL_INT":
                assert "index" not in operand
                assert operand["oparg"] == ins.arg == operand["value"] == 0
                continue
            assert operand["index"] == ins.arg and "oparg" not in operand
            if opcode == "LOAD_CONST":
                assert code.co_consts[ins.arg] == operand["value"] == value
            else:
                assert code.co_names[ins.arg] == operand["name"] == value
        assert (
            root["nodes"][node["binding"]["port"]]["name"] == f"py:driver:{destination}"
        )
    assert all(
        "source_import_store" not in n
        for f in unit["functions"]
        if f is not root
        for n in f["nodes"]
    )


def test_module_call_expression_seals_ordered_operands_once(tmp_path, monkeypatch):
    import dis
    import sys
    from collections import Counter

    from scar.contract import digest, json_bytes, read_member
    from scar.frontend import build

    if sys.implementation.name != "cpython" or sys.version_info[:2] not in {
        (3, 12),
        (3, 14),
    }:
        import pytest

        pytest.skip("this test compares the CPython 3.12 and 3.14 call encodings")

    entry = tmp_path / "notice.py"
    source = (
        'report.warn(f"before {left} and {right!r:>{width}}", category, stacklevel=2)\n'
        "plain(thing)\nreport.warn(*args, **kwargs)\n"
        "try:\n    probe()\nfinally:\n    report.note(1)\n"
        "if False:\n    report.note(2)\n"
        "def nested():\n    report.note(3)\n"
    )
    entry.write_text(source)
    code = compile(source, str(entry), "exec", dont_inherit=True)
    instructions = [i for i in dis.get_instructions(code) if i.positions.lineno == 1]
    counter = Counter()
    compile_ast, decode = builtins.compile, dis.get_instructions

    def count_compile(src, filename, *args, **kw):
        if isinstance(src, ast.Module) and filename == str(entry):
            counter["compile"] += 1
        return compile_ast(src, filename, *args, **kw)

    def count_decode(obj, *args, **kw):
        if obj.co_filename == str(entry) and obj.co_name == "<module>":
            counter["decode"] += 1
        return decode(obj, *args, **kw)

    out = tmp_path / "graph"
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "compile", count_compile)
        patch.setattr(dis, "get_instructions", count_decode)
        build(entry, out)
    graph = json.loads((out / "graph.json").read_bytes())
    unit = read_member(out, graph["units"][0]["graph"])
    expressions = [
        n for f in unit["functions"] for n in f["nodes"] if "source_expression" in n
    ]
    if not any(i.positions.col_offset is not None for i in instructions):
        assert expressions == []
        return
    assert counter == {"compile": 1, "decode": 1}
    [call] = expressions
    assert call["location"]["line"] == 1 and [k["name"] for k in call["keywords"]] == [
        "stacklevel"
    ]
    root = next(f for f in unit["functions"] if f["signature"]["name"] == "__main__")
    receipt = call["source_expression"]
    assert (
        receipt["code_sha256"]
        == root["signature"]["python_scope"]["source_code"]["code_sha256"]
    )
    rows = receipt["instructions"]
    assert (
        rows[-1]["opcode"] == "POP_TOP" and rows[-1]["position"] == receipt["position"]
    )
    for row, ins in zip(rows, instructions, strict=True):
        assert (row["opcode"], row["offset"], row["position"], row.get("oparg")) == (
            ins.opname,
            ins.offset,
            list(ins.positions),
            ins.arg,
        )
        if "name" in row:
            assert row["name"] == ins.argval == code.co_names[row["index"]]
        if "names" in row:
            assert list(code.co_consts[row["index"]]) == row["names"] == ["stacklevel"]
        if "value_sha256" in row:
            value = (
                ins.arg
                if ins.opname == "LOAD_SMALL_INT"
                else code.co_consts[row["index"]]
            )
            assert row["type"] == type(value).__name__ and row[
                "value_sha256"
            ] == digest(json_bytes(value))


def test_module_named_exception_arms_join_native_match_edges_and_alias_cleanup(
    tmp_path,
):
    import dis
    import sys

    from scar.contract import read_member
    from scar.frontend import build

    if sys.implementation.name != "cpython" or sys.version_info[:2] not in {
        (3, 12),
        (3, 14),
    }:
        import pytest

        pytest.skip("this test compares the CPython 3.12 and 3.14 handler encodings")

    path = tmp_path / "handle.py"
    text = (
        "global first, problem\n"
        "try:\n    run()\n"
        "except first as problem:\n    save(problem)\n"
        "except second:\n    raise\n"
        "try:\n    run()\nexcept (first, second):\n    pass\n"
        "try:\n    run()\nexcept first as other:\n    if choose():\n        raise\n    save(other)\n"
        "while flag:\n    try:\n        run()\n    except first as problem:\n        break\n"
        "def nested():\n    try:\n        run()\n    except first:\n        raise\n"
    )
    path.write_text(text)
    code = compile(text, str(path), "exec", dont_inherit=True)
    native = {i.offset: i for i in dis.get_instructions(code)}
    table = [e._asdict() for e in dis.Bytecode(code).exception_entries]
    out = tmp_path / "graph"
    build(path, out)
    unit = read_member(
        out, json.loads((out / "graph.json").read_bytes())["units"][0]["graph"]
    )
    all_receipts = [
        n
        for f in unit["functions"]
        for n in f["nodes"]
        if "source_exception_handler" in n
    ]
    if not any(i.positions.col_offset is not None for i in native.values()):
        assert not all_receipts
        return
    module = next(f for f in unit["functions"] if f["signature"]["name"] == "__main__")
    arms = [n for n in module["nodes"] if "source_exception_handler" in n]
    assert arms == all_receipts and [n["location"]["line"] for n in arms] == [4, 6]
    first, second = [n["source_exception_handler"] for n in arms]
    for index, (node, port) in enumerate(zip(arms, (first, second), strict=True)):
        assert port["load"] == module["nodes"][node["test"]["port"]]["source_name_load"]
        assert (port["arm_index"], port["arm_count"]) == (index, 2)
        assert (
            port["code_sha256"]
            == module["signature"]["python_scope"]["source_code"]["code_sha256"]
        )
        assert port["check"]["opcode"] == "CHECK_EXC_MATCH"
        assert (
            port["branch"]["target"]
            == native[port["branch"]["offset"]].argval
            == port["mismatch"]["offset"]
        )
        assert port["exception_table"] and all(
            e in table for e in port["exception_table"]
        )
        for row in [
            port["entry"],
            port["load"],
            port["check"],
            port["branch"],
            *port["matched"],
            port["mismatch"],
            *port["normal_exit"],
            *port["exceptional_cleanup"],
            *port["unwind"],
        ]:
            ins = native[row["offset"]]
            assert (row["opcode"], row.get("oparg"), row["position"]) == (
                ins.opname,
                ins.arg,
                list(ins.positions) if ins.positions.lineno is not None else None,
            )
            if "name" in row:
                assert code.co_names[row["index"]] == row["name"] == ins.argval
    assert (
        first["load"]["opcode"] == "LOAD_GLOBAL"
        and first["matched"][-1]["name"] == "problem"
    )
    assert first["mismatch"]["offset"] == second["load"]["offset"]
    assert first["entry"] == second["entry"] and first["unwind"] == second["unwind"]
    assert [r["opcode"] for r in first["normal_exit"]] == [
        "POP_EXCEPT",
        "LOAD_CONST",
        "STORE_GLOBAL",
        "DELETE_GLOBAL",
    ]
    assert [r["opcode"] for r in first["exceptional_cleanup"]] == [
        "LOAD_CONST",
        "STORE_GLOBAL",
        "DELETE_GLOBAL",
        "RERAISE",
    ]
    assert first["body_raise"] is None and first["normal_exit"][1]["value"] is None
    assert any(
        e["start"] <= first["body_offset"] < e["end"]
        and e["target"] == first["exceptional_cleanup"][0]["offset"]
        for e in first["exception_table"]
    )
    assert second["matched"][-1]["opcode"] == "POP_TOP"
    assert (
        second["mismatch"]["opcode"] == "RERAISE" and second["mismatch"]["oparg"] == 0
    )
    assert (
        second["normal_exit"] == second["exceptional_cleanup"] == []
        and second["body_raise"]["opcode"] == "RAISE_VARARGS"
    )


def test_scalar_annotation_contract_uses_captured_compiler_semantics(
    tmp_path, monkeypatch
):
    entry = tmp_path / "typed.py"
    entry.write_text("def scale(x: float) -> float:\n    return x * 2.0\n")
    facts = python.parse_unit(entry, root=tmp_path, abi={"exports": {"scale": "scale"}})
    entry.unlink()
    for version in ("cpython 3.12.3", "cpython 3.14.7"):
        with monkeypatch.context() as patch:
            patch.setattr(type(facts), "compiler_id", version)
            result = _offline(
                patch, python, facts, (ast, "parse"), (symtable, "symtable")
            )
        scale = next(f for f in result["functions"] if f["name"] == "scale")
        assert "body" in scale and not result["diagnostics"]


def test_python_project_locks_and_parses_each_imported_file_once(tmp_path, monkeypatch):
    from collections import Counter
    from hashlib import sha256

    from scar.frontend import build

    (tmp_path / "pkg").mkdir()
    sources = {
        tmp_path
        / "main.py": b"\xef\xbb\xbffrom pkg import constants\r\nimport pkg.constants\r\n",
        tmp_path / "pkg/__init__.py": b"from . import constants\r\n",
        tmp_path / "pkg/constants.py": b"# coding: latin-1\r\nname = 'caf\xe9'\r\n",
    }
    for path, raw in sources.items():
        path.write_bytes(raw)
    reads, parses = Counter(), Counter()
    read, parse = Path.read_bytes, ast.parse

    def count_read(path):
        if path in sources:
            reads[path] += 1
        return read(path)

    def count_parse(source, filename="<unknown>", *args, **kwargs):
        if Path(filename) in sources:
            parses[Path(filename)] += 1
        return parse(source, filename, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", count_read)
    monkeypatch.setattr(ast, "parse", count_parse)
    folder = tmp_path / "graph"
    build(tmp_path / "main.py", folder)
    assert reads == parses == Counter({path: 1 for path in sources})
    graph = json.loads((folder / "graph.json").read_bytes())
    captured = {
        name: sha
        for u in graph["units"]
        for name, sha in u["source_facts"]["files"].items()
    }
    assert captured == {
        path.relative_to(tmp_path).as_posix(): sha256(raw).hexdigest()
        for path, raw in sources.items()
    }


def test_script_images_share_syntax_and_preserve_import_bindings(tmp_path, monkeypatch):
    from collections import Counter

    from scar.contract import read_member
    from scar.frontend import build

    shared = tmp_path / "shared.py"
    shared.write_text("import choice\n")
    images = []
    for name in ("left", "right"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "entry.py").write_text("import helper\n")
        (directory / "helper.py").symlink_to(shared)
        (directory / "choice.py").write_text(f"value = {name!r}\n")
        images.append(
            dict(id=name, output=name, entry="__main__", source=f"{name}/entry.py")
        )
    images.append({**images[0], "id": "repeat", "output": "repeat"})
    manifest = tmp_path / "project.json"
    manifest.write_text(json.dumps(dict(root_image="right", images=images)))

    parse, parsed = ast.parse, Counter()

    def count(source, filename="<unknown>", *args, **kwargs):
        path = Path(filename)
        if path.is_relative_to(tmp_path):
            parsed[path.relative_to(tmp_path).as_posix()] += 1
        return parse(source, filename, *args, **kwargs)

    monkeypatch.setattr(ast, "parse", count)
    output = tmp_path / "graph"
    build(manifest, output)
    graph = json.loads((output / "graph.json").read_bytes())
    members = {image["id"]: image["units"] for image in graph["images"]}
    assert graph["root_image"] == "right" and members["repeat"] == members["left"]
    expected = set()
    for name in ("left", "right"):
        paths = {f"{name}/entry.py", f"{name}/choice.py", "shared.py"}
        expected.update(paths)
        units = {graph["units"][i]["path"]: graph["units"][i] for i in members[name]}
        assert set(units) == paths
        shared_graph = read_member(output, units["shared.py"]["graph"])
        (binding,) = (
            n
            for f in shared_graph["functions"]
            for n in f["nodes"]
            if n["op"] == "python_import"
        )
        assert binding["import_resolution"]["file"] == f"{name}/choice.py"
    assert parsed == Counter({path: 1 for path in expected})


def test_entry_environment_separates_launch_declarations_from_analysis_cache(
    tmp_path, monkeypatch
):
    from hashlib import sha256
    import sys
    from types import ModuleType

    import pytest

    from scar.frontend import build
    from scar.frontend.capture import capture_project
    from scar.frontend.input import read_request

    entry = tmp_path / "entry.py"
    active, blocked, foreign, cold = (
        "scar_startup_loaded",
        "scar_startup_blocked",
        "scar_startup_other",
        "scar_startup_cold",
    )
    entry.write_text(f"import {active}, {blocked}, {foreign}, {cold}\n")
    (tmp_path / f"{cold}.py").write_text("raise AssertionError('executed target')\n")
    monkeypatch.setitem(sys.modules, active, ModuleType(active))
    monkeypatch.setitem(sys.modules, blocked, None)
    monkeypatch.setitem(sys.modules, foreign, object())
    monkeypatch.delitem(sys.modules, cold, raising=False)
    assert read_request(entry).python_entries == (
        {"image": None, "declaration": {"kind": "unspecified"}},
    )
    executable = Path(sys.executable)
    standalone = dict(
        kind="standalone_script", interpreter=str(executable), flags=["-I"]
    )
    images = [
        dict(id=name, output=name, entry="__main__", source="entry.py", **fields)
        for name, fields in (
            ("script", {"entry_environment": standalone}),
            ("plugin", {"entry_environment": {"kind": "embedded"}}),
            ("undeclared", {}),
        )
    ]
    manifest, out = tmp_path / "request.json", tmp_path / "graph"
    manifest.write_text(json.dumps(dict(images=images, root_image="script")))
    build(manifest, out)
    graph = json.loads((out / "graph.json").read_bytes())
    environment = graph["source_facts"]["python_entry_environment"]
    assert environment["entries"] == [
        {"image": "script", "declaration": standalone},
        {"image": "plugin", "declaration": {"kind": "embedded"}},
        {"image": "undeclared", "declaration": {"kind": "unspecified"}},
    ]
    assert all(
        "python_entry_environment" not in u["source_facts"] for u in graph["units"]
    )
    analysis = environment["analysis"]
    assert (analysis["scope"], analysis["confidence"]) == (
        "analysis_process_snapshot",
        "observed",
    )
    assert analysis["flags"]["isolated"] == sys.flags.isolated
    assert analysis["import_callable"] == {
        "kind": "builtin_import",
        "type": "builtins.builtin_function_or_method",
    }
    assert analysis["interpreter"]["image"] == dict(
        path=str(executable.resolve()),
        size=executable.stat().st_size,
        sha256=sha256(executable.read_bytes()).hexdigest(),
    )
    assert analysis["selected_module_cache"] == {
        active: "present_module",
        blocked: "present_none",
        foreign: "present_other",
        cold: "absent",
    }
    assert (
        analysis["deployment_equivalence"],
        analysis["application_initial_cache"],
        analysis["audit_hooks"],
    ) == ("unproven", "unproven", "unknown")
    # A second environment may link the same executable and resolve different packages.
    alternate = tmp_path / "other-python"
    alternate.symlink_to(executable)
    manifest.write_text(
        json.dumps(
            dict(
                source="entry.py",
                entry_environment={**standalone, "interpreter": str(alternate)},
            )
        )
    )
    with pytest.raises(ValueError, match="differs from the analysis interpreter"):
        capture_project(read_request(manifest))


def test_python_import_precedence_and_runtime_facts(tmp_path, monkeypatch):
    import codecs
    from importlib.machinery import BuiltinImporter, FrozenImporter
    import os
    import sys
    from types import SimpleNamespace

    from scar.contract import walk_operations

    fs_errors = sys.getfilesystemencodeerrors()
    stdin_errors = "replace" if fs_errors == "strict" else "strict"
    monkeypatch.setattr(
        sys, "stdin", SimpleNamespace(encoding="utf-8", errors=stdin_errors)
    )
    entry = tmp_path / "entry.py"
    entry.write_text(
        "import os, shutil, sys, time, math, subprocess, glob\n"
        "from os.path import join as combine\nopen('never-executed')\n"
    )
    (facts,) = python.capture_project(entry, tmp_path).units
    unit = _offline(monkeypatch, python, facts)
    nodes = list(walk_operations(unit["functions"][0]["partial_body"]))
    imported = next(n for n in nodes if n.get("module") == "os.path")
    assert imported["bound_value"]["external"] == "os.path.join"
    source = imported["source_provider"]
    assert source["module"] == "os" and source["status"] == "verified_stdlib"
    assert (
        source["filesystem_errors"]
        == facts.runtime_facts["filesystem_errors"]
        == fs_errors
    )
    assert facts.runtime_facts["filesystem_encoding"] == sys.getfilesystemencoding()
    assert facts.runtime_facts["stdin_errors"] == stdin_errors
    glob_imports = facts.providers["glob"]["direct_imports"]
    matcher = glob_imports["modules"]["fnmatch"]
    assert glob_imports["scope"] == "absolute_module_suite"
    assert "fnmatch" in glob_imports["order"] and matcher["status"] == "verified_stdlib"
    assert matcher["sha256"] != facts.providers["glob"]["sha256"]
    assert source["aliases"]["os.path"]["module"] == os.path.__name__
    opened = next(
        n for n in nodes if n["op"] == "python_builtin" and n["name"] == "open"
    )
    assert opened["source_runtime"] == facts.runtime_facts
    with entry.open() as stream:
        assert codecs.lookup(
            facts.runtime_facts["default_text_encoding"]
        ) == codecs.lookup(stream.encoding)

    shadows = {"os", "shutil", "sys", "time", "math"}
    for name in shadows:
        (tmp_path / (name + ".py")).write_text(
            "raise RuntimeError('capture executed a module')\n"
        )
    (tmp_path / "fnmatch.py").write_text(
        "raise RuntimeError('capture executed a dependency')\n"
    )
    project = python.capture_project(entry, tmp_path)
    selected = project.units[0].providers["glob"]["direct_imports"]["modules"][
        "fnmatch"
    ]
    if BuiltinImporter.find_spec("fnmatch") or FrozenImporter.find_spec("fnmatch"):
        assert selected == matcher
    else:
        assert selected["status"] == "project" and selected["file"] == "fnmatch.py"
    path_modules = {
        name
        for name in shadows
        if not (BuiltinImporter.find_spec(name) or FrozenImporter.find_spec(name))
    }
    assert set(project.catalog) == {"entry"} | path_modules
    for name in shadows:
        provider = project.units[0].providers[name]
        if name in path_modules:
            assert provider["status"] == "project" and provider["file"] == name + ".py"
        else:
            assert provider == facts.providers[name] and provider["loader"] in {
                "frozen",
                "builtin",
            }


def test_external_import_facts_preserve_ownership_without_execution(
    tmp_path, monkeypatch
):
    from hashlib import sha256
    import py_compile
    import sys

    from scar.contract import python_provider_id, walk_operations
    from scar.frontend.languages.python import normalize
    from scar.frontend.languages.python.imports import Imports, source_member

    project = tmp_path / "project"
    installed = tmp_path / "installed"
    project.mkdir()
    installed.mkdir()
    package = installed / "scar_foreign"
    package.mkdir()
    for path in (
        package / "__init__.py",
        package / "ops.py",
        installed / "scar_compat.py",
    ):
        path.write_text("raise AssertionError('capture executed target code')\n")
    metadata = installed / "foreign_distribution-1.2.3.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: foreign-distribution\nVersion: 1.2.3\n"
    )
    (metadata / "top_level.txt").write_text("scar_foreign\nscar_compat\n")
    (metadata / "RECORD").write_text(
        "scar_foreign/__init__.py,,\nscar_foreign/ops.py,,\nscar_compat.py,,\n"
    )
    monkeypatch.syspath_prepend(str(installed))
    entry = project / "entry.py"
    entry.write_text(
        "import scar_foreign.ops as operations\n"
        "from scar_foreign import ops\nimport scar_compat\n"
        "from json.decoder import JSONDecoder\nfrom json import decoder\n"
        "import sys, math, _frozen_importlib\n"
        "import scar_missing_package.part\n"
    )
    capture = python.capture_project(entry, project)
    units = {unit.abi["module"]: unit for unit in capture.units}
    assert set(units) == {"entry", "json", "json.decoder"}
    facts = units["entry"]
    for name, path in (
        ("scar_foreign", package / "__init__.py"),
        ("scar_foreign.ops", package / "ops.py"),
        ("scar_compat", installed / "scar_compat.py"),
    ):
        record = facts.providers[name]
        assert record["status"] == "verified_distribution"
        assert record["module"] == name
        assert (
            record["distribution"].lower().replace("_", "-") == "foreign-distribution"
        )
        assert record["version"] == "1.2.3"
        assert record["sha256"] == sha256(path.read_bytes()).hexdigest()
    decoder = units["json.decoder"]
    origin = facts.providers["json.decoder"]
    assert origin["status"] == "verified_stdlib"
    assert (origin["loader"], origin["loader_kind"]) == ("path", "source")
    assert origin["source_member"] == "json/decoder.py"
    assert origin["provider_id"] == python_provider_id(
        {k: v for k, v in origin.items() if k not in {"loader_kind", "source_member"}}
    )
    assert decoder.source_key == f"@python/{origin['provider_id']}/json/decoder.py"
    assert decoder.source_origins[decoder.source_key]["sha256"] == decoder.source_hash
    assert (
        origin["exports"]["JSONDecoder"]["candidates"][0]["member"]
        == origin["source_member"]
    )
    frontier = [
        (name, r)
        for name, r in decoder.providers.items()
        if r.get("source_expansion", {}).get("status") == "frontier"
    ]
    assert frontier and all(
        name not in units and source_member(r) for name, r in frontier
    )
    assert facts.providers["sys"]["loader_kind"] == "builtin"
    assert facts.providers["_frozen_importlib"]["loader_kind"] == "frozen"
    assert not source_member(facts.providers["sys"])
    assert not source_member(facts.providers["_frozen_importlib"])
    resolver = Imports(project, facts.compiler_id)
    extension = next(
        r
        for name in ("math", "_struct", "_decimal", "_pickle", "_bz2", "zlib")
        if (r := resolver.record(name)).get("loader_kind") == "extension"
    )
    assert extension["loader"] == "path" and source_member(extension) is None

    # Give the controlled finder a stdlib root with an actual sourceless module.
    # Its bytecode would raise if the capture invoked the loader.
    binary_name = "fixture_bytecode_only"
    path = installed / f"{binary_name}.py"
    path.write_text("raise AssertionError('bytecode executed')\n")
    bytecode = path.with_suffix(".pyc")
    py_compile.compile(str(path), cfile=str(bytecode), doraise=True)
    path.unlink()
    monkeypatch.setattr(
        sys, "stdlib_module_names", sys.stdlib_module_names | {binary_name}
    )
    resolver.stdlib_path.insert(0, str(installed))
    sourceless = resolver.record(binary_name)
    assert (
        sourceless["status"] == "verified_stdlib"
        and sourceless["loader_kind"] == "bytecode"
    )
    assert sourceless["sha256"] == sha256(bytecode.read_bytes()).hexdigest()
    assert source_member(sourceless) is None
    assert facts.providers["scar_missing_package.part"]["status"] == "unresolved"
    entry.unlink()
    shutil.rmtree(installed)
    neutral = _offline(
        monkeypatch, normalize, facts, (ast, "parse"), (builtins, "compile")
    )
    stdlib = _offline(
        monkeypatch, normalize, decoder, (ast, "parse"), (builtins, "compile")
    )
    assert any(
        fn["name"] == "py:json.decoder:JSONDecoder.decode" for fn in stdlib["functions"]
    )
    imports = {
        n["module"]: n
        for function in neutral["functions"]
        for n in walk_operations(function.get("partial_body", ()))
        if n["op"] == "python_import"
    }
    imported = imports["scar_foreign.ops"]
    assert imported["source_provider"] == facts.providers["scar_foreign"]
    assert imported["import_resolution"] == facts.providers["scar_foreign.ops"]
    assert imported["initializers"] == [] and "IO" in imported["effects"]
    assert imports["json.decoder"]["graph_backed"]
    assert imports["json.decoder"]["initializers"] == [
        "py:json:<init>",
        "py:json.decoder:<init>",
    ]
    assert imports["json"]["initializers"] == ["py:json:<init>"]
    assert imports["json"]["conditional_initializers"] == [
        {
            "module": "json.decoder",
            "initializer": "py:json.decoder:<init>",
            "when": {
                "kind": "package_attribute_missing",
                "module": "json",
                "name": "decoder",
            },
        }
    ]
    (check,) = imports["json"]["fromlist_checks"]["items"]
    assert check["child"] == dict(
        module="json.decoder",
        initializer="py:json.decoder:<init>",
        provider_id=origin["provider_id"],
        sha256=origin["sha256"],
    )


def test_fromlist_checks_precede_all_member_bindings(tmp_path, monkeypatch):
    import dis

    from scar.frontend.languages.python import normalize

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "child.py").write_text("")
    statement = "from . import child as first, dynamic, child as second"
    entry = package / "__init__.py"
    entry.write_text(
        f"{statement}\n"
        "def __getattr__(name):\n    raise RuntimeError(name)\n"
        "import math\nfrom . import *\n"
    )
    capture = python.capture_project(entry, tmp_path)
    facts = next(u for u in capture.units if u.abi["module"] == "pkg")
    unit = _offline(
        monkeypatch, normalize, facts, (ast, "parse"), (builtins, "compile")
    )
    main = next(f for f in unit["functions"] if f["name"] == "__main__")
    first, dynamic, second, plain, star = [
        op for op in main["partial_body"] if op["op"] == "python_import"
    ]
    bytecode = [
        (op.opname, op.argval)
        for op in dis.get_instructions(compile(statement, "fixture", "exec"))
        if op.opname.startswith("IMPORT_")
    ]
    assert bytecode == [
        ("IMPORT_NAME", ""),
        ("IMPORT_FROM", "child"),
        ("IMPORT_FROM", "dynamic"),
        ("IMPORT_FROM", "child"),
    ]
    checks = first["fromlist_checks"]
    assert checks["phase"] == "during_IMPORT_NAME_before_IMPORT_FROM"
    assert not any("fromlist_checks" in row for row in (dynamic, second, plain, star))
    assert not any(
        row["conditional_initializers"] for row in (dynamic, second, plain, star)
    )
    assert [row["alias"] for row in (first, dynamic, second)] == [
        "first",
        None,
        "second",
    ]
    for index, (item, binding) in enumerate(
        zip(checks["items"], (first, dynamic, second), strict=True)
    ):
        member = bytecode[index + 1][1]
        assert binding["request"] == dict(
            form="from",
            module="",
            level=1,
            fromlist=["child", "dynamic", "child"],
            binding_index=index,
        )
        when = dict(kind="package_attribute_missing", module="pkg", name=member)
        expected = dict(
            binding_index=index,
            member=member,
            when=when,
            missing_attribute_hooks=[dict(symbol="py:pkg:__getattr__", when=when)],
        )
        if member == "child":
            expected["child"] = dict(
                module="pkg.child", initializer="py:pkg.child:<init>"
            )
        assert item == expected
    assert {"py:pkg.child:<init>", "py:pkg:__getattr__"} <= set(main["calls"])


def test_numpy_linspace_captures_selected_native_dependency_without_importing(
    tmp_path, monkeypatch
):
    from hashlib import sha256
    from importlib.metadata import distribution

    from scar.contract import read_member, walk_operations
    from scar.frontend import build

    entry = tmp_path / "entry.py"
    entry.write_text("from numpy import linspace as grid, array\ngrid(0., 1.)\n")
    original_import = builtins.__import__

    def block_numpy(name, *args, **kwargs):
        assert name.split(".")[0] != "numpy", "capture executed the subject provider"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_numpy)
    facts = python.parse_unit(entry, root=tmp_path)
    exports = facts.providers["numpy"]["exports"]
    dependencies = exports["linspace"]["implementation_dependencies"]
    assert "implementation_dependencies" not in exports["array"]
    loaders = {
        "numpy._core.numeric": "source",
        "numpy._core.multiarray": "source",
        "numpy._core._multiarray_umath": "extension",
    }
    assert dependencies.keys() == loaders.keys()
    installed = distribution("numpy")
    for name, loader in loaders.items():
        item = dependencies[name]
        content = installed.locate_file(item["member"]).read_bytes()
        assert item == dict(
            module=name,
            member=item["member"],
            distribution="numpy",
            version="2.5.5",
            status="verified_distribution",
            loader=loader,
            proof_scope="installed_file_with_digest",
            sha256=sha256(content).hexdigest(),
        )
    unit = _offline(monkeypatch, python, facts)
    operation = next(
        node
        for node in walk_operations(unit["functions"][0]["partial_body"])
        if node["op"] == "python_import"
    )
    assert operation["source_provider"]["exports"]["linspace"] == exports["linspace"]

    extension = dependencies["numpy._core._multiarray_umath"]
    assert operation["source_provider"]["array_runtime"] == extension
    entry.write_text("import numpy as np\nnp.zeros((2, 3))\n")
    graph = tmp_path / "graph"
    build(entry, graph)
    (unit,) = json.loads((graph / "graph.json").read_bytes())["units"]
    zeros = unit["source_facts"]["providers"]["numpy"]
    assert "linspace" not in zeros["exports"]
    assert zeros["array_runtime"] == extension
    payload = read_member(graph, unit["graph"])
    assert (
        next(
            n["source_provider"]["array_runtime"]
            for n in walk_operations(payload)
            if n["op"] == "python_import"
        )
        == extension
    )

    (tmp_path / "numpy.py").write_text("raise AssertionError('executed shadow')\n")
    shadow = python.parse_unit(entry, root=tmp_path).providers["numpy"]
    assert shadow["status"] == "project" and "array_runtime" not in shadow


def test_distribution_source_units_share_capture_and_follow_requested_bodies(
    tmp_path, monkeypatch
):
    from base64 import urlsafe_b64encode
    from collections import Counter
    from hashlib import sha256

    from scar.contract import walk_operations

    project = tmp_path / "project"
    project.mkdir()
    installed = project / ".sdk" / "site-packages"
    package = installed / "orbit_source"
    package.mkdir(parents=True)
    marker = tmp_path / "import-executed"
    contents = {
        "__init__": f"open({str(marker)!r}, 'w').close()\nfrom .ops import apply, unused_binding\n"
        "def __getattr__(name):\n    from .lookup import missing\n    return missing(name)\n"
        "def dormant():\n    from .unused import unused\n    return unused()\n",
        "ops": "def apply(x):\n    from .step import step\n    return step(x)\n"
        "def unused_binding():\n    from .unused import unused\n    return unused()\n",
        "step": "def step(x): return x + 2\n",
        "lookup": "def missing(name): raise AttributeError(name)\n",
        "unused": "def unused(): return 123\n",
    }
    metadata = installed / "orbit_source-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: orbit-source\nVersion: 1.0\n"
    )
    rows = []
    for name, text in contents.items():
        data = text.encode()
        path = package / f"{name}.py"
        path.write_bytes(data)
        checksum = urlsafe_b64encode(sha256(data).digest()).decode().rstrip("=")
        rows.append(f"{path.relative_to(installed)},sha256={checksum},{len(data)}\n")
    (metadata / "RECORD").write_text("".join(rows))
    monkeypatch.syspath_prepend(str(installed))
    entry = project / "entry.py"
    entry.write_text("import orbit_source as namespace\nnamespace.apply(1)\n")

    reads, parses, scopes = Counter(), Counter(), Counter()
    read, parse, scope = Path.read_bytes, ast.parse, symtable.symtable

    def counted_read(path):
        reads[path] += 1
        return read(path)

    def count(function, counter):
        def capture(source, filename, *args, **kwargs):
            counter[Path(filename)] += 1
            return function(source, filename, *args, **kwargs)

        return capture

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", counted_read)
        patch.setattr(ast, "parse", count(parse, parses))
        patch.setattr(symtable, "symtable", count(scope, scopes))
        captured = python.capture_project(entry, project)
    assert set(captured.catalog) == {
        "entry",
        "orbit_source",
        "orbit_source.ops",
        "orbit_source.step",
        "orbit_source.lookup",
    }
    assert not marker.exists() and not reads[package / "unused.py"]
    installed.rename(tmp_path / "hidden")
    entry.unlink()
    for profile, unit in zip(captured.profiles, captured.units, strict=True):
        assert reads[unit.path] == parses[unit.path] == scopes[unit.path] == 1
        module = profile["abi"]["module"]
        row = captured.catalog[module]
        assert row["graph_backed"] and row["path"] == profile["path"] == unit.source_key
        assert unit.source_digests == {
            unit.source_key: captured.source_digests[unit.source_key]
        }
        output = _offline(
            monkeypatch, python, unit, (ast, "parse"), (symtable, "symtable")
        )
        if module == "entry":
            assert "orbit_source.step" not in profile["abi"]["project"]
            imported = next(
                op
                for op in walk_operations(output["functions"])
                if op["op"] == "python_import"
            )
            assert imported["provider"] != "project" and imported["initializers"] == [
                "py:orbit_source:<init>"
            ]
            continue
        provider = row["source_provider"]
        assert (
            provider["status"] == "verified_distribution"
            and provider["loader"] == "source"
        )
        assert (
            unit.source_key == f"@python/{provider['provider_id']}/{provider['member']}"
        )
        assert unit.source_origins[unit.source_key]["origin"] == str(unit.path)
        initializer = next(
            fn for fn in output["functions"] if fn["name"] == f"py:{module}:<init>"
        )
        assert initializer["location"]["file"] == unit.source_key
        if module == "orbit_source":
            assert any(
                fn["name"] == "py:orbit_source:__getattr__"
                for fn in output["functions"]
            )


def test_external_exports_follow_aliases_and_preserve_source_evidence(
    tmp_path, monkeypatch
):
    from base64 import urlsafe_b64encode
    from hashlib import sha256
    from math import copysign
    import sys

    project, installed = tmp_path / "project", tmp_path / "installed"
    project.mkdir()
    package = installed / "source_library"
    package.mkdir(parents=True)
    sources = {
        "source_library/__init__.py": (
            "from .unknown import *\nfrom .first import produce as selected\nsaved = selected\nfrom .second import produce as selected\n"
            "from .values import public, changing, large, zero, infinite, flag, nothing, data\n"
            "def __getattr__(name): return runtime_lookup(name)\n"
        ),
        "source_library/unknown.py": "__all__ = unknown_at_runtime()\n",
        "source_library/first.py": 'def produce(x):\n    """not a statement site"""\n    def nested():\n        return 999\n    return x + 1\n',
        "source_library/second.py": "def produce(x):\n    return x + 2\n",
        "source_library/values.py": (
            "tag = 'science'\npublic = tag\ntag = 'later'\nchanging = 7\n"
            "if runtime_flag(): globals()['changing'] = runtime_value()\n"
            "large = 9223372036854775808\nzero = -0.0\ninfinite = -1e999\n"
            "flag = True\nnothing = None\ndata = b'abc'\n"
        ),
    }
    metadata = installed / "source_library-999.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: source-library\nVersion: 999\n"
    )
    (metadata / "top_level.txt").write_text("source_library\n")
    records = []
    for name, content in sources.items():
        data = content.encode()
        (installed / name).write_bytes(data)
        records.append(
            f"{name},sha256={urlsafe_b64encode(sha256(data).digest()).decode().rstrip('=')},{len(data)}\n"
        )
    (metadata / "RECORD").write_text("".join(records))
    monkeypatch.syspath_prepend(str(installed))
    entry = project / "entry.py"
    entry.write_text(
        "import source_library as api\napi.saved(1)\napi.selected(2)\n"
        "from source_library import public, changing, large, zero, infinite, flag, nothing, data\n"
    )
    assert "source_library" not in sys.modules
    facts = python.parse_unit(entry, root=project)
    exports = facts.providers["source_library"]["exports"]
    (first,) = exports["saved"]["candidates"]
    (second,) = exports["selected"]["candidates"]
    assert (
        first["module"] == "source_library.first"
        and second["module"] == "source_library.second"
    )
    assert exports["selected"]["unresolved"] == []
    assert first["locations"] == {"3": "def nested():", "5": "return x + 1"}
    assert all(step["sha256"] for step in first["via"])
    assert first["kind"] == second["kind"] == "definition"

    (literal,) = exports["public"]["candidates"]
    assert (literal["kind"], literal["source_type"], literal["literal"]) == (
        "literal",
        "builtins.str",
        {"type": "text", "value": "science"},
    )
    assert (
        literal["name"],
        literal["version"],
        literal["line"],
        literal["column"],
    ) == ("tag", "999", 1, 1)
    assert (
        literal["end_line"] == 1 and literal["end_column"] == len("tag = 'science'") + 1
    )
    assert [(s["kind"], s["name"]) for s in literal["via"]] == [
        ("from", "public"),
        ("alias", "public"),
        ("literal", "tag"),
    ]
    for step in literal["via"]:
        assert step["sha256"] == sha256(sources[step["member"]].encode()).hexdigest()
        assert len(step["provider_id"]) == 64
    assert exports["public"]["on_missing"]["hook"] == "__getattr__"
    assert exports["__getattr__"]["candidates"][0]["kind"] == "definition"
    assert exports["changing"]["candidates"][0]["conditions"][0]["branch"] == "else"
    assert {u["reason"] for u in exports["changing"]["unresolved"]} == {
        "namespace_write"
    }
    large = exports["large"]["candidates"][0]
    assert large["literal"] == {"type": "python_int", "value": 1 << 63}
    assert large["representation"] == "requires_integer_range_contract"
    assert copysign(1, exports["zero"]["candidates"][0]["literal"]["value"]) == -1
    assert exports["infinite"]["candidates"][0]["literal"] == {
        "type": "python_float",
        "value_hex": "-inf",
    }
    assert exports["flag"]["candidates"][0]["source_type"] == "builtins.bool"
    assert exports["nothing"]["candidates"][0]["literal"] == {
        "type": "unit",
        "value": None,
    }
    assert (
        exports["data"]["candidates"] == []
        and exports["data"]["unresolved"][0]["source_type"] == "builtins.bytes"
    )
    assert "source_library" not in sys.modules
    (package / "second.py").write_text("def produce(x):\n    return x + 900\n")
    changed = python.parse_unit(entry, root=project).providers["source_library"][
        "exports"
    ]["selected"]
    assert not changed["candidates"] and changed["unresolved"]
    shutil.rmtree(installed)
    entry.unlink()
    neutral = _offline(monkeypatch, python, facts)
    from scar.contract import walk_operations

    imported = next(
        n for n in walk_operations(neutral["functions"]) if n["op"] == "python_import"
    )
    assert imported["source_provider"]["exports"] == exports


def test_python_binding_time_loop_edges_and_class_frames(tmp_path, monkeypatch):
    from scar.contract import walk_operations

    entry = tmp_path / "entry.py"
    source = """import subprocess
command = "first"
def issue():
    subprocess.run(command)
issue()
command = "later"
issue()
def repeated(items):
    command = "x"
    for _ in items:
        subprocess.run(command)
        command = command + "y"
repeated([1, 2, 3])
def stable():
    word = "bounded"
    subprocess.run(word)
stable()
class Base: pass
def nested():
    x = "cell"
    class Choice(Base):
        x = "namespace"
        def selected(self): return x
    return Choice().selected()
def selected(): return command
"""
    entry.write_text(source)
    actual, namespace = [], {}
    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "run", actual.append)
        exec(compile(source, str(entry), "exec"), namespace)
    assert actual == ["first", "later", "x", "xy", "xyy", "bounded"]
    assert namespace["nested"]() == "cell" and namespace["selected"]() == "later"

    (facts,) = python.capture_project(entry, tmp_path).units
    assert facts.exports["command"] == {
        "state": "py:entry:command",
        "literal_sites": [{"line": 2, "value": "first"}, {"line": 6, "value": "later"}],
    }
    # The class forwards the free variable to its method even as it shadows x.
    (nested,) = (s for (n, _), s in facts.scopes.children.items() if n == "nested")
    assert nested.cell == {"x"}
    functions = {f["name"]: f for f in python.normalize_unit(facts)["functions"]}

    def nodes(fn, op):
        return [
            n for n in walk_operations(functions[fn]["partial_body"]) if n["op"] == op
        ]

    for name in ("issue", "repeated", "stable"):
        (launch,) = nodes("py:entry:" + name, "process_launch")
        assert launch["binding_guard"]["status"] == "conditional"
        assert launch["binding_guard"]["on_failure"] == "runtime_lookup"
        if name == "stable":
            assert launch["command_values"]["values"] == ["bounded"]
        else:
            assert launch["command_values"]["status"] == "unresolved"
            guards = [v["guard"] for v in launch["command_values"]["guarded_examples"]]
            expected = ["first", "later"] if name == "issue" else ["x"]
            assert guards == [
                {"argument": 0, "equals": v, "type": "str"} for v in expected
            ]
    (definition,) = nodes("py:entry:nested", "python_define_class")
    owner = definition["symbol"]
    assert definition["bases"][0]["name"] == "py:entry:Base"
    assert definition["class_protocol"]["status"] == "unresolved"
    assert functions[definition["body_symbol"]]["owner_class"] == owner
    assert functions[owner + ".selected"]["owner_class"] == owner
    (method_return,) = nodes(owner + ".selected", "return")
    assert method_return["value"]["cell"]["owner"] == "py:entry:nested"
    assert (
        nodes("py:entry:selected", "return")[0]["value"]["name"] == "py:entry:command"
    )


def test_python_process_handle_and_stream_ports(tmp_path):
    from scar.frontend import build

    entry, out = tmp_path / "entry.py", tmp_path / "graph"
    entry.write_text("""from subprocess import Popen, PIPE
def work(cmd, more, flag, other):
    p = Popen(cmd, stdout=PIPE, cwd=other.cwd(), shell=flag)
    q = p
    wait = p.wait
    p = Popen(cmd, stdin=PIPE, stderr=PIPE)
    wait()
    q.poll()
    for line in q.stdout: pass
    p.stdin
    p.stderr
    p.communicate(input=b"")
    if flag: chosen = q
    else: chosen = p
    chosen.wait()
    for item in more:
        Popen(item).wait()
    other.wait()
""")
    build(entry, out)
    (unit,) = json.loads((out / "graph.json").read_bytes())["units"]
    shard = json.loads((out / unit["graph"]["path"]).read_bytes())
    work = next(
        f for f in shard["functions"] if f["signature"]["name"].endswith(":work")
    )
    nodes = work["nodes"]
    ports = {n["location"]["line"]: n for n in nodes if n["op"] == "process_launch"}
    assert set(ports) == {3, 6, 17}
    assert ports[3]["cwd"] == {"operand": ["keywords", 1, "value"]}
    assert ports[3]["shell"] == {"operand": ["keywords", 2, "value"]}
    assert (
        sum(n["op"] == "python_invoke" and n["location"]["line"] == 3 for n in nodes)
        == 1
    )
    sites = {line: n["result_handle"]["site"] for line, n in ports.items()}
    assert len(set(sites.values())) == 3
    assert all(
        n["result_handle"]["instance"] == "per_successful_invocation"
        for n in ports.values()
    )
    assert set(work["process_operations"]) == {
        i for i, n in enumerate(nodes) if n["op"].startswith("process_")
    }

    operations = {
        n["location"]["line"]: n
        for n in nodes
        if n["op"]
        in {"process_wait", "process_communicate", "process_poll", "process_stream"}
    }
    assert list(sorted(operations)) == [7, 8, 9, 10, 11, 12, 15, 17]
    expected = {7: {3}, 8: {3}, 9: {3}, 10: {6}, 11: {6}, 12: {6}, 15: {3, 6}, 17: {17}}
    for line, row in operations.items():
        assert set(row["handle"]["sites"]) == {sites[i] for i in expected[line]}
        assert row["binding_guard"]["on_failure"] == "runtime_lookup"
        receiver = row
        for field in row["handle"]["operand"]:
            receiver = nodes[receiver[field]["port"]]
        if line == 7:
            assert (
                receiver["name"] == "wait"
                and row["handle"]["projection"] == "bound_receiver"
            )
        elif line == 17:
            assert (
                receiver["op"] == "process_launch"
                and receiver["result_handle"]["site"] == sites[17]
            )
        else:
            assert receiver["op"] == "var"
    assert not operations[8]["waits_for_exit"] and operations[12]["waits_for_exit"]
    assert {
        line: (operations[line]["channel"], operations[line]["direction"])
        for line in (9, 10, 11)
    } == {
        9: ("stdout", "child_to_parent"),
        10: ("stdin", "parent_to_child"),
        11: ("stderr", "child_to_parent"),
    }
    assert all(operations[n]["channel_binding"]["route"] == "pipe" for n in (9, 10, 11))
    assert any(
        n["op"] == "python_invoke" and n["location"]["line"] == 18 for n in nodes
    )


def test_fortran_nonpreprocessed_identity_tracks_opened_includes(tmp_path, monkeypatch):
    import pytest

    from scar.frontend.languages.fortran import facts

    tracer = shutil.which("strace")
    if (
        not tracer
        or subprocess.run(
            [tracer, *facts._OPEN_TRACE, "-o", str(tmp_path / "probe"), "true"]
        ).returncode
    ):
        pytest.skip("observing the selected GNU frontend requires local ptrace")
    source = tmp_path / "value.f90"
    outer = tmp_path / "external declarations"
    inner = tmp_path / "constants"
    source.write_text(
        "integer function value()\ninclude 'external declarations'\nvalue=n\nend\n"
    )
    outer.write_text("include 'constants'\n")
    inner.write_text("integer, parameter :: n=3\n")

    def parse():
        parsed = fortran.parse_unit(source, root=tmp_path)
        contract = parsed.source_numeric_contract
        assert "-cpp" not in contract["validated_command"]
        return parsed.numeric_semantics_sha256, contract["semantic_capture"]

    first, captured = parse()
    assert first and first == parse()[0]
    assert captured["coverage"]["source_includes"] == "same_frontend_read_inventory"
    observed = {row["path"]: row for row in captured["inputs"]}
    for path in (source, outer, inner):
        assert observed[str(path)]["evidence"] == "gnu_frontend_read"
    assert set(captured["open_witness"]["processes"]) == {"driver", "frontend"}
    # Even an edit the compiler erases from its parse dump changes the selected input.
    inner.write_text("integer, parameter :: n=3 ! revised comment\n")
    revised, after = parse()
    assert revised and revised != first
    assert captured["stdout_sha256"] == after["stdout_sha256"]

    module = tmp_path / "parameters.f90"
    module.write_text("module parameters\ninteger, parameter :: n=5\nend module\n")
    subprocess.run(["gfortran", "-fsyntax-only", str(module)], cwd=tmp_path, check=True)
    source.write_text("integer function value()\nuse parameters\nvalue=n\nend\n")
    key, imported = parse()
    module_path = str(tmp_path / "parameters.mod")
    assert module_path in {item["path"] for item in imported["inputs"]}
    module_open = next(
        row
        for row in imported["open_witness"]["opened_files"]
        if row["path"] == module_path
    )
    if module_open["device_inode"] is None:
        assert key is None
        assert {
            "reason": "compiler_open_identity_unobserved",
            "path": module_path,
        } in imported["unresolved"]
    real_which = shutil.which
    monkeypatch.setattr(
        facts.shutil,
        "which",
        lambda name: None if name == "strace" else real_which(name),
    )
    missing_observer, fallback = parse()
    assert missing_observer is None
    assert (
        fallback["coverage"]["source_includes"]
        == "unresolved_without_preprocessor_dependency_inventory"
    )


def test_fortran_io_uses_gnu_filename_and_preserves_source_operand(tmp_path):
    from scar.contract import walk_operations
    from scar.frontend.languages.fortran import normalize_unit, parse_unit

    cwd, src = tmp_path / "build", tmp_path / "src"
    cwd.mkdir()
    src.mkdir()
    path = src / "trace.F90"
    path.write_text(
        'program app\nprint *, __FILE__\n#line 27 "virtual.f90"\nwrite(98,*) 10\nend\n'
    )
    spelling = "../src/./trace.F90"
    parsed = parse_unit(
        path,
        ["-cpp"],
        root=tmp_path,
        working_directory=str(cwd),
        source_operand=spelling,
    )
    path.unlink()
    unit = normalize_unit(parsed)
    assert unit["diagnostics"] == []
    strings = [
        n["value"] for n in walk_operations(unit["functions"]) if n["op"] == "string"
    ]
    assert strings == ["virtual.f90", spelling, "virtual.f90"]
    assert parsed.io_filenames == {2: "virtual.f90", 27: "virtual.f90"}


def test_fortran_capture_then_offline_normalization(tmp_path, monkeypatch):
    from fparser.two import utils
    from fparser.two.parser import ParserFactory

    source = tmp_path / "stream.f90"
    source.write_text(
        "subroutine write_number(path, n)\n"
        "  character(*), intent(in) :: path\n"
        "  integer, intent(in) :: n\n"
        "  open(unit=42, file=path, status='replace', form='unformatted')\n"
        "  write(42) n\n"
        "  close(42)\n"
        "end subroutine\n"
        "real function real_power(i,x)\n"
        "  integer :: i\nreal :: x\nreal_power = i ** x\nend\n"
        "real(8) function mixed_power(x,y)\n"
        "  real :: x\nreal(8) :: y\nmixed_power = x ** y\nend\n"
        "integer function integer_power(i,j)\n"
        "  integer :: i,j\ninteger_power = i ** j\nend\n"
    )
    facts = fortran.parse_unit(source, (), root=tmp_path)
    assert not facts.diagnostics
    assert set(facts.target.files) == {
        ("open", ("file", "form", "status")),
        ("close", ()),
    }
    source.unlink()
    output = _offline(
        monkeypatch, fortran, facts, (ParserFactory, "create"), (utils, "walk")
    )
    assert not output["diagnostics"]
    by_name = {f["name"]: f for f in output["functions"]}
    routine = by_name["write_number_"]
    assert routine["name"] == "write_number_"
    assert routine["params"][-1]["character_of"] == "path"
    assert routine["params"][-1]["type"] == facts.target.size_type
    assert set(f["name"] for f in output["externs"]) == {
        "_gfortran_st_open",
        "_gfortran_st_write",
        "_gfortran_transfer_integer_write",
        "_gfortran_st_write_done",
        "_gfortran_st_close",
    }
    from scar.contract import walk_operations

    for name, source_types, normalized in (
        ("real_power_", ["i32", "f32"], ["f32", "f32"]),
        ("mixed_power_", ["f32", "f64"], ["f64", "f64"]),
        ("integer_power_", ["i32", "i32"], ["i32", "i32"]),
    ):
        power = next(
            n
            for n in walk_operations(by_name[name]["body"])
            if n.get("op") == "intrinsic" and n.get("name") in {"pow", "powi"}
        )
        assert [a["type"] for a in power["args"]] == normalized
        assert power["source_intrinsic_contract"]["argument_types"] == source_types
        assert (
            power["source_intrinsic_contract"]["normalized_argument_types"]
            == normalized
        )
        if name != "integer_power_":
            assert power["args"][0]["op"] == "cast"
            assert power["args"][0]["value"]["type"] == source_types[0]
        else:
            assert power["name"] == "powi"


def test_definition_parameter_access_preconditions_are_source_proven(
    tmp_path, monkeypatch
):
    from scar.frontend.languages import c_family

    declarations = """
typedef double * Q writer;
void add(writer z, const double * Q a, const double * Q b) { *z = *a + *b; }
void plain(double * Q p);
void plain(double * p) { double * Q local = p; *local = 1.0; }
void nested(double * Q * p) {}
"""
    for language, suffix, dialect, qualifier in (
        ("c", ".c", "c17", "restrict"),
        ("c++", ".cpp", "c++17", "__restrict__"),
    ):
        source = tmp_path / ("arguments" + suffix)
        source.write_text(
            'extern "C" {' + declarations + "}" if language == "c++" else declarations
        )
        facts = c_family.parse_unit(
            source,
            ["-std=" + dialect, "-DQ=" + qualifier],
            root=tmp_path,
            language=language,
        )
        source.unlink()
        unit = _offline(monkeypatch, c_family, facts)
        assert not unit["diagnostics"]
        by_name = {fn["name"]: fn["params"] for fn in unit["functions"]}
        assert all(
            "access_precondition" not in p
            for name in ("plain", "nested")
            for p in by_name[name]
        )
        assert all(
            p["access_precondition"]["basis"]["evidence"]["restrict_qualified"]
            for p in by_name["add"]
        )
        expected = {
            p["name"]: {
                k: v for k, v in p["access_precondition"].items() if k != "basis"
            }
            for p in by_name["add"]
        }
        assert {
            n for n, fact in expected.items() if fact["readonly_when_accessed"]
        } == {"a", "b"}

    source = tmp_path / "arguments.f90"
    source.write_text(
        "subroutine add(z,a,b,t,v) bind(c)\nuse iso_c_binding\n"
        "real(c_double), intent(out) :: z\nreal(c_double), intent(in) :: a(*), b(1)\n"
        "real(c_double), target :: t\nreal(c_double), value :: v\n"
        "z = a(1) + b(1) + t + v\nend subroutine\n"
        "subroutine relay(x)\nreal(8), intent(in)::x(4)\n"
        "call write_scalar(x(2))\ncall write_pair(x(1:4:2))\nend subroutine\n"
    )
    facts = fortran.parse_unit(source, (), root=tmp_path)
    source.unlink()
    unit = _offline(monkeypatch, fortran, facts)
    assert not unit["diagnostics"]
    add, relay = unit["functions"]
    params = {p["name"]: p for p in add["params"]}
    assert {n for n, p in params.items() if "access_precondition" in p} == set(expected)
    assert all(
        {k: v for k, v in params[n]["access_precondition"].items() if k != "basis"}
        == claim
        for n, claim in expected.items()
    )
    assert (
        params["a"]["access_precondition"]["basis"]["evidence"]["array_form"]
        == "assumed_size"
    )
    assert (
        params["b"]["access_precondition"]["basis"]["evidence"]["array_form"]
        == "explicit_shape"
    )
    assert (
        not params["a"]["elementwise_noalias"]
        and not params["b"]["elementwise_noalias"]
    )
    assert params["a"]["type"] == {"kind": "ptr", "to": {"kind": "const", "of": "f64"}}
    (native,) = relay["params"]
    assert native["intent"] == "in" and native["type"] == {"kind": "ptr", "to": "f64"}
    assert native["access_precondition"]["readonly_when_accessed"] is False
    scalar_call, packed, _ = relay["body"]
    pair_call, scatter = packed["body"][-2:]
    calls = [scalar_call["value"], pair_call["value"]]
    assert [c["callee"] for c in calls] == ["write_scalar_", "write_pair_"]
    assert all(
        c["parameter_types"] == [native["type"]]
        and c["parameter_type_basis"] == "actual_argument_abi"
        for c in calls
    )
    assert (
        scatter["op"] == "map" and scatter["body"][0]["target"]["base"]["name"] == "x"
    )


def test_clang_localizes_unmodeled_declarations_attributes_and_numeric_kinds(
    tmp_path, monkeypatch
):
    from scar.frontend.graph import unit_graph
    from scar.frontend.languages import c_family

    source = tmp_path / "partial.cpp"
    source.write_text(
        "using I = int;\n"
        "static_assert(sizeof(I) == sizeof(int));\n"
        "struct R { [[maybe_unused]] int n; friend int owned(R r){return r.n;} };\n"
        'extern "C" [[nodiscard]] int good(){return 7;}\n'
        'asm("");\n'
        "[[gnu::weak]] int optional_definition(){return 2;}\n"
        "bool has_add(){return requires(int x){x+x;};}\n"
        "_Float16 half_value(_Float16 x){return x;}\n"
    )
    facts = c_family.parse_unit(source, ["-std=c++20"], root=tmp_path, language="c++")
    source.unlink()
    normalized = _offline(monkeypatch, c_family, facts)
    graph = unit_graph(normalized, "test-revision")
    coverage = graph["source_coverage"]
    assert coverage["language"] == "c++" and not coverage["complete"]
    assert {r["kind"] for r in coverage["records"] if r["status"] == "transparent"} >= {
        "TYPE_ALIAS_DECL",
        "STATIC_ASSERT",
        "WARN_UNUSED_RESULT_ATTR",
    }
    issues = [
        row["diagnostic"]
        for row in coverage["records"]
        if row["status"] == "incomplete"
    ]
    assert issues == normalized["diagnostics"]
    assert all(row["file"] == source.name and row["column"] > 0 for row in issues)
    assert [(row["line"], row["code"]) for row in issues] == [
        (5, "c_family.declaration.unexposed_decl"),
        (6, "c_family.attribute.unexposed_attr"),
        (7, "c_family.expression.cursor_154"),
        (8, "c_family.type.type_32"),
    ]
    assert all("symbol" not in row for row in issues[:2])
    names = {fn["signature"]["name"]: fn for fn in graph["functions"]}
    assert names["good"]["defined"]
    assert any(
        fn["defined"] and fn["signature"]["location"]["line"] == 3
        for fn in names.values()
    )
    for row in issues[2:]:
        fn = names[row["symbol"]]
        assert fn["source_defined"] and not fn["defined"]

    # A block typedef can evaluate a bound, unlike a file-scope type alias.
    c_source = tmp_path / "bound.c"
    c_source.write_text("int bound(int n){typedef int Row[n++];return n;}\n")
    captured = c_family.parse_unit(c_source, ["-std=c99"], root=tmp_path, language="c")
    c_source.unlink()
    c_unit = _offline(monkeypatch, c_family, captured)
    assert c_unit["source_coverage"]["language"] == "c"
    assert [row["code"] for row in c_unit["diagnostics"]] == [
        "c_family.declaration.dynamic_type_bound"
    ]


def test_captured_compiler_paths_preserve_macro_values_and_physical_source_locks(
    tmp_path,
):
    from hashlib import sha256

    from scar.contract import read_member
    from scar.frontend import build

    root = tmp_path / "source"
    for name in ("app", "obj", "include"):
        (root / name).mkdir(parents=True)
    sources = {
        "app/main.cc": (
            "#include <assert.h>\n#include <stdio.h>\n#include <origin.h>\n"
            'extern "C" const char *c_origin(void), *mapped(void);\n'
            "int main(int argc, char **) { assert(argc); puts(__FILE__);\n"
            "  puts(header_origin()); puts(c_origin()); puts(mapped()); return 0; }\n"
        ),
        "app/unit.c": (
            "const char *c_origin(void) { return __FILE__; }\n"
            '#line 80 "mapped.c"\n'
            "const char *mapped(void) { return __FILE__; }\n"
        ),
        "include/origin.h": "static const char *header_origin() { return __FILE__; }\n",
    }
    for name, body in sources.items():
        (root / name).write_text(body)
    (root / "Makefile").write_text(
        "program:\n"
        "\tcd app && g++ -I../include -c ./main.cc -o ../obj/main.o\n"
        "\tcd obj && gcc -c ../app/unit.c -o unit.o\n"
        "\tg++ obj/main.o obj/unit.o -o program\n"
    )
    events = tmp_path / "events.json"
    subprocess.run(
        ["bear", "intercept", "--force-preload", "--output", events, "--", "make"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    observed = subprocess.check_output([root / "program"], text=True).splitlines()
    assert observed == ["./main.cc", "../include/origin.h", "../app/unit.c", "mapped.c"]
    manifest = tmp_path / "request.json"
    manifest.write_text(
        json.dumps(
            dict(
                source_root="source",
                targets=["program"],
                build_trace=dict(provider="bear-3-intercept", path=events.name),
            )
        )
    )
    out = tmp_path / "graph"
    build(manifest, out)
    graph = json.loads((out / "graph.json").read_bytes())
    assert graph["coverage"]["totals"]["diagnostic_sites"] == 0
    profiles = read_member(out, graph["build"]["graph"])["compilation_profiles"]
    assert {
        p["path"]: (p["working_directory"], p["source_operand"]) for p in profiles
    } == {
        "app/main.cc": (str(root / "app"), "./main.cc"),
        "app/unit.c": (str(root / "obj"), "../app/unit.c"),
    }
    locked, spellings, assertion_files = {}, set(), []
    for unit in graph["units"]:
        profile = next(p for p in profiles if p["path"] == unit["path"])
        assert unit["flags"] == profile.get("source_flags", profile["flags"])
        assert unit["working_directory"] == profile["working_directory"]
        assert unit["source_operand"] == profile["source_operand"]
        locked.update(unit["source_facts"]["files"])
        functions = read_member(out, unit["graph"])["functions"]
        for fn in functions:
            spellings.update(n["value"] for n in fn["nodes"] if n["op"] == "string")
            if fn["signature"]["name"] == "mapped" and fn["defined"]:
                assert fn["signature"]["location"] == dict(
                    file="app/unit.c", line=3, column=13
                )
            for node in fn["nodes"]:
                if node.get("callee") == "__assert_fail":
                    assertion_files.append(fn["nodes"][node["args"][1]["port"]])
    assert assertion_files == [dict(op="string", value="./main.cc")]
    assert locked == {
        p: sha256(body.encode()).hexdigest() for p, body in sources.items()
    }
    assert set(observed) <= spellings


def test_clang_capture_locks_included_buffers_then_normalizes_offline(
    tmp_path, monkeypatch
):
    from hashlib import sha256

    from clang import cindex
    import pytest

    from scar.frontend.languages import c_family

    entry, header = tmp_path / "main.cpp", tmp_path / "config.h"
    nested = tmp_path / "constants.h"
    nested.write_bytes(b"#define OFFSET 0\r\n")
    header.write_bytes(
        b'#include "constants.h"\r\n#define ANSWER 7\r\n'
        b"struct Base { virtual int score(const int&) const = 0; };\r\n"
        b"struct Derived : Base { int score(const int& x) const override { return x + ANSWER; } };\r\n"
        b'extern "C" int evaluate(int (*predicate)(int), int value);\r\n'
        b"extern int threshold;\r\n"
    )
    entry.write_text(
        '#include "config.h"\n#include <fstream>\n'
        "static int positive(int threshold) { return threshold > ::threshold; }\n"
        "int main() { Derived d; const Base& b = d; int n = 3; return evaluate(positive, b.score(n)); }\n"
        'extern "C" void read_paths(const char* path, std::istream& erased,\n'
        "                         std::ifstream* other, std::ios::openmode mode) {\n"
        "  char line[32]; std::ifstream file(path); std::ifstream& alias = file;\n"
        "  alias.getline(line, 32); file.close(); other->open(path, mode);\n"
        "  other->getline(line, 32, ','); other->close(); other->open(path);\n"
        "  std::string text; std::getline(erased, text); erased.getline(line, 32);\n"
        "  std::ofstream output; output.open(path); output.close();\n"
        "}\n"
    )
    locks = {f.name: sha256(f.read_bytes()).hexdigest() for f in (entry, header)}
    options = dict(root=tmp_path, language="c++", source_hashes=locks)
    facts = c_family.parse_unit(entry, ["-std=c++17"], **options)
    assert "clang" in facts.compiler_id.lower()
    assert facts.source_digests == {
        **locks,
        nested.name: sha256(nested.read_bytes()).hexdigest(),
    }
    header.write_bytes(header.read_bytes().replace(b"ANSWER 7", b"ANSWER 8"))
    with pytest.raises(ValueError, match="config.h"):
        c_family.parse_unit(entry, ["-std=c++17"], **options)
    header.unlink()
    nested.unlink()
    entry.unlink()
    unit = _offline(
        monkeypatch, c_family, facts, (cindex.Index, "create"), (Path, "resolve")
    )
    assert not unit["diagnostics"]
    functions = {f["name"]: f for f in unit["functions"]}
    instance, base, _, result = functions["main"]["body"]
    assert functions[instance["object_lifetime"]["constructor"]]["method"]["implicit"]
    assert base["init"]["conversion"] == "base_subobject"
    callback, dispatch = result["value"]["args"]
    assert result["value"]["callee"] == "evaluate"
    assert callback["conversion"] == "function_decay"
    assert callback["value"]["name"] in functions
    compare = functions[callback["value"]["name"]]["body"][0]["value"]
    if compare["op"] == "cast":
        compare = compare[
            "value"
        ]  # bool comparison promoted to the predicate's int result
    assert compare["left"] == {"op": "var", "name": "threshold"}
    assert compare["right"] == {"op": "var", "name": "threshold", "scope": "global"}
    assert dispatch["op"] == "object_dispatch" and dispatch["owner"] == "Base"
    (binding,) = dispatch["args"]
    assert binding["conversion"] == "reference_binding"
    assert dispatch["parameter_types"] == [binding["type"]]
    assert binding["value"]["conversion"] == "qualification"
    assert binding["value"]["type"] == {
        "kind": "ref",
        "to": {"kind": "const", "of": "i32"},
    }
    (override,) = (
        f
        for f in functions.values()
        if f.get("method", {}).get("kind") == "virtual_method"
    )
    assert override["method"]["slot"] == dispatch["slot"]
    assert override["body"][0]["value"]["right"] == dict(
        op="literal", type="i32", value=7
    )
    from scar.contract import walk_operations

    streams = [
        op
        for statement in functions["read_paths"]["body"]
        for op in walk_operations(statement)
    ]
    events = [n for n in streams if "artifact_effect" in n]
    effects = [n["artifact_effect"] for n in events]
    assert [effect["interface"] for effect in effects] == [
        "stream.open",
        "stream.getline_into",
        "stream.close",
        "stream.open",
        "stream.getline_into",
        "stream.close",
        "stream.open",
    ]
    assert all(
        e["contract"]["source_profile"] == "cxx_standard_ifstream_char_v1"
        for e in effects
    )
    create, read, close, reopen, delimited, _, implicit_mode = effects
    assert create["contract"]["handle"] == "result" and create["operands"] == {
        "path": ["args", 0]
    }
    assert (
        create["defaults"]
        == implicit_mode["defaults"]
        == {"mode": {"kind": "ios_openmode", "flag": "in"}}
    )
    assert reopen["operands"] == {
        "stream": ["receiver"],
        "path": ["args", 0],
        "mode": ["args", 1],
    }
    assert read["operands"] == {
        "stream": ["receiver"],
        "dst": ["args", 0],
        "count": ["args", 1],
    }
    assert read["defaults"] == {"delimiter": {"kind": "widen_newline"}}
    assert events[1]["receiver"] == {"op": "var", "name": "alias"}
    assert delimited["operands"] == {**read["operands"], "delimiter": ["args", 2]}
    assert events[4]["receiver"]["operator"] == "deref"
    assert (
        close["operands"] == {"stream": ["receiver"]}
        and close["contract"]["result"] == "void"
    )
    assert all(n["effects"] == "unknown" for n in events)
    assert [
        n["name"]
        for n in streams
        if n["op"] == "library_call" and "artifact_effect" not in n
    ] == ["getline", "getline", "open", "close"]


def test_graph_target_ports_and_lexical_regions(tmp_path):
    from collections import defaultdict

    from scar.frontend.graph import function_graph

    entry = tmp_path / "entry.py"
    entry.write_text("""def bound(items, manager, obj, key):
    a, (b, c) = items
    with manager as (x, y):
        obj[key] = a
    try:
        a += b
    except ValueError as err:
        del c
    else:
        del b
    finally:
        del a
    return x
""")
    (facts,) = python.capture_project(entry, tmp_path).units
    function = next(
        f
        for f in python.normalize_unit(facts)["functions"]
        if f["name"] == "py:entry:bound"
    )
    graph = function_graph(function, "bindings")
    modes, domains = defaultdict(list), defaultdict(set)
    for access in graph["accesses"]:
        node = graph["nodes"][access["node"]]
        if node["op"] in {"var", "index"}:
            name = node.get("name", node["op"])
            modes[name].append(access["mode"])
            domains[name].add(access["domain"])
    assert modes == {
        "a": ["write", "read", "read_write", "write"],
        "b": ["write", "read", "write"],
        "c": ["write", "write"],
        "err": ["write", "write"],  # handler entry and its mandatory cleanup
        "x": ["write", "read"],
        "y": ["write"],
        "items": ["read"],
        "manager": ["read"],
        "obj": ["read"],
        "key": ["read"],
        "index": ["write"],
    }
    assert all(len(bindings) == 1 for bindings in domains.values())
    assert domains["index"] == {"memory"}

    # Typed declarations can shadow the outer slot inside independent regions.
    slot = dict(op="var", name="slot")
    local = dict(op="decl", name="slot", type="i32")
    read = dict(op="expr", value=slot)
    handler = dict(op="exception_handler", body=[], cleanup=[local, read])
    protected = dict(
        op="try", body=[], handlers=[handler], **{"finally": [local, read]}
    )
    graph = function_graph(
        dict(
            name="scopes",
            body=[
                {**local, "storage": "static"},
                protected,
                dict(op="return", value=dict(op="unary", operator="addr", value=slot)),
            ],
        ),
        "bindings",
    )
    declarations = [e for e in graph["accesses"] if e["mode"] == "declare"]
    assert [(e["storage"], e["lexical"]) for e in declarations] == [
        ("static", False),
        ("auto", True),
        ("auto", True),
    ]
    outer, cleanup, final = (e["domain"] for e in declarations)
    uses = [e for e in graph["accesses"] if e["mode"] != "declare"]
    assert [(e["domain"], e["mode"]) for e in uses] == [
        (cleanup, "read"),
        (final, "read"),
        (outer, "address"),
    ]


def test_contract_json_preserves_codepoints_and_mapping_keys(tmp_path):
    import pytest

    from scar.contract import (
        GRAPH,
        document,
        json_bytes,
        json_loads,
        member,
        open_product,
        read_member,
        write_json,
    )

    tag = "$scar.wire.v1"
    emoji, pair = "\U0001f600", "\ud83d\ude00"
    ordinary = {"text": ["a\0b", "科学", "é", emoji, r"\ud800", tag], "n": -0.0}
    old_bytes = (
        json.dumps(
            ordinary, separators=(",", ":"), sort_keys=True, allow_nan=False
        ).encode("ascii")
        + b"\n"
    )
    assert json_bytes(ordinary) == old_bytes
    assert json_loads(old_bytes) == ordinary
    assert json_bytes(emoji) != json_bytes(pair)
    colliding_keys = {emoji: 1, pair: 2, "\udc80": 3, "\ud800": 4}
    assert json_bytes(colliding_keys) == json_bytes(
        dict(reversed(colliding_keys.items()))
    )
    values = [
        colliding_keys,
        pair + "\0" + emoji + "\udfff",
        "\udcc3\udca9",
        {tag: ["text", "eda0bdedb880"]},
        {tag: ["map", []], pair: {tag: True}},
    ]
    receipt = member(tmp_path, "values", values)
    assert read_member(tmp_path, receipt) == values
    product = document(GRAPH, values=values, member=receipt)
    write_json(tmp_path / "graph.json", product)
    assert open_product(tmp_path, "graph.json", GRAPH)[2] == product
    assert (
        json.loads((tmp_path / "graph.json").read_bytes().decode("utf-8"))["schema"]
        == GRAPH
    )

    for corrupt in (
        {tag: ["text", "eda0Bdedb880"]},
        {tag: ["text", "c0af"]},
        {tag: ["text", "f09f9880"]},
        {tag: ["text", "eda0bd"], "extra": 0},
        {tag: ["map", [[tag, 1], [tag, 2]]]},
        {tag: ["map", [["z", 1], [tag, 2]]]},
        {tag: ["map", [[0, 1]]]},
        {tag: ["map", [["ordinary", 1]]]},
        {tag: ["invalid", None]},
    ):
        with pytest.raises(ValueError, match="SCAR wire"):
            json_loads(json.dumps(corrupt))


def test_fortran_source_units_account_for_module_and_submodule_coverage(
    tmp_path, monkeypatch
):
    from fparser.two import utils
    from fparser.two.parser import ParserFactory
    from scar.contract import validate_source_coverage

    source = tmp_path / "units.f90"
    text = (
        "module constants\n"
        "implicit none\nprivate\npublic :: increment\n"
        "integer, parameter :: offset=2\ncontains\n"
        "integer function increment(n)\ninteger,intent(in)::n\n"
        "increment=n+offset\nend function\nend module\n"
        "module persistent\ninteger::counter\nend module\n"
        "module types\ntype::point\nreal::x\nend type\n"
        "integer,parameter :: after=3\nend module\n"
        "module parent\ninterface\nmodule subroutine step(x)\n"
        "integer,intent(inout)::x\nend subroutine\nend interface\nend module\n"
        "submodule(parent) child\nend submodule\n"
        "subroutine remaining(n)\ninteger::n\nn=n+1\nend subroutine\n"
    )
    source.write_text(text)
    facts = fortran.parse_unit(source, root=tmp_path)
    assert not facts.diagnostics
    assert facts.source_coverage["complete"] is False
    validate_source_coverage(facts.source_coverage)
    source.unlink()
    output = _offline(
        monkeypatch, fortran, facts, (ParserFactory, "create"), (utils, "walk")
    )
    coverage = output["source_coverage"]
    validate_source_coverage(coverage)
    assert coverage["scope"] == "program_units_and_specification_constructs"
    units = {
        tuple(row["owner"]): row
        for row in coverage["records"]
        if row["role"] == "program_unit"
    }
    assert units[("constants",)]["status"] == "transparent"
    assert units[("constants", "increment")]["status"] == "modeled", units[
        ("constants", "increment")
    ].get("diagnostic")
    assert units[("remaining",)]["status"] == "modeled"
    assert units[("persistent",)]["status"] == "incomplete"
    assert units[("child",)]["kind"] == "Submodule"
    assert units[("child",)]["covers_subtree"] is True
    assert units[("child",)]["diagnostic"]["code"] == "fortran.program_unit"
    constants = next(
        row["facts"]["symbols"]
        for row in coverage["records"]
        if row["kind"] == "Type_Declaration_Stmt" and row["owner"] == ["constants"]
    )
    assert constants[0]["name"] == "offset"
    assert constants[0]["folded_value"] == "2"
    declarations = {
        (tuple(row["owner"]), row["kind"], row["location"]["line"]): row
        for row in coverage["records"]
        if row["role"] == "specification"
    }
    lines = text.splitlines()
    for owner, spelling, code in (
        ("persistent", "integer::counter", "fortran.module_state"),
        ("types", "type::point", "fortran.declaration"),
    ):
        line = lines.index(spelling) + 1
        row = next(
            row
            for (scope, _, at), row in declarations.items()
            if scope == (owner,) and at == line
        )
        assert row["status"] == "incomplete"
        assert row["diagnostic"]["code"] == code
        assert row["diagnostic"]["owner"] == [owner]
        assert row["diagnostic"]["file"].endswith("units.f90")
    assert any(
        row["status"] == "modeled"
        and row.get("facts", {}).get("symbols", [{}])[0].get("name") == "after"
        for row in declarations.values()
        if row["owner"] == ["types"]
    )
    functions = {fn["name"]: fn for fn in output["functions"]}
    assert set(functions) == {"__constants_MOD_increment", "remaining_"}
    assert all("body" in fn for fn in functions.values())
    assert coverage["complete"] is False
    assert any(
        issue["code"] == "fortran.program_unit" for issue in output["diagnostics"]
    )
    assert facts.source_coverage["records"] == coverage["records"]


def test_attach_import_observation_to_relocated_script_image(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from hashlib import sha256
    import os
    import sys

    import scar
    from scar.contract import read_member
    from scar.frontend import build

    source = tmp_path / "src"
    source.mkdir()
    first = source / "first.py"
    first.write_text("from helper import value\nprint(value)\n")
    (source / "second.py").write_text("print('another entry')\n")
    (source / "helper.py").write_text("value = 11\n")
    manifest = source / "request.json"
    manifest.write_text(
        json.dumps(
            dict(
                root_image="first",
                images=[
                    dict(id=name, entry="__main__", source=f"{name}.py", output=name)
                    for name in ("first", "second")
                ],
            )
        )
    )
    sample = tmp_path / "sample.json"
    env = dict(os.environ, PYTHONPATH=str(Path(scar.__file__).resolve().parents[1]))

    def invoke(module, *args):
        return subprocess.run(
            [sys.executable, "-B", "-m", module, *map(str, args)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    captured = invoke(
        "scar.frontend.languages.python.import_observation",
        "--source",
        first,
        "--line",
        1,
        "--out",
        sample,
        first,
    )
    assert captured.returncode == 0 and captured.stdout == "11\n", captured.stderr
    folder = tmp_path / "graph"
    build(manifest, folder)
    single = tmp_path / "single"
    build(first, single)
    shutil.rmtree(source)
    folder = folder.rename(tmp_path / "relocated")
    index = folder / "graph.json"
    before = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}

    def attach(index, image, raw=sample):
        return invoke(
            "scar.frontend.observation",
            "--graph",
            index,
            "--image",
            image,
            "--import-observation",
            raw,
        )

    wrong = attach(index, "second")
    assert wrong.returncode == 2 and "mismatched script bytes" in wrong.stderr
    standalone = invoke(
        "scar.frontend.observation", "--graph", single, "--import-observation", sample
    )
    assert standalone.returncode == 0, standalone.stderr
    assert (
        json.loads(Path(standalone.stdout.strip()).read_bytes())["source_facts"][
            "import_binding_sample"
        ]["image"]
        is None
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: attach(index, "first"), range(2)))
    assert all(run.returncode == 0 for run in results), [run.stderr for run in results]
    assert results[0].stdout == results[1].stdout
    derived = Path(results[0].stdout.strip())
    assert derived.parent == folder
    assert derived.name == f"graph-{sha256(derived.read_bytes()).hexdigest()}.json"
    assert attach(derived, "first").stdout == results[0].stdout
    document = json.loads(derived.read_bytes())
    ref = document["source_facts"].pop("import_binding_sample")
    assert document == json.loads(before[index]) and ref["image"] == "first"
    linked = read_member(folder, ref["evidence"])
    [event] = linked["events"]
    assert event["site"] and event["binding"] == "value"
    assert event["result"] == {"kind": "int", "value": 11}
    other = tmp_path / "different.json"
    other.write_text(json.dumps(dict(json.loads(sample.read_bytes()), dropped=1)))
    rejected = attach(derived, "first", other)
    assert (
        rejected.returncode == 2 and "different import observation" in rejected.stderr
    )
    after = {p for p in folder.rglob("*") if p.is_file()}
    assert after - before.keys() == {derived, folder / ref["evidence"]["path"]}
    assert all(p.read_bytes() == original for p, original in before.items())
    planned = invoke("scar", "plan", "--graph", derived, "--out", tmp_path / "plan")
    assert planned.returncode in (0, 2), planned.stderr
    assert (
        json.loads((tmp_path / "plan/plan.json").read_bytes())["graph_sha256"]
        == sha256(derived.read_bytes()).hexdigest()
    )


def test_plain_import_observation_precedes_frame_callback_and_joins_exact_instruction(
    tmp_path,
):
    from hashlib import sha256
    import os
    import sys

    import scar
    from scar.frontend import build
    from scar.frontend.observation import link_import_observation

    folder = tmp_path / "src"
    folder.mkdir()
    watched = folder / "observed.py"
    unused_names = ", ".join(f"name{i}" for i in range(270))
    watched.write_text(
        f"if False: {unused_names}\nimport subject as alias\nprint(alias.value)\n"
    )
    (folder / "subject.py").write_text("value = 11\n")
    entry = folder / "entry.py"
    entry.write_text(
        "import builtins, sys, types\n"
        "if 'warm' in sys.argv:\n"
        "    cached = types.ModuleType('subject')\n"
        "    cached.value = 37\n"
        "    sys.modules['subject'] = cached\n"
        "if 'deny' in sys.argv:\n"
        "    def denied(*args): raise RuntimeError('frame import was denied')\n"
        "    path = __file__.replace('entry.py', 'observed.py')\n"
        "    with open(path, 'rb') as stream: code = compile(stream.read(), path, 'exec')\n"
        "    exec(code, {'__name__':'observed', '__builtins__':dict(vars(builtins), __import__=denied)})\n"
        "else:\n"
        "    import observed\n"
    )
    out = tmp_path / "graph"
    build(entry, out)
    units = json.loads((out / "graph.json").read_bytes())["units"]
    entry_sha = sha256(entry.read_bytes()).hexdigest()
    env = dict(os.environ, PYTHONPATH=str(Path(scar.__file__).resolve().parents[1]))
    sites = []
    for mode, state, kind in (
        ("cold", "absent", "builtin_function"),
        ("warm", "present_module", "builtin_function"),
        ("deny", "absent", "python_function"),
    ):
        sample = tmp_path / f"{mode}.json"
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "scar.frontend.languages.python.import_observation",
                "--source",
                str(watched),
                "--line",
                "2",
                "--out",
                str(sample),
                str(entry),
                mode,
            ],
            cwd=folder,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if mode == "deny":
            assert result.returncode and "frame import was denied" in result.stderr
        else:
            assert result.returncode == 0, result.stderr
            assert result.stdout == ("11\n" if mode == "cold" else "37\n")
        raw = sample.read_bytes()
        linked = link_import_observation((None, entry_sha, raw), units, None, out)
        assert linked["scope"] == "instrumented_script_import_sites"
        [event] = linked["events"]
        assert event["phase"] == "before_IMPORT_NAME" and event["owner"] == "observed"
        assert (
            event["registry_state"] == state
            and event["import_callable"]["kind"] == kind
        )
        assert event["import_callable"]["qualname"] == (
            "denied" if mode == "deny" else "__import__"
        )
        assert event["instruction"]["name"] == "subject" and "result" not in event
        assert event["instruction"]["index"] > 255
        sites.append(event["site"])
    assert len(set(sites)) == 1
    ambiguous = link_import_observation(
        (None, entry_sha, raw), units + units, None, out
    )
    assert ambiguous["events"][0]["unresolved"] == "ambiguous_source_port"
    for field in (
        "code_sha256",
        "source_sha256",
        "offset",
        "index",
        "name",
        "owner",
        "column",
    ):
        altered = json.loads(raw)
        [event] = altered["events"]
        if field == "source_sha256":
            altered[field] = "0" * 64
        elif field == "code_sha256":
            event[field] = "0" * 64
        elif field in ("name", "owner"):
            (event["instruction"] if field == "name" else event)[field] = "another"
        else:
            (event["location"] if field == "column" else event["instruction"])[
                field
            ] += 1
        linked = link_import_observation(
            (None, entry_sha, json.dumps(altered).encode()), units, None, out
        )
        assert linked["events"][0]["unresolved"] == "no_matching_port"

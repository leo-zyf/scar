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


def test_python_import_precedence_and_runtime_facts(tmp_path, monkeypatch):
    import codecs
    from importlib.machinery import BuiltinImporter, FrozenImporter
    import os

    from scar.contract import walk_operations

    entry = tmp_path / "entry.py"
    entry.write_text(
        "import os, shutil, sys, time, math, subprocess\n"
        "from os.path import join as combine\nopen('never-executed')\n"
    )
    (facts,) = python.capture_project(entry, tmp_path).units
    unit = _offline(monkeypatch, python, facts)
    nodes = list(walk_operations(unit["functions"][0]["partial_body"]))
    imported = next(n for n in nodes if n.get("module") == "os.path")
    assert imported["bound_value"]["external"] == "os.path.join"
    source = imported["source_provider"]
    assert source["module"] == "os" and source["status"] == "verified_stdlib"
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
    project = python.capture_project(entry, tmp_path)
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

    from scar.contract import walk_operations

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
        "import json.decoder\nimport scar_missing_package.part\n"
    )
    capture = python.capture_project(entry, project)
    assert set(capture.catalog) == {"entry"}
    (facts,) = capture.units
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
    assert facts.providers["json.decoder"]["status"] == "verified_stdlib"
    assert facts.providers["scar_missing_package.part"]["status"] == "unresolved"
    entry.unlink()
    shutil.rmtree(installed)
    neutral = _offline(monkeypatch, python, facts)
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


def test_external_exports_follow_aliases_and_preserve_source_evidence(
    tmp_path, monkeypatch
):
    from base64 import urlsafe_b64encode
    from hashlib import sha256
    import sys

    project, installed = tmp_path / "project", tmp_path / "installed"
    project.mkdir()
    package = installed / "source_library"
    package.mkdir(parents=True)
    sources = {
        "source_library/__init__.py": "from .unknown import *\nfrom .first import produce as selected\nsaved = selected\nfrom .second import produce as selected\n",
        "source_library/unknown.py": "__all__ = unknown_at_runtime()\n",
        "source_library/first.py": 'def produce(x):\n    """not a statement site"""\n    def nested():\n        return 999\n    return x + 1\n',
        "source_library/second.py": "def produce(x):\n    return x + 2\n",
    }
    metadata = installed / "source_library-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: source-library\nVersion: 1.0\n"
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
    entry.write_text("import source_library as api\napi.saved(1)\napi.selected(2)\n")
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
    (routine,) = output["functions"]
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
        '#include "config.h"\n'
        "static int positive(int threshold) { return threshold > ::threshold; }\n"
        "int main() { Derived d; const Base& b = d; int n = 3; return evaluate(positive, b.score(n)); }\n"
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

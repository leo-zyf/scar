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

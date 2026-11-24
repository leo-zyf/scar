"""Test the three public stages in mutually isolated interpreter processes."""

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from ctypes import CDLL, c_double
import runpy
import shlex

import pytest

import scar

_PHASE_COMMAND = """
import sys
class BlockOtherStages:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == blocked or fullname.startswith(blocked + '.') for blocked in forbidden):
            raise ImportError('cross-stage import: ' + fullname)

phase = sys.argv[1]
forbidden = {'scar.' + name for name in ('frontend', 'middle', 'backend') if name != phase}
if phase != "frontend":
    forbidden.update(("numpy", "matplotlib"))
sys.meta_path.insert(0, BlockOtherStages())
from scar.cli import main
raise SystemExit(main(sys.argv[2:]))
"""


def _run(command, folder, *, input=None, returncode=0):
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(scar.__file__).resolve().parents[1]),
        PYTHONDONTWRITEBYTECODE="1",
    )
    completed = subprocess.run(
        command, cwd=folder, env=env, input=input, capture_output=True, timeout=120
    )
    assert completed.returncode == returncode, completed.stderr.decode(errors="replace")
    return completed.stdout


def _stage(folder, phase, *arguments, returncode=0):
    return json.loads(
        _run(
            [sys.executable, "-B", "-c", _PHASE_COMMAND, phase, *map(str, arguments)],
            folder,
            returncode=returncode,
        )
    )


def _payloads(graph, document):
    return [
        json.loads((graph / unit["graph"]["path"]).read_bytes())
        for unit in document["units"]
    ]


@pytest.mark.parametrize("controller", [False, True])
def test_make_capture_preserves_mixed_profiles_and_independent_stages(
    tmp_path, controller
):
    from scar.contract import read_member

    source = tmp_path / "source"
    source.mkdir()
    (source / "constants.h").write_text("#define SERIES_INPUT 5\n")
    (source / "caller.c").write_text(
        '#include <stdio.h>\n#include "constants.h"\n'
        "extern int triangle(int);\n"
        'int main(void) { printf("%d\\n", triangle(SERIES_INPUT)); return 0; }\n'
    )
    (source / "series.f90").write_text(
        "module integer_series\nuse iso_c_binding\nimplicit none\ncontains\n"
        "function triangle(n) bind(C) result(v)\n"
        "integer(c_int), value :: n\ninteger(c_int) :: v\n"
        "v=n*(n+1)/2\nend function\nend module\n"
    )
    (source / "Makefile").write_text(
        "hybrid: caller.o series.o\n\tgfortran -o $@ $^ -lm\n"
        "caller.o: caller.c constants.h\n"
        "\tgcc -std=c89 -MMD -MF caller.d -O1 -c $< -o $@\n"
        "series.o: series.f90\n"
        "\tgfortran -std=f2008 -J. -O1 -c $< -o $@\n"
    )
    trace = tmp_path / "events.json"
    _run(
        [
            "bear",
            "intercept",
            "--force-preload",
            "--output",
            trace,
            "--",
            "make",
            "-j2",
            "hybrid",
        ],
        source,
    )
    expected = _run([source / "hybrid"], source)
    assert expected == b"15\n"
    project = {}
    if controller:
        driver = source / "driver.py"
        driver.write_text(
            "import subprocess\n"
            'child = subprocess.Popen(["./hybrid"], stdout=subprocess.PIPE, text=True)\n'
            'for line in child.stdout:\n    print(line, end="")\n'
            "print(child.wait())\n"
        )
        expected = _run([sys.executable, driver], source)
        assert expected == b"15\n0\n"
        project = dict(
            root_image="driver",
            images=[
                dict(id="driver", output="driver", entry="__main__", source="driver.py")
            ],
        )
    manifest = tmp_path / "request.json"
    manifest.write_text(
        json.dumps(
            dict(
                source_root="source",
                entry="main",
                targets=["hybrid"],
                build_trace=dict(provider="bear-3-intercept", path=trace.name),
                **project,
            )
        )
    )
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--manifest", manifest, "--out", graph)
    document = json.loads((graph / "graph.json").read_bytes())
    profiles = {unit["path"]: unit for unit in document["units"]}
    assert "-std=c89" in profiles["caller.c"]["flags"]
    assert "-std=f2008" in profiles["series.f90"]["flags"]
    assert not document["build"]["diagnostics"]
    build_graph = read_member(graph, document["build"]["graph"])
    assert build_graph["nodes"] and build_graph["edges"]
    assert any(r["kind"] == "LIBRARY_REQUEST" for r in document["resources"])
    assert any(r["kind"] == "ELF_LIBRARY_DEPENDENCY" for r in document["resources"])
    assert len(document["images"]) == (2 if controller else 1)
    (binding,) = document["build"]["bindings"]
    assert binding["image_id"] == "hybrid"
    assert next(
        node
        for node in build_graph["nodes"]
        if node["id"] == binding["target_artifact"]
    )["path"] == str(source / "hybrid")
    shutil.rmtree(source)
    trace.unlink()
    manifest.unlink()
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    shutil.rmtree(graph)
    result = _stage(tmp_path, "backend", "compile", "--ir", plan, "--out", delivery)
    shutil.rmtree(plan)
    executable = Path(result["executable"])
    assert _run([executable], executable.parent) == expected


def test_mpi_controller_joins_captured_worker_and_preserves_launch_conventions(
    tmp_path, monkeypatch
):
    if os.geteuid() == 0:
        monkeypatch.setenv("OMPI_ALLOW_RUN_AS_ROOT", "1")
        monkeypatch.setenv("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    source = tmp_path / "source"
    source.mkdir()
    (source / "worker.c").write_text(
        "#include <mpi.h>\n#include <stdio.h>\n"
        "int main(int argc, char **argv) {\n"
        "int rank, size; MPI_Init(&argc, &argv);\n"
        "MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);\n"
        'if (rank == 0) printf("ranks=%d argv0=%s value=%s\\n", size, argv[0], argv[1]);\n'
        "MPI_Finalize(); return 0; }\n"
    )
    (source / "Makefile").write_text(
        "worker: worker.c\n\tmpicc -std=c99 -O2 -o $@ $<\n"
    )
    (source / "driver.py").write_text(
        "import subprocess\n"
        'first = subprocess.Popen(["mpiexec", "-n", "2", "./worker", "data"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n'
        'for line in first.stdout:\n    print(line, end="")\n'
        "print(first.wait())\n"
        'second = subprocess.Popen("mpiexec -n 2 ./worker data", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n'
        'for line in second.stdout:\n    print(line, end="")\n'
        "print(second.wait())\n"
    )
    trace = tmp_path / "events.json"
    _run(
        [
            "bear",
            "intercept",
            "--force-preload",
            "--output",
            trace,
            "--",
            "make",
            "-j2",
        ],
        source,
    )
    expected = _run([sys.executable, source / "driver.py"], source)
    assert expected == b"ranks=2 argv0=./worker value=data\n0\n" * 2
    original_worker = (source / "worker").read_bytes()
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            dict(
                source_root="source",
                root_image="driver",
                images=[
                    dict(
                        id="driver",
                        output="driver",
                        entry="__main__",
                        source="driver.py",
                    )
                ],
                build_trace=dict(provider="bear-3-intercept", path=trace.name),
            )
        )
    )
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(
        tmp_path,
        "frontend",
        "build",
        "--manifest",
        request,
        "--out",
        graph,
        "--jobs",
        2,
    )
    document = json.loads((graph / "graph.json").read_bytes())
    assert document["root_image"] == "driver"
    assert {image["id"] for image in document["images"]} == {"driver", "worker"}
    assert document["build"]["bindings"][0]["image_id"] == "worker"
    shutil.rmtree(source)
    trace.unlink()
    request.unlink()
    logical = _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    assert logical["status"] == "READY"
    shutil.rmtree(graph)
    result = _stage(
        tmp_path,
        "backend",
        "compile",
        "--ir",
        plan,
        "--out",
        delivery,
        "--physical",
        "serial",
    )
    shutil.rmtree(plan)
    executable = Path(result["executable"])
    assert _run([executable], executable.parent) == expected
    copied = tmp_path / "working-directory"
    copied.mkdir()
    shutil.copy2(executable.parent / "worker", copied / "worker")
    assert _run([executable], copied) == expected
    # The launcher must never fall back to the original executable even though
    # it is runnable, implements the same MPI calls and prints the same output.
    (copied / "worker").write_bytes(original_worker)
    failed = subprocess.run([executable], cwd=copied, capture_output=True, timeout=20)
    assert failed.returncode != 0 and not failed.stdout
    assert b"published executable" in failed.stderr


def test_mpi_calls_survive_independent_stages_and_two_rank_execution(
    tmp_path, monkeypatch
):
    if os.geteuid() == 0:
        monkeypatch.setenv("OMPI_ALLOW_RUN_AS_ROOT", "1")
        monkeypatch.setenv("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    source = tmp_path / "source"
    source.mkdir()
    entry = source / "exchange.c"
    entry.write_text("""#include <mpi.h>
#include <stdio.h>
int main(int argc, char **argv) {
  int rank, size, provided, sent, received = -1, total = -1;
  MPI_Request requests[2];
  MPI_Status statuses[2];
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  sent = rank + 1;
  MPI_Irecv(&received, 1, MPI_INT, (rank + 1) % size, 17, MPI_COMM_WORLD, &requests[0]);
  MPI_Isend(&sent, 1, MPI_INT, (rank + size - 1) % size, 17, MPI_COMM_WORLD, &requests[1]);
  MPI_Waitall(2, requests, statuses);
  MPI_Allreduce(&received, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) printf("ranks=%d sum=%d\\n", size, total);
  MPI_Finalize();
  return total == size * (size + 1) / 2 ? 0 : 3;
}
""")
    original = source / "original"
    _run(["mpicc", "-std=c99", "-O2", entry, "-o", original], source)
    expected = _run(["mpiexec", "-n", "2", original], source)
    assert expected == b"ranks=2 sum=3\n"
    includes = shlex.split(_run(["mpicc", "--showme:compile"], source).decode())
    sdk = tmp_path / "sdk" / "include"
    sdk.mkdir(parents=True)
    mpi_header = next(
        Path(flag[2:]) / "mpi.h"
        for flag in includes
        if flag.startswith("-I") and (Path(flag[2:]) / "mpi.h").is_file()
    )
    for name in ("mpi.h", "mpi_portable_platform.h"):
        shutil.copy2(mpi_header.parent / name, sdk / name)
    request = source / "request.json"
    request.write_text(
        json.dumps(
            dict(
                entry="main",
                units=[
                    dict(
                        path=entry.name,
                        language="c",
                        flags=["-std=c99", "-I" + str(sdk), *includes],
                    )
                ],
            )
        )
    )
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--manifest", request, "--out", graph)
    document = json.loads((graph / "graph.json").read_bytes())
    headers = document["units"][0]["source_facts"]["external_headers"]
    assert any(
        Path(path).name == "mpi.h" and row["sha256"] for path, row in headers.items()
    )
    shutil.rmtree(source)
    shutil.rmtree(sdk.parent)
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    ir = json.loads((plan / "exec-ir.json").read_bytes())
    requirement = ir["module"]["operator_requirements"]["mpi"]
    assert {"MPI_Isend", "MPI_Irecv", "MPI_Waitall", "MPI_Allreduce"} <= set(
        requirement["interfaces"]
    )
    assert ir["module"]["resource_model"]["software_headers"] == headers
    shutil.rmtree(graph)
    result = _stage(
        tmp_path,
        "backend",
        "compile",
        "--ir",
        plan,
        "--out",
        delivery,
        "--physical",
        "auto",
    )
    shutil.rmtree(plan)
    physical = json.loads((delivery / "physical-plan.json").read_bytes())
    binding = physical["operator_backends"]["mpi"]
    assert binding["provider"] == "mpi-c"
    assert binding["source_header_evidence"] == "matched"
    assert binding["rank_selection"] == binding["transport_cost"] == "unmodeled"
    assert physical["worker_limit"] == 1
    assert physical["worker_policy"]["caps"]["mpi_intra_rank_limit"] == 1
    assert (
        physical["worker_policy"]["mpi"]["reason"]
        == "rank_placement_and_provided_thread_level_unverified"
    )
    assert json.loads((delivery / "project/mpi-sdk.json").read_bytes()) == binding
    assert _run(["mpiexec", "-n", "2", result["executable"]], tmp_path) == expected


def test_python_project_protocols_preserve_initialization_and_effect_order(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    entry = source / "entry.py"
    entry.write_text(
        "import helper\nfrom helper import echo as show\nimport helper as again\n"
        "print('科学\\0计算', sep='|', end=';\\n')\n"
        "print(show('left'), again.echo('right'), True, None, 42, sep='|')\n",
        encoding="utf-8",
    )
    (source / "helper.py").write_text(
        "print('initialize')\n"
        "def echo(value):\n"
        "    result = value + '✓'\n"
        "    print('call', end=':')\n"
        "    return result\n",
        encoding="utf-8",
    )
    expected = _run([sys.executable, "-B", entry], source)
    assert expected.count(b"initialize") == 1 and b"\x00" in expected
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--entry", entry, "--out", graph)
    shutil.rmtree(source)
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    shutil.rmtree(graph)
    result = _stage(tmp_path, "backend", "compile", "--ir", plan, "--out", delivery)
    shutil.rmtree(plan)
    assert _run([result["executable"]], tmp_path) == expected


def test_array_protocols_preserve_views_copies_and_lifetimes(tmp_path):
    entry = tmp_path / "arrays.py"
    entry.write_text("""import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
alias = a
row = a[-1]
row[0] = 7.0
copied = np.array(a)
negative = -a
a[0] = [8.0, 9.0]
print(a[1, 0] == 7.0, alias[0, 1] == 9.0, copied[0, 0] == 1.0, negative[1, 0] == -7.0)
def retain_row():
    local = np.array([[11.0, 12.0]])
    return local[0]
saved = retain_row()
print(saved[-1] == 12.0)
scalar = np.zeros(())
empty = np.zeros((0, 3))
print(scalar[()] == 0.0)
x = 1.0
items = [x, x]
x = 9.0
snapshot = np.array(items)
print(snapshot[0] == 1.0)
""")
    expected = _run([sys.executable, "-B", entry], tmp_path)
    assert expected == b"True True True True\nTrue\nTrue\nTrue\n"
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--entry", entry, "--out", graph)
    entry.unlink()
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    shutil.rmtree(graph)
    result = _stage(tmp_path, "backend", "compile", "--ir", plan, "--out", delivery)
    shutil.rmtree(plan)
    assert _run([result["executable"]], tmp_path) == expected
    physical = json.loads((delivery / "physical-plan.json").read_bytes())
    selected = physical["operator_backends"]["ndarray"]
    assert selected["provider"] == "xtensor-cpu" and selected["device"] == "cpu"
    target = json.loads((delivery / "target-ir.json").read_bytes())
    assert [h["output"] for h in target["sdk_headers"]] == [
        "include/scar/array.hpp",
        "include/scar/exceptions.hpp",
    ]


def test_numeric_domains_broadcast_and_array_display(tmp_path):
    entry = tmp_path / "entry.py"
    entry.write_text("""import numpy as np
a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
row = a[0]
independent = a + np.array([10.0, 20.0, 30.0])
a *= 2.0
print(a)
print(row)
print(independent)
print(12.0 / a)
print(a ** 2)
def scalar():
    return a[0, 1]
print(scalar() / 2)
empty = np.zeros((0, 3))
print(empty + np.zeros((1, 3)))
zero = np.zeros(())
print(zero + 1.0)
print(['name', 1, 2.5, (True, None)])
print(2.0 ** (5 - 1))
def signed_zero(flag):
    if flag:
        value = -0.0
    else:
        value = 0.0
    return value ** 3
print(signed_zero(True), signed_zero(False))
try:
    print(0.0 ** -1)
except ZeroDivisionError:
    print('zero power')
try:
    print(1e200 ** 2)
except OverflowError:
    print('overflow')
""")
    expected = _run([sys.executable, "-B", entry], tmp_path)
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--entry", entry, "--out", graph)
    entry.unlink()
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    shutil.rmtree(graph)
    result = _stage(tmp_path, "backend", "compile", "--ir", plan, "--out", delivery)
    shutil.rmtree(plan)
    assert _run([result["executable"]], tmp_path) == expected


def test_literal_array_facts_respect_alias_writes_and_activation_boundaries(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    entry = source / "entry.py"
    (source / "settings.py").write_text(
        "import numpy as np\nupper = [7.5, 9.5]\nlower = -np.array(upper)\n"
    )
    entry.write_text("""import numpy as np
import settings
def scratch():
    local = np.zeros((2, 2))
    row = local[0]
    for i in range(2):
        row[i] = 12.5
    print('scratch', row[0] == 12.5)
scratch()
scale = (settings.upper[1] - settings.lower[1]) / (settings.upper[0] - settings.lower[0])
print(int(scale * 3), int(-2.9), int(), int(True), float(), float(7))
original = np.array([[1.9, -2.9], [3.9, 4.9]], dtype=float)
retained = original.copy
alias = original
original = np.array([[8.9, 9.9], [10.9, 11.9]])
copied = retained()
row = alias[-1]
independent = row.copy()
row[0] = 55.0
print(int(copied[0, 0]), int(copied[-1, -1]), int(independent[0]))
print(alias[1, 0] == 55.0, original[0, 0] == 8.9)
def closed():
    local = -np.array([6.9])
    return int(local[0])
print(closed(), closed())
def convert_flag(flag):
    return int(flag)
print(convert_flag(True), convert_flag(False))
for i in range(-5, 6):
    print(i // 3, i % 3, i // -3, i % -3)
""")
    expected = _run([sys.executable, "-B", entry], source)
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--entry", entry, "--out", graph)
    shutil.rmtree(source)
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    shutil.rmtree(graph)
    result = _stage(tmp_path, "backend", "compile", "--ir", plan, "--out", delivery)
    shutil.rmtree(plan)
    assert _run([result["executable"]], tmp_path) == expected

    # Writes kill contents through every alias. Loop bodies are not specialized
    # to their first visit, and function signatures do not freeze a prior input.
    prefixes = {
        "alias": "a = np.array([1.9]); b = a\nb[0] = 7.9\nprint(int(a[0]))\n",
        "callee": "a = np.array([1.9])\ndef change(x):\n    x[0] = 7.9\nchange(a)\nprint(int(a[0]))\n",
        "loop": "a = np.array([1.9]); b = np.array([2.9]); x = a\nfor i in range(2):\n    print(int(b[0]))\n    x = b\n    x[0] = 7.9\n",
        "condition": "a = np.array([1.9])\nwhile int(a[0]) < 5:\n    a[0] = 7.9\n",
        "branch": "a = np.array([1.9])\nif input():\n    a[0] = 7.9\nprint(int(a[0]))\n",
        "exception": "a = np.array([1.9])\ntry:\n    a[0] = 7.9\n    raise ValueError('changed')\nexcept ValueError:\n    print(int(a[0]))\n",
        "unknown": "def convert(x):\n    return int(x)\nprint(convert(2.9))\n",
        "boundary": "print(int(9223372036854775808.0))\n",
    }
    for name, body in prefixes.items():
        folder = tmp_path / name
        folder.mkdir()
        entry = folder / "entry.py"
        entry.write_text("import numpy as np\n" + body)
        graph, plan = folder / "graph", folder / "plan"
        _stage(folder, "frontend", "build", "--entry", entry, "--out", graph)
        entry.unlink()
        report = _stage(
            folder, "middle", "plan", "--graph", graph, "--out", plan, returncode=2
        )
        assert report["status"] == "INCOMPLETE"
        issues = json.loads((plan / "plan.json").read_bytes())["issues"]
        assert issues[0]["code"] == "python.integer_conversion_range", issues


def test_short_circuit_preserves_values_effects_and_activation_edges(tmp_path):
    entry = tmp_path / "entry.py"
    entry.write_text("""import numpy as np
def tap(label, value):
    print(label, end='')
    return value
print(tap('a', 3) < tap('b', 2) < tap('c', 4))
print(tap('d', 1) < tap('e', 2) <= tap('f', 2))
message = 'short'
message += '-circuit'
print(message)
for a in range(3):
    for b in range(3):
        print(tap('a', a) and (tap('b', b) or tap('c', 4)), end='/')
        print(not (tap('x', a) and tap('y', b)))
class Item:
    def __init__(self, value):
        self.value = value
left = Item(2.0)
right = Item(9.0)
def tail():
    print('T', end='')
    left.value = -10.0
    return 1.0
print(0.0 < left.value < tail(), left.value)
print((left or right) is left, (left and right) is right, left is left is not right)
def gate(flag):
    if flag and 7:
        print('yes')
    if flag or 'text':
        print('text')
    return not (flag and 7)
print(gate(False), gate(True))
def never_leaf():
    return 1 / 0
def never():
    return never_leaf()
print(False and never(), True or never(), 3 < 2 < never())
for i in range(0):
    never()
def transfer():
    return 'returned'
    never()
print(transfer())
def repeat():
    print('repeat')
    return True
False and repeat()
print(repeat())
array = np.array([-0.0, 3.0])
print(repr(array[0] and never()), repr(array[1] or never()))
def compare(data):
    return data[0] < data[1] < 4.0
flag = compare(array)
print(repr(flag), str(flag), ascii(flag), format(flag, ''), format(flag, '>5'))
print(int(flag), float(flag), repr(flag and flag), repr(flag or flag))
print([flag], (flag,))
""")
    expected = _run([sys.executable, "-B", entry], tmp_path)
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--entry", entry, "--out", graph)
    entry.unlink()
    # An explicit initializer root remains required even if every ordinary
    # invocation site was proven untaken. The proof must not erase this root.
    manifest = json.loads((graph / "graph.json").read_bytes())
    needed = "py:entry:never"
    for unit in manifest["units"]:
        if any(fn["name"] == needed for fn in unit["functions"]):
            unit["initializer_calls"] = [*unit["initializer_calls"], needed]
            break
    else:
        raise AssertionError("the source function must be present in the input graph")
    pinned = graph / "pinned-root.json"
    pinned.write_text(json.dumps(manifest))
    _stage(
        tmp_path,
        "middle",
        "plan",
        "--graph",
        pinned,
        "--out",
        tmp_path / "pinned",
        returncode=2,
    )
    issues = json.loads((tmp_path / "pinned/plan.json").read_bytes())["issues"]
    assert any(
        i["code"] == "python.reachable_protocol"
        and i["symbol"] in {needed, "py:entry:never_leaf"}
        for i in issues
    )
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    report = json.loads((plan / "plan.json").read_bytes())
    proofs = [
        d for d in report["normalizations"] if d["code"] == "inactive_activation_edges"
    ]
    assert len(proofs) == 1
    assert {"py:entry:never", "py:entry:never_leaf"} <= set(
        proofs[0]["inactive_bodies"]
    )
    assert "py:entry:repeat" not in proofs[0]["inactive_bodies"]
    assert all(e["site"] and e["proof"]["rule"] for e in proofs[0]["removed_edges"])
    shutil.rmtree(graph)
    result = _stage(tmp_path, "backend", "compile", "--ir", plan, "--out", delivery)
    shutil.rmtree(plan)
    assert _run([result["executable"]], tmp_path) == expected

    sources = {
        "union": "def choose(flag):\n    return flag and 7\nprint(choose(False))\n",
        "numpy_identity": "import numpy as np\nprint((np.array([1.0])[0] < 2.0) is True)\n",
        "definition": "def mark():\n    print('definition')\n    return 1\ndef never(x=mark()):\n    return x\nprint(False and never())\n",
    }
    for name, source in sources.items():
        folder = tmp_path / name
        folder.mkdir()
        entry = folder / "entry.py"
        entry.write_text(source)
        graph, plan = folder / "graph", folder / "plan"
        _stage(folder, "frontend", "build", "--entry", entry, "--out", graph)
        entry.unlink()
        _stage(folder, "middle", "plan", "--graph", graph, "--out", plan, returncode=2)
        issue = json.loads((plan / "plan.json").read_bytes())["issues"][0]
        assert (
            issue["code"]
            == {
                "union": "python.short_circuit_union",
                "numpy_identity": "python.comparison_identity",
                "definition": "python.function_creation",
            }[name]
        ), issue


def test_formatted_values_preserve_order_unicode_and_source_domains(tmp_path):
    entry = tmp_path / "entry.py"
    entry.write_text(
        """import numpy as np
text = input('value:')
print(f'{text!r}|{text!a}|{text:🟩^10.3s}')
def value():
    print('value', end=';')
    return 1.25
def width():
    print('width', end=';')
    return 9
print(f'{value():0{width()}.2f}')
print(f'{42:#010x}', f'{1234567:_d}', f'{0.001234:12.3e}', f'{-0.0:z.2f}')
print(format(True, ''), format(True, '-'), f'{None}', format('λ', '🐍>4s'))
print(repr(np.array([1.5])[0]), ascii(['λ', 2, None]))
try:
    print(format(None, '4'))
except TypeError as error:
    print(str(error))
""",
        encoding="utf-8",
    )
    inputs = ("λ\u200b🐍abc\n".encode(), b"\xff'\t\n")
    expected = [_run([sys.executable, "-B", entry], tmp_path, input=v) for v in inputs]
    assert all(b"value;width;000001.25" in v for v in expected)
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--entry", entry, "--out", graph)
    entry.unlink()
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    shutil.rmtree(graph)
    result = _stage(tmp_path, "backend", "compile", "--ir", plan, "--out", delivery)
    shutil.rmtree(plan)
    for data, baseline in zip(inputs, expected):
        assert _run([result["executable"]], tmp_path, input=data) == baseline
    target = json.loads((delivery / "target-ir.json").read_bytes())
    assert "include/scar/text.hpp" in {h["output"] for h in target["sdk_headers"]}
    header = (delivery / "project/include/scar/program.hpp").read_text()
    assert header.count("namespace scar::generated_unicode") == 1


def test_source_sequence_expands_to_native_operators_with_provider_diagnostics(
    tmp_path,
):
    source = tmp_path / "source"
    source.mkdir()
    entry = source / "entry.py"
    entry.write_text("""import numpy as np
def show(values, count):
    for i in range(count):
        print(f'{values[i]:.17g}', end=';')
    print()
show(np.linspace(0.0, 1.0, 6), 6)
show(np.linspace(-2.0, 3.0, 7, endpoint=False), 7)
show(np.linspace(0.0, 5e-324, 6), 6)
show(np.linspace(0.0, -0.0, 3, axis=-1, dtype=np.float64), 3)
empty = np.linspace(-1e308, 1e308, 0)
show(empty, 0)
show(np.linspace(-1e308, 1e308, 1), 1)
show(np.linspace(-1e308, 1e308, 4), 4)
show(np.linspace(-1e308, 1e308, 4), 4)
show(np.linspace(-1e308, 1e308, 4, endpoint=False), 4)
pair = np.linspace(1.0, 4.0, 4, retstep=True)
def report():
    show(pair[0], 4)
    print(repr(pair[1]), abs(pair[1]))
report()
undefined = np.linspace(1.0, 4.0, 1, retstep=True)
print(repr(undefined[1]))
for n in range(-1, 4):
    try:
        show(np.linspace(1.0, 2.0, n), n)
    except ValueError as error:
        print(str(error))
print(abs(True), abs(-3.5), abs(-9223372036854775807), abs(-0.0))
""")
    expected = subprocess.run(
        [sys.executable, "-B", entry], cwd=source, capture_output=True, check=True
    )
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--entry", entry, "--out", graph)
    shutil.rmtree(source)
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    report = json.loads((plan / "plan.json").read_bytes())
    expansions = [
        d for d in report["normalizations"] if d["code"] == "expanded_sequence_region"
    ]
    assert len(expansions) == 12
    ir = json.loads((plan / "exec-ir.json").read_bytes())
    assert (
        "array_iota" in ir["module"]["operator_requirements"]["ndarray"]["operations"]
    )
    shutil.rmtree(graph)
    result = _stage(tmp_path, "backend", "compile", "--ir", plan, "--out", delivery)
    shutil.rmtree(plan)
    native = subprocess.run(
        [result["executable"]], cwd=tmp_path, capture_output=True, check=True
    )
    assert (native.stdout, native.stderr) == (expected.stdout, expected.stderr)


def test_scalar_selection_normalizes_shared_observations_without_runtime_tags(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    entry = source / "entry.py"
    entry.write_text("""import numpy as np
values = np.array([-2.0, 0.0, -0.0, 3.0])
special = np.array([0.0, 1.0, -1.0]) / 0.0
def clamp(value):
    return max(0.0, value)
for i in range(4):
    selected = clamp(values[i])
    print(f'{selected:.17g}', f'{min(values[i], 0.0):.17g}', abs(selected))
    print(f'{min(False, values[i] > 0.0):>6}', not selected)
for i in range(3):
    print(f'{max(0.0, special[i]):.17g}', f'{max(special[i], 0.0):.17g}')
    print(f'{min(0.0, special[i]):.17g}', f'{min(special[i], 0.0):.17g}')
def stamp(label, value):
    print(label, end='|')
    return value
print(max(stamp('first', values[0]), stamp('second', values[3]), stamp('third', values[1])))
print(min(4, 2, 3), max(4, 2, 3))
print(f'{max(0.0, -0.0):.17g}', f'{max(-0.0, 0.0):.17g}')
number = np.array([2.0])[0]
print(repr(number))
number = 3.0
print(repr(number))
number = np.array([4.0])[0]
print(repr(number))
""")
    expected = subprocess.run(
        [sys.executable, "-B", entry], capture_output=True, check=True
    )
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--entry", entry, "--out", graph)
    shutil.rmtree(source)
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    report = json.loads((plan / "plan.json").read_bytes())
    selections = [
        n for n in report["normalizations"] if n["code"] == "scalar_selection"
    ]
    assert any(s["result"] == "binary64_choice" for s in selections)
    ir = json.loads((plan / "exec-ir.json").read_bytes())
    from scar.contract import walk_operations

    functions = [
        json.loads((plan / member["path"]).read_bytes()) for member in ir["functions"]
    ]
    selected = [n for n in walk_operations(functions) if n["op"] == "select"]
    assert selected and all(
        n["then"]["op"] in {"var", "literal"} and n["else"]["op"] in {"var", "literal"}
        for n in selected
    )
    assert not any(n["op"].startswith("sum_") for n in walk_operations(functions))
    shutil.rmtree(graph)
    result = _stage(tmp_path, "backend", "compile", "--ir", plan, "--out", delivery)
    shutil.rmtree(plan)
    native = subprocess.run([result["executable"]], capture_output=True, check=True)
    assert (native.stdout, native.stderr) == (expected.stdout, expected.stderr)
    generated = (delivery / "project/src/program.cpp").read_text()
    assert "std::variant" not in generated and "PyObject" not in generated


def test_scalar_selection_keeps_observable_behavior_and_precision_boundaries(tmp_path):
    for name, statement, issue in (
        ("representation", "print(repr(max(0.0, value)))", "scalar_observation"),
        ("division", "print(max(0.0, value) / 0.0)", "scalar_observation"),
        ("promotion", "print(max(1, value))", "extreme_representation"),
        (
            "integer_rounding",
            "print(max(value, 9007199254740993))",
            "integer_precision",
        ),
        (
            "exact_integer_rounding",
            "print(max(np.array([9007199254740992.0])[0], 9007199254740993))",
            "integer_precision",
        ),
        ("inplace_power", "values **= max(0.5, value)", "array_power_dispatch"),
    ):
        source = tmp_path / name
        source.mkdir()
        entry = source / "entry.py"
        entry.write_text(
            "import numpy as np\n"
            "values = np.array([-1.0, 2.0])\n"
            "for i in range(2):\n"
            "    value = values[i]\n"
            f"    {statement}\n"
        )
        graph, plan = source / "graph", source / "plan"
        _stage(tmp_path, "frontend", "build", "--entry", entry, "--out", graph)
        entry.unlink()
        _stage(
            tmp_path, "middle", "plan", "--graph", graph, "--out", plan, returncode=2
        )
        report = json.loads((plan / "plan.json").read_bytes())
        assert report["status"] == "INCOMPLETE"
        assert report["issues"][0]["code"] == "python." + issue


def test_coordinate_regions_preserve_layout_copies_aliases_and_unpack_order(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    entry = source / "entry.py"
    entry.write_text("""import numpy as np
left = np.array([1.0, 2.0])
right = np.array([3.0, 4.0, 5.0])
x, y = np.meshgrid(left, right)
left[0] = 9.0
x[0, 0] = 7.0
print(x, y, left, right, sep='\\n')
def coordinates(a, b):
    first, second = np.meshgrid(a, b, indexing='ij')
    print(first, second, sep='\\n')
coordinates(left, right)
u, v, w = np.meshgrid([1.0, 2.0], [3.0], [4.0, 5.0], indexing='ij')
print(u, v, w, sep='\\n')
base = np.array([[1.0, 2.0], [3.0, 4.0]])
shared, column = np.meshgrid(base, [5.0, 6.0], sparse=True, copy=False)
shared[0, 2] = 11.0
print(base, shared, column, base[1, 0] == 11.0, sep='\\n')
owned, = np.meshgrid(base[1], sparse=True)
base[1, 1] = 12.0
print(owned)
borrowed, = np.meshgrid(base[1], sparse=True, copy=False)
borrowed[0] = 13.0
print(base, borrowed, sep='\\n')
empty_x, empty_y = np.meshgrid([], [1.0])
print(empty_x, empty_y, sep='\\n')
scalar, = np.meshgrid(2.5)
print(scalar, np.meshgrid())
a, b = 1, 2
a, b = b, a
print(a, b)
index = 0
target = np.array([0.0, 0.0])
index, target[index] = 1, 8.0
print(index, target)
index = 0
for i in range(3):
    print(index)
    index, extra = i, 2
def protected():
    value = 0
    try:
        value, other = 1, 2
        raise ValueError('stop')
    except ValueError:
        print(value)
protected()
""")
    expected = subprocess.run(
        [sys.executable, "-B", entry], capture_output=True, check=True
    )
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--entry", entry, "--out", graph)
    shutil.rmtree(source)
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    report = json.loads((plan / "plan.json").read_bytes())
    regions = [
        n for n in report["normalizations"] if n["code"] == "expanded_coordinate_region"
    ]
    assert len(regions) == 9
    ir = json.loads((plan / "exec-ir.json").read_bytes())
    assert {"array_size", "array_reshape", "array_broadcast_copy"} <= set(
        ir["module"]["operator_requirements"]["ndarray"]["operations"]
    )
    shutil.rmtree(graph)
    result = _stage(tmp_path, "backend", "compile", "--ir", plan, "--out", delivery)
    shutil.rmtree(plan)
    native = subprocess.run([result["executable"]], capture_output=True, check=True)
    assert (native.stdout, native.stderr) == (expected.stdout, expected.stderr)

    excessive = tmp_path / "excessive.py"
    excessive.write_text(
        "import numpy as np\nnp.meshgrid(" + ",".join(["1.0"] * 65) + ",sparse=True)\n"
    )
    _stage(
        tmp_path,
        "frontend",
        "build",
        "--entry",
        excessive,
        "--out",
        tmp_path / "rank-graph",
    )
    _stage(
        tmp_path,
        "middle",
        "plan",
        "--graph",
        tmp_path / "rank-graph",
        "--out",
        tmp_path / "rank-plan",
        returncode=2,
    )
    rejected = json.loads((tmp_path / "rank-plan/plan.json").read_bytes())
    assert rejected["issues"][0]["code"] == "python.array_rank"


def test_numpy_default_warnings_survive_source_removal(tmp_path):
    entry = tmp_path / "entry.py"
    entry.write_text("""import numpy as np
a = np.array([1.0, 0.0])
for i in range(2):
    b = a / 0.0
print(b)
""")
    expected = subprocess.run(
        [sys.executable, "-B", entry], capture_output=True, timeout=30
    )
    assert expected.returncode == 0 and expected.stderr.count(b"RuntimeWarning:") == 2
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--entry", entry, "--out", graph)
    entry.unlink()
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    shutil.rmtree(graph)
    result = _stage(tmp_path, "backend", "compile", "--ir", plan, "--out", delivery)
    shutil.rmtree(plan)
    actual = subprocess.run([result["executable"]], capture_output=True, timeout=30)
    assert (actual.returncode, actual.stdout, actual.stderr) == (
        expected.returncode,
        expected.stdout,
        expected.stderr,
    )


def test_host_protocols_preserve_files_stream_aliases_and_text(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    entry = source / "entry.py"
    entry.write_text(
        """import os
import shutil
import time
from os.path import join as combine
os.makedirs('data/nested', exist_ok=True)
path = combine('data', 'science.txt')
writer = open(path, 'w', encoding='utf-8')
alias = writer
print('科学', 1.25, -0.0, True, None, sep='|', file=writer, flush=True)
print(alias.write('α\\r\\nβ\\rγ\\n'))
alias.close()
writer.close()
reader = open(path)
print(reader.read(), end='')
reader.close()
print(shutil.copyfile(path, 'data/content.txt'))
print(shutil.copy(path, 'data/nested'))
print(shutil.copy2(path, 'data/metadata.txt'))
print(shutil.copytree('data/nested', 'tree'))
print(os.path.isfile('tree/science.txt'), os.path.isdir('tree'))
print(os.path.exists('invalid\\0path'), os.path.isfile('invalid\\0path'))
shutil.rmtree('tree')
shutil.rmtree('absent', ignore_errors=True)
os.chdir('data')
print(os.path.exists('metadata.txt'), time.time() > 0.0)
print(str(1e-5), str(1e15), str(1e16), str(False), str(None))
""",
        encoding="utf-8",
    )
    reference = tmp_path / "reference"
    reference.mkdir()
    expected = _run([sys.executable, "-B", entry], reference)
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--entry", entry, "--out", graph)
    shutil.rmtree(source)
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    shutil.rmtree(graph)
    result = _stage(tmp_path, "backend", "compile", "--ir", plan, "--out", delivery)
    shutil.rmtree(plan)
    native = tmp_path / "native"
    native.mkdir()
    assert _run([result["executable"]], native) == expected
    for relative in (
        "data/science.txt",
        "data/content.txt",
        "data/nested/science.txt",
        "data/metadata.txt",
    ):
        assert (native / relative).read_bytes() == (reference / relative).read_bytes()
    target = json.loads((delivery / "target-ir.json").read_bytes())
    assert [h["output"] for h in target["sdk_headers"]] == [
        "include/scar/exceptions.hpp",
        "include/scar/host.hpp",
        "include/scar/text.hpp",
    ]


def test_exception_regions_preserve_cleanup_and_control_transfers(tmp_path):
    entry = tmp_path / "entry.py"
    entry.write_text("""def protected():
    try:
        try:
            raise ValueError('bad\\0χ')
        except (ValueError, TypeError) as problem:
            print('caught', str(problem))
            raise
        finally:
            print('inner cleanup')
    except Exception:
        print('outer handler')
    finally:
        print('last cleanup')
    with open('data.txt', 'w', encoding='utf-8') as stream, stream as entered:
        alias = stream
        entered.write('saved')
    try:
        alias.write('must fail')
    except ValueError:
        print('closed alias')
    try:
        return 7
    finally:
        print('return cleanup')
print(protected())
for i in range(3):
    try:
        break
    finally:
        continue
else:
    print('exhausted')
for j in range(3):
    try:
        if j == 0: continue
        break
    finally:
        print('iteration', j)
else:
    print('wrong')
""")
    reference = tmp_path / "reference"
    reference.mkdir()
    expected = _run([sys.executable, "-B", entry], reference)
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--entry", entry, "--out", graph)
    entry.unlink()
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    shutil.rmtree(graph)
    result = _stage(tmp_path, "backend", "compile", "--ir", plan, "--out", delivery)
    shutil.rmtree(plan)
    native = tmp_path / "native"
    native.mkdir()
    assert _run([result["executable"]], native) == expected
    assert (native / "data.txt").read_bytes() == (reference / "data.txt").read_bytes()


def test_console_and_process_exit_preserve_handlers_and_finalization(tmp_path):
    entry = tmp_path / "entry.py"
    entry.write_text("""import sys
try:
    text = input('prompt:')
    print('data', text)
except EOFError:
    print('eof')
try:
    sys.exit(3)
except SystemExit as signal:
    print('caught exit', str(signal))
try:
    with open('exit.txt', 'w', encoding='utf-8') as stream:
        print('saved', file=stream)
        sys.exit('stop')
finally:
    print('exit cleanup')
""")
    reference = tmp_path / "reference"
    reference.mkdir()
    inputs = (b"\xff\n", b"")
    expected = [
        subprocess.run(
            [sys.executable, "-B", entry],
            cwd=reference,
            input=data,
            capture_output=True,
            timeout=30,
        )
        for data in inputs
    ]
    assert all(
        result.returncode == 1 and result.stderr == b"stop\n" for result in expected
    )
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--entry", entry, "--out", graph)
    entry.unlink()
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    shutil.rmtree(graph)
    result = _stage(tmp_path, "backend", "compile", "--ir", plan, "--out", delivery)
    shutil.rmtree(plan)
    native = tmp_path / "native"
    native.mkdir()
    for data, baseline in zip(inputs, expected):
        actual = subprocess.run(
            [result["executable"]],
            cwd=native,
            input=data,
            capture_output=True,
            timeout=30,
        )
        assert (actual.returncode, actual.stdout, actual.stderr) == (
            baseline.returncode,
            baseline.stdout,
            baseline.stderr,
        )
    assert (native / "exit.txt").read_bytes() == (reference / "exit.txt").read_bytes()


def test_python_loops_remain_structured_and_preserve_exit_bindings(tmp_path):
    entry = tmp_path / "entry.py"
    entry.write_text("""s = 0
for i in range(1, 6):
    if i == 2: continue
    if i == 4: break
    s += i
else: print('wrong')
print(s, i)
for i in range(3, -2, -2): print(i)
else: print('done')
k = 0
while k < 3:
    k += 1
    if k == 2: continue
    print(k)
else: print('while', k)
n = 4
for j in range(n): n = 0
print(j, n)
for v in range(9223372036854775806, 9223372036854775807, 3): print(v)
""")
    expected = _run([sys.executable, "-B", entry], tmp_path)
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--entry", entry, "--out", graph)
    entry.unlink()
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    ir = json.loads((plan / "exec-ir.json").read_bytes())
    from scar.contract import read_member, walk_operations

    loops = [
        node
        for receipt in ir["functions"]
        for node in walk_operations(read_member(plan, receipt))
        if node["op"] in {"for", "while"}
    ]
    assert len(loops) == 5
    shutil.rmtree(graph)
    result = _stage(tmp_path, "backend", "compile", "--ir", plan, "--out", delivery)
    shutil.rmtree(plan)
    assert _run([result["executable"]], tmp_path) == expected


def test_local_provider_name_does_not_select_an_unrelated_library(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    entry = source / "entry.py"
    entry.write_text("import numpy\nprint(numpy.zeros())\n")
    (source / "numpy.py").write_text("def zeros():\n    return 'local provider'\n")
    expected = _run([sys.executable, "-B", entry], source)
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--entry", entry, "--out", graph)
    shutil.rmtree(source)
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    result = _stage(tmp_path, "backend", "compile", "--ir", plan, "--out", delivery)
    assert _run([result["executable"]], tmp_path) == expected == b"local provider\n"
    physical = json.loads((delivery / "physical-plan.json").read_bytes())
    assert "operator_backends" not in physical


def test_python_protocol_closure_never_assumes_mutable_bindings_constant(tmp_path):
    sources = {
        "global": "x = 'one'\ndef show():\n    print(x)\nshow()\nx = 'two'\nshow()\n",
        "builtin": (
            "def other(value):\n    return None\n"
            "print = other\nprint('must stay silent')\n"
        ),
        "dynamic": "value = input()\nprint(value)\n",
        "generator": "def gen():\n    print('not executed')\n    yield 'x'\ngen()\n",
        "annotation": "def mark():\n    print('annotation')\n    return None\nx: mark() = 'value'\n",
        "object_lookup": (
            "class Item:\n    def __getattribute__(self, name):\n        return 2.5\n"
            "print(Item().value)\n"
        ),
        "object_descriptor": (
            "class Item:\n    @property\n    def value(self):\n"
            "        print('descriptor')\n        return 2.5\n"
            "print(Item().value)\n"
        ),
        "constant_condition": (
            "mode = 'chosen'\n"
            "def check():\n    print('condition')\n    return mode == 'chosen'\n"
            "if check():\n    from os.path import join\n"
            "else:\n    import scar_missing_inactive_package\n"
            "print(join('selected', 'path'))\n"
            "state = 'first'\n"
            "def report():\n    print(state == 'first')\n"
            "report()\nstate = 'second'\nreport()\n"
        ),
    }
    for name, text in sources.items():
        folder = tmp_path / name
        folder.mkdir()
        entry = folder / "entry.py"
        entry.write_text(text)
        expected = _run([sys.executable, "-B", entry], folder, input=b"input\n")
        graph, plan = folder / "graph", folder / "plan"
        _stage(folder, "frontend", "build", "--entry", entry, "--out", graph)
        entry.unlink()
        # A supported binding must preserve behavior; an open protocol must block
        # execution IR. This also allows later principled domain extensions.
        process = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                _PHASE_COMMAND,
                "middle",
                "plan",
                "--graph",
                str(graph),
                "--out",
                str(plan),
            ],
            env={
                **os.environ,
                "PYTHONPATH": str(Path(scar.__file__).resolve().parents[1]),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            capture_output=True,
            timeout=60,
        )
        report = json.loads(process.stdout)
        if name == "constant_condition":
            assert report["status"] == "READY", report
            decisions = json.loads((plan / "plan.json").read_bytes())["normalizations"]
            assert any(d["code"] == "exact_primitive_fact" for d in decisions)
        if report["status"] == "READY":
            assert process.returncode == 0
            result = _stage(
                folder, "backend", "compile", "--ir", plan, "--out", folder / "delivery"
            )
            assert _run([result["executable"]], folder, input=b"input\n") == expected
        else:
            assert process.returncode == 2 and report["blockers"] > 0
            ir = json.loads((plan / "exec-ir.json").read_bytes())
            assert ir["status"] == "INCOMPLETE" and not ir["functions"]


def test_graphics_state_is_logical_ir_and_requires_native_qualification(tmp_path):
    entry = tmp_path / "graphics.py"
    entry.write_text("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as again
x = np.array([0.0, 1.0])
y = np.array([[2.0, 3.0], [4.0, 5.0]])
lines = plt.plot(x, y, color='red', marker='.', linestyle='')
y[0, 0] = 99.0
again.grid(True)
plt.savefig('curves.jpeg')
plt.savefig('curves.pdf')
""")
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--entry", entry, "--out", graph)
    entry.unlink()
    report = _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    assert report["status"] == "READY" and report["blockers"] == 0
    ir = json.loads((plan / "exec-ir.json").read_bytes())
    from scar.contract import read_member, walk_operations

    operations = [
        node
        for receipt in ir["functions"]
        for node in walk_operations(read_member(plan, receipt))
    ]
    contexts = [n for n in operations if n["op"] == "graphics_context"]
    plots = [n for n in operations if n["op"] == "graphics_plot"]
    grids = [n for n in operations if n["op"] == "graphics_grid"]
    saves = [n for n in operations if n["op"] == "graphics_save"]
    assert len(contexts) == len(plots) == len(grids) == 1 and len(saves) == 2
    assert plots[0]["data_capture"] == "snapshot_at_call"
    assert plots[0]["type"] == {"kind": "handle", "resource": "graphics_lines"}
    assert (
        plots[0]["context"]
        == grids[0]["context"]
        == saves[0]["context"]
        == saves[1]["context"]
    )
    assert all(
        n["operator_family"] == "graphics"
        and n["may_raise"]
        and "STATE" in n["effects"]
        for n in contexts + plots + grids + saves
    )
    assert set(ir["module"]["operator_requirements"]) == {"ndarray", "graphics"}
    shutil.rmtree(graph)
    result = _stage(
        tmp_path, "backend", "compile", "--ir", plan, "--out", delivery, returncode=2
    )
    shutil.rmtree(plan)
    assert result["status"] == "INCOMPLETE" and "executable" not in result
    assert {i["requirement"] for i in result["issues"]} == {"graphics"}
    assert not (delivery / "project").exists()


def test_shared_objects_preserve_aliases_bound_receivers_and_field_presence(tmp_path):
    entry = tmp_path / "objects.py"
    entry.write_text(
        """import numpy as np
class Cell:
    'Mutable state shared across aliases.'
    def __init__(self, value, present):
        if present:
            self.value = value
    def add(self, amount):
        self.value += amount
        return self.value
    def unused(self):
        return never_called()
original = Cell(2.5, True)
alias = original
method = original.add
original = Cell(10.0, True)
print(method(1.25), alias.value, original.value, alias is original)
def invoke_saved(amount):
    return method(amount)
print(invoke_saved(0.5), alias.value)
def transfer(value):
    return value
print(transfer(alias) is alias)
pending = Cell(9.0, False)
try:
    print(pending.value)
except AttributeError as error:
    print(str(error))
class Holder:
    def __init__(self, data):
        self.λ = data
data = np.array([1.0, 2.0])
holder = Holder(data)
saved = holder.λ
data[0] = 7.0
saved[1] = 8.0
print(holder.λ[0], data[1])
class Example:
    def read(self):
        return 1
old_type = Example
class Example:
    def read(self):
        return 2
print(old_type().read(), Example().read())
""",
        encoding="utf-8",
    )
    expected = _run([sys.executable, "-B", entry], tmp_path)
    assert expected == (
        b"3.75 3.75 10.0 False\n4.25 4.25\nTrue\n"
        b"'Cell' object has no attribute 'value'\n7.0 8.0\n1 2\n"
    )
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--entry", entry, "--out", graph)
    entry.unlink()
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    ir = json.loads((plan / "exec-ir.json").read_bytes())
    assert len([r for r in ir["module"]["records"] if r.get("object_state")]) == 4
    shutil.rmtree(graph)
    result = _stage(tmp_path, "backend", "compile", "--ir", plan, "--out", delivery)
    shutil.rmtree(plan)
    assert _run([result["executable"]], tmp_path) == expected


def test_import_closure_distinguishes_missing_sources_and_missing_semantics(tmp_path):
    for module, code in (
        ("scar_missing_package.part", "python.import_missing"),
        ("json.decoder", "python.import_semantics"),
        ("pytest", "python.import_semantics"),
    ):
        folder = tmp_path / module
        folder.mkdir()
        entry = folder / "entry.py"
        entry.write_text(f"import {module}\n")
        graph, plan = folder / "graph", folder / "plan"
        _stage(folder, "frontend", "build", "--entry", entry, "--out", graph)
        entry.unlink()
        report = _stage(
            folder, "middle", "plan", "--graph", graph, "--out", plan, returncode=2
        )
        assert report["status"] == "INCOMPLETE" and report["blockers"] == 1
        product = json.loads((plan / "plan.json").read_bytes())
        assert product["issues"][0]["code"] == code
        assert module in product["issues"][0]["detail"]
        execution = json.loads((plan / "exec-ir.json").read_bytes())
        assert execution["status"] == "INCOMPLETE" and not execution["functions"]


def test_automatic_array_in_for_initializer_survives_all_layers(tmp_path):
    entry = tmp_path / "entry.c"
    entry.write_text(
        "#include <stdio.h>\nint main(void) {\n"
        "int n = 3; double sum = 0;\n"
        "for (double scratch[n]; n > 0; --n) {\n"
        "scratch[n-1] = n; sum += scratch[n-1];\n"
        '} printf("%.1f\\n", sum); return 0; }\n'
    )
    original = tmp_path / "original"
    _run(["gcc", "-std=c99", entry, "-o", original], tmp_path)
    expected = _run([original], tmp_path)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            dict(
                source_root=str(tmp_path),
                entry="main",
                units=[dict(path=entry.name, language="c", flags=["-std=c99"])],
            )
        )
    )
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--manifest", request, "--out", graph)
    entry.unlink()
    original.unlink()
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    shutil.rmtree(graph)
    result = _stage(tmp_path, "backend", "compile", "--ir", plan, "--out", delivery)
    shutil.rmtree(plan)
    assert _run([result["executable"]], tmp_path) == expected == b"6.0\n"


def test_declared_process_images_preserve_pipes_and_receiver_identity(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    child = source / "worker.c"
    child.write_text(
        "#include <stdio.h>\nint main(int argc, char **argv) {\n"
        'printf("%s %d\\n", argv[0], argc);\n'
        'if (argc > 1) printf("%s\\n", argv[1]);\n'
        'printf("line\\r\\nlast\\rtail");\n'
        "return argc + 5;\n}\n"
    )
    (source / "helper.py").write_text("""def consume(stream, output):
    for line in stream:
        output.write(line)
        print(line, end='')
    else:
        print('drained')
""")
    parent = source / "driver.py"
    parent.write_text("""import subprocess
from helper import consume
process = subprocess.Popen(['./worker', 'space * raw'], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, encoding='utf-8')
wait_first = process.wait
alias = process
with open('copy.txt', 'w', encoding='utf-8') as output:
    consume(process.stdout, output)
print(wait_first(), alias.wait())
process = subprocess.Popen(['./worker'], stdout=subprocess.PIPE, text=True,
                           encoding='utf-8')
print(process.stdout.read())
print(wait_first(), process.wait())
""")
    _run(["gcc", "-std=c89", child, "-o", source / "worker"], tmp_path)
    expected = subprocess.run(
        [sys.executable, "-B", parent], cwd=source, capture_output=True, check=True
    )
    expected_file = (source / "copy.txt").read_bytes()
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            dict(
                source_root=str(source),
                root_image="controller",
                images=[
                    dict(
                        id="controller",
                        entry="__main__",
                        output="controller",
                        source="driver.py",
                    ),
                    dict(
                        id="worker",
                        entry="main",
                        output="worker",
                        units=[dict(path="worker.c", language="c", flags=["-std=c89"])],
                    ),
                ],
            )
        )
    )
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--manifest", request, "--out", graph)
    shutil.rmtree(source)
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    shutil.rmtree(graph)
    result = _stage(tmp_path, "backend", "compile", "--ir", plan, "--out", delivery)
    shutil.rmtree(plan)
    run = tmp_path / "run"
    run.mkdir()
    native = subprocess.run(
        [result["executables"]["controller"]], cwd=run, capture_output=True, check=True
    )
    assert (native.stdout, native.stderr) == (expected.stdout, expected.stderr)
    assert (run / "copy.txt").read_bytes() == expected_file


def test_mixed_program_images_preserve_link_and_state_boundaries(tmp_path):
    sources = tmp_path / "source"
    sources.mkdir()
    texts = {
        "first.c": (
            "#include <stdio.h>\nint state = 4;\n"
            "extern int increment(void);\nextern void scale_(double*);\n"
            "int main(void) { double x = 2; scale_(&x);\n"
            'printf("first %d %.1f\\n", increment(), x); return 0; }\n'
        ),
        "second.cpp": (
            '#include <cstdio>\nextern "C" {\nint state = 40;\n'
            "int increment(void);\nvoid scale_(double*);\n}\n"
            "int main() { double x = 2; scale_(&x);\n"
            'std::printf("second %d %.1f\\n", increment(), x); return 0; }\n'
        ),
        "shared.c": "extern int state;\nint increment(void) { return ++state; }\n",
        "old.f": (
            "      SUBROUTINE SCALE(X)\n      DOUBLE PRECISION X\n"
            "      X = 3D0 * X\n      END\n"
        ),
        "new.f90": (
            "subroutine scale(x)\n  implicit none\n"
            "  double precision, intent(inout) :: x\n"
            "  x = 5d0 * x\nend subroutine scale\n"
        ),
    }
    profiles = {}
    for name, text in texts.items():
        path = sources / name
        path.write_text(text)
        language, compiler, flags = {
            ".c": ("c", "gcc", ["-std=c89"]),
            ".cpp": ("c++", "g++", ["-std=c++17"]),
            ".f": ("fortran", "gfortran", ["-std=legacy"]),
            ".f90": ("fortran", "gfortran", ["-std=f2008"]),
        }[path.suffix]
        profiles[name] = dict(path=name, language=language, flags=flags)
        _run([compiler, *flags, "-c", path, "-o", sources / (name + ".o")], tmp_path)
    images = [
        dict(
            id="first",
            entry="main",
            output="first",
            units=["first.c", "shared.c", "old.f"],
        ),
        dict(
            id="second",
            entry="main",
            output="second",
            units=["second.cpp", "shared.c", "new.f90"],
        ),
    ]
    expected = {}
    for image in images:
        original = tmp_path / ("original-" + image["id"])
        _run(
            [
                "g++",
                *(sources / (name + ".o") for name in image["units"]),
                "-lgfortran",
                "-o",
                original,
            ],
            tmp_path,
        )
        expected[image["id"]] = _run([original], tmp_path)
        original.unlink()
    assert expected == {"first": b"first 5 6.0\n", "second": b"second 41 10.0\n"}
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            dict(
                source_root=str(sources),
                root_image="second",
                images=[
                    dict(image, units=[profiles[name] for name in image["units"]])
                    for image in images
                ],
            )
        )
    )
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(
        tmp_path,
        "frontend",
        "build",
        "--manifest",
        request,
        "--out",
        graph,
        "--jobs",
        2,
    )
    document = json.loads((graph / "graph.json").read_bytes())
    assert (
        len(document["units"]) == 5
    )  # The shared compilation profile is captured once.
    assert document["images"][0]["units"][1] == document["images"][1]["units"][1]
    shutil.rmtree(sources)
    request.unlink()
    graph = graph.rename(tmp_path / "relocated-graph")
    logical = _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    assert logical["status"] == "READY"
    execution = json.loads((plan / "exec-ir.json").read_bytes())
    assert execution["kind"] == "image_bundle" and execution["root_image"] == "second"
    for image in execution["images"]:
        child = json.loads((plan / image["ir"]["path"]).read_bytes())
        assert child["status"] == "READY" and child["module"]["entry"] == "main"
    # A complete root must not hide an incomplete required child image.
    document["images"][0]["entry"] = "missing_entry"
    (graph / "graph.json").write_text(json.dumps(document))
    partial = tmp_path / "incomplete-plan"
    rejected = _stage(
        tmp_path, "middle", "plan", "--graph", graph, "--out", partial, returncode=2
    )
    assert rejected["status"] == "INCOMPLETE"
    child_states = json.loads((partial / "plan.json").read_bytes())["images"]
    assert [image["status"] for image in child_states] == ["INCOMPLETE", "READY"]
    shutil.rmtree(graph)
    plan = plan.rename(tmp_path / "relocated-plan")
    result = _stage(
        tmp_path,
        "backend",
        "compile",
        "--ir",
        plan,
        "--out",
        delivery,
        "--physical",
        "serial",
    )
    assert result["root_image"] == "second"
    assert result["executable"] == result["executables"]["second"]
    shutil.rmtree(plan)
    for image in images:
        native = Path(result["executables"][image["id"]])
        assert (
            native.name == image["output"]
            and _run([native], tmp_path) == expected[image["id"]]
        )
    built = json.loads((delivery / "build.json").read_bytes())
    assert built["warnings"] == 0 and set(built["executables"]) == {"first", "second"}


def test_python_protocol_qualification_preserves_order_and_control(tmp_path):
    helpers = tmp_path / "helpers.c"
    helpers.write_text(
        "int trace = 0;\n"
        "double first(double x) { trace = trace * 10 + 1; return x; }\n"
        "double second(double x) { trace = trace * 10 + 2; return x + 1; }\n"
        "double pair(double a, double b) { trace = trace * 10 + 3; return a + b; }\n"
    )
    main = tmp_path / "main.c"
    main.write_text(
        "#include <stdio.h>\nextern int trace;\nextern double run(double);\n"
        'int main(void) { double x = run(3.0); printf("%.1f %d\\n", x, trace); return 0; }\n'
    )
    python = tmp_path / "calculation.py"
    python.write_text("""def run(x):
    total = first(x) + second(x)
    scale = pair(first(x), second(x))
    while total < 10.0:
        total += 0.5
        if total > 9.0:
            break
    if x > 0.0:
        return total + scale + ((9007199254740992 + 1) - 9007199254740992) + (1e100 - 1e100)
    return 0.0
""")
    library = tmp_path / "helpers.so"
    _run(["gcc", "-shared", "-fPIC", helpers, "-o", library], tmp_path)
    provider = CDLL(str(library))
    bindings = {name: getattr(provider, name) for name in ("first", "second", "pair")}
    for name, fn in bindings.items():
        fn.argtypes, fn.restype = [c_double] * (2 if name == "pair" else 1), c_double
    result = runpy.run_path(str(python), init_globals=bindings)["run"](3.0)
    from ctypes import c_int

    expected = f"{result:.1f} {c_int.in_dll(provider, 'trace').value}\n".encode()
    assert expected == b"17.5 12123\n"
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            dict(
                entry="main",
                units=[
                    dict(path=main.name, language="c", flags=["-std=c17"]),
                    dict(path=helpers.name, language="c", flags=["-std=c17"]),
                    dict(
                        path=python.name,
                        language="python",
                        abi=dict(
                            exports={"run": "run"},
                            imports={name: name for name in bindings},
                        ),
                    ),
                ],
            )
        )
    )
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--manifest", request, "--out", graph)
    for source in (python, helpers, main):
        source.unlink()
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    result = _stage(
        tmp_path,
        "backend",
        "compile",
        "--ir",
        plan / "exec-ir.json",
        "--out",
        delivery,
        "--physical",
        "serial",
    )
    assert _run([result["executable"]], tmp_path) == expected


def test_python_modules_preserve_structure_and_open_bindings(tmp_path):
    source = tmp_path / "entry.py"
    source.write_text("""import helper
from pathlib import Path

def outer():
    total = 1
    def increment():
        nonlocal total
        total += helper.choose()
        return total
    for step in range(3):
        if step > 0:
            print(increment())

Path("must-not-execute").write_text("executed")
outer()
""")
    (tmp_path / "helper.py").write_text("def choose():\n    return 2\n")
    graph = tmp_path / "graph"
    _stage(tmp_path, "frontend", "build", "--entry", source, "--out", graph)
    document = json.loads((graph / "graph.json").read_bytes())
    assert len(document["units"]) == 2
    functions = [f for unit in _payloads(graph, document) for f in unit["functions"]]
    assert document["coverage"]["totals"]["definitions"] == len(functions)
    assert document["coverage"]["totals"]["qualified_definitions"] == 0
    nodes = [n for f in functions for n in f["nodes"]]
    assert sum(n["op"] == "for" for n in nodes) == 1
    assert sum(n["op"] == "if" for n in nodes) == 1
    assert any(
        state["kind"] == "cell" for f in functions for state in f["states"].values()
    )
    assert not (tmp_path / "must-not-execute").exists()
    source.rename(tmp_path / "hidden-entry")
    result = _stage(
        tmp_path,
        "middle",
        "plan",
        "--graph",
        graph,
        "--out",
        tmp_path / "plan",
        returncode=2,
    )
    assert result["status"] == "INCOMPLETE"
    assert not (tmp_path / "must-not-execute").exists()


def test_python_array_three_stages_and_independent_output(tmp_path):
    np = pytest.importorskip("numpy")
    source = tmp_path / "simulation.py"
    source.write_text(
        """
def prepare(initial, fixed):
    return initial + fixed * fixed

def run(initial, fixed, count):
    for unused in range(count):
        initial = prepare(initial, fixed)
    return initial
""",
        encoding="utf-8",
    )
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    initial = np.arange(15, dtype="<f8").reshape(3, 5) / 8
    fixed = (np.arange(15, dtype="<f8").reshape(3, 5) + 1) / 16
    np.save(tmp_path / "initial.npy", initial)
    np.save(tmp_path / "fixed.npy", fixed)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            dict(
                schema="scar.python-entry",
                schema_version=1,
                source=source.name,
                entry="run",
                observe=dict(
                    arguments=[
                        {"npy": "initial.npy"},
                        {"npy": "fixed.npy"},
                        {"scalar": 4},
                    ]
                ),
            )
        )
    )
    _stage(tmp_path, "frontend", "build", "--manifest", request, "--out", graph)
    document = json.loads((graph / "graph.json").read_bytes())
    functions = [fn for unit in _payloads(graph, document) for fn in unit["functions"]]
    assert any(node["op"] == "for" for fn in functions for node in fn["nodes"])
    assert any(resource["kind"] == "CPU" for resource in document["resources"])
    assert any(
        event["callee"] == "prepare" for event in document["observation"]["events"]
    )
    assert document["entry_abi"]["entry"] == "run"

    shape = tmp_path / "shape.json"
    shape.write_text("[3, 5]\n")
    graph = graph.rename(tmp_path / "relocated-graph")
    _stage(
        tmp_path,
        "middle",
        "plan",
        "--graph",
        graph,
        "--out",
        plan,
        "--input-shapes",
        shape,
        "--trip-count",
        4,
    )
    graph.rename(tmp_path / "hidden-frontend")
    plan = plan.rename(tmp_path / "relocated-plan")
    result = _stage(tmp_path, "backend", "compile", "--plan", plan, "--out", delivery)
    assert result["status"] == "COMPILED"

    inputs, output = tmp_path / "input.bin", tmp_path / "output.bin"
    inputs.write_bytes(initial.tobytes() + fixed.tobytes())
    _run([result["executable"], inputs, output], tmp_path)
    assert (
        output.read_bytes()
        == runpy.run_path(str(source))["run"](initial, fixed, 4).tobytes()
    )


@pytest.mark.parametrize("python_binding", (False, True))
def test_locked_mixed_target_three_stages(tmp_path, python_binding):
    sources = tmp_path / "source"
    sources.mkdir()
    units = [
        (
            "old.c",
            "c",
            "c89",
            "int old(x)\nint x;\n"
            "{ void *opaque = &x; const int *value = opaque; return *value + 1; }\n",
        ),
        (
            "new.c",
            "c",
            "c17",
            "int modern(int x) { return x * 2; }\n"
            "double scale_value(double x) { return 2.0 * x; }\n"
            "void initialize(double *a, double *b, int n) {\n"
            "    int i;\n"
            "    for (i = 0; i < n; ++i) a[i] = 1.0 + (double)i / n;\n"
            "    for (i = 0; i < n; ++i) b[i] = a[i];\n"
            "}\n",
        ),
        (
            "cpp.cpp",
            "c++",
            "c++17",
            """
extern "C" int newest(int x) { return x - 3; }
int destroyed = 0;
struct Member {
    int value;
    Member() : value(42) {}
};
struct Holder { Member member; };
Holder holder;
struct Copied { int value; };
struct Base {
    int initial;
    Base(int x) : initial(x) {}
    virtual int read() const { return initial; }
    virtual ~Base() { destroyed += initial; }
};
struct Derived : Base {
    int& current;
    Derived(int& x) : Base(x), current(x) {}
    int read() const { return initial + current; }
    ~Derived() { destroyed += current; }
};
extern "C" int object_protocol() {
    if (holder.member.value != 42) return 2;
    Copied original;
    original.value = 9;
    Copied copied = original;
    if (copied.value != 9) return 3;
    int x = 3;
    {
        Derived local(x);
        Base* heap = new Derived(x);
        x = 7;
        if (local.read() != 10 || heap->read() != 10) return 1;
        delete heap;
    }
    return destroyed != 20;
}
""",
        ),
        (
            "main.cpp",
            "c++",
            "c++98",
            """
#include <cstdio>
extern "C" int old(int);
extern "C" int modern(int);
extern "C" int newest(int);
extern "C" int object_protocol();
extern "C" int formula_(int&);
extern "C" int extra(int*);
extern "C" void initialize(double*, double*, int);
extern "C" void initialize_fortran(int, double*, double*);
extern "C" void legacy_update__(const int*, double*, const float*);
extern "C" float legacy_bias__(const float*);
extern "C" void legacy_text__(char*, __SIZE_TYPE__);
#ifdef SCAR_PYTHON_ABI
extern "C" double python_adjust(double);
#endif
int main() {
    if (object_protocol()) return 7;
    int n;
    if (std::scanf("%d", &n) != 1) return 1;
    double first[5], second[5], overlapping[6];
    initialize(first, second, 5);
    initialize(overlapping, overlapping + 1, 5);
    for (int i = 0; i < 5; ++i) {
        if (first[i] != second[i] || overlapping[i + 1] != 1.0) return 2;
    }
    initialize_fortran(5, first, second);
    initialize_fortran(0, first, second);
    for (int i = 0; i < 5; ++i) {
        if (first[i] != 2.5 * (i + 1) || second[i] != first[i]) return 3;
    }
    const int extent[] = {2, 3};
    const float factor = -0.5f;
    double field[] = {1., 2., 3., 4., 5., 6.};
    legacy_update__(extent, field, &factor);
    if (legacy_bias__(&factor) != 0.75f) return 4;
    for (int i = 0; i < 6; ++i) {
        if (field[i] != i + 1.75) return 5;
    }
    char message[6] = {};
    legacy_text__(message, sizeof(message));
    if (message[0] != 'S' || message[3] != 'R' || message[4] != ' ') return 6;
    std::printf("%d\\n", old(n) + modern(n) + newest(n) + formula_(n) + extra(&n));
#ifdef SCAR_PYTHON_ABI
    std::printf("%.1f\\n", python_adjust(n));
#endif
    return 0;
}
""",
        ),
        (
            "formula.f",
            "fortran",
            "legacy",
            "      INTEGER FUNCTION FORMULA(N)\n"
            "      INTEGER N\n      FORMULA = N * N + 2\n      END\n",
        ),
        (
            "extra.f90",
            "fortran",
            "f2008",
            """
#if 1
function extra(n) result(value) bind(C)
  use, intrinsic :: iso_c_binding, only: c_int
  implicit none
  integer(c_int), intent(in) :: n
  integer(c_int) :: value
  value = 2*n
end function extra
#else
function extra(n) result(value) bind(C)
  use, intrinsic :: iso_c_binding, only: c_int
  integer(c_int), intent(in) :: n
  integer(c_int) :: value
  value = -n
end function extra
#endif

subroutine fill_pair(n,left,right) bind(C, name="initialize_fortran")
  use, intrinsic :: iso_c_binding, only: c_int, c_double
  implicit none
  integer(c_int), value, intent(in) :: n
  real(c_double), intent(inout) :: left(*), right(*)
  integer(c_int) :: i
  do i = 1,n
    left(i) = 2.5_c_double * real(i,c_double)
  end do
  do i = 1,n
    right(i) = left(i)
  end do
end subroutine fill_pair
""",
        ),
        (
            "legacy.f90",
            "fortran",
            "gnu",
            """
subroutine legacy_update(extent, values, factor)
  implicit none
  integer, intent(in) :: extent(2)
  real(kind=8), intent(inout) :: values(extent(1), extent(2))
  real, intent(in) :: factor
  real :: legacy_bias
  external legacy_bias
  real :: bias
  bias = 0.0 + &
    +legacy_bias(factor)
  values = values + bias
end subroutine legacy_update

real function legacy_bias(factor) result(bias)
  implicit none
  real, intent(in) :: factor
  bias = factor + 1.25
end function legacy_bias

subroutine legacy_text(message)
  implicit none
  character(len=*), intent(out) :: message
  message = "SCAR"
end subroutine legacy_text
""",
        ),
    ]
    files, profiles, objects = {}, [], []
    for index, (name, language, standard, text) in enumerate(units):
        source = sources / name
        raw = text.encode()
        source.write_bytes(raw)
        files[name] = {"sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}
        options = ["-std=" + standard]
        if name == "extra.f90":
            options.append("-cpp")
        if name == "legacy.f90":
            options.append("-fsecond-underscore")
        unit_flags = (
            [*options, "-DSCAR_PYTHON_ABI"]
            if python_binding and name == "main.cpp"
            else options
        )
        profiles.append(dict(path=name, language=language, flags=unit_flags))
        compiler = {"c": "gcc", "c++": "g++", "fortran": "gfortran"}[language]
        object_file = tmp_path / f"baseline-{index}.o"
        _run([compiler, *options, "-O2", "-c", source, "-o", object_file], tmp_path)
        objects.append(object_file)
    baseline = tmp_path / "baseline"
    _run(["g++", *objects, "-lgfortran", "-lm", "-o", baseline], tmp_path)
    case = tmp_path / "input.txt"
    case.write_bytes(b"7\n")
    expected = b"91\n"
    if python_binding:
        source = sources / "adjust.py"
        raw = b"def adjust(value):\n    return native_scale(value) + 0.5\n"
        source.write_bytes(raw)
        files[source.name] = {"sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}
        profiles.append(
            dict(
                path=source.name,
                language="python",
                flags=[],
                abi=dict(
                    exports={"adjust": "python_adjust"},
                    imports={"native_scale": "scale_value"},
                ),
            )
        )
        provider = tmp_path / "original-provider.so"
        _run(["gcc", "-shared", "-fPIC", sources / "new.c", "-o", provider], tmp_path)
        scale = CDLL(provider).scale_value
        scale.argtypes, scale.restype = [c_double], c_double
        result = runpy.run_path(str(source), init_globals={"native_scale": scale})[
            "adjust"
        ](7.0)
        expected += f"{result:.1f}\n".encode()
    manifest = tmp_path / "request.json"
    manifest.write_text(
        json.dumps(
            dict(
                schema="scar.native-entry",
                schema_version=1,
                language="mixed",
                source_root=str(sources),
                entry="main",
                units=profiles,
                source_lock={"files": files},
                baseline=dict(argv=[str(baseline)], cwd=str(tmp_path)),
                cases=(
                    []
                    if python_binding
                    else [dict(id="mixed", stdin=str(case), comparison="exact")]
                ),
            )
        )
    )
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(
        tmp_path,
        "frontend",
        "build",
        "--manifest",
        manifest,
        "--jobs",
        3,
        "--out",
        graph,
    )
    graph_path = graph / "graph.json"
    document = json.loads(graph_path.read_bytes())
    units = _payloads(graph, document)
    bodies = {
        f["signature"]["name"]: f
        for unit in units
        for f in unit["functions"]
        if f["defined"]
    }
    assert {
        "extra",
        "initialize_fortran",
        "legacy_update__",
        "legacy_bias__",
        "legacy_text__",
    } <= bodies.keys()
    assert bodies["legacy_bias__"]["signature"]["result"] == "f32"
    assert not any("fragments" in u for u in units)
    if python_binding:
        assert "scale_value" in bodies["python_adjust"]["calls"]
    original = baseline.rename(tmp_path / "hidden-original")
    sources.rename(tmp_path / "hidden-sources")
    graph = graph.rename(tmp_path / "relocated-graph")
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    graph.rename(tmp_path / "hidden-frontend")
    plan = plan.rename(tmp_path / "relocated-plan")
    (plan / "plan.json").rename(tmp_path / "hidden-logical-report")
    result = _stage(
        tmp_path,
        "backend",
        "compile",
        "--ir",
        plan / "exec-ir.json",
        "--physical",
        "serial",
        "--out",
        delivery,
    )
    assert result["status"] == "COMPILED"
    assert (delivery / "project/src/program.cpp").is_file()
    executable = Path(result["executable"])
    assert _run([original], tmp_path, input=b"7\n") == b"91\n"
    assert _run([executable], tmp_path, input=b"7\n") == expected


def test_fixed_form_program_emits_cpp20(tmp_path):
    data = b"      PROGRAM LEGACY\n      WRITE (*,'(I4)') 8\n      END\n"
    source = tmp_path / "legacy.f"
    source.write_bytes(data)
    baseline = tmp_path / "original"
    _run(["gfortran", str(source), "-o", str(baseline)], tmp_path)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            dict(
                schema="scar.native-entry",
                schema_version=1,
                language="fortran",
                source_root=str(tmp_path),
                entry="main",
                units=[dict(path=source.name, flags=["-std=legacy"])],
                source_lock=dict(
                    files={
                        source.name: dict(
                            sha256=sha256(data).hexdigest(), size_bytes=len(data)
                        )
                    }
                ),
                baseline=dict(argv=[str(baseline)], cwd=str(tmp_path)),
                cases=[dict(id="fixed", comparison="exact")],
            )
        )
    )
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--manifest", request, "--out", graph)
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    result = _stage(tmp_path, "backend", "compile", "--plan", plan, "--out", delivery)
    assert result["status"] == "COMPILED"
    assert _run([result["executable"]], tmp_path) == _run([baseline], tmp_path)


@pytest.fixture(scope="module")
def coordinate_library(tmp_path_factory):
    """Emit one native library so every shape tests the identical physical schedule."""
    from scar.backend.cpp20 import lower_scheduled, print_target
    from scar.backend.planning import schedule_ir
    from scar.middle.execution import prepare, seal
    from scar.middle.normalization import normalize

    folder = tmp_path_factory.mktemp("map-coordinates")

    def var(name):
        return dict(op="var", name=name)

    def integer(value):
        return dict(op="literal", type="i64", value=value)

    def binary(operator, left, right):
        return dict(op="binary", operator=operator, left=left, right=right)

    def cell(name, index):
        return dict(op="index", base=var(name), indices=[index])

    pointer = dict(kind="ptr", to="i64")
    params = [dict(name=k, type="i64") for k in ("n0", "n1", "w0", "w1")]
    params += [dict(name=k, type=pointer) for k in ("out", "input")]
    i = var("i")
    col = binary("mod", i, var("w0"))
    row = binary("mod", binary("div", i, var("w0")), var("w1"))
    read = cell("input", binary("add", col, binary("mul", integer(7), row)))
    write = dict(op="assign", target=cell("out", i), value=binary("add", read, i))
    extent = binary("mul", var("n0"), var("n1"))
    mapping = dict(op="map", site="view/0", index="i", extent=extent, body=[write])
    alias = dict(
        op="decl",
        name="alias",
        type=pointer,
        init=dict(op="unary", operator="addr", value=var("w0")),
    )
    change = dict(
        op="assign",
        target=cell("alias", integer(0)),
        value=binary("add", integer(2), binary("band", i, integer(1))),
    )
    # The x radix is now mutable, while a second, immutable y radix on its own
    # cannot authorize rewriting the dependent quotient/remainder chain.
    mutable = dict(mapping, site="mutable/0", body=[change, write])
    plain = dict(name="map_view", result="void", params=params, body=[mapping])
    other = dict(
        name="map_mutable", result="void", params=params, body=[alias, mutable]
    )
    entry = dict(name="main", result="i32", params=[], body=[])
    bound = dict(
        entry="main",
        functions=[entry, plain, other],
        externs=[],
        globals=[],
        records=[],
        declarations={},
        abi_environment={},
    )
    normalized, issues = normalize(bound)
    assert not issues
    module, facts = prepare(normalized)
    assert facts["iteration_layouts"] == 1
    source = seal(
        module,
        folder,
        status="READY",
        graph_sha256=None,
        contract={},
        logical_optimization={},
    )
    scheduled = schedule_ir(source, folder / "b1", policy="serial")
    physical = json.loads((scheduled.parent / "physical-plan.json").read_bytes())
    assert physical["regions"]["view/0"]["serial_layout"] == "mixed_radix"
    assert physical["regions"]["mutable/0"]["serial_layout"] == "linear"
    target = lower_scheduled(scheduled, folder / "b2a")
    delivery = print_target(target, folder / "b2b")
    folder = Path(delivery["project"])
    names = [str(Path(path).relative_to(folder)) for path in delivery["sources"]]
    # The second map has an escaped radix: the actual indirect update must be
    # visible to subsequent uses of quotient/remainder in the same iteration.
    translation = (folder / names[0]).read_text()
    first, second = translation.split('extern "C" void map_mutable')
    assert "scar_axis_" in first
    assert "v_i % v_w0" in second
    library = folder / "maps.so"
    _run(
        [
            "c++",
            "-std=c++20",
            "-O3",
            "-Werror",
            "-shared",
            "-fPIC",
            "-Iinclude",
            *names,
            "-o",
            str(library),
        ],
        folder,
    )
    return CDLL(str(library))


@pytest.mark.parametrize(
    "width,height,read_width,read_height,overlap,mutable",
    [
        (3, 2, 3, 2, False, False),
        (3, 2, 3, 2, True, False),
        (3, 2, 2, 3, True, False),
        (0, 4, 0, 4, False, False),
        (-3, -2, 3, 2, False, False),
        (3, 2, 3, 2, True, True),
    ],
)
def test_serial_map_coordinates(
    coordinate_library, width, height, read_width, read_height, overlap, mutable
):
    from ctypes import POINTER, byref, cast, c_int64, sizeof

    values = list(range(128))
    expected = values.copy()
    offset = 1 if overlap else 64
    actual = (c_int64 * len(values))(*values)
    procedure = (
        coordinate_library.map_mutable if mutable else coordinate_library.map_view
    )
    pointer = POINTER(c_int64)
    procedure.argtypes = [c_int64] * 4 + [pointer, pointer]
    procedure.restype = None
    procedure(
        width,
        height,
        read_width,
        read_height,
        cast(byref(actual, offset * sizeof(c_int64)), pointer),
        actual,
    )
    for i in range(width * height):
        if mutable:
            read_width = 2 + (i & 1)
        expected[offset + i] = (
            expected[i % read_width + 7 * ((i // read_width) % read_height)] + i
        )
    assert list(actual) == expected


def test_cpp20_global_and_lexical_bindings(tmp_path):
    from scar.contract import IR, document, member, write_json

    def literal(value):
        return dict(op="literal", type="i32", value=value)

    def binary(operator, left, right):
        return dict(op="binary", operator=operator, left=left, right=right)

    def declaration(name, value):
        return dict(op="decl", name=name, type="i32", init=value)

    def assign(target, value):
        return dict(op="assign", target=target, value=value)

    n = dict(op="var", name="n")
    global_n = dict(n, scope="global")
    total = dict(op="var", name="sum")
    accumulate = assign(total, binary("add", total, n))
    difference = dict(
        name="difference",
        params=[dict(name="n", type="i32")],
        result="i32",
        body=[dict(op="return", value=binary("sub", global_n, n))],
    )
    call = dict(op="call", callee="difference", args=[literal(5)])
    body = [
        declaration("sum", n),
        dict(
            op="block",
            body=[
                declaration("n", literal(3)),
                accumulate,
                assign(global_n, binary("add", global_n, n)),
            ],
        ),
        accumulate,
        dict(
            op="for",
            init=[declaration("n", literal(0))],
            condition=binary("lt", n, literal(2)),
            step=dict(op="unary", operator="postinc", value=n),
            body=[accumulate],
        ),
        dict(op="map", site="main/0", extent=literal(2), index="n", body=[accumulate]),
        dict(op="return", value=binary("ne", binary("add", total, call), literal(36))),
    ]
    main = dict(name="main", params=[], result="i32", body=body)
    source = tmp_path / "exec-ir.json"
    write_json(
        source,
        document(
            IR,
            status="READY",
            module=dict(entry="main", globals=[declaration("n", literal(10))]),
            functions=[member(tmp_path, "functions", fn) for fn in (main, difference)],
        ),
    )
    delivery = _stage(
        tmp_path,
        "backend",
        "compile",
        "--ir",
        source,
        "--physical",
        "serial",
        "--out",
        tmp_path / "delivery",
    )
    exe = delivery["executable"]
    assert _run([exe], tmp_path) == b""
    symbols = _run(["nm", "-g", "-P", "--defined-only", exe], tmp_path).splitlines()
    assert any(symbol.startswith(b"n D ") for symbol in symbols)

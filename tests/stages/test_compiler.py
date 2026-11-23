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
sys.meta_path.insert(0, BlockOtherStages())
from scar.cli import main
raise SystemExit(main(sys.argv[2:]))
"""


def _run(command, folder, *, input=None):
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(scar.__file__).resolve().parents[1]),
        PYTHONDONTWRITEBYTECODE="1",
    )
    completed = subprocess.run(
        command, cwd=folder, env=env, input=input, capture_output=True, timeout=120
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    return completed.stdout


def _stage(folder, phase, *arguments):
    return json.loads(
        _run(
            [sys.executable, "-B", "-c", _PHASE_COMMAND, phase, *map(str, arguments)],
            folder,
        )
    )


def _payloads(graph, document):
    return [
        json.loads((graph / unit["graph"]["path"]).read_bytes())
        for unit in document["units"]
    ]


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

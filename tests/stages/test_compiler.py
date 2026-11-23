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
    env = dict(os.environ, PYTHONPATH=str(Path(scar.__file__).resolve().parents[1]), PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(command, cwd=folder, env=env, input=input, capture_output=True, timeout=120)
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    return completed.stdout


def _stage(folder, phase, *arguments):
    return json.loads(_run([sys.executable, "-B", "-c", _PHASE_COMMAND, phase, *map(str, arguments)], folder))


def test_python_array_three_stages_and_independent_output(tmp_path):
    np = pytest.importorskip("numpy")
    source = tmp_path / "simulation.py"
    source.write_text("""
def prepare(initial, fixed):
    return initial + fixed * fixed

def run(initial, fixed, count):
    for unused in range(count):
        initial = prepare(initial, fixed)
    return initial
""", encoding="utf-8")
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    initial = np.arange(15, dtype="<f8").reshape(3, 5) / 8
    fixed = (np.arange(15, dtype="<f8").reshape(3, 5) + 1) / 16
    np.save(tmp_path / "initial.npy", initial)
    np.save(tmp_path / "fixed.npy", fixed)
    request = tmp_path / "request.json"
    request.write_text(json.dumps(dict(schema="scar.python-entry", schema_version=1, source=source.name,
        entry="run", observe=dict(arguments=[{"npy": "initial.npy"}, {"npy": "fixed.npy"}, {"scalar": 4}]))))
    _stage(tmp_path, "frontend", "build", "--manifest", request, "--out", graph)
    document = json.loads((graph / "graph.json").read_bytes())
    assert any(region["kind"].lower() == "loop" for region in document["regions"])
    assert any(resource["kind"] == "CPU" for resource in document["resources"])
    assert any(edge["kind"] == "OBSERVED_CALL" for edge in document["edges"])
    assert document["program"]["elementwise"]["entry"] == "run"

    shape = tmp_path / "shape.json"
    shape.write_text("[3, 5]\n")
    graph = graph.rename(tmp_path / "relocated-graph")
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan, "--input-shapes", shape, "--trip-count", 4)
    graph.rename(tmp_path / "hidden-frontend")
    plan = plan.rename(tmp_path / "relocated-plan")
    result = _stage(tmp_path, "backend", "compile", "--plan", plan, "--out", delivery)
    assert result["status"] == "TRIAL_COMPILED"
    assert result["deployment_authorized"] is False

    inputs, output = tmp_path / "input.bin", tmp_path / "output.bin"
    inputs.write_bytes(initial.tobytes() + fixed.tobytes())
    _run([result["executable"]["path"], inputs, output], tmp_path)
    assert output.read_bytes() == runpy.run_path(str(source))["run"](initial, fixed, 4).tobytes()


@pytest.mark.parametrize("python_binding", (False, True))
def test_locked_mixed_target_three_stages(tmp_path, python_binding):
    if any(shutil.which(name) is None for name in ("gcc", "g++", "gfortran", "cmake", "nm", "readelf")):
        pytest.skip("native C++20 build tools unavailable")
    prefix = Path(os.environ.get("SCAR_CI_F2C_PREFIX", "/usr"))
    arch = _run(["gcc", "-print-multiarch"], tmp_path).decode().strip()
    f2c = {"f2c": prefix / "bin/f2c", "f2c_header": prefix / "include/f2c.h",
           "f2c_archive": prefix / "lib" / arch / "libf2c.a"}
    if not all(path.is_file() for path in f2c.values()):
        pytest.skip("native build requires f2c and libf2c2-dev")
    sources = tmp_path / "source"
    sources.mkdir()
    units = [
        ("old.c", "c", "c89", "int old(x)\nint x;\n{ return x + 1; }\n"),
        ("new.c", "c", "c17", "int modern(int x) { return x * 2; }\n"
         "double scale_value(double x) { return 2.0 * x; }\n"
         "void initialize(double *a, double *b, int n) {\n"
         "    int i;\n"
         "    for (i = 0; i < n; ++i) a[i] = 1.0 + (double)i / n;\n"
         "    for (i = 0; i < n; ++i) b[i] = a[i];\n"
         "}\n"),
        ("cpp.cpp", "c++", "c++17", 'extern "C" int newest(int x) { return x - 3; }\n'),
        ("main.cpp", "c++", "c++98", '''
#include <cstdio>
extern "C" int old(int);
extern "C" int modern(int);
extern "C" int newest(int);
extern "C" int formula_(int*);
extern "C" int extra(int*);
extern "C" void initialize(double*, double*, int);
extern "C" void initialize_fortran(int, double*, double*);
#ifdef SCAR_PYTHON_ABI
extern "C" double python_adjust(double);
#endif
int main() {
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
    std::printf("%d\\n", old(n) + modern(n) + newest(n) + formula_(&n) + extra(&n));
#ifdef SCAR_PYTHON_ABI
    std::printf("%.1f\\n", python_adjust(n));
#endif
    return 0;
}
'''),
        ("formula.f", "fortran", "legacy", "      INTEGER FUNCTION FORMULA(N)\n"
         "      INTEGER N\n      FORMULA = N * N + 2\n      END\n"),
        ("extra.f90", "fortran", "f2008", '''
function extra(n) result(value) bind(C)
  use, intrinsic :: iso_c_binding, only: c_int
  implicit none
  integer(c_int), intent(in) :: n
  integer(c_int) :: value
  value = 2*n
end function extra

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
'''),
    ]
    files, profiles, objects = {}, [], []
    for index, (name, language, standard, text) in enumerate(units):
        source = sources / name
        raw = text.encode()
        source.write_bytes(raw)
        files[name] = {"sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}
        options = ["-std=" + standard]
        unit_flags = [*options, "-DSCAR_PYTHON_ABI"] if python_binding and name == "main.cpp" else options
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
        profiles.append(dict(path=source.name, language="python", flags=[],
                             abi=dict(exports={"adjust": "python_adjust"}, imports={"native_scale": "scale_value"})))
        provider = tmp_path / "original-provider.so"
        _run(["gcc", "-shared", "-fPIC", sources / "new.c", "-o", provider], tmp_path)
        scale = CDLL(provider).scale_value
        scale.argtypes, scale.restype = [c_double], c_double
        result = runpy.run_path(str(source), init_globals={"native_scale": scale})["adjust"](7.0)
        expected += f"{result:.1f}\n".encode()
    manifest = tmp_path / "request.json"
    manifest.write_text(json.dumps(dict(schema="scar.native-entry", schema_version=1, language="mixed",
        source_root=str(sources), entry="main", units=profiles, source_lock={"files": files},
        fortran={name: str(path.resolve()) for name, path in f2c.items()},
        baseline=dict(argv=[str(baseline)], cwd=str(tmp_path)),
        cases=[] if python_binding else [dict(id="mixed", stdin=str(case), comparison="exact")])))
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--manifest", manifest, "--jobs", 3, "--out", graph)
    graph_path = graph / "graph.json"
    document = json.loads(graph_path.read_bytes())
    material_path = Path(document["program"]["native"]["module"]["path"])
    assert not material_path.is_absolute()
    material = json.loads((graph / material_path).read_bytes())
    wrapper = "src/fortran-abi/scar_c_abi.cpp"
    assert {row["symbol"] for row in material["abi_units"][wrapper]} == {"extra", "initialize_fortran"}
    assert wrapper not in material["fragments"]
    assert "include/scar_generated/scar_f2c_call.hpp" not in material["fragments"]
    if python_binding:
        links = {row["value"] for row in document["edges"] if row["kind"] == "CALL"}
        assert {"python_adjust", "scale_value"} <= links
    document["evidence"] = {}
    graph_path.write_text(json.dumps(document))
    original = baseline.rename(tmp_path / "hidden-original")
    case.rename(tmp_path / "hidden-case")
    sources.rename(tmp_path / "hidden-sources")
    graph = graph.rename(tmp_path / "relocated-graph")
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    logical = json.loads((plan / "plan.json").read_bytes())
    assert all(row["resource"] is None and row["start_ns"] is None for row in logical["schedule"])
    rewrites = {row["region"] for row in logical["rewrites"] if row["rule"] == "FUSE_ADJACENT_COUNTED_PRODUCER_COPY"}
    regions = logical["lowering"]["bodies"][0]["regions"]
    typed = {row["symbol"]: row for row in regions if row["region"] in rewrites}
    assert {"initialize", "initialize_fortran"} <= typed.keys()
    assert typed["initialize_fortran"]["fragment"]["authority"] == "TYPED_ABI_EXPORT"
    graph.rename(tmp_path / "hidden-frontend")
    plan = plan.rename(tmp_path / "relocated-plan")
    result = _stage(tmp_path, "backend", "compile", "--plan", plan, "--out", delivery)
    assert result["status"] == ("BUILT" if python_binding else "VALIDATED")
    assert (delivery / "project" / wrapper).is_file()
    assert (delivery / "project/include/scar_generated/scar_f2c_call.hpp").is_file()
    generated = json.loads((delivery / "project/native-region-lowering.json").read_bytes())
    emitted = {row["region"] for row in generated["regions"] if row["status"] == "GUARDED_REGION"}
    assert {typed[name]["region"] for name in ("initialize", "initialize_fortran")} <= emitted
    executable = Path(result["executable"]["path"])
    assert _run([original], tmp_path, input=b"7\n") == b"91\n"
    assert _run([executable], tmp_path, input=b"7\n") == expected

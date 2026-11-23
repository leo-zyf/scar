"""Test the three public stages in mutually isolated interpreter processes."""

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

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
    _stage(tmp_path, "frontend", "build", "--entry", source, "--out", graph)
    document = json.loads((graph / "graph.json").read_bytes())
    assert any(region["kind"].lower() == "loop" for region in document["regions"])
    assert any(resource["kind"] == "CPU" for resource in document["resources"])

    shape = tmp_path / "shape.json"
    shape.write_text("[3, 5]\n")
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan, "--input-shapes", shape, "--trip-count", 4)
    graph.rename(tmp_path / "hidden-frontend")
    plan = plan.rename(tmp_path / "relocated-plan")
    result = _stage(tmp_path, "backend", "compile", "--plan", plan, "--out", delivery)
    assert result["status"] == "TRIAL_COMPILED"
    assert result["deployment_authorized"] is False

    initial = np.arange(15, dtype="<f8").reshape(3, 5) / 8
    fixed = (np.arange(15, dtype="<f8").reshape(3, 5) + 1) / 16
    inputs, output = tmp_path / "input.bin", tmp_path / "output.bin"
    inputs.write_bytes(initial.tobytes() + fixed.tobytes())
    _run([result["executable"]["path"], inputs, output], tmp_path)
    import runpy

    assert output.read_bytes() == runpy.run_path(str(source))["run"](initial, fixed, 4).tobytes()


def test_locked_mixed_target_three_stages(tmp_path):
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
        ("new.c", "c", "c17", "int modern(int x) { return x * 2; }\n"),
        ("cpp.cpp", "c++", "c++17", 'extern "C" int newest(int x) { return x - 3; }\n'),
        ("main.cpp", "c++", "c++98", '''
#include <cstdio>
extern "C" int old(int);
extern "C" int modern(int);
extern "C" int newest(int);
extern "C" int formula_(int*);
extern "C" int extra(int*);
int main() {
    int n;
    if (std::scanf("%d", &n) != 1) return 1;
    std::printf("%d\\n", old(n) + modern(n) + newest(n) + formula_(&n) + extra(&n));
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
'''),
    ]
    files, profiles, objects = {}, [], []
    for index, (name, language, standard, text) in enumerate(units):
        source = sources / name
        raw = text.encode()
        source.write_bytes(raw)
        files[name] = {"sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}
        options = ["-std=" + standard]
        profiles.append(dict(path=name, language=language, flags=options))
        compiler = {"c": "gcc", "c++": "g++", "fortran": "gfortran"}[language]
        object_file = tmp_path / f"baseline-{index}.o"
        _run([compiler, *options, "-O2", "-c", source, "-o", object_file], tmp_path)
        objects.append(object_file)
    baseline = tmp_path / "baseline"
    _run(["g++", *objects, "-lgfortran", "-lm", "-o", baseline], tmp_path)
    case = tmp_path / "input.txt"
    case.write_bytes(b"7\n")
    manifest = tmp_path / "request.json"
    manifest.write_text(json.dumps(dict(schema="scar.native-entry", schema_version=1, language="mixed",
        source_root=str(sources), entry="main", units=profiles, source_lock={"files": files},
        fortran={name: str(path.resolve()) for name, path in f2c.items()},
        baseline=dict(argv=[str(baseline)], cwd=str(tmp_path)),
        cases=[dict(id="mixed", stdin=str(case), comparison="exact")])))
    graph, plan, delivery = (tmp_path / name for name in ("graph", "plan", "delivery"))
    _stage(tmp_path, "frontend", "build", "--manifest", manifest, "--jobs", 3, "--out", graph)
    original = baseline.rename(tmp_path / "hidden-original")
    case.rename(tmp_path / "hidden-case")
    sources.rename(tmp_path / "hidden-sources")
    _stage(tmp_path, "middle", "plan", "--graph", graph, "--out", plan)
    graph.rename(tmp_path / "hidden-frontend")
    plan = plan.rename(tmp_path / "relocated-plan")
    result = _stage(tmp_path, "backend", "compile", "--plan", plan, "--out", delivery)
    assert result["status"] == "VALIDATED"
    executable = Path(result["executable"]["path"])
    assert _run([executable], tmp_path, input=b"7\n") == _run([original], tmp_path, input=b"7\n") == b"91\n"

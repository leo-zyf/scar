"""Mixed native targets exercise the three public compiler stages in separate processes."""
from ctypes import CDLL
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scar.analysis.replay_transaction import replay_transaction
from scar.analysis.target_graph import TargetHardwareGraph
from scar.artifacts import file_reference
from scar.compiler.backend.cpp20 import replay_native_revision
from scar.compiler.contracts import ExecutionPlanRef
from scar.compiler.frontend.clang import stage_source_revision
from scar.compiler.middle.plan import read_native_binding, read_plan_contract
from scar.ir.execution_recipe import read_artifact


ROOT = Path(__file__).resolve().parents[2]
SOURCE_UNITS = (
    ("old.c", "c", "c89", "static int increment(int n) { return n + 1; }\n"
     "int old_value(n)\nint n;\n{ return increment(n); }\n"),
    ("modern.c", "c", "c17", "int c_value(int n) { return n * 2; }\n"),
    ("modern.cpp", "c++", "c++17", 'extern "C" int cpp_value(int n) { return n - 3; }\n'),
    ("main.cpp", "c++", "c++98", r'''#include <cstdio>
extern "C" int old_value(int);
extern "C" int c_value(int);
extern "C" int cpp_value(int);
extern "C" int formula_(int *);
int main() {
    int n;
    if (std::scanf("%d", &n) != 1) return 2;
    std::printf("%d\n", old_value(n) + c_value(n) + cpp_value(n) + formula_(&n));
    return 0;
}
'''),
)
FORTRAN_UNITS = {
    "fixed-f77": ("formula.f", "legacy",
                  "      INTEGER FUNCTION FORMULA(N)\n      INTEGER N\n"
                  "      FORMULA = N * N + 2\n      END\n"),
    "free-f2008": ("formula.f90", "f2008", '''function formula(n) result(value) bind(C, name="formula_")
  use, intrinsic :: iso_c_binding, only: c_int
  implicit none
  integer(c_int), intent(in) :: n
  integer(c_int) :: value
  value = n * n + 2
end function formula
'''),
}
INPUTS = {"positive": (b"7\n", b"77\n"), "negative": (b"-3\n", b"-3\n")}


def _run(directory, name, args, *, stdin=None, environment=None):
    argv = list(map(str, args))
    (directory / f"{name}.command.json").write_text(json.dumps(argv) + "\n", encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", LC_ALL="C")
    env.update(environment or {})
    result = subprocess.run(argv, cwd=directory, env=env, input=stdin, capture_output=True, timeout=120)
    (directory / f"{name}.stdout").write_bytes(result.stdout)
    (directory / f"{name}.stderr").write_bytes(result.stderr)
    assert result.returncode == 0, (f"{name} failed in {directory}\n"
                                    f"{result.stdout.decode(errors='replace')}\n"
                                    f"{result.stderr.decode(errors='replace')}")
    return result.stdout


def _tools(folder, profile):
    required = ("gcc", "g++", "gfortran", "cmake", "ninja", "nm", "readelf", "taskset")
    missing = [name for name in required if shutil.which(name) is None]
    try:
        CDLL("libclang-18.so.18")
    except OSError:
        missing.append("libclang1-18")
    if missing:
        pytest.skip("native handoff requires " + ", ".join(missing))
    prefix = Path(os.environ.get("SCAR_CI_F2C_PREFIX", "/usr"))
    multiarch = _run(folder, "gcc-multiarch", ["gcc", "-print-multiarch"]).decode().strip()
    f2c = {"f2c": prefix / "bin/f2c", "f2c_header": prefix / "include/f2c.h",
           "f2c_archive": prefix / "lib" / multiarch / "libf2c.a"}
    if not all(path.is_file() for path in f2c.values()):
        pytest.skip("native handoff requires f2c and libf2c2-dev or SCAR_CI_F2C_PREFIX")
    config = {key: str(path.resolve()) for key, path in f2c.items()}
    if profile == "free-f2008":
        if parser := os.environ.get("SCAR_CI_FPARSER_PATH"):
            config["fparser_path"] = str(Path(parser).resolve(strict=True))
        elif importlib.util.find_spec("fparser") is None:
            pytest.skip("Fortran 2008 handoff requires fparser or SCAR_CI_FPARSER_PATH")
    return config


@pytest.fixture(scope="module", params=tuple(FORTRAN_UNITS))
def native_request(tmp_path_factory, request):
    profile = request.param
    folder = tmp_path_factory.mktemp(f"scar-handoff-{profile}")
    config = _tools(folder, profile)
    sources = folder / "source"
    sources.mkdir()
    fortran_name, fortran_standard, fortran_text = FORTRAN_UNITS[profile]
    rows = (*SOURCE_UNITS, (fortran_name, "fortran", fortran_standard, fortran_text))
    units, files, objects = [], {}, []
    for index, (name, language, standard, text) in enumerate(rows):
        if name == "modern.c" and profile == "free-f2008":
            text = ("int keep_value(int n) { int same = n; return same; }\n"
                    "int c_value(int n) { return keep_value(n) * 2; }\n")
        path = sources / name
        path.write_text(text, encoding="utf-8")
        data = path.read_bytes()
        files[name] = {"sha256": sha256(data).hexdigest(), "size_bytes": len(data)}
        compiler = {"c": "gcc", "c++": "g++", "fortran": "gfortran"}[language]
        flags = ["-std=" + standard]
        unit = {"path": name, "language": language, "flags": flags}
        units.append(unit)
        obj = folder / f"baseline-{index}.o"
        _run(folder, f"baseline-{index}", [compiler, *flags, "-O2", "-c", path, "-o", obj])
        objects.append(obj)
    original = folder / "original"
    _run(folder, "baseline-link", ["g++", *objects, "-lgfortran", "-lm", "-o", original])
    cases = []
    for name, (data, _) in INPUTS.items():
        path = folder / f"{name}.stdin"
        path.write_bytes(data)
        cases.append({"id": name, "stdin": str(path), "comparison": "exact"})
    document = {"schema": "scar.native-entry", "schema_version": 1, "language": "mixed",
                "source_root": str(sources), "entry": "main", "units": units, "fortran": config,
                "source_lock": {"files": files}, "cases": cases,
                "baseline": {"argv": [str(original)], "cwd": str(folder)}}
    manifest = folder / "entry.json"
    manifest.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return folder, manifest, document


@pytest.fixture(scope="module")
def native_graph(native_request):
    folder, manifest, _ = native_request
    graph = folder / "graph"
    _run(folder, "graph", [sys.executable, "-m", "scar", "build", "--manifest", manifest,
                            "--out", graph, "--jobs", "2"])
    return graph, json.loads((graph / "result.json").read_bytes())


@pytest.fixture(scope="module")
def native_plan(native_graph):
    graph, _ = native_graph
    plan = graph.parent / "plan"
    _run(graph.parent, "plan", [sys.executable, "-m", "scar", "plan", "--graph", graph,
                                 "--out", plan, "--repetitions", "1", "--case-seconds", "60"])
    return plan, json.loads((plan / "result.json").read_bytes())


@pytest.fixture(scope="module")
def cpp20_delivery(native_plan):
    plan, _ = native_plan
    output = plan.parent / "delivery"
    frozen = ExecutionPlanRef.read(plan / "compiler-plan.json")
    _, binding = read_native_binding(frozen)
    environment = None
    if variant := binding.get("trial_variant"):
        trial = Path(variant["uri"]).parent / "project"
        (trial / "CMakeLists.txt").write_text('message(FATAL_ERROR "stale trial build script")\n', encoding="utf-8")
        environment = {"CMAKE_GENERATOR": "SCAR unavailable test generator"}
    _run(plan.parent, "delivery", [sys.executable, "-m", "scar", "compile", "--plan",
                                     plan / "compiler-plan.json", "--out", output, "--repetitions", "1"],
         environment=environment)
    return output, json.loads((output / "result.json").read_bytes())


def test_native_frontend_locks_each_dialect(native_request, native_graph):
    _, _, manifest = native_request
    _, graph = native_graph
    artifacts = graph["artifacts"]
    request = read_artifact(artifacts["request"])
    assert request["source_lock"] == manifest["source_lock"]
    assert request["units"] == manifest["units"]
    evidence = read_artifact(artifacts["graph_evidence"])
    captures = {row["unit"]: row for row in evidence["units"]}
    assert set(captures) == {unit["path"] for unit in manifest["units"]}
    for unit in manifest["units"]:
        row = captures[unit["path"]]
        assert row["language"] == unit["language"] and row["flags"] == unit["flags"]
        assert row["standard"] == unit["flags"][0].removeprefix("-std=")
        assert row["artifacts"]["types"]["schema"] == (
            "gfortran.semantics" if unit["language"] == "fortran" else "scar.clang-unit")
        for role in ("types", "control", "calls"):
            assert read_artifact(row["artifacts"][role], json_document=False)


def test_native_graph_links_calls_and_platform(native_graph):
    _, graph = native_graph
    artifacts = graph["artifacts"]
    program = read_artifact(artifacts["program"])
    assert program["regions"] and graph["graph"]["reachable_regions"] >= len(SOURCE_UNITS) + 1
    calls = read_artifact(artifacts["calls"])
    edges = {row["callee"]["symbol"] for row in calls["calls"]
             if row["status"] == "BOUND" and row["reachable"] and row["operation"]}
    assert {"old_value", "c_value", "cpp_value", "formula_"} <= edges
    hardware = artifacts["hardware"]
    read_artifact(hardware)
    target = TargetHardwareGraph(Path(hardware["uri"]).parent)
    inventory = target.inventory
    assert inventory["cpu"]["online_cpu_ids"] and inventory["execution_constraints"]["allowed_cpu_ids"]


def test_native_middle_replays_cost_and_schedule(native_graph, native_plan, tmp_path):
    _, graph = native_graph
    folder, plan = native_plan
    assert plan["status"] == "PLAN_SCHEDULED"
    for role, status in (("rewrite", "IR_VALIDATED"), ("cost", "OBSERVED"), ("schedule", "SCHEDULED")):
        assert plan["layers"][role] == status
    frozen = ExecutionPlanRef.read(folder / "compiler-plan.json")
    for role in ("program", "hardware", "calls", "request", "graph_evidence"):
        assert frozen.artifacts[role] == graph["artifacts"][role]
    problem, schedule, verification = read_plan_contract(frozen)
    observation, binding = read_native_binding(frozen)
    assert problem.operations and schedule["schema"] == "scar.implementation-execution-plan"
    assert verification["feasible"] and binding["cpus"] == [observation["cpu"]]
    assert set(observation["case_inputs"]) == set(INPUTS)

    original = frozen.artifacts["rewrites"]
    ledger = tmp_path / "rewrites.json"
    ledger.write_bytes(read_artifact(original, json_document=False))
    revision = dict(frozen.artifacts, rewrites=file_reference(ledger, original["schema"]))
    with pytest.raises(ValueError, match="metadata dependency byte identity differs"):
        with replay_transaction() as transaction:
            assert replay_native_revision(revision) == replay_native_revision(revision)
            assert transaction.by_namespace["scar.cpp20-native-revision.v1"] == {"hits": 1, "misses": 1}
            with ledger.open("ab") as stream:
                stream.write(b"\n")
            replay_native_revision(revision)


def test_native_backend_delivers_cpp20_and_matches_locked_cases(native_request, native_plan, cpp20_delivery):
    _, _, request = native_request
    plan_path, plan = native_plan
    folder, delivery = cpp20_delivery
    assert delivery["delivery"] == "COMPARED"
    project = folder / "project"
    receipt = read_artifact(delivery["artifacts"]["cpp20_inputs"])
    cmake = (project / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "LANGUAGES CXX" in cmake and "CXX_STANDARD 20" in cmake
    assert receipt["schema"] == "scar.cpp20-project-inputs" and receipt["cxx_standard"] == 20
    assert all(Path(name).suffix == ".cpp" for name in receipt["configuration"]["sources"])
    translations = json.loads((project / "translation.json").read_bytes())
    old, = [row for row in translations if row["source"] == "old.c"]
    definition, = old["c_identifier_definitions"]
    assert definition["parameter_types"] == ["int"]
    assert b"old_value(int n)" in (project / old["output"]).read_bytes()
    assert delivery["artifacts"]["execution_plan"] == plan["artifacts"]["execution_plan"]
    executable = Path(delivery["executable"]["path"])
    assert executable.read_bytes().startswith(b"\x7fELF")
    frozen = ExecutionPlanRef.read(plan_path / "compiler-plan.json")
    observation, binding = read_native_binding(frozen)
    if any(unit["flags"] == ["-std=f2008"] for unit in request["units"]):
        assert set(observation["variants"]) == {"original", "canonical"}
        calibrated = read_artifact(binding["trial_variant"])["binary"]
        assert delivery["executable"]["sha256"] == calibrated["sha256"]
        assert executable.read_bytes() == read_artifact(calibrated, json_document=False)
    else:
        assert "trial_variant" not in binding

    standalone = folder / "standalone-build"
    _run(folder, "standalone-cmake", ["cmake", "-S", project, "-B", standalone, "-DCMAKE_BUILD_TYPE=Release"])
    _run(folder, "standalone-build", ["cmake", "--build", standalone, "--parallel", "2"])
    assert {row["case"] for row in delivery["validation"] if row["matches"]} == set(INPUTS)
    for case, (stdin, expected) in INPUTS.items():
        retained = folder / "validation" / case
        assert (retained / "1-original.stdout").read_bytes() == expected
        assert (retained / "1-generated.stdout").read_bytes() == expected
        assert _run(folder, f"standalone-{case}", [standalone / "scar_entry"], stdin=stdin) == expected


def test_native_lowering_requires_an_abi_bridge_for_kr_promotions(tmp_path):
    try:
        CDLL("libclang-18.so.18")
    except OSError:
        pytest.skip("C header lowering requires libclang1-18")
    source = tmp_path / "legacy.c"
    source.write_bytes(b"int narrowed(value)\nfloat value;\n{ return (int)value; }\n")
    request = {"language": "c", "source_root": str(tmp_path),
               "units": [{"path": source.name, "flags": ["-std=c89"]}],
               "source_lock": {"files": {source.name: file_reference(source, "scar.native-source")}}}
    plan = tmp_path / "lowering-plan.json"
    plan.write_text('{"schema":"test.lowering-plan"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="parameter promotion or adjustment requires an ABI bridge"):
        stage_source_revision(request, tmp_path / "project", plan=file_reference(plan, "test.lowering-plan"))

"""Recipe/profile replay reaches the existing backend without structural drift."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from test_cpu_hardware_placement import hardware
from test_loop_recipe import setup_recipe
from scar.codegen import loop as backend
from scar.codegen.recipe import build_recipe
from scar.compiler.loop_admission import admit_loop, check_loop_sources
from scar.compiler.loop_model import loop_digest
from scar.compiler.model import document
from scar.ir.execution_recipe import recipe_id
from scar.optimization.loop_recipe import estimate_loop_recipe
from scar.optimization.resource_solver import semantic_sha256


def test_recipe_lowering_replays_and_links_exact_backend_inputs_and_receipts(hardware, tmp_path, monkeypatch):
    recipe, loop, strategy, plan, layout = setup_recipe(hardware, tmp_path)
    assert admit_loop(loop, strategy, plan) == backend._admit(loop, strategy, plan)
    check_loop_sources(loop)
    bad_binding = replace(plan.resource_binding, target_sha256="0" * 64)
    with pytest.raises(ValueError, match="snapshot receipt"):
        admit_loop(loop, strategy, replace(plan, resource_binding=bad_binding))
    calls = []
    mismatch = [False]
    marker = object()
    def run(state, steps): return marker

    def build(actual_loop, directory, **arguments):
        # No compiler is run here. Existing native tests cover numerical/ABI
        # execution; this fixture isolates the recipe-to-backend call boundary.
        calls.append((actual_loop, arguments))
        assert actual_loop == loop
        assert arguments == dict(strategy=strategy, execution_plan=plan, layout=layout)
        directory.mkdir()
        library = directory / "declared-unit-test.so"
        library.write_bytes(b"unit-test-native-stand-in-not-an-ELF")
        capability = recipe["implementation"]["backend_capability_record"]
        metadata = dict(schema=backend.SCHEMA, schema_version=2, backend="cpp", loop_abi_version=1,
            build_id="b" * 64, loop_sha256=loop_digest(loop),
            strategy_sha256=semantic_sha256(strategy), execution_plan_sha256=semantic_sha256(document(plan)),
            contract_sha256="c" * 64, layout_sha256=semantic_sha256(layout),
            compiler=capability["compiler"]["argv"], compiler_version=capability["compiler"]["identity"],
            compiler_options=capability["compiler"]["options"] + (["-ffast-math"] if mismatch[0] else []),
            loop_implementation_sha256=capability["loop_implementation_sha256"])
        manifest = directory / "artifact.json"
        manifest.write_text(json.dumps(metadata))
        report = dict(metadata, metadata_path=str(manifest), library_path=str(library))
        return SimpleNamespace(loop=actual_loop, strategy=strategy, execution_plan=plan,
                               layout=layout, directory=directory, report=report, run=run)

    monkeypatch.setattr(backend, "build", build)
    output = tmp_path / "lowered"
    lowered = build_recipe(tmp_path / "recipe/recipe.json", output)
    estimate = estimate_loop_recipe(recipe)
    report = lowered.report
    assert len(calls) == 1 and lowered.native_directory == output / "native"
    assert lowered.loop == loop and lowered.execution_plan == plan and lowered.layout == layout
    assert lowered.run is run and lowered.run(object(), 400) is marker
    assert report["recipe_id"] == estimate["recipe_id"] == recipe["recipe_id"]
    receipt_path = output / "lowering-receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    assert report["lowering_receipt_sha256"] == sha256(receipt_path.read_bytes()).hexdigest()
    assert receipt["input_artifacts"] == recipe["artifacts"]
    assert receipt["recipe_sha256"] == sha256((output / "recipe.json").read_bytes()).hexdigest()
    assert receipt["native_ids"]["execution_plan_sha256"] == report["execution_plan_sha256"]
    assert receipt["native_library_sha256"] == sha256((output / "native/declared-unit-test.so").read_bytes()).hexdigest()
    assert receipt["compiler"]["options"] == list(backend.layout_compiler_options(plan))
    assert receipt["backend_capability_record"] == recipe["implementation"]["backend_capability_record"]
    assert not receipt["deployment_authorized"] and not receipt["physical_optimality_proven"]
    manifest = json.loads((output / "native/artifact.json").read_bytes())
    assert "recipe_id" not in manifest  # Existing native manifests keep their schema.
    changed = deepcopy(recipe)
    changed["control"]["temporal"]["count"] += 1
    changed["recipe_id"] = recipe_id(changed)
    with pytest.raises(ValueError, match="complete source/control/resource replay"):
        build_recipe(changed, tmp_path / "rejected")
    assert len(calls) == 1 and not (tmp_path / "rejected").exists()
    mismatch[0] = True
    with pytest.raises(ValueError, match="differs from recipe capability"):
        build_recipe(recipe, tmp_path / "wrong-compiler")
    assert len(calls) == 2 and not (tmp_path / "wrong-compiler/lowering-receipt.json").exists()
    source = tmp_path / "leaf.py"
    source.write_bytes(source.read_bytes() + b"# changed after admission\n")
    with pytest.raises(ValueError, match="source snapshot differs"):
        check_loop_sources(loop)

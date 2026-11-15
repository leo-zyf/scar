"""Lower one verified execution recipe through the existing loop backend.

The recipe is the common input to planning/cost and generation. This adapter
does not choose a variant, change its DAG/layout, or authorize deployment.
The unchanged native manifest remains in a separately loadable subdirectory.
"""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path


def _raw_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class RecipeArtifact:
    native: object
    directory: Path
    _recipe: dict
    _receipt: dict
    _receipt_sha256: str

    def __post_init__(self):
        # Keep the native artifact's measured call boundary: no extra Python
        # execution wrapper is inserted around each numerical invocation.
        self.run = self.native.run

    @property
    def native_directory(self): return self.native.directory

    @property
    def loop(self): return self.native.loop

    @property
    def strategy(self): return self.native.strategy

    @property
    def execution_plan(self): return self.native.execution_plan

    @property
    def layout(self): return self.native.layout

    @property
    def recipe(self): return deepcopy(self._recipe)

    @property
    def receipt(self): return deepcopy(self._receipt)

    @property
    def report(self):
        return {**self.native.report,
            "recipe_id": self._receipt["recipe_id"],
            "execution_recipe_id": self._receipt["recipe_id"],
            "recipe_sha256": self._receipt["recipe_sha256"],
            "backend_capability_record": deepcopy(self._receipt["backend_capability_record"]),
            "lowering_receipt_sha256": self._receipt_sha256,
            "recipe_directory": str(self.directory), "native_directory": str(self.native_directory),
            "recipe_path": str(self.directory / "recipe.json"),
            "lowering_receipt_path": str(self.directory / "lowering-receipt.json")}

def build_recipe(recipe, out):
    """Replay a supported recipe and compile only its exact selected kernel."""
    from scar.ir.execution_recipe import read_recipe, validate_recipe, recipe_id, write_recipe
    from scar.optimization.loop_recipe import verify_loop_recipe, unpack_loop_recipe
    from scar.optimization.resource_solver import semantic_sha256
    from scar.compiler.loop_model import loop_digest
    from scar.compiler.model import document
    from . import loop as backend

    recipe = read_recipe(recipe) if isinstance(recipe, (str, Path)) else deepcopy(recipe)
    recipe = validate_recipe(recipe, verify_artifacts=True)
    recipe = verify_loop_recipe(recipe)
    loop, strategy, plan, layout = unpack_loop_recipe(recipe)
    identity = recipe_id(recipe)
    directory = Path(out).resolve()
    if directory.exists():
        raise ValueError("fresh recipe lowering output directory required")
    directory.mkdir(parents=True)
    native = backend.build(loop, directory / "native", strategy=strategy,
                           execution_plan=plan, layout=layout)
    report = native.report
    expected = dict(loop_sha256=loop_digest(loop), strategy_sha256=semantic_sha256(strategy),
                    execution_plan_sha256=semantic_sha256(document(plan)))
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("native artifact differs from the replayed execution recipe")
    if layout is not None and report.get("layout_sha256") != semantic_sha256(layout):
        raise ValueError("native layout differs from the replayed execution recipe")
    capability = recipe["implementation"]["backend_capability_record"]
    actual_capability = dict(schema="scar.loop-backend-capability", schema_version=1,
        family="CPP_SYNCHRONOUS_CPU_TEMPORAL_LOOP", cxx_standard=20,
        loop_abi_version=report["loop_abi_version"],
        operator_abi_version=native.execution_plan.operator_abi_version,
        loop_implementation_sha256=report["loop_implementation_sha256"],
        compiler=dict(argv=report["compiler"], identity=report["compiler_version"],
                      options=report["compiler_options"]), layout_sha256=report.get("layout_sha256"))
    if report.get("backend") != "cpp" or actual_capability != capability:
        raise ValueError("native implementation/compiler/ABI/layout differs from recipe capability")
    write_recipe(recipe, directory / "recipe.json")
    receipt = dict(schema="scar.execution-recipe-lowering", schema_version=1,
        recipe_id=identity, recipe_sha256=_raw_sha(directory / "recipe.json"),
        backend_capability_record=deepcopy(capability),
        profile=recipe["profile"], input_artifacts=deepcopy(recipe["artifacts"]),
        semantic_boundary_sha256=semantic_sha256(recipe["semantic_boundary"]),
        native_directory=str(native.directory),
        native_ids={key: report[key] for key in ("build_id", "loop_sha256", "strategy_sha256",
                                               "execution_plan_sha256", "contract_sha256")},
        native_manifest_sha256=_raw_sha(report["metadata_path"]),
        native_library_sha256=_raw_sha(report["library_path"]),
        compiler=dict(argv=report["compiler"], identity=report["compiler_version"],
                      options=report["compiler_options"]),
        loop_implementation_sha256=report["loop_implementation_sha256"],
        loop_abi_version=report["loop_abi_version"], layout_sha256=report.get("layout_sha256"),
        execution="EXISTING_SYNCHRONOUS_CPU_TEMPORAL_LOOP_BACKEND",
        recipe_replayed=True, target_variants_ranked=False,
        physical_optimality_proven=False, deployment_authorized=False)
    (directory / "lowering-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return RecipeArtifact(native, directory, recipe, receipt, _raw_sha(directory / "lowering-receipt.json"))


__all__ = ["RecipeArtifact", "build_recipe"]

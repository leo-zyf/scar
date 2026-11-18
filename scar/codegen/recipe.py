"""Lower one verified execution recipe through its existing typed backend.

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
    def region(self): return self.native.region

    @property
    def plan(self): return self.native.plan

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

def _build_loop_recipe(recipe, out):
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


def _build_stateful_recipe(recipe, out):
    from scar.ir.execution_recipe import recipe_id, write_recipe
    from scar.optimization.stateful_recipe import verify_stateful_recipe, unpack_stateful_recipe
    from scar.optimization.resource_solver import semantic_sha256
    from scar.compiler.stateful_model import stateful_digest
    from . import stateful as backend

    recipe = verify_stateful_recipe(recipe)
    region, plan = unpack_stateful_recipe(recipe)
    directory = Path(out).resolve()
    if directory.exists(): raise ValueError("fresh recipe lowering output directory required")
    directory.mkdir(parents=True)
    native = backend.build(region, directory / "native", plan=plan)
    report = native.report
    expected = dict(region_sha256=stateful_digest(region), plan_sha256=semantic_sha256(plan))
    if (any(report.get(key) != value for key, value in expected.items())
            or stateful_digest(native.region) != expected['region_sha256']
            or semantic_sha256(native.plan) != expected['plan_sha256']):
        raise ValueError("native stateful artifact differs from replayed execution recipe")
    capability = recipe['implementation']['backend_capability_record']
    actual = dict(schema='scar.stateful-backend-capability', schema_version=1,
        family='CPP_SYNCHRONOUS_CPU_TRANSACTIONAL_STATEFUL_REGION', cxx_standard=20,
        stateful_abi_version=report['stateful_abi_version'], operator_abi_version=1,
        implementation_sha256=report['implementation_sha256'], compiler=report['compiler'],
        provider_runtime=report['provider_runtime'])
    if actual != capability:
        raise ValueError("native implementation/compiler/ABI/provider differs from recipe capability")
    if (report.get('cpu_ids') != recipe['resources']['selected_cpu']['cpu_ids']
            or report.get('cpu_threads') != plan['cpu_threads']
            or any(report.get(key) != plan[key] for key in ('memory_budget_bytes',
                'workspace_payload_bytes', 'external_array_bytes', 'total_region_array_payload_bytes'))):
        raise ValueError("native CPU or payload contract differs from recipe")
    write_recipe(recipe, directory / 'recipe.json')
    receipt = dict(schema='scar.execution-recipe-lowering', schema_version=1,
        recipe_id=recipe_id(recipe), recipe_sha256=_raw_sha(directory / 'recipe.json'),
        profile=recipe['profile'], backend_capability_record=deepcopy(capability),
        input_artifacts=deepcopy(recipe['artifacts']),
        semantic_boundary_sha256=semantic_sha256(recipe['semantic_boundary']),
        task_graph_sha256=semantic_sha256(recipe['task_graph']), native_directory=str(native.directory),
        native_ids={key:report[key] for key in ('build_id','region_sha256','plan_sha256','contract_sha256')},
        native_manifest_sha256=_raw_sha(native.directory / 'artifact.json'),
        native_library_sha256=_raw_sha(native.directory / report['files']['library']),
        compiler=deepcopy(report['compiler']), implementation_sha256=report['implementation_sha256'],
        stateful_abi_version=report['stateful_abi_version'],
        execution='EXISTING_TRANSACTIONAL_CPU_STATEFUL_BACKEND',
        recipe_replayed=True, target_variants_ranked=False,
        makespan_optimality_proven=False, physical_optimality_proven=False, deployment_authorized=False)
    path = directory / 'lowering-receipt.json'
    path.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return RecipeArtifact(native, directory, recipe, receipt, _raw_sha(path))


def build_recipe(recipe, out, *, source_root=None, native_dir=None, jobs=4):
    """Dispatch after envelope validation; each profile replays before building."""
    from scar.ir.execution_recipe import read_recipe, validate_recipe
    value = read_recipe(recipe) if isinstance(recipe, (str, Path)) else validate_recipe(recipe)
    if value['profile'] == 'cpu-sparse-symbolic-solve-v1':
        from .sparse_recipe import build_sparse_lowering

        return build_sparse_lowering(value, out, source_root=source_root, native_dir=native_dir, jobs=jobs)
    if source_root is not None or native_dir is not None:
        raise ValueError('native provider inputs require the sparse symbolic solve profile')
    if value['profile'] == 'cpu-ordered-sparse-solve-v1':
        from .ordered_sparse_recipe import build_ordered_sparse_lowering

        return build_ordered_sparse_lowering(value, out)
    if value['profile'] == 'cpu-temporal-loop-v1': return _build_loop_recipe(value, out)
    if value['profile'] == 'cpu-stateful-array-region-v1': return _build_stateful_recipe(value, out)
    raise ValueError('unsupported execution recipe lowering profile')


__all__ = ["RecipeArtifact", "build_recipe"]

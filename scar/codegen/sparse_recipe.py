"""Bind the canonical sparse solve recipe to its existing scoped C++ backend."""

import json
import os
from pathlib import Path
import subprocess

from scar.ir.execution_recipe import SCHEMA as RECIPE
from scar.ir.execution_recipe import artifact_reference, read_artifact, write_recipe
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.sparse_solve_recipe import verify_sparse_solve_recipe

from . import sparse_lu as backend


SCHEMA = 'scar.execution-recipe-lowering'
_CREATE = 'scar_superlu_symbolic_v1'
_SOLVE = 'scar_superlu_symbolic_solve_v1'
_DESTROY = 'scar_superlu_symbolic_destroy_v1'
_ONCE = 'ONCE_PER_SCOPE'
_PER_CALL = 'PER_CALL'
_TASKS = {
    'admit-boundary': ('COMPUTE', _ONCE, 'scar.runtime.sparse_solve_entry.admit'),
    'load-native': ('COMPUTE', _ONCE, 'scar.runtime.sparse_lu.load_symbolic_solve'),
    'allocate-symbolic': ('ALLOCATION', _ONCE, _CREATE),
    'column-order': ('COMPUTE', _ONCE, _CREATE),
    'preorder': ('COMPUTE', _ONCE, _CREATE),
    'guard-structure': ('COMPUTE', _PER_CALL, _SOLVE),
    'allocate-numeric': ('ALLOCATION', _PER_CALL, _SOLVE),
    'bind-current-columns': ('COMPUTE', _PER_CALL, _SOLVE),
    'numeric-factorization': ('COMPUTE', _PER_CALL, _SOLVE),
    'triangular-solve': ('COMPUTE', _PER_CALL, _SOLVE),
    'release-numeric': ('RELEASE', _PER_CALL, _SOLVE),
    'validate-first-result': ('COMPUTE', 'ON_FIRST_SUCCESS', 'scar.runtime.sparse_lu._Solver._run'),
    'publish-result': ('EXPORT', _PER_CALL, 'ORIGINAL_SOURCE_STATUS_BRANCH'),
    'release-symbolic': ('RELEASE', 'PER_SCOPE_EXIT', _DESTROY),
}
_ABI_SYMBOLS = frozenset((_CREATE, _SOLVE, _DESTROY, 'scar_superlu_abi_v1',
    'scar_superlu_reference_v1', 'scar_superlu_symbolic_bytes_v1'))


def _carrier(recipe):
    scope_ref = recipe['artifacts']['source_scope']
    scope = read_artifact(scope_ref)
    call = scope['source_call']
    carrier = call['carrier']['library']
    evidence = read_artifact(scope['elf_evidence'])
    if (evidence['library'] != carrier
            or evidence['documentation']['matched_tag'] != f'scipy-v{backend.PROVIDER_VERSION}'):
        raise ValueError('sparse recipe original carrier differs from the supported provider build')
    return carrier, call


def _task_bindings(recipe, call):
    graph = recipe['task_graph']
    tasks = graph['tasks']
    if (graph['schema_version'] != 1 or len(tasks) != len(_TASKS)
            or {task['id'] for task in tasks} != set(_TASKS)):
        raise ValueError('the sparse backend requires its exact scoped task set')
    runtime = Path(__file__).resolve().parents[1] / 'runtime'
    python = {
        'admit-boundary': artifact_reference(runtime / 'sparse_solve_entry.py', 'scar.python-source'),
        'load-native': artifact_reference(runtime / 'sparse_lu.py', 'scar.python-source'),
    }
    python['validate-first-result'] = python['load-native']
    abi_binding = artifact_reference(runtime / 'sparse_lu_native.py', 'scar.python-source')
    bindings = []
    for task in tasks:
        name = task['id']
        kind, lifetime, entrypoint = _TASKS[name]
        if (task['kind'], task['attrs'].get('lifetime'), task['attrs'].get('entrypoint')) != (kind, lifetime, entrypoint):
            raise ValueError('unsupported sparse task kind, lifetime or entry point: ' + name)
        if entrypoint in _ABI_SYMBOLS:
            implementation = dict(kind='CXX20_ABI', symbol=entrypoint)
        elif name in python:
            implementation = dict(kind='SCAR_PYTHON', callable=entrypoint, source=python[name])
            if name == 'load-native':
                implementation['abi_binding'] = abi_binding
        else:
            implementation = dict(kind='ORIGINAL_PYTHON', source=call['source'], code_key=call['code_key'],
                call_span=call['source_span'], original_boundary=call['original_python_boundary'])
        bindings.append(dict(task_id=name, kind=kind, lifetime=lifetime, implementation=implementation))
    return bindings


def _validated_bundle(recipe, native, carrier):
    bundle = backend.validate_symbolic_solve(native, carrier)
    capability = recipe['implementation']['backend_capability_record']
    if (bundle['blas'] != recipe['artifacts']['blas'] or capability['blas'] != bundle['blas']
            or capability['original_carrier'] != carrier
            or capability['abi_version'] != bundle['abi_version']
            or capability['provider_version'] != bundle['provider_version']
            or set(capability['symbols'].values()) != _ABI_SYMBOLS):
        raise ValueError('native sparse capability or BLAS artifact differs from the recipe')
    symbols = subprocess.check_output(['readelf', '--dyn-syms', '--wide', bundle['library']['uri']], text=True,
        env={**os.environ, 'LC_ALL': 'C'})
    exported = {row[7] for row in map(str.split, symbols.splitlines())
                if len(row) >= 8 and row[3:6] == ['FUNC', 'GLOBAL', 'DEFAULT'] and row[6] != 'UND'}
    if not _ABI_SYMBOLS <= exported:
        raise ValueError('native sparse library lacks the scoped symbolic ABI')
    return bundle


def _receipt(recipe, recipe_ref, bindings, native, bundle):
    return dict(schema=SCHEMA, schema_version=1, profile=recipe['profile'],
        recipe=recipe_ref, recipe_id=recipe['recipe_id'], recipe_sha256=recipe_ref['sha256'],
        semantic_boundary_sha256=semantic_sha256(recipe['semantic_boundary']),
        control_sha256=semantic_sha256(recipe['control']),
        task_graph_sha256=semantic_sha256(recipe['task_graph']),
        resources_sha256=semantic_sha256(recipe['resources']),
        input_artifacts=recipe['artifacts'],
        backend_capability_record=recipe['implementation']['backend_capability_record'],
        task_bindings=bindings, native=native, native_directory=str(Path(native['uri']).parent),
        native_manifest_sha256=native['sha256'], native_library=bundle['library'],
        native_library_sha256=bundle['library']['sha256'], native_source=bundle['source'],
        native_template=bundle['template'], compiler=bundle['compiler'],
        native_abi_version=bundle['abi_version'],
        execution='EXISTING_SCOPED_SPARSE_SYMBOLIC_BACKEND',
        recipe_replayed=True, generic_graph_executor=False, target_variants_ranked=False,
        makespan_optimality_proven=False, physical_optimality_proven=False, deployment_authorized=False)


def build_sparse_lowering(value, out, *, source_root=None, native_dir=None, jobs=4):
    """Compile, or attest an existing bundle, only after replaying the recipe."""
    if (source_root is None) == (native_dir is None):
        raise ValueError('sparse lowering requires either a SuperLU source root or one existing native bundle')
    recipe = verify_sparse_solve_recipe(value)
    carrier, call = _carrier(recipe)
    bindings = _task_bindings(recipe, call)
    directory = Path(out).resolve()
    receipt_path = directory / 'lowering-receipt.json'
    if receipt_path.exists():
        raise ValueError('fresh sparse lowering receipt required')
    path = directory / 'recipe.json'
    if path.exists():
        recipe_ref = artifact_reference(path, RECIPE)
        if read_artifact(recipe_ref) != recipe:
            raise ValueError('output directory already contains a different execution recipe')
    else:
        recipe_ref = write_recipe(recipe, path)
    if source_root is not None:
        native_dir = directory / 'native'
        backend.build_symbolic_solve(source_root, native_dir, blas=recipe['artifacts']['blas']['uri'],
            jobs=jobs, carrier=carrier)
    native = artifact_reference(Path(native_dir).resolve() / 'artifact.json', backend.SCHEMA)
    bundle = _validated_bundle(recipe, native, carrier)
    receipt = _receipt(recipe, recipe_ref, bindings, native, bundle)
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(receipt_path, SCHEMA)


def verify_sparse_lowering(recipe_ref, receipt_ref, native_ref):
    """Replay and compare the lowering receipt without building or running the target."""
    path = Path(recipe_ref['uri'])
    if (recipe_ref['schema'] != RECIPE or receipt_ref['schema'] != SCHEMA
            or native_ref['schema'] != backend.SCHEMA or path.name != 'recipe.json'
            or receipt_ref['uri'] != str(path.parent / 'lowering-receipt.json')):
        raise ValueError('matching sparse recipe, lowering and native artifact references required')
    receipt = read_artifact(receipt_ref)
    recipe = verify_sparse_solve_recipe(read_artifact(recipe_ref))
    carrier, call = _carrier(recipe)
    bindings = _task_bindings(recipe, call)
    bundle = _validated_bundle(recipe, native_ref, carrier)
    if receipt != _receipt(recipe, recipe_ref, bindings, native_ref, bundle):
        raise ValueError('sparse lowering receipt no longer matches the recipe, tasks or native bytes')
    return receipt

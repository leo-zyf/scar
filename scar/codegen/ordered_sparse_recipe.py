"""Lower the ordered sparse recipe into the existing process runtime."""

from contextlib import nullcontext
import json
from pathlib import Path

from scar.analysis.replay_transaction import current_transaction, replay_transaction
from scar.ir.execution_recipe import SCHEMA as RECIPE
from scar.ir.execution_recipe import artifact_reference, read_artifact, write_recipe
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.sparse_parallel_recipe import verify_sparse_parallel_recipe

from .sparse_recipe import SCHEMA, verify_sparse_lowering


_ENTRY = 'scar.runtime.sparse_parallel_entry'
_POOL = 'scar.runtime.sparse_parallel_lu'
_PREDICTION = 'scar.runtime.sparse_solve_prediction'
_OFFER = _ENTRY + '._OrderedOffers.'
_TEAM = _POOL + '._ParallelSolver.'
_DETACHED = _PREDICTION + '.DetachedCSR.'
_CREATE = 'scar_superlu_symbolic_v1'
_APPLY = 'scar_superlu_symbolic_solve_v1'
_DESTROY = 'scar_superlu_symbolic_destroy_v1'
_PREPARE_ORIGINAL = 'ORIGINAL_SOURCE_ROW_EDIT_AND_NORMALIZATION'
_PUBLISH_ORIGINAL = 'ORIGINAL_SOURCE_STATUS_BRANCH'
_ONCE = 'ONCE_PER_SCOPE'
_CALL = 'PER_CALL'
_EXIT = 'PER_SCOPE_EXIT'
_TASKS = {
    'admit-entry': ('COMPUTE', _ONCE, _ENTRY + '._admit'),
    'allocate-live-ins': ('ALLOCATION', _ONCE, _DETACHED + '__init__'),
    'capture-live-ins': ('COMPUTE', _ONCE, _OFFER + 'capture'),
    'allocate-team': ('ALLOCATION', _ONCE, _TEAM + 'prepare'),
    'bootstrap-team': ('COMPUTE', _ONCE, _TEAM + 'prepare'),
    'allocate-original': ('ALLOCATION', _CALL, _PREPARE_ORIGINAL),
    'prepare-original': ('COMPUTE', _CALL, _PREPARE_ORIGINAL),
    'allocate-call': ('ALLOCATION', _CALL, _TEAM + 'submit'),
    'prepare-private': ('COMPUTE', _CALL, _DETACHED + 'args'),
    'submit': ('COMPUTE', _CALL, _TEAM + 'submit'),
    'deliver-request': ('COMPUTE', _CALL, _POOL + '._Worker.exchange'),
    'solve-worker': ('COMPUTE', _CALL, _APPLY),
    'deliver-result': ('COMPUTE', _CALL, _TEAM + '_execute'),
    'dequeue-oldest': ('COMPUTE', _CALL, _TEAM + '_retire'),
    'compare-request': ('COMPUTE', _CALL, _TEAM + 'consume'),
    'retire-matched': ('COMPUTE', _CALL, _OFFER + '__call__'),
    'release-call': ('RELEASE', _CALL, _TEAM + 'consume'),
    'publish-original': ('EXPORT', _CALL, _PUBLISH_ORIGINAL),
    'close-team': ('RELEASE', _EXIT, _OFFER + 'close'),
}


def _runtime_policy(recipe):
    resources = recipe['resources']
    cpu, dispatch, memory, case = (resources[role] for role in ('selected_cpu', 'dispatch', 'memory', 'case'))
    return dict(launch_cpu_ids=cpu['launch_cpu_ids'], owner_cpu=cpu['owner_cpu_id'],
        numa_node=cpu['cpu_numa_node'], count=case['call_count'], matrix=dict(size=case['size'], nnz=case['nnz']),
        workers=dict(cpu_ids=cpu['worker_cpu_ids'], window=dispatch['window'],
            address_space_limit_bytes=memory['worker_address_space_limit_bytes']),
        symbolic_budget_bytes=memory['per_worker_symbolic_budget_bytes'],
        memory_floor_bytes=memory['minimum_launch_headroom_bytes'])


def _runtime_members():
    runtime = Path(__file__).resolve().parents[1] / 'runtime'
    names = ('sparse_parallel_entry', 'sparse_parallel_lu', 'sparse_solve_prediction', 'sparse_lu_native')
    return {name: artifact_reference(runtime / (name + '.py'), 'scar.python-source') for name in names}


def _task_bindings(recipe, parent, members):
    graph = recipe['task_graph']
    tasks = graph['tasks']
    if (graph['schema_version'] != 1 or len(tasks) != len(_TASKS)
            or {task['id'] for task in tasks} != set(_TASKS)):
        raise ValueError('ordered sparse lowering requires its exact scoped task set')
    scope = read_artifact(recipe['artifacts']['parallel_scope'])
    status = next(item['implementation'] for item in parent['task_bindings'] if item['task_id'] == 'publish-result')
    prepare = dict(kind='ORIGINAL_PYTHON', source=scope['entry']['source'],
        code_key=scope['source_body']['code_key'], row_edit_span=scope['preparation']['row_edit_call_span'],
        solver_source=status['source'], solver_code_key=status['code_key'], solver_call_span=status['call_span'])
    child = dict(callable=_POOL + '._child', source=members['sparse_parallel_lu'],
        abi_binding=members['sparse_lu_native'])

    bindings = []
    for task in tasks:
        name = task['id']
        kind, lifetime, entrypoint = _TASKS[name]
        if (task['kind'], task['attrs'].get('lifetime'), task['attrs'].get('entrypoint')) != (kind, lifetime, entrypoint):
            raise ValueError('unsupported ordered sparse task kind, lifetime or entry point: ' + name)
        if entrypoint == _PREPARE_ORIGINAL:
            implementation = prepare
        elif entrypoint == _PUBLISH_ORIGINAL:
            implementation = status
        elif entrypoint == _APPLY:
            implementation = dict(kind='CXX20_ABI', symbol=_APPLY, worker=child)
        else:
            module = entrypoint.split('.')[2]
            implementation = dict(kind='SCAR_PYTHON', callable=entrypoint, source=members[module])
            if name in ('deliver-request', 'deliver-result'):
                implementation.update(kind='PROCESS_IPC', transport='AF_UNIX_STREAM_CONNECTION',
                    worker=child, direction='OWNER_TO_WORKER' if name == 'deliver-request' else 'WORKER_TO_OWNER')
            elif name in ('bootstrap-team', 'close-team'):
                implementation.update(worker=child, symbol=_CREATE if name == 'bootstrap-team' else _DESTROY)
        bindings.append(dict(task_id=name, kind=kind, lifetime=lifetime, entrypoint=entrypoint,
            implementation=implementation))
    return bindings


def _receipt(recipe, reference):
    capability = recipe['implementation']['backend_capability_record']
    serial_recipe, serial_lowering, native = (capability[role] for role in ('serial_recipe', 'serial_lowering', 'native'))
    launch = read_artifact(recipe['artifacts']['serial_launch'])
    study = read_artifact(launch['study'])
    if (launch['execution_recipe'] != serial_recipe or launch['lowering'] != serial_lowering
            or study['execution_recipe'] != serial_recipe or study['lowering'] != serial_lowering
            or study['native'] != native or capability['symbols'] != dict(create=_CREATE, apply=_APPLY, destroy=_DESTROY)):
        raise ValueError('ordered sparse lowering requires its original serial recipe, lowering and native bundle')
    # The surrounding replay transaction shares the already checked serial
    # source scope while the serial verifier rebuilds the receipt and ELF binding.
    parent = verify_sparse_lowering(serial_recipe, serial_lowering, native)
    serial = parent['backend_capability_record']
    if any(capability[key] != serial[key] for key in ('abi_version', 'cxx_standard', 'original_carrier', 'blas')):
        raise ValueError('ordered worker ABI differs from the verified native parent')
    members = _runtime_members()
    return dict(schema=SCHEMA, schema_version=1, profile=recipe['profile'],
        recipe=reference, recipe_id=recipe['recipe_id'], recipe_sha256=reference['sha256'],
        semantic_boundary_sha256=semantic_sha256(recipe['semantic_boundary']),
        control_sha256=semantic_sha256(recipe['control']), task_graph_sha256=semantic_sha256(recipe['task_graph']),
        resources_sha256=semantic_sha256(recipe['resources']), input_artifacts=recipe['artifacts'],
        backend_capability_record=capability, task_bindings=_task_bindings(recipe, parent, members),
        serial_recipe=serial_recipe, serial_lowering=serial_lowering, native=native,
        native_library=parent['native_library'], native_manifest_sha256=native['sha256'],
        native_library_sha256=parent['native_library_sha256'], native_abi_version=parent['native_abi_version'],
        detached_recipe=recipe['artifacts']['detached'], runtime_policy=_runtime_policy(recipe),
        runtime_members=list(members.values()), execution='EXISTING_ORDERED_PROCESS_SPARSE_BACKEND',
        recipe_replayed=True, generic_graph_executor=False, native_recompiled=False,
        target_variants_ranked=False, makespan_optimality_proven=False, deployment_authorized=False)


def build_ordered_sparse_lowering(value, out):
    """Publish the policy and bindings for an existing independently built worker ABI."""
    directory = Path(out).resolve()
    receipt_path = directory / 'lowering-receipt.json'
    if receipt_path.exists():
        raise ValueError('fresh ordered sparse lowering receipt required')
    with nullcontext() if current_transaction() is not None else replay_transaction():
        recipe = verify_sparse_parallel_recipe(value)
        path = directory / 'recipe.json'
        if path.exists():
            reference = artifact_reference(path, RECIPE)
            if read_artifact(reference) != recipe:
                raise ValueError('output directory already contains a different execution recipe')
        else:
            reference = write_recipe(recipe, path)
        receipt = _receipt(recipe, reference)
        with receipt_path.open('x') as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    return artifact_reference(receipt_path, SCHEMA)


def verify_ordered_sparse_lowering(recipe_ref, receipt_ref):
    """Rebuild the full receipt from an exact recipe, including a pinned local copy."""
    path = Path(receipt_ref['uri'])
    if recipe_ref['schema'] != RECIPE or receipt_ref['schema'] != SCHEMA or path.name != 'lowering-receipt.json':
        raise ValueError('matching ordered sparse recipe and lowering references required')
    with nullcontext() if current_transaction() is not None else replay_transaction():
        receipt = read_artifact(receipt_ref)
        retained = receipt['recipe']
        if retained['schema'] != RECIPE or retained['uri'] != str(path.with_name('recipe.json')):
            raise ValueError('ordered sparse output does not identify its frozen recipe')
        raw = read_artifact(recipe_ref)
        if retained != recipe_ref and read_artifact(retained) != raw:
            raise ValueError('ordered sparse lowering belongs to a different execution recipe')
        recipe = verify_sparse_parallel_recipe(raw)
        if receipt != _receipt(recipe, retained):
            raise ValueError('ordered sparse lowering no longer matches its recipe, policy, tasks or native bytes')
    return receipt

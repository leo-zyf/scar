"""Discover and pin the CPU realization of a source-normalized matrix entry."""

from pathlib import Path

from scar.analysis.nvhpc_source_entry import build_nvhpc_source_recipe, derive_nvhpc_source_entry
from scar.codegen.native_provider_bundle import _content
from scar.compiler.compile_exported_nvhpc import _estimate_body, _plan
from scar.ir.execution_recipe import read_artifact
from scar.optimization.resource_solver import semantic_sha256


FAMILY = 'NVIDIA_CPU_MATRIX_FILE_ENTRY'
SCHEMA = 'scar.nvhpc-source-variants'


def matrix_entry_inputs(contract, argv):
    """Bind the normalized ordered input ports and require fresh output coordinates."""
    paths = list(map(Path, argv))
    if len(paths) != contract['argv_count'] or any(not path.is_absolute() for path in paths):
        raise ValueError('NVHPC file ABI needs the exact number of absolute arguments')
    records = []
    for path, port in zip(paths[:len(contract['inputs'])], contract['inputs'], strict=True):
        if not path.is_file() or path.resolve(strict=True) != path or path.stat().st_size != port['nbytes']:
            raise ValueError('NVHPC input is outside the stable file size and identity domain')
        record = _content(path)
        if record['size_bytes'] != port['nbytes']:
            raise ValueError('matrix input changed while binding the original entry')
        records.append(dict(uri=str(path), schema='scar.source-entry-input', **record))
    outputs = paths[len(records):]
    if (len(set(outputs)) != len(outputs) or any(not path.parent.is_dir() or path.resolve() != path
            or path.exists() or path.is_symlink() for path in outputs)):
        raise ValueError('NVHPC file outputs must be fresh, distinct and canonical')
    return records


def derive_nvhpc_entries(scope, observations):
    """Enumerate before compilation and predict the fixed CPU body where calibrated."""
    result = dict(schema=SCHEMA, schema_version=1, family=FAMILY, scope_sha256=semantic_sha256(scope),
        options=[], coverage_frontier=[], compiled_source_entry=False, target_executed=False)
    if 'entry_obligations_sha256' not in scope:
        result['coverage_frontier'].append(dict(reason='ORIGINAL_F4_ENTRY_OBLIGATIONS_REQUIRED'))
        return result
    from scar.compiler.operator_stack import load_operator_stack

    target = read_artifact(scope['stack_target'])
    if not target['cpu_blas_installed']:
        result['coverage_frontier'].append(dict(reason='PINNED_CPU_BLAS_UNAVAILABLE'))
        return result
    stack_ref, stack = load_operator_stack(target['stack'])
    try:
        recipe = build_nvhpc_source_recipe(scope['entry'])
        contract = derive_nvhpc_source_entry(scope['entry'], recipe)
        if contract['source']['sha256'] != scope['entry_sha256']:
            raise ValueError('source bytes differ from original F4 entry')
        files = matrix_entry_inputs(contract, scope['argv'])
        cpus = scope['process_affinity_cpu_ids']
        memory = scope['memory_budget_bytes']
        problem, storage = _plan(recipe, contract['projection'], contract['program'],
            Path(scope['target']['uri']).parent, memory, cpus, len(cpus), observations=observations, stack=stack)
    except (OSError, SyntaxError, ValueError) as error:
        result['coverage_frontier'].append(dict(reason='SOURCE_CPU_MATRIX_PROFILE_OPEN', detail=str(error)))
        return result
    plan = dict(source_contract=contract, operator_stack=stack_ref, problem=problem, storage=storage,
        cpu_ids=cpus, threads=len(cpus), sgemm_observations=observations, files=files,
        numerical_contract=scope.get('numerical_contract', {'kind': 'EXACT_BYTES'}))
    if observations:
        plan['body_forecast'] = _estimate_body(problem)
    identity = dict(scope_sha256=result['scope_sha256'], plan=plan)
    result['options'].append(dict(id='planned:'+semantic_sha256(identity), legal_status='UNRESOLVED',
        reason='ORIGINAL_IMPORT_INITIALIZATION_EFFECTS_OPEN', implementation_family=FAMILY,
        proof=dict(emission_plan=plan)))
    return result

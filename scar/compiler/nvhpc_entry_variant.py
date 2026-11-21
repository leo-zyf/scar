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


def _cpu_cohorts(hardware, process_cpus):
    from scar.analysis.target_graph import TargetHardwareGraph
    from scar.ir.program.target_hardware import CpuCore, HardwareThread
    from scar.optimization.aten_native_schedule import recorded_cpu_domain

    graph = TargetHardwareGraph(hardware)
    _, eligible = recorded_cpu_domain(graph)
    admitted = set(process_cpus) & set(eligible)
    resources = graph.resources()
    online = {r.cpu_id for r in resources if type(r) is HardwareThread and r.online is True}
    leaders = sorted(min(ids) for core in resources if type(core) is CpuCore
                     if (ids := set(core.cpu_ids) & admitted & online))
    if not leaders:
        raise ValueError('original affinity contains no admitted online physical core')
    return [leaders[:count] for count in range(1, len(leaders) + 1)]


def _body_sample_range(problem, forecast):
    med = forecast['schedule']['makespan_ns']
    if med is None:
        return None
    lower, upper = 0, 0
    samples = {}
    for op in problem['operations']:
        option, = op['options']
        if option['lowering'] == 'REGION_VALUE_HANDOFF':
            continue
        cost = option['cost']
        reference, = cost['source_refs']
        key = (reference['artifact']['uri'], reference['locator'])
        if key not in samples:
            samples[key] = read_artifact(reference['artifact'])['providers'][int(reference['locator'].split('/')[-1])]
        provider = samples[key]
        lower += provider['lower_sample_ns']
        upper += provider['upper_sample_ns']
    return dict(lower=lower, median=med, upper=upper)


def _trial_choice(rows):
    if len(rows) == 1:
        return 'ONLY_DECLARED_COHORT', rows[0]['option_id']
    if any(row['body_sample_range_ns'] is None for row in rows):
        return 'MISSING_MATCHED_BODY_SAMPLES', None
    best = min(rows, key=lambda row: row['body_sample_range_ns']['upper'])
    if any(best['body_sample_range_ns']['upper'] >= row['body_sample_range_ns']['lower']
           for row in rows if row is not best):
        return 'BODY_SAMPLE_RANGES_OVERLAP', None
    return 'SEPARATED_WARM_BODY_SAMPLES', best['option_id']


def derive_nvhpc_entries(scope, observations):
    """Enumerate before compilation and predict the serial CPU body for each team."""
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
        hardware = Path(scope['target']['uri']).parent
        process_cpus = scope['process_affinity_cpu_ids']
        search = scope.get('search_cpu_cohorts', False)
        if search:
            cohorts = _cpu_cohorts(hardware, process_cpus)
            evidence = {tuple(cpus): [] for cpus in cohorts}
            for ref in observations:
                mask = tuple(read_artifact(ref)['cpu_ids'])
                if mask not in evidence:
                    raise ValueError('SGEMM observations do not belong to the declared physical-core prefix domain')
                evidence[mask].append(ref)
        else:
            cohorts = [process_cpus]
            evidence = {tuple(process_cpus): observations}
        plans, rows = [], []
        for cpus in cohorts:
            samples = evidence[tuple(cpus)]
            problem, storage = _plan(recipe, contract['projection'], contract['program'],
                hardware, scope['memory_budget_bytes'], cpus, len(cpus), observations=samples, stack=stack)
            plan = dict(source_contract=contract, operator_stack=stack_ref, problem=problem, storage=storage,
                cpu_ids=cpus, threads=len(cpus), sgemm_observations=samples, files=files,
                numerical_contract=scope.get('numerical_contract', {'kind': 'EXACT_BYTES'}))
            if samples:
                plan['body_forecast'] = _estimate_body(problem)
            option_id = 'planned:'+semantic_sha256(dict(scope_sha256=result['scope_sha256'], plan=plan))
            plans.append(dict(id=option_id, legal_status='UNRESOLVED',
                reason='ORIGINAL_IMPORT_INITIALIZATION_EFFECTS_OPEN', implementation_family=FAMILY,
                proof=dict(emission_plan=plan)))
            if search:
                measured = _body_sample_range(problem, plan['body_forecast']) if samples else None
                rows.append(dict(option_id=option_id, cpu_ids=cpus, body_sample_range_ns=measured))
    except (OSError, SyntaxError, ValueError) as error:
        result['coverage_frontier'].append(dict(reason='SOURCE_CPU_MATRIX_PROFILE_OPEN', detail=str(error)))
        return result
    result['options'] = plans
    if search:
        status, trial = _trial_choice(rows)
        result['cohort_search'] = dict(policy='ONE_ONLINE_THREAD_PER_CORE_SORTED_CPU_PREFIXES',
            original_cpu_ids=process_cpus, rows=rows, trial_option_id=trial, status=status,
            basis='SUMS_OF_MATCHED_SERIAL_WARM_SGEMM_SAMPLE_EXTREMA_AND_MEDIANS',
            whole_process_cost_ranked=False, physical_bounds_proven=False)
    return result

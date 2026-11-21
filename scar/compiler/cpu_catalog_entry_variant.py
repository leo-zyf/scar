"""Match sealed CPU provider evidence to an original F4 invocation before build."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from subprocess import CalledProcessError

from scar.compiler.compile_cpu_entry import CATALOG_SCHEMA, _PROVIDERS, _match, _read_catalog, _reference
from scar.compiler.plan_aten_boundary import _target
from scar.ir.execution_recipe import read_artifact
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.cpu-catalog-entry-variants'
FAMILY = 'CATALOG_CPU_ENTRY'
_ARRAY = 'scar.cpu-array-task-source-provider'
_GRID = 'scar.cpu-fixed-grid-source-provider'
_SPARSE = 'scar.triangle-profile-family'
_STREAM = 'scar.cpu-recursive-stream-source-provider'


class _Frontier(ValueError):
    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason = reason


def _require(condition, reason, detail):
    if not condition:
        raise _Frontier(reason, detail)


def _launch(scope, *, interpreter, cpu_ids, cwd, original_environment):
    _require(bool(cpu_ids) and set(cpu_ids) <= set(scope['process_affinity_cpu_ids']), 'CPU_COHORT_OUTSIDE_F4_SCOPE',
             'provider CPU lanes are outside the original process affinity')
    try:
        target, lanes = _target(Path(scope['target']['uri']).parent, cpu_ids, allow_cpu_subset=True)
    except ValueError as error:
        raise _Frontier('CPU_HARDWARE_COHORT_OPEN', str(error)) from error
    _require(asdict(target) == scope['target'], 'CPU_HARDWARE_SCOPE_DIFFERS',
             'original F4 hardware graph differs from provider admission')
    environment_sha256 = sha256(json.dumps(sorted(original_environment.items()), separators=(',', ':')).encode()).hexdigest()
    matches = dict(cwd=scope['working_directory'] == cwd, affinity=scope['process_affinity_cpu_ids'] == cpu_ids,
        resolved_interpreter=Path(interpreter['uri']).resolve(strict=True) == Path(scope['python_executable']),
        environment=scope['environment_sha256'] == environment_sha256)
    return dict(interpreter=interpreter, cpu_ids=cpu_ids, cwd=cwd, original_environment=original_environment,
        scope_matches=matches, hardware_lanes={str(cpu): [row.id for row in lanes[cpu]] for cpu in cpu_ids})


def _array_memory(scope, decision):
    memory = decision['memory']
    scope_kind = memory['bound_scope']
    owner = memory['explicit_owner_payload_bytes']
    worker = memory['explicit_worker_payload_bytes']
    owner_cap = memory['coordinator_address_space_limit_bytes']
    worker_cap = memory['worker_address_space_limit_bytes']
    workers = decision['worker_count']
    if scope_kind == 'SINGLE_PROCESS_VAS_CAP_INCLUDING_PRIVATE_ARRAYS_AND_OPENMP_CHILD_STACKS':
        processes = 0
    elif scope_kind == 'AGGREGATE_PROCESS_VAS_CAPS_PLUS_EXPLICIT_PAYLOAD_SUBACCOUNT':
        processes = workers
    else:
        raise _Frontier('ARRAY_TASK_ADDRESS_SPACE_CONTRACT_OPEN', 'unsupported native array address-space accounting')
    budget = scope['memory_budget_bytes']
    required = owner_cap + processes * worker_cap
    payload = owner + processes * worker
    _require(all(type(value) is int and value >= 0 for value in (owner, worker, processes))
             and all(type(value) is int and value > 0 for value in (owner_cap, worker_cap, budget))
             and owner < owner_cap and (not processes or worker < worker_cap)
             and required == memory['required_headroom_bytes'] and memory['physical_reservation'] is False,
             'ARRAY_TASK_ADDRESS_SPACE_CONTRACT_DIFFERS', 'retained explicit payloads and process address-space caps differ')
    _require(payload <= budget and required <= budget, 'ARRAY_TASK_ADDRESS_SPACE_BUDGET_EXCEEDED',
             'retained array-task virtual address-space requirement exceeds the requested execution budget')
    return dict(scope=scope_kind, declared_budget_bytes=budget, required_address_space_headroom_bytes=required,
        explicit_payload_bytes=payload, capacity_kind='VIRTUAL_ADDRESS_SPACE_CAPS', physical_dram_reserved=False,
        memory=memory)


def _array(scope, authority, entry, argv):
    from scar.compiler.array_task_source_provider import _read, _result

    _, profiles = _read(authority)
    _require(sum(row['argv'] == argv for row in profiles) == 1, 'ARRAY_TASK_ARGV_NOT_QUALIFIED',
             'array provider requires an exact retained original argv')
    profile = _result(authority, entry, argv)
    invocation = profile['invocation']
    context = _launch(scope, interpreter=invocation['original_executable'], cpu_ids=invocation['cpu_ids'],
                      cwd=invocation['cwd'], original_environment=invocation['original_environment'])
    retained = read_artifact(profile['retention'])
    hardware = read_artifact(retained['schedule'])['hardware']
    historical = {row['cpu_id']: row for row in read_artifact(hardware)['processors']}
    for cpu in invocation['cpu_ids']:
        row = historical.get(cpu)
        _require(row is not None and row.get('permitted') is True and
                 'target_resource:cpu-core/'+'/'.join(row[key] for key in ('package_id', 'die_id', 'core_id'))
                 in context['hardware_lanes'][str(cpu)], 'ARRAY_HOST_TOPOLOGY_DIFFERS',
                 'retained array-task placement differs from the current F4 hardware lanes')
    runtime = read_artifact(profile['runtime'])
    holdout = read_artifact(profile['holdout'])
    return context, dict(profile=profile, parameters=holdout['parameters'], legacy_hardware=hardware,
        native_memory=_array_memory(scope, runtime['decision']), publication=dict(output_index=invocation['output_index'],
            normalized_fields=profile['normalization'])), [dict(reason='RETAINED_PROCESS_SAMPLES_ARE_OBSERVATIONS',
            artifact=profile['retention'], locator='#/retained_vs_original')]


def _fixed_grid(scope, authority, entry, argv):
    from scar.compiler.fixed_grid_source_provider import _argv, _ENV, check

    provider = check(authority['uri'])
    try:
        _argv(provider, argv, require_fresh=True)
    except (OSError, ValueError) as error:
        raise _Frontier('FIXED_GRID_ARGV_NOT_QUALIFIED', str(error)) from error
    selected = provider['source_project']
    manifest = read_artifact(selected)
    root = Path(selected['uri']).parent

    def member(name):
        row = manifest['files'][name]
        return json.loads(read_artifact(dict(uri=str(root / name), schema='application/json', **row), json_document=False))

    execution, rhs = member('plan/execution.json'), member('plan/rhs-recipe.json')
    _require(execution['target'] == scope['target'], 'FIXED_GRID_TARGET_DIFFERS',
             'fixed-grid provider retained a different hardware target')
    shapes = [row['shape'] for row in rhs['inputs']]
    common = scope.get('input_shapes', {}).get('common_shape')
    _require(common is None or all(shape == common for shape in shapes), 'FIXED_GRID_SHAPE_DIFFERS',
             'fixed-grid matrix inputs differ from the declared F4 scope')
    _require(execution['peak_tensor_storage_bytes'] <= scope['memory_budget_bytes'], 'FIXED_GRID_STORAGE_BUDGET_EXCEEDED',
             'fixed-grid retained graph payload exceeds the original requested budget')
    invocation = provider['context']
    _require((Path(invocation['cwd']) / argv[2]).parent.is_dir(), 'FIXED_GRID_OUTPUT_PARENT_REQUIRED',
             'fixed-grid output requires an existing parent directory')
    environment = dict(_ENV, PYTHONPATH=':'.join(invocation['pythonpath']))
    context = _launch(scope, interpreter=invocation['interpreter'], cpu_ids=manifest['cpus'],
                      cwd=invocation['cwd'], original_environment=environment)
    step_ns = manifest['alternatives'][manifest['rhs_mode']]['body_estimate_ns']
    return context, dict(project=selected, inputs=provider['inputs'], input_shapes=shapes, solver=provider['source_binding'],
        steps=int(argv[1]), storage=execution['storage'], body_placement=execution['placement'],
        publication=dict(output_index=2, output_contract='EXACT_FINAL_F32_FILE_DISTINCT_STDOUT_CONTRACTS')), [
            dict(reason='RETAINED_FIXED_GRID_BODY_ONLY', body_step_estimate_ns=step_ns,
                 modeled_body_ns=step_ns*int(argv[1]), artifact=selected)]


def _sparse(scope, authority, entry, argv):
    from scar.compiler.compile_triangle_profile import _launch_environment
    from scar.compiler.triangle_profile_family import check

    _require(len(argv) == 1 and argv[0].isascii() and argv[0].isdecimal() and 0 < len(argv[0]) <= 9
             and int(argv[0]) > 1 and argv[0] == str(int(argv[0])), 'SPARSE_PROFILE_ARGV_OUTSIDE_CONTRACT',
             'iterative sparse provider needs one canonical iteration count greater than one')
    family = check(authority['uri'])
    launch = family['context']
    context = _launch(scope, interpreter=family['interpreter_executable'], cpu_ids=family['launch_cpu_ids'],
                      cwd=launch['cwd'], original_environment=_launch_environment(launch))
    historical = read_artifact(family['release'])
    return context, dict(anchor=family['anchor'], source_model=family['model'], count=int(argv[0]),
        historical_count=historical['count'], historical_placement=family['release'],
        historical_problem=historical['problem'], historical_memory_floor_bytes=historical['memory_floor_bytes'],
        historical_costs_transferred=False, fresh_source_acquisition_required=True), [
            dict(reason='FRESH_SPARSE_PROFILE_AND_SCHEDULING_REQUIRED')]


_PROFILE = {_ARRAY: _array, _GRID: _fixed_grid, _SPARSE: _sparse}


def derive_cpu_catalog_entry(scope, catalog_ref):
    """Derive a build candidate or retain the exact authority and the open frontier."""
    result = dict(schema=SCHEMA, schema_version=1, family=FAMILY, scope_sha256=semantic_sha256(scope),
        options=[], coverage_frontier=[], compiled_source_entry=False, target_executed=False)
    try:
        _require(all(scope.get(key) for key in ('program_sha256', 'source_admission_sha256', 'entry_obligations_sha256')),
                 'ORIGINAL_F4_ENTRY_OBLIGATIONS_REQUIRED', 'source, admission and effect obligations must be pinned')
        _require(scope.get('boundary') == 'PROCESS_LAUNCH_THROUGH_EXIT' and scope.get('native_output_contract') == 'STANDALONE_CPP',
                 'CPU_CATALOG_ENTRY_SCOPE_DIFFERS', 'standalone whole-process entry scope required')
        entry = _reference(scope['entry'], 'scar.python-source')
        _require((entry['uri'], entry['sha256']) == (scope['entry'], scope['entry_sha256']),
                 'ORIGINAL_F4_SOURCE_BYTES_DIFFER', 'original F4 source path or bytes changed')
        authorities = _read_catalog(catalog_ref)
        count = sum(document['entry'] == entry for _, document in authorities)
        _require(count == 1, 'NO_UNIQUE_CPU_CATALOG_AUTHORITY', 'the source path and bytes need one matching authority')
        authority, _ = _match(authorities, entry)
        result['matched_authority'] = authority
        kind = authority['schema']
        if kind == _STREAM:
            raise _Frontier('LIVE_STDIN_PROTOCOL_REQUIRES_INVOCATION_SCOPE',
                            'an argv-only original entry cannot bind the live command stream provider')
        argv = scope['argv']
        context, contract, costs = _PROFILE[kind](scope, authority, entry, argv)
        timing_frontier = [*costs, dict(reason='SAME_ENTRY_FORWARD_LIFECYCLE_COSTS_OPEN')]
        if not all(context['scope_matches'].values()):
            timing_frontier.append(dict(reason='PROVIDER_LAUNCH_CONTEXT_DIFFERS', matches=context['scope_matches']))
        plan = dict(catalog=catalog_ref, authority=authority, entry=entry, argv=argv, provider_family=kind,
            provider_result_schema=_PROVIDERS[kind].result_schema, context=context, contract=contract,
            target=scope['target'], timing_frontier=timing_frontier, full_original_effects_closed=False,
            source_f4_program_sha256=scope['program_sha256'], effect_obligations_sha256=scope['entry_obligations_sha256'])
        option_id = 'planned:'+semantic_sha256(dict(scope_sha256=result['scope_sha256'], plan=plan))
        result['options'].append(dict(id=option_id, implementation_family=FAMILY, legal_status='UNRESOLVED',
            reason='CATALOG_TRIAL_REQUIRES_ORIGINAL_EFFECTS_AND_CLEAN_PROCESS_VALIDATION', proof=dict(emission_plan=plan)))
    except _Frontier as error:
        result['coverage_frontier'].append(dict(reason=error.reason, detail=str(error)))
    except (ValueError, OSError, KeyError, TypeError, CalledProcessError) as error:
        result['coverage_frontier'].append(dict(reason='CPU_CATALOG_PROVIDER_REPLAY_OPEN', detail=str(error)))
    return result

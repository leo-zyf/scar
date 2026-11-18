"""Retarget a sealed ordered sparse scope without transferring cost measurements."""

from contextlib import nullcontext
from hashlib import sha256
import json

from scar.analysis.csr_topology_scope import _require
from scar.analysis.replay_transaction import current_transaction, replay_transaction
from scar.analysis.sparse_parallel_recipe import derive_recipe
from scar.analysis.sparse_parallel_scope import SCHEMA as PARALLEL, check_sparse_parallel_scope
from scar.analysis.target_inventory import _cpu_list
from scar.ir.execution_recipe import read_artifact
from scar.optimization.sparse_parallel_cost import HARDWARE
from scar.runtime.sparse_parallel_entry import SCHEMA as SELECTION


SCHEMA = 'scar.sparse-parallel-scaling-scope'
CHOICE = 'scar.sparse-parallel-preliminary-choice'


def _release_lineage(source_ref, source, release):
    _require(release['schema_version'] == 1 and release['decision'] == 'PARALLEL'
        and release['study_parallel'] is False and release['source'] == source_ref,
        'expected the published parallel selection for the exact original source scope')
    entry, context = source['entry'], release['context']
    owner = release['owner_cpu']
    _require(release['entry'] == entry['source'] and context['argv'] == entry['argv']
        and context['cpu_ids'] == entry['affinity'] == [owner]
        and all(context['environment'].get(k) == v for k,v in entry['thread_environment'].items()),
        'the ordered Python owner or original launch contract differs')
    original = read_artifact(source['source_scope'])
    count = release['count']
    _require(type(count) is int and count > 1 and original['entry'] == entry
        and count == source['loop']['exact_count_under_source_conditions']['repetitions']
        == original['observed']['consumer_completions'] == original['observed']['region_attempts'],
        'the release does not describe the same exact finite loop case')
    loop_inputs = original['observed']['loop_inputs']
    _require(release['matrix'] == dict(size=loop_inputs['matrix_shape'][0],nnz=loop_inputs['nnz']),
        'the released sparse ABI depends on another observed matrix')
    recipe = derive_recipe(read_artifact(entry['source'],json_document=False),source['source_body'],
        source['preparation'],optimize=context['interpreter']['optimize'])
    saved = read_artifact(release['recipe'])
    _require(saved == dict(recipe,schema=release['recipe']['schema'],schema_version=1,source=source_ref),
        'the private arithmetic recipe changed from the original source loop')

    serial = read_artifact(release['serial_launch'])
    serial_study = read_artifact(release['serial_study'])
    _require(serial['study'] == release['serial_study'] and serial['entry'] == release['entry']
        and serial['context'] == context and serial['decision'] == 'LOAD_NATIVE_CANDIDATE'
        and serial_study['source'] == source['solve_scope'] and serial_study['native'] == release['native']
        and serial_study['budget_bytes'] == release['symbolic_budget_bytes'],
        'the released original carrier, native provider or serial fallback changed')
    native = read_artifact(release['native'])
    for ref in (*release['runtime_members'],native['library'],native['blas']):
        read_artifact(ref,json_document=False)
    read_artifact(release['problem'])
    published_plan = read_artifact(release['plan'])
    _require(any(row['option'] == 'PARALLEL' for row in published_plan['selected']),
        'the original release has no selected parallel implementation')
    study = read_artifact(release['study'])
    _require(study['study_parallel'] is True and study['decision'] == 'SERIAL'
        and all(release.get(key) == value for key,value in study.items()
                if key not in ('study_parallel','decision')),
        'the publication changed more than the original study qualification')
    validation = read_artifact(release['validation'])
    cost = read_artifact(release['cost'])
    _require(validation['selection'] == release['study'] and validation['clean'] is True
        and validation['context'] == cost['context']
        and cost['parallel_scope'] == source_ref and cost['source_scope'] == source['source_scope']
        and cost['native'] == release['native'] and cost['owner_cpu_id'] == owner
        and cost['worker_cpu_ids'] == release['workers']['cpu_ids'] and cost['window'] == release['workers']['window'],
        'historical publication or worker identity does not match its own original experiment')
    return original


def _lanes(hardware, cpus, node):
    processors = {row['cpu_id']:row for row in hardware['processors']}
    sources = {row['id']:row for row in hardware['sources']}
    _require(len(processors) == len(hardware['processors']) and len(sources) == len(hardware['sources']),
        'hardware snapshot has duplicate processor or original-source identities')
    _require(node in hardware['allowed_memory_nodes'] and set(cpus) <= set(hardware['launching_affinity']),
        'selected workers or their NUMA domain lie outside observed permissions')
    cores, siblings, lanes = set(), set(), []
    fields = ('physical_package_id','die_id','core_id','thread_siblings_list')
    para = ('package_id','die_id','core_id','siblings')
    for cpu in cpus:
        row = processors[cpu]
        _require(row['permitted'] is True and row['numa_nodes'] == [node],
            'selected CPU is unavailable or lies outside the original NUMA domain')
        receipts = [sources[key] for key in row['source_ids']]
        for field, attr, source in zip(fields,para,receipts,strict=True):
            text = source['text']
            path = f'/sys/devices/system/cpu/cpu{cpu}/topology/{field}'
            _require(source['path'] == path and source['status'] == 'observed' and type(text) is str
                and sha256(text.encode()).hexdigest() == source['sha256'],
                'selected CPU lost its original topology field receipt')
            value = _cpu_list(text) if field == 'thread_siblings_list' else text.strip()
            _require(value == row[attr], 'selected CPU topology does not match the original observation')
        core = tuple(row[k] for k in para[:-1])
        _require(all(type(value) is str and value.lstrip('-').isdecimal() for value in core)
            and core not in cores and cpu in row['siblings'] and not siblings.intersection(row['siblings']),
            'the owner and workers need distinct physical cores without SMT overlap')
        cores.add(core)
        siblings.update(row['siblings'])
        lanes.append(dict(cpu_id=cpu,physical_core=list(core),numa_node=node,
            observed_smt_siblings=row['siblings'],original_topology_sources=row['source_ids']))
    return lanes


def _scaling_choice(release, hardware_ref, hardware, invite, count, *, typed=False):
    current = invite['choice']
    owner, workers, window = current['owner_cpu_id'], current['worker_cpu_ids'], current['window']
    node = current['cpu_numa_node']
    previous = release['workers']
    limit = previous['address_space_limit_bytes']
    _require(invite['schema_version'] == hardware['schema_version'] == 1 and invite['hardware'] == hardware_ref
        and invite['dimensions']['call_count'] == count and release['owner_cpu'] == owner
        and release['numa_node'] == node and sorted(set(workers)) == workers and owner not in workers
        and current['worker_count'] == len(workers) == window
        and previous['window'] < window <= min(invite['dimensions']['max_ordered_window'],count)
        and window <= current['physical_worker_limit']
        and (typed or window <= current['owner_saturation_estimate_workers'])
        and invite['budget']['worker_total_limit_bytes'] == limit
        and current['automatic_admission'] is False and current['resource_snapshot_is_reservation'] is False,
        'choice must enlarge only the worker team/window of the same ordered owner and per-worker contract')
    if 'calibration_seed' in invite:
        _require(invite['calibration_seed'] == release['cost'],
            'different historical calibration cannot qualify this retargeting')
    _require(release['launch_cpu_ids'] == sorted([owner,*previous['cpu_ids']])
        and previous['window'] == len(previous['cpu_ids']), 'published source team differs from its original window')
    return dict(cpu_ids=workers,window=window,address_space_limit_bytes=limit), _lanes(hardware,[owner,*workers],node)


def _typed_anchor(parallel_scope, release, invite, lanes):
    from scar.ir.execution_recipe import SCHEMA as RECIPE, validate_recipe
    from scar.optimization.resource_solver import semantic_sha256
    from scar.optimization.sparse_parallel_recipe import PROFILE

    frontier = read_artifact(invite['team_frontier'])
    recipe_ref, lowering_ref = release['execution_recipe'], release['lowering']
    _require(recipe_ref['schema'] == RECIPE and lowering_ref['schema'] == 'scar.execution-recipe-lowering'
        and frontier['execution_recipe'] == recipe_ref,
        'the model frontier does not retain the original typed parallel recipe')
    recipe = validate_recipe(read_artifact(recipe_ref))
    lowering = read_artifact(lowering_ref)
    artifacts = recipe['artifacts']
    original_cost = read_artifact(release['cost'])
    _require(recipe['profile'] == PROFILE and artifacts['parallel_scope'] == parallel_scope
        and artifacts['serial_launch'] == release['serial_launch']
        and artifacts['detached'] == release['recipe'] and artifacts['choice'] == original_cost['choice']
        and original_cost.get('execution_recipe') == recipe_ref
        and recipe['semantic_boundary']['entry_context'] == release['context']
        and recipe['semantic_boundary']['original_entry'] == release['entry'],
        'typed recipe, costs, detached arguments or original entry belong to a different scope')
    retained_recipe = lowering['recipe']
    capability = recipe['implementation']['backend_capability_record']
    policy_keys = ('launch_cpu_ids', 'owner_cpu', 'numa_node', 'count', 'matrix', 'workers',
                   'symbolic_budget_bytes', 'memory_floor_bytes')
    _require(retained_recipe['schema'] == RECIPE and read_artifact(retained_recipe) == recipe
        and lowering['schema_version'] == 1 and lowering['profile'] == PROFILE
        and lowering['recipe_replayed'] is True and lowering['native_recompiled'] is False
        and lowering['recipe_id'] == recipe['recipe_id']
        and lowering['recipe_sha256'] == retained_recipe['sha256']
        and all(lowering[key + '_sha256'] == semantic_sha256(recipe[key])
                for key in ('semantic_boundary', 'control', 'task_graph', 'resources'))
        and lowering['input_artifacts'] == artifacts and lowering['backend_capability_record'] == capability
        and lowering['native'] == capability['native'] == release['native']
        and lowering['detached_recipe'] == artifacts['detached']
        and lowering['runtime_policy'] == {key:release[key] for key in policy_keys},
        'the established launch does not use the recipe, native bytes and lowered resource policy')
    serial = read_artifact(release['serial_launch'])
    serial_recipe = read_artifact(capability['serial_recipe'])
    serial_lowering = read_artifact(capability['serial_lowering'])
    native = read_artifact(release['native'])
    _require(serial['execution_recipe'] == capability['serial_recipe'] == lowering['serial_recipe']
        == serial_lowering['recipe'] and serial['lowering'] == capability['serial_lowering']
        == lowering['serial_lowering'] and serial_lowering['recipe_id'] == serial_recipe['recipe_id']
        and serial_lowering['native'] == release['native']
        and lowering['native_library'] == serial_lowering['native_library'] == native['library'],
        'the established typed lowering and fallback do not share the admitted original ABI')
    cpu, memory, dispatch, case = (recipe['resources'][key] for key in ('selected_cpu', 'memory', 'dispatch', 'case'))
    previous = release['workers']
    _require(cpu == dict(owner_cpu_id=release['owner_cpu'], worker_cpu_ids=previous['cpu_ids'],
            launch_cpu_ids=release['launch_cpu_ids'], cpu_numa_node=release['numa_node'])
        and case == dict(call_count=release['count'], **release['matrix'])
        and dispatch['window'] == previous['window']
        and memory['worker_address_space_limit_bytes'] == previous['address_space_limit_bytes']
        and memory['per_worker_symbolic_budget_bytes'] == release['symbolic_budget_bytes']
        and memory['minimum_launch_headroom_bytes'] == release['memory_floor_bytes'],
        'the established typed resource coordinates differ from the released policy')
    previous_hardware = read_artifact(read_artifact(artifacts['choice'])['hardware'])
    previous_lanes = _lanes(previous_hardware, release['launch_cpu_ids'], release['numa_node'])
    expanded = {lane['cpu_id']:lane for lane in lanes}
    preserved = ('physical_core', 'numa_node', 'observed_smt_siblings')
    _require(set(release['launch_cpu_ids']) < set(expanded)
        and all(all(expanded[lane['cpu_id']][key] == lane[key] for key in preserved) for lane in previous_lanes),
        'the typed challenger must preserve every original CPU and its physical topology')
    return dict(execution_recipe=recipe_ref, lowering=lowering_ref, recipe_id=recipe['recipe_id'],
        detached_recipe=release['recipe'], native=release['native'], serial_launch=release['serial_launch'],
        serial_recipe=capability['serial_recipe'], serial_lowering=capability['serial_lowering'],
        established_choice=artifacts['choice'], calibration_seed=release['cost'], team_frontier=invite['team_frontier'],
        preserved_cpu_ids=release['launch_cpu_ids'], added_worker_cpu_ids=sorted(set(expanded) - set(release['launch_cpu_ids'])))


def _resources(lanes, window, cap, hardware):
    owner, *workers = lanes
    free = hardware['node_free_snapshot'].get(str(owner['numa_node']),{})
    return dict(owner_lane=owner,worker_lanes=workers,
        execution_edges=[dict(kind='RUNS_ON',activity='ORIGINAL_PYTHON_ISSUE_AND_PUBLICATION',cpu_id=owner['cpu_id']),
            *(dict(kind='RUNS_ON',activity='DETACHED_PRIVATE_NATIVE_WORK',cpu_id=lane['cpu_id']) for lane in workers)],
        order_edges=[dict(kind='VALUE',source='SCOPE_OWNED_SNAPSHOT',target='DETACHED_WORKER_INPUT'),
            dict(kind='VALUE',source='DETACHED_WORKER_RESULT',target='EXACT_ORIGINAL_CARRIER_MATCH'),
            dict(kind='ORDER',source='ORIGINAL_PYTHON_PUBLICATION',target='NEXT_ITERATION_PUBLICATION',iteration_distance=1),
            dict(kind='CAPACITY_RELEASE',source='OWNER_RESULT_RETIREMENT',target='REUSE_WORKER_SLOT')],
        private_state=dict(independent_symbolic_plan_per_worker=True,retained_scope='ONE_ORIGINAL_ENTRY',
            in_flight_window=window,maximum_in_flight_per_worker=1,
            per_worker_virtual_address_limit_bytes=cap,total_worker_virtual_address_limits_bytes=window*cap),
        observed_capacity=dict(cpu_quota_equivalents=hardware['cpu_quota_equivalents'],
            cgroup_memory_headroom_snapshot_bytes=hardware['finite_memory_headroom_snapshot_bytes'],
            numa_free_memory_snapshot_bytes=free.get('bytes'),memory_nodes_allowed=hardware['allowed_memory_nodes'],
            cgroup_hierarchy_observation_complete=hardware['hierarchy_observation_complete']),
        memory_reservation=False,memory_placement_enforced=False,physical_page_residency_observed=False,
        sibling_interference_observed=False)


def derive_sparse_parallel_scaling_scope(parallel_scope, released_selection, hardware, choice):
    """Independently re-admit the original program and bind a fresh study team."""
    refs = (parallel_scope,released_selection,hardware,choice)
    _require(tuple(ref.get('schema') for ref in refs) == (PARALLEL,SELECTION,HARDWARE,CHOICE),
        'expected a sparse parallel scope, published selection, hardware and choice receipts')
    release,observed,invite = tuple(read_artifact(ref) for ref in refs[1:])
    flags = ('execution_recipe' in release, 'lowering' in release, 'team_frontier' in invite)
    typed = any(flags)
    _require(not typed or all(flags), 'typed expansion requires both its established lowering and model choice')
    with nullcontext() if not typed or current_transaction() is not None else replay_transaction():
        if typed:
            from scar.optimization.sparse_parallel_team_choice import check_choice

            _require(invite['released_selection'] == released_selection and invite['parallel_scope'] == parallel_scope
                and invite['calibration_seed'] == release['cost'] and check_choice(choice) == invite,
                'the model choice is not the challenger for this exact typed publication')
        return _derive_scope(parallel_scope, released_selection, hardware, choice, release, observed, invite, typed)


def _derive_scope(parallel_scope, released_selection, hardware, choice, release, observed, invite, typed):
    source = check_sparse_parallel_scope(parallel_scope)
    original = _release_lineage(parallel_scope,source,release)
    count = release['count']
    workers,lanes = _scaling_choice(release,hardware,observed,invite,count,typed=typed)
    owner = release['owner_cpu']
    contract = source['speculation_contract']
    result = dict(schema=SCHEMA,schema_version=1,parallel_scope=parallel_scope,released_selection=released_selection,
        hardware=hardware,choice=choice,source_scope=source['source_scope'],pipeline_scope=source['pipeline_scope'],
        entry=source['entry'],context=release['context'],count=count,owner_cpu=owner,numa_node=release['numa_node'],
        original_matrix=release['matrix'],original_source_count_evidence=original['observed']['exact_count_evidence'],
        previous_workers=release['workers'],selected_workers=workers,
        launch_cpu_ids=sorted([owner,*workers['cpu_ids']]),
        resource_edges=_resources(lanes,workers['window'],workers['address_space_limit_bytes'],observed),
        scope_reuse=dict(source_body=source['source_body'],loop=source['loop']['original_operation'],
            original_native_call=contract['original_raw_dispatch']['call_point'],
            original_preparation=release['recipe'],native_provider=release['native'],
            serial_fallback=release['serial_launch'],
            exact_acceptance_key=contract['acceptance_key'],original_ordered_effects=contract['original_effects_preserved'],
            only_resource_parameters_change=['WORKER_CPU_IDS','ORDERED_WINDOW'],
            source_bytes_count_owner_context_and_provider_preserved=True),
        qualification=dict(same_numa_distinct_physical_core_snapshot=True,source_scope_cold_replayed=True,
            historical_cost_can_seed_the_choice=True,historical_timings_transferred=False,
            original_numa_physical_pages_transferred=False,fresh_resource_preflight_required=True,
            fresh_exact_context_measurement_required=True,automatic_admission=False),
        frontiers=['WORKER_THREAD_AND_BLAS_EXECUTION_MUST_RECHECK_ORIGINAL_FLAGS',
            'CAPACITY_SNAPSHOT_IS_NOT_PHYSICAL_RESERVATION_OR_LOCAL_PAGE_PLACEMENT',
            'MORE_CORES_CAN_CHANGE_IPC_OWNER_WAIT_BOOTSTRAP_AND_BANDWIDTH_COSTS'])
    if typed:
        result['typed_anchor'] = _typed_anchor(parallel_scope,release,invite,lanes)
    return result


def check_sparse_parallel_scaling_scope(reference):
    """Recheck the exact source and all retained resource membership."""
    document = read_artifact(reference) if 'uri' in reference else reference
    _require((document.get('schema'),document.get('schema_version')) == (SCHEMA,1),
        'expected a sparse parallel scaling scope')
    fields = ('parallel_scope','released_selection','hardware','choice')
    expected = derive_sparse_parallel_scaling_scope(*(document[field] for field in fields))
    canonical = lambda value: json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    _require(canonical(expected) == canonical(document),'scaling scope differs from original cold replay')
    return expected

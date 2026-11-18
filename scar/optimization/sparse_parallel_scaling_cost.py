"""Compare two exact sparse solver teams under one paired sampling design."""

from copy import deepcopy

from scar.ir.execution_recipe import SCHEMA as RECIPE, artifact_reference, read_artifact, validate_recipe
from scar.optimization.sparse_parallel_cost import SCHEMA as CALIBRATION, check as check_calibration
from scar.optimization.sparse_solve_cost import _duration, _summary

SCHEMA = 'scar.sparse-parallel-scaling-cost'
DESIGN = SCHEMA + '-design'
_TEAMS = ('baseline', 'expanded')
_ESTIMATES = ('estimate_ns', 'lower_ns', 'upper_ns')
_TYPED_PROFILE = 'cpu-ordered-sparse-solve-v1'
_PHASES = ('queue_ns', 'request_exchange_child_setup_ns', 'native_solve_ns', 'result_exchange_unpack_ns')


def _measure(raw):
    stats = raw['runtime']
    tasks = stats['tasks']
    first = min(task['submitted_ns'] for task in tasks)
    last = max(row['consume_clock_ns'][-1] for row in raw['samples'])
    body_wall = last-first
    per_cpu = {}
    ipc = []
    for task in tasks:
        cpu = task['cpu']
        row = per_cpu.setdefault(cpu, dict(tasks=0, native_cpu_ns=0, sampled_max_rss_bytes=0))
        row['tasks'] += 1
        row['native_cpu_ns'] += task['native_thread_cpu_ns']
        row['sampled_max_rss_bytes'] = max(row['sampled_max_rss_bytes'], task['worker_max_rss_bytes'])
        ipc.append(task['exchange_complete_ns']-task['submitted_ns']
                   - (task['ended_ns']-task['started_ns']))
    for row in per_cpu.values():
        row['native_cpu_fraction_of_measured_body'] = row['native_cpu_ns']/body_wall
    return dict(body_wall_ns=body_wall,
                short_scope_wall_ns=raw['close_clock_ns'][-1]-raw['bootstrap_clock_ns'][0],
                host_ordered_wait_ns=sum(row['consume_wait_ns'] for row in stats['publications']),
                worker_task_outside_c_ns=_summary(ipc),
                per_worker_cpu={str(cpu): row for cpu, row in sorted(per_cpu.items())},
                ownership=dict(snapshot_peak_bytes=stats['peak_snapshot_bytes'],
                               ordered_result_bound_bytes=stats['pending_result_bound_bytes']))


def _family_body(recipe):
    """Remove only the coordinates and capacities changed by a larger team."""
    body = deepcopy(recipe)
    body.pop('recipe_id')
    artifacts = body['artifacts']
    artifacts.pop('choice')
    # The detached source plan may be copied into each prepared recipe directory.
    artifacts['detached'].pop('uri')
    body['control'].pop('window')
    resources = body['resources']
    resources.pop('hardware')
    selected = resources['selected_cpu']
    worker_ids = selected.pop('worker_cpu_ids')
    selected.pop('launch_cpu_ids')
    resources['dispatch'].pop('window')
    memory = resources['memory']
    for scaled in ('worker_address_space_sum_bytes', 'owner_pending_input_bytes',
                   'owner_pending_result_bound_bytes', 'minimum_launch_headroom_bytes'):
        memory.pop(scaled)
    graph = body['task_graph']
    for node in (*graph['tasks'], *graph['materializations']):
        for role in node['resource_refs']:
            if role['kind'] == 'ISOLATED_WORKER':
                if role.pop('cpu_ids') != worker_ids:
                    raise ValueError('worker actor does not use the declared CPU team')
    return body


def _typed_recipes(design, studies):
    if 'execution_recipes' not in design:
        if any('execution_recipe' in study for study in studies.values()):
            raise ValueError('typed calibration requires a paired recipe design')
        return None
    references = design['execution_recipes']
    if set(references) != set(_TEAMS) or references['baseline'] == references['expanded']:
        raise ValueError('two distinct ordered execution recipes are required')
    bodies = []
    for mode in _TEAMS:
        ref, study = references[mode], studies[mode]
        if ref['schema'] != RECIPE or study.get('execution_recipe') != ref:
            raise ValueError('paired clocks do not belong to their frozen execution recipe')
        recipe = validate_recipe(read_artifact(ref))
        if recipe['profile'] != _TYPED_PROFILE or recipe['control']['window'] != study['window']:
            raise ValueError('expected the ordered sparse recipe and its measured window')
        bodies.append(_family_body(recipe))
    if bodies[0] != bodies[1]:
        raise ValueError('paired teams changed source, serial launch, actor graph or lifecycle')
    count = max(study['window'] for study in studies.values())
    if (len(design['groups']) != 2 or any(len(group['iterations']) < 2*count+1 for group in design['groups'])
            or design['team_order'] != [['baseline','expanded'], ['expanded','baseline']]):
        raise ValueError('typed scaling requires two paired full-window samples with reversed team order')
    return references


def _typed_parts(raw):
    tasks = sorted(raw['runtime']['tasks'], key=lambda item: item['issue'])
    phases = {}
    for task in tasks:
        cuts = [task[key] for key in ('submitted_ns', 'exchange_started_ns', 'started_ns',
                                     'ended_ns', 'exchange_complete_ns')]
        for key, start, end in zip(_PHASES, cuts, cuts[1:]):
            phases.setdefault(key, []).append(end-start)
    boot = _duration(raw['bootstrap_clock_ns'])
    close = _duration(raw['close_clock_ns'])
    return dict(phases=phases, bootstrap_ns=boot, team_startup_ns=raw['runtime']['startup_ns'],
                close_ns=close)


def _typed_comparison(design, studies, raw_groups, recipes):
    observations = []
    for index, team_order in enumerate(design['team_order']):
        raw = {mode: raw_groups[mode][index] for mode in _TEAMS}
        first, second = (raw[mode] for mode in team_order)
        if first['close_clock_ns'][-1] >= second['bootstrap_clock_ns'][0]:
            raise ValueError('observed team samples overlap or differ from the frozen order')
        observations.append({mode: _typed_parts(raw[mode]) for mode in _TEAMS})
    phase_delta = {}
    for key in _PHASES:
        paired = [left-right for group in observations
                  for left,right in zip(group['baseline']['phases'][key], group['expanded']['phases'][key])]
        phase_delta[key] = _summary(paired)
    lifecycle_delta = {
        key: _summary([row['baseline'][key]-row['expanded'][key] for row in observations])
        for key in ('bootstrap_ns', 'team_startup_ns', 'close_ns')
    }
    fields = ('worker_virtual_address_caps_bytes', 'worker_symbolic_payload_bytes',
              'owner_peak_pending_input_bytes', 'owner_pending_result_bound_bytes',
              'owner_structure_bytes', 'peak_worker_rss_bytes', 'maximum_native_wall_overlap')
    capacity = {mode: {field: max(group[field] for group in study['resources']) for field in fields}
                for mode,study in studies.items()}
    capacity_delta = {field: capacity['expanded'][field]-capacity['baseline'][field] for field in fields}
    return dict(execution_recipes=recipes, paired_phase_savings=phase_delta, paired_lifecycle_savings=lifecycle_delta,
                observed_team_capacity=capacity, observed_capacity_increase=capacity_delta,
                positive_time_delta_means='BASELINE_DURATION_MINUS_EXPANDED_DURATION',
                ipc_phase_boundaries='QUEUE_REQUEST_EXCHANGE_AND_CHILD_SETUP_NATIVE_RETURN_AND_UNPACK',
                worker_rss_is_single_process_sampled_peak=True)


def project(design_path, cost_paths):
    design_ref = artifact_reference(design_path, DESIGN)
    design = read_artifact(design_ref)
    if design['schema_version'] != 1 or set(cost_paths) != set(_TEAMS):
        raise ValueError('a paired baseline and expanded solver team are required')
    read_artifact(design['harness'], json_document=False)
    refs = {mode: artifact_reference(cost_paths[mode], CALIBRATION) for mode in _TEAMS}
    studies = {mode: check_calibration(refs[mode]['uri']) for mode in _TEAMS}
    typed = _typed_recipes(design, studies)
    baseline, expanded = (studies[name] for name in _TEAMS)
    context = {mode: {key: value for key, value in study['context'].items() if key != 'cpu_ids'}
               for mode, study in studies.items()}
    if (context['baseline'] != context['expanded'] or baseline['context'] != design['context']
            or not set(baseline['worker_cpu_ids']) < set(expanded['worker_cpu_ids'])
            or any(baseline[key] != expanded[key] for key in
                   ('owner_cpu_id', 'source_scope', 'parallel_scope', 'native'))):
        raise ValueError('compared teams must differ only by worker expansion')
    groups = design['groups']
    raw_groups = {}
    fixture = None
    worker = None
    for mode, study in studies.items():
        calibration = read_artifact(study['design'])
        if (study['choice'] != design['choices'][mode] or calibration['groups'] != groups
                or calibration['runtime'] != design['runtime']
                or calibration['worker_address_space_bytes'] != design['worker_address_space_bytes']):
            raise ValueError('solver sampling definition or worker bound differs from the paired design')
        if typed is not None:
            pinned = design.get('calibration_designs')
            if pinned is not None and pinned[mode] != study['design']:
                raise ValueError('paired execution differs from the predeclared calibration')
            if fixture is None:
                fixture, worker = calibration['fixture'], calibration['harness']
            if (fixture != calibration['fixture'] or worker != calibration['harness']
                    or design.get('worker', worker) != worker):
                raise ValueError('typed teams require the same source fixture and execution harness')
        raw_groups[mode] = {}
        for observation in study['observations']:
            raw = read_artifact(observation)
            if raw['mode'] == 'ordered':
                raw_groups[mode][raw['group']] = raw
    paired = []
    for index, group in enumerate(groups):
        raw = {mode: raw_groups[mode][index] for mode in _TEAMS}
        identities = [(item['output_digests'], item['published_digest']) for item in raw.values()]
        if identities[0] != identities[1] or len(identities[0][0]) != len(group['iterations']):
            raise ValueError('expanded team output differs at a paired source iteration')
        paired.append(dict(group=index, measurements={mode: _measure(item) for mode, item in raw.items()}))
    spans = {
        mode: {key: study['projections'][key]['elapsed_projection_ns'] for key in _ESTIMATES}
        for mode, study in studies.items()
    }
    delta = {
        'estimate_ns': spans['baseline']['estimate_ns']-spans['expanded']['estimate_ns'],
        'lower_ns': spans['baseline']['lower_ns']-spans['expanded']['upper_ns'],
        'upper_ns': spans['baseline']['upper_ns']-spans['expanded']['lower_ns'],
    }
    measured_delta = {
        key: _summary([group['measurements']['baseline'][key]-group['measurements']['expanded'][key]
                       for group in paired])
        for key in ('body_wall_ns', 'short_scope_wall_ns', 'host_ordered_wait_ns')
    }
    binding = {} if typed is None else _typed_comparison(design, studies, raw_groups, typed)
    return dict(schema=SCHEMA, schema_version=1, design=design_ref, costs=refs,
                owner_cpu_id=baseline['owner_cpu_id'], source_scope=baseline['source_scope'],
                teams={mode: dict(cpu_ids=row['worker_cpu_ids'], window=row['window'])
                       for mode, row in studies.items()},
                projected_section_ns=spans, projected_marginal_savings_ns=delta,
                paired_diagnostics=paired, paired_short_section_savings_ns=measured_delta,
                output_bitwise=True, memory_bandwidth_bytes_per_second=None,
                exact_peak_native_scratch_bytes=None, physical_memory_residency=None,
                full_entry_cost_ns=None, automatic_admission=False, **binding)


def check(path):
    saved = read_artifact(artifact_reference(path, SCHEMA))
    replayed = project(saved['design']['uri'], {mode: ref['uri'] for mode, ref in saved['costs'].items()})
    if saved != replayed:
        raise ValueError('paired team comparison differs from replay')
    return saved

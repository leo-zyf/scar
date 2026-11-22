"""Qualify an ordered native task entry against independent original process reports."""

import argparse
import json
from pathlib import Path
from statistics import pvariance

from scar.compiler.calibrate_array_entry import _normalize, _ref, _replay, _run, _write
from scar.ir.execution_recipe import read_artifact
from scar.optimization.paired_entry_selection import compare_entry_pairs, replay_entry_pairs
from scar.trace.numpy_partition_entry_validation import normalize_report


SCHEMA = 'scar.native-array-task-entry-calibration'
_ENTRY = 'scar.native-seeded-array-task-entry'
_ORIGINAL = 'scar.array-entry-task-family-holdout'
_STATES = 'scar.array-entry-task-state-oracle'
_LANES = 'scar.native-array-openmp-lanes'
_MPI_RANKS = 'scar.native-array-mpi-ranks'
_TIMING_SCOPE = 'FRESH_EXEC_THROUGH_EXIT_INCLUDES_ORDERED_STDOUT_AND_FILE_PUBLICATION'
_OPENMP = 'ORDERED_OPENMP'
_SERIAL = 'IN_PROCESS_SERIAL'
_PROCESSES = 'ORDERED_LOCAL_PROCESSES'
_MPI = 'ORDERED_LOCAL_MPI'
_PROCESS_TEAMS = {_PROCESSES, _MPI}
_PLACEMENT_REPORTS = {_OPENMP: ('openmp-lanes.json', _LANES), _MPI: ('mpi-ranks.json', _MPI_RANKS)}
_CONTROL_LABELS = {_OPENMP: 'openmp', _SERIAL: 'serial', _PROCESSES: 'processes', _MPI: 'mpi'}
_OUTPUT_FIELDS = {'results', 'pid_list', 'ordinal', 'effective_seed', 'worker_round_count'}
_INPUT_OPTIONS = {'task_count', 'worker_limit', 'seed', 'round_count'}


def _need(condition, detail):
    if not condition:
        raise ValueError('array task entry calibration: ' + detail)


def _load_schedule(reference):
    from scar.optimization import independent_array_task_schedule as process_schedule
    from scar.optimization import independent_array_task_team_schedule as team_schedule

    schema = reference['schema']
    if schema == process_schedule.SCHEMA:
        return process_schedule.check(reference['uri'])
    if schema == team_schedule.SCHEMA:
        return team_schedule.check(reference['uri'])
    from scar.optimization import independent_array_task_mpi_schedule as mpi_schedule

    _need(schema == mpi_schedule.SCHEMA, 'unrecognized task schedule')
    return mpi_schedule.check(reference['uri'])


def _check_mpi_launch(package, schedule):
    selected = package['decision']
    prefix = package.get('launch_prefix')
    graph = package.get('mpi_graph')
    if selected['mode'] != _MPI:
        _need(prefix in (None, []) and graph is None, 'unexpected MPI launcher for a non-MPI mode')
        return
    _need(not schedule['decision'].get('all_eligible_mpi_rejected'),
          'MPI launch is unavailable because the schedule rejected every eligible rank plan')
    _need(schedule['decision']['status'] == 'NOMINATE_MPI_FOR_INDEPENDENT_VALIDATION'
          and schedule['decision']['selected'] == selected,
          'MPI launch requires the exact nominated rank plan')
    _need(graph == schedule.get('mpi_graph') and graph is not None
          and type(prefix) is list and prefix and prefix == selected['launch_prefix'],
          'MPI launcher differs from the priced rank plan')
    runtime = read_artifact(graph)['runtime']['libmpi']
    dependencies = package['linkage']['qualified_runtime_dependencies']
    _need(runtime in dependencies.values(), 'MPI runtime differs from the modeled launcher')
    read_artifact(runtime, json_document=False)


def _local_team(schedule):
    return read_artifact(schedule['team_plan']) if 'mpi_graph' in schedule else schedule


def _load(path, *, checked_source=None, already_replayed_schedule=None):
    from scar.codegen.native_seeded_array_entry import LOOP_SCHEMA
    from scar.compiler.calibrate_seeded_array_entry import _load as load_seeded

    reference = _ref(path, _ENTRY)
    package = read_artifact(reference)
    plan_ref = package['schedule']
    if checked_source is not None:
        parent, native, recipe = checked_source
        serial = parent['native']
        _need(package['native'] == serial and plan_ref == parent['schedule'],
              'validation control changed the independently checked source or schedule')
        schedule = read_artifact(plan_ref)
    else:
        serial, native, recipe = load_seeded(package['native']['uri'])
        if already_replayed_schedule is None:
            schedule = _load_schedule(plan_ref)
        else:
            checked_ref, schedule = already_replayed_schedule
            _need(checked_ref == plan_ref and schedule == read_artifact(plan_ref),
                  'acceptance does not use the schedule replayed in this transaction')
    selected = package['decision']
    control = package.get('execution_control')
    if package.get('serial_control'):
        _need(control in (None, _SERIAL), 'serial control conflicts with execution mode')
        control = _SERIAL
    if control is None:
        _need(checked_source is None and selected == schedule['decision']['selected'],
              'compiled mode differs from selected schedule')
    else:
        _need(checked_source is not None and control == selected['mode'] and selected in schedule['alternatives']
              and not selected['capacity_reasons'] and selected['cost'] is not None,
              'execution control differs from feasible priced schedule alternative')
    _need(serial == package['native'] and native['recipe'] == package['recipe']
          and native['schema'] == LOOP_SCHEMA
          and package['numerical_body_reused'] == native['files']['src/profile_0.cpp']
          and all(package['files'][name]['sha256'] == native['files'][name]['sha256']
                  for name in ('src/profile_0.cpp', 'src/tape_0.inc'))
          and read_artifact(plan_ref) == schedule
          and schedule['native'] == serial and schedule['family'] == package['family']
          and schedule['scope'] == package['scope']
          and package['python_runtime_required'] is False and package['no_worker_visible_output'] is True
          and package['deployment_authorized'] is False and package['original_default_retained'] is True
          and selected['launch_cpu_ids'] == package['cpu_ids'],
          'compiled task scope or decision differs from the checked schedule')
    _need(_OUTPUT_FIELDS <= package['output_fields'].keys()
          and set(package['input_options']) == _INPUT_OPTIONS
          and set(package['input_options'].values()) <= set(recipe['options'])
          and len(set(package['input_options'].values())) == len(_INPUT_OPTIONS),
          'entry has an incomplete task report or parameter binding')
    source = read_artifact(recipe['source_contract'])
    fields = package['output_fields']
    local_team = _local_team(schedule)
    self_pid, = (field['path'] for field in recipe['report_bindings'] if field['role'] == 'pid')
    pids, = (field['path'] for field in recipe['report_bindings'] if field['role'] == 'pid_list')
    _need(self_pid[:2] == [fields['results'], 0] and pids == [fields['pid_list']]
          and selected['worker_count'] <= package['scope']['task_count']
          and selected['mode'] in _CONTROL_LABELS
          and package.get('thread_contract') == (local_team['thread_contract'] if selected['mode'] == _OPENMP else None),
          'source process reporting or selected task ownership differs')
    override = package.get('launch_environment', {})
    _need(override == ({'OMP_STACKSIZE': '8M'} if selected['mode'] == _OPENMP else {})
          and (selected['mode'] != _OPENMP or 'GOMP_STACKSIZE' not in recipe['environment']),
          'native launch environment differs from the bounded OpenMP stack contract')
    if selected['mode'] == _OPENMP:
        libgomp = package['linkage']['qualified_runtime_dependencies']['libgomp.so.1']
        _need(libgomp == local_team['openmp_runtime'], 'OpenMP runtime differs from the modeled dispatcher')
        read_artifact(libgomp, json_document=False)
    _check_mpi_launch(package, schedule)
    for artifact in package['files'].values():
        read_artifact(artifact, json_document=False)
    return reference, package, recipe, native, source['report']['parameters']['field'], self_pid[-1]


def _report(raw, package, parameters_field, pid_field, expected_params, worker_counts):
    report = json.loads(raw)
    fields = package['output_fields']
    scope = package['scope']
    effective = expected_params[package['input_options']['seed']]
    tasks = report[fields['results']]
    _need(report[parameters_field] == expected_params and len(tasks) == scope['task_count'],
          'parameters or output task count differs')
    pids = set()
    for ordinal, result in enumerate(tasks):
        pid = result[pid_field]
        _need(result[fields['ordinal']] == ordinal and result[fields['effective_seed']] == effective + ordinal
              and result[fields['worker_round_count']] == scope['round_count']
              and type(pid) is int and pid > 0, 'task order, seed route or execution identity differs')
        pids.add(pid)
    _need(report[fields['pid_list']] == sorted(pids) and len(pids) in worker_counts,
          'reported active process count differs from the executed ownership boundary')
    return tasks, sorted(pids)


def _holdout(reference, package, recipe, prior, parameters_field, pid_field):
    case = read_artifact(reference)
    options = package['input_options']
    parameters = case['parameters']
    fixed = {key: value for key, value in parameters.items() if key not in options.values()}
    prior_fixed = {key: value for key, value in prior['source_parameters'].items() if key not in options.values()}
    worker_count = case['executed_worker_process_count']
    source_seeds = {row['parameters'][options['seed']] for row in recipe['profiles']}
    schedule = read_artifact(package['schedule'])
    local_team = _local_team(schedule)
    related_seeds = source_seeds | set(read_artifact(local_team['evidence'])['seeds'])
    width = package['scope']['task_count']
    evidence = [(owner[key], width) for owner, key in ((local_team, 'thread_evidence'), (schedule, 'mpi_evidence'))
                if key in owner]
    team_seeds = []
    if 'mpi_preflight' in schedule:
        preflight = read_artifact(schedule['mpi_preflight'])
        evidence.append((preflight['platform_evidence'], preflight['platform_scope']['task_count']))
        if 'local_acceptance' in preflight:
            earlier_acceptance = read_artifact(preflight['local_acceptance'])['holdout']
            team_seeds.append((read_artifact(earlier_acceptance)['parameters'][options['seed']], width))
    team_seeds.extend((base, old_width) for ref, old_width in evidence for base in read_artifact(ref)['seeds'])
    seed = parameters[options['seed']]
    domain = prior['seed_range']
    count = parameters[options['task_count']]
    requested = parameters[options['worker_limit']]
    _need(case['entry'] == recipe['entry'] and case['recipe'] == package['recipe']
          and case['source_contract'] == recipe['source_contract'] and case['interpreter'] == recipe['source_python']
          and case['cpu_ids'] == package['cpu_ids'] and case['environment'] == recipe['environment']
          and case['valid_for'] == 'VALIDATION_ONLY' and case['independent_original_run'] is True
          and case['instrumented'] is False and case['accelerated_backend_used'] is False
          and case['topology_generation_authorized'] is False and case['tuning_authorized'] is False
          and case['normalized_runtime_fields'] == recipe['runtime_fields'] and case['returncode'] == 0
          and fixed == prior_fixed
          and package['scope'] == dict(task_count=count, requested_workers=requested,
                                       round_count=parameters[options['round_count']])
          and 1 < worker_count <= min(count, requested)
          and type(seed) is int and domain['minimum'] <= seed <= domain['maximum'] - (count - 1)
          and all(not seed <= related < seed + count for related in related_seeds)
          and all(not (seed < base + old_width and base < seed + count) for base, old_width in team_seeds),
          'holdout crosses the original process, task scope or seed boundary')
    original = read_artifact(case['stdout'], json_document=False)
    tasks, _ = _report(original, package, parameters_field, pid_field, parameters, (worker_count,))
    normalized = _normalize(original, recipe)
    _need(tasks == case['per_task'] and not read_artifact(case['stderr'], json_document=False)
          and json.loads(original) == json.loads(read_artifact(case['files']['out'], json_document=False))
          and normalized == read_artifact(case['normalized'], json_document=False),
          'independent original report and on-disk publication differ')
    flags, = recipe['options'][recipe['output_option']]['flags']
    argv = case['argv']
    _need(argv[-2:] == [flags, 'published.json'] and argv.count(flags) == 1
          and case['command'] == _commands(package, recipe)['original'],
          'original holdout must identify its executable and sole output file route')
    return case, normalized


def _commands(package, recipe):
    launcher = ['/usr/bin/taskset', '-c', ','.join(map(str, package['cpu_ids']))]
    return dict(original=[*launcher, recipe['source_python']['uri'], '-B', recipe['entry']['uri']],
                native=[*(package.get('launch_prefix') or launcher), package['files'][package['binary']]['uri']])


def _validate_runs(runs, package, recipe, case, expected, parameters_field, pid_field, *, native_label='native'):
    commands = _commands(package, recipe)
    commands[native_label] = commands.pop('native')
    selected = package['decision']
    native_process_count = selected['worker_count'] if selected['mode'] in _PROCESS_TEAMS else 1
    checked, pids = [], []
    for run in runs:
        mode = run['mode']
        override = package.get('launch_environment', {}) if mode == native_label else {}
        _need(mode in commands and run['command'] == commands[mode] and run['argv'] == case['argv'],
              'paired process command or original arguments differ')
        _need(run.get('launch_environment', {}) == override, 'runtime override differs from published package')
        checked.append(_replay(run, recipe, expected))
        # In a fresh original run, the process pool may leave a late process idle.
        workers = ((native_process_count,) if mode == native_label else
                   range(1, package['scope']['requested_workers'] + 1))
        tasks, active = _report(read_artifact(run['stdout'], json_document=False), package,
                                parameters_field, pid_field, case['parameters'], workers)
        observation = dict(pair=run['pair'], mode=mode, active_worker_pids=active)
        if mode == native_label and selected['mode'] in _PROCESS_TEAMS:
            width = native_process_count
            lanes = [row[pid_field] for row in tasks[:width]]
            _need(len(set(lanes)) == width and all(row[pid_field] == lanes[index % width]
                                                  for index, row in enumerate(tasks)),
                  'native ordinal-to-process routing differs from the selected plan')
            if selected['mode'] == _MPI:
                observation['mpi_rank_pids'] = lanes
        pids.append(observation)
    return checked, pids


def _state_oracle(reference, holdout, case, package, recipe):
    oracle = read_artifact(reference)
    _need(oracle['holdout'] == holdout and oracle['valid_for'] == 'VALIDATION_ONLY'
          and oracle['captured_monkeypatch_free'] is True and oracle['performance_measured'] is False
          and len(oracle['workers']) == package['scope']['task_count'],
          'state oracle must cover every independently validated task')
    scope = read_artifact(case['source_contract'])['worker']['code_id']
    result_field = package['output_fields']['results']
    omitted = [(route[-1],) for route in recipe['runtime_fields'] if route[:2] == [result_field, '*']]
    outputs = []
    for ordinal, ref in enumerate(oracle['workers']):
        worker = read_artifact(ref)
        _need(worker['worker_ordinal'] == ordinal and worker['worker_code_id'] == scope
              and worker['original_holdout'] == holdout and worker['original_entry'] == case['entry']
              and worker['captured_monkeypatch_free'] is True and worker['independent_original_worker'] is True
              and worker['capture_is_diagnostic'] is True and worker['tuning_authorized'] is False
              and normalize_report(worker['worker_report'], omit_paths=omitted)
              == normalize_report(case['per_task'][ordinal], omit_paths=omitted)
              and len(worker['states']) == package['scope']['round_count'],
              'state oracle escaped the independently validated task scope')
        trajectory = read_artifact(worker['trajectory'])
        _need(len(trajectory['records']) == len(worker['states']), 'natural result read count differs')
        for index, (row, observed) in enumerate(zip(worker['states'], trajectory['records'], strict=True)):
            output = observed['output']
            data = row['data']
            _need(row['round_index'] == index and row['dtype'] == output['dtype']
                  and row['shape'] == output['shape'] and data['sha256'] == output['sha256']
                  and data['size_bytes'] == output['size_bytes'], 'task state provenance differs')
            outputs.append((ordinal, index, data))
    _need(oracle['bytes'] == sum(data['size_bytes'] for _, _, data in outputs), 'retained state extent differs')
    return outputs


def _equal_states(oracle_rows, checkpoints):
    for (ordinal, index, original), native in zip(oracle_rows, checkpoints, strict=True):
        _need(read_artifact(original, json_document=False) == read_artifact(native, json_document=False),
              f'complete state bytes differ for task {ordinal} round {index}')


def _lane_assignments(reference, package, observation):
    detail = read_artifact(reference)
    selected = package['decision']
    width, cpus = selected['worker_count'], selected['worker_cpu_ids']
    tasks = detail['tasks']
    active_processes = observation['active_worker_pids']
    mpi = selected['mode'] == _MPI
    _need(selected['mode'] in _PLACEMENT_REPORTS and reference['schema'] == _PLACEMENT_REPORTS[selected['mode']][1]
          and type(detail['schema_version']) is int and detail['schema_version'] == 1
          and type(detail['team_size']) is int and detail['team_size'] == width
          and 1 < len(cpus) == width and len(set(cpus)) == width
          and cpus[0] == selected['owner_cpu_id'] and len(active_processes) == (width if mpi else 1)
          and len(tasks) == package['scope']['task_count'] >= width,
          'native workers disagree with the selected core map')
    tids = []
    for ordinal, row in enumerate(tasks):
        lane = ordinal % width
        _need(all(type(row[key]) is int for key in ('ordinal', 'lane', 'cpu', 'tid'))
              and row['ordinal'] == ordinal and row['lane'] == lane and row['cpu'] == cpus[lane]
              and row['tid'] > 0, f'native task {ordinal} used an unplanned core or logical lane')
        if ordinal < width:
            tids.append(row['tid'])
        else:
            _need(row['tid'] == tids[lane], 'logical lane changed Linux thread during the task team')
    if mpi:
        publisher = detail['publisher_pid']
        _need(tids == observation['mpi_rank_pids'] and sorted(tids) == active_processes
              and type(publisher) is int and publisher == tids[0],
              'MPI rank routing or rank zero publisher differs from the visible report')
        return dict(rank_pids=tids, cpu_ids=cpus, observed_tasks=len(tasks), publisher_pid=publisher)
    _need(len(set(tids)) == width and tids[0] == active_processes[0],
          'OpenMP task lanes did not run on distinct threads of the owning process')
    return dict(linux_tids=tids, cpu_ids=cpus, observed_tasks=len(tasks))


def _diagnostics(package, recipe, case, expected, params, pid, out, label, state_ref, original_states):
    placement = _PLACEMENT_REPORTS.get(package['decision']['mode'])
    if original_states is None and placement is None:
        return {}
    directory = out / ('state-diagnostic' if original_states is not None else 'lane-diagnostic')
    override = package.get('launch_environment', {})
    environment = dict(recipe['environment'], **override)
    if original_states is not None:
        environment['SCAR_ARRAY_ENTRY_STATES'] = str(directory / 'state')
    if placement:
        lane_report = directory / placement[0]
        environment['SCAR_ARRAY_ENTRY_LANE_REPORT'] = str(lane_report)
    run = _run(_commands(package, recipe)['native'], case['argv'], environment, directory, 0, label)
    if override:
        run['launch_environment'] = override
    _, observations = _validate_runs([run], package, recipe, case, expected, params, pid, native_label=label)
    diagnostics = {}
    if original_states is not None:
        checkpoints = [_ref(directory / f'state.{ordinal}.{index}.bin', 'scar.ndarray-native-checkpoint')
                       for ordinal, index, _ in original_states]
        _equal_states(original_states, checkpoints)
        diagnostics['state_diagnostic'] = dict(oracle=state_ref, run=run, checkpoints=checkpoints,
            complete_state_count=len(checkpoints), exact_bytes=sum(row['size_bytes'] for row in checkpoints))
    if placement:
        ref = _ref(lane_report, placement[1])
        assignments = _lane_assignments(ref, package, observations[0])
        diagnostics['lane_diagnostic'] = dict(run=run, sidecar=ref, assignments=assignments)
    return diagnostics


def _load_control(path, package, recipe, prior):
    reference, alternate, _, _, _, _ = _load(path, checked_source=(package, prior, recipe))
    _need(all(alternate[key] == package[key] for key in (
              'native', 'family', 'schedule', 'scope', 'recipe', 'cpu_ids', 'input_options', 'output_fields',
              'numerical_body_reused'))
          and alternate['decision']['mode'] != package['decision']['mode'],
          'validation control must share the numerical body and change execution mode only')
    return _CONTROL_LABELS[alternate['decision']['mode']], reference, alternate


def _pair(checked, incumbent, challenger, warmups, repetitions, *, archived=None):
    context = dict(incumbent=incumbent, challenger=challenger, warmup_pairs=warmups, measurement_pairs=repetitions)
    return (compare_entry_pairs(checked, **context) if archived is None
            else replay_entry_pairs(checked, archived, **context))


def calibrate(package_path, holdout_path, out, *, state_oracle=None, control_paths=(), warmups=1, repetitions=3):
    _need(type(warmups) is int and warmups >= 1 and type(repetitions) is int and repetitions >= 3,
          'at least one warmup pair and three clean pairs required')
    reference, package, recipe, prior, params, pid = _load(package_path)
    heldout = _ref(holdout_path, _ORIGINAL)
    case, expected = _holdout(heldout, package, recipe, prior, params, pid)
    commands = _commands(package, recipe)
    overrides = {'original': {}, 'native': package.get('launch_environment', {})}
    controls = {}
    for path in control_paths:
        label, control_ref, alternate = _load_control(path, package, recipe, prior)
        _need(label not in controls, 'duplicate alternate mode')
        controls[label] = (control_ref, alternate)
        commands[label] = _commands(alternate, recipe)['native']
        overrides[label] = alternate.get('launch_environment', {})
    clean_env = recipe['environment']
    _need(not {'SCAR_ARRAY_ENTRY_STATES', 'SCAR_ARRAY_ENTRY_LANE_REPORT'} & clean_env.keys(),
          'diagnostic output requested in the clean execution environment')
    state_reference = _ref(state_oracle, _STATES) if state_oracle is not None else None
    original_states = _state_oracle(state_reference, heldout, case, package, recipe) if state_reference else None
    out = Path(out).resolve()
    out.mkdir(parents=True)
    native_runs, control_runs = [], {label: [] for label in controls}
    for pair in range(warmups + repetitions):
        modes = tuple(commands)
        order = modes if pair % 2 == 0 else reversed(modes)
        for mode in order:
            rows = control_runs.get(mode, native_runs)
            override = overrides[mode]
            run = _run(commands[mode], case['argv'], dict(clean_env, **override), out / f'{pair}-{mode}', pair, mode)
            if override:
                run['launch_environment'] = override
            rows.append(run)
    checked, pids = _validate_runs(native_runs, package, recipe, case, expected, params, pid)
    pairing = _pair(checked, 'original', 'native', warmups, repetitions)
    report = dict(schema=SCHEMA, schema_version=1, entry=reference, holdout=heldout,
                  schedule=package['schedule'], selected=package['decision'], cpu_ids=package['cpu_ids'],
                  clean_pairs=native_runs, verified_processes=pids, parameters=case['parameters'], pairing=pairing,
                  population_variance_ns2={name: pvariance(row['samples_ns'])
                                           for name, row in pairing['metrics'].items()},
                  normalized_only=recipe['runtime_fields'],
                  timing_scope=_TIMING_SCOPE,
                  instrumentation_in_timing=False, compiled_from_holdout=False, deployment_authorized=False)
    report.update(_diagnostics(package, recipe, case, expected, params, pid, out, 'native',
                               state_reference, original_states))
    if controls:
        report['validation_controls'] = {}
    for label, (control_ref, alternate) in controls.items():
        observed, identities = _validate_runs(control_runs[label], alternate, recipe, case, expected,
                                              params, pid, native_label=label)
        compared = [row for row in checked if row['mode'] == 'native'] + observed
        comparison = _pair(compared, 'native', label, warmups, repetitions)
        row = dict(entry=control_ref, selected=alternate['decision'], clean_pairs=control_runs[label],
                   verified_processes=identities, pairing=comparison,
                   selected_vs_control=_pair(compared, label, 'native', warmups, repetitions),
                   deployment_authorized=False)
        if state_reference is not None or alternate['decision']['mode'] in _PLACEMENT_REPORTS:
            directory = out / f'{label}-diagnostics'
            directory.mkdir()
            row.update(_diagnostics(alternate, recipe, case, expected, params, pid, directory, label,
                                    state_reference, original_states))
        report['validation_controls'][label] = row
    _write(out / 'result.json', report)
    return report


def _check_diagnostics(report, holdout, package, recipe, case, expected, params, pid, label):
    lane = report.get('lane_diagnostic')
    placement = _PLACEMENT_REPORTS.get(package['decision']['mode'])
    _need(bool(lane) == bool(placement), 'parallel launch requires a retained placement diagnostic')
    clean_outputs = {row['stdout']['uri'] for row in report['clean_pairs']}
    if lane:
        run = lane['run']
        _need(run['stdout']['uri'] not in clean_outputs
              and Path(lane['sidecar']['uri']) == Path(run['stdout']['uri']).with_name(placement[0]),
              'placement evidence must come from a separate diagnostic invocation')
        _, observations = _validate_runs([run], package, recipe, case, expected, params, pid, native_label=label)
        _need(lane['assignments'] == _lane_assignments(lane['sidecar'], package, observations[0]),
              'retained placement replay disagrees with recorded lanes')
    if diagnostic := report.get('state_diagnostic'):
        _need(diagnostic['run']['stdout']['uri'] not in clean_outputs and (not lane or diagnostic['run'] == lane['run']),
              'diagnostic states must share the separate lane observation when present')
        original_states = _state_oracle(diagnostic['oracle'], holdout, case, package, recipe)
        checkpoints = diagnostic['checkpoints']
        _validate_runs([diagnostic['run']], package, recipe, case, expected, params, pid, native_label=label)
        _equal_states(original_states, checkpoints)
        _need(len(checkpoints) == diagnostic['complete_state_count']
              and diagnostic['exact_bytes'] == sum(row['size_bytes'] for row in checkpoints),
              'diagnostic state accounting differs')


def check(path):
    return _check(path)


def _check(path, *, already_replayed_schedule=None):
    stored = read_artifact(_ref(path, SCHEMA))
    reference, package, recipe, prior, params, pid = _load(
        stored['entry']['uri'], already_replayed_schedule=already_replayed_schedule)
    case, expected = _holdout(stored['holdout'], package, recipe, prior, params, pid)
    checked, processes = _validate_runs(stored['clean_pairs'], package, recipe, case, expected, params, pid)
    previous = stored['pairing']
    warmups, repetitions = previous['warmup_pairs'], previous['measurement_pairs']
    pairing = _pair(checked, 'original', 'native', warmups, repetitions, archived=previous)
    _need(stored['entry'] == reference and stored['schedule'] == package['schedule']
          and stored['selected'] == package['decision'] and stored['cpu_ids'] == package['cpu_ids']
          and stored['parameters'] == case['parameters'] and stored['normalized_only'] == recipe['runtime_fields']
          and stored['verified_processes'] == processes and stored['pairing'] == pairing
          and stored['timing_scope'] == _TIMING_SCOPE and stored['instrumentation_in_timing'] is False
          and stored['compiled_from_holdout'] is False and warmups >= 1
          and stored['deployment_authorized'] is False
          and stored['population_variance_ns2'] == {name: pvariance(row['samples_ns'])
                                                  for name, row in pairing['metrics'].items()},
          'stored scheduling, output or timing facts differ on cold replay')
    _check_diagnostics(stored, stored['holdout'], package, recipe, case, expected, params, pid, 'native')
    for label, row in stored.get('validation_controls', {}).items():
        actual_label, entry, alternate = _load_control(row['entry']['uri'], package, recipe, prior)
        alternate_runs, identities = _validate_runs(row['clean_pairs'], alternate, recipe, case, expected,
                                                    params, pid, native_label=label)
        compared = [item for item in checked if item['mode'] == 'native'] + alternate_runs
        comparison = _pair(compared, 'native', label, warmups, repetitions, archived=row['pairing'])
        _need(actual_label == label and entry == row['entry'] and alternate['decision'] == row['selected']
              and identities == row['verified_processes'] and comparison == row['pairing']
              and row['deployment_authorized'] is False, 'validation control replay differs')
        if 'selected_vs_control' in row:
            _need(row['selected_vs_control'] == _pair(compared, label, 'native', warmups, repetitions,
                                                       archived=row['selected_vs_control']),
                  'selected backend did not reproduce its control comparison')
        _check_diagnostics(row, stored['holdout'], alternate, recipe, case, expected, params, pid, label)
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    entry = parser.add_mutually_exclusive_group(required=True)
    entry.add_argument('--package')
    entry.add_argument('--check')
    parser.add_argument('--holdout')
    parser.add_argument('--state-oracle')
    parser.add_argument('--control', action='append', default=[],
                        help='calibrate another feasible compiled mode from the same schedule')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check and not any((args.holdout, args.state_oracle, args.out, args.control)):
        result = check(args.check)
    elif args.package and args.holdout and args.out:
        result = calibrate(args.package, args.holdout, args.out, state_oracle=args.state_oracle,
                           control_paths=args.control)
    else:
        parser.error('use --check RESULT or --package FILE --holdout FILE --out DIRECTORY')
    output = dict(selected=result['selected'], metrics=result['pairing']['metrics'])
    if controls := result.get('validation_controls'):
        output['validation_controls'] = {key: row['pairing']['metrics'][key] for key, row in controls.items()}
    print(json.dumps(output, sort_keys=True))


if __name__ == '__main__':
    main()

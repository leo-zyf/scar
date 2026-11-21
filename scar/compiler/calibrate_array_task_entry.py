"""Qualify an ordered native task entry against independent original process reports."""

import argparse
import json
from pathlib import Path
from statistics import pvariance

from scar.compiler.calibrate_array_entry import _normalize, _ref, _replay, _run, _write
from scar.ir.execution_recipe import read_artifact
from scar.optimization.paired_entry_selection import compare_entry_pairs
from scar.trace.numpy_partition_entry_validation import normalize_report


SCHEMA = 'scar.native-array-task-entry-calibration'
_ENTRY = 'scar.native-seeded-array-task-entry'
_ORIGINAL = 'scar.array-entry-task-family-holdout'
_STATES = 'scar.array-entry-task-state-oracle'
_OUTPUT_FIELDS = {'results', 'pid_list', 'ordinal', 'effective_seed', 'worker_round_count'}
_INPUT_OPTIONS = {'task_count', 'worker_limit', 'seed', 'round_count'}


def _need(condition, detail):
    if not condition:
        raise ValueError('array task entry calibration: ' + detail)


def _load(path):
    from scar.codegen.native_seeded_array_entry import LOOP_SCHEMA
    from scar.compiler.calibrate_seeded_array_entry import _load as load_seeded
    from scar.optimization.independent_array_task_schedule import check as check_schedule

    reference = _ref(path, _ENTRY)
    package = read_artifact(reference)
    serial, native, recipe = load_seeded(package['native']['uri'])
    schedule = check_schedule(package['schedule']['uri'])
    selected = schedule['decision']['selected']
    _need(serial == package['native'] and native['recipe'] == package['recipe']
          and native['schema'] == LOOP_SCHEMA
          and package['numerical_body_reused'] == native['files']['src/profile_0.cpp']
          and read_artifact(package['schedule']) == schedule
          and schedule['native'] == serial and schedule['family'] == package['family']
          and schedule['scope'] == package['scope'] and selected is not None and selected == package['decision']
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
    self_pid, = (field['path'] for field in recipe['report_bindings'] if field['role'] == 'pid')
    pids, = (field['path'] for field in recipe['report_bindings'] if field['role'] == 'pid_list')
    _need(self_pid[:2] == [fields['results'], 0] and pids == [fields['pid_list']]
          and selected['worker_count'] <= package['scope']['task_count']
          and selected['mode'] in ('IN_PROCESS_SERIAL', 'ORDERED_LOCAL_PROCESSES'),
          'source process reporting or selected task ownership differs')
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
    evidence = read_artifact(read_artifact(package['schedule'])['evidence'])
    related_seeds = source_seeds | set(evidence['seeds'])
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
          and all(seed + ordinal not in related_seeds for ordinal in range(count)),
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
                native=[*launcher, package['files'][package['binary']]['uri']])


def _validate_runs(runs, package, recipe, case, expected, parameters_field, pid_field):
    commands = _commands(package, recipe)
    checked, pids = [], []
    for run in runs:
        mode = run['mode']
        _need(mode in commands and run['command'] == commands[mode] and run['argv'] == case['argv'],
              'paired process command or original arguments differ')
        checked.append(_replay(run, recipe, expected))
        # In a fresh original run, the process pool may leave a late process idle.
        workers = ((package['decision']['worker_count'],) if mode == 'native' else
                   range(1, package['scope']['requested_workers'] + 1))
        tasks, active = _report(read_artifact(run['stdout'], json_document=False), package,
                                parameters_field, pid_field, case['parameters'], workers)
        if mode == 'native':
            width = package['decision']['worker_count']
            lanes = [row[pid_field] for row in tasks[:width]]
            _need(len(set(lanes)) == width and all(row[pid_field] == lanes[index % width]
                                                  for index, row in enumerate(tasks)),
                  'native ordinal-to-process routing differs from the selected plan')
        pids.append(dict(pair=run['pair'], mode=mode, active_worker_pids=active))
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


def calibrate(package_path, holdout_path, out, *, state_oracle=None, warmups=1, repetitions=3):
    _need(type(warmups) is int and warmups >= 1 and type(repetitions) is int and repetitions >= 3,
          'at least one warmup pair and three clean pairs required')
    reference, package, recipe, prior, params, pid = _load(package_path)
    heldout = _ref(holdout_path, _ORIGINAL)
    case, expected = _holdout(heldout, package, recipe, prior, params, pid)
    commands = _commands(package, recipe)
    state_reference = _ref(state_oracle, _STATES) if state_oracle is not None else None
    original_states = _state_oracle(state_reference, heldout, case, package, recipe) if state_reference else None
    out = Path(out).resolve()
    out.mkdir(parents=True)
    runs = []
    for pair in range(warmups + repetitions):
        order = ('original', 'native') if pair % 2 == 0 else ('native', 'original')
        for mode in order:
            runs.append(_run(commands[mode], case['argv'], recipe['environment'],
                             out / f'{pair}-{mode}', pair, mode))
    checked, pids = _validate_runs(runs, package, recipe, case, expected, params, pid)
    pairing = compare_entry_pairs(checked, incumbent='original', challenger='native',
                                  warmup_pairs=warmups, measurement_pairs=repetitions)
    report = dict(schema=SCHEMA, schema_version=1, entry=reference, holdout=heldout,
                  schedule=package['schedule'], selected=package['decision'], cpu_ids=package['cpu_ids'],
                  clean_pairs=runs, verified_processes=pids, parameters=case['parameters'], pairing=pairing,
                  population_variance_ns2={name: pvariance(row['samples_ns'])
                                           for name, row in pairing['metrics'].items()},
                  normalized_only=recipe['runtime_fields'],
                  timing_scope='FRESH_EXEC_THROUGH_EXIT_INCLUDES_ORDERED_STDOUT_AND_FILE_PUBLICATION',
                  instrumentation_in_timing=False, compiled_from_holdout=False, deployment_authorized=False)
    if original_states is not None:
        directory = out / 'state-diagnostic'
        prefix = directory / 'state'
        observed = _run(commands['native'], case['argv'], dict(recipe['environment'], SCAR_ARRAY_ENTRY_STATES=str(prefix)),
                        directory, 0, 'native')
        _validate_runs([observed], package, recipe, case, expected, params, pid)
        checkpoints = [_ref(directory / f'state.{ordinal}.{index}.bin', 'scar.ndarray-native-checkpoint')
                       for ordinal, index, _ in original_states]
        _equal_states(original_states, checkpoints)
        report['state_diagnostic'] = dict(oracle=state_reference, run=observed, checkpoints=checkpoints,
            complete_state_count=len(checkpoints), exact_bytes=sum(row['size_bytes'] for row in checkpoints))
    _write(out / 'result.json', report)
    return report


def check(path):
    stored = read_artifact(_ref(path, SCHEMA))
    reference, package, recipe, prior, params, pid = _load(stored['entry']['uri'])
    case, expected = _holdout(stored['holdout'], package, recipe, prior, params, pid)
    checked, processes = _validate_runs(stored['clean_pairs'], package, recipe, case, expected, params, pid)
    previous = stored['pairing']
    pairing = compare_entry_pairs(checked, incumbent='original', challenger='native',
                                  warmup_pairs=previous['warmup_pairs'], measurement_pairs=previous['measurement_pairs'])
    _need(stored['entry'] == reference and stored['schedule'] == package['schedule']
          and stored['selected'] == package['decision'] and stored['cpu_ids'] == package['cpu_ids']
          and stored['parameters'] == case['parameters'] and stored['normalized_only'] == recipe['runtime_fields']
          and stored['verified_processes'] == processes and stored['pairing'] == pairing
          and stored['deployment_authorized'] is False
          and stored['population_variance_ns2'] == {name: pvariance(row['samples_ns'])
                                                  for name, row in pairing['metrics'].items()},
          'stored scheduling, output or timing facts differ on cold replay')
    if diagnostic := stored.get('state_diagnostic'):
        original_states = _state_oracle(diagnostic['oracle'], stored['holdout'], case, package, recipe)
        checkpoints = diagnostic['checkpoints']
        _validate_runs([diagnostic['run']], package, recipe, case, expected, params, pid)
        _equal_states(original_states, checkpoints)
        _need(len(checkpoints) == diagnostic['complete_state_count']
              and diagnostic['exact_bytes'] == sum(row['size_bytes'] for row in checkpoints),
              'diagnostic state accounting differs')
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    entry = parser.add_mutually_exclusive_group(required=True)
    entry.add_argument('--package')
    entry.add_argument('--check')
    parser.add_argument('--holdout')
    parser.add_argument('--state-oracle')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check and not any((args.holdout, args.state_oracle, args.out)):
        result = check(args.check)
    elif args.package and args.holdout and args.out:
        result = calibrate(args.package, args.holdout, args.out, state_oracle=args.state_oracle)
    else:
        parser.error('use --check RESULT or --package FILE --holdout FILE --out DIRECTORY')
    print(json.dumps(dict(selected=result['selected'], metrics=result['pairing']['metrics']), sort_keys=True))


if __name__ == '__main__':
    main()

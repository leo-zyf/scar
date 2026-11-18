"""Freeze and validate a full-process cost interpolation at one exact argv."""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

from scar.compiler.stateful_holdout import _write
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.compiled_deployment import thread_environment

DESIGN = 'scar.snapshot-entry-cost-design'
TRAIN = 'scar.snapshot-entry-cost-training'
MODEL = 'scar.snapshot-entry-cost-model'
TICKET = 'scar.snapshot-entry-cost-holdout-ticket'
HOLDOUT = 'scar.snapshot-entry-cost-holdout'
SCHEMA = 'scar.snapshot-entry-cost-qualification'
WORKER_DESIGN = 'scar.borrowed-snapshot-long-entry-design'
CANDIDATE = 'scar.compiled-workspace-borrowed-roots'
OUTPUT = '{output}'
MODES = ('original', 'borrowed')
POINTS = ('short', 'long', 'holdout')
PAIR_COUNT = 4
_ENV_KEYS = ('MPLBACKEND', 'MPLCONFIGDIR', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD',
    'PYTHONHASHSEED', 'PYTHONOPTIMIZE', 'PYTHONUSERBASE', 'LC_ALL', 'LANG', 'TZ',
    'GLIBC_TUNABLES', 'MALLOC_ARENA_MAX', 'OMP_PROC_BIND', 'OMP_PLACES')


def _need(condition, detail):
    if not condition:
        raise ValueError(detail)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _ref(value, schema):
    return value if type(value) is dict else artifact_reference(Path(value).resolve(), schema)


def _read(ref, schema):
    saved = read_artifact(ref)
    _need(ref['schema'] == schema and saved['schema_version'] == 1, 'expected versioned calibration receipt')
    return saved


def environment():
    return dict(python_executable=sys.executable, python_version=sys.version, cwd=os.getcwd(),
        variables={key:os.environ.get(key) for key in _ENV_KEYS},
        native_thread_environment=thread_environment())


def _coordinate(argv):
    _need(set(argv) == set(POINTS), 'two endpoint argv lists and one interior argv list required')
    columns = list(zip(*(argv[key] for key in POINTS), strict=True))
    changed = [index for index, values in enumerate(columns) if len(set(values)) > 1]
    _need(len(changed) == 1 and all(type(arg) is str for row in argv.values() for arg in row)
        and all(row.count(OUTPUT) == 1 for row in argv.values()),
        'exact command family with one varying token and one output placeholder required')
    position, = changed
    values = {key:Fraction(argv[key][position]) for key in POINTS}
    _need(0 < values['short'] < values['holdout'] < values['long'], 'holdout must be strictly inside trained endpoints')
    return position, values


def prepare(candidate, previous_validation, runner, argv, out):
    from scar.compiler.compile_workspace_borrowed_roots import check

    _coordinate(argv)
    candidate_ref = _ref(candidate, CANDIDATE)
    compiled = check(candidate_ref['uri'])
    _need(compiled == _read(candidate_ref, CANDIDATE), 'candidate source replay differs')
    validation_ref = _ref(previous_validation, 'scar.borrowed-snapshot-long-validation')
    validation = _read(validation_ref, validation_ref['schema'])
    previous = _read(validation['design'], WORKER_DESIGN)
    runner_ref = _ref(runner, 'scar.python-source')
    _need(validation['validated'] is True
        and validation['modes']['borrowed']['guard']['fallback_calls'] == 0
        and previous['bundles']['borrowed']['scope_admission'] == compiled['scope_admission']
        and runner_ref == previous['runner']
        and previous['argv'] not in argv.values(), 'same previously validated artifact and fresh argv required')
    output = Path(out).resolve()
    output.mkdir(parents=True)
    scope = compiled['scope_admission']
    common = dict(entry=previous['entry'], candidate=candidate_ref, scope_admission=scope,
        output_parent_device=output.stat().st_dev, output_token=OUTPUT, environment=environment(),
        caller_cpu_ids=sorted(os.sched_getaffinity(0)))
    formats = {}
    for point in POINTS:
        design = dict(schema=WORKER_DESIGN, schema_version=1, entry=common['entry'], argv=argv[point],
            caller_cpu_ids=common['caller_cpu_ids'], python=platform.python_version(), runner=runner_ref,
            bundles={'borrowed':{'bundle':compiled['bundle'], 'scope_admission':scope}})
        formats[point] = _write(output/(point+'-worker-design.json'), design)
    return _write(output/'design.json', dict(schema=DESIGN, schema_version=1,
        context=common, runner=runner_ref, previous_correctness=validation_ref,
        argv=argv, worker_designs=formats, pair_count=PAIR_COUNT,
        policy=dict(minimum_gain=0.05, max_relative_error=0.2),
        root=str(output), prepared_at=_now(), protocol='FRESH_PROCESS_LAUNCH_TO_EXIT_INCLUDING_STARTUP_AND_IO',
        implementation=_ref(__file__, 'scar.implementation-source')))


def _design(value):
    ref = _ref(value, DESIGN)
    saved = _read(ref, DESIGN)
    original = _read(saved['previous_correctness'], 'scar.borrowed-snapshot-long-validation')
    previous = _read(original['design'], WORKER_DESIGN)
    context = saved['context']
    compiled = _read(context['candidate'], CANDIDATE)
    _need(original['validated'] is True and previous['runner'] == saved['runner']
        and previous['entry'] == context['entry']
        and previous['argv'] not in saved['argv'].values()
        and previous['bundles']['borrowed']['scope_admission'] == compiled['scope_admission'] == context['scope_admission']
        and saved['implementation'] == _ref(__file__, 'scar.implementation-source')
        and saved['pair_count'] == PAIR_COUNT and saved['policy'] == dict(minimum_gain=0.05, max_relative_error=0.2),
        'frozen software or policy differs from the admitted source benchmark')
    _coordinate(saved['argv'])
    for member in (context['entry'], saved['runner'], compiled['library'], compiled['source']):
        read_artifact(member, json_document=False)
    from scar.runtime.session_scope_bundle import inspect_scope_bundle
    inspected = inspect_scope_bundle(compiled['bundle'], expected_admission_sha256=context['scope_admission']['sha256'])
    _need(inspected['admission']['snapshot_rewrite'] == compiled['native'],
        'compact bundle native identity differs from source freeze')
    for point in POINTS:
        worker = _read(saved['worker_designs'][point], WORKER_DESIGN)
        expected = dict(schema=WORKER_DESIGN, schema_version=1, entry=context['entry'], argv=saved['argv'][point],
            caller_cpu_ids=context['caller_cpu_ids'], python=platform.python_version(), runner=saved['runner'],
            bundles={'borrowed':{'bundle':compiled['bundle'], 'scope_admission':context['scope_admission']}})
        _need(worker == expected, 'worker argv or compiled candidate differs from frozen source')
    _need(Path(saved['root']).stat().st_dev == context['output_parent_device'], 'calibration output device changed')
    return ref, saved


def _live(design):
    _need(design['context']['environment'] == environment()
        and design['context']['caller_cpu_ids'] == sorted(os.sched_getaffinity(0)),
        'benchmark caller context differs from the frozen calibration')


def _command(design, point, mode, directory, observe):
    worker = design['worker_designs'][point]
    return [design['context']['environment']['python_executable'], '-B', design['runner']['uri'],
        worker['uri'], worker['sha256'], mode, str(directory), str(int(observe))]


def _files(directory):
    return {str(path.relative_to(directory)):{key:ref[key] for key in ('sha256', 'size_bytes')}
        for path in sorted(directory.rglob('*')) if path.is_file()
        for ref in [_ref(path, 'scar.benchmark-output')]}


def _sample(design, point, mode, directory, observe=False):
    command = _command(design, point, mode, directory, observe)
    directory.mkdir()
    with (directory/'stdout').open('xb') as stdout, (directory/'stderr').open('xb') as stderr:
        start = perf_counter_ns()
        process = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        stop = perf_counter_ns()
    _need(process.returncode == 0, 'frozen worker failed: '+str(directory/'stderr'))
    return dict(command=command, work_directory=str(directory), start_ns=start if not observe else None,
        stop_ns=stop if not observe else None, worker=_ref(directory/'worker.json', 'scar.snapshot-entry-worker'),
        outputs=_files(directory/'images'), stdout=_ref(directory/'stdout', 'scar.benchmark-output'),
        stderr=_ref(directory/'stderr', 'scar.benchmark-output'), exit_code=process.returncode)


def _output_identity(row):
    return ({key:(row[key]['sha256'], row[key]['size_bytes']) for key in ('stdout', 'stderr')}, row['outputs'])


def _worker(row, design, point, mode, directory, observe, expected_steps=None):
    _need(row['command'] == _command(design, point, mode, directory, observe)
        and row['work_directory'] == str(directory) and row['exit_code'] == 0
        and row['outputs'] == _files(directory/'images'), 'worker command or configured outputs differ')
    for key in ('stdout', 'stderr', 'worker'):
        expected = directory/(key if key != 'worker' else 'worker.json')
        _need(row[key]['uri'] == str(expected), 'worker receipt belongs to another invocation')
        read_artifact(row[key], json_document=False)
    result = json.loads(Path(row['worker']['uri']).read_bytes())
    _need(result['mode'] == mode and result['observer'] is observe, 'unexpected worker path')
    if observe:
        _need(row['start_ns'] is None and row['stop_ns'] is None and result['trajectory_sha256']
            and type(result['trajectory_steps']) is int and result['trajectory_steps'] > 0
            and result['final'] is not None, 'independent trajectory and final state required')
        expected_steps = result['trajectory_steps']
    else:
        _need(type(row['start_ns']) is int and type(row['stop_ns']) is int
            and 0 < row['start_ns'] < row['stop_ns'] and result['trajectory_steps'] == 0,
            'clean run must retain monotone wall-clock endpoints and no observer')
    if mode == 'borrowed':
        _need(result['guard'] == dict(native_calls=expected_steps, fallback_calls=0, reasons={})
            and result['stats']['resident_calls'] == 0, 'the exact dirty-step native path must succeed every time')
    return result


def _samples(design_ref, design, point, *, ticket=None):
    _live(design)
    root = Path(design['root'])/point
    root.mkdir()
    start = _now()
    validation = {mode:_sample(design, point, mode, root/('validate-'+mode), True) for mode in MODES}
    pairs = []
    for index in range(PAIR_COUNT):
        order = list(MODES if index % 2 == 0 else reversed(MODES))
        runs = {mode:_sample(design, point, mode, root/f'{index}-{mode}') for mode in order}
        pairs.append(dict(index=index, order=order, runs=runs))
    _live(design)
    ref = _write(root/'observations.json', dict(schema=HOLDOUT if ticket else TRAIN, schema_version=1,
        design=design_ref, point=point, ticket=ticket, validation=validation, pairs=pairs,
        observed_from=start, observed_until=_now()))
    return ref


def _point(observations, design_ref, design, point, *, ticket=None):
    data = _read(observations, HOLDOUT if ticket else TRAIN)
    _need(data['design'] == design_ref and data['point'] == point and data['ticket'] == ticket
        and data['observed_from'] < data['observed_until'] and len(data['pairs']) == PAIR_COUNT,
        'evidence is outside its frozen sampling phase')
    root = Path(design['root'])/point
    validation = data['validation']
    oracle, native = [_worker(validation[mode], design, point, mode, root/('validate-'+mode), True) for mode in MODES]
    _need(all(oracle[key] == native[key] for key in ('trajectory_steps', 'trajectory_sha256', 'final'))
        and _output_identity(validation['original']) == _output_identity(validation['borrowed']),
        'per-step trajectory, final state or external outputs differ')
    native_id = _read(_read(design['context']['candidate'], CANDIDATE)['native'],
        'scar.native-workspace-borrowed-roots')['execution_plan_sha256']
    steps = oracle['trajectory_steps']
    clocks = {mode:[] for mode in MODES}
    previous = 0
    for index, pair in enumerate(data['pairs']):
        order = list(MODES if index % 2 == 0 else reversed(MODES))
        _need(pair['index'] == index and pair['order'] == order, 'AB/BA pair order differs')
        for mode in order:
            row = pair['runs'][mode]
            worker = _worker(row, design, point, mode, root/f'{index}-{mode}', False, steps)
            _need(row['start_ns'] >= previous and _output_identity(row) == _output_identity(validation['original'])
                and (mode == 'original' or worker['kernel_execution_identity'] == native_id),
                'clean process overlaps another sample, differs from its output oracle, or ran another ELF')
            clocks[mode].append(row['stop_ns']-row['start_ns'])
            previous = row['stop_ns']
    return dict(steps=steps, clocks_ns=clocks,
        medians_ns={mode:median(map(Fraction, values)) for mode, values in clocks.items()},
        observed_from=data['observed_from'], observed_until=data['observed_until'])


def _fit(design_ref, design, short, long, fitted_at):
    a, b = [_point(ref, design_ref, design, point) for ref, point in ((short, 'short'), (long, 'long'))]
    _need(design['prepared_at'] < a['observed_from'] < a['observed_until'] < b['observed_from']
        < b['observed_until'] < fitted_at and 0 < a['steps'] < b['steps'],
        'independent ordered training observations required before fitting')
    index, x = _coordinate(design['argv'])
    weights = (x['holdout'] - x['short'])/(x['long'] - x['short'])
    predicted = {mode:round(a['medians_ns'][mode] + weights*(b['medians_ns'][mode]-a['medians_ns'][mode]))
        for mode in MODES}
    return dict(schema=MODEL, schema_version=1, design=design_ref, short=short, long=long,
        fitted_at=fitted_at, type='AFFINE_INTERPOLATION_OF_INDEPENDENT_ENDPOINT_MEDIANS',
        scalar_argv_index=index, training_steps=[a['steps'], b['steps']],
        prediction=dict(original_ns=predicted['original'], native_ns=predicted['borrowed']))


def train(value):
    design_ref, design = _design(value)
    short, long = [_samples(design_ref, design, point) for point in ('short', 'long')]
    model = _fit(design_ref, design, short, long, _now())
    return _write(Path(design['root'])/'model.json', model)


def _model(value):
    reference = _ref(value, MODEL)
    saved = _read(reference, MODEL)
    design_ref, design = _design(saved['design'])
    _need(saved == _fit(design_ref, design, saved['short'], saved['long'], saved['fitted_at']),
        'frozen model differs from the training endpoints')
    return reference, saved, design_ref, design


def _evaluation(model_path, model, design_ref, design, ticket, holdout):
    frozen = _read(ticket, TICKET)
    _need(frozen['model'] == model_path and frozen['design'] == design_ref
        and model['fitted_at'] < frozen['frozen_at'], 'holdout must bind the pre-frozen model')
    data = _point(holdout, design_ref, design, 'holdout', ticket=ticket)
    low, high = model['training_steps']
    _need(frozen['frozen_at'] < data['observed_from'] and low < data['steps'] < high,
        'holdout is not a new interior observation after the model freeze')
    prediction = model['prediction']
    original, native = (data['medians_ns'][mode] for mode in MODES)
    predicted_original, predicted_native = (prediction[key] for key in ('original_ns', 'native_ns'))
    expected_gain = Fraction(predicted_original-predicted_native, predicted_original)
    gain = (original-native)/original
    original_error = abs(predicted_original-original)/original
    native_error = abs(predicted_native-native)/native
    qualifies = (min(expected_gain, gain) >= Fraction(1, 20)
        and max(original_error, native_error) <= Fraction(1, 5))
    context = {**design['context'], 'argv':design['argv']['holdout']}
    return dict(schema=SCHEMA, schema_version=1, decision='LOAD_NATIVE_CANDIDATE' if qualifies else 'NO_OP',
        qualifies=qualifies, context=context, design=design_ref, model=model_path, holdout=holdout,
        holdout_ticket=ticket, prediction=prediction, verified_steps=data['steps'],
        evaluation=dict(predicted_gain=float(expected_gain), observed_gain=float(gain),
            original_error=float(original_error), native_error=float(native_error),
            original_ns=float(original), native_ns=float(native), raw_process_clocks_ns=data['clocks_ns']),
        acceptance_thresholds=design['policy'], scope=design['protocol'],
        default_backend='NO_OP', exact_verified_argv_only=True)


def holdout(value):
    model_ref, model, design_ref, design = _model(value)
    ticket = _write(Path(design['root'])/'holdout-ticket.json', dict(schema=TICKET, schema_version=1,
        model=model_ref, design=design_ref, frozen_at=_now()))
    sampled = _samples(design_ref, design, 'holdout', ticket=ticket)
    result = _evaluation(model_ref, model, design_ref, design, ticket, sampled)
    return _write(Path(design['root'])/'result.json', result)


def check(value):
    saved = _read(_ref(value, SCHEMA), SCHEMA)
    model_ref, model, design_ref, design = _model(saved['model'])
    replayed = _evaluation(model_ref, model, design_ref, design, saved['holdout_ticket'], saved['holdout'])
    _need(saved == replayed, 'complete entry qualification differs from independent clocks and source receipts')
    return replayed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('prepare')
    for name in ('candidate', 'previous-validation', 'runner', 'argv-json', 'out'):
        freeze.add_argument('--'+name, required=True)
    for name, key in (('train', 'design'), ('holdout', 'model'), ('check', 'result')):
        commands.add_parser(name).add_argument('--'+key, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare(args.candidate, args.previous_validation, args.runner, json.loads(args.argv_json), args.out)
    elif args.command == 'train':
        result = train(args.design)
    elif args.command == 'holdout':
        result = holdout(args.model)
    else:
        result = check(args.result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

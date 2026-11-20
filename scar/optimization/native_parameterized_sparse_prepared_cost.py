"""Choose prepared sparse ingress using fresh matched whole-process evidence."""

import argparse
import json
from pathlib import Path

from scar.analysis.replay_transaction import replay_transaction
from scar.ir.execution_recipe import read_artifact
from scar.optimization import native_parameterized_sparse_plan as sparse
from scar.optimization import scalar_sorted_transition_cost as cost
from scar.optimization.native_parameterized_sparse_width import _command, _context, _digest


BENCHMARK = 'scar.native-parameterized-sparse-prepared-benchmark'
SCHEMA = 'scar.native-parameterized-sparse-prepared-selection'
_ARMS = ('original', 'native', 'prepared')
_TASKSET = '/usr/bin/taskset'
_STREAM = 'application/octet-stream'


def _require(condition, detail):
    if not condition:
        raise ValueError('prepared sparse cost: ' + detail)


def _treatments(receipt):
    from scar.codegen import native_parameterized_sparse_prepared as prepared

    arms = receipt['arms']
    _require(set(arms) == set(_ARMS) and arms['original'] == {'source': receipt['source']}
        and all(set(arms[mode]) == {'artifact', 'executable'} for mode in ('native', 'prepared')),
        'expected direct original, runtime preparation, and prepared ingress arms')
    native_ref = arms['native']['artifact']
    artifact = read_artifact(native_ref)
    plan_ref = artifact['parameterized_plan']
    plan = read_artifact(plan_ref)
    sparse._native(arms['native'], plan, plan_ref)
    scope, resources = plan['scope'], plan['resources']
    _require(receipt['source'] == scope['source'] and receipt['cpu_ids'] == resources['placement']['launch_cpu_ids']
        and resources['width'] == resources['dispatch']['window'] == 6
        and plan['decision']['native_experiment_authorized'] is True,
        'baseline must bind the same admitted source and full W6 owner/worker cohort')
    read_artifact(receipt['source'], json_document=False)
    optimized = arms['prepared']
    ref = optimized['artifact']
    _require(ref['schema'] == prepared.SCHEMA, 'prepared arm requires the checked executable manifest')
    manifest = read_artifact(ref)
    _require(manifest == prepared.check(ref['uri']), 'prepared artifact differs from checked delivery')
    root = Path(ref['uri']).parent
    binary = sparse._member(root, manifest['binary'])
    baseline_solver = Path(native_ref['uri']).parent / 'solver'
    expected_solver = dict(artifact['solver'], uri='solver/artifact.json')
    native_solver = read_artifact(dict(expected_solver, uri=str(baseline_solver / 'artifact.json')))
    executable_ref = native_solver['stream']['executable']
    expected_executable = dict(executable_ref, uri=str(Path('solver') / executable_ref['uri']))
    _require(manifest['entry'] == native_ref and manifest['source'] == receipt['source'] and manifest['scope'] == scope
        and manifest['protocol'] == artifact['protocol'] and manifest['solver'] == expected_solver
        and manifest['solver_executable'] == expected_executable and optimized['executable'] == binary
        and manifest['cxx_standard'] == 20 and manifest['python_runtime_required'] is False
        and manifest['source_count_read_at_runtime'] is True and manifest['deployment_authorized'] is False
        and read_artifact(binary, json_document=False).startswith(b'\x7fELF'),
        'prepared and runtime preparation must share the exact source, solver bytes, count ABI and placement')
    return plan, manifest['plan']


def _launch_context(receipt, plan):
    interpreter = receipt['interpreter']
    _require(set(interpreter) == {'python_executable', 'python_version'}
        and type(interpreter['python_version']) is str and interpreter['python_version'].strip()
        and _context(plan) == (interpreter['python_executable'], receipt['cwd'], receipt['environment']),
        'original and C++ launches must share the qualified interpreter, working directory and environment')
    _require(receipt['launcher']['schema'] == 'scar.python-source', 'expected a retained benchmark launcher')
    read_artifact(receipt['launcher'], json_document=False)


def _timings(samples):
    measurements, source_comparison = cost._envelopes(samples, minimum_gain_bps=0)
    _, native_comparison = cost._envelopes(dict(original=samples['native'], prepared=samples['prepared']),
        minimum_gain_bps=0)
    treatment = native_comparison['prepared']
    treatment['paired_native_minus_prepared_ns'] = treatment.pop('paired_original_minus_candidate_ns')
    treatment['native_over_prepared_median'] = treatment.pop('original_over_candidate_median')
    return measurements, source_comparison, treatment


def _case(case, receipt, plan, observed_files):
    count, argv = case['count'], case['argv']
    work = sparse.work_for_count(plan, count)
    _require(argv == [str(count)], 'argv must bind the exact positive runtime count')
    rows = case['rows']
    _require(type(rows) is list and len(rows) >= 12 and len(rows) % len(_ARMS) == 0,
        'each count needs one warmup and at least three whole-process measured rounds')
    arms = receipt['arms']
    samples = {arm: [] for arm in _ARMS}
    original = [receipt['interpreter']['python_executable'], '-B', receipt['source']['uri'], *argv]
    commands = dict(original=original, **{arm: [arms[arm]['executable']['uri'], *argv] for arm in ('native', 'prepared')})
    orders, launchers, digests = [], set(), set()
    for index, row in enumerate(rows):
        round_, slot = divmod(index, len(_ARMS))
        shift = round_ % len(_ARMS)
        order = (*_ARMS[shift:], *_ARMS[:shift])
        if slot == 0:
            orders.append(list(order))
        arm = order[slot]
        expected = 'OBSERVED_WARMUP' if not round_ else 'CLEAN_MEASURED'
        _require(type(row['round']) is int and row['round'] == round_ and row['arm'] == arm
            and row['kind'] == expected and type(row['wall_ns']) is int and row['wall_ns'] > 0,
            'trial did not follow the rotating, paired warmup/clean measurement layout')
        launch, command = _command(row['command'], receipt, argv)
        _require(launch[0] == _TASKSET and command == commands[arm],
            'trial did not launch the checked direct source or exact ELF')
        launchers.add(launch)
        for descriptor in (row['stdout'], row['stderr']):
            path = Path(descriptor['uri'])
            _require(descriptor['schema'] == _STREAM and path == path.resolve(strict=True)
                and path not in observed_files, 'each original or native result needs its own output files')
            observed_files.add(path)
        digests.add(_digest(row))
        if round_:
            samples[arm].append(row['wall_ns'])
    _require(len(digests) == len(launchers) == 1, 'trial output or exact launcher changed across rounds or arms')
    measurements, source_comparison, prepared_gain = _timings(samples)
    profitable = prepared_gain['verified_observed_gain']
    return dict(count=count, argv=argv, work=work, orders=orders, warmup_rounds=1,
        clean_rounds=len(samples['original']), exact_stdout_hex=digests.pop().decode('ascii').rstrip('\n'),
        measurements=measurements, native_vs_prepared=prepared_gain, original_vs_prepared=source_comparison['prepared'],
        original_vs_native=source_comparison['native'], selected='prepared' if profitable else 'native',
        observed_preparation_gain=profitable)


@replay_transaction()
def select(path):
    """Select only counts where every fresh pair and strict envelope improve."""
    benchmark = sparse._ref(path, BENCHMARK)
    receipt = read_artifact(benchmark)
    _require(type(receipt['schema_version']) is int and receipt['schema_version'] == 1,
        'expected version 1 three-arm benchmark receipt')
    plan, preparation_ref = _treatments(receipt)
    _launch_context(receipt, plan)
    observed_files = set()
    cases = [_case(case, receipt, plan, observed_files) for case in receipt['cases']]
    choices = {str(case['count']): case['selected'] for case in cases}
    _require(cases and len(choices) == len(cases), 'expected distinct actually measured runtime counts')
    scope = plan['scope']
    return dict(schema=SCHEMA, schema_version=1, benchmark=benchmark, preparation_plan=preparation_ref,
        source=receipt['source'], arms=receipt['arms'], cpu_ids=receipt['cpu_ids'],
        minimum_count=scope['minimum_count'], maximum_count=scope['maximum_count'],
        cases=cases, exact_count_to_arm=choices, default_arm='native',
        selection_scope='EXACT_MEASURED_COUNTS_ON_ADMITTED_W6_COHORT',
        timing_scope='FRESH_WHOLE_PROCESS_WALL_CLOCK_WITHOUT_DIAGNOSTIC_INSTRUMENTATION',
        deployment_authorized=False, historical_costs_used_for_selection=False, globally_optimal=False)


def check(path):
    stored = read_artifact(sparse._ref(path, SCHEMA))
    rebuilt = select(stored['benchmark']['uri'])
    _require(sparse._canonical(stored) == sparse._canonical(rebuilt), 'selection differs from cold replay')
    return rebuilt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('benchmark', 'check', 'out'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    if args.check and not any((args.benchmark, args.out)):
        result = check(args.check)
    elif args.benchmark and args.out and args.check is None:
        result = select(args.benchmark)
        with args.out.open('x', encoding='utf8') as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    else:
        parser.error('provide --check or --benchmark and --out')
    print(sparse._canonical({key: result[key] for key in ('exact_count_to_arm', 'default_arm')}))


if __name__ == '__main__':
    main()

"""Choose a guarded search region using matched whole-entry evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import median, stdev

from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.scalar-sorted-transition-cost'
HISTORY = SCHEMA + '-historical'
BENCHMARK = 'scar.scalar-sorted-transition-paired-run'
NORMALIZATION = 'UCI_INFO_TIME_AND_NPS_ONLY'
_TIMING = re.compile(rb'\s+(?:time|nps) \d+')
_DEPTH = re.compile(rb'^info depth ([1-9][0-9]*)\b')
_OLD = {
    'scar.scalar-sorted-search-benchmark': 'scar.scalar-sorted-source-study',
    'scar.scalar-transition-search-benchmark': 'scar.scalar-transition-source-study',
}
_REGION = {
    'sort_only': 'scar.scalar-sorted-source-study',
    'fused': 'scar.scalar-sorted-transition-source-study',
}
_AUX_COUNTERS = {'validated_mutation_epochs', 'shadow_verified'}


def _require(valid, reason):
    if not valid:
        raise ValueError(reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _raw_json(ref):
    return json.loads(read_artifact(ref, json_document=False))


def canonical_stdout(raw):
    return b'\n'.join(_TIMING.sub(b'', line) if line.startswith(b'info depth ') else line
                      for line in raw.splitlines())


def _request_count(commands, depth):
    _require(type(depth) is int and depth > 0 and type(commands) is list
        and all(type(line) is str and line.isascii() and line and '\n' not in line and '\r' not in line
                for line in commands)
        and commands[-1:] == ['quit'], 'expected the exact finite UCI command stream')
    searches = [line for line in commands if line.startswith('go ')]
    _require(searches and all(line == f'go depth {depth}' for line in searches),
        'each search request must bind the declared depth')
    return len(searches)


def _output(raw, depth, requests):
    moves = []
    depths = []
    for line in raw.splitlines():
        match = _DEPTH.match(line)
        if match:
            depths.append(int(match[1]))
        if line.startswith(b'bestmove '):
            _require(depths and max(depths) == depth, 'search output did not reach the requested depth')
            moves.append(line.decode('ascii'))
            depths.clear()
    _require(len(moves) == requests and not depths, 'search output did not complete every request')
    exact = canonical_stdout(raw)
    return sha256(exact).hexdigest(), moves


def _summary(samples):
    return dict(samples_ns=samples, median_ns=median(samples), min_ns=min(samples), max_ns=max(samples),
                sample_stdev_ns=stdev(samples))


def _envelopes(samples, minimum_gain_bps):
    _require(type(minimum_gain_bps) is int and 0 <= minimum_gain_bps < 10000,
        'invalid observed timing separation threshold')
    result = {mode: _summary(values) for mode,values in samples.items()}
    original = result['original']
    comparisons = {}
    for mode, row in result.items():
        if mode == 'original':
            continue
        diff = [a-b for a,b in zip(samples['original'], row['samples_ns'], strict=True)]
        gap = original['min_ns']-row['max_ns']
        qualifies = gap > 0 and all(value > 0 for value in diff)
        qualifies &= gap*10000 >= original['min_ns']*minimum_gain_bps
        comparisons[mode] = dict(paired_original_minus_candidate_ns=diff,
            paired_delta_median_ns=median(diff), paired_delta_range_ns=[min(diff),max(diff)],
            original_over_candidate_median=original['median_ns']/row['median_ns'],
            observed_worst_separation_ns=gap, verified_observed_gain=qualifies)
    return result, comparisons


def _groups(rows, modes, warmups, measured, *, key):
    _require(type(warmups) is int and warmups >= 1 and type(measured) is int and measured >= 3,
        'search acceptance needs at least one warmup and three clean paired rounds')
    expected = list(range(warmups+measured))
    sequence = [row['round'] for row in rows]
    counts = Counter((row['round'], row[key]) for row in rows)
    _require(sequence == sorted(sequence)
        and counts == Counter((index, mode) for index in expected for mode in modes),
        'each warmup and measured block must include each arm exactly once')
    orders = [[row[key] for row in rows if row['round'] == index] for index in expected]
    _require(len({tuple(row) for row in orders[warmups:]}) > 1, 'clean arms must vary launch order across blocks')
    return orders


def _call_cost(counts, phase_ns, successful=('native',)):
    _require(type(counts) is dict and type(phase_ns) is dict and
        all(type(value) is int and value >= 0 for value in (*counts.values(), *phase_ns.values())),
        'expected an instrumented guarded operation with timing and counters')
    completed = {key:counts.get(key, 0) for key in successful}
    native = sum(completed.values())
    _require(bool(native) == bool(phase_ns), 'native phase costs must cover actual successful calls')
    fallback = {key:value for key,value in counts.items() if key not in successful and key not in _AUX_COUNTERS}
    return dict(native_calls=native, fallback_calls=sum(fallback.values()), fallback_reasons=fallback,
        observed_attempts=native+sum(fallback.values()), native_status_counts=completed, total_success_phase_ns=phase_ns,
        mean_instrumented_success_phase_ns={key:value/native for key,value in phase_ns.items()},
        timed_fallback_phase_ns=None, clean_call_frequency_proven=False)


def _legacy_search(path, input_ref, driver_path, source_ref):
    schema = json.loads(Path(path).read_bytes())['schema']
    _require(schema in _OLD, 'unsupported historical native boundary study')
    benchmark_ref = _ref(path, schema)
    value = read_artifact(benchmark_ref)
    driver = _ref(driver_path, 'scar.python-source')
    script = read_artifact(driver, json_document=False)
    # Both archived drivers load the same captured command transcript and
    # substitute only the requested search depth. Source launch is external.
    archive_dir = '/'.join(Path(input_ref['uri']).parts[-3:-1])
    _require(f"source_run = experiment/'{archive_dir}'".encode() in script
        and b"uci = json.loads((source_run/'run.json').read_bytes())['stdin']" in script
        and b"request=f'go depth {depth}'" in script
        and b"args.cpu,i==0" in script, 'archived driver does not bind the retained command and warmup contract')
    request = [f"go depth {value['depth']}" if line.startswith('go depth ') else line
               for line in _raw_json(input_ref)['stdin']]
    positions = _request_count(request, value['depth'])
    _require(value['schema_version'] == 1 and value['reference_normalization'] == NORMALIZATION
        and value['positions'] == positions and value['native_search_complete'] is False
        and type(value['cpu_ids']) is list and len(value['cpu_ids']) == 1
        and all(type(cpu) is int and cpu >= 0 for cpu in value['cpu_ids']),
        'historical timing receipt differs from the archived launch boundary')
    warmup, measured = value['warmup_pairs'], value['measured_pairs']
    rows = value['raw']
    orders = _groups(rows, ('original', 'native'), warmup, measured, key='arm')
    _require(len({row['report'] for row in rows}) == len(rows), 'historical samples must have distinct raw reports')
    samples = {'original': [], 'native': []}
    retained, outputs, observed = [], set(), None
    identity = None
    for row in rows:
        ref = _ref(row['report'], _OLD[schema])
        report = read_artifact(ref)
        stdout = _ref(Path(row['report']).with_suffix('.stdout'), 'application/octet-stream')
        stderr = _ref(Path(row['report']).with_suffix('.stderr'), 'application/octet-stream')
        canon, moves = _output(read_artifact(stdout, json_document=False), value['depth'], positions)
        execution = report['execution']
        _require(report['schema_version'] == 1 and report['admitted_for_deployment'] is False
            and execution['entry_path'] == source_ref['uri'] and execution['argv'] == [] and execution['exit_code'] == 0
            and not read_artifact(stderr, json_document=False) and row['canonical_stdout_sha256'] == canon
            and row['bestmoves'] == moves and type(row['elapsed_ns']) is int and row['elapsed_ns'] > 0,
            'historical arm differs from the exact observed output or successful launch')
        outputs.add(canon)
        if row['arm'] == 'original':
            _require(report['library'] is None and execution['installed_bindings'] == 0,
                'historical original arm installed a native replacement')
        else:
            library = report['library']
            read_artifact(library)
            exact = (library, report.get('guard'), report.get('epoch_helper'), report['original_site'])
            if identity is None:
                identity = exact
                for auxiliary in exact[1:3]:
                    if auxiliary is not None:
                        read_artifact(auxiliary)
            _require(exact == identity and library['uri'] == value['library'] and library['sha256'] == value['library_sha']
                and report['source'] == source_ref and execution['installed_bindings'] == 1
                and not report.get('shadow', False), 'historical native arm differs from its pinned region')
            if row['round'] < warmup:
                _require(observed is None and report['statistics'] == value['guarded_observed_calls']
                    and report['phase_ns'] == value['guarded_phase_ns'] and report['table_calls'] == value['guarded_table_calls'],
                    'instrumented warmup evidence differs from its complete report')
                observed = _call_cost(report['statistics'], report['phase_ns'])
                _require(observed['native_calls'] > 0 and sum(report['table_calls'].values()) == observed['native_calls'],
                    'observed version counts must account for all native calls')
            else:
                _require(all(report.get(name) is None for name in ('statistics', 'table_calls', 'phase_ns')),
                    'instrumented sample cannot participate in clean timing')
        if row['round'] >= warmup:
            samples[row['arm']].append(row['elapsed_ns'])
        retained.append(dict(report=ref, stdout=stdout, stderr=stderr, arm=row['arm'], round=row['round']))
    _require(len(outputs) == 1 and observed is not None, 'historical arms must publish identical canonical output')
    summary, compared = _envelopes(samples, 0)
    _require(summary == value['measurements']
        and compared['native']['original_over_candidate_median'] == value['median_speedup'],
        'historical measurement summary failed cold recalculation')
    return dict(benchmark=benchmark_ref, driver=driver, rows=retained, canonical_stdout_sha256=outputs.pop(),
        reconstructed_stdin_sha256=sha256(('\n'.join(request)+'\n').encode('ascii')).hexdigest(),
        depth=value['depth'], positions=positions, cpu_ids=value['cpu_ids'], warmup_pairs=warmup, measured_pairs=measured,
        orders=orders, observes_full_search=True, includes_boundary_adapter_and_fallback=True,
        diagnostics=observed, clean=samples, measurements=summary, comparison=compared['native'],
        study_decision='CONTINUE_STUDY' if compared['native']['verified_observed_gain'] else 'KEEP_ORIGINAL')


def _prepared(path, *, resident):
    tag = 'scar.prepared-resident-transition-study' if resident else 'scar.prepared-sorted-region-study'
    reference = _ref(path, tag)
    raw = _raw_json(reference)
    if resident:
        labels = {0: 'TAGGED_WIRE_NATIVE_SCORE', 1: 'RESIDENT_NATIVE_SCORE', 2: 'RESIDENT_ARCHIVED_PRODUCER_SCORE_STUDY'}
        _require(raw['scope'] == 'C11_DIRECT_IN_PROCESS_REAL_TWO_VERSION_SAMPLES_INPUT_TOKEN_ALREADY_PREPARED'
            and raw['verified_native_score_and_transported_state'] is True
            and raw['production_bypass_proven'] is False and raw['source_level_speedup_proven'] is False,
            'prepared resident fixture differs from its output and source boundary')
        dependencies = [_ref(Path(folder)/'artifact.json', json.loads(
            (Path(folder)/'artifact.json').read_bytes())['schema']) for folder in raw['package_sha256']]
        _require(all(ref['sha256'] == raw['package_sha256'][str(Path(ref['uri']).parent)] for ref in dependencies),
            'resident fixture package no longer matches the measured artifact')
    else:
        labels = {'original': 'original', 'native': 'native'}
        driver_root = Path(reference['uri']).parent
        dependencies = [_ref(driver_root/'measure-prepared.cpp', 'scar.cpp-source'),
                        _ref(driver_root/'measure-prepared', 'scar.native-executable-file')]
        _require(raw['scope'] == 'PREPARED_EXACT_REGION_INPUTS_AND_NATIVE_OUTPUT_RECORDS'
            and raw['includes_python_to_native_conversion'] is False and raw['original_full_output_verified'] is True
            and len({row['checksum'] for row in raw['raw']}) == 1
            and [ref['sha256'] for ref in dependencies] == [raw['driver_sha256'], raw['binary_sha256']],
            'prepared sorting fixture differs from its output and checked benchmark binary')
    warmup, measured = raw['warmup_rounds'], raw['measured_rounds']
    _groups(raw['raw'], labels, warmup, measured, key='mode')
    calls = raw['calls_per_round']
    _require(type(calls) is int and calls > 0
        and all(type(row['elapsed_ns']) is int and row['elapsed_ns'] > 0 for row in raw['raw']),
        'prepared elapsed measurements need positive exact call counts')
    rates = {}
    for mode, name in labels.items():
        one = _summary([row['elapsed_ns']/calls for row in raw['raw'] if row['mode'] == mode and row['round'] >= warmup])
        if resident:
            expected = dict(samples_per_call_ns=one['samples_ns'], median_per_call_ns=one['median_ns'],
                min_per_call_ns=one['min_ns'], max_per_call_ns=one['max_ns'])
            original = raw['results'][name]
        else:
            expected = dict(samples_ns_per_region=one['samples_ns'], median_ns_per_region=one['median_ns'],
                min_ns_per_region=one['min_ns'], max_ns_per_region=one['max_ns'],
                stdev_ns_per_region=one['sample_stdev_ns'])
            original = raw['measurements'][name]
        _require(expected == original, 'prepared timing summary differs from raw samples')
        rates[name] = expected
    return dict(receipt=reference, dependencies=dependencies, rates=rates, calls_per_round=calls,
        warmup_rounds=warmup, measured_rounds=measured, scope=raw['scope'],
        source_entry_adapter_included=False, compatible_for_whole_entry_selection=False)


def history(sort_path, per_call_path, input_run_path, prepared_sort_path, resident_path, score_transport_path):
    """Build a typed historical ledger without transferring costs between boundaries."""
    from scar.optimization import scalar_score_reuse_plan as score_transport

    input_ref = _ref(input_run_path, 'scar.archived-uci-capture-launch')
    capture = _ref(Path(input_ref['uri']).with_name('capture.json'), 'scar.scalar-transition-capture')
    source = read_artifact(capture)['source']
    read_artifact(source, json_document=False)
    sort, per_call = (_legacy_search(path, input_ref, Path(path).with_name('run-study.py'), source)
        for path in (sort_path, per_call_path))
    _require(sort['canonical_stdout_sha256'] == per_call['canonical_stdout_sha256']
        and sort['reconstructed_stdin_sha256'] == per_call['reconstructed_stdin_sha256']
        and sort['depth'] == per_call['depth'] and sort['cpu_ids'] == per_call['cpu_ids'],
        'historical studies must refer to the same fixed requests')
    plan_ref = _ref(score_transport_path, score_transport.SCHEMA)
    plan = score_transport.check(plan_ref['uri'])
    _require(plan == read_artifact(plan_ref) and read_artifact(plan['capture'])['source'] == source
        and plan['guards']['live_producer_token_implemented'] is False
        and plan['native_score_bypass_qualified'] is False, 'historical score witnesses must stay scoped to sampled replay')
    return dict(schema=HISTORY, schema_version=1, source=source, original_input=input_ref, capture=capture,
        historical_search=dict(epoch_sort=sort, per_call_transition=per_call),
        prepared_sort=_prepared(prepared_sort_path, resident=False), prepared_resident=_prepared(resident_path, resident=True),
        score_transport=dict(plan=plan_ref, sample_counts=plan['verified_sample_counts'],
            total_samples=sum(plan['verified_sample_counts'].values()), scope='ORIGINAL_WITNESSES_AND_PREPARED_VALUE_REPLAY',
            live_match_frequency=None, whole_entry_savings_ns=None),
        historical_cross_study_ranking_authorized=False, deployment_authorized=False)


def _arm_contract(mode, arm, source):
    refs = ('library', 'source_guard', 'provenance', 'epoch_helper')
    _require(set(refs) <= arm.keys(), 'each arm needs explicit guard and native package declarations')
    if mode == 'original':
        _require(all(arm[name] is None for name in refs), 'original arm must retain the original program')
        return
    library_ref, guard_ref, flow_ref, epoch_ref = (arm[name] for name in refs)
    lib = read_artifact(library_ref)
    guard = read_artifact(guard_ref)
    epoch = read_artifact(epoch_ref)
    _require(library_ref['schema'] == ('scar.scalar-sorted-library' if mode == 'sort_only'
            else 'scar.scalar-sorted-transition-library')
        and guard_ref['schema'] == 'scar.scalar-sorted-source-guard'
        and epoch_ref['schema'] == 'scar.sorted-study-epoch' and epoch['schema_version'] == 1
        and guard['source'] == source and lib['cxx_standard'] == 20 and lib['deployment_authorized'] is False,
        'candidate must bind the original source, epoch guard and checked native library')
    binary = lib['library']
    read_artifact({**binary, 'uri':str(Path(library_ref['uri']).parent/binary['uri'])}, json_document=False)
    if mode == 'sort_only':
        _require(flow_ref is None and guard['plan'] == lib['region'], 'sort-only study must retain its exact eager boundary')
    else:
        flow = read_artifact(flow_ref)
        parent_sort = read_artifact(lib['sorted'])
        resident = read_artifact(lib['resident'])
        _require(flow_ref['schema'] == 'scar.scalar-sorted-transition-provenance'
            and flow['source_guard'] == guard_ref and flow['source'] == source
            and guard['plan'] == parent_sort['region'] and flow['transport'] == resident['transport']
            and flow['token']['fallback'] == 'CALL_ORIGINAL_TRANSITION_WITH_ORIGINAL_MOVE_AND_KEEP_ORIGINAL_RECURSION',
            'fused study must bind the exact guarded two-site transition')


def _argument(command, flag):
    positions = [i for i,value in enumerate(command) if value == flag]
    _require(len(positions) == 1 and positions[0]+1 < len(command), 'missing exact launch flag ' + flag)
    return command[positions[0]+1]


def _launch(row, source, cpus, report, arm):
    mode = row['mode']
    command = row['command']
    execution = report['execution']
    _require(type(command) is list and len(command) >= 6 and all(type(item) is str for item in command)
        and Path(command[0]).name == 'taskset' and command[1] == '-c'
        and command[2] == ','.join(map(str,cpus)) and command.count('--') == 1,
        'each launched arm must retain the same exact CPU cohort')
    cut = command.index('--')
    module = ('scar.runtime.scalar_sorted_transition_study' if mode == 'fused'
        else 'scar.runtime.scalar_sorted_study')
    flags = command[:cut]
    _require(command[cut+1:] == [source['uri'], *execution['argv']]
        and _argument(flags, '-m') == module and _argument(flags, '--report') == row['report']['uri']
        and '-B' in flags and '--shadow' not in flags,
        'runtime report corresponds to a different original entry or trial command')
    if mode == 'original':
        _require('--isolated-study' not in flags and '--observe' not in flags and '--library' not in flags,
            'baseline launch may not install an interception or diagnostic')
    else:
        guard_flag = '--provenance' if mode == 'fused' else '--source-guard'
        digest_flag = '--provenance-sha256' if mode == 'fused' else '--guard-sha256'
        guard_ref = arm['provenance'] if mode == 'fused' else arm['source_guard']
        _require('--isolated-study' in flags and ('--observe' in flags) == (row['kind'] == 'OBSERVED_WARMUP')
            and _argument(flags, '--library') == arm['library']['uri']
            and _argument(flags, '--sha256') == arm['library']['sha256']
            and _argument(flags, guard_flag) == guard_ref['uri']
            and _argument(flags, digest_flag) == guard_ref['sha256']
            and _argument(flags, '--epoch-helper') == arm['epoch_helper']['uri']
            and _argument(flags, '--epoch-sha256') == arm['epoch_helper']['sha256'],
            'native launch must bind its exact runtime artifacts and instrumentation mode')
    return tuple(execution['argv']), tuple(flags[:flags.index('-m')])


def _paired_row(row, arm, source, cpus, depth, searches):
    mode = row['mode']
    report = read_artifact(row['report'])
    schema = _REGION.get(mode, 'scar.scalar-sorted-source-study')
    _require(report['schema'] == schema and report['schema_version'] == 1
        and report['admitted_for_deployment'] is False and report['library'] == arm['library']
        and report['execution']['entry_path'] == source['uri'] and report['execution']['exit_code'] == 0
        and report['execution']['installed_bindings'] == int(mode != 'original')
        and type(row['wall_ns']) is int and row['wall_ns'] > 0 and type(row['returncode']) is int
        and row['returncode'] == 0
        and not report.get('shadow', False) and not read_artifact(row['stderr'], json_document=False),
        'a paired observation failed, emitted errors or installed an unmeasured boundary')
    output, _ = _output(read_artifact(row['stdout'], json_document=False), depth, searches)
    argv = _launch(row, source, cpus, report, arm)
    if mode == 'original':
        return output, argv, None
    guard_key = 'source_guard' if mode == 'fused' else 'guard'
    _require(report['source'] == source and report[guard_key] == arm['source_guard']
        and report.get('provenance') == arm['provenance'] and report['epoch_helper'] == arm['epoch_helper'],
        'candidate report differs from its admitted guard, library or provenance')
    pairs = (('sort', 'statistics', 'phase_ns', ('native',)),) if mode == 'sort_only' else (
        ('sort', 'sort_statistics', 'sort_phase_ns', ('native',)),
        ('transition', 'transition_statistics', 'transition_phase_ns', ('transported', 'native_recompute')))
    if row['kind'] == 'CLEAN_MEASURED':
        _require(report['table_calls'] is None
            and all(report[counts] is None and report[timings] is None for _,counts,timings,_ in pairs),
            'instrumented phase intervals cannot enter the paired performance result')
        return output, argv, None
    costs = {label: _call_cost(report[counts], report[timing], success) for label,counts,timing,success in pairs}
    _require(all(type(value) is int and value >= 0 for value in report['table_calls'].values())
        and sum(report['table_calls'].values()) == costs['sort']['native_calls'],
        'instrumented sort counts must bind every table-version observation')
    return output, argv, costs


def _choice(measurements, comparisons, diagnostics, samples):
    eligible = {}
    for mode, comparison in comparisons.items():
        exercised = all(any(row[phase]['native_calls'] > 0 for row in diagnostics[mode])
            for phase in (('sort', 'transition') if mode == 'fused' else ('sort',)))
        eligible[mode] = exercised and comparison['verified_observed_gain']
        comparison['guarded_boundary_exercised'] = exercised
    if not any(eligible.values()):
        return 'original', 'NO_QUALIFIED_OBSERVED_GAIN'
    if eligible.get('sort_only') and eligible.get('fused'):
        fused = measurements['fused']
        simpler = measurements['sort_only']
        supersedes = fused['max_ns'] < simpler['min_ns'] and all(a < b for a,b in zip(
            samples['fused'], samples['sort_only'], strict=True))
        return ('fused', 'FUSED_SEPARATES_FROM_ORIGINAL_AND_SORT') if supersedes else (
            'sort_only', 'KEEP_QUALIFIED_NARROWER_BOUNDARY')
    winner = next(mode for mode, valid in eligible.items() if valid)
    return winner, 'QUALIFIED_OBSERVED_GAIN'


def accept(benchmark_path, *, minimum_gain_bps=0):
    """Cold-check clean paired UCI search timing and choose a measured isolated study."""
    reference = _ref(benchmark_path, BENCHMARK)
    receipt = read_artifact(reference)
    arm_order = ('original', 'sort_only', 'fused')
    modes = tuple(mode for mode in arm_order if mode in receipt['arms'])
    _require(receipt['schema_version'] == 1 and receipt['normalization'] == NORMALIZATION
        and receipt['scope'] == 'ORIGINAL_SEARCH_NATIVE_SORT_AND_LAZY_TRANSITION_SITE_STUDY'
        and set(modes) == set(receipt['arms']) and 'original' in modes and len(modes) >= 2,
        'expected original and one or more named guarded paired arms')
    source, cpus, depth = (receipt[key] for key in ('source', 'cpu_ids', 'depth'))
    _require(type(cpus) is list and cpus and all(type(cpu) is int and cpu >= 0 for cpu in cpus)
        and cpus == sorted(set(cpus)) and source['schema'] == 'scar.python-source'
        and receipt['stdin']['schema'] == 'application/octet-stream',
        'expected the original Python entry, exact stdin and fixed CPU cohort')
    read_artifact(source, json_document=False)
    stdin = read_artifact(receipt['stdin'], json_document=False)
    _require(stdin.endswith(b'\n') and b'\r' not in stdin, 'stdin must retain the fixed newline transcript')
    searches = _request_count(stdin.decode('ascii').splitlines(), depth)
    definitions, rows = receipt['arms'], receipt['rows']
    _require(all(len({row[name]['uri'] for row in rows}) == len(rows) for name in ('report', 'stdout', 'stderr')),
        'paired samples each need their own raw report and output artifacts')
    for mode in modes:
        _arm_contract(mode, definitions[mode], source)
    clocks = {'OBSERVED_WARMUP', 'CLEAN_MEASURED'}
    by_kind = {kind: {row['round'] for row in rows if row['kind'] == kind} for kind in clocks}
    _require(all(row['kind'] in clocks and type(row['round']) is int for row in rows)
        and not by_kind['OBSERVED_WARMUP'] & by_kind['CLEAN_MEASURED'], 'timing and diagnostic rounds must remain distinct')
    warms, measured = len(by_kind['OBSERVED_WARMUP']), len(by_kind['CLEAN_MEASURED'])
    orders = _groups(rows, modes, warms, measured, key='mode')
    _require(all((row['round'] < warms) == (row['kind'] == 'OBSERVED_WARMUP') for row in rows),
        'warmups must precede all clean rounds')
    for index, order in enumerate(orders):
        offset = index % len(modes)
        expected = modes[offset:] + modes[:offset]
        if index % 2:
            expected = tuple(reversed(expected))
        _require(tuple(order) == expected, 'paired round differs from the frozen launch order')
    samples = {mode: [] for mode in modes}
    diagnostics = {mode: [] for mode in modes if mode != 'original'}
    launches, detail = set(), set()
    for row in rows:
        output, argv, costs = _paired_row(row, definitions[row['mode']], source, cpus, depth, searches)
        detail.add(output)
        launches.add(argv)
        if row['kind'] == 'CLEAN_MEASURED':
            samples[row['mode']].append(row['wall_ns'])
        if costs is not None:
            diagnostics[row['mode']].append(costs)
    _require(len(launches) == len(detail) == 1 and all(len(group) == warms for group in diagnostics.values()),
        'paired arms must match canonical stdout, arguments and diagnostic coverage')
    measurements, comparisons = _envelopes(samples, minimum_gain_bps)
    chosen, reason = _choice(measurements, comparisons, diagnostics, samples)
    argv, prefix = launches.pop()
    return dict(schema=SCHEMA, schema_version=1, benchmark=reference, source=source, stdin=receipt['stdin'],
        selected=chosen, decision=reason, depth=depth, positions=searches, cpu_ids=cpus,
        argv=list(argv), shared_command_prefix=list(prefix),
        orders=orders, warmup_pairs=warms, measured_pairs=measured, minimum_gain_bps=minimum_gain_bps,
        exact_canonical_stdout_sha256=detail.pop(), measurements=measurements, comparisons=comparisons,
        instrumented_boundaries=diagnostics, observed_frequency_transferred_to_clean=False,
        cost_boundary=dict(kind='WHOLE_ORIGINAL_SEARCH_ENTRY_FRESH_PROCESS',
            included=['startup', 'guard', 'marshalling', 'ffi', 'native', 'fallback', 'retained_source_search', 'output'],
            pure_native_time_known=False, isolated_regions_only=False),
        uncertainty_scope='OBSERVED_PAIRED_SAMPLE_ENVELOPES', historical_or_prepared_timing_used_for_decision=False,
        full_source_effects_closed=False, deployment_authorized=False)


def check(path):
    schema = json.loads(Path(path).read_bytes())['schema']
    _require(schema in (SCHEMA, HISTORY), 'unsupported search cost receipt')
    stored = read_artifact(_ref(path, schema))
    if schema == SCHEMA:
        replay = accept(stored['benchmark']['uri'], minimum_gain_bps=stored['minimum_gain_bps'])
    else:
        own = stored['historical_search']
        replay = history(own['epoch_sort']['benchmark']['uri'], own['per_call_transition']['benchmark']['uri'],
            stored['original_input']['uri'], stored['prepared_sort']['receipt']['uri'],
            stored['prepared_resident']['receipt']['uri'], stored['score_transport']['plan']['uri'])
    _require(replay == stored, 'search cost receipt differs from cold replay')
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--benchmark', type=Path)
    inputs.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--minimum-gain-bps', type=int, default=0)
    args = parser.parse_args(argv)
    if args.check and not args.out and args.minimum_gain_bps == 0:
        result = check(args.check)
    elif args.benchmark and args.out:
        result = accept(args.benchmark,minimum_gain_bps=args.minimum_gain_bps)
        with args.out.open('x',encoding='utf-8') as stream:
            json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --benchmark with --out, or --check')
    print(json.dumps({key:result[key] for key in ('schema','selected','decision','deployment_authorized')
                      if key in result}))


if __name__ == '__main__':
    main()

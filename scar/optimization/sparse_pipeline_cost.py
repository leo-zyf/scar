"""Replay paired serial update and sparse solve timing cuts."""

from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.sparse_solve_cost import _duration, _summary

SCHEMA = 'scar.sparse-pipeline-paired-cost'
DESIGN = SCHEMA + '-design'
WORKER = SCHEMA + '-worker'
MODES = ('original', 'solve_only', 'joint')


def _chain(worker, count):
    samples = worker['samples']
    activation = worker['activation']
    mode = worker['mode']
    close_cut = worker['close_clock_ns']
    admit_cut = worker['admit_clock_ns']
    if mode == 'original' and (close_cut is not None or admit_cut is not None):
        raise ValueError('original arm cannot carry a native lifecycle')
    close = _duration(close_cut) if close_cut is not None else 0
    admit = _duration(admit_cut) if admit_cut is not None else 0
    costs = dict(first_update=[], first_solve=[], first_pair=[], warm_update=[], warm_solve=[],
                 warm_pair=[], load_update=[], load_solve=[], close_scope=[close],
                 admit_scope=[admit], non_warm_lifecycle=[],
                 observed_scope=[])
    previous_end = admit_cut[1] if admit_cut is not None else 0
    for index, sample in enumerate(samples):
        start, split, end = sample['clock_cuts_ns']
        if (not previous_end < start < split < end
                or close_cut is not None and end > close_cut[0]):
            raise ValueError('update or solve overlaps its causal predecessor')
        phase = 'first' if index == 0 else 'warm'
        costs[phase+'_update'].append(_duration((start, split)))
        costs[phase+'_solve'].append(_duration((split, end)))
        costs[phase+'_pair'].append(end - start)
        previous_end = end
    active = dict(update=mode == 'joint', solve=mode != 'original')
    qualified = not worker['frontier'] and (mode == 'original' or close > 0 and admit > 0)
    for stage, expected in active.items():
        load = activation.get(stage)
        statistics = worker['admission'].get(stage)
        if expected:
            report = worker['stages']['row' if stage == 'update' else stage]
            release = report['release']
            cache_key = 'topology' if stage == 'update' else 'symbolic'
            retained_key = cache_key + '_retained_bytes'
            cache = report[cache_key]
            qualified &= (statistics is not None and statistics['native_calls'] == count
                          and statistics['fallback_calls'] == 0
                          and cache is not None and cache['builds'] == 1 and cache['hits'] == count-1
                          and release is not None and release['closed'] is True
                          and release['retained_after_close_bytes'] == 0
                          and release['retained_before_close_bytes'] == worker['resource'][retained_key])
            if load is None or load['attempts'] != 1:
                qualified = False
                continue
            duration = load['load_ns']
            if type(duration) is not int or not 0 < duration <= costs['first_'+stage][0]:
                raise ValueError('native load must be nested inside the first stage')
            costs['load_'+stage].append(duration)
        elif load is not None or statistics is not None:
            raise ValueError('observed unexpected attachment')
    fixed = admit + close + costs['first_pair'][0]
    costs['non_warm_lifecycle'].append(fixed)
    costs['observed_scope'].append(fixed + sum(costs['warm_pair']))
    return costs, qualified


def _repeated_projection(count, fixed, warm):
    return {key: fixed[key] + (count-1) * warm[key]
            for key in ('estimate_ns', 'lower_ns', 'upper_ns')} | {
                'calls': count, 'bound_kind': 'conditional_observed_sample_envelope_projection'}


def project(design_path, worker_paths):
    design_ref = artifact_reference(design_path, DESIGN)
    design = read_artifact(design_ref)
    composition = read_artifact(design['composition'])
    scope = read_artifact(design['source_scope'])
    entry = scope['entry']
    count = entry['argv_repetitions']
    if (design['schema_version'] != 1 or design['entry'] != entry['source']
            or composition['entry'] != entry or composition['source_scope'] != design['source_scope']
            or design['context']['argv'] != entry['argv']
            or design['context']['cpu_ids'] != entry['affinity']
            or any(design['context']['environment'][key] != value
                   for key, value in entry['thread_environment'].items())
            or scope['observed']['region_attempts'] != count
            or scope['observed']['consumer_completions'] != count):
        raise ValueError('pipeline study does not match the source scope')
    solve_scope = read_artifact(composition['solve_scope'])
    program = read_artifact(read_artifact(solve_scope['source_graph'])['program'])
    target = design['hardware']['target']
    read_artifact(target)
    if (target['sha256'] != program['target']['sha256']
            or target['uri'] != program['target']['path']
            or [design['hardware']['calling_cpu_id']] != design['context']['cpu_ids']):
        raise ValueError('pipeline hardware and calling CPU differ from the admitted graph')
    for stage in ('update', 'solve'):
        selection = read_artifact(design[stage+'_selection'])
        native_ref = design[stage+'_native']
        component = selection['topology']['proof'] if stage == 'update' else selection['source']
        expected = composition['topology_scope' if stage == 'update' else 'solve_scope']
        if (selection['entry'] != entry['source'] or selection['native'] != native_ref
                or component != expected):
            raise ValueError('pipeline selection, entry or native producer differs')
        native = read_artifact(native_ref)
        read_artifact(native['library'], json_document=False)
    for reference in (design['entry'], design['harness'], design['fixture_parser'],
                      *design['runtime'].values()):
        read_artifact(reference, json_document=False)
    refs = [artifact_reference(path, WORKER) for path in worker_paths]
    if (not design['groups'] or len(refs) != len(MODES) * len(design['groups'])
            or len({ref['uri'] for ref in refs}) != len(refs)):
        raise ValueError('one worker per frozen case and group required')
    workers = iter(refs)
    diagnostics = []
    times_by_mode = {}
    marginal = {key: [] for key in ('first_pair', 'warm_pair', 'non_warm_lifecycle', 'observed_scope')}
    all_equal = True
    all_admitted = True
    resource = {mode: [] for mode in MODES}
    for group in design['groups']:
        if set(group['order']) != set(MODES) or len(group['order']) != len(MODES):
            raise ValueError('group requires the full mode comparison')
        if (len(group['iterations']) < 2 or group['iterations'][0] != 0
                or group['iterations'] != sorted(set(group['iterations']))
                or group['iterations'][-1] >= count):
            raise ValueError('one actual first iteration per group required')
        by_mode = {}
        outputs = []
        for mode in group['order']:
            raw = read_artifact(next(workers))
            if (raw['schema_version'] != 1 or raw['design'] != design_ref
                    or raw['group'] != group or raw['mode'] != mode
                    or raw['context'] != design['context']
                    or [sample['iteration'] for sample in raw['samples']] != group['iterations']):
                raise ValueError('pipeline worker identity or iteration order differs')
            costs, admitted = _chain(raw, len(group['iterations']))
            all_admitted &= admitted
            diagnostics.append(dict(group=group['id'], mode=mode, admitted=admitted,
                                    activation=raw['activation'], admission=raw['admission']))
            by_mode[mode] = costs
            series = times_by_mode.setdefault(mode, {key: [] for key in costs})
            for key, values in costs.items():
                series[key].extend(values)
            outputs.append([sample['value_signatures'] for sample in raw['samples']])
            resource[mode].append(raw['resource'])
        all_equal &= outputs[0] == outputs[1] == outputs[2]
        for key, values in marginal.items():
            values.extend(old - new for old, new in zip(by_mode['solve_only'][key],
                                                         by_mode['joint'][key]))
    summaries = {mode: {key: _summary(values) if values else None for key, values in series.items()}
                 for mode, series in times_by_mode.items()}
    paired = {key: _summary(values) for key, values in marginal.items()}
    projections = None
    if all_equal and all_admitted:
        projections = {mode: _repeated_projection(count, summary['non_warm_lifecycle'], summary['warm_pair'])
                       for mode, summary in summaries.items()}
        projections['joint_savings_over_solve_only'] = _repeated_projection(
            count, paired['non_warm_lifecycle'], paired['warm_pair'])
    return dict(schema=SCHEMA, schema_version=1, design=design_ref, workers=refs,
                composition=design['composition'], source_scope=design['source_scope'],
                context=design['context'], update_native=design['update_native'],
                solve_native=design['solve_native'], hardware=design['hardware'],
                iteration_template=dict(
                    count=count, operations=['update', 'solve'],
                    edges=[dict(producer='update', consumer='solve', kind='VALUE',
                                iteration_distance=0, scope_locator='#/value_routes'),
                           dict(producer='solve', consumer='update', kind='ORDER',
                                iteration_distance=1, scope_locator='#/loop')],
                    cpu_lease=dict(cpu_id=design['hardware']['calling_cpu_id'], capacity=1),
                    memory_domain=dict(cpu_numa_node=design['hardware']['cpu_numa_node'],
                        actual_residency=design['hardware']['actual_numa_residency'])),
                timings_by_mode=summaries, paired_joint_savings=paired,
                conditional_serial_scope=projections,
                resource={mode: dict(
                    topology_retained_bytes=max(row['topology_retained_bytes'] for row in values),
                    symbolic_retained_bytes=max(row['symbolic_retained_bytes'] for row in values),
                    peak_simultaneous_persistent_bytes=max(row['topology_retained_bytes'] +
                                                           row['symbolic_retained_bytes'] for row in values),
                    transient_numeric_workspace_bytes=None) for mode, values in resource.items()},
                diagnostics=diagnostics,
                qualification=dict(all_values_bitwise=all_equal, all_expected_attachments=all_admitted,
                    observed_pair_clocks_include_lazy_load=True,
                    scope_cost_excludes_assembly_and_output_consumption=True,
                    full_entry_cost_ns=None, automatic_rewrite=False))


def check(path):
    result = read_artifact(artifact_reference(Path(path), SCHEMA))
    replay = project(result['design']['uri'], [ref['uri'] for ref in result['workers']])
    if result != replay:
        raise ValueError('paired update and solve cost does not replay')
    return result

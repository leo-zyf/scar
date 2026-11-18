"""Compare two exact sparse solver teams under one paired sampling design."""

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.sparse_parallel_cost import SCHEMA as CALIBRATION, check as check_calibration
from scar.optimization.sparse_solve_cost import _summary

SCHEMA = 'scar.sparse-parallel-scaling-cost'
DESIGN = SCHEMA + '-design'
_TEAMS = ('baseline', 'expanded')
_ESTIMATES = ('estimate_ns', 'lower_ns', 'upper_ns')


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


def project(design_path, cost_paths):
    design_ref = artifact_reference(design_path, DESIGN)
    design = read_artifact(design_ref)
    if design['schema_version'] != 1 or set(cost_paths) != set(_TEAMS):
        raise ValueError('a paired baseline and expanded solver team are required')
    read_artifact(design['harness'], json_document=False)
    refs = {mode: artifact_reference(cost_paths[mode], CALIBRATION) for mode in _TEAMS}
    studies = {mode: check_calibration(refs[mode]['uri']) for mode in _TEAMS}
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
    for mode, study in studies.items():
        calibration = read_artifact(study['design'])
        if (study['choice'] != design['choices'][mode] or calibration['groups'] != groups
                or calibration['runtime'] != design['runtime']
                or calibration['worker_address_space_bytes'] != design['worker_address_space_bytes']):
            raise ValueError('solver sampling definition or worker bound differs from the paired design')
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
    return dict(schema=SCHEMA, schema_version=1, design=design_ref, costs=refs,
                owner_cpu_id=baseline['owner_cpu_id'], source_scope=baseline['source_scope'],
                teams={mode: dict(cpu_ids=row['worker_cpu_ids'], window=row['window'])
                       for mode, row in studies.items()},
                projected_section_ns=spans, projected_marginal_savings_ns=delta,
                paired_diagnostics=paired, paired_short_section_savings_ns=measured_delta,
                output_bitwise=True, memory_bandwidth_bytes_per_second=None,
                exact_peak_native_scratch_bytes=None, physical_memory_residency=None,
                full_entry_cost_ns=None, automatic_admission=False)


def check(path):
    saved = read_artifact(artifact_reference(path, SCHEMA))
    replayed = project(saved['design']['uri'], {mode: ref['uri'] for mode, ref in saved['costs'].items()})
    if saved != replayed:
        raise ValueError('paired team comparison differs from replay')
    return saved

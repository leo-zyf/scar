"""Project ordered sparse owner work and its measurable costs from pinned evidence."""

import argparse
import json
from pathlib import Path

from scar.analysis.replay_transaction import replay_transaction
from scar.analysis.sparse_parallel_scaling_scope import _release_lineage
from scar.ir.execution_recipe import SCHEMA as RECIPE, artifact_reference, read_artifact
from scar.optimization.sparse_parallel_cost import SCHEMA as CALIBRATION, check as check_calibration
from scar.optimization.sparse_parallel_recipe import verify_sparse_parallel_recipe
from scar.optimization.sparse_solve_cost import _duration, _summary
from scar.runtime.sparse_parallel_entry import SCHEMA as RELEASE

SCHEMA = 'scar.sparse-parallel-owner-cost'
EXECUTION = 'scar.sparse-parallel-entry-execution'


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _boundary(recipe, scope, pipeline):
    source = recipe['semantic_boundary']
    normalization = pipeline['normalization']
    detached = recipe['artifacts']['detached']
    status = source['original_python_status_boundary']
    contract = scope['speculation_contract']
    _require(pipeline['source_scope'] == source['source_scope']
        and pipeline['solve_scope'] == scope['solve_scope']
        and pipeline['regions']['sparse_solve']['call_point'] == source['original_call']
        and normalization['original_normalization_required'] is True
        and normalization['order'][-1] == 'NATIVE_CALL'
        and contract['original_effects_preserved'] == source['original_effects']
        and contract['raw_inputs_always_revalidated'] is True
        and contract['acceptance_key'] == source['Q']['acceptance_key']
        and status['preserve_in_python'] is True,
        'owner work does not match the ordered recipe source and visible effects')

    tasks = {row['id']: row for row in recipe['task_graph']['tasks']}
    stages = [
        ('capture-live-ins', 'ONCE', dict(detached=detached, locator='#/roles')),
        ('prepare-original', 'PER_CALL', dict(entry=source['original_entry'],
            source_spans=source['original_preparation_spans'], row_edit=pipeline['regions']['row_edit'],
            operand_routes='#/value_routes', expands='#/graph/original_normalization')),
        ('bootstrap-team', 'ONCE_AFTER_FIRST_ORIGINAL_REQUEST', dict(original_call=source['original_call'])),
        ('prepare-private', 'PER_CALL', dict(detached=detached, locator='#/scale')),
        ('compare-request', 'PER_CALL', dict(original_call=source['original_call'],
            normalized_operands='#/normalization/terminal_call', acceptance_key=contract['acceptance_key'])),
        ('publish-original', 'PER_CALL', dict(sparse_status=status,
            preserved_effects=source['original_effects'])),
    ]
    overlay = []
    for task, cadence, source_binding in stages:
        row = tasks[task]
        _require(any(lease['cpu_id'] == recipe['resources']['selected_cpu']['owner_cpu_id']
            for lease in row['resource_refs'] if lease.get('kind') == 'OWNER_CPU'),
            'expanded preparation task lost its ordered CPU owner')
        overlay.append(dict(task=task, cadence=cadence, source=source_binding,
            reads=row['reads'], writes=row['writes'], depends_on=row['depends_on'],
            resource_refs=row['resource_refs']))
    steps = normalization['order'][:-1]
    return dict(stages=overlay, indexed_control=recipe['control']['cross_instance_dependencies'],
        original_normalization=dict(parent='prepare-original', axis='i',
            operations=[dict(id=step, duration_ns=None) for step in steps],
            normal_path_order=[dict(before=left, after=right) for left, right in zip(steps, steps[1:])],
            retained_f4_targets='#/normalization/retained_original_call_targets',
            native_operand_ports=normalization['terminal_call']['input_ports'],
            native_world_in=normalization['terminal_call']['world_before'],
            native_world_out=normalization['terminal_call']['world_after']),
        outside_recipe=dict(original_entry=source['original_entry'], before='capture-live-ins',
            setup_source=pipeline['entry']['setup_source'],
            exact_setup_prefix_bytes=pipeline['entry']['exact_setup_prefix_bytes']),
        logical_values_distinct=['original-arguments', 'private-arguments'],
        cross_region_storage_alias=normalization['actual_cross_region_storage_alias'],
        raw_keys_interchangeable=pipeline['composition_contract']['raw_keys_interchangeable'],
        original_normalization_required=True, exact_match_at_source_carrier=True)


def _execution(execution, release_ref, release, recipe):
    count = release['count']
    offers = execution['parallel'] or {}
    team = offers.get('workers') or {}
    workers = recipe['resources']['selected_cpu']['worker_cpu_ids']
    _require(execution['schema_version'] == 1 and execution['selection_sha256'] == release_ref['sha256']
        and execution['entry_path'] == release['entry']['uri'] and execution['argv'] == release['context']['argv']
        and execution['applied'] == 'PARALLEL' and execution['exit_code'] == 0
        and execution['statistics']['fallback_calls'] == 0 and execution['statistics']['native_calls'] == count
        and offers['inputs_captured'] is True and offers['matched'] == count
        and offers['serial_calls'] == offers['rejected'] == 0 and offers['peak_window'] == release['workers']['window']
        and team['closed'] is True and team['retained_after_close_bytes'] == team['pending_snapshot_bytes'] == 0
        and team['native_calls'] == team['submissions'] == count and team['fallback_calls'] == 0
        and sorted(row['cpu'] for row in team['workers']) == workers
        and [(row['issue'], row['native']) for row in team['publications']]
            == [(index, True) for index in range(count)],
        'live clocks are not from the exact released ordered execution')
    return team


def _costs(calibration, team):
    live = []
    for key, boundary in (
            ('owner_validation_ns', 'First serial provider validation and close, after raw request capture'),
            ('startup_ns', 'Create courier and workers, exchange structure, await symbolic plans')):
        value = team[key]
        _require(type(value) is int and value > 0, 'one positive clock per released startup interval required')
        live.append(dict(task='bootstrap-team', interval=boundary, duration_ns=value, evidence=key))
    ordered = calibration['ordered']
    phase = calibration['phase_cost_ledger']['measured']
    diagnostic = {
        'prepare-private': dict(evidence='ordered/prepare_ns', sample=ordered['prepare_ns']),
        'prepare-original': dict(evidence='ordered/replay_before_wait_ns', sample=ordered['replay_before_wait_ns'],
            coverage='Harness row edit and partial normalization on non-first calls'),
        'publish-original': dict(evidence='phase_cost_ledger/measured/publish_digest_proxy_ns',
            sample=phase['publish_digest_proxy_ns'], coverage='Harness output digest'),
    }
    return dict(actual_released_bootstrap_subphases=live,
        actual_released_bootstrap_observed_disjoint_elapsed_ns=sum(row['duration_ns'] for row in live),
        sample_diagnostics_by_task=diagnostic,
        unresolved_actual_duration_ns={
            'entry-assembly-and-exit': None, 'capture-live-ins': None, 'prepare-original': None,
            'bootstrap-team': None, 'prepare-private': None, 'compare-request': None, 'publish-original': None},
        section_projection_is_not_full_entry=True, full_entry_cost_ns=None,
        charge_policy='LIVE_SUBPHASES_AND_CALIBRATION_DIAGNOSTICS_ARE_SEPARATE')


_LIVE_BOUNDARY = {
    'snapshot': ('capture-live-ins', 'Live-in capture body, including validated detached snapshot construction'),
    'initial_match': ('bootstrap-team', 'First prediction/raw-request comparison before starting the team'),
    'initial_prediction': ('bootstrap-team', 'Prediction for the first-request guard, separate from scheduled private offers'),
    'row_edit': ('prepare-original', 'Original source row edit callable, after argument evaluation'),
    'normalize': ('prepare-original', 'Original sealed solver entry to guarded raw carrier offer'),
    'prepare_private': ('prepare-private', 'Original runtime detached args for one scheduled offer'),
    'capture_request': ('compare-request', 'Live ABI validation and current raw request capture before waiting'),
    'future_wait': ('compare-request', 'Future.result inclusive of the instrumented call and wait'),
    'finish_match': ('compare-request', 'After Future.result through exact comparison and runtime bookkeeping'),
    'status_return': ('publish-original', 'Raw offer return through protected original solver status, normalization and return'),
    'publication': ('publish-original', 'Original hash update on the original result'),
}


def _sample_intervals(raw, team, count, width, samples):
    from scar.runtime.sparse_parallel_owner_observe import SAMPLE_PHASES

    run_begin, run_end = raw['context_elapsed']
    rows, predictions = raw['rows'], raw['predictions']
    _duration(raw['context_elapsed'])
    _require(len(rows) == count and [item['index'] for item in predictions] == [0, *range(count)]
        and set(raw['cuts']) == {'snapshot', 'initial_match'},
        'owner clocks must cover the full released request and prediction order')
    snapshot, guard = (raw['cuts'][name] for name in ('snapshot', 'initial_match'))
    initial, *offers = [row['interval'] for row in predictions]
    _require(run_begin <= snapshot[0] < snapshot[1] <= rows[0]['row_edit'][0]
        and rows[0]['normalize'][1] <= initial[0] < initial[1] <= guard[0] < guard[1]
        <= offers[0][0] and offers[width-1][1] <= rows[0]['capture_request'][0],
        'clock order differs from source capture, initial match and worker startup')
    for phase, interval in (*raw['cuts'].items(), ('initial_prediction', initial)):
        samples[phase].append(_duration(interval))
    tasks = sorted(team['tasks'], key=lambda row: row['issue'])
    _require([task['issue'] for task in tasks] == list(range(count)),
        'owner clocks require one measured native task per source occurrence')
    previous = snapshot[1]
    for index, (row, task, prediction) in enumerate(zip(rows, tasks, offers, strict=True)):
        _require(set(row) == set(SAMPLE_PHASES) and previous <= row['row_edit'][0]
            and row['publication'][1] <= run_end
            and all(row[left][1] <= row[right][0] for left, right in zip(SAMPLE_PHASES, SAMPLE_PHASES[1:]))
            and row['future_wait'][1] == row['finish_match'][0]
            and prediction[1] <= task['submitted_ns'] <= task['exchange_complete_ns'] <= row['future_wait'][1],
            'owner intervals are out of order or differ from the native offer and completion')
        if index >= width:
            refilled = rows[index-width]
            _require(refilled['finish_match'][1] <= prediction[0] < prediction[1] <= refilled['status_return'][0],
                'the measured private offer did not follow its ordered refill credit')
        previous = row['publication'][1]
        samples['prepare_private'].append(_duration(prediction))
        for phase in SAMPLE_PHASES:
            samples[phase].append(_duration(row[phase]))


def _timeline_run(raw, team, lanes):
    """Replay placement and separately account the disjoint owner and worker clocks."""
    workers = {row['cpu']: row['pid'] for row in team['workers']}
    native = {cpu: dict(calls=0, solve_wall_ns=0, native_thread_cpu_ns=0) for cpu in lanes}
    counts = {}
    ends = {}
    ordered = sorted(team['tasks'], key=lambda row: row['issue'])
    owner_intervals = list(raw['cuts'].items())
    first, *predictions = [row['interval'] for row in raw['predictions']]
    owner_intervals.append(('initial_prediction', first))
    await_ns = exchange_before_await = 0
    for index, (row, task, prediction) in enumerate(zip(raw['rows'], ordered, predictions, strict=True)):
        cpu = task['cpu']
        begin, end = task['started_ns'], task['ended_ns']
        thread_cpu = task['native_thread_cpu_ns']
        submitted, exchanging, exchange_done = (task[key] for key in
            ('submitted_ns', 'exchange_started_ns', 'exchange_complete_ns'))
        _require(cpu == lanes[index % len(lanes)] and workers[cpu] == task['pid']
            and task['status'] == 0 and ends.get(cpu, 0) <= submitted
            and prediction[1] <= submitted <= exchanging <= begin < end <= exchange_done
            <= row['future_wait'][1] <= row['finish_match'][1] <= row['publication'][0]
            and type(thread_cpu) is int and 0 < thread_cpu <= end-begin,
            'native solve differs from the issued task, placed lane or original ordered publication')
        ends[cpu] = exchange_done
        lane = native[cpu]
        lane['calls'] += 1
        lane['solve_wall_ns'] += end-begin
        lane['native_thread_cpu_ns'] += thread_cpu
        await_ns += _duration(row['future_wait'])
        exchange_before_await += exchange_done <= row['future_wait'][0]
        owner_intervals.append(('prepare_private', prediction))
        owner_intervals.extend(row.items())
    owner_intervals.sort(key=lambda pair: pair[1])
    _require(all(left[1] <= right[0] for (_, left), (_, right) in zip(owner_intervals, owner_intervals[1:])),
        'measured owner intervals overlap on the single owner lane')
    for phase, interval in owner_intervals:
        if phase != 'future_wait':
            task, _ = _LIVE_BOUNDARY[phase]
            counts[task] = counts.get(task, 0) + _duration(interval)
    return dict(pid=raw['pid'], calls=len(ordered), owner_measured_service_by_task_ns=counts,
        owner_measured_service_ns=sum(counts.values()), owner_future_result_elapsed_ns=await_ns,
        exchange_complete_before_await_calls=exchange_before_await,
        native_solve_wall_span_ns=max(row['ended_ns'] for row in ordered)-min(row['started_ns'] for row in ordered),
        native_solve_wall_sum_ns=sum(row['solve_wall_ns'] for row in native.values()),
        native_thread_cpu_sum_ns=sum(row['native_thread_cpu_ns'] for row in native.values()),
        native_lanes={str(cpu):value for cpu,value in native.items()})


def _instrumented_costs(paths, release_ref, release, recipe, *, timeline=False):
    from scar.runtime import sparse_parallel_owner_observe as observer

    refs = [artifact_reference(path, observer.SCHEMA) for path in paths]
    _require(refs and len({ref['sha256'] for ref in refs}) == len(refs),
        'independent instrumented entry receipts are required')
    validated = read_artifact(release['validation'])
    outputs = {(row['stdout_sha256'], row['stderr_sha256']) for row in validated['runs']}
    _require(len(outputs) == 1, 'released comparison has no single exact stdout and stderr')
    expected_context = {**release['context'], 'cpu_ids': release['launch_cpu_ids']}
    plan = observer.probe_plan(release)
    implementation = artifact_reference(observer.__file__, observer.SOURCE)
    samples = {key: [] for key in _LIVE_BOUNDARY}
    processes, timeline_runs, previous = set(), [], 0
    for reference in refs:
        raw = read_artifact(reference)
        _require(raw['schema_version'] == 1 and raw['release'] == release_ref
            and raw['context'] == expected_context and raw['plan'] == plan
            and raw['instrument'] == implementation and raw['instrumented'] is True
            and (raw['stdout_sha256'], raw['stderr_sha256']) in outputs
            and type(raw['pid']) is int and raw['pid'] > 0 and raw['pid'] not in processes
            and previous < raw['context_elapsed'][0],
            'instrumented clocks do not match the sealed source, released result or independent process order')
        team = _execution(raw['execution'], release_ref, release, recipe)
        _sample_intervals(raw, team, release['count'], release['workers']['window'], samples)
        if timeline:
            timeline_runs.append(_timeline_run(raw, team, release['workers']['cpu_ids']))
        processes.add(raw['pid'])
        previous = raw['context_elapsed'][1]
    phases = {phase: dict(task=task, coverage=coverage, **_summary(samples[phase]))
              for phase, (task, coverage) in _LIVE_BOUNDARY.items()}
    result = dict(observations=refs, distinct_processes=len(processes), calls_per_process=release['count'],
        instrumented=True, standalone_cxx_measurement=False, comparable_clean_benchmark=False,
        subphases=phases, summable_with_baseline_and_calibration=False, full_entry_cost_ns=None,
        still_unresolved=['entry_assembly_exit', 'original_caller_scale_and_argument_evaluation',
            'individual_normalization_operations', 'retire_and_request_capture_bookkeeping_gaps',
            'non_hash_publication_work', 'uninstrumented_duration_at_same_boundaries'])
    if timeline:
        result['source_order_timeline'] = dict(runs=timeline_runs,
            owner_service_excludes_wait=True, owner_service_includes_probe_overhead=True,
            future_result_elapsed_includes_unblocked_calls=True, future_ready_instant_ns=None,
            worker_solve_span_includes_lane_overlap=True,
            unknown_owner_gap_ns=None, comparable_clean_benchmark=False)
    return result


def _entry_dependence(path, scope, pipeline):
    from scar.analysis import entry_call_dependence as dependence
    from scar.analysis.sparse_pipeline_scope import _interface

    reference = artifact_reference(path, dependence.SCHEMA)
    witness = read_artifact(reference)
    source = read_artifact(scope['source_scope'])
    shadow = scope['shadow_inputs']
    expected = [row['formal_ingress'] for row in shadow['inherited_live_ins'].values()]
    expected.extend(shadow['rhs_and_scalar_entry_routes'])
    live_ins = witness['live_ins']
    actual = [row['original_route'] for row in live_ins]
    order = lambda route: route['formal']['port']['ordinal']
    _require(witness['entry_calls'] == pipeline['source_graphs']['sparse_solve']
        and witness['selected_body'] == source['f4_loop']['source_body']
        and sorted(expected, key=order) == sorted(actual, key=order),
        'entry dependence does not match the original F4 body and exact live-in ports')
    row, row_link, row_attempt = _interface(read_artifact(scope['source_graph']), scope['loop']['ingress'])
    solve, solve_link, solve_attempt = _interface(read_artifact(witness['entry_calls']),
                                                  witness['original_interface'])
    _require(row_link == solve_link and row['caller_scope'] == solve['caller_scope']
        and row_attempt['operation'] == solve_attempt['operation'] == witness['operation']
        and all(row_attempt['observation'][key] == solve_attempt['observation'][key]
            for key in ('caller_site', 'target_code')),
        'entry dependence does not name the same observed source call and typed interface')
    dependence.check(reference['uri'])
    read_artifact(reference, json_document=False)
    return dict(evidence=reference, original_call=witness['operation'],
        capture_input_edges=[dict(evidence_locator=f'#/live_ins/{index}', source_ports=route['actuals'],
            formal_port=route['formal'], value='source-live-ins', consumer='capture-live-ins')
            for index, route in enumerate(actual)],
        original_return_consumers='#/return_consumers', original_world_out=witness['original_world_out'],
        effect_discharge_proven=False)


@replay_transaction()
def project(recipe_path, calibration_path, release_path, execution_path, *, instrumented_paths=(),
            entry_dependence_path=None):
    """Bind one released execution and same-recipe calibration to original F4 work."""
    refs = {key: artifact_reference(path, schema) for key, path, schema in (
        ('recipe', recipe_path, RECIPE), ('calibration', calibration_path, CALIBRATION),
        ('release', release_path, RELEASE), ('execution', execution_path, EXECUTION))}
    release = read_artifact(refs['release'])
    _require(release['execution_recipe'] == refs['recipe'] and release['cost'] == refs['calibration'],
        'owner evidence must bind the recipe and calibration of the exact release')
    recipe = read_artifact(refs['recipe'])
    scope_ref = recipe['artifacts']['parallel_scope']
    scope = read_artifact(scope_ref)
    pipeline = read_artifact(scope['pipeline_scope'])
    entry_dependencies = None if entry_dependence_path is None else _entry_dependence(
        entry_dependence_path, scope, pipeline)
    recipe = verify_sparse_parallel_recipe(recipe)
    calibration = check_calibration(refs['calibration']['uri'])
    _release_lineage(scope_ref, scope, release)
    _require(calibration['execution_recipe'] == refs['recipe']
        and calibration['calibration_is_section_only'] is True and calibration['full_entry_cost_ns'] is None,
        'owner diagnostics require the original section-scoped calibration')
    team = _execution(read_artifact(refs['execution']), refs['release'], release, recipe)
    memory = recipe['resources']['memory']
    graph = _boundary(recipe, scope, pipeline)
    result = dict(schema=SCHEMA, schema_version=1, inputs=refs, recipe_id=recipe['recipe_id'],
        source_pipeline=scope['pipeline_scope'], owner=recipe['resources']['selected_cpu'],
        iteration=recipe['control']['iteration'], graph=graph,
        existing_storage=dict(detached_snapshot_bound_bytes=memory['owner_detached_snapshot_bound_bytes'],
            pending_worker_input_peak_bytes=team['peak_snapshot_bytes'], physical_reservation=False),
        costs=_costs(calibration, team), legality=dict(
            delete_original_normalization=False, replace_original_publication=False,
            independent_private_compute_only=True, standalone_cxx_scope='PREPARED_INPUT_TO_ORDERED_NUMERIC_OUTPUT',
            automatic_rewrite=False))
    if instrumented_paths:
        result['schema_version'] = 2
        result['costs']['instrumented_original_entry'] = _instrumented_costs(
            instrumented_paths, refs['release'], release, recipe, timeline=entry_dependence_path is not None)
    if entry_dependence_path is not None:
        result['schema_version'] = 3
        graph['outside_recipe']['entry_dependence'] = entry_dependencies
    return result


def check(path):
    """Recreate the complete Stage II overlay from its original evidence."""
    record = read_artifact(artifact_reference(path, SCHEMA))
    refs = record['inputs']
    live = record['costs'].get('instrumented_original_entry', {})
    ingress = record['graph']['outside_recipe'].get('entry_dependence', {})
    rebuilt = project(*(refs[key]['uri'] for key in ('recipe', 'calibration', 'release', 'execution')),
                      instrumented_paths=[ref['uri'] for ref in live.get('observations', ())],
                      entry_dependence_path=ingress['evidence']['uri'] if ingress else None)
    _require(record == rebuilt, 'owner cost and legality projection differs from cold replay')
    return rebuilt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', type=Path)
    parser.add_argument('--owner-observation', action='append', default=[], type=Path)
    parser.add_argument('--entry-dependence', type=Path)
    for name in ('recipe', 'calibration', 'release', 'execution', 'out'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    if args.check:
        print(check(args.check)['recipe_id'])
        return
    if not all((args.recipe, args.calibration, args.release, args.execution, args.out)):
        parser.error('provide recipe, calibration, release, execution and output')
    result = project(args.recipe, args.calibration, args.release, args.execution,
                     instrumented_paths=args.owner_observation, entry_dependence_path=args.entry_dependence)
    with args.out.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()

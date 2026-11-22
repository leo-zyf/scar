"""Project source-bound fixed-grid cost coverage without ranking the entry."""

from pathlib import Path
from statistics import median

from scar.analysis.replay_transaction import replay_transaction
from scar.ir.execution_recipe import read_artifact
from scar.optimization.cpu_fixed_grid_cli_review import SCHEMA as REVIEW, derive_cpu_fixed_grid_cli_review

SCHEMA = 'scar.cpu-fixed-grid-lifecycle-cost'
_GRAPH = 'scar.cpu-fixed-grid-lifecycle'
_PYTHON = 'scar.cpu-fixed-grid-python-lifecycle-observation'
_NATIVE = 'scar.cpu-fixed-grid-native-lifecycle'
_LOADER = 'scar.cpu-fixed-grid-loader-proxy'
_PY, _CPP = 'original_python', 'native_cpp'
_PYTHON_WINDOWS = {
    'startup': ('launch', 'initialization'), 'imports': ('initialization',),
    'ingress': ('ingress', 'provider_setup'), 'loading': ('@loads',), 'execution': ('recurrence',),
    'terminal': ('terminal',), 'file_publication': ('file_publication',),
    'stdout_submission': ('stdout_publication',), 'teardown': ('teardown',),
    'unattributed': (), 'total': (),
}
_NATIVE_WINDOWS = {
    'launcher_to_main': ('launch', 'initialization'), 'main_ingress': ('ingress',),
    'runtime_team_setup': ('provider_setup',), 'input_preflight': ('io_preflight',),
    'input_ingestion': ('@loads',), 'recurrence': ('recurrence',),
    'file_publication': ('file_publication',), 'stdout_submission': ('stdout_publication',),
    'owner_local_teardown': ('teardown',), 'main_end_to_reap': ('teardown',),
}


def _require(condition, detail):
    if not condition:
        raise ValueError('CPU fixed-grid lifecycle cost: ' + detail)


def _context(review_ref, lifecycle_ref):
    from scar.analysis.cpu_fixed_grid_lifecycle import verify_cpu_fixed_grid_lifecycle

    _require(review_ref['schema'] == REVIEW and lifecycle_ref['schema'] == _GRAPH,
             'exact source profile review and lifecycle graph required')
    review = read_artifact(review_ref)
    graph = verify_cpu_fixed_grid_lifecycle(lifecycle_ref)
    _require(graph['delivery'] == review['delivery']
             and all(graph['effects'][key] == review['bridge'][key] for key in ('schema', 'sha256', 'size_bytes'))
             and graph['source'] == review['source'] and graph['project'] == review['source_cli']['project']
             and set(graph['routes']) == {_PY, _CPP}, 'review and lifecycle evidence describe different programs')
    delivery = read_artifact(review['delivery'])
    main = read_artifact(delivery['main_chain'])
    scope = read_artifact(delivery['domain'])['scope']
    options = read_artifact(main['variant_discovery'])['options']
    option, = (row for row in options if row['id'] == review['option_id'])
    _require(review == derive_cpu_fixed_grid_cli_review(option, scope, review['bridge'], review['delivery'])
             and graph['target'] == delivery['target'], 'stored review differs from its pinned original option')
    return review, graph, delivery


def _native_step(graph, delivery):
    resources = read_artifact(graph['resources'])
    project = delivery['project']
    manifest = read_artifact(project)
    member = 'validation/calibration.json'
    calibration_ref = dict(uri=str(Path(project['uri']).parent / member),
                           schema='scar.tensor-body-observation', **manifest['files'][member])
    calibration = read_artifact(calibration_ref)
    binding = read_artifact(resources['binding'])
    loop, costs = resources['loop'], binding['costs']
    step, steps = costs['native_body_per_step_estimate_ns'], delivery['steps']
    _require(resources['project'] == project and resources['target'] == delivery['target']
             and resources['argv'] == delivery['argv'] and loop['steps'] == steps
             and binding['placement'] == calibration['placement'] == delivery['native_placement']
             and costs['body_estimate_scope'] == 'REPLAYED_NATIVE_STEP_SCHEDULE_ONLY'
             and type(step) is type(steps) is int and min(step, steps) > 0
             and loop['native_step_schedule_estimate_ns'] == step and loop['body_only_estimate_ns'] == step * steps,
             'physical step timing did not bind the original counted recurrence and placement')
    files = [{key: value for key, value in row.items() if key != 'port'} for row in delivery['file_inputs']]
    matched = files == calibration['inputs']
    return dict(
        basis='ARCHIVED_CPU_PHYSICAL_STEP_CALIBRATION', binding=resources['binding'], calibration=calibration_ref,
        graph_operation=f'#/routes/{_CPP}/operations/recurrence', steps=steps, step_estimate_ns=step,
        calibration_repeats=calibration['repeats'], calibration_warmups=calibration['discarded_warmups'],
        calibration_matches_exact_inputs=matched, recurrence_point_estimate_ns=step * steps if matched else None,
        input_logical_payload_bytes=costs['logical_input_payload_bytes'],
        final_logical_payload_bytes=costs['logical_final_payload_bytes'],
        complete_process_interval_ns=dict(lower=None, upper=None), physical_duration_bound_proven=False)


def _statistics(samples):
    _require(type(samples) is list and bool(samples)
             and all(type(value) is int and value >= 0 for value in samples),
             'diagnostic requires nonnegative integer phase spans')
    return dict(samples_ns=samples, min_ns=min(samples), median_ns=median(samples), max_ns=max(samples))


def _project_windows(route, operations, stats, mapping, *, locator_prefix):
    loads = [key for key in operations if key.startswith('load:')]
    output = {}
    for name, values in stats.items():
        targets = [key for key in mapping[name] if key != '@loads']
        if '@loads' in mapping[name]:
            targets.extend(loads)
        _require(set(targets) <= operations.keys(), 'diagnostic phase names operations outside its route')
        output[name] = dict(
            **values, evidence=f'{locator_prefix}/{name}',
            graph_operations=[f'#/routes/{route}/operations/{key}' for key in targets],
            covers_exactly_one_operation=False)
    return output


def _same_context(observation, review, delivery):
    _require(observation['delivery'] == review['delivery']
             and all(observation[key] == delivery[key]
                     for key in ('source', 'project', 'argv', 'cpu_ids', 'working_directory'))
             and observation.get('environment_sha256', delivery['environment_sha256'])
                 == delivery['environment_sha256'],
             'diagnostic must be replayed against the exact original entry and project launch')


def _python(reference, review, graph, delivery):
    from scar.compiler.observe_cpu_fixed_grid_python_lifecycle import verify_cpu_fixed_grid_python_lifecycle

    _require(reference['schema'] == _PYTHON, 'source-marked Python observation required')
    data = verify_cpu_fixed_grid_python_lifecycle(reference)
    _same_context(data, review, delivery)
    phases = data['phases_ns']
    _require(set(phases) == _PYTHON_WINDOWS.keys(), 'unsupported original Python diagnostic phases')
    for values in phases.values():
        _require(all(values[key] == expected for key, expected in _statistics(values['samples_ns']).items()),
                 'Python diagnostic statistics differ from retained samples')
    import_only = data['import_only']
    return dict(artifact=reference, basis='SOURCE_MARKED_ORIGINAL_PYTHON_PROCESS',
        windows=_project_windows(_PY, graph['routes'][_PY]['operations'], phases, _PYTHON_WINDOWS,
                                 locator_prefix='#/phases_ns'),
        import_only=None if import_only is None else dict(evidence='#/import_only',
            basis='ISOLATED_SOURCE_IMPORT_DIAGNOSTIC', phases_ns=import_only['phases_ns'],
            fills_production_startup_cost=False),
        clean_benchmark=False, observation_used_as_model_bound=False)


def _native(reference, review, graph, delivery):
    from scar.compiler.observe_cpu_fixed_grid_native_lifecycle import verify_cpu_fixed_grid_native_lifecycle

    _require(reference['schema'] == _NATIVE, 'source-marked C++ sibling observation required')
    data = verify_cpu_fixed_grid_native_lifecycle(reference)
    _same_context(data, review, delivery)
    _require(data['target'] == delivery['target'] and data['steps'] == delivery['steps']
             and data['file_inputs'] == delivery['file_inputs']
             and data['native_executable'] == delivery['native_executable']
             and data['diagnostic_executable'] != data['native_executable']
             and data['diagnostic_only'] is True, 'C++ observation crossed the admitted diagnostic executable or inputs')
    windows, stats = data['windows'], {}
    _require(set(windows) <= _NATIVE_WINDOWS.keys() and 'recurrence' in windows,
             'unsupported native source windows')
    for name, (start, stop) in windows.items():
        intervals = []
        for row in data['samples']:
            marks = dict(row['markers_ns'], launcher_start_ns=row['launcher_start_ns'],
                         process_reaped_ns=row['process_reaped_ns'])
            intervals.append(marks[stop] - marks[start])
        stats[name] = _statistics(intervals)
    return dict(artifact=reference, basis='VERIFIED_SOURCE_MARKED_CXX_SIBLING',
        windows=_project_windows(_CPP, graph['routes'][_CPP]['operations'], stats, _NATIVE_WINDOWS,
                                 locator_prefix='#/windows'),
        samples_evidence='#/samples', limits=data['limits'],
        clean_benchmark=False, observation_used_as_model_bound=False)


def _loader(reference, review, delivery):
    from scar.compiler.calibrate_cpu_fixed_grid_loader import verify_cpu_fixed_grid_loader

    _require(reference['schema'] == _LOADER, 'verified fixed-grid native loader proxy required')
    proxy = verify_cpu_fixed_grid_loader(reference)
    _require(proxy['delivery'] == review['delivery']
             and all(proxy[key] == delivery[key] for key in ('project', 'source', 'cpu_ids', 'working_directory'))
             and proxy['loader_and_arg_guard_and_reap_are_combined'] is True
             and proxy['complete_entry_executed'] is False and proxy['physical_cost_bound'] is False,
             'loader diagnostic crossed the original native launch scope')
    stats = _statistics(proxy['timing_ns']['samples_ns'])
    _require(all(proxy['timing_ns'][key] == value for key, value in stats.items()),
             'loader proxy interval differs from the observed samples')
    return dict(artifact=reference, basis='EXACT_NATIVE_LAUNCH_ARGC_REJECTION_AND_REAP_PROXY',
                observed_argv=proxy['argv'], timing_evidence='#/timing_ns', timing_ns=stats,
                includes_loader_and_guard_and_reap=True, replaces_production_startup_cost=False,
                added_to_diagnostic_or_model=False)


def _deviation(model, native):
    estimate = model['recurrence_point_estimate_ns']
    if estimate is None or native is None:
        return None
    sample = native['windows']['recurrence']
    return dict(graph_operation=model['graph_operation'], model_point_ns=estimate,
        diagnostic_median_ns=sample['median_ns'], diagnostic_minus_model_point_ns=sample['median_ns'] - estimate,
        diagnostic_sample_range_ns=[sample['min_ns'], sample['max_ns']],
        basis='POINT_COMPARISON_TO_INSTRUMENTED_NATIVE_RECURRENCE', updates_step_model=False)


def _partial_proxy_sum(model, loader):
    point = model['recurrence_point_estimate_ns']
    if point is None or loader is None:
        return None
    proxy = loader['timing_ns']['median_ns']
    return dict(
        basis='CONDITIONAL_SUM_OF_INDEPENDENT_LOOP_POINT_AND_EARLY_EXIT_PROXY',
        recurrence_point_estimate_ns=point, loader_guard_reap_proxy_median_ns=proxy,
        sum_ns=point + proxy,
        production_loader_proxy_equivalence_proven=False,
        production_timeline_partition_proven=False,
        unpriced_production_operations=['INPUT_INGESTION_AND_PAGE_STATE', 'PROVIDER_AND_TEAM_SETUP',
                                        'TERMINAL_AND_FILE_PUBLICATION', 'STDOUT_DRAIN_AND_PRODUCTION_TEARDOWN'],
        full_entry_model_bound_proven=False)


def _route(graph, route, diagnostic, estimate):
    open_operations = (key for key in graph['routes'][route]['operations']
                       if key != 'recurrence' or estimate is None)
    return dict(diagnostic=diagnostic, forward_recurrence_estimate_ns=estimate,
                unpriced_operations=[f'#/routes/{route}/operations/{key}' for key in open_operations])


def derive_cpu_fixed_grid_lifecycle_cost(review_ref, *, lifecycle_ref, python_diagnostic=None,
                                         native_diagnostic=None, loader_probe=None):
    """Bind measured markers, independent probes and native step cost to one graph."""
    with replay_transaction():
        review, graph, delivery = _context(review_ref, lifecycle_ref)
        model = _native_step(graph, delivery)
        python = None if python_diagnostic is None else _python(python_diagnostic, review, graph, delivery)
        native = None if native_diagnostic is None else _native(native_diagnostic, review, graph, delivery)
        loader = None if loader_probe is None else _loader(loader_probe, review, delivery)
    numerical = model['recurrence_point_estimate_ns'] is not None
    return dict(
        schema=SCHEMA, schema_version=1, review=review_ref, lifecycle=lifecycle_ref,
        scope_sha256=review['scope_sha256'], delivery=review['delivery'], resources=graph['resources'],
        target=graph['target'], source=graph['source'], project=graph['project'],
        python_diagnostic=python_diagnostic, native_diagnostic=native_diagnostic, loader_probe=loader_probe,
        status='CONDITIONAL_NATIVE_STEP_POINT' if numerical else 'NATIVE_STEP_INPUT_SCOPE_OPEN',
        routes={_PY: _route(graph, _PY, python, None),
                _CPP: _route(graph, _CPP, native, model['recurrence_point_estimate_ns'])},
        independent_loader_probe=loader, native_step_model=model, recurrence_model_deviation=_deviation(model, native),
        conditional_partial_proxy_sum=_partial_proxy_sum(model, loader),
        coverage_frontier=[
            dict(role='PROCESS_STARTUP', status='UNPRICED',
                 blockers=['PRODUCTION_ELF_AND_PYTHON_INITIALIZATION', 'PROVIDER_LOAD_AND_TEAM_STARTUP']),
            dict(role='INPUT_MATERIALIZATION', status='LOGICAL_PAYLOAD_AND_STORAGE_ROUTE_ONLY',
                 blockers=['FILE_PAGE_CACHE_AND_FIRST_TOUCH_STATE', 'STORAGE_AND_DRAM_PHYSICAL_TRAFFIC']),
            dict(role='PUBLICATION', status='SCOPED_CONTENT_AND_MODE_EVIDENCE',
                 exact_observed_mode=review['source_cli']['exact_publication_mode'],
                 blockers=['FILE_MODE_AND_METADATA_RUNTIME_COST', 'DURABILITY_AND_FAILURE_BEHAVIOR']),
            dict(role='TEARDOWN', status='UNPRICED', blockers=['STDOUT_DRAIN_AND_PROCESS_REAP']),
            dict(role='NUMERICAL', status='NATIVE_CONDITIONAL_POINT_ONLY' if numerical else 'SCOPE_OPEN',
                 blockers=['ORIGINAL_PYTHON_NUMERICAL_MODEL', 'STRICT_NATIVE_DURATION_BOUNDS'])],
        diagnostic_windows_added_to_forward_model=False, diagnostics_used_for_gain_gate=False,
        whole_entry_model_interval_ns=dict(lower=None, upper=None), physical_duration_bound_proven=False,
        whole_entry_ranked=False, execution_default='NO_OP', target_executed=False, deployment_authorized=False)


def verify_cpu_fixed_grid_lifecycle_cost(reference):
    """Replay the original source review, graph and scoped cost evidence."""
    saved = read_artifact(reference)
    _require(reference['schema'] == SCHEMA and type(saved['schema_version']) is int and saved['schema_version'] == 1,
             'unsupported fixed-grid lifecycle cost report')
    replay = derive_cpu_fixed_grid_lifecycle_cost(
        saved['review'], lifecycle_ref=saved['lifecycle'], python_diagnostic=saved['python_diagnostic'],
        native_diagnostic=saved['native_diagnostic'], loader_probe=saved['loader_probe'])
    _require(saved == replay, 'saved fixed-grid lifecycle cost differs at cold replay')
    return replay

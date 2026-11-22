"""Place measured components on the native route timeline and expose unpriced work."""

from statistics import median

from scar.analysis import cpu_array_native_route_lifecycle as lifecycle
from scar.analysis.cpu_array_native_route import verify_cpu_array_native_route
from scar.analysis.replay_transaction import replay_transaction
from scar.ir.execution_recipe import read_artifact
from . import cpu_array_native_route_component_cost as component_cost
from .cpu_entry_forward_cost import _interval


SCHEMA = 'scar.cpu-array-native-route-forward'
_ROUTES = ('retained_process', 'grouped_openmp')
_DISPATCH = ('team_setup', 'worker_setup', 'task_ingress', 'result_handoff', 'join')
_PROFILE = ('task_prepare', 'task_allocate', 'numerical_body', 'task_result')
_OTHER = ('launch', 'ingress', 'reduction', 'file_publication', 'stdout_publication', 'teardown')


def _need(condition, reason):
    if not condition:
        raise ValueError('CPU native route timeline: ' + reason)


def _coverage(graph, prices):
    operations = graph['operations']
    _need(set(operations) == set(_DISPATCH + _PROFILE + _OTHER),
          'every retained lifecycle operation needs one cost coverage role')
    profile, dispatch = (prices[key] for key in ('conditional_numeric_critical_interval_ns',
                                                 'transport_proxy_interval_ns'))
    has_profile = profile is not None
    _need(prices['executable'] == graph['executable'], 'cost and route use different C++ executables')
    report = dict(
        included_profile_operations=list(_PROFILE), priced_as_one_service=has_profile,
        empty_dispatch_proxy_operations=list(_DISPATCH),
        exposed_unpriced_operations=[*(_PROFILE if not has_profile else ()), *_OTHER],
        explicitly_unpriced_interactions=['LOADED_DISPATCH_OVERLAP', 'PHYSICAL_FIRST_TOUCH_AND_NUMA',
                                          'CONTENTION_DIFFERENCE_ACROSS_ROUTES'],
        timing_assumptions=prices['assumptions'], lanes=prices['per_lane'],
        task_service_critical_path_ns=None if not has_profile else _interval(profile),
        empty_dispatch_proxy_ns=_interval(dispatch),
        conditional_body_proxy_ns=prices['conditional_additive_component_interval_ns'],
        worker_overlap_in_cost_model=(
            'ORDERED_OWNER_RETIREMENT_WITH_PER_LANE_REFILL'
            if graph['mode'] == 'LOCAL_PROCESS_ORDERED_SOCKET_WINDOW'
            else 'MAX_OVER_STATIC_ORDINAL_LANES'),
        cost_components_share_an_entry_clock=False, whole_entry_interval_ns=dict(lower=None, upper=None))
    if 'loaded_dispatch_span_ns' in prices:
        span = _interval(prices['loaded_dispatch_span_ns'])
        model = _interval(prices['conditional_additive_component_interval_ns'])
        report.update(loaded_dispatch_observation_ns=span, empty_proxy_added_to_loaded_dispatch=False,
                      loaded_dispatch_minus_model_point_ns=span['estimate_ns'] - model['estimate_ns'],
                      measured_task_service_replay=prices['loaded_service_replay'])
    return report


def _statistics(samples):
    return dict(samples_ns=samples, median_ns=median(samples), min_ns=min(samples), max_ns=max(samples))


def _diagnostic_report(reference, graph, package_ref, checked_package):
    from scar.compiler.cpu_array_native_route_phase_probe import _verify

    diagnostic = _verify(reference, checked_package)
    plan = read_artifact(diagnostic['plan'])
    _need((checked_package is not None or diagnostic['route_package'] == package_ref) and plan['proof'] == graph['proof']
          and diagnostic['target'] == graph['hardware'] and diagnostic['clean_benchmark'] is False,
          'diagnostic siblings must bind the same C++ package, source proof and hardware')
    routes = {}
    for name in _ROUTES:
        arm = diagnostic['routes'][name]
        _need(arm['original_executable'] == graph['routes'][name]['executable'],
              'diagnostic was not taken from the routed executable')
        windows = {key: [] for key in ('parent_launch_to_main', 'native_ingress', 'team_and_join',
                                       'report_statements', 'after_report_to_parent_reap')}
        for row in arm['samples']:
            marks = row['owner_markers_ns']
            clocks = read_artifact(row['clocks'])
            for key, begin, end in (
                ('parent_launch_to_main', clocks['runner_job_start_ns'], marks['MAIN_ENTER']),
                ('native_ingress', marks['MAIN_ENTER'], marks['DISPATCH_BEGIN']),
                ('team_and_join', marks['TEAM_ENTER'], marks['TEAM_RETURN']),
                ('report_statements', marks['PUBLICATION_BEGIN'], marks['PUBLICATION_END']),
                ('after_report_to_parent_reap', marks['PUBLICATION_END'], clocks['runner_reaped_ns']),
            ):
                windows[key].append(end - begin)
        routes[name] = {key: _statistics(values) for key, values in windows.items()}
    return dict(artifact=reference, routes=routes, sample_count=diagnostic['sample_count'],
                overlapping_task_intervals_retained_in_artifact=True, costs_calibrated_from_diagnostics=False)


def _publication_report(reference, graph, routes):
    from scar.compiler.calibrate_cpu_array_native_route_boundary import _verify

    publication = _verify(reference, graph)
    proof = read_artifact(graph['proof'])
    _need(publication['proof'] == graph['proof'] and publication['exact_stdout_and_file_bytes']
          and publication['cpu_ids'] == proof['resources']['launch_cpu_ids']
          and publication['normalized_oracle'] == proof['workload']['normalized_oracle'],
          'publication clocks differ from the admitted source, output contract or CPU scope')
    results = {}
    for name in _ROUTES:
        arm, graph_route = publication['routes'][name], graph['routes'][name]
        _need(arm['original_executable'] == graph_route['executable']
              and any(row['source'] == 'file_publication' and row['target'] == 'stdout_publication'
                      for row in graph_route['edges']), 'publication must preserve the exact file then stdout order')
        windows = {key: _statistics([row['markers_ns'][end] - row['markers_ns'][begin] for row in arm['samples']])
                   for key, (begin, end) in publication['windows'].items()}
        results[name] = windows
        body = routes[name].get('loaded_dispatch_observation_ns')
        if body is not None:
            report = windows['publication_statements']
            results[name]['conditional_accounted_windows_ns'] = {
                'lower_ns': body['lower_ns'] + report['min_ns'],
                'estimate_ns': body['estimate_ns'] + report['median_ns'],
                'upper_ns': body['upper_ns'] + report['max_ns'],
            }
    return dict(artifact=reference, routes=results, sample_count=publication['sample_count'],
                composed_windows=['loaded_dispatch', 'native_publication_statements'],
                composition_basis='SEQUENTIAL_COMPONENT_PROXIES_WITH_PRODUCER_HISTORY_ASSUMPTION',
                excluded=['LOADER_AND_INGRESS', 'ORDERED_REDUCTION', 'STDOUT_DRAIN_AND_PROCESS_EXIT', 'FILE_DURABILITY'],
                physical_entry_bounds=False)


def _project(lifecycle_ref, component_ref, graph, components, package_ref, diagnostics, checked_package=None,
             *, publication=None):
    _need(graph['proof'] == components['proof'] and graph['hardware'] == components['resources']['hardware']
          and graph['scope'] == components['scope'] and set(graph['routes']) == set(components['routes'])
          == set(_ROUTES), 'lifecycle and component inputs refer to different route scopes')
    if package_ref is not None:
        package = read_artifact(package_ref)
        plan = read_artifact(package['plan'])
        _need(package_ref['schema'] == 'scar.cpu-array-native-route-package'
              and plan['proof'] == graph['proof'] and package['hardware'] == graph['hardware']
              and all(package['routes'][mode]['executable'] == graph['routes'][mode]['executable']
                      for mode in _ROUTES), 'optional package differs from the analyzed native routes')
    _need(diagnostics is None or package_ref is not None, 'diagnostics require their exact launch package')
    observed = None if diagnostics is None else _diagnostic_report(diagnostics, graph, package_ref, checked_package)
    routes = {name: _coverage(graph['routes'][name], components['routes'][name]) for name in _ROUTES}
    process, openmp = (routes[name]['conditional_body_proxy_ns'] for name in _ROUTES)
    comparison = None
    if process is not None and openmp is not None:
        comparison = dict(process_minus_grouped_point_ns=process['estimate_ns'] - openmp['estimate_ns'],
                          process_low_minus_grouped_high_ns=process['lower_ns'] - openmp['upper_ns'],
                          grouped_low_minus_process_high_ns=openmp['lower_ns'] - process['upper_ns'],
                          prices_cover_different_contention_assumptions=components.get('matched_probe') is None,
                          decision_authority=False)
    result = dict(schema=SCHEMA, schema_version=1, proof=graph['proof'], lifecycle=lifecycle_ref,
                components=component_ref, package=package_ref, diagnostics=observed, routes=routes,
                conditional_body_comparison=comparison, scope=graph['scope'], hardware=graph['hardware'],
                status='CONDITIONAL_SUBGRAPH_MODEL' if comparison is not None else 'PARTIAL_SUBGRAPH_MODEL',
                whole_entry_cost_closed=False, physical_duration_bound=False, whole_entry_ranked=False,
                target_executed=False, execution_default='NO_OP', deployment_authorized=False)
    if publication is not None:
        result['publication'] = _publication_report(publication, graph, routes)
    return result


def derive_cpu_array_native_route_forward(lifecycle_ref, component_ref, *, package_ref=None, diagnostics=None,
                                         publication=None):
    """Compare matched body models while keeping both full-entry costs unresolved."""
    return _derive(lifecycle_ref, component_ref, package_ref, diagnostics, None, publication=publication)


def _derive(lifecycle_ref, component_ref, package_ref, diagnostics, checked_package, *, publication=None):
    with replay_transaction():
        graph, components = read_artifact(lifecycle_ref), read_artifact(component_ref)
        proof_ref = graph['proof']
        _need(lifecycle_ref['schema'] == lifecycle.SCHEMA and component_ref['schema'] == component_cost.SCHEMA
              and proof_ref == components['proof'], 'expected one exact proof for lifecycle and component inputs')
        proof, checked_package = _read_proof(proof_ref, package_ref, checked_package)
        _need(lifecycle._canonical(graph) == lifecycle._canonical(lifecycle._project(proof_ref, proof))
              and components == component_cost._project(proof_ref, proof, components.get('grouped_probe'),
                    matched_probe=components.get('matched_probe'), checked_package=checked_package),
              'saved lifecycle or component costs differ after replay')
        return _project(lifecycle_ref, component_ref, graph, components, package_ref, diagnostics, checked_package,
                        publication=publication)


def _read_proof(proof_ref, package_ref, checked_package=None):
    if package_ref is None:
        _need(checked_package is None, 'checked package without a requested package')
        return verify_cpu_array_native_route(proof_ref), None
    if checked_package is None:
        from scar.codegen.cpu_native_route_package import verify_cpu_array_native_route_package

        checked_package = package_ref, verify_cpu_array_native_route_package(package_ref)
    checked_ref, package = checked_package
    _need(checked_ref == package_ref and read_artifact(package['plan'])['proof'] == proof_ref,
          'checked package does not cover the required source proof')
    return read_artifact(proof_ref), checked_package


def verify_cpu_array_native_route_forward(reference):
    return _verify(reference)


def _verify(reference, checked_package=None):
    saved = read_artifact(reference)
    _need(reference['schema'] == SCHEMA and saved['schema_version'] == 1, 'unsupported forward cost report')
    diagnostics = saved['diagnostics']
    publication = saved.get('publication')
    replay = _derive(saved['lifecycle'], saved['components'], saved['package'],
                     None if diagnostics is None else diagnostics['artifact'], checked_package,
                     publication=None if publication is None else publication['artifact'])
    _need(replay == saved, 'native route timeline differs at replay')
    return replay

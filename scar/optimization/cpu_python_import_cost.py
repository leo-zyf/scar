"""Replay Python import attribution without inferring a startup time bound."""

from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.cpu-python-import-cost'
IMPORT_GRAPH = 'scar.f4-cpu-python-import-graph'
OBSERVATION = 'scar.cpu-original-import-observation'


def _require(condition, detail):
    if not condition:
        raise ValueError('CPU Python import cost: ' + detail)


def _task_cost(graph_ref, names, run_index, task_index, task):
    pointer = f'#/runs/{run_index}/tasks/{task_index}'
    timed, cached, open_clocks, total, peak = 0, 0, 0, 0, None
    for index, event in enumerate(task['events']):
        module = event['module']
        exclusive = event['self_us']
        inclusive = event['cumulative_us']
        status = event['clock_status']
        _require(type(module) is int and 0 <= module < len(names) and type(event['cached']) is bool,
                 'import event needs an exact module and cache observation')
        if event['cached']:
            _require(status == 'CACHED_NO_TIMING' and exclusive is None and inclusive is None,
                     'cached import has no retained timing fields')
            cached += 1
            continue
        if status == 'CLOCK_ACCOUNTING_OPEN':
            open_clocks += 1
            continue
        _require(status == 'REPORTED_SELF_AND_CUMULATIVE_US'
                 and type(exclusive) is type(inclusive) is int and 0 <= exclusive <= inclusive,
                 'invalid exclusive and inclusive import accounting')
        timed += 1
        total += exclusive
        if peak is None or exclusive > peak['self_us']:
            peak = dict(module=names[module], self_us=exclusive, cumulative_us=inclusive,
                        evidence=dict(artifact=graph_ref, locator=f'{pointer}/events/{index}'))
    return dict(task_id=task['task_id'], timed_imports=timed, cached_imports=cached, clock_open_imports=open_clocks,
                recorded_self_us=total if timed else None, largest_self_event=peak,
                evidence=dict(artifact=graph_ref, locator=pointer))


def _process_cost(process, tasks, evidence):
    recorded = [row['recorded_self_us'] for row in tasks if row['recorded_self_us'] is not None]
    peaks = [row['largest_self_event'] for row in tasks if row['largest_self_event'] is not None]
    return dict(process_id=process, evidence=evidence, tasks=[row['evidence'] for row in tasks],
                timed_imports=sum(row['timed_imports'] for row in tasks),
                cached_imports=sum(row['cached_imports'] for row in tasks),
                clock_open_imports=sum(row['clock_open_imports'] for row in tasks),
                recorded_self_us=sum(recorded) if recorded else None,
                largest_self_event=max(peaks, key=lambda row: row['self_us'], default=None))


def _run_cost(graph_ref, names, index, run, observed):
    start, reap = observed['started_ns'], observed['reaped_ns']
    _require(type(run['ordinal']) is int and run['ordinal'] == observed['ordinal'] == index
             and type(start) is type(reap) is int and 0 < start < reap,
             'import accounting must name one exact observed process-launch run')
    processes, unassigned = {}, []
    for task_index, task in enumerate(run['tasks']):
        process = task['process_id']
        result = _task_cost(graph_ref, names, index, task_index, task)
        if process is None:
            unassigned.append(result)
        else:
            _require(type(process) is int and process > 0,
                     'invalid graph-proven process instance identity')
            processes.setdefault(process, []).append(result)
    _require(len(run['processes']) == len(processes), 'inconsistent process grouping in the import graph')
    accounting = []
    for process_index, group in enumerate(run['processes']):
        process = group['process_id']
        tasks = processes.pop(process)
        _require(sorted(task['task_id'] for task in tasks) == group['task_ids'], 'import process task membership differs')
        pointer = dict(artifact=graph_ref, locator=f'#/runs/{index}/processes/{process_index}')
        accounting.append(_process_cost(process, tasks, pointer))
    return dict(ordinal=index, process_accounting=accounting,
                unattributed_task_accounting=unassigned, diagnostic_launch_reap_ns=reap - start,
                per_process_timing_complete=False,
                all_captured_timed_rows_process_attributed=not any(
                    row['timed_imports'] or row['clock_open_imports'] for row in unassigned))


def derive_cpu_python_import_cost(import_graph_ref, observation_ref):
    """Project one original-side diagnostic with process-local event accounting."""
    from scar.analysis.cpu_python_import_graph import verify_cpu_python_import_graph

    _require(import_graph_ref['schema'] == IMPORT_GRAPH and observation_ref['schema'] == OBSERVATION,
             'original import graph and observation evidence required')
    graph = verify_cpu_python_import_graph(import_graph_ref)
    _require(graph['observation'] == observation_ref, 'graph belongs to another import observation')
    observation = read_artifact(observation_ref)
    names = [module['name'] for module in graph['modules']]
    runs = [_run_cost(import_graph_ref, names, index, run, observed)
            for index, (run, observed) in enumerate(zip(graph['runs'], observation['runs'], strict=True))]
    return dict(schema=SCHEMA, schema_version=1, import_graph=import_graph_ref, observation=observation_ref,
        scope_sha256=graph.get('scope_sha256'), side='ORIGINAL_PYTHON_ENTRY', status='OBSERVED_IMPORT_DIAGNOSTIC_ONLY',
        timing_basis='SUM_OF_ACCOUNTED_EXCLUSIVE_IMPORT_ROWS_WITHIN_EACH_IDENTIFIED_PROCESS',
        inclusive_times_aggregated=False, process_totals_aggregated=False, self_time_establishes_cpu_service=False,
        runs=runs, startup_latency_ns=None, critical_import_path_ns=None, import_wall_interval_ns=None,
        import_initialization_effects_closed=False, removable_imports_proven=False,
        observer_interference_quantified=False, cross_side_usage='DIAGNOSTIC_CONTEXT_ONLY',
        lifecycle_frontier=['STARTUP_CUT_UNOBSERVED', 'PER_IMPORT_WALL_INTERVALS_UNOBSERVED',
                            'PROCESS_AND_THREAD_OVERLAP_OPEN', 'IMPORT_RESOLUTION_AND_INITIALIZATION_EFFECTS_OPEN'],
        whole_entry_model_bound_ns=dict(lower=None, upper=None), whole_entry_ranked=False,
        clean_benchmark=False, execution_default='NO_OP', deployment_authorized=False)


def verify_cpu_python_import_cost(cost_ref):
    saved = read_artifact(cost_ref)
    _require(cost_ref['schema'] == SCHEMA and type(saved['schema_version']) is int and saved['schema_version'] == 1,
             'unsupported import cost report')
    replay = derive_cpu_python_import_cost(saved['import_graph'], saved['observation'])
    _require(replay == saved, 'import diagnostic costs differ at cold replay')
    return replay

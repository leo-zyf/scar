"""Plan one resident owner for a suspended recursive generator and its probes."""

import argparse
import json
from pathlib import Path
import re

from scar.analysis import scalar_recursive_driver as source_driver
from scar.ir.execution_recipe import artifact_reference, make_recipe, read_artifact
from scar.optimization import scalar_recursive_scope_plan as owner_plan
from scar.optimization import scalar_recursive_validation as prepared_assessment
from scar.optimization import scalar_sorted_transition_cost as source_cost
from scar.optimization.schedule import LogicalValue, Materialization, ResourceSchedule, Task, TaskKind


SCHEMA = 'scar.scalar-recursive-driver-plan'
PROFILE = 'scalar-recursive-driver-owner-cpu'
BENCHMARK = 'scar.scalar-recursive-uci-paired-run'
COMPARISON = 'scar.scalar-recursive-uci-paired-selection'
STUDY = 'scar.scalar-recursive-driver-source-study'
_MODULE = 'scar.runtime.scalar_recursive_driver_study'
_ARMS = ('original', 'driver_native')
_ARTIFACT_FLAGS = dict(library=('--library', '--sha256'), source_driver=('--source-driver', '--driver-sha256'),
                       ui_contract=('--ui-contract', '--ui-sha256'))


def _require(condition, message):
    if not condition:
        raise ValueError('recursive driver placement: ' + message)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _schedule(region_id):
    # One symbolic pass through the producing path. The generator and its
    # suspended continuation carry state through the control graph in the recipe.
    scope = '$DRIVER_PROBE'
    values = (LogicalValue('session_inputs', call_scope=scope),
              LogicalValue('probe_output', producer='probe', escape=True, call_scope=scope))
    materials = (Materialization('session_inputs', 'session_inputs', 'CPU_OWNER', external_ready=True,
                                 ownership='EXTERNAL'),
                 Materialization('probe_output', 'probe_output', 'CPU_OWNER', ready_task='probe',
                                 allocated_by='output_slot'))
    tasks = (Task('start_or_resume', TaskKind.COMPUTE, reads=('session_inputs',)),
             Task('output_slot', TaskKind.ALLOCATION, writes=('probe_output',), depends_on=('start_or_resume',)),
             Task('probe', TaskKind.COMPUTE, reads=('session_inputs',), writes=('probe_output',),
                  depends_on=('output_slot',), attrs={'recursion': 'SAME_OWNER', 'service_duration_ns': None}),
             Task('publish_yield', TaskKind.EXPORT, reads=('probe_output',), depends_on=('probe',)))
    return ResourceSchedule(region_id, values, materials, tasks, call_scope=scope,
        proof=dict(scope='SYMBOLIC_NORMAL_YIELD_PATH'), unresolved=(
            dict(reason='OWNER_ASSOCIATION_AND_RUNTIME_EFFECTS_OPEN'),
            dict(reason='REGION_MEMORY_AND_WHOLE_ENTRY_COST_UNKNOWN'),))


def _transition(event, before, after, effects):
    return dict(event=event, before=before, after=after, ordered_effects=effects)


def _field_roles(driver, model):
    source, scope = driver['driver'], model['scope']
    probe, yielded, initialization = driver['probe'], driver['yield'], driver['initialization']
    _require(driver['source'] == model['source']
        and driver['recursive_region'] == model['artifacts']['recursive_region']
        and driver['qualification']['original_compiled_and_f4_bodies_joined'] is True
        and source['receiver'] == scope['abi']['receiver']
        and probe['source_recursive_owner'] == scope['owner']
        and probe['root_flag'] is True and probe['call']['code'] == source['code']['key']
        and probe['formals'] == scope['abi']['positional_formals'][1:]
        and yielded['bounds_updated_before_yield'] is True and yielded['gamma_updated_only_after_resume'] is True,
        'the selected source driver does not own the nominated recursive call and suspension')
    reset = [initialization[name]['field'] for name in ('counter', 'history', 'score_table', 'root')]
    retained = [yielded['move_lookup']['field']]
    clock = scope['clock']
    _require(clock is not None and clock['counter_field'] == reset[0]
        and driver['time_and_exit']['soft']['scope'] == 'BETWEEN_COMPLETED_DEPTHS', 'missing original clock cut')
    deadlines = [clock['deadline_field'], driver['time_and_exit']['soft']['deadline_field']]
    _require(len(set(reset + retained + deadlines)) == len(reset + retained + deadlines)
        and set(reset + retained + deadlines[:1]) <= set(scope['abi']['receiver_fields_observed_or_referenced']),
        'source reset and retained state fields do not match the owner layout')
    return dict(reset=reset, retained=retained, clock=deadlines)


def _control(driver, field_names):
    return dict(source_driver=driver['driver'], source_probe=driver['probe'], source_yield=driver['yield'],
        source_initialization=driver['initialization'], source_bracket=driver['bracket'],
        source_time_and_exit=driver['time_and_exit'], state_machine=[
            _transition('START', 'IDLE_OWNER', 'CREATED', ['BIND_OWNER_AND_HISTORY', 'PRESERVE_GENERATOR_LAZINESS']),
            _transition('ADVANCE', 'CREATED', 'RUNNING', ['RESET_SESSION_FIELDS_IN_SOURCE_ORDER', 'SELECT_TABLE_VERSION',
                'INITIALIZE_GAMMA_AND_BRACKET', 'ENTER_FIRST_ROOT_PROBE']),
            _transition('ADVANCE', 'SUSPENDED', 'RUNNING', ['RESUME_AFTER_YIELD', 'ADVANCE_GAMMA_FROM_UPDATED_BRACKET',
                'CHECK_SOFT_CLOCK_IF_BRACKET_FINISHED', 'CONTINUE_CURRENT_OR_START_NEXT_DEPTH']),
            _transition('PUBLISH_YIELD', 'RUNNING', 'SUSPENDED', ['ACCEPT_NORMAL_ROOT_RETURN',
                'TIGHTEN_FAIL_HIGH_OR_FAIL_LOW_BOUND', 'READ_MOVE_FROM_CURRENT_RECEIVER_STATE', 'PUBLISH_PROBE_RECORD',
                'SUSPEND_BEFORE_DEFERRED_GAMMA_UPDATE_OR_CLOCK']),
            _transition('REQUEST_STOP', ['RUNNING', 'SUSPENDED'], 'UNCHANGED_ACTIVATION',
                ['PUBLISH_EXTERNAL_DEADLINE_OR_CANCELLATION', 'RETAIN_SOURCE_POLL_AND_CONSUMER_DECISION']),
            _transition('STOP', 'RUNNING', 'STOPPED', ['PRESERVE_COMMITTED_STATE_PREFIX',
                'RETURN_ORIGINAL_HARD_STOP_TO_CONSUMER']),
            _transition('COMPLETE', 'RUNNING', 'COMPLETED', ['RETURN_AT_ORIGINAL_NORMAL_EXIT', 'KEEP_OWNER_STATE']),
            _transition('CLOSE', ['CREATED', 'SUSPENDED', 'STOPPED', 'COMPLETED'], 'IDLE_OWNER',
                ['DESTROY_LOCAL_FRAME_AND_KEEP_COMMITTED_RECEIVER', 'EXECUTE_NO_DEFERRED_PROBE_OR_CLOCK']),
        ], state_versions=dict(search_reset=field_names['reset'], retain_between_searches=field_names['retained'],
            external_clock_ports=field_names['clock'], cross_epoch='SAME_LIVE_RECEIVER_ONLY',
            fallback='RETAIN_ORIGINAL_BEFORE_FIRST_EFFECT_OR_REPLAY_EXACT_COMMITTED_CONTINUATION'),
        iteration=dict(unroll=False, probe_count=None, session_count=None,
            carry=['SAME_OWNER_STATE', 'GAMMA', 'DEPTH', 'LOWER', 'UPPER', 'SUSPENDED_CONTINUATION']))


def _obligations(observed):
    return [
        dict(id='source_scope', status='PROVEN', detail='Source owner, iterative probe, yield and recursive region are replayed.'),
        dict(id='completed_session_observation', status='PROVEN' if observed else 'UNRESOLVED',
            detail='Bind actual selected root calls and yield/resume/close to the same source driver and owner.'),
        dict(id='shared_receiver_and_visibility', status='RUNTIME_GUARD',
            detail='Bind one live owner and publish its node counter and ordered table accesses at original UI observations.'),
        dict(id='effects_and_cancellation', status='RUNTIME_GUARD',
            detail='Keep live deadline updates, independent controller events, exceptions and close before deferred effects.'),
        dict(id='lowering', status='LOWERING_CHECK',
            detail='Preserve source value layout, table version, insertion order, recursive state and the generator continuation.'),
        dict(id='resource_capacity', status='UNRESOLVED',
            detail='Measure or bound owner stack, resident tables, native allocator overhead and host placement.'),
        dict(id='entry_profitability', status='UNRESOLVED',
            detail='Compare matched clean full UCI launches with Python UI, any bridge, startup, stdout and fallback included.'),
    ]


def _observed_sessions(sessions):
    result = []
    last_move_table = {}
    for row in sessions:
        initial, initialized, final = (row[key] for key in ('initial', 'initialized', 'final'))
        _require(initialized['nodes'] == 0 and initialized['root'] == row['root_input']
            and all(initialized[role]['identity'] != initial[role]['identity']
                and final[role]['identity'] == initialized[role]['identity'] for role in ('score_table', 'history'))
            and initialized['score_table']['size'] == 0
            and initialized['history']['size'] <= row['history_input_size'],
            'selected source session did not renew its per-search state')
        identity = row['identity']
        move_table = initial['move_table']['identity']
        _require(all(part['move_table']['identity'] == move_table for part in (initialized, final))
            and move_table == last_move_table.get(identity, move_table),
            'retained move table identity changed in a selected owner session')
        last_move_table[identity] = move_table
        result.append(dict(owner_token=identity, move_table_token=move_table, thread_id=row['thread_id'],
            table_version=row['table_version'], initial_move_keys=initial['move_table']['size'],
            final_move_keys=final['move_table']['size'], final_score_keys=final['score_table']['size'],
            final_nodes=final['nodes'], yielded=len(row['yields']), close_site=row['exit']['site']))
    return result


def _plan(refs, driver, scope, prepared):
    fields = _field_roles(driver, scope)
    observed = bool(driver['sessions'])
    control = _control(driver, fields)
    recipe = make_recipe(PROFILE, semantic_boundary=dict(id='owner-bound-root-probe-driver', source=scope['source'],
        original_full_entry_effects_closed=False, ui_outside_lowering=True), control=control,
        task_graph=_schedule(scope['scope']['owner']['key']), artifacts=refs,
        resources=dict(owner=dict(kind='CPU_OWNER', lanes=1, affinity_policy='PINNED_BY_MATCHED_LAUNCH', cpu_ids=None),
            controller=dict(kind='CPU_CONTROLLER', independent=True, allocation_verified=False),
            memory=dict(domain='HOST', receiver_lifetime='OWNER_EPOCH', residency_verified=False,
                peak_bytes=None, live_frame_limit=None)),
        implementation=dict(family='CXX20_RESIDENT_RECURSIVE_OWNER', backend_inputs={role:role for role in refs},
            abi=dict(entrypoints=['START', 'ADVANCE', 'CLOSE'], out_of_band=['REQUEST_STOP', 'DEADLINE_UPDATE'],
                probe_fields=['DEPTH', 'GAMMA', 'SCORE', 'MOVE'], terminal=['COMPLETE', 'STOP', 'DECLINE'],
                live_python_reads=['NODE_COUNT', 'MOVE_TABLE_LOOKUP'], prepared_receiver_replay_only=False),
            status='EXPERIMENT_ONLY'), obligations=_obligations(observed))
    prior = [dict(case=row['case'], speedup=row['median_speedup'], kind=row['original_case_kind'])
             for row in prepared['profiles']]
    observation = dict(capture=driver['capture'], sessions=len(driver['sessions']),
        published_probes=sum(len(row['yields']) for row in driver['sessions']),
        close_reasons=[row['exit']['kind'] for row in driver['sessions']], used_as_clean_cost=False)
    if observed:
        observation['owner_sessions'] = _observed_sessions(driver['sessions'])
    return dict(schema=SCHEMA, schema_version=1, artifacts=refs, source=scope['source'], recipe=recipe,
        source_observation=observation,
        costs=dict(prepared_body=dict(scope=prepared['comparison_scope'], profiles=prior, transferable=False),
            cpp_clean_driver=dict(status='UNKNOWN', estimate_ns=None, boundary='RESIDENT_DRIVER_WITH_ROOT_TABLE'),
            full_original_entry=dict(status='UNKNOWN', estimate_ns=None, boundary='MATCHED_LIVE_UCI'),
            full_native_entry=dict(status='UNKNOWN', estimate_ns=None, boundary='MATCHED_LIVE_UCI',
                included=['PROCESS_STARTUP', 'UI_CONTROLLER', 'GUARDS', 'STATE_INTERFACES', 'PROBE_BRIDGE',
                    'NATIVE_SERVICE', 'STDOUT', 'FALLBACK']),
            wrapper_overhead=dict(status='UNKNOWN', estimate_ns=None, method='MATCHED_CLEAN_BOUNDARIES_REQUIRED'),
            source_probe_frequency_transferred_to_clean=False),
        decision=dict(selected='RETAIN_ORIGINAL_SOURCE', prototype='CXX20_RESIDENT_RECURSIVE_DRIVER',
            source_nomination='OBSERVED_SOURCE_FRONTIER' if observed else 'STATIC_SOURCE_FRONTIER',
            runtime_consumers_verified_for_all_scopes=False, deployment_authorized=False))


def propose(driver_path, scope_plan_path, assessment_path):
    refs = dict(driver=_ref(driver_path, source_driver.SCHEMA), scope_plan=_ref(scope_plan_path, owner_plan.SCHEMA),
        prepared_assessment=_ref(assessment_path, prepared_assessment.SCHEMA))
    prepared = prepared_assessment.check(refs['prepared_assessment']['uri'])
    _require(prepared['scope_plan'] == refs['scope_plan']
        and prepared['comparison_scope'] == 'PREPARED_SAME_OWNER_RECURSIVE_BODY'
        and prepared['whole_program_performance_status'] == 'UNKNOWN', 'prepared measurement binds another scope')
    scope = read_artifact(refs['scope_plan'])
    driver = source_driver.check(refs['driver']['uri'])
    return _plan(refs, driver, scope, prepared)


def _load_plan(path):
    stored = read_artifact(_ref(path, SCHEMA))
    refs = stored['artifacts']
    result = propose(refs['driver']['uri'], refs['scope_plan']['uri'], refs['prepared_assessment']['uri'])
    _require(json.dumps(stored, sort_keys=True, allow_nan=False) == json.dumps(result, sort_keys=True, allow_nan=False),
        'driver ownership plan differs from cold replay')
    return result


def _launch(row, launch, report, arm):
    command = row['command']
    cpus = ','.join(map(str, launch['cpu_ids']))
    _require(type(command) is list and all(type(item) is str for item in command) and len(command) >= 8
        and Path(command[0]).name == 'taskset' and command[1:3] == ['-c', cpus]
        and command.count('--') == 1, 'runtime command does not retain the exact CPU cohort')
    split = command.index('--')
    flags = command[:split]
    native = row['arm'] == 'driver_native'
    _require(command[split + 1:] == [launch['source']['uri'], *launch['argv']]
        and report['execution']['argv'] == launch['argv']
        and source_cost._argument(flags, '--report') == row['report']['uri'] and {'-B', '-u'} <= set(flags)
        and source_cost._argument(flags, '-m') == _MODULE and ('--isolated-study' in flags) == native
        and ('--observe' in flags) == (native and row['kind'] == 'OBSERVED_WARMUP'),
        'runtime command differs from source arguments or instrumentation scope')
    for key, (uri_flag, sha_flag) in _ARTIFACT_FLAGS.items():
        if native:
            reference = arm[key]
            _require(source_cost._argument(flags, uri_flag) == reference['uri']
                and source_cost._argument(flags, sha_flag) == reference['sha256'],
                'runtime launch omits the byte identity of ' + key)
        else:
            _require(uri_flag not in flags and sha_flag not in flags, 'baseline launch must run original source')
    implementation = launch.get('implementation')
    if implementation is not None:
        executable = flags[flags.index('-B') - 1]
        _require(Path(executable).is_absolute() and str(Path(executable).resolve(strict=True)) ==
            implementation['python_executable'] and report.get('bridge') == implementation['bridge']
            and report.get('python_version') == implementation['python_version'],
            'trial launch interpreter or Python bridge differs from the measured implementation')
    return tuple(flags[:flags.index('-m') + 2])


def _diagnostic(report, source_sessions):
    counts, phases = report['statistics'], report['phase_ns']
    _require(type(counts) is dict and type(phases) is dict
        and all(type(item) is int and item >= 0 for item in (*counts.values(), *phases.values()))
        and counts.get('native_sessions') == counts.get('native_closed') == len(source_sessions)
        and counts.get('native_yields') == sum(len(row['yields']) for row in source_sessions)
        and counts.get('fallback', 0) == 0 and len(report['sessions']) == len(source_sessions),
        'the independent observed run did not complete the exact selected native sessions')
    map_digest_sessions = 0
    for actual, expected in zip(report['sessions'], source_sessions, strict=True):
        _require(actual['table_version'] == expected['table_version']
            and actual['yields'] == [{key:row[key] for key in ('value','nodes')} for row in expected['yields']]
            and actual['outcome'] == 'consumer_close' and expected['exit']['kind'] == 'GENERATOR_CLOSE'
            and actual['initial_move_entries'] == expected['initialized']['move_table']['size']
            and actual['initial_score_entries'] == expected['initialized']['score_table']['size']
            and actual['final_nodes'] == expected['final']['nodes']
            and actual['final_move_entries'] == expected['final']['move_table']['size']
            and actual['final_score_entries'] == expected['final']['score_table']['size'],
            'the observed native probe/state sequence differs from selected original sessions')
        if 'ordered_map_digests' in expected['final']:
            expected_maps = expected['final']['ordered_map_digests']
            _require(type(expected_maps) is dict and len(expected_maps) == 2
                and all(type(field) is str and field and type(digest) is str and re.fullmatch(r'[a-f0-9]{64}', digest)
                        for field, digest in expected_maps.items())
                and actual.get('ordered_map_digests') == expected_maps,
                'the observed native ordered map contents or insertion order differs from source')
            map_digest_sessions += 1
    result = dict(statistics=counts, phase_ns=phases, exact_probe_sequence=True)
    if map_digest_sessions:
        result['exact_ordered_map_digest_sessions'] = map_digest_sessions
    return result


def _rows(benchmark, depth, source_sessions):
    rows = benchmark['rows']
    warmups = len({row['round'] for row in rows if row['kind'] == 'OBSERVED_WARMUP'})
    measured = len({row['round'] for row in rows if row['kind'] == 'CLEAN_MEASURED'})
    rounds = source_cost._groups(rows, _ARMS, warmups, measured, key='arm')
    samples = {arm:[] for arm in _ARMS}
    checks, launchers, diagnostics = set(), set(), []
    for i, row in enumerate(rows):
        mode, block = row['arm'], row['round']
        observed = row['kind'] == 'OBSERVED_WARMUP'
        _require(block == i // 2 and mode == _ARMS[(i % 2 + block) % 2]
            and row['kind'] in ('OBSERVED_WARMUP', 'CLEAN_MEASURED')
            and observed == (block < warmups)
            and type(row['returncode']) is int and row['returncode'] == 0
            and type(row['wall_ns']) is int and row['wall_ns'] > 0, 'invalid paired run order or completion')
        report = read_artifact(row['report'])
        arm = benchmark['arms'][mode]
        expected_source = None if mode == 'original' else benchmark['source']
        _require(report['schema'] == STUDY and report['schema_version'] == 1
            and report['source'] == expected_source and report['admitted_for_deployment'] is False
            and report['execution']['entry_path'] == benchmark['source']['uri']
            and report['execution']['exit_code'] == 0 and all(report[key] == value for key, value in arm.items()),
            'trial runtime report differs from its source or admission')
        launchers.add(_launch(row, benchmark, report, arm))
        raw, errors = (read_artifact(row[key], json_document=False) for key in ('stdout', 'stderr'))
        _require(b'info string aborted at ' not in raw, 'the live search stopped before completing its requested scope')
        output, _ = source_cost._output(raw, depth, len(source_sessions))
        checks.add((output, errors))
        if observed and mode == 'driver_native':
            diagnostics.append(_diagnostic(report, source_sessions))
        else:
            _require(report['statistics'] is None and report['phase_ns'] is None
                and report.get('sessions') is None and report.get('frontiers') is None,
                'instrumented runs may not enter the clean comparison')
        if not observed:
            samples[mode].append(row['wall_ns'])
    _require(len(checks) == 1 and len(launchers) == 1 and diagnostics,
        'paired execution must use one launcher and identical stdout and stderr')
    return samples, rounds, checks.pop()[0], diagnostics


def _implementation(benchmark):
    if 'implementation' not in benchmark:
        return None
    value = benchmark['implementation']
    _require(type(value) is dict and set(value) == {'bridge', 'launcher', 'python_executable', 'python_version'}
        and type(value['python_executable']) is str and Path(value['python_executable']).is_absolute()
        and type(value['python_version']) is str and bool(value['python_version'].strip()),
        'paired execution requires a recorded bridge, launcher and interpreter identity')
    python = Path(value['python_executable'])
    _require(python.is_file() and python.resolve(strict=True) == python,
        'paired interpreter is no longer available at its recorded real path')
    for role in ('bridge', 'launcher'):
        read_artifact(value[role], json_document=False)
    return value


def compare(plan_path, benchmark_path):
    """Choose inside complete, same-launch UCI sample envelopes only."""
    refs = dict(plan=_ref(plan_path, SCHEMA), benchmark=_ref(benchmark_path, BENCHMARK))
    plan = _load_plan(refs['plan']['uri'])
    benchmark = read_artifact(refs['benchmark'])
    _require(benchmark['schema_version'] == 1 and benchmark['driver_plan'] == refs['plan']
        and benchmark['source'] == plan['source'] and benchmark['normalization'] == source_cost.NORMALIZATION
        and type(benchmark['cwd']) is str and Path(benchmark['cwd']).is_absolute()
        and type(benchmark['argv']) is list and all(type(arg) is str for arg in benchmark['argv'])
        and type(benchmark['environment']) is dict and all(type(key) is str and type(value) is str
            for key, value in benchmark['environment'].items())
        and type(benchmark['cpu_ids']) is list and len(benchmark['cpu_ids']) == 1
        and type(benchmark['cpu_ids'][0]) is int and benchmark['cpu_ids'][0] >= 0,
        'paired run differs from the planned source and explicit single-CPU launch')
    implementation = _implementation(benchmark)
    original, native = (benchmark['arms'][arm] for arm in _ARMS)
    keys = set(_ARTIFACT_FLAGS)
    _require(set(benchmark['arms']) == set(_ARMS) and set(original) == set(native) == keys
        and all(original[key] is None and native[key] is not None for key in keys),
        'baseline and native must retain the same source/UI contract with only native work added')
    allowed_drivers = [plan['artifacts']['driver']]
    capture = plan['source_observation']['capture']
    if capture is not None:
        allowed_drivers.append(read_artifact(capture)['design'])
    _require(native['source_driver'] in allowed_drivers, 'timing uses a different source-driver recipe')
    _require(native['library']['schema'] == 'scar.scalar-recursive-driver-library'
        and native['ui_contract']['schema'] == 'scar.scalar-recursive-ui-contract',
        'paired run requires an exact C++ provider and UI contract')
    provider, ui = (read_artifact(native[key]) for key in ('library', 'ui_contract'))
    prepared = read_artifact(plan['artifacts']['prepared_assessment'])
    _require(provider['code'] == plan['recipe']['control']['source_driver']['code']
        and provider['native'] == read_artifact(prepared['timing'])['artifact']
        and provider['cxx_standard'] == 20 and provider['whole_driver_cpp'] is True
        and ui['source_driver'] == native['source_driver'], 'paired provider must retain the exact original proof chain')
    base = Path(native['library']['uri']).parent
    member = provider['library']
    library = (base / member['uri']).resolve(strict=True)
    _require(library.is_relative_to(base), 'runtime library escaped its published provider')
    read_artifact(dict(member, uri=str(library)), json_document=False)
    read_artifact(ui['ui_source'], json_document=False)
    commands = read_artifact(benchmark['stdin'], json_document=False).decode('ascii').splitlines()
    depths = {line.removeprefix('go depth ') for line in commands if line.startswith('go ')}
    _require(len(depths) == 1 and next(iter(depths)).isdigit(), 'expected one finite fixed-depth search contract')
    depth = int(depths.pop())
    searches = source_cost._request_count(commands, depth)
    source_sessions = read_artifact(plan['artifacts']['driver'])['sessions']
    _require(searches == len(source_sessions), 'whole-entry selection requires the exact original observed sessions')
    samples, orders, digest, diagnostics = _rows(benchmark, depth, source_sessions)
    measurements, comparisons = source_cost._envelopes(samples, minimum_gain_bps=0)
    decision = comparisons['driver_native']
    selection = 'driver_native' if decision['verified_observed_gain'] else 'original'
    result = dict(schema=COMPARISON, schema_version=1, artifacts=refs, scope='SAME_FIXED_DEPTH_WHOLE_UCI_PROCESS',
        recipe_id=plan['recipe']['recipe_id'], matched_launch={key:benchmark[key] for key in
            ('source', 'stdin', 'cpu_ids', 'cwd', 'argv', 'environment', 'normalization')},
        treatment=dict(native_library=native['library'], native_world_scope=provider['abi']['control'],
            original_python_ui_retained=True,
            observed_owner_snapshot_bytes=[row['statistics'].get('snapshot_bytes') for row in diagnostics]),
        external_shell_and_wrapper_included=True, exact_canonical_stdout_sha256=digest, depth=depth,
        searches=searches, clean_pairs=len(samples['original']), warmup_pairs=len(diagnostics), orders=orders,
        measurements=measurements, original_minus_native_evidence=decision, instrumented_native=diagnostics,
        selected=selection, uncertainty='OBSERVED_PAIRED_SAMPLE_ENVELOPES', deployment_authorized=False,
        prepared_local_speedup_used=False, standalone_cxx_driver_wall_ns=None, separate_wrapper_overhead_ns=None)
    if implementation is not None:
        result['implementation'] = implementation
    return result


def check(path):
    schema = json.loads(Path(path).read_bytes())['schema']
    if schema == SCHEMA:
        return _load_plan(path)
    _require(schema == COMPARISON, 'unsupported driver plan or comparison')
    stored = read_artifact(_ref(path, COMPARISON))
    refs = stored['artifacts']
    result = compare(refs['plan']['uri'], refs['benchmark']['uri'])
    _require(json.dumps(stored, sort_keys=True, allow_nan=False) == json.dumps(result, sort_keys=True, allow_nan=False),
        'matched UCI comparison differs from cold replay')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('driver', 'scope', 'assessment', 'check', 'out', 'plan', 'benchmark'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    options = (args.driver, args.scope, args.assessment, args.out)
    comparison = (args.plan, args.benchmark)
    if args.check and not any(options + comparison):
        result = check(args.check)
    elif args.check is None and all(options) and not any(comparison):
        result = propose(args.driver, args.scope, args.assessment)
    elif args.check is None and all(comparison) and args.out and not any(options[:-1]):
        result = compare(args.plan, args.benchmark)
    else:
        parser.error('supply --check, plan inputs, or --plan and --benchmark with --out')
    if args.out:
        with args.out.open('x') as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    summary = dict(result['decision'], recipe_id=result['recipe']['recipe_id']) if result['schema'] == SCHEMA else {
        key:result[key] for key in ('selected', 'measurements', 'recipe_id')}
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()

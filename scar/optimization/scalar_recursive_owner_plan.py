"""Plan state materialization across adjacent source generator activations."""

import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import scalar_recursive_driver_plan as driver_plan
from scar.optimization import scalar_sorted_transition_cost as source_cost


SCHEMA = 'scar.scalar-recursive-owner-plan'
INVENTORY = 'scar.scalar-recursive-owner-continuity'
MODE = 'ISOLATED_FIXED_ADJACENT_SOURCE_GENERATORS'
BENCHMARK = 'scar.scalar-recursive-owner-paired-run'
COMPARISON = 'scar.scalar-recursive-owner-paired-selection'
_ARMS = ('driver_native', 'owner_native')
_OWNER_MODULE = 'scar.runtime.scalar_recursive_owner_study'
_OWNER_REPORT = 'scar.scalar-recursive-owner-source-study'


def _require(condition, detail):
    if not condition:
        raise ValueError('recursive owner materialization: ' + detail)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _state(recipe):
    control = recipe['control']
    init = control['source_initialization']
    move = control['source_yield']['move_lookup']['field']
    score = init['score_table']
    versions = control['state_versions']
    _require(versions['retain_between_searches'] == [move] and score['field'] in versions['search_reset'],
        'the selected owner does not have distinct persistent and reset ordered maps')
    return dict(move=dict(source_field=move, active_idle='NATIVE_LOOKUP_PROXY', carry='SAME_HANDLE',
            original_binding='KEEP_ORIGINAL_DICTIONARY_FOR_MATERIALIZATION',
            supported_reads=['KEYED_GET', 'PRESENCE_TEST'], writes='ORIGINAL_MATERIALIZATION_BARRIER'),
        score=dict(source_field=score['field'], original_reset=score['store'], reset='ORIGINAL_STORE_ATTR_THEN_NATIVE_RESET',
            discard_before_reset=False, keep_old_version_until=['ORIGINAL_RESET_CONFIRMED', 'NO_OLD_ALIAS_OBSERVATION'],
            legacy_materialization='WRITE_ONLY_IF_ORIGINAL_BINDING_STILL_MATCHES_REPRESENTED_VERSION'),
        scalar_ports=dict(counter=init['counter']['field'], history=init['history']['field'],
            root=init['root']['field'], clock=versions['external_clock_ports']))


def _barriers():
    return [
        dict(event='PUBLISH_AND_SUSPEND', order=['FINISH_NATIVE_PROBE', 'PUBLISH_NODES_AND_PROBE_VALUE',
            'SERVE_ADMITTED_MOVE_QUERIES'], state='WORLD_ACTIVE_OR_SUSPENDED'),
        dict(event='SOURCE_GENERATOR_CLOSE_WITH_OUTGOING_PAIR',
            order=['KEEP_SUCCEEDED_SOURCE_PREFIX', 'DO_NOT_RESUME_SUSPENDED_SOURCE',
            'KEEP_MOVE_PROXY_AT_ADMITTED_IDLE_READS', 'KEEP_SCORE_VERSION_UNTIL_SOURCE_RESET_OR_FLUSH'],
            state='WORLD_IDLE'),
        dict(event='SOURCE_GENERATOR_CLOSE_WITHOUT_OUTGOING_PAIR',
            order=['KEEP_SUCCEEDED_SOURCE_PREFIX', 'DO_NOT_RESUME_SUSPENDED_SOURCE',
                'MATERIALIZE_MOVE_AND_STILL_CURRENT_SCORE', 'RESTORE_ORIGINAL_DICTIONARY_OBJECTS', 'RELEASE_NATIVE_WORLD'],
            state='ORIGINAL'),
        dict(event='ADJACENT_SOURCE_RESET', order=['CONFIRM_SAME_LIVE_OWNER_AND_ORIGINAL_SCORE_RESET_SITE',
            'CONFIRM_FRESH_EMPTY_SCORE_BINDING_AND_NO_OLD_ALIAS_OBSERVATION', 'REBIND_V2_ATOMICALLY',
            'PRESERVE_MOVE_AND_REVOKE_OLD_SCORE', 'BEGIN_NEXT_SOURCE_ROOT_PROBE'], state='WORLD_ACTIVE'),
        dict(event='BEFORE_UNSUPPORTED_ACCESS_OR_ALIAS_ESCAPE', order=['QUIESCE_OWNER_BEFORE_OBSERVATION',
            'MATERIALIZE_MOVE_AND_STILL_CURRENT_SCORE', 'RESTORE_ORIGINAL_DICTIONARY_OBJECTS',
            'RUN_ORIGINAL_ACCESS'], state='ORIGINAL'),
        dict(event='BEFORE_REBIND_REFUSAL_FALLBACK', order=['KEEP_PRIOR_COMMITTED_WORLD',
            'MATERIALIZE_MOVE_AND_STILL_CURRENT_SCORE', 'DO_NOT_RESTORE_REVOKED_SCORE_INTO_FRESH_BINDING',
            'RESUME_EXACT_ORIGINAL_CONTINUATION'], state='ORIGINAL'),
        dict(event='OWNER_CHANGE_OR_PROCESS_EXIT', order=['QUIESCE_OWNER', 'MATERIALIZE_MOVE_AND_STILL_CURRENT_SCORE',
            'RESTORE_ORIGINAL_DICTIONARY_OBJECTS', 'RELEASE_NATIVE_WORLD'], state='ORIGINAL'),
        dict(event='FAILURE_AFTER_UNREPLAYABLE_EFFECT', order=['DO_NOT_REPEAT_SOURCE_PREFIX',
            'REPORT_SCOPE_FAILURE_IF_EXACT_CONTINUATION_UNAVAILABLE'], state='UNRESOLVED'),
    ]


def _prior(path, plan_ref, source):
    if path is None:
        return None
    ref = _ref(path, driver_plan.COMPARISON)
    prior = read_artifact(ref)
    _require(prior['artifacts']['plan'] == plan_ref and prior['matched_launch']['source'] == source
        and prior['scope'] == 'SAME_FIXED_DEPTH_WHOLE_UCI_PROCESS'
        and prior['treatment']['native_world_scope'] == 'ONE_WORLD_PER_SOURCE_GENERATOR',
        'archived baseline belongs to a different source or state materialization boundary')
    baseline = read_artifact(prior['artifacts']['benchmark'])
    _require(baseline['driver_plan'] == plan_ref and baseline['stdin'] == prior['matched_launch']['stdin'],
        'archived comparison does not bind the same original launch')
    # The previous comparison stays attached as an archived observation only.
    # Replaying a prospective plan never treats old bridge times as new costs.
    diagnostic = []
    for row in prior['instrumented_native']:
        total, amount = row['phase_ns']['snapshot_and_publish'], row['statistics']['snapshot_bytes']
        _require(type(total) is type(amount) is int and total > 0 and amount > 0,
            'missing measured original bridge publication boundary')
        diagnostic.append(dict(snapshot_and_publish_ns=total, snapshot_bytes=amount,
                               world_closes=row['statistics']['native_closed']))
    return dict(selection=ref, stdin=baseline['stdin'], archived_treatment='ONE_WORLD_PER_SOURCE_GENERATOR',
        archived_native_whole_entry_median_ns=prior['measurements']['driver_native']['median_ns'],
        instrumented_publication=diagnostic, reusable_for_new_profitability=False)


def _bind_inventory(scope, state, inventory):
    ui = read_artifact(inventory['ui_contract'])
    source = read_artifact(scope['source_driver'])
    move, score = state['move']['source_field'], state['score']['source_field']
    static = inventory['static']
    _require(inventory['source_driver'] == ui['source_driver'] == scope['source_driver']
        and source['source'] == scope['source'] and static['driver']['code'] == scope['owner_code']
        and inventory['roles'] == dict(move_table=move, score_table=score)
        and static['reset']['site'] == state['score']['original_reset']
        and ui['native_scope'] == 'FIXED_DEPTH_ORIGINAL_CONSUMER_WITHOUT_OBSERVERS',
        'caller inventory differs from the isolated driver and original table reset')
    scope.update(ui_contract=inventory['ui_contract'], stdin=inventory['stdin'],
                 original_access_coverage=static['proof_scope'], outside_scope_frontiers=static['unresolved'])
    _require(read_artifact(scope['stdin'], json_document=False), 'the isolated source transcript is empty')
    sessions = source['sessions']
    original = [link for link in source['state_reuse'] if link['after_session'] == link['before_session'] + 1]
    adjacent, blockers = [], []
    for ordinal, window in enumerate(inventory['windows']):
        anchors = window['anchors']
        holders = anchors['close']['holders']
        matched = []
        for link in original:
            before, after = (sessions[link[key]] for key in ('before_session', 'after_session'))
            final = before['final']
            digests = final.get('ordered_map_digests', {})
            if (digests.get(move) == holders['move_table']['ordered_sha256'] == after.get('initial_move_digest')
                    and digests.get(score) == holders['score_table']['ordered_sha256']
                    and final['move_table']['size'] == holders['move_table']['size']
                    and final['score_table']['size'] == holders['score_table']['size']
                    and before['identity'] == after['identity'] and before['thread_id'] == after['thread_id']
                    and link['score_table_reset'] is True):
                matched.append(link)
        if len(matched) != 1:
            blockers.append('CALLER_WINDOW_LACKS_UNIQUE_ORIGINAL_STATE_PAIR')
            continue
        unsupported = [row for row in window['events'] if row['role'] != 'move_table'
                       or row['kind'] not in state['move']['supported_reads']]
        unsupported.extend(window['unsupported_events'])
        if unsupported:
            blockers.append('CALLER_WINDOW_REQUIRES_ORIGINAL_DICTIONARY_MATERIALIZATION')
        link = matched[0]
        adjacent.append(dict(before_session=link['before_session'], after_session=link['after_session'],
            caller_window=ordinal, owner_token_in_continuity_capture=window['owner_identity'],
            move_digest=holders['move_table']['ordered_sha256'],
            revoked_score_digest=holders['score_table']['ordered_sha256'],
            fresh_score_reset_anchor=anchors['fresh_score_reset'],
            source_anchors=anchors, permitted_idle_reads=[row for row in window['events'] if row not in unsupported],
            access_barriers=unsupported, runtime_guard_required=True))
    expected = [(index, index + 1) for index in range(len(sessions) - 1)]
    if [(row['before_session'], row['after_session']) for row in adjacent] != expected:
        blockers.append('INCOMPLETE_ADJACENT_SESSION_COVERAGE')
    if inventory['frontiers']:
        blockers.append('UNRESOLVED_ORIGINAL_CAPTURE_FRONTIERS')
    return adjacent, sorted(set(blockers))


def propose(driver_plan_path, caller_inventory_path=None, selection_path=None):
    """Admit only the checked source window as an isolated resident experiment."""
    plan_ref = _ref(driver_plan_path, driver_plan.SCHEMA)
    plan = driver_plan.check(plan_ref['uri'])
    source_ref, driver_ref = plan['source'], plan['artifacts']['driver']
    state = _state(plan['recipe'])
    prior = _prior(selection_path, plan_ref, source_ref)
    inventory_ref = _ref(caller_inventory_path, INVENTORY) if caller_inventory_path is not None else None
    scope = dict(mode=MODE, source=source_ref, source_driver=driver_ref, ui_contract=None, stdin=None,
        owner_code=plan['recipe']['control']['source_driver']['code'], owner_identity_rule='SAME_LIVE_SOURCE_RECEIVER_AND_OWNER_THREAD',
        transcript_order='WAIT_FOR_PUBLICATION_BEFORE_NEXT_COMMAND_WHEN_SOURCE_PROTOCOL_REQUIRES',
        unknown_external_aliases='OUTSIDE_ISOLATED_EXPERIMENT')
    adjacent, blockers = [], ['MISSING_ORIGINAL_CALLER_ACCESS_INVENTORY']
    if inventory_ref is not None:
        from scar.analysis import scalar_recursive_owner_continuity
        inventory = scalar_recursive_owner_continuity.check(inventory_ref['uri'])
        adjacent, blockers = _bind_inventory(scope, state, inventory)
        if prior is not None:
            _require(read_artifact(prior['stdin'], json_document=False) == read_artifact(scope['stdin'], json_document=False),
                'archived diagnostic and original caller inventory used different input bytes')
    admitted = bool(adjacent) and not blockers
    return dict(schema=SCHEMA, schema_version=1, artifacts=dict(driver_plan=plan_ref, caller_inventory=inventory_ref,
            prior_selection=prior['selection'] if prior is not None else None), scope=scope, state=state,
        adjacent_sessions=adjacent, barriers=_barriers(),
        backend_contract=dict(cxx_standard=20, owner_lane_count=1, bridge='PYTHON_UI_WITH_NATIVE_OWNER',
            abi='scar_recursive_driver_rebind_v2', rebind_refusal='LEAVES_PRIOR_WORLD_UNCHANGED',
            lookup_when_idle='EXACT_ORIGINAL_KEY_AND_DEFAULT_SEMANTICS',
            baseline='ONE_WORLD_PER_SOURCE_GENERATOR', phase='ISOLATED_FIXED_INPUT_EXPERIMENT'),
        costs=dict(status='UNKNOWN_FOR_RESIDENT_TREATMENT', historical=prior, whole_entry_resident_median_ns=None,
            changed_boundary=['INTERMEDIATE_CLOSE_AND_MOVE_REINGRESS', 'FINAL_MATERIALIZATION', 'IDLE_PROXY_AND_GUARDS'],
            peak_resident_bytes=None, score_and_move_breakdown_ns=None),
        decision=dict(selected='RETAIN_ORIGINAL_SOURCE', runtime_experiment_authorized=admitted,
            deployment_authorized=False, blockers=blockers))


def _load_plan(path):
    stored = read_artifact(_ref(path, SCHEMA))
    refs = stored['artifacts']
    result = propose(refs['driver_plan']['uri'],
        refs['caller_inventory']['uri'] if refs['caller_inventory'] is not None else None,
        refs['prior_selection']['uri'] if refs['prior_selection'] is not None else None)
    _require(json.dumps(stored, sort_keys=True, allow_nan=False) == json.dumps(result, sort_keys=True, allow_nan=False),
        'owner materialization differs from cold replay')
    return result


def _implementation(document):
    keys = {'launcher', 'python_executable', 'python_version', 'driver_bridge', 'owner_bridge'}
    _require(type(document) is dict and set(document) == keys
        and type(document['python_executable']) is str and Path(document['python_executable']).is_absolute()
        and type(document['python_version']) is str and document['python_version'].strip(),
        'clean comparison lacks the launcher, bridges or interpreter identity')
    python = Path(document['python_executable'])
    _require(python.is_file() and python.resolve(strict=True) == python,
        'the paired interpreter must resolve to its original real path')
    for name in ('launcher', 'driver_bridge', 'owner_bridge'):
        read_artifact(document[name], json_document=False)


def _provider(ref, expected_version):
    _require(ref['schema'] == 'scar.scalar-recursive-driver-library', 'missing checked native driver manifest')
    provider = read_artifact(ref)
    base = Path(ref['uri']).parent
    member = provider['library']
    library = (base / member['uri']).resolve(strict=True)
    _require(library.is_relative_to(base) and type(provider['schema_version']) is int
        and provider['schema_version'] == expected_version
        and provider['whole_driver_cpp'] is True and provider['deployment_authorized'] is False,
        'native provider does not match the scoped executable ABI')
    read_artifact(dict(member, uri=str(library)), json_document=False)
    return provider


def _launch(row, benchmark, report):
    command, mode = row['command'], row['arm']
    observed = row['kind'] == 'OBSERVED_WARMUP'
    arm = benchmark['arms'][mode]
    if mode == 'driver_native':
        launch = {key:value for key, value in benchmark.items() if key != 'implementation'}
        driver_plan._launch(row, launch, report, arm)
    else:
        cpus = ','.join(map(str, benchmark['cpu_ids']))
        _require(type(command) is list and all(type(arg) is str for arg in command) and len(command) >= 8
            and Path(command[0]).name == 'taskset' and command[1:3] == ['-c', cpus]
            and source_cost._argument(command, '-m') == _OWNER_MODULE and {'-B','-u'} <= set(command)
            and source_cost._argument(command, '--report') == row['report']['uri']
            and ('--observe' in command) == observed and '--' not in command,
            'owner study launch does not match the isolated runtime and instrumentation scope')
        for role in ('library','plan'):
            _require(source_cost._argument(command, '--' + role) == arm[role]['uri']
                and source_cost._argument(command, '--' + role + '-sha256') == arm[role]['sha256'],
                'owner launcher passed a different exact ' + role)
    implementation = benchmark['implementation']
    bridge = implementation['driver_bridge' if mode == 'driver_native' else 'owner_bridge']
    executable = Path(command[command.index('-B') - 1])
    _require(executable.is_absolute() and str(executable.resolve(strict=True)) == implementation['python_executable']
        and report.get('bridge') == bridge and report.get('python_version') == implementation['python_version'],
        'paired trial implementation differs from the retained bridge and interpreter')
    return tuple(command[:command.index('-m')])


def _paired_rows(benchmark, depth, sessions, adjacent):
    rows = benchmark['rows']
    _require(type(rows) is list and len(rows) >= 6 and len(rows) % 2 == 0,
        'expected one observed warmup and at least two complete clean pairs')
    orders = [[rows[index]['arm'], rows[index + 1]['arm']] for index in range(0, len(rows), 2)]
    idle_gets = sum(row['kind'] == 'KEYED_GET' for pair in adjacent for row in pair['permitted_idle_reads'])
    diagnostics, samples, stdout, launchers = {}, {mode:[] for mode in _ARMS}, set(), set()
    for i, row in enumerate(rows):
        round_, slot = divmod(i, 2)
        observed = round_ == 0
        mode = _ARMS[(slot + round_) % 2]
        _require(row['arm'] == mode and type(row['round']) is int and row['round'] == round_
            and row['kind'] == ('OBSERVED_WARMUP' if observed else 'CLEAN_MEASURED')
            and type(row['returncode']) is int and row['returncode'] == 0
            and type(row['wall_ns']) is int and row['wall_ns'] > 0,
            'paired row was out of order, instrumented or did not complete')
        report = read_artifact(row['report'])
        arms = benchmark['arms'][mode]
        _require(report['schema'] == (driver_plan.STUDY if mode == 'driver_native' else _OWNER_REPORT)
            and report['schema_version'] == 1 and report['admitted_for_deployment'] is False
            and report['execution']['entry_path'] == benchmark['source']['uri']
            and report['execution']['argv'] == benchmark['argv'] and report['execution']['exit_code'] == 0
            and all(report[key] == value for key, value in arms.items()),
            'paired runtime execution is outside its declared source, ABI or experiment')
        if mode == 'owner_native':
            expected = dict(qualified=True, closed_sessions=len(sessions), rebinds=len(adjacent))
            completion = report.get('completion')
            _require(type(completion) is dict and completion.get('qualified') is True
                and type(completion.get('closed_sessions')) is type(completion.get('rebinds')) is int
                and completion == expected,
                'resident trial lacked the clean original probe and rebind completion checks')
        launchers.add(_launch(row, benchmark, report))
        raw, errors = (read_artifact(row[key], json_document=False) for key in ('stdout','stderr'))
        _require(b'info string aborted at ' not in raw, 'source search was interrupted before reaching its scope')
        digest, _ = source_cost._output(raw, depth, len(sessions))
        stdout.add((digest, errors))
        if observed:
            evidence = driver_plan._diagnostic(report, sessions)
            _require(evidence.get('exact_ordered_map_digest_sessions') == len(sessions),
                'the observed resident treatment must preserve complete ordered map contents')
            if mode == 'owner_native':
                counts = evidence['statistics']
                _require(counts.get('native_worlds_created') == counts.get('native_worlds_released') == 1
                    and counts.get('terminal_flushes') == 1 and counts.get('native_rebinds') == len(adjacent)
                    and counts.get('idle_lookups', 0) == idle_gets
                    and type(counts.get('published_snapshot_bytes')) is int
                    and 0 < counts['published_snapshot_bytes'] <= counts['snapshot_bytes'],
                    'observed owner did not reuse one native world and publish at one terminal barrier')
                evidence['snapshot_bytes_include_extra_validation'] = True
            diagnostics[mode] = evidence
        else:
            _require(all(report.get(key) is None for key in ('statistics','phase_ns','sessions','frontiers')),
                'instrumented observations cannot enter clean service pricing')
            samples[mode].append(row['wall_ns'])
    _require(len(stdout) == len(launchers) == 1 and set(diagnostics) == set(_ARMS),
        'paired treatment changed the launcher, canonical stdout or stderr')
    return samples, orders, stdout.pop()[0], diagnostics


def compare(plan_path, benchmark_path):
    """Check resident and materializing arms at the same full source boundary."""
    refs = dict(plan=_ref(plan_path, SCHEMA), benchmark=_ref(benchmark_path, BENCHMARK))
    plan = _load_plan(refs['plan']['uri'])
    scope = plan['scope']
    benchmark = read_artifact(refs['benchmark'])
    _require(plan['decision']['runtime_experiment_authorized'] is True and benchmark['schema_version'] == 1
        and benchmark['owner_plan'] == refs['plan'] and benchmark['source'] == scope['source']
        and benchmark['normalization'] == source_cost.NORMALIZATION and benchmark['argv'] == []
        and type(benchmark['cwd']) is str and Path(benchmark['cwd']).is_absolute()
        and type(benchmark['environment']) is dict and all(type(key) is type(value) is str
            for key, value in benchmark['environment'].items())
        and type(benchmark['cpu_ids']) is list and len(benchmark['cpu_ids']) == 1
        and type(benchmark['cpu_ids'][0]) is int and benchmark['cpu_ids'][0] >= 0,
        'the measured isolated launch must retain the admitted source and CPU boundary')
    _implementation(benchmark['implementation'])
    arms = benchmark['arms']
    original, resident = (arms[mode] for mode in _ARMS)
    _require(set(arms) == set(_ARMS) and set(original) == {'library','source_driver','ui_contract'}
        and set(resident) == {'library','plan'} and original['source_driver'] == scope['source_driver']
        and original['ui_contract'] == scope['ui_contract'] and resident['plan'] == refs['plan'],
        'paired treatment must preserve the same source driver and UI')
    each, held = (_provider(entry['library'], version) for entry, version in zip((original,resident),(1,2),strict=True))
    _require(each['native'] == held['native'] and each['code'] == held['code'] == scope['owner_code']
        and each['abi']['control'] == 'ONE_WORLD_PER_SOURCE_GENERATOR'
        and held['abi']['control'] == 'ONE_WORLD_ACROSS_REBOUND_SOURCE_GENERATORS' and 'rebind' in held['abi'],
        'resident ABI or original numerical implementation changed')
    raw = read_artifact(benchmark['stdin'], json_document=False)
    _require(raw == read_artifact(scope['stdin'], json_document=False), 'measured command input differs from the admitted transcript')
    commands = raw.decode('ascii').splitlines()
    depths = {line.removeprefix('go depth ') for line in commands if line.startswith('go ')}
    _require(len(depths) == 1 and next(iter(depths)).isdigit(), 'expected one fixed requested depth in the isolated source launch')
    depth = int(depths.pop())
    sessions = read_artifact(scope['source_driver'])['sessions']
    _require(source_cost._request_count(commands, depth) == len(sessions), 'the measured original session count changed')
    samples, orders, digest, diagnostics = _paired_rows(benchmark, depth, sessions, plan['adjacent_sessions'])
    measurements, comparisons = source_cost._envelopes(dict(original=samples['driver_native'],
        owner_native=samples['owner_native']), minimum_gain_bps=0)
    measurements['driver_native'] = measurements.pop('original')
    outcome = comparisons['owner_native']
    return dict(schema=COMPARISON, schema_version=1, artifacts=refs, scope='SAME_FIXED_DEPTH_WHOLE_UCI_PROCESS',
        matched_launch={key:benchmark[key] for key in ('source','stdin','argv','cpu_ids','cwd','environment','normalization')},
        implementation=benchmark['implementation'], selected='owner_native' if outcome['verified_observed_gain'] else 'driver_native',
        depth=depth, searches=len(sessions), clean_pairs=len(samples['driver_native']), warmup_pairs=1,
        orders=orders, exact_canonical_stdout_sha256=digest,
        measurements=measurements, materializing_minus_resident_evidence=outcome, instrumented=diagnostics,
        deployment_authorized=False, clean_transfer_bytes=None, historical_cost_used_for_selection=False)


def check(path):
    schema = json.loads(Path(path).read_bytes())['schema']
    if schema == SCHEMA:
        return _load_plan(path)
    _require(schema == COMPARISON, 'unsupported owner plan or paired comparison')
    stored = read_artifact(_ref(path, COMPARISON))
    refs = stored['artifacts']
    result = compare(refs['plan']['uri'], refs['benchmark']['uri'])
    _require(json.dumps(stored, sort_keys=True, allow_nan=False) == json.dumps(result, sort_keys=True, allow_nan=False),
        'paired owner decision differs from cold replay')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('driver-plan', 'caller-inventory', 'selection', 'check', 'out', 'plan', 'benchmark'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    options = (args.driver_plan, args.caller_inventory, args.selection)
    if args.check and not any((*options,args.plan,args.benchmark,args.out)):
        result = check(args.check)
    elif args.check is None and args.driver_plan and args.out and not any((args.plan,args.benchmark)):
        result = propose(args.driver_plan, args.caller_inventory, args.selection)
    elif args.check is None and args.plan and args.benchmark and args.out and not any(options):
        result = compare(args.plan,args.benchmark)
    else:
        parser.error('supply --check, plan inputs, or --plan and --benchmark with --out')
    if args.out:
        with args.out.open('x') as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    summary = result['decision'] if result['schema'] == SCHEMA else {
        key:result[key] for key in ('selected','measurements','exact_canonical_stdout_sha256')}
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()

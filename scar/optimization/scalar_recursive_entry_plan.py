"""Plan a standalone C++ entry for a checked finite source protocol."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import re

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import scalar_recursive_owner_plan as owner_plan
from scar.optimization import scalar_sorted_transition_cost as source_cost


SCHEMA = 'scar.scalar-recursive-entry-plan'
SOURCE = 'scar.scalar-recursive-entry-source'
MODE = 'CHECKED_PACED_FINITE_SOURCE_TAPE'
BENCHMARK = 'scar.scalar-recursive-entry-paired-run'
COMPARISON = 'scar.scalar-recursive-entry-paired-selection'
EXECUTABLE = 'scar.scalar-recursive-entry-executable'
OBSERVATION = 'scar.scalar-recursive-entry-observation'
_ARMS = ('original', 'entry_native')
_TIMING_FIELDS = re.compile(rb'^(info depth [0-9]+) time [0-9]+( nodes [0-9]+) nps [0-9]+ ')
_REPLY_NODES = (ast.Assign, ast.Attribute, ast.BinOp, ast.Call, ast.Constant, ast.Continue, ast.Expr, ast.FloorDiv, ast.For,
    ast.FormattedValue, ast.If, ast.JoinedStr, ast.Load, ast.Name, ast.Store, ast.Tuple)


def _require(condition, detail):
    if not condition:
        raise ValueError('standalone source entry: ' + detail)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _historical(path, owner_ref, scope):
    if path is None:
        return None
    ref = _ref(path, owner_plan.COMPARISON)
    old = read_artifact(ref)
    _require(old['artifacts']['plan'] == owner_ref and old['matched_launch']['source'] == scope['source']
        and old['scope'] == 'SAME_FIXED_DEPTH_WHOLE_UCI_PROCESS' and old['selected'] == 'owner_native',
        'archived result describes a different source or resident owner boundary')
    launch = read_artifact(old['artifacts']['benchmark'])
    _require(launch['owner_plan'] == owner_ref and launch['stdin'] == old['matched_launch']['stdin'] == scope['stdin'],
        'archived result did not use the exact admitted source tape')
    return dict(artifact=ref, boundary='ORIGINAL_UI_WITH_NATIVE_RESIDENT_SEARCH',
        median_ns=old['measurements']['owner_native']['median_ns'],
        paired_with='ORIGINAL_UI_WITH_PER_SEARCH_MATERIALIZATION', transferable=False)


def _barriers(scope, resident):
    return dict(input='MATCH_EXACT_COMMAND_BEFORE_THIS_COMMAND_EFFECTS', stdin=scope['stdin'],
        publication='WAIT_FOR_REQUIRED_TERMINATOR_BEFORE_NEXT_COMMAND',
        source_driver='PRESERVE_ROOT_PROBE_ORDER_AND_OBSERVED_LOOKAHEAD',
        owner_state='ONE_OWNER_WITH_SOURCE_RESET_AND_REBIND_BARRIERS', resident=resident['barriers'],
        mismatch='REJECT_BEFORE_CURRENT_COMMAND_EFFECTS_WITH_NO_AUTOMATIC_MIDSTREAM_RESTART',
        clock='SOURCE_CLOCK_FRONTIER_RETAINS_ORIGINAL_FIXED_INPUT_CONTRACT',
        terminal='PUBLISH_LAST_SOURCE_OUTPUT_BEFORE_RELEASING_EXACT_OWNER_STATE')


def _static_branch(ui, row):
    tree = ast.parse(read_artifact(ui, json_document=False), filename=ui['uri'])
    branches = [node for node in ast.walk(tree) if type(node) is ast.If
                and [node.lineno,node.end_lineno,node.col_offset,node.end_col_offset] == row['source_span']]
    _require(len(branches) == 1, 'static reply has no exact original branch')
    nodes = [node for stmt in branches[0].body for node in ast.walk(stmt)]
    if any(type(node) not in _REPLY_NODES for node in nodes):
        return False
    calls = [node.func for node in nodes if type(node) is ast.Call]
    # Selected replies only read pinned module metadata and publish strings.
    # Attribute or item mutation, reflective calls and other providers stay open.
    return all((type(callee) is ast.Name and callee.id in {'print','hasattr','getattr'})
               or (type(callee) is ast.Attribute and callee.attr == 'items'
                   and type(callee.value) is ast.Attribute and type(callee.value.value) is ast.Name)
               for callee in calls) and all(type(node.ctx) is ast.Load for node in nodes if type(node) is ast.Attribute)


def _controller(scope, source, resident):
    _require(source['owner_scope'] == resident['artifacts']['caller_inventory']
        and all(source[key] == scope[key] for key in ('source','stdin','ui_contract','source_driver')),
        'controller and output evidence does not match the admitted resident state and original run')
    qualification = source['qualification']
    _require(qualification['tape_only'] is True and qualification['other_commands_admitted'] is False
        and qualification['search_outputs_not_replayed'] is True and qualification['deployment_authorized'] is False
        and qualification['prepared_values_and_visible_table_effects_closed_under_exact_pinned_providers'] is True,
        'source entry evidence is missing its isolated output contract')
    raw = read_artifact(scope['stdin'], json_document=False)
    lines = raw.decode('ascii').splitlines()
    commands = source['commands']
    _require(commands and len(lines) == len(commands)
        and all(row['ordinal'] == index and text == ' '.join(row['tokens'])
                for index,(row,text) in enumerate(zip(commands,lines,strict=True)))
        and raw == ''.join(text + '\n' for text in lines).encode('ascii'),
        'typed controller commands are not the immutable exact input stream')

    startup = source['startup']
    ui_source = read_artifact(scope['ui_contract'])['ui_source']
    io = startup['io']
    initial = startup['initial_owner']['fields']
    state = resident['state']
    _require(io['locale'] == 'C' and io['stdin_encoding'] == io['stdout_encoding'] == 'utf-8'
        and io['stdout_isatty'] is False and initial[state['move']['source_field']] == []
        and initial[state['score']['source_field']] == [] and initial[state['scalar_ports']['history']] == []
        and initial[state['scalar_ports']['counter']] == 0,
        'pinned startup needs an empty original owner and the observed text output providers')
    replies = {row['command_ordinal']:row for row in startup['static_replies']}
    searches = source['searches']
    positions = {row['position_ordinal']:index for index,row in enumerate(searches)}
    requests = {row['command_ordinal']:index for index,row in enumerate(searches)}
    _require(searches and len(positions) == len(requests) == len(searches)
        and all(row['position_ordinal'] + 1 == row['command_ordinal'] for row in searches)
        and commands[-1]['role'] == 'CLOSE' and sum(row['role'] == 'CLOSE' for row in commands) == 1,
        'every selected native search must follow its exact position and finish before the single terminal close')
    _require(set(replies) == {row['ordinal'] for row in commands if row['role'] == 'STATIC_REPLY'}
        and len(replies) == len(startup['static_replies']) and all(index < min(positions) for index in replies),
        'static replies must occupy only the original startup with no active search')

    steps, blockers = [], []
    for text, row in zip(lines,commands,strict=True):
        index, role = row['ordinal'], row['role']
        step = dict(row, index=index, line=text, input_guard='EXACT_LINE_BEFORE_EFFECTS')
        if role == 'STATIC_REPLY':
            reply = replies[index]
            if (reply['authority'] != 'EXACT_ORIGINAL_NON_SEARCH_OUTPUT_AT_A_PINNED_CONTROLLER_SITE'
                    or reply['flush'] is not True or not _static_branch(ui_source,row)):
                blockers.append('STARTUP_REPLY_EFFECT_PROVIDER_UNRESOLVED')
            step.update(operation='STATIC_REPLY', output_policy='PINNED_STARTUP_TEXT_ONLY',
                required_publication=reply['lines'][-1], reply_lines=reply['lines'])
        elif role == 'POSITION':
            _require(index in positions, 'position lacks a typed prepared state for its native search')
            step.update(operation='POSITION', output_policy='NONE', search_ordinal=positions[index])
        elif role == 'SEARCH':
            _require(index in requests, 'native search is missing its checked source input')
            step.update(operation='SEARCH', output_policy='COMPUTE_FROM_LIVE_PROBES_AND_MOVE_MAP',
                search_ordinal=requests[index], required_publication='SOURCE_FINAL_MOVE_AND_OPTIONAL_REPLY')
        elif role == 'CLOSE':
            step.update(operation='CLOSE', output_policy='NONE')
        else:
            blockers.append('UNSUPPORTED_ORIGINAL_COMMAND_BRANCH')
        steps.append(step)
    _require(source['output_protocol']['search_results'] == 'EVALUATE_NATIVE_SEARCH_AND_ORIGINAL_OUTPUT_DEPENDENCIES',
        'output protocol requires current native search and original consumers')
    preparations = source['position_preparations']
    _require(len(preparations) == len(searches)
        and startup['initial_table_version'] == searches[0]['position_table_version']
        and startup['initial_owner']['configuration'] == searches[0]['configuration'],
        'the frozen preparation must start with the observed original providers')
    for index,(row,preparation) in enumerate(zip(searches,preparations,strict=True)):
        pure = preparation['normal_pure_value']
        command = commands[row['position_ordinal']]
        _require(row['history_authority'] == 'EXACT_ORIGINAL_PREPARED_VALUE_ON_CLOSED_COMMAND_AND_PROVIDERS'
            and row['history'] and row['history'][-1] == row['root']
            and type(row['requested_depth']) is int and row['requested_depth'] > 0
            and pure['root'] == row['root'] and preparation['ordinal'] == row['position_ordinal']
            and preparation['source_selector'] == command['selector'] and preparation['observable_holder_writes'] == []
            and pure.get('table_version', row['position_table_version']) == row['position_table_version']
            and (index == 0 or row['position_table_version'] == searches[index-1]['search_table_version']),
            'the original position, requested depth or cross-search scoring table requires an open provider')
    return steps,startup,searches,source['output_protocol'],sorted(set(blockers))


def propose(owner_plan_path, source_entry_path=None, prior_selection_path=None):
    """Nominate only the exact checked source controller and its output dataflow."""
    owner_ref = _ref(owner_plan_path, owner_plan.SCHEMA)
    resident = owner_plan.check(owner_ref['uri'])
    old_scope = resident['scope']
    scope = {key:old_scope[key] for key in ('source','stdin','source_driver','ui_contract')}
    scope.update(mode=MODE, owner_plan=owner_ref, normalization=source_cost.NORMALIZATION)
    prior = _historical(prior_selection_path, owner_ref, scope)
    source_ref = _ref(source_entry_path, SOURCE) if source_entry_path is not None else None
    commands, startup, searches, protocol = [], None, [], None
    blockers = []
    if resident['decision']['runtime_experiment_authorized'] is not True:
        blockers.append('SOURCE_OWNER_CONTINUITY_UNAVAILABLE')
    if source_ref is None:
        blockers.append('SOURCE_CONTROLLER_AND_OUTPUT_EVIDENCE_UNAVAILABLE')
    elif not blockers:
        from scar.analysis import scalar_recursive_entry
        source = scalar_recursive_entry.check(source_ref['uri'])
        commands, startup, searches, protocol, open_edges = _controller(scope, source, resident)
        blockers.extend(open_edges)
    return dict(schema=SCHEMA, schema_version=1, artifacts=dict(owner_plan=owner_ref, source_entry=source_ref,
            prior_selection=prior['artifact'] if prior is not None else None), scope=scope,
        commands=commands,startup=startup,searches=searches,output_protocol=protocol,
        barriers=_barriers(scope, resident), resident_state=resident['state'],
        requirements=dict(cxx_standard=20, owner_lanes=1, fixed_tape_only=True, original_runtime_required=False,
            search_outputs_from_live_native_data=True, copy_captured_search_output=False, output_timing_only_normalization=True),
        costs=dict(status='UNKNOWN_FOR_STANDALONE_ENTRY', historical=prior, whole_entry_native_median_ns=None,
            live_clock_behavior_outside_fixed_tape=None, peak_native_state_bytes=None),
        decision=dict(selected='RETAIN_ORIGINAL_SOURCE', native_experiment_authorized=not blockers,
                      deployment_authorized=False, blockers=blockers))


def _load_plan(path):
    stored = read_artifact(_ref(path, SCHEMA))
    refs = stored['artifacts']
    rebuilt = propose(refs['owner_plan']['uri'], refs['source_entry']['uri'] if refs['source_entry'] else None,
        refs['prior_selection']['uri'] if refs['prior_selection'] else None)
    _require(_canonical(stored) == _canonical(rebuilt), 'source entry differs from cold replay')
    return rebuilt


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _executable(arm, plan_ref, source, expected_driver):
    ref = arm['artifact']
    _require(ref['schema'] == EXECUTABLE, 'standalone process requires the exact published executable manifest')
    artifact = read_artifact(ref)
    _require(artifact['schema_version'] == 1 and artifact['entry_plan'] == plan_ref
        and artifact['source'] == source and artifact['cxx_standard'] == 20
        and artifact['python_runtime_required'] is artifact['deployment_authorized'] is False,
        'compiled executable does not implement the admitted source entry')
    driver_ref = artifact['driver_library']
    driver = read_artifact(driver_ref)
    _require(driver_ref['schema'] == 'scar.scalar-recursive-driver-library'
        and driver['schema_version'] == 2 and driver['code'] == expected_driver
        and driver['abi']['control'] == 'ONE_WORLD_ACROSS_REBOUND_SOURCE_GENERATORS',
        'standalone process does not bind the admitted resident driver ABI')
    root = Path(ref['uri']).parent
    binary = artifact['binary']
    _require(binary['uri'] == 'bin/scar_recursive_entry' and binary['schema'] == 'scar.elf',
        'expected an exact package-local C++ entry ELF')
    path = (root / binary['uri']).resolve(strict=True)
    _require(path.is_relative_to(root) and arm['executable'] == dict(binary, uri=str(path)),
        'launched C++ binary escaped or differs from its published manifest')
    _require(read_artifact(arm['executable'], json_document=False).startswith(b'\x7fELF'), 'entry binary is not an ELF')


def _launch(row, run):
    command, mode = row['command'], row['arm']
    cpus = ','.join(map(str, run['cpu_ids']))
    _require(type(command) is list and len(command) >= 8 and all(type(arg) is str for arg in command)
        and Path(command[0]).name == 'taskset' and command[1:3] == ['-c', cpus]
        and Path(command[3]).name == 'timeout' and command[4:7] == ['-k','2','60'],
        'entry launch must share the direct process CPU and timeout')
    tail = command[7:]
    arm = run['arms'][mode]
    if mode == 'original':
        _require(len(tail) == 4 and Path(tail[0]).is_absolute()
            and str(Path(tail[0]).resolve(strict=True)) == arm['python_executable']
            and tail[1:] == ['-B','-u',run['source']['uri']], 'baseline must directly run the original source')
        return None
    _require(tail[:1] == [arm['executable']['uri']], 'native arm must execute only the published ELF')
    observed = row['kind'] == 'OBSERVED_WARMUP'
    if observed:
        _require(len(tail) == 3 and tail[1] == '--observe' and Path(tail[2]).is_absolute(),
            'native oracle capture requires its exact out-of-band observation directory')
        return Path(tail[2]).resolve(strict=True)
    _require(len(tail) == 1, 'instrumentation cannot enter a clean native trial')
    return None


def _observed_state(reference, arm, sessions, directory):
    _require(reference is not None and reference['schema'] == OBSERVATION, 'missing independent native process trace')
    result = read_artifact(reference)
    _require(result['schema_version'] == 1 and result['artifact'] == arm['artifact']
        and result['observed_from_exact_launch'] is result['original_sessions_matched'] is True
        and len(result['sessions']) == len(sessions), 'native observation does not bind all selected source sessions')
    files = result['trace_files']
    paths = [Path(ref['uri']).resolve(strict=True) for ref in files]
    _require(len(set(paths)) == len(paths) == 2 * len(sessions)
        and all(path.is_relative_to(directory) for path in paths)
        and sum(path.suffix == '.probes' for path in paths) == sum(path.suffix == '.world' for path in paths) == len(sessions),
        'native probe and state observations must originate from their exact observed process run')
    from scar.codegen import scalar_recursive_entry as executable
    _require(_canonical(result) == _canonical(executable.project_observation(arm['artifact']['uri'], directory)),
        'standalone source observation differs from independently decoded C++ trace bytes')
    for actual, expected in zip(result['sessions'], sessions, strict=True):
        _require(actual['table_version'] == expected['table_version']
            and actual['yields'] == [{key:row[key] for key in ('value','nodes')} for row in expected['yields']]
            and actual['outcome'] == 'consumer_close' and expected['exit']['kind'] == 'GENERATOR_CLOSE'
            and actual['initial_move_entries'] == expected['initialized']['move_table']['size']
            and actual['initial_score_entries'] == expected['initialized']['score_table']['size']
            and actual['final_nodes'] == expected['final']['nodes']
            and actual['final_move_entries'] == expected['final']['move_table']['size']
            and actual['final_score_entries'] == expected['final']['score_table']['size']
            and actual['ordered_map_digests'] == expected['final']['ordered_map_digests'],
            'native probe order, selected table, or complete ordered map differs from original source')
    return dict(artifact=reference, trace_files=files, exact_probe_records=sum(len(row['yields']) for row in sessions),
                ordered_map_sessions=len(sessions))


def _paired_rows(run, sessions, depth):
    rows = run['rows']
    _require(type(rows) is list and len(rows) % 2 == 0 and len(rows) >= 6,
        'expected one observed warmup pair and at least two complete clean pairs')
    samples, checks, observation = {arm:[] for arm in _ARMS}, set(), None
    orders = []
    for i, row in enumerate(rows):
        number, slot = divmod(i, 2)
        order = _ARMS if number % 2 == 0 else _ARMS[::-1]
        if slot == 0:
            orders.append(list(order))
        warmup = number == 0
        _require(type(row['round']) is int and row['round'] == number and row['arm'] == order[slot]
            and row['kind'] == ('OBSERVED_WARMUP' if warmup else 'CLEAN_MEASURED')
            and type(row['returncode']) is int and row['returncode'] == 0
            and type(row['wall_ns']) is int and row['wall_ns'] > 0,
            'paired row has an unmatched launch order, timing scope or exit status')
        directory = _launch(row, run)
        raw, stderr = (read_artifact(row[key], json_document=False) for key in ('stdout','stderr'))
        lines = raw.splitlines()
        info = [line for line in lines if line.startswith(b'info depth ')]
        _require(raw.endswith(b'\n') and b'\r' not in raw and info and all(_TIMING_FIELDS.match(line) for line in info)
            and b'info string aborted at ' not in raw, 'native or source output lost typed timing fields or was interrupted')
        source_cost._output(raw, depth, len(sessions))
        canonical = b'\n'.join(_TIMING_FIELDS.sub(rb'\1\2 ', line, count=1) for line in lines)
        digest = sha256(canonical).hexdigest()
        checks.add((digest, stderr))
        if directory is not None:
            observation = _observed_state(row['observation'], run['arms']['entry_native'], sessions, directory)
        else:
            _require(row['observation'] is None, 'row unexpectedly attached an instrumented observation')
        if not warmup:
            samples[row['arm']].append(row['wall_ns'])
    _require(len(checks) == 1 and observation is not None,
        'native or original execution changed canonical stdout, stderr or omitted correctness evidence')
    return samples, orders, checks.pop()[0], observation


def compare(plan_path, benchmark_path):
    """Choose on direct fresh-process observations at the admitted source boundary."""
    refs = dict(plan=_ref(plan_path, SCHEMA), benchmark=_ref(benchmark_path, BENCHMARK))
    plan = _load_plan(refs['plan']['uri'])
    run = read_artifact(refs['benchmark'])
    scope = plan['scope']
    _require(plan['decision']['native_experiment_authorized'] is True and run['schema_version'] == 1
        and run['entry_plan'] == refs['plan'] and run['source'] == scope['source'] and run['stdin'] == scope['stdin']
        and run['normalization'] == scope['normalization'] and run['argv'] == []
        and type(run['cpu_ids']) is list and len(run['cpu_ids']) == 1
        and type(run['cpu_ids'][0]) is int and run['cpu_ids'][0] >= 0
        and type(run['cwd']) is str and Path(run['cwd']).is_absolute() and type(run['environment']) is dict
        and all(type(k) is type(v) is str for k,v in run['environment'].items()),
        'the measured native and original processes differ from the admitted source tape and launch cohort')
    read_artifact(run['launcher'], json_document=False)
    arms = run['arms']
    original, candidate = (arms[name] for name in _ARMS)
    _require(set(arms) == set(_ARMS) and set(original) == {'python_executable','python_version'}
        and set(candidate) == {'artifact','executable'} and type(original['python_executable']) is str
        and type(original['python_version']) is str and original['python_version'].strip(),
        'comparison must name one direct original interpreter and one standalone C++ executable')
    python = Path(original['python_executable'])
    _require(python.is_absolute() and python.is_file() and python.resolve(strict=True) == python,
        'original interpreter must remain at the recorded real path')
    source_driver = read_artifact(scope['source_driver'])
    _executable(candidate, refs['plan'], scope['source'], source_driver['driver']['code'])
    commands = read_artifact(run['stdin'], json_document=False).decode('ascii').splitlines()
    sessions = source_driver['sessions']
    depths = {row['requested_depth'] for row in plan['searches']}
    _require(len(depths) == 1, 'paired decision needs the same requested depth for the admitted original searches')
    depth = depths.pop()
    _require(source_cost._request_count(commands,depth) == len(sessions), 'pinned original search cardinality changed')
    samples, orders, digest, observation = _paired_rows(run, sessions, depth)
    measures, comparisons = source_cost._envelopes(samples, minimum_gain_bps=0)
    result = comparisons['entry_native']
    return dict(schema=COMPARISON, schema_version=1, artifacts=refs, scope='MATCHED_DIRECT_WHOLE_SOURCE_PROCESS',
        matched_launch={k:run[k] for k in ('source','stdin','normalization','cwd','argv','environment','cpu_ids')},
        launcher=run['launcher'], executable=candidate, original_interpreter=original, depth=depth,
        searches=len(sessions), clean_pairs=len(samples['original']), warmup_pairs=1, orders=orders,
        exact_canonical_stdout_sha256=digest, observation=observation, measurements=measures,
        original_minus_native_evidence=result, selected='entry_native' if result['verified_observed_gain'] else 'original',
        deployment_authorized=False, historical_cost_used_for_selection=False)


def check(path):
    schema = json.loads(Path(path).read_bytes())['schema']
    if schema == SCHEMA:
        return _load_plan(path)
    _require(schema == COMPARISON, 'unsupported standalone entry plan or comparison')
    stored = read_artifact(_ref(path, COMPARISON))
    refs = stored['artifacts']
    rebuilt = compare(refs['plan']['uri'], refs['benchmark']['uri'])
    _require(_canonical(stored) == _canonical(rebuilt), 'matched standalone decision differs from cold replay')
    return rebuilt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('owner-plan','source-entry','prior-selection','check','out','plan','benchmark'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    optionals = (args.owner_plan,args.source_entry,args.prior_selection)
    comparison = (args.plan,args.benchmark)
    if args.check and not any((*optionals,*comparison,args.out)):
        result = check(args.check)
    elif args.check is None and args.owner_plan and args.out and not any(comparison):
        result = propose(args.owner_plan,args.source_entry,args.prior_selection)
    elif args.check is None and all(comparison) and args.out and not any(optionals):
        result = compare(*comparison)
    else:
        parser.error('supply --check, plan inputs, or --plan and --benchmark with --out')
    if args.out:
        with args.out.open('x') as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    summary = result['decision'] if result['schema'] == SCHEMA else {
        key:result[key] for key in ('selected','measurements','exact_canonical_stdout_sha256')}
    print(_canonical(summary))


if __name__ == '__main__':
    main()

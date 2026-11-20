"""Plan a checked sequential native owner for parameterized command streams."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import scalar_recursive_entry_plan as entry_plan
from scar.optimization import scalar_sorted_transition_cost as source_cost


SCHEMA = 'scar.scalar-recursive-parameterized-plan'
GRAMMAR = 'scar.scalar-recursive-parameterized-entry'
MODE = 'PARAMETERIZED_PACED_FIXED_DEPTH'
BENCHMARK = 'scar.scalar-recursive-parameterized-paired-run'
COMPARISON = 'scar.scalar-recursive-parameterized-paired-selection'
EXECUTABLE = 'scar.scalar-recursive-parameterized-executable'
OBSERVATION = 'scar.scalar-recursive-parameterized-observation'
_KINDS = ('EMPTY','UCI','ISREADY','POSITION_START','POSITION_FEN','GO_DEPTH','NEW_GAME','QUIT')
_ARMS = ('original','entry_native')
_DECIMAL = re.compile(r'[+-]?[0-9]+')
_TERMINAL_INFO = re.compile(rb' score cp -?[0-9]+$')


def _require(condition, detail):
    if not condition:
        raise ValueError('parameterized recursive entry: ' + detail)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _historical(path, anchor_ref, source):
    if path is None:
        return None
    ref = _ref(path, entry_plan.COMPARISON)
    old = read_artifact(ref)
    _require(old['artifacts']['plan'] == anchor_ref and old['matched_launch']['source'] == source
        and old['scope'] == 'MATCHED_DIRECT_WHOLE_SOURCE_PROCESS', 'historical timings belong to another source plan')
    launch = read_artifact(old['artifacts']['benchmark'])
    _require(launch['entry_plan'] == anchor_ref and launch['stdin'] == old['matched_launch']['stdin'],
        'archived fixed-tape measurements changed their launch boundary')
    return dict(selection=ref, scope='ARCHIVED_DIRECT_FIXED_TAPE_ONLY',
        original_median_ns=old['measurements']['original']['median_ns'],
        native_median_ns=old['measurements']['entry_native']['median_ns'],
        reusable_for_new_profitability=False)


def _state(anchor):
    owner = anchor['resident_state']
    return dict(move=dict(source_field=owner['move']['source_field'], storage='NATIVE_ORDERED_MAP',
            carry='ACROSS_ADJACENT_GO_ON_ONE_RECEIVER', lifetime_end='NEW_GAME_OR_TERMINATION',
            read='SOURCE_ORDERED_PV_AND_MOVE_LOOKUPS'),
        score=dict(source_field=owner['score']['source_field'], reset='FRESH_MAP_AT_EACH_GO',
            original_reset_site=owner['score']['original_reset'], lifetime_end='NEXT_GO_OR_RECEIVER_DESTRUCTION'),
        receiver=dict(clock=owner['scalar_ports']['clock'], counter=owner['scalar_ports']['counter'],
            history=owner['scalar_ports']['history'], root=owner['scalar_ports']['root'],
            lifetime='FROM_ORIGINAL_RECEIVER_CONSTRUCTION_TO_NEW_GAME_OR_TERMINATION'),
        controller=dict(history='ORDERED_ORIGINAL_POSITION_HISTORY', halfmove_clock='ORIGINAL_UI_OWNED_VALUE',
            history_reset='POSITION_ONLY', lag_prev_reset='NEW_GAME'),
        module_table=dict(placement='SINGLE_LOGICAL_CONTROLLER_EPOCH',
            use='POSITION_PREPARATION_BEFORE_NEXT_GO_TABLE_SELECTION',
            update='SOURCE_SELECTOR_AT_GO', survives_new_game=True))


def _step(operation, *, effects=(), retains=(), publication=None):
    return dict(operation=operation, precondition='IDLE_AND_COMMAND_TYPED_BEFORE_EFFECTS',
        effects=list(effects), retains=list(retains), publication=publication)


def _transitions():
    position = dict(effects=('PREPARE_COMPLETE_HISTORY_AND_HALFMOVE_IN_PRIVATE_STORAGE',
        'READ_CURRENT_MODULE_TABLE', 'COMMIT_NEW_CONTROLLER_POSITION'),
        retains=('OWNER_MOVE', 'OWNER_SCORE', 'MODULE_TABLE'))
    return [
        _step('EMPTY'),
        _step('UCI', effects=('READ_IMMUTABLE_STARTUP_METADATA',), publication='SOURCE_UCI_ACK'),
        _step('ISREADY', publication='SOURCE_READY_ACK'),
        _step('POSITION_START', **position),
        _step('POSITION_FEN', **position),
        _step('GO_DEPTH', effects=('ARM_ORIGINAL_SOURCE_CLOCK_PORTS', 'RESET_COUNTER_HISTORY_SCORE_AND_ROOT',
            'SELECT_AND_COMMIT_ORIGINAL_MODULE_TABLE', 'RESUME_BALANCED_NATIVE_PROBES',
            'READ_LIVE_ORDERED_MOVE_MAP_FOR_PV', 'CLOSE_AT_ORIGINAL_CONSUMER_FRONTIER'),
            retains=('SAME_RECEIVER', 'MOVE_MAP_EPOCH', 'CONTROLLER_HISTORY', 'CONTROLLER_HALFMOVE'),
            publication='SOURCE_INFO_AND_FINAL_BESTMOVE_WITH_OPTIONAL_PONDER'),
        _step('NEW_GAME', effects=('DESTROY_AND_REPLACE_OWNER', 'RESET_LAG_AND_PREVIOUS_SAMPLE'),
            retains=('CONTROLLER_HISTORY', 'CONTROLLER_HALFMOVE', 'MODULE_TABLE')),
        _step('QUIT', effects=('SOURCE_IDLE_STOP_AND_JOIN', 'DESTROY_OWNER_AND_TERMINATE')),
    ]


def _obligations():
    return [
        dict(id='command_guard', status='LOWERING_CHECK',
            detail='Validate the whole command, including argument domains and moves, before any current-command effect.'),
        dict(id='owner_versions', status='LOWERING_CHECK',
            detail='Keep MOVE on the current receiver, renew SCORE on GO, and preserve the old module table across NEW_GAME and POSITION.'),
        dict(id='position_types', status='LOWERING_CHECK',
            detail='Lower both FEN orientations, original mailbox and rights, prepared history and halfmove updates using the live table.'),
        dict(id='numeric_and_stack', status='RUNTIME_GUARD',
            detail='Keep checked integer arithmetic and native recursion limits. A request outside the proven domain must explicitly fail.'),
        dict(id='time_and_mode', status='RUNTIME_GUARD',
            detail='Keep fixed-depth lookahead and original clock ports. Concurrent commands and adaptive time modes are outside the paced scope.'),
        dict(id='open_provider_effects', status='UNRESOLVED',
            detail='Broader Python provider effects, arbitrary command coverage, and completion for every position require further proofs.'),
        dict(id='profitability', status='UNRESOLVED',
            detail='Use fresh same-stream original and native direct-process measurements, with correctness validated separately.'),
    ]


def _admit_grammar(original, anchor, scope):
    _require(original['source_entry'] == anchor['artifacts']['source_entry']
        and all(original[key] == scope[key] for key in ('source','source_driver','ui_contract')),
        'source grammar does not join the exact initialized original entry')
    grammar = original['grammar']
    commands = grammar['commands']
    _require([row['kind'] for row in commands] == list(_KINDS)
        and len({tuple(row['tokens']) for row in commands}) == len(commands)
        and all(type(value) is str and value and value.isascii() and not any(c.isspace() for c in value)
                for row in commands for value in row['tokens'])
        and grammar['tokenization'] == 'ASCII_SUBSET_OF_ORIGINAL_WHITESPACE_SPLIT'
        and grammar['dispatch'] == 'MATCH_COMPLETE_COMMAND_BEFORE_EFFECTS'
        and grammar['protocol'] == 'ORDERED_ACKNOWLEDGED_IDLE_COMMANDS'
        and grammar['malformed_or_unsupported'] == 'DECLINE_BEFORE_CURRENT_COMMAND_EFFECTS',
        'controller grammar cannot support the selected serial command effects')
    low, high = (grammar['depth_range'][key] for key in ('minimum','maximum'))
    by_kind = {row['kind']:row for row in commands}
    _require(type(low) is type(high) is int and 1 <= low <= high <= 4
        and by_kind['GO_DEPTH']['arguments'] == [dict(name='depth',type='ASCII_SIGNED_I64',minimum=low,maximum=high)]
        and all(by_kind[kind]['optional_tail'] is None for kind in ('EMPTY','UCI','ISREADY','GO_DEPTH','NEW_GAME','QUIT')),
        'typed depth or control bindings differ from the bounded solver')

    positions, config, effects = (original[key] for key in ('positions','configuration','effects'))
    topology = positions['fen']['controller_orientation']
    defaults = anchor['startup']['initial_owner']
    _require(topology['direct_history_records'] == 1 and topology['alternate_history_records'] == 2
        and positions['fen']['read_table']['use'] == 'CURRENT_MODULE_VERSION_BEFORE_NEXT_SEARCH'
        and positions['fen']['read_table']['versions']
        and all(by_kind[key]['optional_tail']['marker'] == positions['suffix_marker']
                for key in ('POSITION_START','POSITION_FEN'))
        and effects['new_game']['fields'] == defaults['fields']
        and set(effects['new_game']['retains']) == {'CONTROLLER_POSITION_HISTORY','CONTROLLER_HALFMOVE_CLOCK',
            'MODULE_SCORING_VERSION','DEFAULT_OPTIONS'}
        and effects['close']['requires_idle_owner'] is True
        and effects['historical_startup_replies'] == anchor['startup']['static_replies'],
        'position preparation, owner replacement or observable startup effects are open')

    clocks, values = config['clocks'], config['values']
    budget, cap, soft = (clocks[key] for key in ('default_budget_seconds','hard_cap_seconds','soft_initial_epoch'))
    clock_fields = anchor['resident_state']['scalar_ports']['clock']
    _require(values == defaults['configuration'] and set(config['source_origins']) == set(values)
        and all(type(value) is int and -(1 << 63) <= value < (1 << 63) for value in values.values())
        and type(budget) is type(cap) is type(soft) is int and 0 < cap <= budget <= 1 << 53
        and 0 < soft <= 1 << 63 and int(float(soft)) == soft
        and [clocks['hard_field'],clocks['soft_field']] == clock_fields,
        'numeric defaults or fixed clock ports differ from the checked owner ABI')
    frontiers = original['proof_frontiers']
    _require(frontiers['held_out_input_used'] is False and frontiers['deployment_authorized'] is False,
        'experimental grammar may not borrow held-out measurements or deployment authority')
    return dict(grammar=grammar,ports=original['ports'],positions=positions,configuration=config,
                source_effects=effects,proof_frontiers=frontiers), []


def propose(entry_plan_path, source_grammar_path=None, prior_selection_path=None):
    anchor_ref = _ref(entry_plan_path, entry_plan.SCHEMA)
    anchor = entry_plan.check(anchor_ref['uri'])
    scope = {key:anchor['scope'][key] for key in ('source','source_driver','ui_contract','normalization')}
    scope.update(mode=MODE,entry_plan=anchor_ref)
    prior = _historical(prior_selection_path,anchor_ref,scope['source'])
    grammar_ref = _ref(source_grammar_path,GRAMMAR) if source_grammar_path is not None else None
    fields = dict(grammar=None,ports=None,positions=None,configuration=None,source_effects=None,proof_frontiers=None)
    blockers = []
    if anchor['decision']['native_experiment_authorized'] is not True:
        blockers.append('MISSING_CHECKED_SOURCE_ENTRY')
    if grammar_ref is None:
        blockers.append('MISSING_SOURCE_COMMAND_GRAMMAR')
    elif not blockers:
        from scar.analysis import scalar_recursive_parameterized_entry
        grammar = scalar_recursive_parameterized_entry.check(grammar_ref['uri'])
        fields, grammar_blockers = _admit_grammar(grammar,anchor,scope)
        blockers.extend(grammar_blockers)
    return dict(schema=SCHEMA,schema_version=1,artifacts=dict(entry_plan=anchor_ref,source_grammar=grammar_ref,
            prior_selection=prior['selection'] if prior is not None else None), scope=scope,**fields,
        startup=anchor['startup'],state=_state(anchor),state_transitions=_transitions(),
        resources=dict(cpu_owner_lanes=1,concurrent_command_effects=False,placement='COMMON_CPU_IN_MATCHED_LAUNCH',
            peak_owner_memory_bytes=None),
        requirements=dict(cxx_standard=20,paced_only=True,held_out_input_not_embedded=True,
            parse_before_effects=True,search_and_pv_from_live_native_values=True,checked_signed_integer=True,
            clock_outputs_normalized=['time','nps'], mid_command_restart=False),
        costs=dict(status='UNKNOWN_FOR_FRESH_PARAMETERIZED_STREAMS',historical=prior,estimates_ns=None),
        obligations=_obligations(),decision=dict(selected='RETAIN_ORIGINAL_SOURCE',
            native_experiment_authorized=not blockers,deployment_authorized=False,blockers=blockers))


def _load_plan(path):
    stored = read_artifact(_ref(path,SCHEMA))
    refs = stored['artifacts']
    rebuilt = propose(refs['entry_plan']['uri'],refs['source_grammar']['uri'] if refs['source_grammar'] else None,
        refs['prior_selection']['uri'] if refs['prior_selection'] else None)
    _require(_canonical(stored) == _canonical(rebuilt), 'parameterized plan differs from cold replay')
    return rebuilt


def _native(arm, plan, plan_ref):
    manifest = arm['artifact']
    _require(manifest['schema'] == EXECUTABLE, 'native process must use a published parameterized ELF')
    native = read_artifact(manifest)
    _require(native['schema_version'] == 1 and native['parameterized_plan'] == plan_ref
        and native['source'] == plan['scope']['source'] and native['cxx_standard'] == 20
        and native['python_runtime_required'] is native['deployment_authorized'] is False,
        'native executable does not follow the authorized typed command plan')
    driver_ref = native['driver_library']
    driver = read_artifact(driver_ref)
    _require(driver_ref['schema'] == 'scar.scalar-recursive-driver-library' and driver['schema_version'] == 2
        and driver['code'] == plan['ports']['driver']['code']
        and driver['abi']['control'] == 'ONE_WORLD_ACROSS_REBOUND_SOURCE_GENERATORS',
        'parameterized binary changed its exact native recursive provider')
    root = Path(manifest['uri']).parent
    binary = native['binary']
    _require(binary['uri'] == 'bin/scar_recursive_parameterized' and binary['schema'] == 'scar.elf',
        'expected the package-local ELF that implements the parameterized controller')
    path = (root / binary['uri']).resolve(strict=True)
    _require(path.is_relative_to(root) and arm['executable'] == dict(binary,uri=str(path))
        and read_artifact(arm['executable'],json_document=False).startswith(b'\x7fELF'),
        'timed process does not bind the exact package-local compiled ELF')


def _tape_tokens(raw):
    _require(raw.endswith(b'\n') and all(32 <= byte <= 126 or 9 <= byte <= 13 for byte in raw),
        'benchmark input must be a finite line-oriented ASCII stream')
    return [text.split() for text in raw.decode('ascii').split('\n')[:-1]]


def _tape(raw, plan):
    commands = _tape_tokens(raw)
    anchor = read_artifact(plan['scope']['entry_plan'])
    _require(commands != _tape_tokens(read_artifact(anchor['scope']['stdin'],json_document=False)),
        'the fresh benchmark stream must differ semantically from the admission tape')
    grammar = plan['grammar']
    heads = {row['kind']:row['tokens'] for row in grammar['commands']}
    literals = {tuple(head):kind for kind,head in heads.items() if kind in ('EMPTY','UCI','ISREADY','NEW_GAME','QUIT')}
    lo, hi = (grammar['depth_range'][key] for key in ('minimum','maximum'))
    replies = {tuple(tokens):row['lines'][-1].encode('ascii')
               for tokens,row in zip((heads['UCI'],heads['ISREADY']),plan['startup']['static_replies'],strict=True)}
    effects, depths = [], []
    for command in commands:
        literal = literals.get(tuple(command))
        if literal in ('UCI','ISREADY'):
            effects.append(('REPLY',replies[tuple(command)]))
        elif literal in ('EMPTY','NEW_GAME','QUIT'):
            continue
        elif command[:len(heads['GO_DEPTH'])] == heads['GO_DEPTH']:
            _require(len(command) == len(heads['GO_DEPTH']) + 1 and _DECIMAL.fullmatch(command[-1]),
                'timed stream contains a search outside the fixed-depth typed grammar')
            depth = int(command[-1])
            _require(lo <= depth <= hi, 'timed stream contains a depth outside the checked planner domain')
            effects.append(('SEARCH',depth))
            depths.append(depth)
        else:
            # The native parser and original differential certify actual position values.
            _require(any(command[:len(heads[kind])] == heads[kind] for kind in ('POSITION_START','POSITION_FEN')),
                'benchmark includes an unsupported controller command')
    _require(depths and commands[-1] == heads['QUIT'] and commands.count(heads['QUIT']) == 1,
        'benchmark must complete a finite fresh stream with one terminal quit')
    return effects,depths


def _stdout(raw, effects):
    _require(raw.endswith(b'\n') and b'\r' not in raw and b'info string aborted at ' not in raw,
        'process output interrupted or left the original source publication format')
    reply_markers = {value for kind,value in effects if kind == 'REPLY'}
    active, observed = [], []
    lines = raw.splitlines()
    for line in lines:
        if line.startswith(b'info depth '):
            match = entry_plan._TIMING_FIELDS.match(line)
            _require(match is not None, 'depth output lost its ordered and typed timing fields')
            active.append((int(match[1].rsplit(b' ',1)[1]),bool(_TERMINAL_INFO.search(line))))
        elif line.startswith(b'bestmove '):
            _require(active, 'native or source process omitted search publication')
            observed.append(('SEARCH',max(depth for depth,_ in active),active[-1][1]))
            active.clear()
        elif line in reply_markers:
            _require(not active,'reply broke the paced search publication barrier')
            observed.append(('REPLY',line,False))
    _require(not active and len(observed) == len(effects), 'process outputs differ in command acknowledgements')
    for (kind,value),(actual,seen,terminal) in zip(effects,observed,strict=True):
        _require(kind == actual and (seen == value or (kind == 'SEARCH' and terminal and 1 <= seen < value)),
            'process output reached a different depth or acknowledgement order')
    canonical = b'\n'.join(entry_plan._TIMING_FIELDS.sub(rb'\1\2 ',line,count=1) for line in lines)
    return sha256(canonical).hexdigest()


def _observation(reference, run, directory, count):
    _require(reference is not None and reference['schema'] == OBSERVATION, 'missing held-out native probe and state bytes')
    stored = read_artifact(reference)
    arm = run['arms']['entry_native']['artifact']
    traces = stored['trace_files']
    files = {Path(ref['uri']).resolve(strict=True) for ref in traces}
    expected = {directory / f'{i}.{suffix}' for i in range(count) for suffix in ('probes','world','prepare')}
    _require(stored['schema_version'] == 1 and stored['artifact'] == arm and len(traces) == 3 * count and files == expected,
        'held-out native trace does not cover the exact observed process files')
    from scar.codegen import scalar_recursive_parameterized as executable
    replay = executable.project_observation(arm['uri'],directory,run['original_capture']['uri'])
    _require(_canonical(stored) == _canonical(replay), 'held-out state result differs from independently decoded trace bytes')
    return reference


def _paired_rows(run, effects, search_count):
    rows = run['rows']
    _require(type(rows) is list and len(rows) >= 6 and len(rows) % 2 == 0,
        'at least two clean paired process launches and one observed warmup are required')
    samples, outputs, orders, observation = {mode:[] for mode in _ARMS}, set(), [], None
    for i,row in enumerate(rows):
        number, slot = divmod(i,2)
        order = _ARMS if number % 2 == 0 else _ARMS[::-1]
        if slot == 0:
            orders.append(list(order))
        _require(type(row['round']) is int and row['round'] == number and row['arm'] == order[slot]
            and row['kind'] == ('OBSERVED_WARMUP' if number == 0 else 'CLEAN_MEASURED')
            and type(row['returncode']) is int and row['returncode'] == 0
            and type(row['wall_ns']) is int and row['wall_ns'] > 0,
            'trial does not belong to the matched fresh-process scope')
        directory = entry_plan._launch(row,run)
        raw, stderr = (read_artifact(row[key],json_document=False) for key in ('stdout','stderr'))
        outputs.add((_stdout(raw,effects),stderr))
        if directory is not None:
            observation = _observation(row['observation'],run,directory,search_count)
        else:
            _require(row['observation'] is None, 'direct clean launch unexpectedly has diagnostic evidence')
        if number:
            samples[row['arm']].append(row['wall_ns'])
    _require(len(outputs) == 1 and observation is not None, 'matched standalone output or correctness evidence changed')
    return samples,orders,outputs.pop()[0],observation


def compare(plan_path,benchmark_path):
    """Choose only from fresh matched parameterized direct process observations."""
    refs = dict(plan=_ref(plan_path,SCHEMA),benchmark=_ref(benchmark_path,BENCHMARK))
    plan = _load_plan(refs['plan']['uri'])
    run = read_artifact(refs['benchmark'])
    scope = plan['scope']
    _require(plan['decision']['native_experiment_authorized'] is True and run['schema_version'] == 1
        and run['parameterized_plan'] == refs['plan'] and run['source'] == scope['source']
        and run['normalization'] == scope['normalization'] and run['argv'] == []
        and type(run['cpu_ids']) is list and len(run['cpu_ids']) == 1
        and type(run['cpu_ids'][0]) is int and run['cpu_ids'][0] >= 0
        and type(run['cwd']) is str and Path(run['cwd']).is_absolute() and type(run['environment']) is dict
        and all(type(k) is type(v) is str for k,v in run['environment'].items()),
        'fresh benchmark must bind the admitted typed controller and shared CPU launch')
    read_artifact(run['launcher'],json_document=False)
    original, native = (run['arms'][mode] for mode in _ARMS)
    _require(set(run['arms']) == set(_ARMS) and set(original) == {'python_executable','python_version'}
        and set(native) == {'artifact','executable'} and type(original['python_executable']) is str
        and type(original['python_version']) is str and original['python_version'].strip(),
        'comparison requires direct original and native executable arms')
    python = Path(original['python_executable'])
    _require(python.is_absolute() and python.is_file() and python.resolve(strict=True) == python,
        'original launcher must keep the measured interpreter identity')
    _native(native,plan,refs['plan'])
    effects, depths = _tape(read_artifact(run['stdin'],json_document=False),plan)
    capture = read_artifact(run['original_capture'])
    original_capture = read_artifact(capture['raw_capture'])
    captured_launch = json.loads(read_artifact(capture['captured_launch'],json_document=False))
    ui = read_artifact(scope['ui_contract'])
    _require(capture['source'] == scope['source'] and capture['stdin'] == run['stdin']
        and capture['original_execution'] is True and capture['performance_measurement'] is False
        and capture['launch']['entry_path'] == scope['source']['uri'] and capture['launch']['exit_code'] == 0
        and all(capture[key] == value for key,value in original_capture.items())
        and captured_launch['stdin'] == run['stdin']['uri']
        and source_cost._argument(captured_launch['command'],'--out') == capture['raw_capture']['uri']
        and len(capture['sessions']) == len(depths)
        and all(row['consumer_scalars'][ui['request_port']] == row['consumer_scalars'][ui['limit_port']] == depth
                for row,depth in zip(capture['sessions'],depths,strict=True)),
        'held-out source oracle did not observe the same complete fixed-depth stream')
    samples,orders,digest,observation = _paired_rows(run,effects,len(depths))
    _require(digest == _stdout(read_artifact(capture['stdout'],json_document=False),effects),
        'held-out original observer and clean source execution differ in visible output')
    measurements, comparisons = source_cost._envelopes(samples,minimum_gain_bps=0)
    gain = comparisons['entry_native']
    return dict(schema=COMPARISON,schema_version=1,artifacts=refs,scope='MATCHED_DIRECT_HELD_OUT_PARAMETERIZED_PROCESS',
        matched_launch={key:run[key] for key in ('source','stdin','normalization','cwd','argv','environment','cpu_ids')},
        launcher=run['launcher'],original_capture=run['original_capture'],original_interpreter=original,executable=native,
        depths=depths,clean_pairs=len(samples['original']),warmup_pairs=1,orders=orders,exact_canonical_stdout_sha256=digest,
        observation=observation,measurements=measurements,original_minus_native_evidence=gain,
        selected='entry_native' if gain['verified_observed_gain'] else 'original',
        deployment_authorized=False,historical_cost_used_for_selection=False)


def check(path):
    schema = json.loads(Path(path).read_bytes())['schema']
    if schema == SCHEMA:
        return _load_plan(path)
    _require(schema == COMPARISON,'unsupported parameterized command plan or selection')
    recorded = read_artifact(_ref(path,COMPARISON))
    refs = recorded['artifacts']
    replay = compare(refs['plan']['uri'],refs['benchmark']['uri'])
    _require(_canonical(recorded) == _canonical(replay),'held-out native selection differs from cold replay')
    return replay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ('entry-plan','source-grammar','prior-selection','check','out','plan','benchmark'):
        parser.add_argument('--' + role,type=Path)
    args = parser.parse_args(argv)
    inputs = (args.entry_plan,args.source_grammar,args.prior_selection)
    comparison = (args.plan,args.benchmark)
    if args.check and not any((*inputs,*comparison,args.out)):
        result = check(args.check)
    elif args.check is None and args.entry_plan and args.out and not any(comparison):
        result = propose(args.entry_plan,args.source_grammar,args.prior_selection)
    elif args.check is None and all(comparison) and args.out and not any(inputs):
        result = compare(*comparison)
    else:
        parser.error('supply --check, plan inputs, or --plan and --benchmark with --out')
    if args.out:
        with args.out.open('x') as stream:
            stream.write(json.dumps(result,indent=2,sort_keys=True,allow_nan=False) + '\n')
    summary = result['decision'] if result['schema'] == SCHEMA else {
        key:result[key] for key in ('selected','measurements','exact_canonical_stdout_sha256')}
    print(_canonical(summary))


if __name__ == '__main__':
    main()

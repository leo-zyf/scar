"""Join a retained original command stream to typed F4 source endpoints."""

import ast
from hashlib import sha256
import json
from pathlib import Path

from scar.analysis.build import BuildSession
from scar.analysis.build_admission import verify_build_admission
from scar.analysis.entry_obligations import verify_entry_obligations
from scar.analysis.scalar_recursive_driver import CAPTURE, _replay
from scar.analysis.scalar_score_reuse_evidence import _compile, _event, _span
from scar.analysis.source_references import ReferenceSlicer, source_port
from scar.analysis.static.query import _Fragments
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program.source_syntax import SourceSyntaxMaterial
from scar.ir.program.target_hardware import HardwareThread
from scar.ir.record_codec import decode, encode
from scar.trace.code_snapshot import describe_code, site

SCHEMA = 'scar.f4-cpu-stream-capture'
_PROVIDER = 'scar.cpu-recursive-stream-source-provider'
_OBSERVER = 'scar.runtime.scalar_recursive_driver_observe'


def _need(ok, detail):
    if not ok:
        raise ValueError('original CPU stream: ' + detail)


def _retained(capture_ref, scope):
    capture = read_artifact(capture_ref)
    raw = read_artifact(capture['raw_capture'])
    launch = capture['launch']
    source = capture['source']
    _need(capture_ref == scope['cpu_stream_capture'] and capture_ref['schema'] == CAPTURE
          and capture['raw_capture']['schema'] == CAPTURE and raw['schema_version'] == 1
          and set(capture) == set(raw) | {'raw_capture', 'captured_launch', 'stdin', 'stdout'}
          and all(capture[key] == value for key, value in raw.items())
          and capture['original_execution'] is True and capture['performance_measurement'] is False
          and launch['exit_code'] == 0 and launch['entry_path'] == source['uri']
          and launch['installed_bindings'] == 0
          and launch['argv'] == scope['argv'] and not capture['frontiers'],
          'bound capture differs from the successful original observer')
    design = read_artifact(capture['design'])
    _need(design['source'] == source and design['capture'] is None,
          'observer design describes another original source')
    _replay(capture['sessions'], design)
    context = json.loads(read_artifact(capture['captured_launch'], json_document=False))
    command = context['command']
    index = command.index(_OBSERVER)
    executable = Path(command[index - 4])
    tail = command[index + 1:]
    options = {'--design': capture['design']['uri'], '--out': capture['raw_capture']['uri']}
    _need(index >= 4 and command[index - 3:index] == ['-B', '-u', '-m'] and executable.is_absolute()
          and all(tail.count(key) == 1 and tail.index(key) + 1 < len(tail)
                  and tail[tail.index(key) + 1] == value for key, value in options.items())
          and context['stdin'] == capture['stdin']['uri'],
          'retained observer launcher does not name the exact design, stdin and capture')
    cpu = None
    if command[:2] == ['/usr/bin/taskset', '-c']:
        cpu_text = command[2]
        _need((index == 7 or (index == 11 and command[3:7] == ['/usr/bin/timeout', '-k', '2', '120']))
              and cpu_text.isascii() and cpu_text.isdecimal() and len(cpu_text) <= 10
              and cpu_text == str(int(cpu_text)), 'unsupported captured taskset placement')
        cpu = int(cpu_text)
        _need(scope['process_affinity_cpu_ids'] == [cpu] == context.get('cpu_ids', [cpu]),
              'capture CPU does not equal the original process cohort')
    context_digest = sha256(json.dumps(sorted(context['environment'].items()), separators=(',', ':')).encode()).hexdigest()
    _need(context['cwd'] == scope['working_directory'] and context_digest == scope['environment_sha256']
          and executable.resolve(strict=True) == Path(scope['python_executable']).resolve(strict=True),
          'retained command environment, working directory or interpreter differs')
    return capture, context, executable, context_digest, cpu


def _anchor(slicer, session, refs, source, code, instruction, role, slot):
    _need(site(code, instruction['offset']) == instruction, 'stream instruction differs from original code')
    row = dict(role=role, source=source, code=describe_code(code), instruction=instruction, authority=slot)
    key = source['uri']
    if key not in refs:
        return dict(row, status='SOURCE_CODE_ONLY_F4_UNIT_NOT_ADMITTED')
    matches = refs[key].get(tuple(instruction['position']), ())
    if len(matches) != 1:
        return dict(row, status='ORIGINAL_F4_CALL_ENDPOINT_OPEN')
    facts, index, port = matches[0]
    point = session.view.source_port(port)
    return dict(row, status='BOUND_ORIGINAL_F4_CALL', operation=encode(session.view.get(point).owner),
                original_call=encode(slicer.fact(facts, 'calls', index)))


def _f4_calls(catalog, sources):
    lookup = {}
    store = catalog.fact_store
    for unit, declaration in sorted(store.rows.items()):
        source = sources.get(declaration['path'])
        if source is None or store.original_source_units.get(unit, unit) != unit:
            continue
        model = store.model_units[unit]
        _need(model['raw_sha256'] == source['sha256'], 'original F4 unit differs from stream source bytes')
        syntax = decode(SourceSyntaxMaterial, _Fragments(store.model_for(unit)).read(model['artifacts']['source_syntax']))
        _need((syntax.input.path, syntax.input.raw_sha256) == (source['uri'], source['sha256']),
              'original F2 syntax differs from stream source bytes')
        attrs = {item.name: item.attributes for item in syntax.profile.ast_schema}
        facts = store.load(unit)
        by_position = lookup.setdefault(source['uri'], {})
        for index, call in enumerate(facts.calls):
            port = source_port(facts, call.result)
            node = syntax.nodes[port.port.template.occurrence]
            fields = dict(zip(attrs[node.constructor], node.attributes, strict=True))
            position = tuple(fields[name].scalar.to_python() for name in
                             ('lineno', 'end_lineno', 'col_offset', 'end_col_offset'))
            by_position.setdefault(position, []).append((facts, index, port))
    return lookup


def _call_roles(plan, plan_ref, source_entry, source_entry_ref, ui_source):
    entry = source_entry['entry']
    _, entry_codes = _compile(entry['source'])
    tree, ui_codes = _compile(ui_source)
    controller = plan['ports']['controller']
    consumer = plan['ports']['output']['consumer']
    anchor_code = entry_codes[entry['body']['qualname']]
    controller_code, consumer_code = (ui_codes[row['qualname']] for row in (controller, consumer))
    _need([describe_code(code) for code in (anchor_code, controller_code, consumer_code)]
          == [entry['body'], controller, consumer], 'checked controller source CodeID differs')
    roles = [(entry['source'], anchor_code, entry['call'], 'ENTRY_TO_CONTROLLER', dict(ref=source_entry_ref, slot='#/entry/call'))]
    token_store = plan['ports']['input']['store']
    _need(site(controller_code, token_store['offset']) == token_store, 'command token assignment differs')
    assignment = [row for row in ast.walk(tree) if type(row) is ast.Assign and
                  any(_span(target) == token_store['position'] for target in row.targets)]
    read_calls = [row for stmt in assignment for row in ast.walk(stmt.value)
                  if type(row) is ast.Call and type(row.func) is ast.Name and row.func.id == 'input'
                  and not row.args and not row.keywords]
    if len(read_calls) == 1:
        roles.append((ui_source, controller_code, _event(controller_code, read_calls[0], 'CALL'), 'STDIN_READ_CALL',
                      dict(ref=plan_ref, slot='#/ports/input/store', relation='CALL_FEEDS_PINNED_TOKEN_ASSIGNMENT')))
    for index, reply in enumerate(plan['source_effects']['historical_startup_replies']):
        roles.append((ui_source, controller_code, reply['sites'][-1], f'STATIC_ACK_{index}', dict(
            ref=plan_ref, slot=f'#/source_effects/historical_startup_replies/{index}/sites/{len(reply["sites"])-1}')))
    terminal_positions = {tuple(_span(row)) for row in ast.walk(tree) if type(row) is ast.Call
                          and type(row.func) is ast.Name and row.func.id == 'print' and row.args
                          and type(row.args[0]) is ast.Constant and row.args[0].value == 'bestmove'}
    for index, instruction in enumerate(plan['ports']['output']['search_print_sites']):
        if tuple(instruction['position']) in terminal_positions:
            roles.append((ui_source, consumer_code, instruction, 'SEARCH_ACK_CALL_ALTERNATIVE', dict(
                ref=plan_ref, slot=f'#/ports/output/search_print_sites/{index}')))
    return roles


def _ranges(raw):
    offset = 0
    for line in raw.splitlines(keepends=True):
        digest = sha256(line).hexdigest()
        yield line, dict(offset_bytes=offset, size_bytes=len(line), sha256=digest)
        offset += len(line)


def _observables(tape, stdout, capture, plan, ui, context, paired_ref):
    from scar.compiler.cpu_entry_replay import _paced_transcript
    from scar.optimization.scalar_recursive_parameterized_plan import _stdout

    transcript, effects, depths = _paced_transcript(tape, plan)
    _need(len(depths) == len(capture['sessions']), 'command stream differs from captured original search count')
    canonical = _stdout(stdout, effects)
    commands, edges, acknowledgements, pacing, searches = [], [], [], [], []
    outputs = iter(_ranges(stdout))
    sessions = iter(enumerate(zip(capture['sessions'], depths, strict=True)))
    literal_heads = {tuple(row['tokens']): row['kind'] for row in plan['grammar']['commands']}
    variable_heads = {tuple(row['tokens']): row['kind'] for row in plan['grammar']['commands']
                      if row['arguments'] or row['optional_tail'] is not None}
    reply_roles = {'UCI': 'STATIC_ACK_0', 'ISREADY': 'STATIC_ACK_1', 'GO_DEPTH': 'SEARCH_ACK_CALL_ALTERNATIVE'}
    for index, (phase, (line, position)) in enumerate(zip(transcript, _ranges(tape), strict=True)):
        _need(line == phase['line'].encode('ascii'), 'protocol step differs from original input range')
        tokens = tuple(line.decode('ascii').split())
        kind = literal_heads.get(tokens)
        if kind is None:
            choices = [value for head, value in variable_heads.items() if tokens[:len(head)] == head]
            _need(len(choices) == 1, 'typed stream command selector is not unique')
            kind = choices[0]
        command = f'command:{index}'
        commands.append(dict(id=command, kind=kind, stdin=position, source_role='STDIN_READ_CALL'))
        if index:
            edges.append(dict(source=f'command:{index - 1}', target=command, kind='ADJACENT_STDIN_RANGES'))
        event = dict(input_offset=position['offset_bytes'], input_bytes=position['size_bytes'],
                     ack_stdout_offset=None, ack_stdout_bytes=None)
        if phase['acknowledgement'] is not None:
            marker = phase['acknowledgement'].encode('ascii')
            ack = f'acknowledgement:{index}'
            for actual, span in outputs:
                if actual.startswith(marker):
                    break
            else:
                raise ValueError('original CPU stream: expected acknowledgement bytes are missing')
            acknowledgements.append(dict(id=ack, command=command, stdout=span, source_role=reply_roles[kind],
                                         source_invocation_observed=False))
            event.update(ack_stdout_offset=span['offset_bytes'], ack_stdout_bytes=span['size_bytes'])
            edges.append(dict(source=command, target=ack, kind='TYPED_INPUT_TO_OBSERVED_OUTPUT'))
            if index + 1 < len(transcript):
                if context.get('stdin_mode') == 'PACED_PIPE':
                    edges.append(dict(source=ack, target=f'command:{index + 1}', kind='CAPTURE_ACK_BEFORE_NEXT_SEND',
                                      slot=f'#/pacing_events/{index}'))
                if paired_ref is not None:
                    edges.append(dict(source=ack, target=f'command:{index + 1}', kind='PAIRED_HARNESS_ACK_BEFORE_NEXT_SEND',
                                      slot=f'#/invocation/transcript/{index}/acknowledgement'))
        if kind == 'GO_DEPTH':
            ordinal, (session, depth) = next(sessions)
            chain = session['caller_chain']
            output = plan['ports']['output']
            _need([row['code'] for row in chain] == [output['relay'], output['consumer']]
                  and session['entry']['code'] == plan['ports']['driver']['code']['key']
                  and all(type(session['consumer_scalars'][role]) is int
                          and session['consumer_scalars'][role] == depth
                          for role in (ui['request_port'], ui['limit_port'])),
                  'captured search invocation differs from original relay, consumer or requested depth')
            search = f'original-search:{ordinal}'
            searches.append(dict(id=search, capture_slot=f'#/sessions/{ordinal}', requested_depth=depth,
                entry=session['entry'], caller_chain=chain))
            edges.append(dict(source=command, target=search, kind='TYPED_COMMAND_AND_CAPTURED_SEARCH_ORDINAL'))
            edges.append(dict(source=search, target=ack, kind='CAPTURED_SEARCH_AND_OUTPUT_ORDINAL'))
        pacing.append(event)
    if context.get('stdin_mode') == 'PACED_PIPE':
        _need(context['pacing_events'] == pacing and context['normalized_stdout_sha256'] == canonical,
              'paced launcher events differ from pinned command and output byte ranges')
    return dict(commands=commands, acknowledgements=acknowledgements, edges=edges,
                search_invocations=searches, normalized_stdout_sha256=canonical)


def bind_cpu_stream_capture_graph(program_ref, scope, capture_ref, *, source_admission,
                                  entry_obligations=None, provider=None):
    """Cold-join original source stream bytes. Admission never executes the target."""
    return _bind_cpu_stream_capture_graph(program_ref, scope, capture_ref, source_admission=source_admission,
        entry_obligations=entry_obligations, provider=provider)


def _bind_cpu_stream_capture_graph(program_ref, scope, capture_ref, *, source_admission,
                                   entry_obligations, provider, verified_paired=None):
    program = read_artifact(program_ref)
    capture, context, executable, environment, cpu = _retained(capture_ref, scope)
    source = capture['source']
    target = program['target']
    target_ref = dict(uri=target['path'], sha256=target['sha256'], size_bytes=target['bytes'],
                      schema='scar.target-hardware-graph')
    _need((scope['entry'], scope['entry_sha256'], scope['boundary'], scope['program_sha256'])
          == (source['uri'], source['sha256'], 'PROCESS_LAUNCH_THROUGH_EXIT', program_ref['sha256'])
          and program['entry'] == source['uri'] and (scope['target'], scope['target_sha256']) == (target_ref, target['sha256'])
          and scope['source_admission_sha256'] == source_admission['sha256'],
          'stream capture differs from original entry or requested target')
    admitted = verify_build_admission(source_admission, program=program_ref, hardware=target,
        source_filter=read_artifact(source_admission)['source']['filter_path'])
    origin = admitted['source']['entry']['source']
    _need(all(origin[key] == source[key] for key in ('uri', 'sha256', 'size_bytes')),
          'stream source differs from admitted original bytes')
    tape, stdout = (read_artifact(capture[key], json_document=False) for key in ('stdin', 'stdout'))
    paired_cpus, paired_ref = (), None
    if verified_paired is not None:
        paired_ref, paired = verified_paired
        _need(paired['capture'] == capture_ref and paired['invocation']['entry'] == source
              and paired['invocation']['scope'] == 'PACED_SOURCE_COMMAND_STREAM'
              and b''.join(row['line'].encode('ascii') for row in paired['invocation']['transcript']) == tape,
              'verified replay covers a different paced command stream')
        paired_cpus = paired['invocation']['cpu_ids']
    frontiers = [dict(kind='UNOBSERVED_ORIGINAL_EFFECTS_OPEN')]
    if cpu is None:
        frontiers.append(dict(kind='CAPTURE_CPU_AFFINITY_UNRECORDED'))
    if paired_ref is None and context.get('stdin_mode') != 'PACED_PIPE':
        frontiers.append(dict(kind='CAPTURE_HAS_NO_OBSERVED_PACING'))
    refs = {source['uri']: source}
    anchors, roles, observable = [], [], dict(stdin_ranges=[span for _, span in _ranges(tape)])
    plan_ref = context.get('protocol_plan')
    if provider is not None:
        authority = read_artifact(provider)
        _need(provider['schema'] == _PROVIDER and authority['entry'] == source
              and authority['argv'] == scope['argv'] and plan_ref in (None, authority['parameterized_plan']),
              'typed stream provider belongs to another source, argv or protocol')
        plan_ref = authority['parameterized_plan']
    if context.get('stdin_mode') == 'PACED_PIPE':
        _need(context.get('protocol_plan') == plan_ref and plan_ref is not None, 'paced capture has no pinned protocol')
    if plan_ref is None:
        frontiers.append(dict(kind='TYPED_STREAM_PROTOCOL_NOT_BOUND'))
    else:
        from scar.optimization import scalar_recursive_parameterized_plan as planning

        plan = read_artifact(plan_ref)
        _need(plan_ref['schema'] == planning.SCHEMA and planning.check(plan_ref['uri']) == plan
              and plan['scope']['source'] == source
              and plan['scope']['mode'] == planning.MODE and plan['requirements']['paced_only'] is True,
              'parameterized command plan differs from original typed authority')
        anchor = read_artifact(plan['scope']['entry_plan'])
        source_entry_ref = anchor['artifacts']['source_entry']
        source_entry = read_artifact(source_entry_ref)
        ui = read_artifact(plan['scope']['ui_contract'])
        ui_source = ui['ui_source']
        refs[ui_source['uri']] = ui_source
        roles = _call_roles(plan, plan_ref, source_entry, source_entry_ref, ui_source)
        observable = _observables(tape, stdout, capture, plan, ui, context, paired_ref)
        frontiers.extend(dict(kind='ORIGINAL_STREAM_DOMAIN_OPEN', detail=value) for value in plan['proof_frontiers']['unbound'])
    with BuildSession(Path(program_ref['uri']).parent) as session:
        projection = None
        if entry_obligations is not None:
            _need(entry_obligations['sha256'] == scope['entry_obligations_sha256'],
                  'stream capture must bind the original entry obligation graph')
            projection = verify_entry_obligations(entry_obligations, program=program_ref, session=session)
        if roles:
            catalog = session.view.catalog
            calls = _f4_calls(catalog, refs)
            slicer = ReferenceSlicer(catalog)
            anchors = [_anchor(slicer, session, calls, *row) for row in roles]
        if projection is not None:
            indices = {row['operation']: index for index, row in enumerate(projection['operations'])}
            for row in anchors:
                operation = row.get('operation')
                if operation is not None and operation['node']['wire'] in indices:
                    row['entry_operation'] = '#/operations/' + str(indices[operation['node']['wire']])
        threads = {row.cpu_id: row.id.wire for row in session.resources.target_resources if type(row) is HardwareThread}
    for row in anchors:
        if row['status'] != 'BOUND_ORIGINAL_F4_CALL':
            frontiers.append(dict(kind=row['status'], role=row['role'], source=row['source']))
    if plan_ref is not None:
        declared_roles = {row['source_role'] for row in
                          (*observable['commands'], *observable['acknowledgements'])}
        for role in sorted(declared_roles - {row['role'] for row in anchors}):
            frontiers.append(dict(kind='ORIGINAL_STREAM_ROLE_SOURCE_SITE_OPEN', role=role))
    for recorded_cpu in sorted(set(paired_cpus) | ({cpu} if cpu is not None else set())):
        if recorded_cpu not in threads:
            frontiers.append(dict(kind='ORIGINAL_STREAM_TARGET_CPU_ENDPOINT_OPEN', cpu_id=recorded_cpu))
    return dict(schema=SCHEMA, schema_version=1, program=program_ref, source_admission=source_admission,
        entry_obligations=entry_obligations, capture=capture_ref, provider=provider, paired_replay=paired_ref,
        protocol_plan=plan_ref,
        source=source, target=target_ref,
        original_observer=dict(launcher=capture['captured_launch'], argv=scope['argv'], cwd=context['cwd'],
            interpreter_at_capture_path=artifact_reference(executable, 'application/x-executable'),
            environment_sha256=environment, taskset_cpu=cpu, exit_status=0,
            stdin_mode=context.get('stdin_mode', 'INJECTION_MODE_UNRECORDED'),
            instrumentation_scope=capture['launch']['boundary_observation_scope']),
        requested_cpu_scope=[dict(cpu_id=cpu, target_thread=threads.get(cpu)) for cpu in scope['process_affinity_cpu_ids']],
        verified_paired_cpu_scope=[dict(cpu_id=cpu, target_thread=threads.get(cpu)) for cpu in paired_cpus],
        stdin=capture['stdin'], stdout=capture['stdout'], observable_graph=observable, source_anchors=anchors,
        coverage_frontier=frontiers, whole_program_effects_closed=False, deployment_authorized=False)

"""Bind a verified CPU provider's observed boundary to the original F4 entry."""

from hashlib import sha256
import json
from pathlib import Path
from types import CodeType

from scar.analysis.build import BuildSession
from scar.analysis.build_admission import verify_build_admission
from scar.analysis.cpu_stream_graph import _bind_cpu_stream_capture_graph
from scar.analysis.cpu_stream_graph import bind_cpu_stream_capture_graph as bind_cpu_stream_capture_graph
from scar.analysis.entry_obligations import verify_entry_obligations
from scar.analysis.source_references import ReferenceSlicer, source_port
from scar.analysis.source_role_materials import SyntaxRoleView
from scar.analysis.static_query import _Fragments
from scar.compiler.cpu_entry_replay import CheckedCpuPair, SCHEMA as PAIRED, check as check_paired
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program.normal_graph import FactTable
from scar.ir.program.source_syntax import SourceSyntaxMaterial
from scar.ir.program.target_hardware import HardwareThread
from scar.ir.record_codec import decode, encode
from scar.trace.code_snapshot import code_key


SCHEMA = 'scar.f4-cpu-provider-boundary'
_PROCESS = 'PROCESS_LAUNCH_THROUGH_EXIT'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _only(rows, message):
    choices = tuple(rows)
    _need(len(choices) == 1, message)
    return choices[0]


def _provider_boundary(authority_ref, authority, compiled):
    """Read only documented authority links, without searching for field names."""
    family = authority_ref['schema']
    model = authority.get('model')
    target, effects, entry = None, [], None
    if family == 'scar.cpu-fixed-grid-source-provider':
        from scar.analysis.fixed_grid_entry_abi import derive_fixed_grid_file_entry

        project = authority['source_project']
        manifest = read_artifact(project)

        def member(name):
            raw = (Path(project['uri']).parent/name).read_bytes()
            _need(manifest['files'][name] == dict(sha256=sha256(raw).hexdigest(), size_bytes=len(raw)),
                  'fixed-grid source project member differs from the checked provider')
            return json.loads(raw)

        target = member('plan/execution.json')['target']
        entry = member('plan/source-entry.json')['entry']
        abi = derive_fixed_grid_file_entry(member('plan/source-entry-binding.json'), entry)
        for index, node in enumerate(abi['graph']['nodes']):
            kind = {'FILE_PUBLICATION': 'FILE', 'STDOUT_PUBLICATION': 'STDOUT'}.get(node['kind'])
            if kind is not None:
                first, left, last, right = node['source_span']
                effects.append(dict(kind=kind, position=[first, last, left, right], declaration=project,
                    slot=f'DERIVED_FIXED_GRID_ABI#/graph/nodes/{index}', expected_code=None))
    elif family == 'scar.cpu-array-task-source-provider':
        retention = read_artifact(compiled['retention'])
        target = read_artifact(retention['schedule'])['hardware']
        recipe = read_artifact(compiled['recipe'])
        contract_ref = recipe['source_contract']
        contract = read_artifact(contract_ref)
        _need(contract['entry'] == compiled['entry'], 'array source contract differs from the admitted entry')
        entry = contract['entry_function']['name']
        for field, kind in (('path_output', 'FILE'), ('stdout', 'STDOUT')):
            material = contract['report'].get(field)
            if material is not None:
                source = material['source']
                effects.append(dict(kind=kind, position=source['position'], declaration=contract_ref,
                    slot=f'#/report/{field}/source', expected_code=source['code_id']))
    elif family == 'scar.triangle-profile-family':
        target = compiled['hardware']
    elif family != 'scar.cpu-recursive-stream-source-provider':
        raise ValueError('unsupported checked CPU provider boundary')
    if target is not None:
        read_artifact(target)
    if model is not None:
        read_artifact(model)
    return model, target, entry, effects


def _source_anchors(session, source, entry, effects, projection):
    if not effects:
        return []
    catalog = session.view.catalog
    unit = session.report['f4']['entry_unit']
    facts = catalog.fact_store.load(unit)
    model = catalog.fact_store.model_units[unit]
    syntax = decode(SourceSyntaxMaterial, _Fragments(catalog.fact_store.model_for(unit)).read(
        model['artifacts']['source_syntax']))
    _need((syntax.input.path, syntax.input.raw_sha256) == (source['uri'], source['sha256']),
          'provider observable anchors differ from original F2 source bytes')
    roles = SyntaxRoleView(syntax)
    attributes = {row.name: row.attributes for row in syntax.profile.ast_schema}
    function = _only((row for row in facts.functions
        if syntax.nodes[row.template.occurrence].constructor in ('FunctionDef', 'AsyncFunctionDef')
        and roles.field(row.template.occurrence, 'name').scalar.to_python() == entry),
        'provider observable has no unique original F4 function')
    compiled = compile(read_artifact(source, json_document=False), source['uri'], 'exec', dont_inherit=True)
    code = _only((row for row in compiled.co_consts if type(row) is CodeType and row.co_qualname == entry),
                 'provider observable has no unique original CodeID')
    identity = code_key(code)
    by_position = {}
    for group in facts.scope_groups:
        if group.owner != function.body_scope.owner:
            continue
        for span in group.tables:
            if span.table is not FactTable.CALLS:
                continue
            for index in range(span.start, span.stop):
                port = source_port(facts, facts.calls[index].result)
                occurrence = syntax.nodes[port.port.template.occurrence]
                values = dict(zip(attributes[occurrence.constructor], occurrence.attributes, strict=True))
                position = tuple(values[key].scalar.to_python() for key in
                    ('lineno', 'end_lineno', 'col_offset', 'end_col_offset'))
                by_position.setdefault(position, []).append((index, port))
    operation_indices = {} if projection is None else {row['operation']: i for i, row in enumerate(projection['operations'])}
    slicer = ReferenceSlicer(catalog)
    bound = []
    for effect in effects:
        _need(effect['expected_code'] in (None, identity), 'provider publication proof is from a different original CodeID')
        index, port = _only(by_position.get(tuple(effect['position']), ()),
                            'provider publication does not match one original F4 call')
        point = session.view.source_port(port)
        operation = session.view.get(point).owner
        row = dict(effect, code_id=identity, operation=encode(operation),
                   original_call=encode(slicer.fact(facts, 'calls', index)))
        if operation.node.wire in operation_indices:
            row['entry_operation'] = '#/operations/'+str(operation_indices[operation.node.wire])
        bound.append(row)
    return bound


def _observable_contract(spec, anchors):
    original, candidate = (spec['arms'][role] for role in ('original', 'candidate'))
    report = spec.get('json_report')
    paced = bool(spec['transcript'])
    return dict(anchors=anchors, exit_status=0, stderr='EMPTY',
        stdin_kind='ADMITTED_PACED_TRANSCRIPT' if paced else 'DEVNULL',
        transcript=spec['transcript'] if paced else [],
        stdout=dict(original=original['stdout'], candidate=candidate['stdout'], rule=(
            'ADMITTED_PACED_REPLY_NORMALIZATION' if paced else 'NAMED_JSON_RUNTIME_FIELD_NORMALIZATION' if report
            else 'ARM_SPECIFIC_ADMITTED_BYTE_STREAMS')),
        publications=[dict(name=name, expected=ref, rule='NORMALIZED_JSON' if report else 'EXACT_BYTES')
                      for name, ref in sorted(spec['files'].items())],
        json_report=report, publication_path_scope='DECLARED_OUTPUT_SLOT_RELOCATED_IN_EACH_PAIR' if spec['files']
            else 'NO_DECLARED_FILE_OUTPUT', observed_effects_close_original_world=False)


def bind_cpu_provider_graph(program_ref, scope, replay_ref, *, source_admission, entry_obligations=None,
                            checked_pair=None):
    """Read back the same F4 entry, sealed provider and paired observable scope."""
    replay = read_artifact(replay_ref)
    program = read_artifact(program_ref)
    compilation = read_artifact(replay['compilation'])
    source = compilation['entry']
    target_row = program['target']
    target = dict(uri=target_row['path'], sha256=target_row['sha256'], size_bytes=target_row['bytes'],
                  schema='scar.target-hardware-graph')
    spec = replay['invocation']
    original = spec['arms']['original']
    environment = sha256(json.dumps(sorted(original['environment'].items()), separators=(',', ':')).encode()).hexdigest()
    _need(replay_ref['schema'] == PAIRED and program['entry'] == source['uri'] == scope['entry']
          and (scope['entry_sha256'], scope['argv'], scope['boundary'], scope['program_sha256'])
              == (source['sha256'], compilation['argv'], _PROCESS, program_ref['sha256'])
          and (scope['target'], scope['target_sha256'], scope['source_admission_sha256'])
              == (target, target['sha256'], source_admission['sha256'])
          and scope['working_directory'] == spec['cwd'] and scope['environment_sha256'] == environment
          and ((scope['python_executable'] == original['executable']['uri'])
               or Path(original['executable']['uri']).resolve() == Path(scope['python_executable']))
          and set(spec['cpu_ids']) <= set(scope['process_affinity_cpu_ids']),
          'CPU paired observation differs from original source, argv, target or process scope')
    read_artifact(source, json_document=False)
    _need(checked_pair is None or type(checked_pair) is CheckedCpuPair, 'expected a checked CPU pair')
    checked = check_paired(replay_ref['uri']) if checked_pair is None else checked_pair.report(replay_ref)
    _need(checked == replay, 'paired observable result differs from current provider replay')
    provider = compilation['provider']
    authority = read_artifact(provider)
    compiled = read_artifact(compilation['compilation'])
    model, provider_target, entry, declared_effects = _provider_boundary(provider, authority, compiled)

    saved_admission = read_artifact(source_admission)
    admitted = verify_build_admission(source_admission, program=program_ref, hardware=target_row,
        source_filter=saved_admission['source']['filter_path'])
    origin = admitted['source']['entry']
    _need(all(origin['source'][key] == source[key] for key in ('uri', 'sha256', 'size_bytes')),
          'CPU provider source differs from the original F4 byte receipt')
    with BuildSession(Path(program_ref['uri']).parent) as session:
        if entry_obligations is not None:
            _need(entry_obligations['sha256'] == scope['entry_obligations_sha256'],
                  'CPU observation does not retain its original entry obligation graph')
        projection = None if entry_obligations is None else verify_entry_obligations(
            entry_obligations, program=program_ref, session=session)
        anchors = _source_anchors(session, source, entry, declared_effects, projection)
        threads = {row.cpu_id: row.id.wire for row in session.resources.target_resources
                   if type(row) is HardwareThread}
    original_scope = [dict(cpu_id=cpu, target_thread=threads.get(cpu)) for cpu in spec['cpu_ids']]
    frontier = [dict(kind='ORIGINAL_WORLD_EFFECTS_OPEN')]
    if model != program_ref:
        frontier.append(dict(kind='PROVIDER_ORIGINAL_F4_REFERENCE_ABSENT' if model is None
                             else 'PROVIDER_ORIGINAL_F4_REFERENCE_DIFFERS', provider_model=model))
    if provider_target != target:
        frontier.append(dict(kind='PROVIDER_EXACT_TARGET_ABSENT' if provider_target is None
                             else 'PROVIDER_TARGET_REPRESENTATION_DIFFERS', provider_target=provider_target))
    if any(row['target_thread'] is None for row in original_scope):
        frontier.append(dict(kind='ORIGINAL_PLATFORM_THREAD_ENDPOINT_OPEN'))
    if not anchors:
        frontier.append(dict(kind='ORIGINAL_PUBLICATION_CALL_BINDING_OPEN'))
    stream = None
    if stream_ref := scope.get('cpu_stream_capture'):
        _need(spec['transcript'] and stream_ref == replay['capture'],
              'explicit stream capture differs from paired original observations')
        stream = _bind_cpu_stream_capture_graph(program_ref, scope, stream_ref, source_admission=source_admission,
            entry_obligations=entry_obligations, provider=provider, verified_paired=(replay_ref, replay))
    elif spec['transcript']:
        frontier.append(dict(kind='PACED_STDIN_TRANSCRIPT_REMAINS_AN_EXTRA_REQUEST_CONSTRAINT'))
    if spec['files']:
        frontier.append(dict(kind='ORIGINAL_ARBITRARY_OUTPUT_PATH_EFFECTS_OPEN'))
    if scope['python_executable'] != original['executable']['uri']:
        frontier.append(dict(kind='ORIGINAL_INTERPRETER_LAUNCH_PATH_SPELLING_DIFFERS'))
    read_artifact(replay_ref, json_document=False)
    result = dict(schema=SCHEMA, schema_version=1, program=program_ref, source_admission=source_admission,
        entry_obligations=entry_obligations, entry=origin, target=target, replay=replay_ref, provider=provider,
        compilation=replay['compilation'], argv=compilation['argv'], source_launch=original['argv'],
        working_directory=spec['cwd'], original_environment_sha256=environment, observed_launch_cpu_ids=original_scope,
        observable_contract=_observable_contract(spec, anchors), coverage_frontier=frontier,
        status='ORIGINAL_SOURCE_AND_SCOPED_PROVIDER_OBSERVABLES_BOUND',
        native_argv_proven_for_unobserved_paths=False, whole_program_effects_closed=False, deployment_authorized=False)
    if stream is not None:
        result['stream_capture'] = stream
    return result

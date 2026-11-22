"""Locate typed instrumentation boundaries in one retained native CPU entry."""

from hashlib import sha256
from pathlib import Path

from scar.analysis.cpu_entry_lifecycle_graph import verify_cpu_entry_lifecycle_graph
from scar.analysis.native_call_materials import read_static_cpp_source
from scar.codegen import native_array_entry, native_seeded_array_task_entry as generator
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.record_codec import encode
from scar.optimization.cpu_entry_capacity_contract import ADMISSIBLE

SCHEMA = 'scar.cpu-native-phase-sites'
_MAIN = 'src/main.cpp'


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU native phase sites: ' + detail)


def _one(rows, detail):
    rows = tuple(rows)
    _need(len(rows) == 1, detail)
    return rows[0]


def _fact(artifact, locator):
    return dict(artifact=artifact, locator=locator)


def _retained_runtime(capacity):
    evidence = capacity['evidence']
    reference = evidence['runtime']
    runtime = read_artifact(reference)
    binary = runtime['files'][runtime['binary']]
    source = runtime['files'][_MAIN]
    project = Path(reference['uri']).parent
    _need(reference['schema'] == generator.SCHEMA and runtime['schema_version'] == 1
          and type(runtime['cxx_standard']) is int and runtime['cxx_standard'] == 20
          and runtime['schedule'] == evidence['schedule']
          and runtime['thread_contract'] == evidence['thread_contract']
          and runtime['python_runtime_required'] is False and runtime['deployment_authorized'] is False
          and binary == capacity['native_launch']['executable']
          and Path(source['uri']) == project / _MAIN
          and Path(binary['uri']) == project / runtime['binary'],
          'native C++20 source is not a member of the scoped candidate runtime')
    _need(read_artifact(binary, json_document=False).startswith(b'\x7fELF'), 'retained native member is not ELF')
    exact = read_artifact(source, json_document=False)
    recipe, prior, family, thread = (read_artifact(runtime[key])
                                    for key in ('recipe', 'native', 'family', 'thread_contract'))
    _, rendered = generator._render(recipe, prior, family, runtime['decision'], runtime['scope'], thread)
    header, indices = native_array_entry._argv_header(recipe['options'], recipe['allow_abbrev'])
    header_ref = runtime['files']['include/scar_generated/array_entry_argv.hpp']
    _need(exact == rendered[_MAIN] and header == read_artifact(header_ref, json_document=False),
          'retained main body or public ingress header differs from mechanical provider rendering')
    contract_ref = recipe['source_contract']
    contract = read_artifact(contract_ref)
    _need(contract['entry'] == recipe['entry'] and contract['report']['stdout']['kind'] == 'JSON_TEXT_PRINT'
          and (contract['report']['path_output'] is None
               or contract['report']['path_output']['kind'] == 'CONDITIONAL_PATH_WRITE_TEXT'),
          'typed original publication differs from generated provider ingress')
    return source, exact, binary, contract_ref, contract, indices[recipe['output_option']]


def _source_spans(raw, model, output_index):
    ix = model.index()
    function = _one((row for row in model.functions if row.symbol == 'main' and not row.namespaces
                    and not row.preprocessor), 'expected a unique unconditional C++ main')
    body = ix.occurrences[function.body]

    def text(row):
        return raw[row.byte_start:row.byte_end]

    def owned(row):
        return body.byte_start <= row.byte_start < row.byte_end <= body.byte_end

    _need(body.kind == 'compound_statement' and not body.has_parse_error and not model.holes
          and all(not owned(ix.occurrences[atom]) for atom in model.errors),
          'generated main contains open source syntax')

    def call(callee, arguments):
        point = _one((row for row in model.calls if row.owner_function == function.occurrence
            and not row.preprocessor and row.callee is not None
            and text(ix.occurrences[row.callee]) == callee
            and tuple(text(ix.occurrences[arg]) for arg in row.arguments) == arguments),
            'provider source call differs from its typed phase role')
        return ix.occurrences[point.occurrence]

    def syntax(kind, syntax_text, *, prefix=False):
        return _one((row for row in model.occurrences if row.kind == kind and owned(row)
                    and (text(row).startswith(syntax_text) if prefix else text(row) == syntax_text)),
                    'provider source statement differs from its typed phase role')

    def line(offset):
        start = raw.rfind(b'\n', body.byte_start, offset) + 1
        end = raw.index(b'\n', offset)
        return start, raw[start:end]

    def statement(anchor, expected):
        offset, raw_line = line(anchor.byte_start)
        indentation = raw_line[:len(raw_line) - len(raw_line.lstrip(b' '))]
        _need(bool(indentation) and raw_line == indentation + expected,
              'expected a unique complete generated statement line')
        return offset, indentation.decode('ascii'), anchor

    def earlier_statement(anchor, expected):
        following, _ = line(anchor.byte_start)
        offset = raw.rfind(b'\n', body.byte_start, following-1) + 1
        raw_line = raw[offset:following-1]
        indentation = raw_line[:len(raw_line) - len(raw_line.lstrip(b' '))]
        _need(bool(indentation) and raw_line == indentation + expected,
              'generated statement adjacent to its source occurrence differs')
        return offset, indentation.decode('ascii'), anchor

    parse = call(b'scar::public_argv::parse', (b'argc', b'argv'))
    hardware = call(b'hardware_scope', ())
    limit = call(b'enforce_coordinator_limit', ())
    dispatch = call(b'execute', (b'result',))
    result_check = syntax('if_statement', b'if (status != 0) {', prefix=True)
    file_guard = syntax('if_statement', b'if (!path.empty()) {', prefix=True)
    file_output = call(b'write_result', (b'file', b'result', b'true'))
    close = call(b'file.close', ())
    stdout = call(b'write_result', (b'std::cout', b'result', b'false'))
    return_ = syntax('return_statement', b'return std::cout ? 0 : 1;')
    _need(parse.byte_start < hardware.byte_start < limit.byte_start < dispatch.byte_start
          < result_check.byte_start < file_guard.byte_start < file_output.byte_start
          < close.byte_start < file_guard.byte_end <= stdout.byte_start < return_.byte_start,
          'success/publication ordering differs from the generated candidate entry')
    earlier_statement(return_, rb"std::cout << '\n';")
    path = b'const auto& path = values[' + str(output_index).encode('ascii') + b'].text;'
    return [
        ('MAIN_ENTER', 'boundary', *statement(parse, b'const auto values = scar::public_argv::parse(argc, argv);')),
        ('DISPATCH_BEGIN', 'work', *statement(dispatch, b'const auto status = execute(result);')),
        ('DISPATCH_RETURN', 'join', *statement(result_check, b'if (status != 0) {')),
        ('PUBLICATION_BEGIN', 'publication', *earlier_statement(file_guard, path)),
        ('PUBLICATION_END', 'publication', *statement(return_, b'return std::cout ? 0 : 1;')),
    ], return_


def derive_cpu_native_phase_sites(lifecycle_graph_ref):
    """Bind insertion locations and original roles for a scoped C++20 candidate."""
    graph = verify_cpu_entry_lifecycle_graph(lifecycle_graph_ref)
    capacity_ref = graph['capacity_contract']
    _need(capacity_ref is not None and graph['scope_sha256'] is not None, 'candidate lifecycle scope is unresolved')
    capacity = read_artifact(capacity_ref)
    _need(capacity['status'] == ADMISSIBLE and capacity['scope_sha256'] == graph['scope_sha256'],
          'exact scoped provider capacity contract required')
    lane_index, lane = _one(((index, lane) for index, lane in enumerate(graph['lanes'])
        if lane['role'] == 'CONDITIONAL_NATIVE_CANDIDATE' and lane['option_id'] == capacity['option_id']),
        'expected the exact candidate lifecycle lane')
    operations = {row['id']: index for index, row in enumerate(lane['operations'])}
    template = artifact_reference(generator._TEMPLATE, 'scar.native-cpp-source-template')
    source, raw, binary, original_ref, original, output_index = _retained_runtime(capacity)
    read_artifact(template, json_document=False)
    original_entry = original['entry']
    _need(original_entry == read_artifact(capacity['evidence']['provider'])['entry']
          and original_entry['uri'] == read_artifact(graph['program'])['entry'], 'provider source is outside original F4')
    read_artifact(original_entry, json_document=False)
    model = read_static_cpp_source(source['uri'])
    _need((model.source.path, model.source.sha256, model.source.size_bytes)
          == (source['uri'], source['sha256'], source['size_bytes']), 'C++ parser source receipt differs')
    points, return_ = _source_spans(raw, model, output_index)

    def operation(name):
        _need(name in operations, 'bound lifecycle operation is absent')
        return _fact(lifecycle_graph_ref, f'#/lanes/{lane_index}/operations/{operations[name]}')

    def anchor(point):
        span = raw[point.byte_start:point.byte_end]
        return dict(source_occurrence=encode(point.source), kind=point.kind, byte_start=point.byte_start,
                    byte_end=point.byte_end, selection_sha256=sha256(span).hexdigest())

    sites = [dict(id=identity, byte_offset=offset, indentation=indentation, operation=role,
                  operation_source=operation(role), placement='BEFORE_CPP_STATEMENT', anchor=anchor(point))
             for identity, role, offset, indentation, point in points]
    _need(all(left['byte_offset'] < right['byte_offset'] for left, right in zip(sites, sites[1:])),
          'native phase insertion offsets are not strictly ordered')
    return dict(schema=SCHEMA, schema_version=1, lifecycle_graph=lifecycle_graph_ref,
        scope_sha256=graph['scope_sha256'], option_id=capacity['option_id'], program=graph['program'],
        target=graph['target'], capacity_contract=capacity_ref, cpp_source=source, binary=binary,
        runtime=capacity['evidence']['runtime'], schedule=capacity['evidence']['schedule'],
        generator_template=template,
        original_source_contract=original_ref, original_entry=original_entry,
        original_source_roles=dict(ingress=_fact(original_ref, '#/argv'),
            stdout=_fact(original_ref, '#/report/stdout'),
            file=None if original['report']['path_output'] is None else _fact(original_ref, '#/report/path_output')),
        sites=sites, main_return=dict(anchor=anchor(return_), operation='exit', operation_source=operation('exit'),
                                     exposes_cpp_return_selection_only=True),
        timed_segments=[
            dict(id='ingress', begin='MAIN_ENTER', end='DISPATCH_BEGIN', scope='NATIVE_MAIN_PARSE_VALIDATE_AND_SETUP'),
            dict(id='dispatch', begin='DISPATCH_BEGIN', end='DISPATCH_RETURN', scope='EXECUTE_CALL_INCLUDES_JOIN_AND_FINAL_REDUCTION'),
            dict(id='post_dispatch', begin='DISPATCH_RETURN', end='PUBLICATION_BEGIN', scope='DISPATCH_STATUS_HANDLING_BEFORE_PUBLICATION'),
            dict(id='publication', begin='PUBLICATION_BEGIN', end='PUBLICATION_END', scope='CPP_FILE_AND_STDOUT_PUBLICATION_STATEMENTS'),
        ],
        coverage_frontier=['LAUNCHER_PRE_EXEC_AND_LOADER_STARTUP_OPEN', 'STDOUT_BUFFER_DRAIN_AFTER_PUBLICATION_END_OPEN',
            'COMPLETED_PROCESS_TEARDOWN_OPEN', 'SEPARATE_WORKER_AND_JOIN_TIMES_OPEN', 'FAILURE_PATHS_OPEN'],
        target_executed=False, runtime_marker_execution_proven=False, source_to_machine_location_proven=False,
        performance_bounds_proven=False, transformation_authorized=False)


def verify_cpu_native_phase_sites(reference):
    report = read_artifact(reference)
    _need(reference['schema'] == SCHEMA and report == derive_cpu_native_phase_sites(report['lifecycle_graph']),
          'saved native phase sites differ from their original scoped projection')
    return report

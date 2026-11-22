"""Bind diagnostic cuts to sealed native source and output effects."""

from scar.analysis.cpu_array_native_route_lifecycle import _anchor, verify_cpu_array_native_route_lifecycle
from scar.analysis.cpu_native_phase_sites import _source_spans
from scar.analysis.replay_transaction import replay_transaction
from scar.analysis.native_call_materials import read_static_cpp_source
from scar.codegen.native_array_entry import _argv_header
from scar.ir.execution_recipe import _canonical, read_artifact


SCHEMA = 'scar.cpu-array-native-route-probe-sites'
_MAIN = 'src/main.cpp'


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU array native route probe sites: ' + detail)


def _unique(rows, detail):
    rows = tuple(rows)
    _need(len(rows) == 1, detail)
    return rows[0]


def _source_ranges(reference, output_index):
    raw = read_artifact(reference, json_document=False)
    model = read_static_cpp_source(reference['uri'])
    _need((model.source.path, model.source.sha256, model.source.size_bytes)
          == tuple(reference[key] for key in ('uri', 'sha256', 'size_bytes')),
          'parsed entry source differs from its exact source receipt')
    functions = model.index().occurrences
    main = _unique((row for row in model.functions if row.symbol == 'main' and not row.preprocessor
                    and not row.namespaces), 'expected one unconditional main')
    main_body = functions[main.body]
    points, _ = _source_spans(raw, model, output_index)

    def site(offset, start, end):
        _need(0 < offset < len(raw) and raw[offset - 1:offset] == b'\n',
              'source cut does not fall before a real source line')
        line_end = raw.index(b'\n', offset)
        line = raw[offset:line_end]
        indent = line[:len(line) - len(line.lstrip(b' '))]
        return dict(byte_offset=offset, indentation=indent.decode('ascii'),
                    anchor=_anchor(_MAIN, raw, start, end))

    def statement(text, *, after=False):
        literal = b'\n' + text + b'\n'
        start = raw.find(literal, main_body.byte_start, main_body.byte_end)
        _need(start >= 0 and raw.find(literal, start + 1, main_body.byte_end) < 0,
              'expected unique successful-path statement in main')
        start += 1
        end = start + len(text)
        return site(end + 1 if after else start, start, end)

    structural = {name: (offset, occurrence) for name, _, offset, _, occurrence in points}
    sites = {
        'MAIN_ENTER': site(main_body.byte_start + 2, main_body.byte_start, main_body.byte_start + 1),
        'INGRESS_BEGIN': site(structural['MAIN_ENTER'][0], structural['MAIN_ENTER'][1].byte_start,
                              structural['MAIN_ENTER'][1].byte_end),
        'DISPATCH_BEGIN': statement(b'        const auto status = execute(result);'),
        'DISPATCH_RETURN': statement(b'        const auto status = execute(result);', after=True),
        'PUBLICATION_BEGIN': statement(b'        const auto& path = values['
                                       + str(output_index).encode('ascii') + b'].text;'),
        'FILE_OPEN_BEGIN': statement(b'            std::ofstream file(path, std::ios::binary | std::ios::trunc);'),
        'FILE_CLOSE_COMPLETE': statement(b'            file.close();', after=True),
        'STDOUT_BEGIN': statement(b'        write_result(std::cout, result, false);'),
        'STDOUT_SUBMIT_END': statement(rb"        std::cout << '\n';", after=True),
        'NORMAL_RETURN': statement(b'        return std::cout ? 0 : 1;'),
    }
    _need(raw[main_body.byte_start:main_body.byte_start + 2] == b'{\n'
          and sites['DISPATCH_BEGIN']['byte_offset'] == structural['DISPATCH_BEGIN'][0]
          and sites['DISPATCH_RETURN']['byte_offset'] == structural['DISPATCH_RETURN'][0]
          and sites['PUBLICATION_BEGIN']['byte_offset'] == structural['PUBLICATION_BEGIN'][0]
          and sites['STDOUT_SUBMIT_END']['byte_offset'] == sites['NORMAL_RETURN']['byte_offset']
          == structural['PUBLICATION_END'][0], 'successful main intervals differ from the parsed provider shape')
    starts = [row['byte_offset'] for role, row in sites.items() if role != 'NORMAL_RETURN']
    _need(all(a < b for a, b in zip(starts, starts[1:])), 'successful-path boundaries are not strictly ordered')
    begin, report = (sites[role]['byte_offset'] for role in ('DISPATCH_BEGIN', 'PUBLICATION_BEGIN'))

    writers = {}
    for name in ('write_worker', 'write_records', 'write_pids', 'write_result'):
        function = _unique((row for row in model.functions if row.symbol == name and not row.preprocessor),
                           'expected one generated report helper: ' + name)
        region = functions[function.occurrence]
        _need(region.byte_end < main_body.byte_start and not region.has_parse_error,
              'report helper must be a complete retained declaration')
        writers[name] = _anchor(_MAIN, raw, region.byte_start, region.byte_end)
    return sites, dict(
        ingress_prefix=_anchor(_MAIN, raw, 0, begin),
        replace_execute=_anchor(_MAIN, raw, begin, report),
        publication_suffix=_anchor(_MAIN, raw, report, len(raw))), writers


def _qualified_loader(providers, *, openmp):
    dependencies = dict(providers['common_libraries'])
    if openmp:
        dependencies.update({name: dict(kind='HOST_FILE', path=ref['uri'], sha256=ref['sha256'],
                                        size_bytes=ref['size_bytes'])
                             for name, ref in providers['openmp_runtime'].items()})
    dynamic = providers['openmp_dynamic' if openmp else 'local_dynamic']
    _need(set(dynamic['needed']) <= dependencies.keys(), 'qualified loader has unresolved direct dependencies')
    return dict(dynamic=dynamic, dependencies=dependencies)


def derive_cpu_array_native_route_probe_sites(lifecycle_ref):
    """Locate normal-success probe intervals without rebuilding or running a target."""
    with replay_transaction():
        lifecycle = verify_cpu_array_native_route_lifecycle(lifecycle_ref)
        proof = read_artifact(lifecycle['proof'])
        return _project(lifecycle_ref, lifecycle, proof)


def _project(lifecycle_ref, lifecycle, proof):
    workload = proof['workload']
    recipe = read_artifact(workload['recipe'])
    contract_ref = lifecycle['original_source_contract']
    contract = read_artifact(contract_ref)
    report = contract['report']
    _, indices = _argv_header(recipe['options'], recipe['allow_abbrev'])
    _need(recipe['source_contract'] == contract_ref and contract['entry'] == workload['source']
          and report['path_output']['kind'] == 'CONDITIONAL_PATH_WRITE_TEXT'
          and report['path_output']['option'] == recipe['output_option'] and report['path_output']['newline'] is True
          and report['stdout']['kind'] == 'JSON_TEXT_PRINT' and report['stdout']['newline'] is True
          and workload['argv'][workload['output_index']], 'expected the sealed file then stdout publication ABI')
    providers = proof['evidence']['providers']
    routes = {}
    for name, route in lifecycle['routes'].items():
        openmp = name == 'grouped_openmp'
        expected = proof['evidence']['openmp_executable' if openmp else 'retained_executable']
        runtime = read_artifact(route['runtime'])
        source = route['sources'][_MAIN]
        _need(route['executable'] == expected == runtime['files'][runtime['binary']]
              and source == runtime['files'][_MAIN], 'lifecycle source or executable differs from the qualified package')
        sites, ranges, writers = _source_ranges(source, indices[recipe['output_option']])
        routes[name] = dict(source=source, runtime=route['runtime'], executable=route['executable'],
                            sites=sites, source_ranges=ranges, report_writers=writers,
                            expected_loader=_qualified_loader(providers, openmp=openmp),
                            output_fields=read_artifact(route['origin_runtime'])['output_fields'])
    return dict(schema=SCHEMA, schema_version=1, lifecycle=lifecycle_ref, proof=lifecycle['proof'],
        scope=lifecycle['scope'], hardware=lifecycle['hardware'], workload=workload, routes=routes,
        publication=dict(original_source_contract=contract_ref, original_file='#/report/path_output',
                         original_stdout='#/report/stdout', file_close_precedes_stdout=True,
                         file_close_proves_durable_storage=False, stdout_submit_end_proves_stream_drain=False,
                         runtime_field_paths=recipe['runtime_fields']),
        qualification=dict(structural_cuts_proven=True, original_loader_closure_pinned=True,
            new_executable_needs_loader_requalification=True, same_loader_closure_implies_same_duration=False,
            same_source_implies_same_duration=False, report_fixture_needs_independent_validation=True),
        status='PROVEN_SCOPED_NORMAL_SUCCESS_PROBE_CUTS', target_executed=False, transformation_authorized=False)


def verify_cpu_array_native_route_probe_sites(reference):
    _need(reference['schema'] == SCHEMA, 'unexpected probe-site schema')
    saved = read_artifact(reference)
    _need(_canonical(saved) == _canonical(derive_cpu_array_native_route_probe_sites(saved['lifecycle'])),
          'saved probe boundaries differ from cold replay')
    return saved

"""Certify the publication tail of a scoped generated C++20 entry."""

from hashlib import sha256
from pathlib import Path

from scar.analysis.cpu_native_phase_sites import _one, verify_cpu_native_phase_sites
from scar.analysis.static_cpp_source import read_static_cpp_source
from scar.ir.execution_recipe import read_artifact
from scar.ir.record_codec import encode


SCHEMA = 'scar.cpu-native-publication-order'
_CONDITION = 'DECLARED_NONEMPTY_OUTPUT_SLOT_AND_NORMAL_PUBLICATION_COMPLETION'
_FILE_BODY = (
    b'{',
    b'std::ofstream file(path, std::ios::binary | std::ios::trunc);',
    b'file.exceptions(std::ios::failbit | std::ios::badbit);',
    b'write_result(file, result, true);',
    b"file << '\\n';",
    b'file.close();',
    b'}',
)


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU native publication order: ' + detail)


def derive_cpu_native_publication_order(sites_ref):
    """Locate and prove the ordinary statement order of the admitted suffix."""
    # This re-renders the provider's source and argv header and verifies their
    # exact package membership, original entry and selected launch contract.
    sites = verify_cpu_native_phase_sites(sites_ref)
    capacity = read_artifact(sites['capacity_contract'])
    launch = capacity['native_launch']
    contract = read_artifact(sites['original_source_contract'])
    output = contract['report']['path_output']
    index = launch['output_index']
    argv = launch['argv']
    _need(output is not None and output['kind'] == 'CONDITIONAL_PATH_WRITE_TEXT'
          and type(index) is int and index == len(argv) - 2 and index > 0
          and argv[0] == sites['binary']['uri'] and type(argv[index + 1]) is str
          and argv[index + 1] not in ('', '.', '..') and Path(argv[index + 1]).name == argv[index + 1],
          'expected the selected nonempty terminal file publication slot')

    source = sites['cpp_source']
    raw = read_artifact(source, json_document=False)
    model = read_static_cpp_source(source['uri'])
    _need((model.source.path, model.source.sha256, model.source.size_bytes)
          == (source['uri'], source['sha256'], source['size_bytes']), 'C++ source receipt differs')
    view = model.index()
    main = _one((row for row in model.functions if row.symbol == 'main' and not row.namespaces
                 and not row.preprocessor), 'expected one unconditional generated main')

    def text(row):
        return raw[row.byte_start:row.byte_end]

    def anchor(saved, kind):
        return _one((row for row in model.occurrences if row.kind == kind
                     and row.byte_start == saved['byte_start'] and row.byte_end == saved['byte_end']
                     and encode(row.source) == saved['source_occurrence']), 'phase site no longer identifies its C++ occurrence')

    def call(callee, arguments):
        row = _one((row for row in model.calls if row.owner_function == main.occurrence
                    and not row.preprocessor and row.callee is not None and text(view.occurrences[row.callee]) == callee
                    and tuple(text(view.occurrences[arg]) for arg in row.arguments) == arguments),
                   'expected one typed publication call in generated main')
        return view.occurrences[row.occurrence]

    def parent(row):
        return _one((edge.source for edge in view.incoming(row.id) if edge.role == 'contains-source-occurrence'),
                    'publication call lacks a unique structural parent')

    start = _one((row for row in sites['sites'] if row['id'] == 'PUBLICATION_BEGIN'), 'missing publication site')
    guard = anchor(start['anchor'], 'if_statement')
    result = anchor(sites['main_return']['anchor'], 'return_statement')
    branch = {edge.role: view.occurrences[edge.target] for edge in view.outgoing(guard.id)
              if edge.role in ('condition', 'consequence', 'alternative')}
    _need(branch.keys() == {'condition', 'consequence'} and text(branch['condition']) == b'(!path.empty())'
          and branch['consequence'].kind == 'compound_statement'
          and tuple(line.strip() for line in text(branch['consequence']).splitlines()) == _FILE_BODY,
          'file arm must be the canonical straight-line generated block')

    file = call(b'write_result', (b'file', b'result', b'true'))
    close = call(b'file.close', ())
    stdout = call(b'write_result', (b'std::cout', b'result', b'false'))
    _need(parent(file) == parent(close) == branch['consequence'].id
          and parent(guard) == parent(stdout) == parent(result) == main.body
          and guard.byte_end <= stdout.byte_start < stdout.byte_end < result.byte_start
          and not raw[guard.byte_end:stdout.byte_start].strip()
          and tuple(line.strip() for line in raw[stdout.byte_end:result.byte_start].strip().splitlines())
          == (b';', b"std::cout << '\\n';"), 'stdout must directly follow the generated file arm on the normal tail')

    def operation(identity, row):
        return dict(id=identity, call_id=encode(row.id), source_occurrence=encode(row.source),
                    byte_start=row.byte_start, byte_end=row.byte_end, selection_sha256=sha256(text(row)).hexdigest())

    return dict(
        schema=SCHEMA, schema_version=1, phase_sites=sites_ref, lifecycle_graph=sites['lifecycle_graph'],
        capacity_contract=sites['capacity_contract'], scope_sha256=sites['scope_sha256'], option_id=sites['option_id'],
        program=sites['program'], target=sites['target'], runtime=sites['runtime'], cpp_source=source,
        binary=sites['binary'], generator_template=sites['generator_template'],
        output_slot=dict(argv_index=index, argument=argv[index + 1], original_option=output['option']),
        branch_condition=_CONDITION, publication_operation=start['operation_source'], file_guard=encode(guard.source),
        operations=[operation('FILE_WRITE', file), operation('FILE_CLOSE', close), operation('STDOUT_PUBLICATION', stdout)],
        edges=[dict(kind='HAPPENS_BEFORE', source=before, target=after, when=_CONDITION) for before, after in
               (('FILE_WRITE', 'FILE_CLOSE'), ('FILE_CLOSE', 'STDOUT_PUBLICATION'))],
        scope='EXACT_GENERATED_CPP_NORMAL_PUBLICATION_STATEMENTS', native_target_executed=False,
        external_visibility_order_proven=False, failure_paths_equivalent=False, deployment_authorized=False,
    )


def verify_cpu_native_publication_order(reference):
    report = read_artifact(reference)
    _need(reference['schema'] == SCHEMA and report == derive_cpu_native_publication_order(report['phase_sites']),
          'saved native publication proof differs at cold replay')
    return report

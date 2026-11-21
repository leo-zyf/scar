"""Index original entry effect obligations alongside bounded CPU observations."""

from collections import defaultdict

from scar.analysis.cpu_provider_graph import SCHEMA as CPU_PROVIDER
from scar.analysis.cpu_stream_graph import SCHEMA as CPU_STREAM
from scar.analysis.entry_obligations import SCHEMA as ENTRY
from scar.ir.execution_recipe import read_artifact

SCHEMA = 'scar.f4-entry-effect-frontier'


def _need(ok, detail):
    if not ok:
        raise ValueError('entry effect frontier: ' + detail)


def _index(value, rows, name):
    _need(type(value) is int and 0 <= value < len(rows), 'invalid original ' + name + ' reference')
    return rows[value]


def _body_frontier(original):
    return [dict(original=f'#/bodies/{index}', status=row['status'])
            for index, row in enumerate(original['bodies']) if row['status'] != 'PROJECTED']


def _static(original, observed):
    requirements = defaultdict(list)
    operations = original['operations']
    bodies = original['bodies']
    for index, row in enumerate(operations):
        for name in row['obligations']:
            requirements[name].append(index)
    calls, imports = [], []
    for index, call in enumerate(original['calls']):
        op = call['operation']
        _index(op, operations, 'call operation')
        alternatives = defaultdict(list)
        for ordinal, row in enumerate(call['alternatives']):
            status = row['status']
            if status == 'SOURCE_BODY':
                body = _index(row['body'], bodies, 'call body')
                status = 'PROJECTED_SOURCE_BODY' if body['status'] == 'PROJECTED' else body['status']
            alternatives[status].append(ordinal)
        calls.append(dict(original=f'#/calls/{index}', operation=op, listed_alternatives=dict(alternatives),
                          listing_enumerated=call['provider_enumerated'],
                          has_provider_frontiers=bool(call['provider_frontiers']),
                          named_effects=observed.get(op, []), obligation=call['obligation'],
                          status='TARGET_ACTIVATION_AND_EFFECTS_OPEN'))
    for index, row in enumerate(original['imports']):
        op = row['operation']
        _index(op, operations, 'import operation')
        alternatives = defaultdict(list)
        for ordinal, option in enumerate(row['alternatives']):
            status = option['status']
            if status == 'POSSIBLE_MODULE_BODY':
                body = _index(option['body'], bodies, 'import body')
                status = 'PROJECTED_POSSIBLE_MODULE_BODY' if body['status'] == 'PROJECTED' else body['status']
            alternatives[status].append(ordinal)
        imports.append(dict(original=f'#/imports/{index}', operation=op, listed_alternatives=dict(alternatives),
                            obligation=row['obligation'], status='RESOLUTION_AND_INITIALIZATION_EFFECTS_OPEN'))
    return dict(operation_obligations=dict(sorted(requirements.items())), unprojected_bodies=_body_frontier(original),
                calls=calls, imports=imports)


class _Observations:
    def __init__(self, original):
        self.original = original
        self.anchors = []
        self.effects = []
        self.boundary = []
        self.linked_operations = defaultdict(list)

    def _anchor(self, row, slot):
        pointer = row.get('entry_operation')
        operation = None
        if pointer is not None:
            prefix, _, component = pointer.rpartition('/')
            _need(prefix == '#/operations' and component.isascii() and component.isdecimal(),
                  'provider anchor does not point to an original entry operation')
            operation = int(component)
            original = _index(operation, self.original['operations'], 'anchored operation')
            point = row['operation']
            namespace = _index(original['unit'], self.original['units'], 'anchored unit')['namespace']
            _need(point['node']['wire'] == original['operation'] and point['namespace'] == namespace,
                  'provider anchor original operation or namespace differs')
        status = ('ORIGINAL_ENTRY_OPERATION' if operation is not None else
                  'F4_CALL_OUTSIDE_ENTRY_PROJECTION' if 'operation' in row else 'SOURCE_CALL_ENDPOINT_OPEN')
        anchor = len(self.anchors)
        self.anchors.append(dict(original=slot, operation=operation, status=status))
        return anchor

    def _effect(self, kind, slot, basis, anchors=()):
        anchors = list(anchors)
        number = len(self.effects)
        self.effects.append(dict(kind=kind, original=slot, basis=basis, source_anchors=anchors,
                                 source_attribution='COMPATIBLE_ANCHORS' if anchors else 'UNBOUND'))
        operations = {self.anchors[anchor]['operation'] for anchor in anchors}
        for operation in sorted(operations - {None}):
            self.linked_operations[operation].append(number)

    def _frontiers(self, graph, prefix):
        self.boundary.extend(dict(original=f'{prefix}/coverage_frontier/{index}', kind=row['kind'])
                             for index, row in enumerate(graph['coverage_frontier']))

    def stream(self, graph, prefix):
        self._frontiers(graph, prefix)
        roles = defaultdict(list)
        for index, row in enumerate(graph['source_anchors']):
            roles[row['role']].append(self._anchor(row, f'{prefix}/source_anchors/{index}'))
        self._effect('PROCESS_EXIT', prefix+'/original_observer/exit_status', 'ORIGINAL_CAPTURE')
        self._effect('STDIN_BYTES', prefix+'/stdin', 'ORIGINAL_CAPTURE')
        self._effect('STDOUT_BYTES', prefix+'/stdout', 'ORIGINAL_CAPTURE')
        observed = graph['observable_graph']
        base = prefix+'/observable_graph'
        for index, row in enumerate(observed.get('commands', ())):
            self._effect('STDIN_RANGE', f'{base}/commands/{index}/stdin', 'TYPED_CAPTURED_INPUT',
                         roles[row['source_role']])
        for index, row in enumerate(observed.get('acknowledgements', ())):
            self._effect('STDOUT_ACKNOWLEDGEMENT', f'{base}/acknowledgements/{index}/stdout',
                         'CAPTURED_STDOUT_RANGE', roles[row['source_role']])
        for index, _ in enumerate(observed.get('search_invocations', ())):
            self._effect('INVOCATION_CHAIN', f'{base}/search_invocations/{index}', 'BOUNDED_ORIGINAL_CALLER_CHAIN')
        for index, row in enumerate(observed.get('edges', ())):
            if row['kind'] in ('CAPTURE_ACK_BEFORE_NEXT_SEND', 'PAIRED_HARNESS_ACK_BEFORE_NEXT_SEND'):
                self._effect('ACKNOWLEDGEMENT_ORDER', f'{base}/edges/{index}', row['kind'])

    def provider(self, graph):
        self._frontiers(graph, '#')
        contract = graph['observable_contract']
        base = '#/observable_contract'
        channels = defaultdict(list)
        for index, row in enumerate(contract['anchors']):
            channels[row['kind']].append(self._anchor(row, f'{base}/anchors/{index}'))
        self._effect('PROCESS_EXIT', base+'/exit_status', 'CHECKED_PAIRED_CONTRACT')
        self._effect('STDERR', base+'/stderr', 'CHECKED_PAIRED_CONTRACT')
        self._effect('STDOUT_BYTES', base+'/stdout', 'CHECKED_PAIRED_CONTRACT', channels['STDOUT'])
        for index, _ in enumerate(contract['publications']):
            self._effect('FILE_PUBLICATION', f'{base}/publications/{index}', 'CHECKED_PAIRED_CONTRACT', channels['FILE'])
        stream = graph.get('stream_capture')
        if stream is not None:
            self.stream(stream, '#/stream_capture')


def _check_graph(graph, entry_obligations, original, program):
    target = program['target']
    expected = dict(uri=target['path'], sha256=target['sha256'], size_bytes=target['bytes'],
                    schema='scar.target-hardware-graph')
    _need(graph['schema_version'] == 1 and graph['program'] == original['program']
          and graph['entry_obligations'] == entry_obligations and graph['target'] == expected
          and graph['whole_program_effects_closed'] is False and graph['deployment_authorized'] is False,
          'CPU observation differs from its original entry, target, or evidence boundary')
    source = graph['source'] if graph['schema'] == CPU_STREAM else graph['entry']['source']
    _need(source['uri'] == program['entry'], 'CPU observation describes a different source')


def derive_entry_effect_frontier(entry_obligations, cpu_graph=None):
    """Read pinned metadata and retain original JSON pointers for every obligation."""
    original = read_artifact(entry_obligations)
    _need(entry_obligations['schema'] == ENTRY and original['schema_version'] == 1
          and original['scope'] == 'NORMAL_SUCCESS_SOURCE_MAY_GRAPH'
          and original['effects_closed'] is False and original['runtime_path_complete'] is False,
          'expected original static entry effect obligations')
    program = read_artifact(original['program'])
    observed = _Observations(original)
    if cpu_graph is not None:
        _need(cpu_graph['schema'] in (CPU_PROVIDER, CPU_STREAM), 'unsupported CPU evidence graph')
        graph = read_artifact(cpu_graph)
        _check_graph(graph, entry_obligations, original, program)
        if graph['schema'] == CPU_STREAM:
            observed.stream(graph, '#')
        else:
            if graph.get('stream_capture') is not None:
                _check_graph(graph['stream_capture'], entry_obligations, original, program)
            observed.provider(graph)
    return dict(schema=SCHEMA, schema_version=1, program=original['program'], entry_obligations=entry_obligations,
                cpu_graph=cpu_graph, static=_static(original, observed.linked_operations),
                observed=dict(named_effects=observed.effects, source_anchors=observed.anchors),
                provider_frontiers=observed.boundary, effect_scope='SCOPED_NAMED_OBSERVATIONS_AND_OPEN_SOURCE_REQUIREMENTS',
                observes_all_entry_effects=False, operation_equivalence_proven=False,
                whole_entry_equivalence_proven=False, transformation_authorized=False)


def verify_entry_effect_frontier(reference):
    """Cold-replay the index from the exact original snapshots without target execution."""
    report = read_artifact(reference)
    _need(reference['schema'] == SCHEMA and report == derive_entry_effect_frontier(
        report['entry_obligations'], report['cpu_graph']), 'saved frontiers differ from original pinned entries')
    return report

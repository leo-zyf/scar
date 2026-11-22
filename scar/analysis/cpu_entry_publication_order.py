"""Prove one scoped original publication order over F4 control ports."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from scar.analysis.build import BuildSession
from scar.analysis.cpu_provider_graph import SCHEMA as PROVIDER
from scar.analysis.cpu_stream_graph import SCHEMA as STREAM
from scar.analysis.entry_obligations import SCHEMA as OBLIGATIONS
from scar.analysis.source_control import _control_states
from scar.ir.execution_recipe import read_artifact
from scar.ir.program import normal_graph as G, source_productions as P
from scar.ir.program.control import LanguageCompletionKind
from scar.ir.program.dependencies import (
    DependencyCondition, DependencyEdge, DependencyEvidence, DependencyKind, DependencyMeaning,
)
from scar.ir.program.providers import ProviderArtifact
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.program.source_declarations import SourceTemplateKey
from scar.ir.record_codec import decode, encode

SCHEMA = 'scar.f4-cpu-entry-publication-order'
_NORMAL = ('FALLTHROUGH', 'RETURN')
_PROVEN = 'PROVEN_SCOPED_NORMAL_SUCCESS'


@dataclass(frozen=True, slots=True)
class _Control:
    target: int
    transition: G.Transition
    site: int | None


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU entry publication order: ' + detail)


def _at(rows, ordinal):
    _need(type(ordinal) is int and 0 <= ordinal < len(rows), 'invalid original table ordinal')
    return rows[ordinal]


def _locator(reference, pointer):
    artifact = decode(ProviderArtifact, reference)
    return SourceArtifactLocator(artifact, artifact.sha256, artifact.size_bytes, 'canonical-json', pointer)


def _operation(original, anchor):
    pointer = anchor.get('entry_operation')
    if pointer is None:
        return None
    prefix, _, index = pointer.rpartition('/')
    _need(prefix == '#/operations' and index.isascii() and index.isdecimal(),
          'provider anchor does not reference an original operation')
    index = int(index)
    row = _at(original['operations'], index)
    point = anchor['operation']
    _need(point['node']['wire'] == row['operation']
          and point['namespace'] == _at(original['units'], row['unit'])['namespace'],
          'provider role changed its original operation or namespace')
    return index


def _port(original, ordinal):
    row = _at(original['ports'], ordinal)
    owner = _at(original['owners'], row['owner'])
    template = _at(original['templates'], owner['template'])
    return row, owner, template


def _body_ports(view, original, body):
    unit = _at(original['units'], body['unit'])['unit']
    source = view.unit(unit)
    matches = []
    for function in source.facts.functions:
        if function.body is None:
            continue
        point = view.source_region_ref(unit, function.template, function.body)
        if encode(point) == body['point']:
            matches.append((function, point))
    if len(matches) != 1:
        return None
    function, point = matches[0]
    owners = {index for index, row in enumerate(original['owners'])
              if row['original'] == function.body.wire
              and (template := _at(original['templates'], row['template']))['unit'] == body['unit']
              and template['template'] == encode(function.template)}
    if len(owners) != 1:
        return None
    begins, normal = [], set()
    for index, row in enumerate(original['ports']):
        if row['owner'] in owners:
            if row['kind'] == G.PortKind.ENTER.value:
                begins.append(index)
            elif row['kind'] == G.PortKind.EXIT.value and row['completion'] in _NORMAL:
                normal.add(index)
    if len(begins) != 1 or not normal:
        return None
    return unit, point, begins[0], normal


def _control(original, body):
    result = defaultdict(list)
    for index in body['edges']:
        row = _at(original['edges'], index)
        if row['kind'] != G.EdgeKind.CONTROL.value:
            continue
        source, target = row['source'], row['target']
        _need(all(_port(original, endpoint)[2]['unit'] == body['unit'] for endpoint in (source, target)),
              'body control edge left its declared source unit')
        site = row['site']
        if site is not None:
            _need(_at(original['sites'], site)['unit'] == body['unit'],
                  'body control edge changed its original call site')
        result[source].append(_Control(target, G.Transition(row['transition']), site))
    return result


def _prove(controls, body_enter, normal, file_enter, file_end, stdout_enter):
    try:
        reached = _control_states(((body_enter, ()),), controls, ())
        if any(edge.transition is G.Transition.RESUME
               for port, _ in reached for edge in controls.get(port, ())):
            return 'ORIGINAL_RESUMPTION_CONTEXT_OPEN'
        file_states = [state for state in reached if state[0] == file_end]
        stdout_states = [state for state in reached if state[0] == stdout_enter]
        if not file_states or not stdout_states:
            return 'PUBLICATION_OPERATION_CONTEXT_UNREACHABLE'
        suffix = _control_states(file_states, controls, ())
        if not any(port in normal and not stack for port, stack in suffix):
            return 'FILE_COMPLETION_HAS_NO_NORMAL_BODY_EXIT_WITNESS'
        without_stdout = _control_states(file_states, controls, {stdout_enter})
        if any(port in normal and not stack for port, stack in without_stdout):
            return 'NORMAL_BODY_EXIT_CAN_BYPASS_STDOUT_AFTER_FILE'
        backwards = _control_states(stdout_states, controls, ())
        if any(port in (file_enter, file_end) for port, _ in backwards):
            return 'STDOUT_CAN_PRECEDE_FILE_IN_THE_SAME_BODY'
    except ValueError as error:
        if not str(error).endswith('budget exhausted'):
            raise
        return 'MATCHED_CONTROL_TRAVERSAL_BUDGET_EXHAUSTED'
    return None


def _point(view, original, unit, ordinal):
    row, owner, template = _port(original, ordinal)
    _need(not row['path'] and row['target'] is None, 'expected an ordinary source publication control port')
    prefix, _, payload = owner['original'].partition(':')
    _need(prefix == P.ProductionStepID.prefix, 'publication endpoint must reference an original source step')
    port = G.Port(decode(SourceTemplateKey, template['template']), P.ProductionStepID(payload),
        G.PortKind(row['kind']), row['ordinal'],
        None if row['completion'] is None else LanguageCompletionKind(row['completion']))
    return view.unit(unit).port(port)


def derive_cpu_entry_publication_order(provider_graph_ref):
    """Close only same-activation normal-call order when the original graph proves it."""
    graph = read_artifact(provider_graph_ref)
    _need(provider_graph_ref['schema'] in (PROVIDER, STREAM) and graph['schema_version'] == 1
          and graph['whole_program_effects_closed'] is False, 'expected original scoped CPU provider evidence')
    original_ref = graph['entry_obligations']
    program = graph['program']
    result = dict(schema=SCHEMA, schema_version=1, provider_graph=provider_graph_ref, program=program,
        entry_obligations=original_ref, status='UNRESOLVED', reason='FILE_AND_STDOUT_ROLES_NOT_BOUND',
        scope='ORIGINAL_F4_NORMAL_SUCCESS_SAME_ACTIVATION', edge=None, anchors=None,
        physical_visibility_order_proven=False, whole_entry_effect_equivalence_proven=False,
        transformation_authorized=False)
    if provider_graph_ref['schema'] == STREAM or original_ref is None:
        return result
    original = read_artifact(original_ref)
    _need(original_ref['schema'] == OBLIGATIONS and original['schema_version'] == 1
          and original['program'] == program and original['scope'] == 'NORMAL_SUCCESS_SOURCE_MAY_GRAPH',
          'provider observation changed its original F4 obligations')
    claims = graph['observable_contract']['anchors']
    file, stdout = ([index for index, row in enumerate(claims) if row['kind'] == role]
                    for role in ('FILE', 'STDOUT'))
    if len(file) != 1 or len(stdout) != 1:
        return result
    file_op, stdout_op = (_operation(original, claims[index]) for index in (file[0], stdout[0]))
    if file_op is None or stdout_op is None:
        return result
    bodies = [(index, body) for index, body in enumerate(original['bodies'])
              if body['status'] == 'PROJECTED' and {file_op, stdout_op} <= set(body['operations'])]
    if len(bodies) != 1 or file_op == stdout_op:
        return dict(result, reason='ONE_PROJECTED_ORIGINAL_PUBLICATION_BODY_REQUIRED')
    index, body = bodies[0]
    first, second = original['operations'][file_op], original['operations'][stdout_op]
    if first['unit'] != body['unit'] or second['unit'] != body['unit']:
        return dict(result, reason='SAME_ORIGINAL_ACTIVATION_UNBOUND')
    file_enter, stdout_enter = first['reach_port'], second['reach_port']
    start, _, _ = _port(original, file_enter)
    target, _, _ = _port(original, stdout_enter)
    _need(start['kind'] == target['kind'] == G.PortKind.ENTER.value,
          'publication source operation lost its ENTER port')
    exits = [ordinal for ordinal, row in enumerate(original['ports'])
             if row['owner'] == start['owner'] and row['path'] == start['path']
             and row['kind'] == G.PortKind.EXIT.value and row['completion'] == 'FALLTHROUGH']
    if len(exits) != 1:
        return dict(result, reason='ORIGINAL_FILE_NORMAL_COMPLETION_PORT_OPEN')
    file_end = exits[0]
    program_document = read_artifact(program)
    target_receipt = program_document['target']
    _need(graph['target'] == dict(uri=target_receipt['path'], sha256=target_receipt['sha256'],
          size_bytes=target_receipt['bytes'], schema='scar.target-hardware-graph'),
          'provider effect graph has a different original target')
    with BuildSession(Path(program['uri']).parent) as session:
        _need(session.report == program_document, 'original F4 report differs')
        body_ports = _body_ports(session.view, original, body)
        if body_ports is None:
            return dict(result, reason='ORIGINAL_BODY_NORMAL_COMPLETION_OPEN')
        unit, body_point, enter, normal = body_ports
        gap = _prove(_control(original, body), enter, normal, file_enter, file_end, stdout_enter)
        if gap is not None:
            return dict(result, reason=gap)
        from_port = _point(session.view, original, unit, file_end)
        to_port = _point(session.view, original, unit, stdout_enter)
    body_slot, from_slot, to_slot = f'#/bodies/{index}', f'#/ports/{file_end}', f'#/ports/{stdout_enter}'
    roles = [f'#/observable_contract/anchors/{ordinal}' for ordinal in (file[0], stdout[0])]
    sources = (_locator(original_ref, body_slot), *(_locator(provider_graph_ref, slot) for slot in roles))
    proof = DependencyEvidence('scar.f4.publication-normal-order.v1', sources, (body_point, from_port, to_port))
    conditions = (DependencyCondition('scar.f4.publication-normal-order.v1',
        'SAME_ORIGINAL_BODY_ACTIVATION_NORMAL_COMPLETION', body_point, operands=(from_port, to_port)),)
    edge = DependencyEdge(from_port, to_port, DependencyKind.ORDER, DependencyMeaning.ORDER, (proof,), conditions)
    return dict(result, status=_PROVEN, reason='ALL_NORMAL_EXITS_AFTER_FILE_REQUIRE_STDOUT_AND_NO_REVERSE_ROUTE',
                edge=encode(edge), anchors=dict(body=body_slot, file_completion=from_slot, stdout_entry=to_slot,
                                               file_role=roles[0], stdout_role=roles[1],
                                               file_operation=f'#/operations/{file_op}',
                                               stdout_operation=f'#/operations/{stdout_op}'))


def verify_cpu_entry_publication_order(reference):
    """Recompute the typed proof from retained originals without launching them."""
    result = read_artifact(reference)
    _need(reference['schema'] == SCHEMA and result == derive_cpu_entry_publication_order(result['provider_graph']),
          'saved publication order differs after original F4 replay')
    return result

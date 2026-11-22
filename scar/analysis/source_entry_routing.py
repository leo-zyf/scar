"""Source entry routing."""

from __future__ import annotations

from collections import Counter
import json
from scar.analysis.function_interfaces import validate_body_interface
from scar.analysis.program_composition import QualifiedSourceBodyReference
from scar.ir.program.call_edges import ReferenceOccurrence, ReferenceRole
from scar.ir.program.functions import FunctionSignature, ParameterKind, FunctionKind
from scar.ir.record_codec import decode, encode
from scar.ir.program.providers import ProviderFactReference


# Independent checks of local entry snapshots against original formal ports.
#
# These checks do not authorize caller actual-expression routes or object/content
# identities. A live binder supplies original typed observations and body readers.
class EntryRoutingError(ValueError):
    pass


def _same(actual, expected):
    # Transport equality is exact JSON equality, including bool versus int;
    # Python container equality alone would erase that distinction.
    try:
        options = dict(sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        return json.dumps(actual, **options) == json.dumps(expected, **options)
    except (ValueError, TypeError):
        return False


def require_code_layout(descriptor, signature):
    """CPython code layout differs from the source signature's canonical order."""
    ranks = {ParameterKind.POSITIONAL_ONLY: 0, ParameterKind.POSITIONAL: 0,
             ParameterKind.KEYWORD_ONLY: 1, ParameterKind.VAR_POSITIONAL: 2,
             ParameterKind.VAR_KEYWORD: 3}
    expected = tuple(p.name for p in sorted(signature.parameters, key=lambda p: ranks[p.kind]))
    if descriptor.parameters != expected:
        raise EntryRoutingError('entry_code_layout: captured names differ from exact source kinds/order')
    for kind, flag in ((ParameterKind.VAR_POSITIONAL, 4), (ParameterKind.VAR_KEYWORD, 8)):
        if bool(descriptor.flags & flag) != any(p.kind is kind for p in signature.parameters):
            raise EntryRoutingError('entry_code_layout: captured variadic flags differ from source signature')
    kind = (FunctionKind.ASYNC_GENERATOR if descriptor.flags & 0x200 else
            FunctionKind.ASYNC if descriptor.flags & 0x80 else
            FunctionKind.GENERATOR if descriptor.flags & 0x20 else FunctionKind.SYNC)
    if kind is not signature.kind:
        raise EntryRoutingError('entry_code_layout: captured function kind differs from source signature')


def parameter_source(fact, ordinal):
    return ProviderFactReference(fact.artifact, fact.locator + '/parameters/' + str(ordinal))


def entry_routing_errors(rows, *, entries, captures, bindings, contexts, members):
    """Check the entire actual denominator, without accepting a scope/hash seal.

    contexts originate in the ordinary qualified body reader. No serialized
    result or caller boolean reconstructs the binder's live session.
    """
    errors = []

    def check(condition, code, path, message):
        if not condition: errors.append((code, path, message))
        return condition

    if type(rows) not in (tuple, list):
        return [('entry_inventory', '#', 'entry projection must be an ordered row vector')]
    ordered = sorted(entries.values(), key=lambda e: (e.invocation.process_id, e.anchor.index))
    if len(rows) != len(ordered):
        return [('entry_inventory', '#', 'projection omits or adds original invocations')]
    top_fields = {'kind', 'invocation', 'anchor', 'code', 'source_binding_index', 'point',
                  'source', 'qualified_body', 'namespace', 'status', 'frontiers', 'parameters'}
    fields = {'ordinal', 'formal_position', 'formal_slot', 'formal_kind', 'occurrence', 'source'}
    for number, (row, entry) in enumerate(zip(rows, ordered, strict=True)):
        path = '#/' + str(number)
        if not check(type(row) is dict and set(row) == top_fields, 'entry_fields', path,
                     'entry projection has missing/extra fields'): continue
        binding_index, binding = bindings[entry.invocation.process_id, entry.code]
        context = contexts.get(binding_index)
        descriptor = captures[entry.invocation.process_id, entry.code].code
        fact = members.fact(entry.anchor, entry.invocation.process_id)
        point = members.point(entry.anchor, entry.invocation.process_id)
        check(_same(row['kind'], 'CALLEE_ENTRY_PARAMETERS') and _same(row['invocation'], encode(entry.invocation))
              and _same(row['anchor'], encode(entry.anchor)) and _same(row['code'], entry.code)
              and type(row['source_binding_index']) is int and row['source_binding_index'] == binding_index,
              'entry_identity', path, 'projection lost its exact original invocation/row/code')
        check(_same(row['source'], encode(fact)) and _same(row['point'], encode(point)),
              'entry_source', path, 'entry point is not the exact coordinate-bound raw fact')
        reasons = []
        if not entry.parameters_complete: reasons.append('ENTRY_PARAMETERS_INCOMPLETE')
        if entry.suspended: reasons.append('SUSPENDED_ENTRY_UNSUPPORTED')
        if point is None: reasons.append('ENTRY_PROCESS_MEMBERSHIP_UNRESOLVED')
        if not binding['complete_descriptor_match']: reasons.append('COMPLETE_CODE_SOURCE_WITNESS_UNAVAILABLE')
        if context is None: reasons.append('QUALIFIED_CANONICAL_BODY_UNAVAILABLE')
        eligible = not reasons
        check(_same(row['frontiers'], reasons) and _same(row['status'], 'QUALIFIED_CALLEE_ENTRY_SNAPSHOT' if eligible else 'FRONTIER'),
              'entry_qualification', path, 'entry qualification/frontier differs from original evidence')
        check(_same(row['qualified_body'], None if context is None else context['reference'])
              and _same(row['namespace'], None if context is None else context['namespace']),
              'entry_namespace', path, 'formal port lost its original qualified body/namespace')
        if entry.parameters_complete:
            check(tuple(p.name for p in entry.parameters) == descriptor.parameters,
                  'entry_parameter_names', path, 'complete entry names differ from captured descriptor')
        if context is not None: require_code_layout(descriptor, context['signature'])
        if not check(type(row['parameters']) is list and len(row['parameters']) == len(entry.parameters),
                     'entry_parameter_inventory', path, 'entry parameter vector was omitted/expanded'): continue
        for ordinal, (projected, observed) in enumerate(zip(row['parameters'], entry.parameters, strict=True)):
            slot_path = path + '/parameters/' + str(ordinal)
            if not check(type(projected) is dict and set(projected) == fields,
                         'entry_parameter_fields', slot_path, 'parameter has missing/extra fields'): continue
            parameter = None if not eligible else context['by_name'].get(observed.name)
            if eligible and not check(parameter is not None, 'entry_parameter_name', slot_path,
                                      'captured formal is absent from exact source signature'): continue
            position = None if parameter is None else context['positions'][parameter.slot]
            port = None if parameter is None else context['ports'][parameter.slot]
            expected = ReferenceOccurrence(entry.anchor, entry.invocation, entry.invocation,
                ReferenceRole.PARAMETER, observed.name, observed.ref,
                None if parameter is None else parameter.source, core_value=None if port is None else port.value)
            check(type(projected['ordinal']) is int and projected['ordinal'] == ordinal
                  and _same(projected['occurrence'], encode(expected)),
                  'entry_parameter_occurrence', slot_path, 'parameter reference/name/owner/core value differs from original entry')
            check(_same(projected['formal_position'], position)
                  and _same(projected['formal_slot'], None if parameter is None else encode(parameter.slot))
                  and _same(projected['formal_kind'], None if parameter is None else parameter.kind.value),
                  'entry_parameter_port', slot_path, 'parameter selected another formal slot/kind/order')
            check(_same(projected['source'], encode(parameter_source(fact, ordinal))),
                  'entry_parameter_source', slot_path, 'parameter source selects another original ordered observation')
    return errors


def require_entry_routing(rows, **original):
    errors = entry_routing_errors(rows, **original)
    if errors:
        raise EntryRoutingError('; '.join(code + ' at ' + path + ': ' + message for code, path, message in errors))


# Owned local projections of original entry snapshots onto existing ports.
#
# No caller actual route, FunctionObject, ObjectIdentity, logical version or full
# ExecutionModel is created. Saved rows cannot reconstruct this live session.
ENTRY_ROUTING_PROFILE = 'CALLEE_ENTRY_PARAMETER_ROUTING_V1'
_OWNER = object()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


class EntryRoutingSession:
    """Actual source-bound query/check session, not a serialized authority."""

    def __init__(self, key, composition, entries, captures, binding_rows, members):
        if key is not _OWNER: raise TypeError('entry routing needs the actual fresh source binder')
        self._composition, self._members = composition, members
        self._entries, self._captures = dict(entries), dict(captures)
        self._bindings = {(r['process_id'], r['code']): (i, json.loads(_canonical(r)))
                          for i, r in enumerate(binding_rows)}
        self._contexts = {}
        # One normal reader/static-layout lookup per code/body, never per event.
        for index, row in enumerate(binding_rows):
            body = row.get('body')
            if body is None: continue
            if body.get('kind') != 'QUALIFIED_SOURCE_BODY':
                continue  # Other body protocols remain in the explicit frontier denominator.
            reference = decode(QualifiedSourceBodyReference, body['reference'])
            typed = composition.typed_qualified_body(reference)
            item = composition._qualified_entry_item(reference)
            original = item['record']
            if (original['entry_key'] != item['entry'].key
                    or original['definition'] != encode(reference.definition)):
                raise ValueError('entry routing cached signature differs from original checked publication')
            signature = decode(FunctionSignature, original['signature'])
            validate_body_interface(typed.body, typed.interface, signature)
            if (encode(reference.definition) != row['definition'] or body != composition.qualified_body_record(reference)
                    or typed.interface.definition != reference.definition.definition):
                raise ValueError('entry routing changes original code/source/body/definition')
            descriptor = captures[row['process_id'], row['code']].code
            require_code_layout(descriptor, signature)
            self._contexts[index] = {'reference': encode(reference), 'namespace': typed.namespace,
                'signature': signature, 'by_name': {p.name: p for p in signature.parameters},
                'ports': {p.slot: p for p in typed.interface.parameters},
                'positions': {p.slot: i for i, p in enumerate(signature.parameters)}}
        rows = []
        for entry in sorted(entries.values(), key=lambda e: (e.invocation.process_id, e.anchor.index)):
            index, binding = self._bindings[entry.invocation.process_id, entry.code]
            context = self._contexts.get(index)
            fact = members.fact(entry.anchor, entry.invocation.process_id)
            point = members.point(entry.anchor, entry.invocation.process_id)
            reasons = []
            if not entry.parameters_complete: reasons.append('ENTRY_PARAMETERS_INCOMPLETE')
            if entry.suspended: reasons.append('SUSPENDED_ENTRY_UNSUPPORTED')
            if point is None: reasons.append('ENTRY_PROCESS_MEMBERSHIP_UNRESOLVED')
            if not binding['complete_descriptor_match']: reasons.append('COMPLETE_CODE_SOURCE_WITNESS_UNAVAILABLE')
            if context is None: reasons.append('QUALIFIED_CANONICAL_BODY_UNAVAILABLE')
            parameters = []
            for ordinal, observed in enumerate(entry.parameters):
                parameter = None if reasons else context['by_name'][observed.name]
                port = None if parameter is None else context['ports'][parameter.slot]
                occurrence = ReferenceOccurrence(entry.anchor, entry.invocation, entry.invocation,
                    ReferenceRole.PARAMETER, observed.name, observed.ref,
                    None if parameter is None else parameter.source, core_value=None if port is None else port.value)
                parameters.append({'ordinal': ordinal,
                    'formal_position': None if parameter is None else context['positions'][parameter.slot],
                    'formal_slot': None if parameter is None else encode(parameter.slot),
                    'formal_kind': None if parameter is None else parameter.kind.value,
                    'occurrence': encode(occurrence), 'source': encode(parameter_source(fact, ordinal))})
            rows.append({'kind': 'CALLEE_ENTRY_PARAMETERS', 'invocation': encode(entry.invocation),
                'anchor': encode(entry.anchor), 'code': entry.code, 'source_binding_index': index,
                'point': encode(point), 'source': encode(fact),
                'qualified_body': None if context is None else context['reference'],
                'namespace': None if context is None else context['namespace'],
                'status': 'FRONTIER' if reasons else 'QUALIFIED_CALLEE_ENTRY_SNAPSHOT',
                'frontiers': reasons, 'parameters': parameters})
        self.check_rows(rows)
        self._rows = _canonical(rows)

    def rows(self):
        return tuple(json.loads(self._rows))

    def check_rows(self, rows):
        # Guard the real body reader, then use independent original-row/port
        # predicates. Changed candidates receive semantic errors, not seal errors.
        for context in self._contexts.values():
            self._composition._qualified_entry_item(decode(QualifiedSourceBodyReference, context['reference']))
        require_entry_routing(rows, entries=self._entries, captures=self._captures,
            bindings=self._bindings, contexts=self._contexts, members=self._members)

    def summary(self):
        rows = self.rows()
        qualified = [r for r in rows if r['status'] == 'QUALIFIED_CALLEE_ENTRY_SNAPSHOT']
        return {'profile': ENTRY_ROUTING_PROFILE, 'invocations': len(rows),
            'observed_parameter_occurrences': sum(len(r['parameters']) for r in rows),
            'qualified_entry_invocations': len(qualified),
            'qualified_formal_ports': sum(len(r['parameters']) for r in qualified),
            'frontier_invocations': len(rows) - len(qualified),
            'frontiers': dict(Counter(g for r in rows for g in r['frontiers'])),
            'caller_actual_expression_routing_established': False, 'object_identity_established': False,
            'logical_contents_established': False, 'full_execution_model_established': False,
            'function_environment_state_ready_established': False,
            'reference_tokens_are_occurrence_carriers_not_object_count': True}


def _make_entry_routing(composition, entries, captures, bindings, members):
    return EntryRoutingSession(_OWNER, composition, entries, captures, bindings, members)

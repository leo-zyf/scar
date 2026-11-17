"""Conditional MAY-summary composition over admitted, shared source bodies.

Only the existing source interfaces cross CALL boundaries.  A finite token
lattice and call-graph SCCs summarize recursion without inlining, invocation
identities, value equality, effect closure, or future-dispatch claims.
"""
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
import json

import networkx as nx

from scar.ir.execution_recipe import artifact_reference
from scar.ir.program import normal_targets as T
from scar.ir.record_codec import decode, encode
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.interprocedural-summary'
_CHANNELS = ('data', 'environment', 'state', 'unknown')
_EMPTY = tuple(frozenset() for _ in _CHANNELS)
_FORMALS = {'FORMAL_DATA', 'FORMAL_ENVIRONMENT', 'FORMAL_STATE'}
_CALLS = {'CALL_RESULT', 'CALL_WORLD'}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _join(*values):
    return tuple(frozenset().union(*(value[index] for value in values))
        for index in range(len(_CHANNELS)))


def _unknown(reason, **source):
    return (*_EMPTY[:3], frozenset((_canonical(dict(reason=reason, **source)),)))


def _pack(value):
    return {channel: ([json.loads(item) for item in sorted(value[index])]
        if channel == 'unknown' else sorted(value[index]))
        for index, channel in enumerate(_CHANNELS)}


class _DependencyTable:
    """Intern only serialization; the fixed-point lattice is unchanged."""

    def __init__(self):
        self.sets, self.unknowns, self.values, self.unknown_ids = {}, {}, {}, {}

    def intern(self, value):
        found = self.values.get(value)
        if found is not None:
            return found
        unknown_refs = []
        for raw in sorted(value[3]):
            reference = self.unknown_ids.get(raw)
            if reference is None:
                fact = json.loads(raw)
                reference = 'unknown-fact:'+semantic_sha256(fact)
                self.unknowns[reference] = fact
                self.unknown_ids[raw] = reference
            unknown_refs.append(reference)
        record = dict(data=sorted(value[0]), environment=sorted(value[1]),
            state=sorted(value[2]), unknown_refs=unknown_refs)
        reference = 'dependency-set:'+semantic_sha256(record)
        self.sets[reference] = record
        self.values[value] = reference
        return reference


def resolve_dependency_set(report, dependency_ref):
    """Return detached original channels after checking both content IDs.

    This validates a referenced metadata record, not its source admission.
    ``verify_interprocedural_summary`` still cold-replays the entire report.
    """
    if type(dependency_ref) is dict and set(dependency_ref)=={'dependency_ref'}:
        dependency_ref = dependency_ref['dependency_ref']
    if type(dependency_ref) is not str:
        raise ValueError('one content-addressed dependency_ref is required')
    record = report.get('dependency_sets', {}).get(dependency_ref)
    if (type(record) is not dict or set(record)!={'data','environment','state','unknown_refs'}
            or any(type(record[field]) is not list or any(type(value) is not str for value in record[field])
                for field in record)
            or dependency_ref != 'dependency-set:'+semantic_sha256(record)):
        raise ValueError('dependency reference is missing or its content ID differs')
    result = {field:list(record[field]) for field in _CHANNELS[:3]}
    if any(result[field] != sorted(set(result[field])) for field in result):
        raise ValueError('dependency token channels must retain their canonical MAY sets')
    unknown = []
    for reference in record['unknown_refs']:
        facts = report.get('unknown_facts', {})
        if reference not in facts or reference != 'unknown-fact:'+semantic_sha256(facts[reference]):
            raise ValueError('unknown fact reference is missing or its content ID differs')
        unknown.append(deepcopy(facts[reference]))
    canonical = [_canonical(value) for value in unknown]
    if canonical != sorted(set(canonical)):
        raise ValueError('unknown facts must retain their canonical MAY set')
    result['unknown'] = unknown
    return result


def expand_interprocedural_dependencies(report):
    """Display-only projection with the original expanded dependency fields.

    Local producer reports are unchanged. This projection is not the retained
    normalized report and cannot replace it during independent verification.
    """
    result = deepcopy(report)
    for row in result['functions']:
        selected = row['conditional_selected_target']
        for field in ('normal_return', 'normal_return_world'):
            selected[field] = resolve_dependency_set(report, selected[field])
        for field in ('reads', 'writes'):
            for access in selected[field]:
                access['dependencies'] = resolve_dependency_set(report, access.pop('dependency_ref'))
    for row in result['calls']:
        row['dispatch_environment_dependencies'] = resolve_dependency_set(report, row['dispatch_environment_dependencies'])
        for value in row['selected_target_dependence']:
            value['dependencies'] = resolve_dependency_set(report, value.pop('dependency_ref'))
    for field in ('dependency_sets', 'unknown_facts', 'dependency_encoding'):
        result.pop(field)
    return result


def _collect(view, entry):
    from scar.analysis.source_function_summary import derive_source_function_summary

    functions = []
    for row in entry['source_bodies']:
        body = decode(T.Body, row['body'])
        source = view.unit(body.unit)
        matches = [function for function in source.facts.functions
            if function.template == body.template and function.body == body.region]
        if len(matches) != 1 or matches[0].body_scope is None:
            raise ValueError('entry summary requires one original ordinary source function')
        local = derive_source_function_summary(view, body.unit, matches[0])
        if (local['body'] != row['body'] or local['definition'] != row['definition']
                or local['body_point'] != row['point']):
            raise ValueError('local summary differs from the admitted original entry body')
        functions.append(dict(function_id=local['function_id'], local=local,
            body=deepcopy(row['body']), source_function=deepcopy(local['source_function']),
            definition=deepcopy(row['definition']), point=deepcopy(row['point']),
            ports=deepcopy(local['ports']), body_scope=encode(matches[0].body_scope.owner)))
    return sorted(functions, key=lambda row: row['function_id'])


def _compose(functions, entry):
    """Pure finite-set propagation; all rows still address original records."""
    functions = deepcopy(functions)
    by_id = {row['function_id']: row for row in functions}
    if len(by_id) != len(functions):
        raise ValueError('one shared source function may occur only once')
    by_body = {_canonical(row['body']): row['function_id'] for row in functions}
    if len(by_body) != len(functions):
        raise ValueError('shared original body has conflicting function summaries')
    tokens, contexts, accesses = {}, {}, {}
    function_indices = {row['function_id']: index for index, row in enumerate(functions)}
    graph = nx.DiGraph()
    graph.add_nodes_from(by_id)
    for index, row in enumerate(functions):
        local = row['local']
        local_tokens = {token['id']: token for token in local['tokens']}
        if len(local_tokens) != len(local['tokens']):
            raise ValueError('local summary repeats an original dependency token')
        for key, token in local_tokens.items():
            if key in tokens and tokens[key] != token:
                raise ValueError('original dependency token has conflicting source identity')
            tokens[key] = token
        ports = {_canonical(item['port']): item for item in local['port_dependencies']}
        formal_ids = {key for key, token in local_tokens.items() if token['kind'] in _FORMALS}
        contexts[row['function_id']] = dict(row=row, tokens=local_tokens, ports=ports,
            formal_ids=formal_ids, calls=[], base_accesses={})
        for table in ('reads', 'writes'):
            for ordinal, access in enumerate(local[table]):
                ref = '#/functions/'+str(index)+'/local/'+table+'/'+str(ordinal)
                accesses[ref] = dict(source_function_id=row['function_id'], source_access_ref=ref)
                contexts[row['function_id']]['base_accesses'][ref] = (table, access['dependencies'])

    call_values, effect_values, calls = {}, {key: {} for key in by_id}, []
    for fid, context in contexts.items():
        for ordinal, call in enumerate(context['row']['local']['call_sites']):
            local_ref = '#/functions/'+str(function_indices[fid])+'/local/call_sites/'+str(ordinal)
            candidates = []
            for index, selected in enumerate(entry['interfaces']):
                interface = selected['interface']
                link = interface['link']
                if (interface['query'] != call['query'] or link['result'] != call['result']
                        or link['callee'] != call['callee']
                        or selected.get('caller_scope') != context['row']['body_scope']):
                    continue
                target_id = None if link['body'] is None else by_body.get(_canonical(link['body']))
                candidates.append(dict(function_id=target_id,
                    interface_ref='#/interfaces/'+str(index), link=link,
                    reference_document='ENTRY_PROJECTION',
                    ordinary_body_entry=selected['ordinary_body_entry'],
                    conditions=deepcopy(link['conditions']), binding=link['binding'],
                    typed_route_edge_refs=deepcopy(selected.get('typed_route_edge_refs', []))))
                if target_id is not None and selected['ordinary_body_entry']:
                    graph.add_edge(fid, target_id)
            call_row = dict(caller_function_id=fid, call_id=call['call_id'], query=call['query'],
                local_call_ref=local_ref, targets=candidates,
                future_dispatch=dict(status='UNKNOWN',
                    reason='BOUNDED_SELECTED_TARGETS_DO_NOT_ENUMERATE_FUTURE_DISPATCH'))
            call_index = len(calls)
            calls.append(call_row)
            context['calls'].append((call_index, call))
            for token_id, token in context['tokens'].items():
                if token['kind'] in _CALLS and token['port'] in (call['result'], call['result_world']):
                    call_values[fid, token_id] = _EMPTY

    def local_deps(fid, dependency):
        context = contexts[fid]
        value = _EMPTY
        for channel_index, channel in enumerate(_CHANNELS):
            for item in dependency[channel]:
                token = context['tokens'].get(item) if type(item) is str else None
                if token is not None and token['kind'] in _CALLS:
                    value = _join(value, call_values.get((fid, item),
                        _unknown('CALL_TOKEN_HAS_NO_ORIGINAL_CALL_SITE', function_id=fid, token=item)))
                elif channel == 'unknown':
                    value = _join(value, (*_EMPTY[:3], frozenset((_canonical(item),))))
                else:
                    part = list(_EMPTY)
                    part[channel_index] = frozenset((item,))
                    value = _join(value, tuple(part))
        return value

    def port_deps(fid, port):
        dependency = contexts[fid]['ports'].get(_canonical(port))
        return (local_deps(fid, dependency) if dependency is not None else
            _unknown('ACTUAL_OR_RETURN_PORT_HAS_NO_LOCAL_DEPENDENCY_ROW', function_id=fid, port=port))

    def bindings(caller, call, candidate):
        target = candidate['function_id']
        if target is None or not candidate['ordinary_body_entry']:
            return {}
        callee = contexts[target]
        formal_tokens = {_canonical(token['port']): key for key, token in callee['tokens'].items()
            if key in callee['formal_ids']}
        result = {}
        link = candidate['link']
        for route in link['parameters']:
            key = formal_tokens.get(_canonical(route['formal']))
            if key is None:
                continue
            if route['kind'] in ('INITIAL_DEFAULT', 'CONDITIONAL_DEFAULT'):
                # The initial expression is provenance, not current default
                # identity. Retain the admitted provider route and its gap.
                value = _join(port_deps(caller, call['callee']), port_deps(caller, call['world']),
                    _unknown('CURRENT_DEFAULT_STATE_NOT_INITIAL_VALUE_ALIAS',
                        interface_ref=candidate['interface_ref'], formal=route['formal'],
                        typed_route_edge_refs=candidate['typed_route_edge_refs']))
            else:
                value = _join(*(port_deps(caller, actual) for actual in route['actuals']))
                if not route['actuals'] and route['kind'] not in ('PACK_POSITIONALS', 'PACK_KEYWORDS'):
                    value = _join(value, _unknown('PARAMETER_ROUTE_HAS_NO_DIRECT_OPERAND',
                        interface_ref=candidate['interface_ref'], formal=route['formal']))
            result[key] = _join(result.get(key, _EMPTY), value)
        for projection in link['projections']:
            key = formal_tokens.get(_canonical(projection['formal']))
            if key is not None:
                result[key] = _join(result.get(key, _EMPTY), port_deps(caller, projection['pack']),
                    _unknown('SYMBOLIC_PACK_CONTENT_NOT_SELECTED_REFERENCE_BINDING',
                        interface_ref=candidate['interface_ref'], projection=projection))
        for key in callee['formal_ids']-result.keys():
            # An input envelope is deliberately not a parameter alias.
            envelope = link.get('input_envelope')
            value = _unknown('FORMAL_ARGUMENT_BINDING_REMAINS_OPEN',
                interface_ref=candidate['interface_ref'], token=key,
                input_envelope=deepcopy(envelope))
            if envelope is not None:
                actuals = [envelope['callee'], envelope['world'], *envelope['positional'],
                    *(item[1] for item in envelope['keywords']),
                    *(envelope[field] for field in ('positional_pack', 'keyword_pack')
                        if envelope[field] is not None)]
                value = _join(value, *(port_deps(caller, actual) for actual in actuals))
            result[key] = value
        return result

    def substitute(target, value, replacements):
        result = (*_EMPTY[:3], value[3])
        for index in range(3):
            for key in value[index]:
                if key in contexts[target]['formal_ids']:
                    result = _join(result, replacements[key])
                else:
                    part = list(_EMPTY)
                    part[index] = frozenset((key,))
                    result = _join(result, tuple(part))
        return result

    def transfer(fid):
        context = contexts[fid]
        effects = {ref: (table, local_deps(fid, dependency))
            for ref, (table, dependency) in context['base_accesses'].items()}
        values = {}
        for call_index, call in context['calls']:
            selected = calls[call_index]
            dispatch = port_deps(fid, call['callee'])
            envelope = None

            def open_inputs():
                nonlocal envelope
                if envelope is None:
                    points = [call['callee'], call['world'],
                        *(item['port'] for item in call['positional']),
                        *(item['port'] for item in call['keywords']),
                        *(call[field] for field in ('positional_pack', 'keyword_pack')
                            if call[field] is not None)]
                    envelope = _join(*(port_deps(fid, point) for point in points))
                return envelope
            relevant = [(key, token) for key, token in context['tokens'].items()
                if token['kind'] in _CALLS and token['port'] in (call['result'], call['result_world'])]
            for key, _ in relevant:
                values[fid, key] = _EMPTY
            if not selected['targets']:
                for key, _ in relevant:
                    values[fid, key] = _join(open_inputs(), _unknown('NO_ADMITTED_SELECTED_SOURCE_TARGET',
                        function_id=fid, call_id=call['call_id']))
            for candidate in selected['targets']:
                target = candidate['function_id']
                if target is None or not candidate['ordinary_body_entry']:
                    for key, _ in relevant:
                        values[fid, key] = _join(values[fid, key],
                            open_inputs(),
                            _unknown('SELECTED_INTERFACE_HAS_NO_ORDINARY_LOCAL_SUMMARY',
                                interface_ref=candidate['interface_ref']))
                    continue
                replacements = bindings(fid, call, candidate)
                for key, token in relevant:
                    returns = [source for source, destination in candidate['link']['returns']
                        if destination == token['port']]
                    value = (_join(*(substitute(target, port_deps(target, source), replacements)
                        for source in returns)) if returns else
                        _unknown('SELECTED_INTERFACE_HAS_NO_MATCHING_NORMAL_RETURN',
                            interface_ref=candidate['interface_ref'], port=token['port']))
                    values[fid, key] = _join(values[fid, key], value, dispatch)
                for ref, (table, dependency) in effect_values[target].items():
                    value = _join(substitute(target, dependency, replacements), dispatch)
                    previous = effects.get(ref, (table, _EMPTY))
                    effects[ref] = table, _join(previous[1], value)
        return values, effects

    components = list(nx.strongly_connected_components(graph))
    condensation = nx.condensation(graph, components)
    convergence = []
    for component in reversed(list(nx.topological_sort(condensation))):
        members = sorted(condensation.nodes[component]['members'])
        iterations = 0
        while True:
            iterations += 1
            changed = False
            for fid in members:
                values, effects = transfer(fid)
                for key, value in values.items():
                    merged = _join(call_values[key], value)
                    if merged != call_values[key]:
                        call_values[key] = merged
                        changed = True
                for ref, (table, value) in effects.items():
                    previous = effect_values[fid].get(ref)
                    merged = _join(previous[1] if previous is not None else _EMPTY, value)
                    if previous != (table, merged):
                        effect_values[fid][ref] = table, merged
                        changed = True
            if not changed:
                break
        convergence.append(dict(function_ids=members, iterations=iterations,
            recursive=len(members)>1 or graph.has_edge(members[0], members[0]), converged=True))

    dependency_table = _DependencyTable()
    for fid, context in contexts.items():
        row = context['row']
        local = row['local']
        row['conditional_selected_target'] = dict(
            normal_return=dict(dependency_ref=dependency_table.intern(local_deps(fid, local['normal_return']))),
            normal_return_world=dict(dependency_ref=dependency_table.intern(local_deps(fid, local['normal_return_world']))),
            **{table: [dict(**accesses[ref], dependency_ref=dependency_table.intern(dependency))
                for ref, (kind, dependency) in sorted(effect_values[fid].items()) if kind == table]
                for table in ('reads', 'writes')})
        reachable = {fid, *nx.descendants(graph, fid)}
        row['future_dispatch_frontier'] = [dict(source_function_id=owner, call_id=call['call_id'],
            status='UNKNOWN', reason='BOUNDED_SELECTED_TARGETS_DO_NOT_ENUMERATE_FUTURE_DISPATCH')
            for owner in sorted(reachable) for _, call in contexts[owner]['calls']]
        row['effect_closure'] = 'UNKNOWN_EMPTY_SOURCE_ACCESS_SET_IS_NOT_EFFECT_ABSENCE'
        row['normal_completion_only'] = True
        row['termination_proven'] = False
        row['transformation_authorized'] = False
    for row in calls:
        fid = row['caller_function_id']
        context = contexts[fid]
        original_call = next(call for _, call in context['calls'] if call['call_id']==row['call_id'])
        row['dispatch_environment_dependencies'] = dict(dependency_ref=dependency_table.intern(port_deps(fid, original_call['callee'])))
        row['selected_target_dependence'] = [dict(token=key, dependency_ref=dependency_table.intern(call_values[fid, key]))
            for key, token in sorted(context['tokens'].items()) if (fid, key) in call_values
            and any(call['call_id']==row['call_id'] and token['port'] in (call['result'], call['result_world'])
                for _, call in context['calls'])]
        for candidate in row['targets']:
            candidate.pop('link')
    return dict(functions=functions, tokens=[tokens[key] for key in sorted(tokens)], calls=calls,
        dependency_sets=dependency_table.sets, unknown_facts=dependency_table.unknowns,
        dependency_encoding='CONTENT_ADDRESSED_DERIVED_DEPENDENCY_SETS_V1_LOCAL_REPORTS_UNCHANGED',
        sccs=convergence, summary=dict(functions=len(functions), calls=len(calls),
            conditional_targets=sum(len(row['targets']) for row in calls),
            recursive_sccs=sum(row['recursive'] for row in convergence), tokens=len(tokens)))


def derive_interprocedural_summary(session, entry_calls=None):
    """Replay one admitted entry projection, then reuse that session's view."""
    from scar.analysis.bounded_entry_calls import _admit_entry_calls

    with _admit_entry_calls(session, entry_calls) as admission:
        return _derive_interprocedural_summary_admitted(session, admission)


def _derive_interprocedural_summary_admitted(session, admission):
    from scar.analysis.bounded_entry_calls import _entry_projection, _entry_admission_dependencies

    entry = _entry_projection(session, admission)
    result = _compose(_collect(session._f4.view, entry), entry)
    result.update(schema=SCHEMA, schema_version=1,
        scope='CONDITIONAL_SELECTED_SOURCE_TARGET_MAY_DEPENDENCIES',
        program=deepcopy(entry['program']), producer=artifact_reference(__file__, 'scar.python-producer'),
        entry_projection=dict(sha256=semantic_sha256(entry), program=deepcopy(entry['program']),
            producer=deepcopy(entry['producer'])),
        source_frontiers=deepcopy(entry['frontiers']),
        observed_target_frontiers=deepcopy(entry['observed_target_frontiers']),
        qualification=dict(strength='MAY', fixed_point='FINITE_ORIGINAL_TOKEN_SETS_UNION_ONLY',
            selected_targets_conditional=True, future_dispatch_exhaustive=False,
            selected_target_conditions_discharged=False,
            normal_completion_only=True, termination_proven=False, whole_program_complete=False,
            source_bodies_inlined=False, loops_unrolled=False, recursion_unrolled=False,
            runtime_invocations_created=False, source_unknowns_are_not_zero_or_purity=True,
            reference_aliases_or_content_versions_proven=False, effects_closed=False,
            transformation_authorized=False, scheduling_authorized=False,
            cost_or_timing_inferred=False, target_executed=False))
    result['id'] = 'interprocedural-summary:'+semantic_sha256(result)
    _entry_admission_dependencies(session, admission,
        (result['producer'], *(row['local']['producer'] for row in result['functions']),
         *(reference for row in result['functions'] for reference in row['local']['source_refs'])))
    return result


def verify_interprocedural_summary(session, report):
    """Recompute from the same original admission, including every CALL route."""
    with _verified_interprocedural_summary(session, report) as (expected, _):
        return expected


@contextmanager
def _verified_interprocedural_summary(session, report, *, entry_calls=None):
    """Lend a genuine entry capability across summary and witness consumers.

    The public verifier remains cold. Only this active context can reuse the
    session's already replayed entry, and its exit rechecks original material.
    """
    from scar.analysis.bounded_entry_calls import _admit_entry_calls

    with _admit_entry_calls(session, entry_calls) as admission:
        expected = _derive_interprocedural_summary_admitted(session, admission)
        if _canonical(expected) != _canonical(report):
            raise ValueError('interprocedural summary differs from original entry and source-route replay')
        yield deepcopy(expected), admission


__all__ = ['derive_interprocedural_summary', 'verify_interprocedural_summary',
    'resolve_dependency_set', 'expand_interprocedural_dependencies']

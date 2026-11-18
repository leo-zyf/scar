"""Replay generated wrapper argument routes against retained F4 and ATen calls."""

from collections import defaultdict
import re

from scar.analysis.source_binding_native_graph import BindingNativeStore, SCHEMA as GRAPH_SCHEMA
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program.binding_returns import ReceiverMode
from scar.ir.program.providers import ProviderFactReference
from scar.ir.program.python_bindings import BindingKind
from scar.ir.record_codec import decode, encode


SCHEMA = 'scar.aten-declared-wrapper-routes'
_SCHEMA_POINTER = re.compile(r'#/declarations/([0-9]+)/schema\Z')
_NONE_BRANCH = re.compile(r'(!)?_r\.isNone\(([0-9]+)\)\Z')
_SINGLE_LIST = re.compile(r'(?:int|SymInt)\[\d*\]\??\Z')
_OPEN_EFFECTS = (
    'EXECUTED_WRAPPER_TO_GENERATED_SOURCE_BINDING_UNVERIFIED',
    'RUNTIME_PARSER_ACCEPTANCE_AND_CONVERSIONS_UNVERIFIED',
    'WRAPPER_OVERRIDE_PRE_POST_AND_EXCEPTION_EFFECTS_OPEN',
    'NATIVE_DISPATCH_EFFECTS_AND_READINESS_OPEN',
)


def _shape(candidate, definition):
    site = candidate.site
    if site.positional_pack is not None or site.keyword_pack is not None:
        return None, 'PACK_LAYOUT_UNRESOLVED'
    if (candidate.receiver_mode is not ReceiverMode.FUNCTION or candidate.receiver is not None
            or definition.kind is not BindingKind.FUNCTION or definition.implicit_receiver is not None):
        return None, 'RECEIVER_LAYOUT_UNRESOLVED'
    parameters = {argument.name: argument for argument in definition.arguments}
    order = tuple(sorted((argument for argument in definition.arguments if not argument.keyword_only),
                         key=lambda argument: -1 if argument.position is None else argument.position))
    if (len(parameters) != len({a.parser_index for a in definition.arguments})
            or len(parameters) != len(definition.arguments)
            or tuple(a.position for a in order) != tuple(range(len(order)))
            or any(a.keyword_only and a.position is not None for a in definition.arguments)):
        raise ValueError('generated parser parameter layout is inconsistent')
    if len(site.positional) > len(order):
        if len(order) == 1 and _SINGLE_LIST.fullmatch(order[0].type):
            return None, 'POSITIONAL_LIST_AGGREGATION_UNRESOLVED'
        return None, 'DECLARED_POSITIONAL_ARITY_MISMATCH'
    supplied = {arg.name: port for arg, port in zip(order, site.positional)}
    for name, port in site.keywords:
        if name not in parameters:
            return None, 'KEYWORD_ALIAS_LAYOUT_UNRESOLVED'
        if name in supplied:
            return None, 'DECLARED_DUPLICATE_PARAMETER'
        supplied[name] = port
    if any(name not in supplied and arg.default is None and arg.default_init is None
           for name, arg in parameters.items()):
        return None, 'DECLARED_REQUIRED_PARAMETER_OMITTED'

    branch = definition.branch_condition
    match = _NONE_BRANCH.fullmatch(branch) if branch is not None else None
    if match is None:
        return supplied, 'UNCONDITIONAL_DECLARATION' if branch is None else 'DECLARED_BRANCH_CONDITION_OPEN'
    negate, index = match.groups()
    parameter, = (arg for arg in definition.arguments if arg.parser_index == int(index))
    if parameter.name in supplied or (parameter.default, parameter.default_init) != ('None', None):
        return supplied, 'DECLARED_BRANCH_CONDITION_OPEN'
    if negate:
        return None, 'DECLARED_NONE_BRANCH_EXCLUDED_BY_OMISSION'
    return supplied, 'DECLARED_NONE_BRANCH_MATCHES_OMISSION'


def _same_site(call, site):
    return (encode(site.positional), encode(site.keywords), encode(site.result),
            encode(site.world), encode(site.result_world)) == (
                call['positional'], call['keywords'], call['result'],
                call['world_before'], call['world_after'])


def _schema(store, definition, materials):
    _, _, boundary = store.natives[definition.selector]
    artifact = boundary.signature.artifact
    if artifact not in materials:
        materials[artifact] = read_artifact(artifact)
    position = _SCHEMA_POINTER.fullmatch(boundary.signature.locator)
    if position is None:
        raise ValueError('canonical native boundary lacks its original schema pointer')
    declared = materials[artifact]['declarations'][int(position[1])]
    if declared['selector'] != definition.selector:
        raise ValueError('canonical native declaration changed its selector')
    return boundary, declared


def _selection(indexed_inputs, match, definition, supplied, branch_status, boundary, declared, op):
    deps_by_port = indexed_inputs[(match.candidate.site.result, definition.id)]
    ordered = []
    parser_uses = set()
    for slot, parameter in zip(boundary.inputs, declared['inputs'], strict=True):
        deps = deps_by_port.get(slot.port, ())
        parser_uses.update(dep.dependency.parser_parameter for dep in deps)
        ordered.append(dict(ordinal=slot.position, native_parameter=parameter['name'],
            declared_port=encode(slot.evidence), observed_port=dict(
                artifact=op['eto']['artifact'], locator=op['eto']['locator'] + '/inputs/values/' + str(slot.position)),
            dependencies=[encode(dep) for dep in deps]))
    unmapped = [dict(parser_parameter=arg.name, parser_index=arg.parser_index, tensor_option=arg.tensor_option,
        actual=encode(supplied[arg.name]) if arg.name in supplied else None,
        default=arg.default, default_init=arg.default_init)
        for arg in definition.arguments if arg.name not in parser_uses]
    mutable = [slot.position for slot, parameter in zip(boundary.inputs, declared['inputs'], strict=True)
               if parameter['alias'] is not None and parameter['alias']['may_write'] is True]
    return dict(binding=encode(definition.id), declaration=encode(match.declaration),
        generated_wrapper=encode(definition.wrapper), cpp_symbol=definition.cpp_symbol,
        parser_variant=definition.parser_index, deprecated=definition.deprecated,
        branch=definition.branch_condition, branch_status=branch_status,
        native_schema=encode(boundary.signature), ordered_routes=ordered,
        wrapper_only_parameters=unmapped, supplied_unmapped_actuals=[row for row in unmapped if row['actual'] is not None],
        declared_native_mutability=declared['declared_mutable'], declared_write_ordinals=mutable)


def _site_row(store, indexed_inputs, materials, matches, call, scoped_call):
    row = dict(attempt=call['attempt'], source_call=call['source_call'], operand=scoped_call['operand'],
               scope=scoped_call['declaration'], unresolved_effects=list(_OPEN_EFFECTS))
    children = scoped_call['operators']
    row.update(operators=[dict(schema=op['schema'], eto=op['eto'], kineto=op['kineto']) for op in children],
               selection=None, rejected_alternatives=[], open_alternatives=[])
    if len(children) != 1:
        return dict(row, status='OPERATOR_SCOPE_CARDINALITY_UNRESOLVED')
    op, = children
    candidates = []
    for match in matches:
        definition = store.parents.by_id[match.candidate.binding]
        if not _same_site(call, match.candidate.site):
            raise ValueError('named binding site differs from the retained F4 call ABI')
        alternative = dict(declaration=encode(match.declaration), native_selector=definition.selector,
                           parser_variant=definition.parser_index, deprecated=definition.deprecated)
        if definition.selector != op['schema'].partition('(')[0]:
            row['rejected_alternatives'].append(dict(alternative, reason='OBSERVED_NATIVE_SELECTOR_DIFFERS'))
            continue
        boundary, declared = _schema(store, definition, materials)
        if declared['namespace'] + '::' + declared['schema'] != op['schema']:
            row['rejected_alternatives'].append(dict(alternative, reason='OBSERVED_FULL_NATIVE_SCHEMA_DIFFERS'))
            continue
        supplied, reason = _shape(match.candidate, definition)
        if supplied is not None:
            candidates.append((match, definition, supplied, reason, boundary, declared))
        else:
            alternatives = 'open_alternatives' if reason.endswith('_UNRESOLVED') else 'rejected_alternatives'
            row[alternatives].append(dict(alternative, reason=reason))
    if len(candidates) != 1 or row['open_alternatives']:
        row['open_alternatives'].extend(dict(declaration=encode(match.declaration), native_selector=definition.selector,
            parser_variant=definition.parser_index, deprecated=definition.deprecated, reason=reason)
            for match, definition, _, reason, _, _ in candidates)
        return dict(row, status='DECLARED_ROUTE_AMBIGUOUS' if row['open_alternatives'] else 'NO_DECLARED_ROUTE')
    selection = _selection(indexed_inputs, *candidates[0], op)
    if selection['deprecated']:
        row['unresolved_effects'].append('DECLARED_DEPRECATION_WARNING_POSSIBLE')
    if selection['declared_native_mutability']:
        row['unresolved_effects'].append('DECLARED_NATIVE_ARGUMENT_WRITE')
    if selection['supplied_unmapped_actuals']:
        row['unresolved_effects'].append('WRAPPER_ONLY_ACTUAL_EFFECTS_OPEN')
    return dict(row, status='DECLARED_ROUTE_SELECTED', selection=selection)


def read_aten_declared_routes(binding_graph, source, scoped):
    """Cold-replay one F4 activation against its checked ATen observations.

    Call before importing the runtime provider parser. A unique route describes
    the retained declaration under the checked call shape and schema.
    """
    store = binding_graph if isinstance(binding_graph, BindingNativeStore) else BindingNativeStore(binding_graph)
    if store.parents.binding_report['source_revision'] != scoped.revision:
        raise ValueError('native wrapper declaration revision differs from the original capture')
    unit = source['body']['unit']
    graph = store.read_unit(unit)
    methods = defaultdict(list)
    inputs = defaultdict(lambda: defaultdict(list))
    for match in store.parents.matches(unit):
        methods[match.call].append(match)
    for row in graph.inputs:
        dependency = row.dependency
        inputs[(dependency.call, dependency.binding)][dependency.native_port].append(row)
    calls = {row['attempt']: row for row in scoped.calls}
    if (len(calls) != len(scoped.calls) or len(calls) != len(source['calls'])
            or set(calls) != {row['attempt'] for row in source['calls']}):
        raise ValueError('declared route inputs do not name the same retained source calls')

    materials, rows = {}, []
    for call in source['calls']:
        captured = calls[call['attempt']]
        if (captured['invocation'] != source['invocation'] or captured['site'] != call['site']
                or captured['outcome'] != 'C_RETURN' or captured['status'] != 'CLOSED_SCOPE_OBSERVED'):
            raise ValueError('declared wrapper route requires the original normally completed F4 call scope')
        matches = methods[decode(ProviderFactReference, call['source_call'])]
        rows.append(_site_row(store, inputs, materials, matches, call, captured))
    store.validate()
    return dict(schema=SCHEMA, schema_version=1,
        binding_graph=artifact_reference(store.path / 'result.json', GRAPH_SCHEMA),
        invocation=source['invocation'], coordinate_binding=encode(scoped.coordinates.artifact), site_rows=rows,
        runtime_parser_selected=False, binary_source_bound=False, effect_contract_established=False,
        replacement_authorized=False)

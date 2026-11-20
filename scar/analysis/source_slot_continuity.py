"""Check direct receiver-field writes on balanced original F4 control slices."""

from collections import defaultdict, deque

from scar.analysis.source_address_paths import AddressResolver
from scar.analysis.source_content_interfaces import ContentInterfaceIndex
from scar.analysis.source_effects import classify_instruction
from scar.analysis.source_loop_dependencies import _control_states
from scar.analysis.source_namespace_interfaces import NamespacePlanView
from scar.analysis.source_reference_slice import ReferenceSlicer, source_port
from scar.analysis.source_value_flow import project_value_flow
from scar.ir.program import normal_graph as G, normal_targets as T, source_productions as P
from scar.ir.program.control import LanguageCompletionKind as Completion
from scar.ir.program.source_address_paths import AddressRootKind, FieldSelector
from scar.ir.program.source_effects import EffectResolution
from scar.ir.record_codec import encode


SCHEMA = 'scar.source-slot-continuity'
_ATTRIBUTES = frozenset((P.LanguageActionKind.ATTRIBUTE_GET, P.LanguageActionKind.ATTRIBUTE_SET,
                         P.LanguageActionKind.ATTRIBUTE_DELETE))
_NORMAL_END = frozenset((Completion.RETURN, Completion.FALLTHROUGH))


def _enter(endpoint):
    return G.Port(endpoint.template, endpoint.owner, G.PortKind.ENTER)


def _normal_window(flow, start, stops, backedge):
    """Find source steps on balanced paths with exactly the selected carry."""
    adjacent = defaultdict(list)
    prefix = defaultdict(list)
    for edge in flow.edges:
        if (edge.kind is not G.EdgeKind.CONTROL or edge.transition is G.Transition.RESUME
                or Completion.RAISE in (edge.source.completion, edge.target.completion)):
            continue
        if edge.transition is not G.Transition.BACKEDGE:
            prefix[edge.source].append(edge)
        elif edge.target != backedge:
            continue
        adjacent[edge.source].append(edge)
    body = flow.body
    head = G.Port(body.template, body.region, G.PortKind.ENTER)
    required = int(backedge is not None)
    seeds = [(port, stack, 0) for port, stack in _control_states(((head, ()),), prefix, set())
             if port == start]
    seen = set(seeds)
    pending = deque(seeds)
    previous = defaultdict(list)
    endings = []
    while pending:
        state = pending.popleft()
        port, stack, crossed = state
        if port in stops:
            if crossed == required:
                endings.append(state)
            continue
        for edge in adjacent[port]:
            following = stack
            count = crossed
            if edge.transition is G.Transition.CALL:
                if edge.site is None or len(stack) >= 128:
                    raise ValueError('source slot window lacks a bounded original call context')
                following = (*stack, edge.site)
            elif edge.transition is G.Transition.RETURN:
                if not stack or stack[-1] != edge.site:
                    continue
                following = stack[:-1]
            elif edge.transition is G.Transition.BACKEDGE:
                if crossed == required:
                    continue
                count += 1
            successor = edge.target, following, count
            previous[successor].append((state, edge))
            if successor in seen:
                continue
            if len(seen) >= 20_000:
                raise ValueError('source slot window exceeds balanced control-state budget')
            seen.add(successor)
            pending.append(successor)

    possible = set(endings)
    edges = set()
    pending.extend(endings)
    while pending:
        for predecessor, edge in previous[pending.popleft()]:
            edges.add(edge)
            if predecessor not in possible:
                possible.add(predecessor)
                pending.append(predecessor)
    return {port for port, _, _ in possible if port not in stops}, edges


def _field_path(path):
    if (path.frontier is None and path.root_kind is AddressRootKind.FUNCTION_PARAMETER
            and all(type(field) is FieldSelector for field in path.selectors)):
        return path.root, tuple(field.name for field in path.selectors)
    return None


class _Function:
    def __init__(self, view, body):
        self.source = view.unit(body.unit)
        function, = (row for row in self.source.facts.functions
                     if (row.template, row.body) == (body.template, body.region))
        self.function = function
        self.body = body
        self.head = G.Port(body.template, body.region, G.PortKind.ENTER)
        self.flow = project_value_flow(self.source.material, source_port(self.source.facts, function.value), body)
        self.ends = {edge.target for edge in self.flow.edges if edge.target.owner == body.region
                     and edge.target.kind is G.PortKind.EXIT and edge.target.completion in _NORMAL_END}
        self.addresses = AddressResolver(view.catalog.fact_store, self.source.facts, function.body_scope.owner)
        self.attributes = ContentInterfaceIndex(self.source)
        self.contexts = {}

    def paths(self, slicer, endpoint):
        if endpoint.unit != self.source.unit:
            return ()
        _, ordinal = slicer.local(endpoint)
        return self.addresses.paths(ordinal)

    def rebase(self, slicer, link, fields):
        result = {}
        for name, location in fields.items():
            routes = tuple(route for route in link.parameters if location is not None
                           and route.kind is T.RouteKind.ARGUMENT and route.formal == location[0])
            path = (self.paths(slicer, routes[0].actuals[0]) if len(routes) == 1
                    and len(routes[0].actuals) == 1 else ())
            root = _field_path(path[0]) if len(path) == 1 else None
            result[name] = None if root is None else (root[0], (*root[1], *location[1]))
        return result

    def attribute(self, entry):
        key = entry.template
        if key not in self.contexts:
            self.source._admit((key,))
            self.contexts[key] = NamespacePlanView(self.source.unit, self.source.plans[key])
        context = self.contexts[key]
        return self.attributes.identify(context, context.steps[entry.owner])


def _write_relation(location, addresses, selector):
    if location is None:
        return 'PROTECTED_FIELD_SOURCE_ROUTE_OPEN'
    root, fields = location
    if selector not in fields:
        return 'DIFFERENT_FIELD_SELECTOR'
    paths = tuple(_field_path(path) for path in addresses)
    if any(path is not None and path[0] == root and (*path[1], selector) == fields[:len(path[1]) + 1]
           for path in paths):
        return 'DIRECT_SELECTED_PATH_OR_PREFIX_WRITE_POSSIBLE'
    return 'RECEIVER_ALIAS_TO_SELECTED_PATH_UNRESOLVED'


def _project_window(slicer, function, roots, start, stops, backedge=None):
    operations, edges = _normal_window(function.flow, start, stops, backedge)
    calls = {_enter(source_port(function.source.facts, row.call.result).port): row.fact
             for row in function.flow.calls}
    writes = []
    unresolved = []
    for op in function.flow.operations:
        if op.entry not in operations:
            continue
        ins = op.instruction
        if type(ins) is P.ApplyLanguageAction and ins.action in _ATTRIBUTES:
            reason = 'DESCRIPTOR_OR_ATTRIBUTE_HOOK_EFFECT_OPEN'
            if ins.action is not P.LanguageActionKind.ATTRIBUTE_GET:
                interface = function.attribute(op.entry)
                selector = interface.selector.name
                addresses = function.paths(slicer, interface.receiver)
                writes.append(dict(operation=op.entry, action=interface.kind, selector=selector,
                    receiver_paths=addresses, original_fact=slicer.fact(
                        function.source.facts, 'attributes', interface.attribute_fact),
                    roots={root: _write_relation(location, addresses, selector) for root, location in roots.items()}))
        elif op.entry in calls:
            reason = 'CALL_BODY_OR_EXTERNAL_CALLBACK_EFFECT_OPEN'
        elif classify_instruction(ins).resolution is EffectResolution.DYNAMIC_TARGET_OR_PROTOCOL:
            reason = 'PROTOCOL_EFFECT_OPEN'
        else:
            continue
        unresolved.append(dict(operation=op.entry, reason=reason, call_fact=calls.get(op.entry)))
    function.source.check()
    edge_order = lambda edge: (repr(edge.source), repr(edge.target), repr(edge.condition), repr(edge.when))
    return dict(body=function.body, material=function.flow.source_templates, start=start,
                stops=tuple(sorted(stops, key=repr)), has_compatible_normal_path=bool(operations),
                control_backedges=tuple(sorted((edge for edge in edges if edge.transition is G.Transition.BACKEDGE),
                                              key=edge_order)),
                guarded_edges=tuple(sorted((edge for edge in edges if edge.condition is not None), key=edge_order)),
                direct_field_writes=writes, effect_frontiers=unresolved)


def project_slot_continuity(view, region, outer_scope, outer_call, selected_call, loop_header):
    """Project direct field writes between two selected nested call entries.

    Endpoints and field routes refer to original F4. The four structural
    windows are a bounded normal-path overapproximation. Descriptor, callback,
    effect, exceptional-route, and actual backedge evidence stay separate.
    """
    source = view.unit(outer_scope.unit)
    outer_function, = (row for row in source.facts.functions if row.body_scope is not None
                       and row.body_scope.owner == outer_scope)
    outer_body = T.Body(source.unit, outer_function.template, outer_function.body)
    outer = _Function(view, outer_body)
    caller = _Function(view, outer_call.body)
    selected = _Function(view, selected_call.body)
    origin, = (row for row in region.source.origins if row.get('kind') == 'source_identity')
    if (outer_call.result.unit != outer.body.unit or selected_call.result.unit != caller.body.unit
            or any(link.binding is not T.ArgumentBinding.DIRECT for link in (outer_call, selected_call))
            or selected.body.unit != origin['unit'] or selected.body.template.occurrence != origin['occurrence']
            or selected.function.parameter_names != region.parameter_names
            or loop_header.unit != outer.body.unit or loop_header.port.kind is not G.PortKind.ENTER):
        raise ValueError('original field windows differ from their selected source call interfaces')
    slicer = ReferenceSlicer(view.catalog)
    initial = {}
    for spec in region.inputs:
        if spec.kind != 'array':
            continue
        index = selected.function.parameter_names.index(spec.path[0])
        initial[spec.id] = (source_port(selected.source.facts, selected.function.formals[index + 1]), spec.path[1:])
    caller_roots = caller.rebase(slicer, selected_call, initial)
    outer_roots = outer.rebase(slicer, outer_call, caller_roots)
    child_entry = _enter(selected_call.result.port)
    child_exit = G.Port(child_entry.template, child_entry.owner, G.PortKind.EXIT, completion=Completion.FALLTHROUGH)
    parent_entry = _enter(outer_call.result.port)
    parent_exit = G.Port(parent_entry.template, parent_entry.owner, G.PortKind.EXIT, completion=Completion.FALLTHROUGH)
    windows = {
        'FIRST_SELECTED_BODY': _project_window(slicer, selected, initial, selected.head, selected.ends),
        'FIRST_CALLER_SUFFIX': _project_window(slicer, caller, caller_roots, child_exit, caller.ends),
        'OUTER_CARRY': _project_window(slicer, outer, outer_roots, parent_exit, {parent_entry}, loop_header.port),
        'NEXT_CALLER_PREFIX': _project_window(slicer, caller, caller_roots, caller.head, {child_entry}),
    }
    roots = {}
    for name in initial:
        effects = [write['roots'][name] for window in windows.values() for write in window['direct_field_writes']]
        if ('DIRECT_SELECTED_PATH_OR_PREFIX_WRITE_POSSIBLE' in effects
                or 'RECEIVER_ALIAS_TO_SELECTED_PATH_UNRESOLVED' in effects):
            status = 'POSSIBLE_DIRECT_FIELD_OVERWRITE'
        elif (not initial[name][1] or caller_roots[name] is None or outer_roots[name] is None
                or not all(window['has_compatible_normal_path'] for window in windows.values())):
            status = 'SOURCE_FIELD_ROUTE_OR_NORMAL_CONTROL_OPEN'
        else:
            status = 'NO_DIRECT_FIELD_OVERWRITE_IN_NORMAL_SOURCE_PROJECTION'
        roots[name] = dict(status=status, selected_field_path=initial[name], caller_field_path=caller_roots[name],
                           outer_field_path=outer_roots[name])
    return encode(dict(schema=SCHEMA, schema_version=1, inputs=roots, windows=windows,
                conditions=['ORIGINAL_NORMAL_CALL_AND_RETURN_ROUTES',
                    'DIRECT_RECEIVER_FIELD_ROUTE_SELECTED_WITH_ORDINARY_ATTRIBUTE_SLOTS',
                    'DESCRIPTORS_CALLBACKS_AND_EXTERNAL_BINDING_EFFECTS_REQUIRE_INDEPENDENT_EVIDENCE'],
                actual_branch_choices_selected=False, intervening_effects_closed=False,
                slot_binding_continuity_verified=False, transformation_authorized=False))

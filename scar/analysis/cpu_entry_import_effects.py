"""Join original initializer namespace effects to their operands and possible readers."""

from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

from scar.analysis.build import BuildSession
from scar.analysis.cpu_entry_import_initializers import SCHEMA as INITIALIZERS
from scar.analysis.cpu_entry_import_initializers import _coordinates, verify_cpu_entry_import_initializers
from scar.analysis.entry_obligations import _body
from scar.analysis.source_decoded_bodies import reuse_decoded_source_bodies
from scar.analysis.source_effects import classify_instruction
from scar.analysis.source_loop_dependencies import _control_states
from scar.analysis.source_namespace_interfaces import NamespacePlanView
from scar.analysis.source_reference_slice import ReferenceSlicer, source_port
from scar.analysis.source_state import SourceStateProvider
from scar.ir.execution_recipe import read_artifact
from scar.ir.program import normal_graph as G, source_productions as P
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.source_bindings import EnvironmentInterface
from scar.ir.program.source_effects import EffectResolution
from scar.ir.record_codec import encode

SCHEMA = 'scar.f4-cpu-entry-import-effects'
_BODY_SCOPE = 'ORIGINAL_NORMAL_SUCCESS_SOURCE_INITIALIZER'
_READ_SCOPE = 'LATER_LOOKUP_MAY_HIT_SAME_SYMBOLIC_LOCATION'
_READ_CONDITIONS = ('SAME_INITIALIZER_ACTIVATION', 'LOOKUP_BRANCH_AND_LOCATION_INSTANCE_SELECTED',
                    'NO_INTERVENING_BINDING_INVALIDATION')
_STORES = {P.NamespaceActionKind.STORE_NAME: (1, 'ROOT_LOCALS'),
           P.NamespaceActionKind.STORE_GLOBAL: (0, 'ROOT_GLOBALS')}


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU entry initializer effects: ' + detail)


def _control(adjacency, seeds):
    try:
        return _control_states(seeds, adjacency, set(), max_states=100_000)
    except ValueError as error:
        if str(error) not in ('loop membership control-state budget exhausted',
                              'loop membership recipe-stack budget exhausted'):
            raise
        return None


class _OriginalBody:
    def __init__(self, original, row, source):
        self.source = source
        self.original = original
        self.root = source.facts.root_scope
        self.coordinates, _ = _coordinates(original, row)
        steps, edges = _body(source, self.root.template, self.root.region,
                             original['limits']['templates_per_body'])
        self.steps = {(key, step.id): step for key, step in steps}
        self.operations = {(key, step.id): self.coordinates[key, step.id.wire] for key, step in steps}
        _need([self.operations[key, step.id] for key, step in steps] == row['operations']
              and len(edges) == len(row['edges']), 'original initializer recipe or edge inventory differs')
        self.control, self.in_data, self.out_data = defaultdict(list), defaultdict(list), defaultdict(list)
        self.port_ids, self.edge_ids = {}, {}
        for index, edge in zip(row['edges'], edges):
            published = original['edges'][index]
            for name in ('source', 'target', 'condition'):
                self._bind_port(getattr(edge, name), published[name], row['unit'])
            site = None if published['site'] is None else original['sites'][published['site']]
            value_type = None if published['value_type'] is None else original['value_types'][published['value_type']]
            _need(site == (None if edge.site is None else dict(unit=row['unit'], site=encode(edge.site)))
                  and value_type == encode(edge.value_type)
                  and published['kind'] == edge.kind.value and published['transition'] == edge.transition.value
                  and published['when'] == edge.when, 'original initializer edge differs from sealed F4')
            self.edge_ids[edge] = index
            if edge.kind is G.EdgeKind.CONTROL:
                self.control[edge.source].append(edge)
            elif edge.kind is G.EdgeKind.DATA:
                self.in_data[edge.target].append(edge)
                self.out_data[edge.source].append(edge)
        self.stacks = defaultdict(list)
        states = _control(self.control, ((G.Port(self.root.template, self.root.region, G.PortKind.ENTER), ()),))
        self.control_complete = states is not None and all(
            edge.transition is not G.Transition.RESUME for edges in self.control.values() for edge in edges)
        for port, stack in states or ():
            if port.kind is G.PortKind.ENTER:
                self.stacks[port].append(stack)
        # A typed constant fact always refers to a specific original result port.
        self.constants = defaultdict(list)
        self.slicer = ReferenceSlicer(source.material.catalog)
        for index, fact in enumerate(source.facts.constants):
            self.constants[source_port(source.facts, fact.value).port].append((index, fact.literal))

    def _bind_port(self, port, index, unit):
        if port is None:
            _need(index is None, 'original edge guard changed')
            return
        _need(type(index) is int and 0 <= index < len(self.original['ports']), 'original edge lost a typed endpoint')
        if port in self.port_ids:
            _need(index == self.port_ids[port], 'multiple original ordinals for one port')
            return
        row = self.original['ports'][index]
        owner = self.original['owners'][row['owner']]
        template = self.original['templates'][owner['template']]
        _need(owner['original'] == port.owner.wire and template == dict(unit=unit, template=encode(port.template))
              and all(row[field] == encode(getattr(port, field))
                      for field in ('kind', 'ordinal', 'completion', 'target', 'path')), 'original typed port differs')
        self.port_ids[port] = index

    def _transport(self, operand, stack):
        """Follow one unambiguous existing data route backwards in its recipe context."""
        route, seen = [], set()
        while (operand, stack) not in seen and len(route) < 256:
            seen.add((operand, stack))
            choices = []
            for edge in self.in_data[operand]:
                if edge.transition is G.Transition.RETURN:
                    previous = (*stack, edge.site)
                elif edge.transition is G.Transition.CALL:
                    if not stack or stack[-1] != edge.site:
                        continue
                    previous = stack[:-1]
                elif edge.transition is G.Transition.DIRECT and edge.site is None:
                    previous = stack
                else:
                    return None
                choices.append((edge, previous))
            if not choices:
                # A filtered-out CALL cannot turn into an input with no producer.
                return (operand, tuple(route)) if not self.in_data[operand] else None
            if len(choices) != 1:
                return None
            edge, stack = choices[0]
            route.append(self.edge_ids[edge])
            operand = edge.source
        return None

    def origin(self, operand, site):
        if not self.stacks[site]:
            return None
        routes = {self._transport(operand, stack) for stack in self.stacks[site]}
        if len(routes) == 1:
            return routes.pop()
        return None

    def payload(self, operand, site):
        route = self.origin(operand, site)
        if route is None:
            return None
        origin, edges = route
        key = origin.template, origin.owner
        literal = None
        step = self.steps.get(key)
        if step is not None and origin.kind is G.PortKind.RESULT and origin.ordinal == 0 and origin.completion is C.FALLTHROUGH:
            instruction = step.instruction
            if type(instruction) is P.DefineLiteral:
                literal = dict(value=encode(instruction.value), original_fact=None)
            elif type(instruction) is P.ConstructValue and instruction.construction is P.ConstructionKind.LITERAL:
                facts = self.constants[origin]
                if len(facts) == 1:
                    index, value = facts[0]
                    literal = dict(value=encode(value), original_fact=encode(
                        self.slicer.fact(self.source.facts, 'constants', index)))
        return dict(origin_port=self.port_ids[origin], origin_operation=self.operations.get(key),
                    edges=list(reversed(edges)), literal=literal)

    def namespace_store(self, provider, context, step):
        info = _STORES.get(step.instruction.kind)
        site = context.site(step).port
        if info is None or context.owner(step.region).owner != self.root.owner:
            return None
        ordinal, kind = info
        handle, key, value, _ = (context.value(argument).port for argument in step.arguments)
        owner = self.origin(handle, site)
        formal = G.Port(self.root.template, self.root.region, G.PortKind.PARAMETER, ordinal)
        literal_route = self.payload(key, site)
        if owner is None or owner[0] != formal or literal_route is None or literal_route['literal'] is None:
            return None
        origin = self.origin(key, site)[0]
        instruction = self.steps[origin.template, origin.owner].instruction
        if type(instruction) is not P.DefineLiteral or type(name := instruction.value.to_python()) is not str:
            return None
        receiver = EnvironmentInterface(kind, self.root.source_scope)
        location = provider._location(receiver, name)
        before, after = context.worlds(step)
        return dict(kind='WRITE', alternatives=[dict(location=encode(location), interface=provider.interface_ids.get(receiver),
                    lookup_ordinal=None, on_hit=None, on_miss=None, on_exception=None)],
                    binding_port=None, missing_outcome=None, payload=self.port_ids[value],
                    payload_route=self.payload(value, site), world_in=self.port_ids[before.port],
                    world_out=self.port_ids[after.port], namespace_owner=dict(argument=self.port_ids[handle],
                    formal=self.port_ids[formal], edges=list(reversed(owner[1])), key=literal_route))

    def binding(self, access):
        payload = None if access.payload is None else access.payload.port
        read = access.kind.value in ('READ', 'HANDLE')
        return dict(kind=access.kind.value, binding_port=access.binding_port, alternatives=encode(access.alternatives),
                    missing_outcome=access.missing_outcome, payload=self.port_ids[payload] if payload else None,
                    world_in=self.port_ids[access.world_in.port], world_out=self.port_ids[access.world_out.port],
                    payload_route=self.payload(payload, access.site.port) if payload and not read else None,
                    direct_data_consumers=[self.edge_ids[edge] for edge in self.out_data[payload]] if read else None,
                    namespace_owner=None)

    def readers(self, accesses):
        reads = defaultdict(list)
        for read_site, row in accesses:
            if row['kind'] == 'READ':
                for target in row['alternatives']:
                    reads[target['location']['wire']].append((read_site, row['operation'], target['lookup_ordinal']))
        routes = []
        for site, row in accesses:
            if row['kind'] != 'WRITE' or not self.stacks[site]:
                continue
            candidates = [(target, reads[target['location']['wire']]) for target in row['alternatives']
                          if target['location']['wire'] in reads]
            if not candidates:
                continue
            completed = G.Port(site.template, site.owner, G.PortKind.EXIT, completion=C.FALLTHROUGH)
            states = _control(self.control, ((completed, stack) for stack in self.stacks[site]))
            if states is None:
                self.control_complete = False
                continue
            reached = {port for port, _ in states}
            for target, matching in candidates:
                for read_site, operation, branch in matching:
                    if read_site in reached:
                        routes.append(dict(write=row['operation'], read=operation, location=target['location'],
                            lookup_ordinal=branch, normal_completion=self.port_ids[completed]))
        return routes


def _initializer(session, original, selected):
    pointer = selected['original']
    body = original['bodies'][int(pointer.rsplit('/', 1)[1])]
    source = session.view.unit(original['units'][body['unit']]['unit'])
    graph = _OriginalBody(original, body, source)
    provider = SourceStateProvider(session.view, source.unit)
    accesses, open_addresses = [], []
    for (template, identity), step in graph.steps.items():
        number = graph.operations[template, identity]
        rule = classify_instruction(step.instruction)
        point = source._node(template, identity)
        _need(original['operations'][number]['operation'] == point.node.wire
              and original['operations'][number]['effect_rule'] == encode(rule),
              'initializer instruction is not its original operation')
        if rule.resolution is not EffectResolution.ADDRESS_RULE:
            continue
        site = G.Port(template, identity, G.PortKind.ENTER)
        if type(step.instruction) is P.ApplyNamespaceAction:
            plan = source.material.plans((template,))[template]
            row = graph.namespace_store(provider, NamespacePlanView(source.unit, plan), step)
        else:
            records = provider.effect_members(point)['bindings']
            _need(len(records) <= 1, 'initializer operation has multiple binding records')
            row = graph.binding(records[0][1]) if records else None
        if row is None:
            open_addresses.append(number)
            continue
        row.update(operation=number)
        accesses.append((site, row))
    readers = graph.readers(accesses)
    provider.check()
    source.material.validate_parents()
    return dict(original=pointer, scope=_BODY_SCOPE, status='SCOPED_NAMESPACE_ROUTES_PROJECTED',
                binding_materials=encode(provider.materials), locations=encode(list(provider.locations.values())),
                accesses=[row for _, row in accesses], potential_reads=readers, open_address_operations=open_addresses,
                local_reader_control_search_complete=graph.control_complete)


@reuse_decoded_source_bodies()
def derive_cpu_entry_import_effects(initializer_graph_ref, *, session=None):
    """Check only retained F4 and qualified bindings. No target is executed."""
    retained = read_artifact(initializer_graph_ref)
    _need(initializer_graph_ref['schema'] == INITIALIZERS, 'original initializer graph required')
    needs_source = any(row['operation_inventory'] is not None for row in retained['initializers'])
    query = (BuildSession(Path(retained['program']['uri']).parent) if needs_source and session is None
             else nullcontext(session))
    bodies = []
    with query as active:
        initializers = verify_cpu_entry_import_initializers(initializer_graph_ref, session=active)
        if needs_source:
            original = read_artifact(initializers['entry_obligations'])
            bodies = [_initializer(active, original, row) for row in initializers['initializers']
                      if row['operation_inventory'] is not None]
    return dict(schema=SCHEMA, schema_version=1, initializer_graph=initializer_graph_ref,
                entry_obligations=initializers['entry_obligations'], program=initializers['program'],
                initializers=bodies, status='SCOPED_NAMESPACE_ROUTES_PROJECTED' if bodies else 'SOURCE_INITIALIZER_BODIES_OPEN',
                access_contract=dict(runtime_address_instance='UNKNOWN', indirect_or_external_effects='UNKNOWN',
                    outside_initializer_readers='UNKNOWN', later_lookup=_READ_SCOPE,
                    later_lookup_conditions=list(_READ_CONDITIONS), reaching_version_proven=False),
                runtime_import_effects_closed=False, whole_entry_effect_equivalence_proven=False,
                transformation_authorized=False, target_executed=False)


def verify_cpu_entry_import_effects(reference, *, session=None):
    """Cold-replay the effect and data routes from the sealed original graph."""
    saved = read_artifact(reference)
    _need(reference['schema'] == SCHEMA
          and saved == derive_cpu_entry_import_effects(saved['initializer_graph'], session=session),
          'saved initializer effects differ from original F4 replay')
    return saved

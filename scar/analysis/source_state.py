"""Source state."""

from collections import ChainMap, defaultdict
from copy import copy
from dataclasses import replace
from pathlib import Path
from scar.analysis.program_view import _digest
from scar.analysis.source_effects import classify_instruction
from scar.analysis.source_stack_routes import _bindings, anchor_name_accesses
from scar.analysis.source_name_definitions import _NameDefinitionControl, _reaching_name_definitions
from scar.analysis.source_namespace_contents import NamespacePlanView
from scar.analysis.source_references import source_port
from scar.analysis.source_value_flow import project_value_flow
from scar.analysis.source_consumer_parents import _stamp
from scar.ir.program import (
    state_ports as S,
    source_productions as P,
    normal_graph as G,
    normal_graph as F,
    normal_graph as T,
    view as V,
    model as M,
)
from scar.ir.program.source_bindings import EnvironmentInterface
from scar.ir.program.name_flow import NameDefinitionKind as D
from scar.ir.program.dependencies import (
    DependencyPoint as Point,
    DependencyEdge,
    DependencyKind as K,
    DependencyMeaning as Meaning,
    DependencyStrength as Strength,
    DependencyEvidence,
    DependencyCondition,
    DependencyFrontier,
    DependencyNeighborhood,
    DependencyEdge as Edge,
    DependencyCondition as Condition,
    DependencyEvidence as Evidence,
    DependencyNeighborhood as Neighborhood,
    DependencyFrontier as Frontier,
)
from scar.ir.program.ids import LocationID, OperationID, ValueID
from hashlib import sha256
from scar.artifacts import read_record_table as _table
from scar.ir.program.providers import ProviderArtifact
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.record_codec import decode, encode
import ast
from scar.ir.program.state_addresses import (
    AddressCoverage,
    AddressEffect,
    Element,
    InvokeStateOperand,
    ObjectField,
    ReferenceOwner,
)
from scar.ir.program.state_addresses import AccessMode
from scar.ir.program.structured import PythonProtocolKind as Kind
from scar.ir.semantics_v2 import Opcode


# Attach source binding interfaces and may versions to the common graph.
#
# No source parsing or target execution. Original C2 interfaces select symbolic
# addresses; the existing matched-control solver supplies definition choices.
# This adapter does not infer object contents from a stable reference binding.
class _AppendBuffer:
    """Reserve final ordinals while keeping one operation's rows private."""
    def __init__(self, base):self.base=base;self.added=[]
    def __len__(self):return len(self.base)+len(self.added)
    def append(self,value):self.added.append(value)
    def __getitem__(self,index):
        return self.base[index]if index<len(self.base)else self.added[index-len(self.base)]


def _publish_edges(target, staged):
    for rows in staged.out_edges.values():
        for edge in rows:
            if edge not in target.out_edges[edge.source.node]:
                target.out_edges[edge.source.node].append(edge)
                target.in_edges[edge.target.node].append(edge)


class SourceStateProvider:
    def __init__(self, view, unit):
        self.view = view; self.source = view.unit(unit)
        self.material = self.source.material; self.namespace = self.source.namespace
        self.bindings, self.declarations, self.materials = _bindings(self.material, {})
        self.interface_ids = {descriptor:index for index,descriptor in enumerate(self.bindings.interfaces)}
        self._guard = {Path(m.uri):_stamp(Path(m.uri)) for m in self.materials}
        self.locations = {}; self.versions = {}; self.accesses = []; self.requests = {}
        self.environment_values = {}
        self.in_edges = defaultdict(list); self.out_edges = defaultdict(list)
        self.tracks = []; self.track_ids = {}; self.selections = []; self.track_limits = []
        self.flows = {}; self._definition_controls = {}; self.anchors = None
        self.read_frontiers = defaultdict(list)
        self.content_provider = None
        self.content_lineage = None
        root=self.source.facts.root_scope
        if root is None or root.owner.unit!=unit:
            raise ValueError('state source requires its original root scope witness')
        self.root_scopes={root.source_scope}
        self.class_scopes={row.scope for row in self.source.facts.classes}
        if self.root_scopes & self.class_scopes or any(row.body_scope is not None and
                (row.body_scope.source_scope!=row.scope or row.body_scope.owner.unit!=unit
                    or row.body_scope.template!=row.template or row.body_scope.region!=row.body)
                for row in self.source.facts.classes):
            raise ValueError('state root/class source scope witnesses differ')
        self.analyzed_operations=set();self._building=set();self._failures={}
        self.bindings_complete=False
        self.source.effect_provider = self
        view.register(self)

    def content(self, *, complete=True):
        """Attach contents; the old no-argument call explicitly requests all."""
        if self.content_provider is None:
            from scar.analysis.source_content import SourceContentProvider
            provider=SourceContentProvider(self,register=False)
            self.content_provider=provider
            self.view.register(provider)
            from scar.analysis.source_content_lineage import SourceContentLineageProvider
            self.content_lineage=SourceContentLineageProvider(provider)
        if complete:self.ensure_all(include_content=True)
        return self.content_provider

    def _operation_context(self, operation):
        point=operation if type(operation)is Point else Point(self.namespace,operation,None)
        self.source.ensure(point)
        step=self.source.records[point.node]
        if type(step)is not P.ProductionStep:
            raise TypeError('state request requires an original source operation')
        key=self.source.nodes[point.node].anchor.template
        context=self.source.effect_contexts.get(key)
        if context is None:
            context=NamespacePlanView(self.source.unit,self.source.plans[key])
            self.source.effect_contexts[key]=context
        if context.steps.get(step.id)!=step:
            raise ValueError('state request differs from its original plan operation')
        return point,context,step

    def ensure_operation(self, operation, *, include_content=False):
        """Identify one operation once; newly admitted plans need no sweep."""
        point,context,step=self._operation_context(operation);identity=point.node
        if identity in self._building:return self.requests[identity]
        if identity in self._failures:raise ValueError(self._failures[identity])
        if identity not in self.analyzed_operations:
            rule=classify_instruction(step.instruction)
            pending=('CALLBACK_OBJECT_CONTENT_ALIAS_RNG_EXTERNAL_AND_ORDERING_EFFECTS_NOT_CLOSED',)
            self.requests[identity]=S.SourceEffectRequest(identity,rule,(),
                ('SOURCE_STATE_EFFECT_REQUEST_PENDING',),pending)
            self._building.add(identity)
            try:
                staged=copy(self);staged.locations=ChainMap({},self.locations)
                staged.accesses=_AppendBuffer(self.accesses)
                staged.environment_values={};staged.in_edges=defaultdict(list);staged.out_edges=defaultdict(list)
                accesses=staged._identify(context,step,identity);projection=None
                if type(step.instruction)is P.ReadClassTypeParameters:
                    from scar.analysis.class_bindings import identify_class_type_parameters, attach_class_type_parameters
                    projection=identify_class_type_parameters(self.source,context,step,identity)
                    staged.environment_values[identity]=projection
                    attach_class_type_parameters(staged,projection)
                missing=(()if accesses or projection is not None else
                    ('SOURCE_CONTENT_EFFECT_REQUEST_NOT_ANALYZED',))if rule.resolution.value=='ADDRESS_RULE'else ()
                request=S.SourceEffectRequest(identity,rule,accesses,missing,pending)
                self.check(point)
                self.locations.update(staged.locations.maps[0]);self.accesses.extend(staged.accesses.added)
                self.environment_values.update(staged.environment_values);_publish_edges(self,staged)
                self.requests[identity]=request;self.analyzed_operations.add(identity)
            except Exception as error:
                self.requests[identity]=replace(self.requests[identity],missing_rules=('SOURCE_STATE_EFFECT_ANALYSIS_FAILED',))
                self._failures[identity]='state operation identification failed: '+str(error)
                raise
            finally:self._building.remove(identity)
        if include_content:self.content(complete=False).ensure_operation(point)
        return self.requests[identity]

    def ensure_regions(self, key, regions, *, include_content=False):
        """Identify direct steps of selected original regions, not child code."""
        regions=tuple(dict.fromkeys(regions));self.source._expand_regions(key,regions)
        index=self.source.region_indexes[key]
        requests=[]
        for region in regions:
            for step in index._emitter.regions[region].steps:
                point=self.source._node(key,step)
                requests.append(self.ensure_operation(point,include_content=include_content))
        return tuple(requests)

    def ensure_all(self, *, include_content=True):
        """Explicit complete unit projection, preserving the former full path."""
        self.check()
        if self.bindings_complete and (not include_content or
                self.content_provider is not None and self.content_provider.complete):return self
        keys=tuple(self.material.interfaces['nodes']);self.source._admit(keys)
        for key in keys:
            self.ensure_regions(key,(region.id for region in self.source.plans[key].regions),
                include_content=include_content)
        self.check();self.bindings_complete=True
        if include_content:self.content(complete=False).complete=True
        return self

    def _name(self, name):
        return self.declarations.names[name.index].value if name.kind == 'SOURCE' else name.symbol

    def _location(self, descriptor, key):
        # Lexical interfaces refer to the same original slot even when reached
        # through different environment descriptors. A frame/cell is still a
        # parameter, not a physical identity inferred from the source slot.
        if descriptor.kind in ('LEXICAL_BINDING', 'CLASS_PROTOCOL_CELL'):
            slot = self.bindings.bindings[descriptor.binding]
            if self._name(slot.name) != key:
                raise ValueError('state key differs from original lexical binding')
            descriptor = EnvironmentInterface('LEXICAL_BINDING', slot.owner, descriptor.binding)
        identity = LocationID(_digest(('scar.source-binding-location.v1', self.source.basis, descriptor, key)))
        self.locations.setdefault(identity, S.BindingLocationTemplate(identity, self.material.unit, descriptor, key))
        return identity

    def _alternative(self, interface, key, ordinal=None, lookup=None):
        descriptor = self.bindings.interfaces[interface]
        return S.BindingAccessAlternative(self._location(descriptor, key), interface, ordinal,
            None if lookup is None else lookup.on_hit,
            None if lookup is None else lookup.on_miss,
            None if lookup is None else lookup.on_exception)

    def _identify(self, context, step, operation):
        ins = step.instruction; kind = None; index = None; alternatives = (); missing = None
        payload = None; owner_handles = ()
        if type(ins) in (P.AccessBinding, P.DeleteExceptionBinding, P.BindTypeParameter):
            index = ins.parameter.binding_port if type(ins) is P.BindTypeParameter else ins.port
            port = self.bindings.ports[index]
            if port.name is None: raise ValueError('binding access lacks its original name')
            key = self._name(port.name)
            kind = (S.BindingAccessKind.DELETE if type(ins) is P.DeleteExceptionBinding else
                S.BindingAccessKind.WRITE if type(ins) is P.BindTypeParameter else S.BindingAccessKind(ins.action.value))
            if kind is S.BindingAccessKind.READ:
                if port.read_plan is None: raise ValueError('read lacks original lookup plan')
                plan = self.bindings.plans[port.read_plan]
                alternatives = tuple(self._alternative(s.interface, key, i, s) for i,s in enumerate(plan.steps))
                missing = plan.missing_outcome; payload = context.result(step)
            else:
                # Exception-handler cleanup uses the original Store port under
                # CLEAR_ON_HANDLER_EXIT; do not manufacture a source Del port.
                if type(ins) is P.BindTypeParameter:
                    if port.declref is None or port.resolution is None:
                        raise ValueError('type parameter lacks its original declaration/resolution')
                    contribution = self.declarations.contributions[port.declref]
                    resolution = self.bindings.resolutions[port.resolution]
                    binding = resolution.binding
                    if (contribution.port != index or 'TYPE_PARAMETER_INTERFACE' not in contribution.operations
                            or binding is None or self.bindings.bindings[binding].domain != 'TYPE_PARAMETER_BINDING'
                            or self.bindings.bindings[binding].name != port.name or resolution.name != port.name):
                        raise ValueError('type parameter is not an original binding admission')
                    slot = self.bindings.bindings[binding]
                    descriptor = EnvironmentInterface('LEXICAL_BINDING', slot.owner, binding)
                    alternatives = (S.BindingAccessAlternative(self._location(descriptor, key), self.interface_ids.get(descriptor), None),)
                else:
                    target = port.write_target if kind is S.BindingAccessKind.WRITE or type(ins) is P.DeleteExceptionBinding else port.delete_target
                    if target is None: raise ValueError('binding update lacks original target')
                    alternatives = (self._alternative(target, key),)
                if kind is S.BindingAccessKind.WRITE: payload = context.value(step.arguments[0])
        elif type(ins) is P.AccessClassProtocol:
            slot = self.bindings.bindings[ins.binding]
            if ins.binding not in ins.interface.protocol_bindings:
                raise ValueError('class protocol access is outside its declared interface')
            descriptor = EnvironmentInterface('CLASS_PROTOCOL_CELL', slot.owner, ins.binding)
            alternatives = (S.BindingAccessAlternative(self._location(descriptor, self._name(slot.name)), self.interface_ids.get(descriptor), None),)
            kind = S.BindingAccessKind.HANDLE if ins.action is P.ClassProtocolActionKind.CELL_REFERENCE else S.BindingAccessKind.WRITE
            payload = context.result(step) if kind is S.BindingAccessKind.HANDLE else context.value(step.arguments[1])
        else:
            from scar.analysis.source_namespace_contents import identify_implicit_namespace
            from scar.analysis.annotation_state import identify_annotation_namespace
            implicit = identify_annotation_namespace(self.bindings,self.declarations,
                self.root_scopes,self.class_scopes,context,step)
            if implicit is None:
                implicit = identify_implicit_namespace(self.bindings, self.declarations, context, step)
            if implicit is not None:
                kind = implicit.kind; payload = implicit.payload; missing = implicit.missing_outcome
                owner_handles = implicit.owner_handles
                read = kind is S.BindingAccessKind.READ
                alternatives = tuple(S.BindingAccessAlternative(self._location(descriptor, implicit.key),
                    self.interface_ids.get(descriptor), ordinal if read else None,
                    'RETURN_VALUE' if read else None,
                    ('NEXT' if ordinal+1<len(implicit.descriptors) else missing) if read else None,
                    'PROPAGATE' if read else None)
                    for ordinal,descriptor in enumerate(implicit.descriptors))
        if kind is None: return ()
        before, after = context.worlds(step)
        record = S.BindingAccessTemplate(operation, context.site(step), context.owner(step.region).owner,
            kind, index, alternatives, payload, before, after, missing)
        ordinal = len(self.accesses); self.accesses.append(record)
        op = Point(self.namespace, operation)
        for branch, alternative in enumerate(alternatives):
            location = Point(self.namespace, alternative.location)
            condition = DependencyCondition('scar.source-state.address', 'SELECTED_ACTIVATION_AND_LOOKUP_BRANCH',
                op, branch, (location,))
            read = kind in (S.BindingAccessKind.READ, S.BindingAccessKind.HANDLE)
            self._edge(location if read else op, op if read else location,
                K.ENVIRONMENT if kind is S.BindingAccessKind.HANDLE else K.STATE_INPUT if read else K.STATE_OUTPUT,
                Meaning.STATE_INTERFACE, (condition,), 'scar.source-state.binding-interface.v1')
            if owner_handles and owner_handles[branch] is not None:
                handle = self.view.source_port(owner_handles[branch])
                selected = DependencyCondition('scar.source-state.address-owner',
                    self.locations[alternative.location].interface.kind, op, branch, (handle, location))
                self._edge(handle, location, K.ENVIRONMENT, Meaning.STATE_INTERFACE, (selected,),
                    'scar.source-state.explicit-namespace-owner.v1')
        return (ordinal,)

    def _edge(self, source, target, kind, meaning, conditions=(), rule='scar.source-state.binding-version.v1', strength=Strength.EXACT):
        evidence = DependencyEvidence(rule, (self.source.locator,), (source, target))
        edge = DependencyEdge(source, target, kind, meaning, (evidence,), tuple(conditions), strength)
        if edge not in self.out_edges[source.node]:
            self.out_edges[source.node].append(edge); self.in_edges[target.node].append(edge)
        return edge

    def track(self, source_function, namespace, **limits):
        """Join the original may definitions; no static last-writer shortcut.

        Only a function's own lexical slot can use this intraprocedural solver.
        Captured/mapping/object-content state remains an explicit interface.
        """
        self.check()
        if (type(namespace) is not F.Namespace or namespace.kind is not F.NamespaceKind.LEXICAL
                or type(namespace.owner) is not int or not 0 <= namespace.owner < len(self.bindings.bindings)):
            raise ValueError('binding track requires a complete original lexical slot')
        if type(source_function) is not T.GlobalPort:
            raise ValueError('binding track source is not a unique original function')
        unit = self.source.facts
        control = self._definition_controls.get(source_function)
        if control is None:
            matches = [(i, f) for i, f in enumerate(unit.functions) if source_port(unit, f.value) == source_function]
            if len(matches) != 1 or matches[0][1].body_scope is None:
                raise ValueError('binding track source is not a unique original function')
            function_ordinal, function = matches[0]
        else:
            function_ordinal = control.function_ordinal
            function = unit.functions[function_ordinal]
        slot = self.bindings.bindings[namespace.owner]
        if (slot.domain != 'LOCAL_BINDING' or slot.owner != function.environment.scope
                or self._name(slot.name) != namespace.name):
            raise ValueError('captured/nonlocal/name-mismatched slot needs its selected environment interface')
        key = source_function, namespace
        limits = dict(max_states=limits.get('max_states', 20000), max_stack=limits.get('max_stack', 128),
            max_updates=limits.get('max_updates', 1000000), **{k:v for k,v in limits.items()
                if k not in ('max_states', 'max_stack', 'max_updates')})
        if key in self.track_ids:
            index = self.track_ids[key]
            if limits != self.track_limits[index]: raise ValueError('cached binding track has different traversal limits')
            return self.tracks[index]
        if self.anchors is None:
            self.anchors = anchor_name_accesses(self.view.catalog, self.material.unit, material=self.material)
        flow = self.flows.get(source_function)
        if flow is None:
            body = T.Body(self.material.unit, function.template, function.body)
            flow = project_value_flow(self.material, source_function, body)
            self.flows[source_function] = flow
        if control is None or control.flow is not flow:
            control = _NameDefinitionControl(self.material, unit, function_ordinal, flow)
            self._definition_controls[source_function] = control
        graph = _reaching_name_definitions(self.view.catalog, source_function, namespace,
            anchor_graph=self.anchors, material=self.material, _control=control, **limits)
        ordinal = len(self.tracks); self.track_ids[key] = ordinal; self.tracks.append(graph); self.track_limits.append(limits)
        location = self._location(EnvironmentInterface('LEXICAL_BINDING', slot.owner, namespace.owner), namespace.name)
        address = Point(self.namespace, location)
        by_definition = {}
        for index, definition in enumerate(graph.definitions):
            identity = ValueID(_digest(('scar.source-binding-version.v1', self.source.basis, source_function,
                namespace, index, definition)))
            bound = (S.BindingBoundness.BOUND if definition.kind in (D.FORMAL, D.WRITE) else
                S.BindingBoundness.UNBOUND if definition.kind is D.DELETE else S.BindingBoundness.UNKNOWN)
            version = S.BindingVersionTemplate(identity, location, source_function, namespace,
                index, definition.kind, bound, definition.value)
            self.versions[identity] = version
            by_definition[index] = identity
            value = Point(self.namespace, identity)
            self._edge(address, value, K.STATE_LOCATION, Meaning.STRUCTURE)
            site = self.view.source_port(T.GlobalPort(self.material.unit, definition.site))
            selected = DependencyCondition('scar.source-state.version', 'SAME_ACTIVATION_AND_DEFINITION_OCCURRENCE',
                site, index, (value, address))
            self._edge(site, value, K.STATE_OUTPUT, Meaning.STATE_INTERFACE, (selected,))
            if definition.value is not None:
                payload = self.view.source_port(definition.value)
                self._edge(payload, value, K.BINDING_PAYLOAD, Meaning.BINDING, (selected,))
        frontier_reasons = tuple(frontier.reason for frontier in graph.frontiers)
        function_points = (self.view.source_port(source_function),) if graph.reads else ()
        for index, read in enumerate(graph.reads):
            access = graph.anchor_graph.accesses[read.access]
            self.view.source_port(access.site)
            point=self.source._node(access.site.port.template,access.site.port.owner)
            self.ensure_operation(point,include_content=False)
            op=point.node
            target = Point(self.namespace, op)
            versions = tuple(by_definition[n] for n in read.definitions)
            self.selections.append(S.BindingReadSelection(ordinal, index, read.access, read.state, versions))
            self.read_frontiers[op].extend(frontier_reasons)
            condition = DependencyCondition('scar.source-state.read', 'MATCHED_RECIPE_STATE_AND_FRAME_SELECTED',
                target, index, function_points)
            for identity in versions:
                self._edge(Point(self.namespace, identity), target, K.BINDING_READ, Meaning.BINDING,
                    (condition,), strength=Strength.MAY)
        return graph

    def effects(self, operation):
        return self.ensure_operation(operation,include_content=False)

    def effect_members(self, operation):
        """Expose the exact access rows referenced by a field-level summary."""
        request = self.effects(operation)
        identity = request.operation
        content = self.content_provider
        return dict(bindings=tuple((n, self.accesses[n]) for n in request.accesses),
            contents=() if content is None else tuple((n, content.accesses[n])
                for n in content.operation_accesses.get(identity, ())))

    def check(self, point=None):
        self.source.check(point)
        if any(_stamp(path) != stamp for path,stamp in self._guard.items()):
            raise ValueError('admitted original binding/declaration material changed')

    def describe(self, point):
        if point.namespace != self.namespace: return None
        self.check(point)
        if point.node in self.locations:
            return V.ProgramNode(point, V.ProgramEntityKind.STATE_LOCATION, V.ProgramNodeRole.STATE_LOCATION, None, None)
        if point.node in self.versions:
            version = self.versions[point.node]
            return V.ProgramNode(point, V.ProgramEntityKind.VALUE, V.ProgramNodeRole.CORE_VALUE, None,
                M.ValueType(M.ValueKind.STATE, location=version.location))
        return None

    def original(self, point):
        if self.describe(point) is None: raise ValueError('state endpoint is not admitted')
        return self.locations[point.node] if point.node in self.locations else self.versions[point.node]

    def expand(self, point):
        self.original(point)
        if point.node not in self.locations: return ()
        return tuple(Point(self.namespace, v.id, point.context) for v in self.versions.values() if v.location == point.node)

    def explain(self, point):
        return dict(node=self.describe(point), original=self.original(point),
            source_material=self.material.result_artifact, c2_materials=self.materials,
            binding_versions_not_object_or_tensor_content=True, runtime_instance_selected=False)

    def _neighbors(self, point, incoming):
        if point.namespace != self.namespace: return DependencyNeighborhood()
        self.check(point)
        if type(point.node)is OperationID and point.node in self.source.nodes:
            self.ensure_operation(point,include_content=False)
        rows = self.in_edges[point.node] if incoming else self.out_edges[point.node]
        edges = tuple(replace(edge, source=replace(edge.source, context=point.context),
            target=replace(edge.target, context=point.context), conditions=tuple(replace(c,
                subject=None if c.subject is None else replace(c.subject, context=point.context),
                operands=tuple(replace(p, context=point.context) for p in c.operands)) for c in edge.conditions)) for edge in rows)
        frontiers = []
        if not self.bindings_complete and (type(point.node)is LocationID or point.node in self.versions
                or point.node in self.source.nodes and type(point.node)is not OperationID):
            frontiers.append(DependencyFrontier(point,'SOURCE_STATE_ADJACENCY_PARTIAL',
                'Only requested operations have binding accesses; ensure_all() explicitly projects the remaining original unit.'))
        if point.node in self.locations or point.node in self.versions:
            frontiers.append(DependencyFrontier(point, 'STATE_INSTANCE_AND_OBJECT_CONTENT_NOT_SELECTED',
                'Static binding versions require a frame/occurrence and do not identify tensor contents.'))
        for reason in dict.fromkeys(self.read_frontiers.get(point.node, ())):
            frontiers.append(DependencyFrontier(point, 'SOURCE_BINDING_'+reason,
                'Original matched-control definition analysis retains this unresolved alternative.'))
        return DependencyNeighborhood(edges, tuple(dict.fromkeys(frontiers)))

    def incoming(self, point): return self._neighbors(point, True)
    def outgoing(self, point): return self._neighbors(point, False)


# State materialization.
# Select local binding analyses from original access/owner types, not names.
def local_read_tracks(state):
    """Return every function/slot pair this intraprocedural model can track.

    A capture is still an address interface: it is not silently analyzed as
    a slot of the reading activation. Different function creations can select
    the same body, so all original creation endpoints are retained.
    """
    # This API explicitly enumerates the whole unit, unlike point queries.
    state.ensure_all(include_content=False)
    functions = {}
    for function in state.source.facts.functions:
        if function.body_scope is not None:
            functions.setdefault((function.body_scope.owner, function.environment.scope), []).append(function)
    selected = {}
    for access in state.accesses:
        if access.kind is not S.BindingAccessKind.READ: continue
        for alternative in access.alternatives:
            descriptor = state.locations[alternative.location].interface
            if descriptor.kind != 'LEXICAL_BINDING': continue
            slot = state.bindings.bindings[descriptor.binding]
            if slot.domain != 'LOCAL_BINDING': continue
            namespace = F.Namespace(F.NamespaceKind.LEXICAL, descriptor.binding, state._name(slot.name))
            for function in functions.get((access.scope, slot.owner), ()):
                endpoint = source_port(state.source.facts, function.value)
                selected.setdefault((endpoint, namespace), None)
    return tuple(selected)


def track_local_reads(state, **limits):
    """Solve each selected pair once, sharing the provider's anchors/bodies."""
    selected = local_read_tracks(state)
    for function, namespace in selected: state.track(function, namespace, **limits)
    return selected


# Typed readback of additional binding-version relations, without solving.
#
# The enclosing observer admits producer/input receipts and actual body choices.
# This unit reader checks saved table bytes, types and endpoint closure against
# the original source projection. Independent source rederivation remains the
# separate state-view reader; it is not repeated for graph navigation.
TABLES=frozenset(('rules','requests','locations','accesses','versions','selections','tracks','edges',
    'content_addresses','content_states','content_accesses','content_edges'))
LINEAGE_TABLES=frozenset(('content_lineage','content_lineage_edges'))


class StoredBindingVersions:
    def __init__(self,view,folder,header,summary):
        self.folder=Path(folder);self.header=header;self.summary=summary
        self.source=view.project_unit(header['unit']);self.namespace=self.source.namespace
        expected_tables=TABLES
        if header['schema_version']==2:
            from scar.analysis.program_state_view import ENCODING,compatible_storage_pins
            if (header.get('track_encoding') != ENCODING
                    or not compatible_storage_pins(header.get('storage_pins'))):
                raise ValueError('stored binding track encoding differs')
            expected_tables=TABLES|{'controls'}
        elif header['schema_version']!=1:raise ValueError('unsupported stored binding schema')
        if (encode(self.namespace)!=header['namespace']
                or encode(self.source.material.result_artifact)!=header['source_material']
                or set(header['tables']) not in (expected_tables,expected_tables|LINEAGE_TABLES)
                or header['original_body_bytes_copied']!=0
                or any(header[k]is not False for k in ('loops_unrolled','target_executed',
                    'tensor_content_versions_proven','effect_membership_closed','first_stage_complete'))):
            raise ValueError('stored binding source domain, table grammar or authority differs')
        self.guards={};self.locations={};self.versions={};self.in_edges=defaultdict(list);self.out_edges=defaultdict(list)
        for artifact in header['binding_materials']:
            p=Path(artifact['uri']);raw=p.read_bytes()
            if len(raw)!=artifact['size_bytes'] or sha256(raw).hexdigest()!=artifact['sha256']:
                raise ValueError('stored binding original C2 material changed')
            self.guards[p]=_stamp(p)
        # Check every saved table, including proof records not needed for this
        # neighborhood. Hash compressed bytes incrementally; do not decode the
        # full per-variable control histories just to follow a version edge.
        for name,receipt in header['tables'].items():
            if receipt['path']!=name+'.jsonl.gz':raise ValueError('stored binding table path differs')
            p=self.folder/receipt['path'];before=_stamp(p);digest=sha256();size=0
            with p.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block);size+=len(block)
            if size!=receipt['bytes'] or digest.hexdigest()!=receipt['sha256'] or _stamp(p)!=before:
                raise ValueError('stored binding table receipt changed')
            self.guards[p]=before
        for name,cls,target in (('locations',S.BindingLocationTemplate,self.locations),
                ('versions',S.BindingVersionTemplate,self.versions)):
            for wire in _table(self.folder,header['tables'][name]):
                row=decode(cls,wire,record_tags=P.PRODUCTION_RECORD_TAGS)
                if row.id in target:raise ValueError('duplicate stored binding identity')
                target[row.id]=row
            if len(target)!=header['counts'][name]:raise ValueError('stored binding node denominator differs')
        for row in self.versions.values():
            bound=(S.BindingBoundness.BOUND if row.kind in (D.FORMAL,D.WRITE) else
                S.BindingBoundness.UNBOUND if row.kind is D.DELETE else S.BindingBoundness.UNKNOWN)
            if row.location not in self.locations or row.source_function.unit!=header['unit'] or row.boundness is not bound:
                raise ValueError('stored binding version location/function/boundness differs')
        raw=(self.folder/'result.json').read_bytes()
        artifact=ProviderArtifact(str(self.folder/'result.json'),sha256(raw).hexdigest(),len(raw),'scar.source-state-view')
        self.locator=SourceArtifactLocator(artifact,artifact.sha256,artifact.size_bytes,'canonical-json')
        point=lambda identity:Point(self.namespace,identity,None)
        known=lambda identity:identity in self.source.nodes or identity in self.locations or identity in self.versions
        count=0;self.projected_relations=0
        for wire in _table(self.folder,header['tables']['edges']):
            row=decode(S.StoredBindingEdge,wire,record_tags=P.PRODUCTION_RECORD_TAGS);count+=1
            if not all(known(n) for n in (row.source,row.target,
                    *(c.subject for c in row.conditions if c.subject is not None),
                    *(n for c in row.conditions for n in c.operands))):
                raise ValueError('stored binding edge has an undeclared source/state endpoint')
            if row.source not in self.versions and row.target not in self.versions:continue
            conditions=tuple(Condition(c.domain,c.name,None if c.subject is None else point(c.subject),
                c.index,tuple(point(n) for n in c.operands)) for c in row.conditions)
            edge=Edge(point(row.source),point(row.target),row.kind,row.meaning,
                (Evidence(row.rule,(self.locator,),(point(row.source),point(row.target))),),conditions,row.strength)
            self.in_edges[row.target].append(edge);self.out_edges[row.source].append(edge);self.projected_relations+=1
        if count!=header['counts']['relations']:raise ValueError('stored binding edge denominator differs')
        self.check()

    def check(self,point=None):
        self.source.check(point)
        if any(_stamp(p)!=stamp for p,stamp in self.guards.items()):
            raise ValueError('admitted stored binding unit changed')

    def describe(self,point):
        if point.namespace!=self.namespace:return None
        self.check(point)
        if point.node in self.locations:
            return V.ProgramNode(point,V.ProgramEntityKind.STATE_LOCATION,V.ProgramNodeRole.STATE_LOCATION,None,None)
        if point.node in self.versions:
            row=self.versions[point.node]
            return V.ProgramNode(point,V.ProgramEntityKind.VALUE,V.ProgramNodeRole.CORE_VALUE,None,
                M.ValueType(M.ValueKind.STATE,location=row.location))
        return None

    def original(self,point):
        if self.describe(point)is None:raise ValueError('point is not a stored binding state node')
        return self.locations.get(point.node) or self.versions[point.node]

    def expand(self,point):
        self.original(point)
        return tuple(Point(self.namespace,row.id,point.context) for row in self.versions.values() if row.location==point.node)

    def explain(self,point):
        return dict(node=self.describe(point),original=self.original(point),stored_source=self.locator,
            source_rederived_now=False,admission='PINNED_TYPED_STORED_STATE_WITH_ENDPOINT_CLOSURE',
            binding_versions_not_tensor_contents=True,runtime_instance_selected=False,
            track_summary=self.summary,independent_replay='scar state-view --read '+str(self.folder))

    def _neighbors(self,point,incoming):
        if point.namespace!=self.namespace:return Neighborhood()
        self.check(point);rows=(self.in_edges if incoming else self.out_edges).get(point.node,())
        edges=tuple(replace(e,source=replace(e.source,context=point.context),target=replace(e.target,context=point.context),
            conditions=tuple(replace(c,subject=None if c.subject is None else replace(c.subject,context=point.context),
                operands=tuple(replace(p,context=point.context) for p in c.operands)) for c in e.conditions)) for e in rows)
        frontiers=[]
        if point.node in self.locations or point.node in self.versions:
            frontiers.append(Frontier(point,'STATE_INSTANCE_AND_OBJECT_CONTENT_NOT_SELECTED',
                'Static binding versions require frame/occurrence selection and are not tensor contents.'))
        if rows and self.summary['budget_or_control_incomplete']:
            frontiers.append(Frontier(point,'SOURCE_BINDING_TRACK_COVERAGE_REQUIRES_SELECTION',
                'Some tracks in this unit are open; saved track/selection records qualify each read.'))
        return Neighborhood(edges,tuple(frontiers))

    def incoming(self,point):return self._neighbors(point,True)
    def outgoing(self,point):return self._neighbors(point,False)


# Additional source/owner checks for the two heap store protocols.
#
# Core validation owns use typing/CFG; the environment validator owns the
# normal/raise state signature. Exact source evaluation is checked by replay.
def validate_heap_store(operation, kind, link, row, semantic):
    definition = semantic.definitions.get(link.semantic_operation)
    atom = semantic.source_atoms.get(link.source.atom_id)
    attribute = kind is Kind.STORE_ATTRIBUTE
    expected_kind = "Attribute" if attribute else "Subscript"
    if (definition is None or atom is None or atom.kind != expected_kind
            or definition.metadata.get("role") != "binding"
            or definition.code_id != link.source.atom_id.wire
            or link.opcode is not Opcode.OPAQUE
            or not definition.label.startswith("bind ")):
        raise ValueError("heap store must reference its exact SG assignment target")
    target = ast.parse(definition.label[5:], mode="eval").body
    if attribute:
        if not isinstance(target, ast.Attribute) or len(operation.uses) != 3:
            raise ValueError("attribute store needs its owner, assigned reference and world")
        selector = ObjectField(target.attr)
    else:
        if (not isinstance(target, ast.Subscript) or isinstance(target.slice, ast.Slice)
                or len(operation.uses) != 4):
            raise ValueError("subscript store needs its owner, key, assigned reference and world")
        selector = Element(operation.uses[1].id)
    effect = AddressEffect(ReferenceOwner(operation.uses[0].id), (selector,), AccessMode.WRITE)
    if (row is None or row.site != InvokeStateOperand(operation.id, operation.uses[-1].id)
            or row.coverage is not AddressCoverage.PARTIAL or row.protocol is not None
            or row.effects != (effect,)):
        raise ValueError("heap store address must retain its exact owner/key/world Uses")

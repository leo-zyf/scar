"""Conditional local-READ effects from original F4/C2 witnesses.

This is a separate projection, not a purity flag or a replacement for strict
binding definitions. No obligation is discharged by choosing this analysis.
Writes, releases, calls, operators and resumptions retain their strict effects.
"""
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from scar.analysis.source_consumer_parents import _stamp
from scar.analysis.source_name_access_anchors import _bindings, anchor_name_accesses
from scar.analysis.source_reference_slice import ReferenceSlicer, source_port
from scar.analysis.source_value_flow import project_value_flow
from scar.ir.program import normal_graph as G, normal_graph as F, normal_graph as T
from scar.ir.program import source_productions as P, name_flow as D
from scar.ir.program import name_flow as A, source_bindings as B
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference


MODE = 'EXACT_LOCAL_READ_EFFECTS_UNDER_UNDISCHARGED_FRAME_Q_V1'
REQUIRED_Q = (
    'Q_SELECTED_C2_LOCAL_INTERFACE_IS_ORDINARY_CURRENT_FUNCTION_FRAME',
    'Q_NO_AMBIENT_FRAME_INTROSPECTION_DEBUGGER_OR_FOREIGN_FRAME_WRITES',
    'Q_NO_TRACE_PROFILE_MONITORING_CALLBACK_FRAME_WRITES',
    'Q_NO_SIGNAL_ASYNC_REENTRY_OR_CONCURRENT_SELECTED_FRAME_WRITES',
    'Q_LOCAL_READ_REFERENCE_TRANSFER_HAS_NO_RELEASE_FINALIZER_CALLBACK',
)


@dataclass(frozen=True, slots=True)
class LocalReadEffectWitness:
    operation: int
    site: T.GlobalPort
    source_function: T.GlobalPort
    scope: F.CodeScopeRef
    binding: int
    binding_port: int
    read_plan: int
    interface: int
    name_fact: ProviderFactReference
    materials: tuple[ProviderArtifact, ...]
    world_in: T.GlobalPort
    world_out: T.GlobalPort
    rule: str = 'EXACT_CURRENT_NONCAPTURE_LOCAL_READ_DOES_NOT_WRITE_SELECTED_FRAME_SLOT'
    conditions: tuple[str, ...] = REQUIRED_Q


@dataclass(frozen=True, slots=True)
class ConditionalBindingProjection:
    graph: D.NameDefinitionGraph
    witnesses: tuple[LocalReadEffectWitness, ...]
    analysis_sources: tuple[ProviderArtifact, ...]
    obligations: tuple[str, ...] = field(default=REQUIRED_Q, init=False)
    mode: str = field(default=MODE, init=False)
    obligations_discharged: bool = field(default=False, init=False)
    tensor_content_versions_proven: bool = field(default=False, init=False)
    runtime_activation_selected: bool = field(default=False, init=False)


class LocalReadEffectContext:
    """Receipt-bound C2 context reusable within one owning material session."""
    def __init__(self, material):
        remembered = {}
        self.bindings, self.declarations, self.materials = _bindings(material, remembered)
        self.material = material
        self.unit = material.fact_store.load(material.unit)
        slicer = ReferenceSlicer(SimpleNamespace(fact_store=material.fact_store))
        self.name_facts = {slicer.fact(self.unit, 'names', i): row for i, row in enumerate(self.unit.names)}
        self._cache = {}
        self.guards = {path: _stamp(path) for path in remembered}
        implementation = (Path(__file__).resolve(), Path(__file__).with_name('source_name_definitions.py').resolve())
        self.analysis_sources = tuple(ProviderArtifact(str(path), sha256(path.read_bytes()).hexdigest(),
            path.stat().st_size, 'scar.conditional-binding-effect-rule-source') for path in implementation)
        self.guards.update((path, _stamp(path)) for path in implementation)
        self.check()

    def check(self):
        self.material.validate_parents()
        if any(_stamp(path) != stamp for path, stamp in self.guards.items()):
            raise ValueError('conditional local-read original C2 material changed')

    def witnesses(self, material, anchors, function, scope, flow):
        self.check()
        if (material is not self.material or type(anchors) is not A.NameAccessAnchorGraph
                or anchors.unit != material.unit or scope not in anchors.scopes
                or not set(self.materials) <= set(anchors.materials)):
            raise ValueError('local-read effect context differs from original material/anchors')
        # A negative capture claim needs the complete C2 demand domain. A
        # cell_required=False flag alone says nothing about compiler layout.
        bindings = self.bindings
        if (type(bindings) is not B.SourceBindingsMaterial or bindings.status != 'RESOLVED'
                or bindings.frontiers or function.environment.kind is not P.DeferredBodyKind.FUNCTION):
            return ()
        cache_key = id(anchors), id(flow), scope
        if cache_key in self._cache:
            return self._cache[cache_key][2]
        captured = {c.binding for c in bindings.captures} | {r.binding for r in bindings.relays}
        operations = {op.entry: (i, op) for i, op in enumerate(flow.operations)}
        witnesses = []
        for anchor in anchors.anchors:
            if anchor.scope != scope or anchor.record.write:
                continue
            if self.name_facts.get(anchor.fact) != anchor.record:
                raise ValueError('local-read effect anchor is not its original name fact')
            access = anchors.accesses[anchor.access]
            candidate = operations.get(access.site.port)
            if candidate is None:
                continue
            ordinal, operation = candidate
            ins = operation.instruction
            if (type(ins) is not P.AccessBinding or ins.action is not P.BindingActionKind.READ
                    or access.action is not P.BindingActionKind.READ or ins.port != access.binding_port
                    or access.owner.owner != scope or access.site.unit != material.unit
                    or access.target is not None or access.read_plan is None or len(access.lookup) != 1
                    or anchor.lookup_ordinal != 0 or anchor.namespace.kind is not F.NamespaceKind.LEXICAL):
                continue
            port = bindings.ports[ins.port]
            if (port.conflicts or port.read_plan != access.read_plan or port.resolution is None
                    or port.name != access.name):
                continue
            lookup = access.lookup[0]
            if (lookup.ordinal != 0 or lookup.interface.unit != material.unit
                    or lookup.on_hit != 'RETURN_VALUE' or lookup.on_exception != 'PROPAGATE'):
                continue
            plan = bindings.plans[access.read_plan]
            interface = bindings.interfaces[lookup.interface.index]
            if (len(plan.steps) != 1 or plan.steps[0].interface != lookup.interface.index
                    or (plan.steps[0].on_hit, plan.steps[0].on_miss, plan.steps[0].on_exception)
                        != (lookup.on_hit, lookup.on_miss, lookup.on_exception)
                    or interface.kind != 'LEXICAL_BINDING' or interface.binding is None
                    or interface.owner != function.environment.scope):
                continue
            binding = interface.binding; slot = bindings.bindings[binding]
            resolution = bindings.resolutions[port.resolution]
            if (slot.domain != 'LOCAL_BINDING' or slot.owner != function.environment.scope
                    or slot.cell_required or binding in captured or slot.name != access.name
                    or resolution.binding != binding or resolution.name != access.name
                    or resolution.scope != function.environment.scope
                    or anchor.namespace.owner != binding
                    or anchor.record.namespace != anchor.namespace
                    or anchor.record.namespace.name != access.key):
                continue
            if source_port(self.unit, anchor.record.value) != access.value:
                raise ValueError('local-read effect fact differs from its original F4 output')
            witnesses.append(LocalReadEffectWitness(ordinal, access.site, flow.source_function, scope,
                binding, ins.port, access.read_plan, lookup.interface.index, anchor.fact,
                (material.result_artifact, *self.materials), access.world_in, access.world_out))
        self.check()
        result = tuple(dict.fromkeys(witnesses))
        self._cache[cache_key] = anchors, flow, result
        return result


def conditional_reaching_name_definitions(snapshot, source_function, namespace, *, anchor_graph=None,
        material=None, flow=None, max_states=20000, max_stack=128, max_updates=1000000,
        effect_context=None):
    """Derive conditional may definitions; Q remains explicit and unresolved."""
    from scar.analysis.source_graph_material import source_graph_material
    from scar.analysis.source_name_definitions import _reaching_name_definitions
    if material is None:
        material = source_graph_material(getattr(snapshot, 'catalog', snapshot), source_function.unit)
    if anchor_graph is None:
        anchor_graph = anchor_name_accesses(snapshot, source_function.unit, material=material)
    context = LocalReadEffectContext(material) if effect_context is None else effect_context
    if type(context) is not LocalReadEffectContext:
        raise TypeError('expected original LocalReadEffectContext, never a pure boolean')
    unit = material.fact_store.load(material.unit)
    function = next(f for f in unit.functions if source_port(unit, f.value) == source_function)
    if flow is None:
        flow = project_value_flow(material, source_function, T.Body(material.unit, function.template, function.body))
    graph = _reaching_name_definitions(snapshot, source_function, namespace, anchor_graph=anchor_graph,
        material=material, flow=flow, max_states=max_states, max_stack=max_stack, max_updates=max_updates,
        _local_read_effects=context)
    witnesses = context.witnesses(material, anchor_graph, function, graph.scope, flow)
    frontiers = tuple(D.NameDefinitionFrontier(None, None, None, requirement) for requirement in REQUIRED_Q)
    graph = replace(graph, conditions=(*graph.conditions, MODE, *REQUIRED_Q),
        frontiers=(*graph.frontiers, *frontiers), semantics='CONDITIONAL_LOCAL_READ_EFFECT_PROJECTION')
    return ConditionalBindingProjection(graph, witnesses, context.analysis_sources)


def conditional_local_read_tracks(state):
    """Analyze one source provider without attaching conditional data edges."""
    from scar.analysis.state_materialization import local_read_tracks
    if state.anchors is None:
        state.anchors = anchor_name_accesses(state.view.catalog, state.material.unit, material=state.material)
    context = LocalReadEffectContext(state.material)
    result = []
    for function, namespace in local_read_tracks(state):
        if function not in state.flows:
            record = next(f for f in state.source.facts.functions
                if source_port(state.source.facts, f.value) == function)
            state.flows[function] = project_value_flow(state.material, function,
                T.Body(state.material.unit, record.template, record.body))
        result.append(conditional_reaching_name_definitions(state.view.catalog, function, namespace,
            anchor_graph=state.anchors, material=state.material, flow=state.flows[function], effect_context=context))
    return tuple(result)

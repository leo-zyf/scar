"""Selected-frame binding separation, conditional on explicit isolation Q.

Only the selected lexical slot's opaque weak definitions are filtered. The
original operations, world edges, heap/content effects and strict solver stay
unchanged. This projection never grants executable transformation authority.
"""
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path

from scar.analysis.source_binding_effects import LocalReadEffectContext
from scar.analysis.source_effects import classify_instruction
from scar.analysis.source_reference_slice import source_port
from scar.analysis.source_value_flow import project_value_flow
from scar.analysis.source_world_flow import _state
from scar.ir.program import normal_graph as G, normal_links as F, normal_targets as T
from scar.ir.program import source_productions as P, source_name_definitions as D
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.program.control import LanguageCompletionKind as C


FRAME_MODE = 'SELECTED_LOCAL_FRAME_BINDING_ISOLATION_UNDER_UNDISCHARGED_Q_V1'
FRAME_REQUIRED_Q = (
    'Q_SELECTED_C2_LOCAL_IS_UNALIASED_ORDINARY_OPTIMIZED_CURRENT_FUNCTION_FRAME_SLOT',
    'Q_NO_UNMODELED_SAME_FRAME_OR_FOREIGN_FRAME_BINDING_WRITES',
    'Q_NO_FRAME_PROXY_ESCAPED_PROXY_NATIVE_FRAME_API_OR_DYNAMIC_EXEC_BINDING_WRITES',
    'Q_NO_DEBUG_TRACE_PROFILE_MONITOR_SIGNAL_ASYNC_REENTRY_OR_CONCURRENT_FRAME_BINDING_WRITES',
    'Q_ALL_SELECTED_SLOT_DIRECT_WRITES_AND_DELETES_ARE_RECORDED_ACCESS_BINDING_NORMAL_EXITS',
)


@dataclass(frozen=True, slots=True)
class ExplicitFrameBindingWrite:
    operation: int
    site: T.GlobalPort
    action: P.BindingActionKind
    access: int
    binding_port: int
    interface: int
    fact: ProviderFactReference | None


@dataclass(frozen=True, slots=True)
class FrameBindingIsolationWitness:
    source_function: T.GlobalPort
    scope: F.CodeScopeRef
    namespace: F.Namespace
    binding: int
    qualified_read_facts: tuple[ProviderFactReference, ...]
    writers: tuple[ExplicitFrameBindingWrite, ...]
    filtered_operations: tuple[int, ...]
    materials: tuple[ProviderArtifact, ...]
    rule: str = 'UNCAPTURED_SELECTED_LOCAL_SLOT_DISJOINT_FROM_CALLEE_AND_HEAP_EFFECTS_UNDER_FRAME_Q'
    conditions: tuple[str, ...] = FRAME_REQUIRED_Q


@dataclass(frozen=True, slots=True)
class FrameBindingIsolationProjection:
    graph: D.NameDefinitionGraph
    witness: FrameBindingIsolationWitness | None
    analysis_sources: tuple[ProviderArtifact, ...]
    refusal_reasons: tuple[str, ...]
    obligations: tuple[str, ...] = field(default=FRAME_REQUIRED_Q, init=False)
    mode: str = field(default=FRAME_MODE, init=False)
    obligations_discharged: bool = field(default=False, init=False)
    heap_rng_external_and_array_content_effects: str = field(default='UNCHANGED_UNKNOWN', init=False)
    tensor_content_versions_proven: bool = field(default=False, init=False)
    runtime_activation_selected: bool = field(default=False, init=False)


class FrameBindingIsolationContext:
    def __init__(self, material):
        self.local = LocalReadEffectContext(material)
        self._cache = {}; self.refusals = {}
        path = Path(__file__).resolve()
        self.analysis_sources = (*self.local.analysis_sources,
            ProviderArtifact(str(path), sha256(path.read_bytes()).hexdigest(), path.stat().st_size,
                'scar.conditional-frame-isolation-rule-source'))
        from scar.analysis.source_consumer_parents import _stamp
        self.local.guards[path] = _stamp(path)

    def witness(self, material, anchors, function, scope, flow, namespace):
        self.local.check()
        key = id(anchors), id(flow), namespace
        if key in self._cache:return self._cache[key][2]
        reasons = []
        ordinary_reads = self.local.witnesses(material, anchors, function, scope, flow)
        selected = tuple(w for w in ordinary_reads if w.binding == namespace.owner
            and self.local.name_facts[w.name_fact].namespace == namespace)
        if not selected or namespace.kind is not F.NamespaceKind.LEXICAL:
            reasons.append('SELECTED_SLOT_NOT_CERTIFIED_CURRENT_NONCAPTURE_LOCAL')
        if any(edge.transition is G.Transition.RESUME for edge in flow.edges):
            reasons.append('FRAME_ISOLATION_RESUMPTION_NOT_SUPPORTED')
        binding = namespace.owner
        bindings = self.local.bindings
        # These operations directly modify namespace interfaces and are not
        # all represented by the existing AccessBinding GEN/KILL anchors.
        unmodeled_writers = (P.DeleteExceptionBinding, P.BindTypeParameter,
            P.ApplyNamespaceAction, P.ImportStar)
        writers = []
        access_by_site = {}
        for i, access in enumerate(anchors.accesses):
            if access.owner.owner == scope:
                access_by_site.setdefault(access.site.port, []).append((i, access))
        for ordinal, operation in enumerate(flow.operations):
            instruction = operation.instruction
            try:classify_instruction(instruction)
            except TypeError:
                reasons.append('UNKNOWN_INSTRUCTION_SELECTED_FRAME_EFFECT');continue
            if type(instruction) in unmodeled_writers:
                reasons.append('UNMODELED_DIRECT_NAMESPACE_WRITER:' + type(instruction).__name__);continue
            if type(instruction) in (P.EnterRootBody, P.EnterClassBody, P.EnterTypeParameters, P.EnterEvaluationRequest):
                reasons.append('NONFUNCTION_NAMESPACE_ENTRY_NOT_PROVEN_DISJOINT');continue
            if type(instruction) is not P.AccessBinding or instruction.action is P.BindingActionKind.READ:
                continue
            port = bindings.ports[instruction.port]
            interface_index = port.write_target if instruction.action is P.BindingActionKind.WRITE else port.delete_target
            if interface_index is None:
                reasons.append('EXPLICIT_BINDING_WRITER_TARGET_UNRESOLVED');continue
            interface = bindings.interfaces[interface_index]
            if interface.kind != 'LEXICAL_BINDING' or interface.binding != binding:
                continue
            candidates = [(i, a) for i, a in access_by_site.get(operation.entry, ())
                if a.action is instruction.action and a.binding_port == instruction.port
                and a.target is not None and a.target.unit == material.unit and a.target.index == interface_index]
            if len(candidates) != 1 or interface.owner != function.environment.scope:
                reasons.append('SELECTED_SLOT_EXPLICIT_WRITER_ANCHOR_UNRESOLVED');continue
            i, access = candidates[0]
            if instruction.action is P.BindingActionKind.WRITE:
                matches = [a for a in anchors.anchors if a.scope == scope and a.access == i
                    and a.namespace == namespace and a.record.write
                    and self.local.name_facts.get(a.fact) == a.record]
                if len(matches) != 1:
                    reasons.append('SELECTED_SLOT_WRITE_HAS_NO_ORIGINAL_STRONG_DEFINITION');continue
                evidence = matches[0].fact
            else:
                matches = [a for a in anchors.deletes if a.access == i and a.namespace == namespace]
                if len(matches) != 1:
                    reasons.append('SELECTED_SLOT_DELETE_HAS_NO_ORIGINAL_STRONG_DEFINITION');continue
                evidence = None
            writers.append(ExplicitFrameBindingWrite(ordinal, access.site, instruction.action, i,
                instruction.port, interface_index, evidence))
        reasons = tuple(dict.fromkeys(reasons)); self.refusals[key] = reasons
        if reasons:witness = None
        else:
            witness = FrameBindingIsolationWitness(flow.source_function, scope, namespace, binding,
                tuple(dict.fromkeys(w.name_fact for w in selected)), tuple(writers),
                tuple(i for i, operation in enumerate(flow.operations)
                    if any(_state(p.value_type) for p in operation.inputs)
                    and any(_state(p.value_type) and p.port.completion is C.FALLTHROUGH
                        and p.port.target is None for p in operation.outputs)),
                (material.result_artifact, *self.local.materials))
        self.local.check(); self._cache[key] = anchors, flow, witness
        return witness


def frame_isolated_reaching_name_definitions(snapshot, source_function, namespace, *, anchor_graph=None,
        material=None, flow=None, max_states=20000, max_stack=128, max_updates=1000000, effect_context=None):
    from scar.analysis.source_graph_material import source_graph_material
    from scar.analysis.source_name_access_anchors import anchor_name_accesses
    from scar.analysis.source_name_definitions import _reaching_name_definitions
    if material is None:
        material = source_graph_material(getattr(snapshot, 'catalog', snapshot), source_function.unit)
    if anchor_graph is None:
        anchor_graph = anchor_name_accesses(snapshot, source_function.unit, material=material)
    unit = material.fact_store.load(material.unit)
    function = next(f for f in unit.functions if source_port(unit, f.value) == source_function)
    if flow is None:
        flow = project_value_flow(material, source_function, T.Body(material.unit, function.template, function.body))
    context = FrameBindingIsolationContext(material) if effect_context is None else effect_context
    if type(context) is not FrameBindingIsolationContext:raise TypeError('expected original frame-isolation context')
    graph = _reaching_name_definitions(snapshot, source_function, namespace, anchor_graph=anchor_graph,
        material=material, flow=flow, max_states=max_states, max_stack=max_stack, max_updates=max_updates,
        _frame_isolation_effects=context)
    witness = context.witness(material, anchor_graph, function, graph.scope, flow, namespace)
    if witness is not None:
        for writer in witness.writers:
            kind = D.NameDefinitionKind.WRITE if writer.action is P.BindingActionKind.WRITE else D.NameDefinitionKind.DELETE
            if not any(d.kind is kind and d.operation == writer.operation for d in graph.definitions):
                raise ValueError('selected-frame writer is not modeled by the original strong definition')
    reasons = context.refusals[id(anchor_graph), id(flow), namespace]
    graph = replace(graph, conditions=(*graph.conditions, FRAME_MODE, *FRAME_REQUIRED_Q),
        frontiers=(*graph.frontiers, *(D.NameDefinitionFrontier(None, None, None, q)
            for q in (*FRAME_REQUIRED_Q, *reasons))), semantics='CONDITIONAL_SELECTED_FRAME_BINDING_ISOLATION')
    return FrameBindingIsolationProjection(graph, witness, context.analysis_sources, reasons)


def frame_isolated_local_read_tracks(state):
    from scar.analysis.source_name_access_anchors import anchor_name_accesses
    from scar.analysis.source_local_state import local_read_tracks
    if state.anchors is None:
        state.anchors = anchor_name_accesses(state.view.catalog, state.material.unit, material=state.material)
    context = FrameBindingIsolationContext(state.material); result = []
    for function, namespace in local_read_tracks(state):
        if function not in state.flows:
            record = next(f for f in state.source.facts.functions if source_port(state.source.facts, f.value) == function)
            state.flows[function] = project_value_flow(state.material, function,
                T.Body(state.material.unit, record.template, record.body))
        result.append(frame_isolated_reaching_name_definitions(state.view.catalog, function, namespace,
            anchor_graph=state.anchors, material=state.material, flow=state.flows[function], effect_context=context))
    return tuple(result)

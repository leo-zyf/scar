"""Source namespace contents."""

from dataclasses import dataclass
from scar.ir.program import (
    normal_graph as F,
    normal_graph as T,
    name_flow as N,
    normal_graph as G,
    source_productions as P,
    source_bindings as B,
    state_ports as S,
)
from scar.ir.program.source_exec_roots import ExecNamespaceKind
from scar.ir.literals import PythonLiteral
from collections import defaultdict, deque
from scar.analysis.normal_graph import _Emitter
from scar.analysis.namespace_values import extract_captures
from scar.ir.program.control import LanguageCompletionKind as K
from scar.ir.program.model import ValueType, ValueKind
from scar.ir.record_codec import decode
from scar.analysis.source_references import ReferenceSlicer, source_port, DECLARED


# Source namespace interfaces.
# Recover ordered C2 namespace interfaces from original F4 operations.
#
# No target execution or new source rules: each access retains the compiler
# read plan, original world ports and the shared source scope witness.
def _state(value_type):
    return type(value_type) is ValueType and value_type.kind is ValueKind.STATE


class NamespacePlanView:
    def __init__(self, unit, plan):
        self.unit, self.plan = unit, plan
        self.steps = {s.id:s for s in plan.steps}
        self.regions = {r.id:r for r in plan.regions}
        self.contexts = {p.region:p for p in plan.compilation_flow.ports}
        self.key = G.SourceTemplateKey(plan.source.occurrence, plan.purpose)
        self.emitter = _Emitter(plan, self.key, {})

    def gp(self, port):
        return T.GlobalPort(self.unit, port)

    def value(self, value):
        return self.gp(self.emitter.value(value)[0])

    def result(self, step, index=0):
        normal = next(c for c in step.completions if c.kind is K.FALLTHROUGH)
        return self.gp(self.emitter.port(step.id, G.PortKind.RESULT, index, normal))

    def site(self, step):
        return self.gp(self.emitter.port(step.id, G.PortKind.ENTER))

    def owner(self, region):
        context = self.contexts[region]
        return F.CodeScopeWitness(F.CodeScopeRef(self.unit, context.code_scope),
            self.key, region, context.source_scope)

    def worlds(self, step):
        inputs = [v for v in step.arguments if _state(self.emitter.value(v)[1])]
        normal = next(c for c in step.completions if c.kind is K.FALLTHROUGH)
        outputs = [i for i,t in enumerate(normal.types) if _state(t)]
        if len(inputs)!=1 or len(outputs)!=1:
            raise ValueError('namespace operation needs original single-world ABI')
        return self.value(inputs[0]), self.result(step, outputs[0])


def extract_namespace_interfaces(unit, plan, bindings, declarations):
    view = NamespacePlanView(unit, plan)
    accesses, explicit = [], []
    references, captures, obligations = extract_captures(view)
    for step in plan.steps:
        ins = step.instruction
        if type(ins) is P.ApplyNamespaceAction:
            before, after = view.worlds(step)
            explicit.append(N.ExplicitNamespaceAction(view.site(step), view.owner(step.region), ins.kind,
                tuple(view.value(v) for v in step.arguments if not _state(view.emitter.value(v)[1])), before, after))
        if type(ins) is not P.AccessBinding:
            continue
        port = bindings.ports[ins.port]
        if port.name is None:
            raise ValueError('namespace access lost original C2 name')
        name = port.name
        key = declarations.names[name.index].value if name.kind=='SOURCE' else name.symbol
        lookup, missing, read_plan, target = (), None, None, None
        value = None
        if ins.action is P.BindingActionKind.READ:
            read_plan = port.read_plan
            if read_plan is None:
                raise ValueError('namespace read has no C2 lookup plan')
            selected = bindings.plans[read_plan]
            lookup = tuple(N.NamespaceLookupStep(i, N.NamespaceInterfaceRef(unit,s.interface),
                s.on_hit, s.on_miss, s.on_exception) for i,s in enumerate(selected.steps))
            missing = selected.missing_outcome
            value = view.result(step)
        else:
            index = port.write_target if ins.action is P.BindingActionKind.WRITE else port.delete_target
            if index is None:
                raise ValueError('namespace write/delete has no original target')
            target = N.NamespaceInterfaceRef(unit,index)
            if ins.action is P.BindingActionKind.WRITE:
                value = view.value(step.arguments[0])
        before, after = view.worlds(step)
        accesses.append(N.NamespaceAccess(view.site(step), view.owner(step.region), ins.port, ins.action,
            name, key, read_plan, lookup, missing, target, value, before, after))
    return tuple(accesses), tuple(explicit), references, captures, obligations


def bind_exec_namespaces(connection, bindings, references, captures):
    """Root dict references and function globals stay separate from modules."""
    unit = connection.root.owner.unit
    root = connection.root
    formal = lambda i:T.GlobalPort(unit, G.Port(root.template, root.region, G.PortKind.PARAMETER, i))
    common = (*connection.conditions, 'SELECTED_EXEC_NAMESPACE_BRANCH')
    result = []
    for index, descriptor in enumerate(bindings.interfaces):
        ref = N.NamespaceInterfaceRef(unit, index)
        options = [None]
        if descriptor.kind=='ENV_GLOBALS':
            options = [c for c in captures if c.body.source_scope==descriptor.owner] or [None]
        for capture in options:
            actual, parameter = None, None
            if descriptor.kind=='ROOT_GLOBALS':
                actual,parameter,origin = connection.globals.value,formal(0),'ROOT_GLOBALS_ARGUMENT'
            elif descriptor.kind=='ROOT_LOCALS':
                actual,parameter,origin = connection.locals.value,formal(1),'ROOT_LOCALS_ARGUMENT'
            elif descriptor.kind=='ENV_GLOBALS':
                actual = None if capture is None else capture.globals
                origin = 'CAPTURED_FUNCTION_GLOBALS' if capture is not None else 'CAPTURED_GLOBALS_INTERFACE_UNRESOLVED'
            elif descriptor.kind=='BUILTINS':
                # Root frame builtins are selected at entry; a function retains
                # its builtins selected at creation, not a later dict lookup.
                origin = ('ROOT_FRAME_BUILTINS_SELECTED_AT_ENTRY' if descriptor.owner==root.source_scope else
                    'CALLABLE_FRAME_BUILTINS_SELECTED_AT_CREATION')
            else:
                origin = descriptor.kind+'_TEMPLATE'
            conditions = common if capture is None else (*common,'CURRENT_CALLABLE_IS_SELECTED_FUNCTION_CREATION')
            result.append(N.NamespaceInterfaceBinding(connection.id, ref, descriptor, origin, actual, parameter,
                conditions, None if capture is None else capture.function))
    reference_bindings = tuple(N.GlobalsReferenceBinding(connection.id, r.result, connection.globals.value,
        r.owner, (*common,'ACTIVE_FRAME_GLOBALS_FROM_SELECTED_ROOT_OR_ITS_CAPTURE_CHAIN'))
        for r in references if r.globals_scope==root.source_scope)
    return tuple(result), reference_bindings


class CallerReferencePaths:
    """Sparse directed aliases and factored name channels, with row witnesses."""
    def __init__(self, unit):
        self.unit = unit
        self.ports = {i:T.GlobalPort(unit.unit, decode(G.Port,p,record_tags=P.PRODUCTION_RECORD_TAGS)
            if type(p) is dict else p) for i,p in enumerate(unit.ports)}
        self.indices = {p:i for i,p in self.ports.items()}
        self.adj = defaultdict(list)
        for i,(a,b) in enumerate(unit.aliases):
            self.adj[a].append((b,F.FactTable.ALIASES,i))
        for i,row in enumerate(unit.names):
            channel = ('namespace',row.namespace)
            a,b = (row.value,channel) if row.write else (channel,row.value)
            self.adj[a].append((b,F.FactTable.NAMES,i))
        self.cache = {}

    def path(self, source, target):
        """None means not established, () means the same original port."""
        if target not in self.indices:
            return None
        if source not in self.cache:
            parents = {source:None}; queue = deque((source,))
            while queue:
                node = queue.popleft()
                for successor,table,row in self.adj.get(node,()):
                    if successor not in parents:
                        parents[successor] = (node,table,row); queue.append(successor)
            self.cache[source] = parents
        parents = self.cache[source]; end = self.indices[target]
        if end not in parents:
            return None
        reversed_edges = []
        while parents[end] is not None:
            before,table,row = parents[end]
            reversed_edges.append((before,end,table,row)); end = before
        edges = list(reversed(reversed_edges)); path = []; i = 0
        while i<len(edges):
            a,b,table,row = edges[i]
            if type(b) is tuple:
                _,c,other_table,other_row = edges[i+1]
                if table is not F.FactTable.NAMES or other_table is not table:
                    raise ValueError('invalid factored namespace path')
                path.append(N.NamespaceReferenceHop(self.ports[a],self.ports[c],table,(row,other_row))); i+=2
            else:
                path.append(N.NamespaceReferenceHop(self.ports[a],self.ports[b],table,(row,))); i+=1
        return tuple(path)


# Connect source namespace writes and import exports without executing them.
#
# Namespace identity/current versions stay open. Every source writer/export is
# conditional, and importable file candidates are never selected by class name.
# IMPORT_FROM's attribute branch and import-registry fallback remain separate.
_NAMESPACE_GUARDS = (*DECLARED, 'RECORDED_LOOKUP_NAMESPACE_ALTERNATIVE_SELECTED',
    'SOURCE_WRITER_SCOPE_ENTERED_AND_WRITE_COMPLETED', 'CURRENT_SLOT_VERSION_SELECTS_RECORDED_SOURCE_WRITE',
    'LOOKUP_PATH_AND_NAMESPACE_WORLDS_ARE_COMPATIBLE', 'NO_CURRENT_NAMESPACE_OR_REFERENCE_IDENTITY_PROOF')
_IMPORT_GUARDS = (*_NAMESPACE_GUARDS, 'STANDARD_IMPORTER_AND_MODULE_RETURN_SEMANTICS_SELECTED',
    'SOURCE_SEARCH_ROOT_AND_SYS_MODULES_SELECT_RECORDED_PROVIDER_AND_VERSION',
    'MODULE_INITIALIZATION_AND_CURRENT_EXPORT_SELECT_RECORDED_WRITE')


class NamespaceReferenceResolver:
    def __init__(self, snapshot):
        self.snapshot = snapshot; self.slicer = ReferenceSlicer(snapshot); self.units = {}
        self.modules = defaultdict(list)
        store = snapshot.fact_store
        # This is an import candidate index. Compiled exec dictionaries are
        # deliberately excluded, even when they share an original filename.
        for identity in sorted(store.importable_units):
            row = store.rows[identity]; model = store.model_units[identity]
            names = (row.get('module'), *model.get('module_names', ()),
                *(mapping['name'] for mapping in model.get('module_mappings', ())))
            for name in dict.fromkeys(name for name in names if type(name) is str and name):
                self.modules[name].append(identity)

    def unit(self, identity):
        if identity not in self.units:
            unit, _, aliases, reads, writes, formals, terminals = self.slicer.unit(identity)
            owners = defaultdict(set)
            for group in unit.scope_groups:
                for span in group.tables:
                    if span.table is F.FactTable.NAMES:
                        for i in range(span.start, span.stop): owners[i].add(group.owner)
            attributes = defaultdict(list); imports = defaultdict(list); exports = defaultdict(list)
            for i, row in enumerate(unit.attributes):
                if row.action in (F.AttributeAction.GET, F.AttributeAction.IMPORT_MEMBER):
                    attributes[row.value].append((i, row))
            for i, row in enumerate(unit.imports): imports[row.value].append((i, row))
            for i, row in enumerate(unit.names):
                if row.write and row.namespace.kind is F.NamespaceKind.MODULE:
                    exports[row.namespace.name].append((i, row))
            self.units[identity] = unit, aliases, reads, writes, formals, terminals, owners, attributes, imports, exports
        return self.units[identity]

    def resolve(self, point, *, max_states=4000, max_selectors=32):
        return _Query(self, point, max_states, max_selectors).run()


class _Query:
    def __init__(self, resolver, point, max_states, max_selectors):
        resolver.slicer.local(point)
        if type(max_states) is not int or max_states < 1 or type(max_selectors) is not int or max_selectors < 1:
            raise ValueError('namespace source query needs positive state and selector limits')
        self.resolver = resolver; self.max_states = max_states; self.max_selectors = max_selectors
        root = N.NamespaceReferenceState(point)
        self.states = [root]; self.ids = {root: 0}; self.pending = deque((0,))
        self.edges = []; self.origins = []; self.frontiers = []; self.frontier_ids = set()

    def frontier(self, state, reason, evidence=()):
        key = state, reason, tuple(evidence)
        if key not in self.frontier_ids:
            self.frontier_ids.add(key); self.frontiers.append(N.NamespaceReferenceFrontier(*key))

    def edge(self, state, point, selectors, kind, evidence, conditions):
        if len(selectors) > self.max_selectors:
            self.frontier(state, 'NAMESPACE_SELECTOR_DEPTH_FRONTIER', evidence); return
        value = N.NamespaceReferenceState(point, selectors)
        if value not in self.ids:
            if len(self.states) >= self.max_states:
                self.frontier(state, 'NAMESPACE_REFERENCE_STATE_BUDGET_FRONTIER', evidence); return
            self.ids[value] = len(self.states); self.states.append(value); self.pending.append(self.ids[value])
        self.edges.append(N.NamespaceReferenceEdge(state, self.ids[value], kind, tuple(evidence), tuple(conditions)))

    def module(self, state, symbol, selectors, evidence, conditions):
        providers = tuple(self.resolver.modules.get(symbol, ()))
        if not selectors:
            self.origins.append(N.NamespaceReferenceOrigin(state, N.NamespaceOriginKind.MODULE_CANDIDATE,
                tuple(evidence), tuple(conditions), symbol, providers))
            if not providers: self.frontier(state, 'IMPORTED_MODULE_SOURCE_PROVIDER_NOT_RETAINED', evidence)
            return
        selected, rest = selectors[0], selectors[1:]
        guards = (*conditions, 'MODULE_ATTRIBUTE_READ_RETURNS_RECORDED_STORED_REFERENCE')
        found = False
        for identity in providers:
            unit, *_, exports = self.resolver.unit(identity)
            for ordinal, writer in exports.get(selected.member, ()):
                found = True
                ref = self.resolver.slicer.fact(unit, 'names', ordinal)
                self.edge(state, source_port(unit, writer.value), rest, N.NamespaceEdgeKind.MODULE_EXPORT,
                    (*evidence, selected.access, ref), guards)
        self.frontier(state, 'CURRENT_MODULE_EXPORT_VERSION_OR_DYNAMIC_ATTRIBUTE_PATH_OPEN', (*evidence, selected.access))
        if not found: self.frontier(state, 'DECLARED_MODULE_EXPORT_SOURCE_UNRESOLVED', (*evidence, selected.access))
        if selected.import_from:
            # IMPORT_FROM may recover a registered submodule after its normal
            # attribute path. It does not turn every member into a submodule.
            self.module(state, symbol+'.'+selected.member, rest, (*evidence, selected.access),
                (*conditions, 'IMPORT_FROM_ATTRIBUTE_PATH_MISSES_AND_REGISTRY_FALLBACK_SELECTED',
                    'IMPORT_REGISTRY_ENTRY_IS_RECORDED_MODULE_NAMESPACE'))

    def run(self):
        r = self.resolver
        while self.pending:
            index = self.pending.popleft(); state = self.states[index]
            unit, local = r.slicer.local(state.port)
            unit, aliases, reads, writes, formals, terminals, owners, attributes, imports, _ = r.unit(unit.unit)
            before = len(self.edges); before_origins = len(self.origins); before_frontiers = len(self.frontiers)
            for source, ordinal in aliases.get(local, ()):
                self.edge(index, source_port(unit, source), state.selectors, N.NamespaceEdgeKind.ALIAS,
                    (r.slicer.fact(unit, 'aliases', ordinal),), DECLARED)
            for namespace, ordinal in reads.get(local, ()):
                read = r.slicer.fact(unit, 'names', ordinal)
                if namespace.kind is F.NamespaceKind.BUILTINS:
                    self.frontier(index, 'BUILTIN_MAPPING_SLOT_BINDING_NOT_RESOLVED', (read,)); continue
                if namespace.kind is F.NamespaceKind.LEXICAL:
                    kind = N.NamespaceEdgeKind.LEXICAL_WRITE
                    if len(owners[ordinal]) != 1:
                        self.frontier(index, 'LEXICAL_READ_SCOPE_UNRESOLVED', (read,)); continue
                    guards = (*_NAMESPACE_GUARDS, 'SAME_LEXICAL_FRAME_AND_WRITE_VERSION_SELECTED')
                elif namespace.kind is F.NamespaceKind.MODULE:
                    kind = N.NamespaceEdgeKind.GLOBAL_WRITE
                    compiled = (unit.unit in r.snapshot.fact_store.original_source_units
                        or unit.unit in getattr(r.snapshot.fact_store, 'observed_compiled_units', ()))
                    guards = (*_NAMESPACE_GUARDS, 'CURRENT_FRAME_GLOBALS_SELECT_RECORDED_COMPILE_CONTEXT_DICT' if compiled
                        else 'CURRENT_FRAME_GLOBALS_SELECT_RECORDED_SOURCE_MODULE_NAMESPACE')
                    self.frontier(index, 'CURRENT_GLOBAL_NAMESPACE_IDENTITY_AND_CONTENTS_OPEN', (read,))
                else:
                    kind = N.NamespaceEdgeKind.CLASS_WRITE
                    guards = (*_NAMESPACE_GUARDS, 'CLASS_NAMESPACE_MAPPING_SELECTS_RECORDED_SOURCE_WRITE')
                    self.frontier(index, 'CURRENT_CLASS_NAMESPACE_MAPPING_AND_VERSION_OPEN', (read,))
                found = False
                for source, writer in writes.get(namespace, ()):
                    if namespace.kind is F.NamespaceKind.LEXICAL and owners[writer] != owners[ordinal]:
                        self.frontier(index, 'CAPTURE_CREATION_FRAME_NOT_SELECTED', (read,)); continue
                    found = True
                    self.edge(index, source_port(unit, source), state.selectors, kind,
                        (r.slicer.fact(unit, 'names', writer), read), guards)
                if not found: self.frontier(index, 'DECLARED_NAMESPACE_WRITER_UNRESOLVED', (read,))
            for ordinal, attribute in attributes.get(local, ()):
                ref = r.slicer.fact(unit, 'attributes', ordinal)
                selector = N.NamespaceSelector(attribute.name, attribute.action is F.AttributeAction.IMPORT_MEMBER, ref)
                self.edge(index, source_port(unit, attribute.receiver), (selector, *state.selectors),
                    N.NamespaceEdgeKind.ATTRIBUTE_RECEIVER, (ref,),
                    (*DECLARED, 'PENDING_ATTRIBUTE_SELECTION_IS_NOT_RECEIVER_RESULT_ALIAS'))
            for ordinal, imported in imports.get(local, ()):
                ref = r.slicer.fact(unit, 'imports', ordinal)
                self.module(index, imported.module, state.selectors, (ref,), _IMPORT_GUARDS)
                requested = imported.requested_module
                if (imported.module and requested is not None
                        and requested.startswith(imported.module+'.')):
                    # import a.b.c returns a, but the standard loader also
                    # installs b on a and c on a.b. Retain every witnessed
                    # prefix: later slot mutation can affect any selection.
                    suffix = requested[len(imported.module)+1:].split('.')
                    prefix = []; selected = []
                    for component, selector in zip(suffix, state.selectors):
                        if selector.member != component: break
                        prefix.append(component); selected.append(selector.access)
                        self.module(index, imported.module+'.'+'.'.join(prefix),
                            state.selectors[len(prefix):], (ref, *selected), (*_IMPORT_GUARDS,
                                'REQUESTED_CHILD_IMPORT_COMPLETES_AND_INSTALLS_STANDARD_PARENT_ATTRIBUTE',
                                'CURRENT_PARENT_ATTRIBUTE_SELECTS_RECORDED_IMPORTED_CHILD_MODULE'))
            if local in terminals:
                kind, table, ordinal = terminals[local]; ref = r.slicer.fact(unit, table, ordinal)
                if state.selectors:
                    self.frontier(index, 'ATTRIBUTE_OF_CLASS_OR_VALUE_REQUIRES_DESCRIPTOR_OR_MEMBER_MODEL', (ref,))
                else:
                    self.origins.append(N.NamespaceReferenceOrigin(index, N.NamespaceOriginKind(kind), (ref,),
                        (*DECLARED, 'SOURCE_DEFINITION_CREATION_AND_REFERENCE_PATH_SELECTED')))
            if local in formals:
                self.frontier(index, 'FUNCTION_PARAMETER_REQUIRES_SELECTED_CALL_CONTEXT')
            if (len(self.edges), len(self.origins), len(self.frontiers)) == (before, before_origins, before_frontiers):
                self.frontier(index, 'SOURCE_REFERENCE_PRODUCER_UNRESOLVED')
        r.snapshot.validate_parents()
        return N.NamespaceReferenceSlice(tuple(self.states), tuple(self.edges), tuple(self.origins), tuple(self.frontiers))


# Mechanical addresses for sealed implicit class namespace operations.
#
# Descriptors denote source environment templates, not current dictionaries or
# cells. The caller retains lookup selection, frame identity and world guards;
# this projection neither selects the standard builder nor enters a class body.
@dataclass(frozen=True, slots=True)
class ImplicitNamespaceAccess:
    kind: S.BindingAccessKind
    key: str
    descriptors: tuple[B.EnvironmentInterface, ...]
    owner_handles: tuple[T.GlobalPort | None, ...]
    payload: T.GlobalPort | None
    missing_outcome: str | None


def _at(rows, index, label):
    if type(index) is not int or not 0 <= index < len(rows):
        raise ValueError('implicit namespace invalid original '+label)
    return rows[index]


def _name(declarations, name):
    if type(name) is not B.NameRef:
        raise ValueError('implicit namespace lacks original name reference')
    if name.kind == 'SOURCE':
        return _at(declarations.names, name.index, 'name').value
    return name.symbol


def _globals(bindings, resolution_index, resolution):
    """Prefer original C2 routes, including explicit escaped-root routes."""
    found = set()
    for port in bindings.ports:
        if port.resolution != resolution_index:
            continue
        if port.source_scope != resolution.scope or port.name != resolution.name:
            raise ValueError('implicit namespace resolution/port scope or name differs')
        indices = [i for i in (port.write_target, port.delete_target) if i is not None]
        if port.read_plan is not None:
            plan = _at(bindings.plans, port.read_plan, 'lookup plan')
            indices.extend(step.interface for step in plan.steps)
        for index in indices:
            descriptor = _at(bindings.interfaces, index, 'interface')
            if descriptor.kind in ('ROOT_GLOBALS', 'ENV_GLOBALS'):
                found.add(descriptor)
    if len(found) > 1:
        raise ValueError('implicit namespace original globals routes disagree')
    if found:
        return next(iter(found))
    # C2 global_interface can redirect an escaped walrus to the root. Without
    # its original plan/target, class_scope alone cannot establish that route.
    if resolution.reason == 'ESCAPED_WALRUS_GLOBAL':
        return None
    return B.EnvironmentInterface('ENV_GLOBALS', resolution.scope)


def identify_implicit_namespace(bindings, declarations,
        context: NamespacePlanView, step) -> ImplicitNamespaceAccess | None:
    ins = step.instruction
    if type(ins) not in (P.ReadBuildClass, P.AccessClassName):
        return None
    if context.steps.get(step.id) != step:
        raise ValueError('implicit namespace step is outside its original plan')
    scope = context.owner(step.region).source_scope
    if type(scope) is not int or scope < 0:
        raise ValueError('implicit namespace lacks original source scope')
    if type(ins) is P.ReadBuildClass:
        if len(step.arguments) != 1:
            raise ValueError('implicit builder lookup world ABI differs')
        return ImplicitNamespaceAccess(S.BindingAccessKind.READ, '__build_class__',
            (B.EnvironmentInterface('BUILTINS', scope),), (None,),
            context.result(step, 0), 'RAISE_NAME_ERROR')

    if ins.interface.scope != scope or type(ins.name) is not str:
        raise ValueError('implicit class access original scope/name differs')
    if ins.action not in (P.BindingActionKind.READ, P.BindingActionKind.WRITE):
        return None
    read = ins.action is P.BindingActionKind.READ
    if len(step.arguments) != (3 if read else 4):
        raise ValueError('implicit class access environment/value/world ABI differs')
    prepared = B.EnvironmentInterface('CLASS_PREPARED_MAPPING', scope)
    builtin = B.EnvironmentInterface('BUILTINS', scope)
    missing = 'RAISE_NAME_ERROR' if read else None
    if ins.resolution is None:
        if any(r.scope == scope and _name(declarations, r.name) == ins.name
                for r in bindings.resolutions):
            raise ValueError('implicit class access omitted its original resolution')
        descriptors = ((prepared, B.EnvironmentInterface('ENV_GLOBALS', scope), builtin)
            if read else (prepared,))
    else:
        resolution = _at(bindings.resolutions, ins.resolution, 'resolution')
        if resolution.scope != scope or _name(declarations, resolution.name) != ins.name:
            raise ValueError('implicit class resolution belongs to another scope/name')
        classification = resolution.classification
        if classification in ('LOCAL', 'FREE'):
            slot = _at(bindings.bindings, resolution.binding, 'binding')
            if (_name(declarations, slot.name) != ins.name
                    or (slot.owner == scope) != (classification == 'LOCAL')
                    or slot.domain not in ('LOCAL_BINDING', 'TYPE_PARAMETER_BINDING', 'CLASS_PROTOCOL')):
                raise ValueError('implicit class cell differs from original binding')
            cell = B.EnvironmentInterface('CLASS_PROTOCOL_CELL' if slot.domain == 'CLASS_PROTOCOL'
                else 'LEXICAL_BINDING', slot.owner, resolution.binding)
            descriptors = (prepared, cell) if read and classification == 'FREE' else (cell,)
            if read:
                missing = 'RAISE_UNBOUND_FREE' if classification == 'FREE' else 'RAISE_UNBOUND_LOCAL'
        elif classification in ('CLASS_LOCAL', 'GLOBAL_IMPLICIT', 'GLOBAL_EXPLICIT'):
            if not read and classification == 'CLASS_LOCAL':
                descriptors = (prepared,)
            else:
                global_descriptor = _globals(bindings, ins.resolution, resolution)
                if global_descriptor is None:
                    return None
                descriptors = ((global_descriptor, builtin) if classification == 'GLOBAL_EXPLICIT'
                    else (prepared, global_descriptor, builtin)) if read else (global_descriptor,)
        else:
            return None
    environment = context.value(step.arguments[0])
    mapping = context.value(step.arguments[1])
    handles = tuple(mapping if d.kind == 'CLASS_PREPARED_MAPPING' else environment
        for d in descriptors)
    payload = context.result(step, 0) if read else context.value(step.arguments[2])
    return ImplicitNamespaceAccess(S.BindingAccessKind.READ if read else S.BindingAccessKind.WRITE,
        ins.name, descriptors, handles, payload, missing)


# Conditional dictionary definitions reaching original namespace read ports.
@dataclass(frozen=True, slots=True)
class _Definition:
    mapping: int
    assigned: int
    key: str
    key_source: T.GlobalPort | PythonLiteral
    key_literal_source: T.GlobalPort | None
    key_path: tuple[N.NamespaceReferenceHop, ...]
    world: T.GlobalPort
    result_world: T.GlobalPort
    table: F.FactTable
    row: int
    member: int | None


def _keys(paths, key):
    if type(key) is PythonLiteral:
        value = key.to_python()
        if type(value) is str:
            yield value,key,None,()
        return
    for row in paths.unit.constants:
        value = row.literal.to_python()
        if type(value) is not str:
            continue
        path = paths.path(row.value, paths.ports[key])
        if path is not None:
            yield value,paths.ports[key],paths.ports[row.value],path


def content_definitions(paths, targets, interface_index):
    """Select source mappings first; recover each needed F4 plan once."""
    unit = paths.unit
    relevant = lambda point:any(paths.path(point,target) is not None for target in targets)
    selected = [(i,row) for i,row in enumerate(unit.containers)
        if row.kind is F.ContainerKind.MAPPING and row.pairs and relevant(row.value)]
    keys = {paths.ports[row.value].port.template for _,row in selected}
    if keys:
        interface_index._prefetch(unit, keys)
    for index,row in selected:
        point = paths.ports[row.value]
        plan = interface_index._plan(unit,point.port.template)[0]
        view = NamespacePlanView(unit.unit,plan)
        step = view.steps[point.port.owner]
        before,after = view.worlds(step)
        for member,(key,value) in enumerate(row.pairs):
            for text,key_source,literal,path in _keys(paths,key):
                yield _Definition(row.value,value,text,key_source,literal,path,before,after,
                    F.FactTable.CONTAINERS,index,member)
    for index,row in enumerate(unit.items):
        if not row.write or not relevant(row.receiver):
            continue
        for text,key_source,literal,path in _keys(paths,row.key):
            yield _Definition(row.receiver,row.value,text,key_source,literal,path,
                paths.ports[row.world],paths.ports[row.result_world],F.FactTable.ITEMS,index,None)


def link_namespace_contents(connection, accesses, bindings, reference_bindings, caller, interface_index):
    """May-reaching content is guarded by worlds, lookup misses and callable."""
    paths = CallerReferencePaths(caller)
    references = {r.reference:r.actual for r in reference_bindings}
    targets = set()
    for binding in bindings:
        actual = references.get(binding.actual,binding.actual)
        if type(actual) is T.GlobalPort and actual.unit==caller.unit:
            targets.add(actual)
    definitions = tuple(content_definitions(paths,targets,interface_index))
    by_interface = {}
    for binding in bindings:
        by_interface.setdefault(binding.interface,[]).append(binding)
    links = []
    for access in accesses:
        for lookup in access.lookup:
            for binding in by_interface.get(lookup.interface,()):
                actual = references.get(binding.actual,binding.actual)
                if type(actual) is not T.GlobalPort or actual.unit!=caller.unit:
                    continue
                for definition in definitions:
                    if definition.key!=access.key:
                        continue
                    path = paths.path(definition.mapping,actual)
                    if path is None:
                        continue
                    conditions = (*binding.conditions,
                        'SOURCE_MAPPING_REFERENCE_PATH_SELECTED_AT_EXEC_CALL',
                        'SOURCE_KEY_REFERENCE_PATH_HAS_RECORDED_STRING_VALUE',
                        'DEFINITION_REACHES_READ_WORLD_WITHOUT_INTERVENING_REPLACEMENT',
                        'ALL_EARLIER_LOOKUP_STEPS_REPORT_MISSING',
                        'SELECTED_MAPPING_PROTOCOL_RETURNS_RECORDED_CONTENT')
                    links.append(N.NamespaceContentLink(connection.id,lookup.interface,lookup.ordinal,
                        access.value,access.world_in,paths.ports[definition.mapping],paths.ports[definition.assigned],
                        definition.world,definition.result_world,definition.row,definition.table,definition.member,
                        definition.key,definition.key_source,definition.key_literal_source,definition.key_path,
                        path,conditions,binding.capturing_function,
                        binding.actual if binding.capturing_function is not None else None))
    return tuple(links)

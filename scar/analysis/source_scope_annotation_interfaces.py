"""Source-bound variable annotation groups from static compiler queries.

The primary query replaces annotation expressions in a temporary AST restored
from original F2 references. Marker loads recover emission order, duplicate
finally emissions and conditional indices. Pruned source templates use an
explicitly marked ancestor-control projection. No source/name/line join,
parsing, target execution or query capture-layout reuse is involved.
"""
import ast
import dis
from collections import defaultdict
from copy import deepcopy
from types import CodeType
from scar.analysis.source_syntax import _restore_ast
from scar.ir.program import source_productions as P
from scar.ir.program.protocol_signatures import ExceptionABI


def _require(condition, path, detail, *, code='SCOPE_ANNOTATION_INTERFACE'):
    if not condition:
        from scar.analysis.source_productions import ProductionError
        raise ProductionError(code, str(path), detail)


def _codes(root):
    pending=[root];seen=set()
    while pending:
        item=pending.pop()
        if id(item) in seen:continue
        seen.add(id(item));yield item
        pending.extend(c for c in item.co_consts if type(c) is CodeType)


_CONDITIONAL_CONTAINERS = frozenset(
    ('If', 'While', 'For', 'AsyncFor', 'Try', 'TryStar', 'With', 'AsyncWith', 'Match'))


class ScopeAnnotationIndex:
    """Own variable annotation sites, publication contracts and emission queries."""

    def __init__(self, source):
        self.source = source
        self._interfaces = None
        self._publications = {}
        self._deferred_publications = None
        self._creators = {}
        by_scope = defaultdict(list)
        for occurrence, node in enumerate(source.context.syntax.nodes):
            if node.constructor == 'AnnAssign':
                creator = source.context.entry_context(occurrence).environment
                self._creators[occurrence] = creator
                by_scope[creator].append(occurrence)
        self.by_scope = defaultdict(tuple, {creator: tuple(occurrences)
            for creator, occurrences in by_scope.items()})

    def publication(self, occurrence, exception_abi):
        # Only exact original coordinates and enum values share a cache key.
        if type(occurrence) is not int or type(exception_abi) is not ExceptionABI:
            return self._publication(occurrence, exception_abi)
        key = occurrence, exception_abi
        if key not in self._publications:
            self._publications[key] = self._publication(occurrence, exception_abi)
        return self._publications[key]

    def _publication(self, occurrence, exception_abi):
        source = self.source
        roles, nodes = source.roles, source.context.syntax.nodes
        _require(nodes[occurrence].constructor == 'AnnAssign', occurrence,
            'expected annotated assignment', code='STATEMENT_INTERFACE')
        annotation = P.SlotRef(occurrence, 'FIELD', roles.fields['AnnAssign']['annotation'])
        context = source.context.context_at_slot(annotation).state
        region = source.context.material.regions[context.region]
        simple = roles.field(occurrence, 'simple').scalar.to_python()
        if region.suppression_reason in ('FUNCTION_LOCAL_ANNOTATION', 'NON_SIMPLE_ANNOTATION'):
            return None
        target = roles.field(occurrence, 'target').reference
        _require(simple == 1 and nodes[target].constructor == 'Name', occurrence,
            'published variable annotation requires a simple original name', code='STATEMENT_INTERFACE')
        name = P.SlotRef(target, 'FIELD', roles.fields['Name']['id'])
        if region.suppression_reason == 'FUTURE_ANNOTATION_STRINGIZED':
            return P.PublishAnnotation(annotation, name, P.AnnotationPublicationKind.STRINGIZED, False, exception_abi)
        _require(region.eligibility == 'ELIGIBLE' and region.kind == 'VARIABLE_ANNOTATION', occurrence,
            'unknown original variable annotation policy', code='STATEMENT_INTERFACE')
        environment = self._creators[occurrence]
        scope = source.context.material.scopes[environment]
        _require(scope.kind in ('ROOT_EXEC', 'ROOT_INTERACTIVE', 'CLASS'), occurrence,
            'published annotation requires an original module/class namespace', code='STATEMENT_INTERFACE')
        conditional = scope.kind != 'CLASS'
        current = occurrence
        while current in roles.parents:
            edge = roles.parents[current]
            parent = nodes[edge.occurrence].constructor
            if parent in ('ClassDef', 'FunctionDef', 'AsyncFunctionDef'):
                break  # compiler conditional depth restarts in each code scope
            conditional |= parent in _CONDITIONAL_CONTAINERS
            current = edge.occurrence
        return P.PublishAnnotation(annotation, name, P.AnnotationPublicationKind.DEFERRED, conditional, exception_abi)

    def deferred_publications(self):
        if self._deferred_publications is None:
            self._deferred_publications = {occurrence: publication for occurrence in self._creators
                if (publication := self.publication(occurrence, ExceptionABI.EXCEPTION_WORLD)) is not None
                and publication.publication is P.AnnotationPublicationKind.DEFERRED}
        return self._deferred_publications

    def restore_source_objects(self):
        """Build a working AST for one marker query from the accepted occurrences."""
        syntax, roles = self.source.context.syntax, self.source.roles
        tree = _restore_ast(syntax.root, syntax.nodes)
        objects = {syntax.root: tree}
        pending = [syntax.root]
        while pending:
            occurrence = pending.pop()
            node, restored = syntax.nodes[occurrence], objects[occurrence]
            for name, value in zip(roles.fields[node.constructor], node.fields):
                if value.kind == 'NODE':
                    objects[value.reference] = getattr(restored, name)
                    pending.append(value.reference)
                elif value.kind == 'LIST':
                    for member, child in zip(value.items, getattr(restored, name)):
                        if member.kind == 'NODE':
                            objects[member.reference] = child
                            pending.append(member.reference)
        return tree, objects

    def _emissions(self,code,markers,level):
        by_scope={}
        for unit in _codes(code):
            ops=[op for op in dis.get_instructions(unit) if op.opname not in ('EXTENDED_ARG','NOT_TAKEN')]
            entries=[];scopes=set()
            for i,op in enumerate(ops):
                if op.opname!='LOAD_CONST' or type(op.argval) is not bytes or op.argval not in markers:continue
                publication=markers[op.argval]
                scope=self.source.context.context_at_slot(publication.annotation).state.environment
                scopes.add(scope);tail=ops[i+1:i+4]
                _require(len(tail)==3 and tail[0].opname=='COPY' and tail[0].arg==2
                    and tail[1].opname=='LOAD_CONST' and type(tail[1].argval) is str and tail[2].opname=='STORE_SUBSCR',
                    publication.annotation,'controlled compiler marker/store shape differs')
                registration=None
                if publication.conditional_registration:
                    guard=ops[i-4:i]
                    _require(len(guard)==4 and guard[0].opname in ('LOAD_CONST','LOAD_SMALL_INT')
                        and type(guard[0].argval) is int and guard[0].argval>=0
                        and guard[1].opname in ('LOAD_DEREF','LOAD_GLOBAL')
                        and guard[1].argval=='__conditional_annotations__'
                        and guard[2].opname=='CONTAINS_OP' and guard[2].arg==0
                        and guard[3].opname=='POP_JUMP_IF_FALSE',publication.annotation,
                        'controlled compiler conditional registration shape differs')
                    registration=P.AnnotationRegistration(scope,level,guard[0].argval)
                entries.append(P.ScopeAnnotationEntry(publication.annotation,publication.name,tail[1].argval,registration))
            if not entries:continue
            _require(len(scopes)==1,scopes,'one callback must not mix original source annotation groups')
            scope=scopes.pop();entries=tuple(entries)
            _require(scope not in by_scope or by_scope[scope]==entries,scope,
                'repeated code-unit emission has incompatible annotation interfaces')
            by_scope[scope]=entries
        return by_scope

    def _projection(self,objects,scope):
        """Admit an otherwise pruned class template, retaining its whole body.

        Only ancestor/sibling control is relaxed. Its query's capture/frame
        layout is discarded. An explicit flag distinguishes this source
        template from code retained by the original compilation.
        """
        source=self.source;roles=source.roles;syntax=source.context.syntax
        owner=source.context.material.scopes[scope].owner.occurrence
        protected={owner};pending=[owner]
        while pending:
            row=syntax.nodes[pending.pop()]
            for value in row.fields:
                children=(value.reference,) if value.kind=='NODE' else tuple(
                    v.reference for v in value.items if v.kind=='NODE') if value.kind=='LIST' else ()
                protected.update(children);pending.extend(children)
        query=deepcopy(objects)
        names={n.value for n in source.frontend.declarations.names};suffix=0
        while (name:='_scar_static_annotation_condition_'+str(suffix)) in names:suffix+=1
        def unknown(location):return ast.copy_location(ast.Name(id=name,ctx=ast.Load()),location)
        def condition(value):
            return ast.copy_location(ast.Compare(left=value,ops=[ast.Is()],comparators=[unknown(value)]),value)
        try_bodies=[]
        for original,node in query.items():
            if original in protected:continue
            if type(node) in (ast.If,ast.While,ast.Assert):node.test=condition(node.test)
            elif type(node) in (ast.For,ast.AsyncFor):
                node.iter=ast.copy_location(ast.Tuple(elts=[node.iter,unknown(node.iter)],ctx=ast.Load()),node.iter)
            elif type(node) is ast.match_case:
                if node.guard is not None:node.guard=condition(node.guard)
            elif type(node) is ast.Match:node.subject=condition(node.subject)
            elif type(node) in (ast.Try,ast.TryStar):
                try_bodies.append(node)
            elif type(node) in (ast.Return,ast.Raise,ast.Break,ast.Continue):
                edge=roles.parents[original];parent=query[edge.occurrence]
                field=tuple(roles.fields[syntax.nodes[edge.occurrence].constructor])[edge.ordinal]
                values=([node.value] if type(node) is ast.Return and node.value is not None else
                    [v for v in (node.exc,node.cause) if v is not None] if type(node) is ast.Raise else [])
                expression=values[0] if len(values)==1 else ast.copy_location(ast.Tuple(elts=values,ctx=ast.Load()),node)
                replacement=ast.copy_location(ast.Expr(value=expression) if values else ast.Pass(),node)
                if edge.item_path:
                    container=getattr(parent,field)
                    for i in edge.item_path[:-1]:container=container[i]
                    container[edge.item_path[-1]]=replacement
                else:setattr(parent,field,replacement)
        # Preserve original list-slot indices until every replacement is done.
        for node in try_bodies:node.body.insert(0,ast.copy_location(ast.Expr(value=unknown(node)),node))
        return query[syntax.root]

    def _collect(self):
        source, syntax = self.source, self.source.context.syntax
        publications = self.deferred_publications()
        if not publications:
            return {}
        tree, objects = self.restore_source_objects()
        # Reserve markers absent from all original compiled constants, including
        # folded tuples/bytes, for every supported target optimization policy.
        constants=set()
        for level in (0,1,2):
            code=compile(tree,'<scar-static-original-annotations>',syntax.mode,dont_inherit=True,optimize=level)
            values=[c for unit in _codes(code) for c in unit.co_consts]
            while values:
                value=values.pop()
                if type(value) is bytes:constants.add(value)
                elif type(value) in (tuple,frozenset):values.extend(value)
        markers={};groups={}
        for occurrence,publication in publications.items():
            ordinal=0
            marker=('SCAR-ANNOTATION:'+str(occurrence)+':0').encode()
            while marker in constants:
                ordinal+=1;marker=('SCAR-ANNOTATION:'+str(occurrence)+':'+str(ordinal)).encode()
            constants.add(marker);markers[marker]=publication
            expression=objects[occurrence].annotation
            objects[occurrence].annotation=ast.copy_location(ast.Constant(value=marker),expression)
            scope=source.context.context_at_slot(publication.annotation).state.environment
            groups.setdefault(scope,{})
        for level in (0,1,2):
            code=compile(tree,'<scar-static-annotation-emissions>',syntax.mode,dont_inherit=True,optimize=level)
            by_scope=self._emissions(code,markers,level)
            for scope in groups:
                emitted=scope in by_scope
                if not emitted:
                    projection=self._projection(objects,scope)
                    selected=self._emissions(compile(projection,'<scar-static-annotation-template>',
                        syntax.mode,dont_inherit=True,optimize=level),markers,level)
                    _require(scope in selected,scope,'compiler template projection omitted a source annotation group')
                    by_scope[scope]=selected[scope]
                groups[scope][level]=P.ScopeAnnotationVariant(level,by_scope[scope],emitted)
        result={}
        for scope,variants in groups.items():
            creator=source.context.material.scopes[scope].parent
            # Use an eligible published annotation anchor, including when the
            # original shared group was first created by a non-simple annotation.
            anchor=variants[0].entries[0].source
            result[creator]=P.ScopeAnnotationInterface(anchor,scope,creator,
                tuple(source.capture_paths.captures[scope]),tuple(variants[i] for i in (0,1,2)))
        return result

    def interface(self,creator):
        if self._interfaces is None:self._interfaces=self._collect()
        return self._interfaces.get(creator)

    def request(self,scope):
        creator=self.source.context.material.scopes[scope].parent
        result=self.interface(creator)
        _require(result is not None and result.scope==scope,scope,'scope has no original variable annotation request')
        return result

    def registration(self,reference,site):
        request=self.request(reference.scope)
        _require(reference.optimization_level in (0,1,2) and reference.index>=0
            and any(e.source==site and e.registration==reference for e in request.variants[reference.optimization_level].entries),
            reference,'registration must identify a source-bound compiler/template emission of this slot')
        return reference


def _require_state(condition, path, detail):
    _require(condition, path, detail, code='ANNOTATION_STATE_INTERFACE')


class AnnotationStateIndex:
    """Project namespace access and native registration inputs for a publication."""

    def __init__(self,source):
        self.source=source;self._parameters={}

    def access(self,creator,name,*,request=False):
        source=self.source;kind=source.context.material.scopes[creator].kind
        _require_state(kind in ('ROOT_EXEC','ROOT_INTERACTIVE','CLASS'),creator,'annotation state requires its original class/root scope')
        if name is P.AnnotationStateName.CONDITIONAL and kind=='CLASS':
            rows=[(i,r) for i,r in enumerate(source.frontend.bindings.implicit)
                if r.scope==creator and source.frontend.namespace.name(r.name)==name.value and r.binding is not None]
            _require_state(len(rows)==1,creator,'class registry requires its original implicit protocol binding')
            index,row=rows[0];binding=source.frontend.bindings.bindings[row.binding]
            _require_state(binding.owner==creator and binding.domain=='CLASS_PROTOCOL',creator,'registry must use the original class protocol cell')
            return P.AnnotationStateAccess(creator,P.AnnotationLookupKind.CLASS_PROTOCOL_CELL,name,row.binding,index,row.read_plan)
        lookup=P.AnnotationLookupKind.GLOBALS_CHAIN if request else P.AnnotationLookupKind.LOCALS_CHAIN
        return P.AnnotationStateAccess(creator,lookup,name)

    def publication_access(self,publication):
        scope=self.source.context.entry_context(publication.annotation.occurrence).environment
        name=(P.AnnotationStateName.ANNOTATIONS if publication.publication is P.AnnotationPublicationKind.STRINGIZED
            else P.AnnotationStateName.CONDITIONAL)
        return self.access(scope,name)

    def request_access(self,scope):
        creator=self.source.context.material.scopes[scope].parent
        return self.access(creator,P.AnnotationStateName.CONDITIONAL,request=True)

    def key(self,publication):
        source=self.source
        declarations = source.frontend.declarations
        names = {declarations.names[name].value for ordinal in source.role_ports(publication.name)
            if (name := declarations.ports[ordinal].name) is not None}
        _require_state(len(names)==1,publication.name,'annotation key needs its original projected declaration name')
        return names.pop()

    def parameter(self,publication):
        _require_state(publication.publication is P.AnnotationPublicationKind.DEFERRED and publication.conditional_registration,
            publication.annotation,'only conditional declarations have compiler registration indices')
        if publication in self._parameters:return self._parameters[publication]
        scope=self.source.context.context_at_slot(publication.annotation).state.environment
        request=self.source.scope_annotations.request(scope);variants=[]
        for variant in request.variants:
            selected=tuple(e.registration for e in variant.entries if e.source==publication.annotation and e.registration is not None)
            _require_state(bool(selected),publication.annotation,'compiler query omitted a conditional source registration')
            status,sites,unverified=self.source.compilation_contexts.sites(scope,variant.optimization_level,emitted=variant.emitted_in_original_unit)
            variants.append(P.AnnotationEmissionVariant(variant.optimization_level,selected,variant.emitted_in_original_unit,
                tuple(site for site in sites if site.registration in selected),status,unverified))
        result=P.AnnotationEmissionParameter(publication,tuple(variants));self._parameters[publication]=result
        return result

    def verify_access(self,access,*,request=False):
        _require_state(access==self.access(access.scope,access.name,request=request),access,
            'namespace/cell source or original implicit/read-plan reference differs')
        return access

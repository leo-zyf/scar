"""Declaration scopes, deferred entries and closure paths over accepted F2/F3.

Source-order parameters and capture references describe logical bindings.
Native frame and free-variable layout have separate compiler evidence.
"""
import __future__
import ast
from collections import defaultdict
from dataclasses import dataclass, replace
from types import CodeType

from scar.analysis.source_syntax import _restore_ast

from scar.ir.program import source_productions as P
from scar.ir.program.protocol_signatures import ExceptionABI


def _require(condition, code, path, detail):
    if not condition:
        from scar.analysis.source_productions import ProductionError
        raise ProductionError(code, str(path), detail)


def _parameter_ports(source, site, scope, operation):
    bindings, declarations = source.frontend.bindings, source.frontend.declarations
    result = []
    for ordinal in source.role_ports(site):
        port = bindings.ports[ordinal]
        if port.declref is None or port.resolution is None:
            continue
        contribution = declarations.contributions[port.declref]
        binding = bindings.resolutions[port.resolution].binding
        if (operation in contribution.operations and binding is not None
                and bindings.bindings[binding].owner == scope):
            result.append(ordinal)
    return result


class DeclarationScopeIndex:
    """Index body scopes by their original declaration occurrence."""

    def __init__(self, scopes):
        self.deferred = {}
        self.classes = {}
        self.type_parameters = {}
        for ordinal, scope in enumerate(scopes):
            if scope.kind in ('FUNCTION', 'LAMBDA', 'GENERATOR_EXPRESSION'):
                self.deferred[scope.owner.occurrence] = ordinal
            elif scope.kind == 'CLASS':
                self.classes[scope.owner.occurrence] = ordinal
            elif scope.kind == 'TYPE_PARAMETERS':
                self.type_parameters[scope.owner.occurrence] = ordinal


class DeferredInterfaceIndex:
    def __init__(self, source):
        self.source = source
        self.scopes = source.declaration_scopes.deferred
        self.captures = source.capture_paths.captures
        self.yield_owners, self.await_owners = set(), set()
        eager, direct_async = set(), set()
        for ordinal, node in enumerate(source.context.syntax.nodes):
            if node.constructor in ('Yield', 'YieldFrom', 'Await'):
                target = self.await_owners if node.constructor == 'Await' else self.yield_owners
                target.add(source.context.entry_context(ordinal).owner.occurrence)
            if node.constructor in ('ListComp', 'SetComp', 'DictComp', 'GeneratorExp'):
                if node.constructor != 'GeneratorExp':
                    eager.add(ordinal)
                if any(source.roles.field(item.reference, 'is_async').scalar.to_python()
                        for item in source.roles.field(ordinal, 'generators').items):
                    direct_async.add(ordinal)
        # Eager comprehensions implicitly await their body; lazy generator
        # construction does not. C1 already routes the first iterable/defaults
        # to their creator, so this worklist respects those phase boundaries.
        self.async_owners = set(self.await_owners) | direct_async
        pending = list(self.async_owners & eager)
        while pending:
            creator = source.context.entry_context(pending.pop()).owner.occurrence
            if creator not in self.async_owners:
                self.async_owners.add(creator)
                if creator in eager:
                    pending.append(creator)
        self._parameters = {}
        self._annotations = {}

    def parameters(self, occurrence):
        if occurrence in self._parameters:
            return self._parameters[occurrence]
        roles, syntax = self.source.roles, self.source.context.syntax
        node = syntax.nodes[occurrence]
        if node.constructor == 'GeneratorExp':
            generators = roles.field(occurrence, 'generators')
            clause = generators.items[0].reference
            site = P.SlotRef(clause, 'FIELD', roles.fields['comprehension']['iter'])
            result = (P.DeferredParameter(P.DeferredParameterKind.INITIAL_ITERATOR, site, None),)
        elif node.constructor in ('Lambda', 'FunctionDef', 'AsyncFunctionDef'):
            arguments = roles.field(occurrence, 'args').reference
            result = []
            for field, kind in (
                ('posonlyargs', P.DeferredParameterKind.POSITIONAL_ONLY),
                ('args', P.DeferredParameterKind.POSITIONAL_OR_KEYWORD),
                ('vararg', P.DeferredParameterKind.VAR_POSITIONAL),
                ('kwonlyargs', P.DeferredParameterKind.KEYWORD_ONLY),
                ('kwarg', P.DeferredParameterKind.VAR_KEYWORD)):
                value = roles.field(arguments, field)
                members = value.items if value.kind == 'LIST' else (value,) if value.kind == 'NODE' else ()
                for item in members:
                    site = P.SlotRef(item.reference, 'FIELD', roles.fields['arg']['arg'])
                    ports = _parameter_ports(self.source, site, self.scopes.get(occurrence),
                        'PARAMETER_INTERFACE')
                    if len(ports) != 1:
                        raise ValueError('formal has no unique original F3 admission/binding port')
                    result.append(P.DeferredParameter(kind, site, ports[0]))
            positional = [i for i, p in enumerate(result) if p.kind in (
                P.DeferredParameterKind.POSITIONAL_ONLY, P.DeferredParameterKind.POSITIONAL_OR_KEYWORD)]
            defaults = roles.field(arguments, 'defaults').items
            if len(defaults) > len(positional):
                raise ValueError('positional defaults exceed original positional formals')
            for ordinal, index in enumerate(positional[len(positional)-len(defaults):]):
                result[index] = replace(result[index], default_source=P.SlotRef(arguments, 'FIELD',
                    roles.fields['arguments']['defaults'], (ordinal,)))
            keywords = [i for i, p in enumerate(result) if p.kind is P.DeferredParameterKind.KEYWORD_ONLY]
            default_values = roles.field(arguments, 'kw_defaults').items
            if len(keywords) != len(default_values):
                raise ValueError('keyword default slots differ from original keyword-only formals')
            for ordinal, (index, value) in enumerate(zip(keywords, default_values)):
                if value.kind == 'NODE':
                    result[index] = replace(result[index], default_source=P.SlotRef(arguments, 'FIELD',
                        roles.fields['arguments']['kw_defaults'], (ordinal,)))
            result = tuple(result)
        else:
            raise ValueError('source occurrence has no declared deferred interface')
        self._parameters[occurrence] = result
        return result

    def interface(self, occurrence, world, exception_abi):
        if occurrence not in self.scopes:
            raise ValueError('deferred body has no original child scope')
        scope = self.scopes[occurrence]
        return P.DeferredEnvironment(scope, self.source.context.material.root_context.environment,
            self.parameters(occurrence), tuple(self.captures[scope]), self.kind(occurrence), world, exception_abi,
            self.source.capture_paths.owned_cells(scope))

    def kind(self, occurrence):
        syntax, roles = self.source.context.syntax, self.source.roles
        node = syntax.nodes[occurrence]
        asynchronous = node.constructor == 'AsyncFunctionDef'
        generator = node.constructor == 'GeneratorExp' or occurrence in self.yield_owners
        if node.constructor == 'GeneratorExp':
            asynchronous = occurrence in self.async_owners
        return (P.DeferredBodyKind.ASYNC_GENERATOR if generator else P.DeferredBodyKind.COROUTINE) if asynchronous else (
            P.DeferredBodyKind.GENERATOR if generator else P.DeferredBodyKind.FUNCTION)

    def annotations(self, occurrence):
        if occurrence in self._annotations:
            return self._annotations[occurrence]
        roles = self.source.roles
        node = self.source.context.syntax.nodes[occurrence]
        if node.constructor not in ('FunctionDef','AsyncFunctionDef'):
            raise ValueError('header annotation interface requires an original function definition')
        parameters = {p.source.occurrence:p for p in self.parameters(occurrence)}
        args = roles.field(occurrence, 'args').reference
        entries = []
        # CPython annotation evaluation order differs from formal admission:
        # regular positional args precede positional-only annotations.
        for name in ('args','posonlyargs','vararg','kwonlyargs','kwarg'):
            value = roles.field(args,name)
            members = value.items if value.kind == 'LIST' else (value,) if value.kind == 'NODE' else ()
            for member in members:
                annotation = roles.field(member.reference,'annotation')
                if annotation.kind != 'NODE':
                    continue
                parameter = parameters[member.reference]
                name_index = self.source.frontend.declarations.ports[parameter.binding_port].name
                key = self.source.frontend.declarations.names[name_index].value
                entries.append(P.FunctionAnnotationEntry(parameter.source,key,
                    P.SlotRef(member.reference,'FIELD',roles.fields['arg']['annotation'])))
        if roles.field(occurrence,'returns').kind == 'NODE':
            entries.append(P.FunctionAnnotationEntry(None,'return',P.SlotRef(occurrence,'FIELD',roles.fields[node.constructor]['returns'])))
        if not entries:
            result = None
        else:
            contexts = [self.source.context.context_at_slot(e.value).state for e in entries]
            scope = contexts[0].environment
            if any(c.environment != scope for c in contexts):
                raise ValueError('function header annotations do not share their original C1 scope')
            regions = [self.source.context.material.regions[c.region] for c in contexts]
            stringized = regions[0].suppression_reason == 'FUTURE_ANNOTATION_STRINGIZED'
            if any((r.suppression_reason == 'FUTURE_ANNOTATION_STRINGIZED') != stringized for r in regions):
                raise ValueError('function header annotation policies differ')
            result = P.FunctionAnnotationInterface(scope,tuple(self.captures[scope]),tuple(entries),stringized)
        self._annotations[occurrence] = result
        return result


def _require_type_parameter(condition,path,detail):
    _require(condition, 'TYPE_PARAMETER_INTERFACE', path, detail)


class TypeParameterInterfaceIndex:
    def __init__(self,source):
        self.source=source
        self.scopes=source.declaration_scopes.type_parameters
        self._specs={}

    def lazy(self,site,*,allow_starred=False):
        _,value=self.source.roles.locate(site)
        if value.kind!='NODE':return None
        context=self.source.context.context_at_slot(site).state
        scope=self.source.context.material.scopes[context.environment]
        _require_type_parameter(scope.kind in ('TYPE_PARAMETER_EVALUATOR','TYPE_ALIAS') and scope.anchor==site,site,
            'lazy expression requires its original type-parameter/alias evaluator scope')
        starred=self.source.context.syntax.nodes[value.reference].constructor=='Starred'
        _require_type_parameter(not starred or allow_starred,site,'starred evaluator is permitted only for TypeVarTuple defaults')
        return P.LazyExpressionInterface(site,context.environment,
            tuple(self.source.deferred.captures[context.environment]),starred)

    def spec(self,occurrence):
        if occurrence in self._specs:return self._specs[occurrence]
        roles=self.source.roles;nodes=self.source.context.syntax.nodes
        kind=nodes[occurrence].constructor
        kinds={'TypeVar':P.TypeParameterKind.TYPE_VAR,'TypeVarTuple':P.TypeParameterKind.TYPE_VAR_TUPLE,'ParamSpec':P.TypeParameterKind.PARAM_SPEC}
        _require_type_parameter(kind in kinds,occurrence,'expected an original type-parameter constructor')
        name=P.SlotRef(occurrence,'FIELD',roles.fields[kind]['name'])
        scope=self.source.context.entry_context(occurrence).environment
        ports = _parameter_ports(self.source, name, scope, 'TYPE_PARAMETER_INTERFACE')
        _require_type_parameter(len(ports)==1,name,'type parameter needs one original declaration/resolution interface')
        bound=self.lazy(P.SlotRef(occurrence,'FIELD',roles.fields[kind]['bound'])) if kind=='TypeVar' else None
        default=self.lazy(P.SlotRef(occurrence,'FIELD',roles.fields[kind]['default_value']),allow_starred=kind=='TypeVarTuple')
        constraints=bound is not None and nodes[roles.locate(bound.source)[1].reference].constructor=='Tuple'
        result=P.TypeParameterSpec(name,kinds[kind],ports[0],bound,default,constraints)
        self._specs[occurrence]=result
        return result

    def environment(self,owner,world,exception_abi):
        _require_type_parameter(owner in self.scopes,owner,'definition has no original type-parameter scope')
        roles=self.source.roles;nodes=self.source.context.syntax.nodes
        parameters=tuple(self.spec(item.reference) for item in roles.field(owner,'type_params').items)
        _require_type_parameter(bool(parameters),owner,'type-parameter factory requires nonempty original parameters')
        seen=False
        for p in parameters:
            _require_type_parameter(not seen or p.default is not None,p.name,'non-default type parameter follows a default')
            seen|=p.default is not None
        inputs=(tuple(p.default_source for p in self.source.deferred.parameters(owner) if p.default_source is not None)
            if nodes[owner].constructor in ('FunctionDef','AsyncFunctionDef') else ())
        scope=self.scopes[owner]
        return P.TypeParameterEnvironment(scope,tuple(self.source.deferred.captures[scope]),parameters,inputs,world,exception_abi,
            self.source.capture_paths.owned_cells(scope))


class DefinitionInterfaceIndex:
    def __init__(self,source):
        self.source=source
        self.classes = source.declaration_scopes.classes
        self.protocols={scope:[] for scope in self.classes.values()}
        for i,binding in enumerate(source.frontend.bindings.bindings):
            if binding.domain=='CLASS_PROTOCOL':self.protocols[binding.owner].append(i)

    def class_body(self,owner,world,exception_abi):
        _require(owner in self.classes, 'DEFINITION_INTERFACE', owner, 'class requires its original C1 child scope')
        scope=self.classes[owner]
        protocols=tuple(self.protocols[scope])
        params=(P.SlotRef(owner,'FIELD',self.source.roles.fields['ClassDef']['type_params'])
            if self.source.roles.field(owner,'type_params').items else None)
        return P.ClassBodyInterface(scope,self.source.context.material.root_context.environment,
            tuple(self.source.deferred.captures[scope]),protocols,self.source.statements.declaration_docstring(owner),
            params,world,exception_abi,self.source.capture_paths.owned_cells(scope))

    def alias(self,owner,exception_abi):
        roles=self.source.roles;nodes=self.source.context.syntax.nodes
        _require(nodes[owner].constructor=='TypeAlias', 'DEFINITION_INTERFACE', owner, 'alias requires an original TypeAlias declaration')
        target=roles.field(owner,'name').reference
        _require(nodes[target].constructor=='Name', 'DEFINITION_INTERFACE', owner, 'alias binder must be an original Name')
        name=P.SlotRef(target,'FIELD',roles.fields['Name']['id'])
        source=P.SlotRef(owner,'FIELD',roles.fields['TypeAlias']['value'])
        expression=self.source.type_parameters.lazy(source)
        params=P.SlotRef(owner,'FIELD',roles.fields['TypeAlias']['type_params']) if roles.field(owner,'type_params').items else None
        return P.BuildTypeAlias(name,expression,params,exception_abi)


@dataclass(frozen=True,slots=True)
class CapturePathHop:
    scope: int
    binding: int
    parent: P.CaptureReference | None


class CaptureInterfaceIndex:
    """Retain original F3 relay ordinals and index missing routes by owner and binding."""

    def __init__(self,source):
        self.source=source;self.data=source.frontend.bindings;self.scopes=source.context.material.scopes
        self.captures=defaultdict(list);self.cells=defaultdict(set);self.transports={};self._by_pair={}
        active=set()
        for demand in self.data.captures:
            if demand.activation in ('TEMPLATE_ONLY','TYPE_GRAMMAR'):continue
            hop=demand.hop
            while hop is not None and hop not in active:
                active.add(hop);hop=self.data.relays[hop].parent
        for index,hop in enumerate(self.data.relays):
            if index in active:
                self._by_pair[hop.scope,hop.binding]=index;self.captures[hop.scope].append(index)
        for index,binding in enumerate(self.data.bindings):
            if binding.cell_required:self.cells[binding.owner].add(index)
        native=source.frontend.symbols;topology=source.frontend.topology
        constants={c.name:c.value for c in source.frontend.native.profile.constants}
        _require('FREE' in constants, 'CAPTURE_TRANSPORT', 'compiler profile', 'original FREE symbol constant is required')
        free=defaultdict(list)
        for index,slot in enumerate(native.slots):
            block=topology.blocks[slot.block]
            if block.source_scope is None or block.projection in ('FUTURE_DETACHED','DETACHED_DESCENDANT'):continue
            if slot.final_fact is not None and native.facts[slot.final_fact].scope_code==constants['FREE']:
                free[block.source_scope,source.frontend.compiler.name(slot.name)].append(index)
        self.native_requirements=[]
        for resolution,row in enumerate(self.data.resolutions):
            witnesses=free.get((row.scope,source.frontend.namespace.name(row.name)),())
            if row.classification=='FREE' and row.binding is not None and witnesses:
                self.native_requirements.append((resolution,tuple(witnesses)))
                self._ensure(row.scope,row.binding,tuple(witnesses))
        # Some implicit reads are recorded at their creator (e.g. conditional
        # annotation registry), while the actual consumer is the request scope.
        # Use its original anchor/read target; compiler witnesses supplement
        # rather than replace the implicit source dependency.
        for ordinal,row in enumerate(self.data.implicit):
            if row.binding is None:continue
            if row.analysis_visibility!='SYMBOL_VISIT' or row.activation in ('TEMPLATE_ONLY','TYPE_GRAMMAR'):continue
            state=source.context.context_at_slot(row.anchor).state
            if source.context.material.regions[state.region].eligibility!='ELIGIBLE':continue
            scope=state.environment
            witnesses=free.get((scope,source.frontend.namespace.name(row.name)),())
            if self.data.bindings[row.binding].owner!=scope:
                self._ensure(scope,row.binding,tuple(witnesses),(ordinal,))

    def _ensure(self,scope,binding,witnesses,implicit=()):
        owner=self.data.bindings[binding].owner;missing=[];current=scope;seen=set()
        while current!=owner and (current,binding) not in self._by_pair:
            _require(current is not None and current not in seen, 'CAPTURE_TRANSPORT', scope, 'capture must reach its original owner through source scope parents')
            seen.add(current);missing.append(current);current=self.scopes[current].parent
        parent=None if current==owner else self._by_pair[current,binding]
        for current in reversed(missing):
            reference=P.DerivedCaptureRef(current,binding)
            self.transports[reference]=P.CaptureTransport(reference,parent,witnesses,implicit)
            self._by_pair[current,binding]=reference;self.captures[current].append(reference);parent=reference
        # Multiple consumers may share a carrier; keep the union of original
        # compiler witnesses without cloning the shared path.
        current=self._by_pair.get((scope,binding))
        while type(current) is P.DerivedCaptureRef:
            row=self.transports[current]
            self.transports[current]=replace(row,compiler_slots=tuple(sorted(set(row.compiler_slots)|set(witnesses))),
                implicit_requirements=tuple(sorted(set(row.implicit_requirements)|set(implicit))))
            current=row.parent
        self.cells[owner].add(binding)

    def relay(self,reference):
        if type(reference) is int:
            _require(0<=reference<len(self.data.relays), 'CAPTURE_TRANSPORT', reference, 'original relay is outside its F3 table')
            return self.data.relays[reference]
        _require(type(reference) is P.DerivedCaptureRef and reference in self.transports, 'CAPTURE_TRANSPORT', reference,
            'derived relay must have original compiler/source evidence')
        # Same logical fields as the old ForwardHop; its reference domain stays separate.
        row=self.transports[reference]
        return CapturePathHop(reference.scope,reference.binding,row.parent)

    def owned_cells(self,scope):return tuple(sorted(self.cells[scope]))

    def export_references(self,references):
        selected=set();pending=list(references)
        while pending:
            reference=pending.pop()
            if type(reference) is int:continue
            _require(reference in self.transports, 'CAPTURE_TRANSPORT', reference, 'unknown derived capture reference')
            if reference in selected:continue
            selected.add(reference);parent=self.transports[reference].parent
            if parent is not None:pending.append(parent)
        return tuple(self.transports[r] for r in sorted(selected,key=lambda r:(r.scope,r.binding)))

    def export_steps(self,steps):
        return self.export_references(s.instruction.source.relay for s in steps if type(s.instruction) is P.ReferenceEnvironment
            and type(s.instruction.source) is P.ClosureEnvironment)


class EvaluationInterfaceIndex:
    """Admit evaluation requests and recover compiler-produced annotation strings."""

    def __init__(self,source):
        self.source=source;self._strings={}

    def environment(self,request,world,abi):
        if type(request) is P.FunctionAnnotationInterface:
            anchor=request.entries[0].value;scope=request.scope;captures=request.captures;default=None
        elif type(request) is P.ScopeAnnotationInterface:
            anchor=request.source;scope=request.scope;captures=request.captures;default=None
        else:
            _require(type(request) is P.LazyExpressionInterface, 'EVALUATION_REQUEST_INTERFACE', 'request','expected an original function annotation or lazy expression interface')
            anchor=request.source;scope=request.scope;captures=request.captures;default=request.format_default
        return P.EvaluationRequestEnvironment(anchor,scope,self.source.context.material.root_context.environment,
            captures,default,world,abi,self.source.capture_paths.owned_cells(scope))

    def expected(self,environment):
        source=self.source;anchor=environment.source;scope=source.context.material.scopes[environment.scope]
        if scope.kind=='ANNOTATION' and scope.owner.kind=='VARIABLE_ANNOTATIONS':
            request=source.scope_annotations.request(environment.scope)
        elif scope.kind in ('TYPE_PARAMETER_EVALUATOR','TYPE_ALIAS'):
            request=source.type_parameters.lazy(anchor,allow_starred=True)
        else:
            _require(scope.kind=='ANNOTATION' and scope.owner.kind=='HEADER_ANNOTATIONS', 'EVALUATION_REQUEST_INTERFACE', anchor,
                'request must use an original evaluator/function-annotation scope')
            owner=scope.owner.occurrence
            # C1 owns function annotations at the FunctionDef, not at each arg.
            request=source.deferred.annotations(owner)
        _require(request is not None, 'EVALUATION_REQUEST_INTERFACE', anchor,'request has no original source interface')
        return self.environment(request,environment.world,environment.exception_abi)

    def stringize(self,site):
        if site in self._strings:return self._strings[site]
        source=self.source;_,selected=source.roles.locate(site)
        state=source.context.context_at_slot(site).state
        region=source.context.material.regions[state.region]
        _require(selected.kind=='NODE' and region.suppression_reason=='FUTURE_ANNOTATION_STRINGIZED', 'EVALUATION_REQUEST_INTERFACE', site,
            'compiler stringization needs an original future annotation expression')
        nodes=source.context.syntax.nodes;order=[];seen=set();pending=[selected.reference]
        def references(value):
            if value.kind=='NODE':return (value.reference,)
            if value.kind=='LIST':return tuple(r for item in value.items for r in references(item))
            return ()
        while pending:
            current=pending.pop()
            if current in seen:continue
            seen.add(current);order.append(current)
            pending.extend(reversed(tuple(r for value in (*nodes[current].fields,*nodes[current].attributes) for r in references(value))))
        remap={original:i for i,original in enumerate(order)}
        def value(item):
            return replace(item,reference=remap[item.reference]) if item.kind=='NODE' else (
                replace(item,items=tuple(value(v) for v in item.items)) if item.kind=='LIST' else item)
        subset=tuple(replace(nodes[o],fields=tuple(value(v) for v in nodes[o].fields),
            attributes=tuple(value(v) for v in nodes[o].attributes)) for o in order)
        expression=_restore_ast(0,subset)
        argument=ast.arg(arg='_value',annotation=expression)
        arguments=ast.arguments(posonlyargs=[],args=[],vararg=argument,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[])
        function=ast.FunctionDef(name='_scar_annotation_query',args=arguments,body=[ast.Pass()],
            decorator_list=[],returns=None,type_comment=None,type_params=[])
        wrapper=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
        code=compile(wrapper,'<scar-static-annotation-string>', 'exec',
            flags=__future__.annotations.compiler_flag,dont_inherit=True,optimize=0)
        # This wrapper has one compiler-created annotation callable and a pass
        # function. Its single map entry loads the fixed key before the string.
        callbacks=[c for c in code.co_consts if type(c) is CodeType and c.co_name=='__annotate__']
        _require(len(callbacks)==1, 'EVALUATION_REQUEST_INTERFACE', site,'controlled compiler wrapper annotation shape differs')
        strings=[v for v in callbacks[0].co_consts if type(v) is str]
        _require(1<=len(strings)<=2 and strings[0]=='_value', 'EVALUATION_REQUEST_INTERFACE', site,'controlled compiler wrapper constant order differs')
        self._strings[site]=strings[-1]
        return strings[-1]


ROOT_FIELDS={'Module':('body','type_ignores'),'Interactive':('body',),'Expression':('body',)}


_CLASS_NAMES=frozenset(('__name__','__module__','__qualname__','__firstlineno__','__type_params__',
    '__doc__','__static_attributes__','__classdictcell__','__classcell__','__annotate_func__'))


def _require_root(condition,path,detail):
    _require(condition, 'ROOT_INTERFACE', path, detail)


def clean_docstring(value):
    """Pinned _PyCompile_CleanDoc, retaining blank lines and ASCII-space rules."""
    value=value.expandtabs()
    value.encode('utf-8')  # the compiler requires a valid UTF-8 representation
    lines=value.split('\n')
    nonblank=[len(line)-len(line.lstrip(' ')) for line in lines[1:] if line.lstrip(' ')]
    margin=min(nonblank,default=0)
    return '\n'.join((lines[0].lstrip(' '),*(line[min(margin,len(line)-len(line.lstrip(' '))):] for line in lines[1:])))


class RootInterfaceIndex:
    def __init__(self,source):
        self.source=source;self.syntax=source.context.syntax;self.root=self.syntax.root
        self.scope=source.context.material.root_context.environment
        self.kind=self.syntax.nodes[self.root].constructor
        annotations = source.scope_annotations
        self.annotations = annotations.by_scope[self.scope]
        self.publications = tuple(publication for occurrence in self.annotations
            if (publication := annotations.publication(occurrence, ExceptionABI.EXCEPTION_WORLD)) is not None)
        self.deferred=any(p.publication is P.AnnotationPublicationKind.DEFERRED for p in self.publications)
        self.future=False
        if self.kind in ('Module','Interactive'):
            for ordinal,value in enumerate(source.roles.field(self.root,'body').items):
                owner=value.reference
                if ordinal==0 and self.syntax.nodes[owner].constructor=='Expr':
                    child=source.roles.field(owner,'value').reference
                    if self.syntax.nodes[child].constructor=='Constant' and type(source.roles.field(child,'value').scalar.to_python()) is str:continue
                if self.syntax.nodes[owner].constructor!='ImportFrom':break
                module=source.roles.field(owner,'module').scalar.to_python()
                if module!='__future__' or source.roles.field(owner,'level').scalar.to_python()!=0:break
                self.future |= any(source.roles.field(v.reference,'name').scalar.to_python()=='annotations'
                    for v in source.roles.field(owner,'names').items)
        for kind,fields in ROOT_FIELDS.items():
            _require_root(tuple(source.roles.fields[kind])==fields,kind,'original root field grammar differs')

    def interface(self,world,abi):
        _require_root(self.kind in ROOT_FIELDS and self.syntax.mode=={'Module':'exec','Interactive':'single','Expression':'eval'}[self.kind],
            self.root,'root constructor and original parser mode differ')
        _require_root(self.source.context.material.scopes[self.scope].kind=={
            'Module':'ROOT_EXEC','Interactive':'ROOT_INTERACTIVE','Expression':'ROOT_EVAL'}[self.kind],
            self.root,'root requires its original external namespace scope')
        return P.RootBodyInterface(P.ProductionOccurrence(self.root),self.scope,world,abi)

    def store_kind(self,name):
        """Implicit codegen_nameop honors an explicit root global directive."""
        declarations=self.source.frontend.declarations;bindings=self.source.frontend.bindings
        explicit=any(row.scope==self.scope and row.classification=='GLOBAL_EXPLICIT' and row.name.kind=='SOURCE'
            and declarations.names[row.name.index].value==name for row in bindings.resolutions)
        return P.NamespaceActionKind.STORE_GLOBAL if explicit else P.NamespaceActionKind.STORE_NAME

    def interactive_expression(self,occurrence):
        context=self.source.context.entry_context(occurrence)
        return self.kind=='Interactive' and context.owner.kind=='ROOT' and context.environment==self.scope


def _require_class(condition,path,detail):
    _require(condition, 'CLASS_NAMESPACE_INTERFACE', path, detail)


class ClassNamespaceIndex:
    def __init__(self,source):
        self.source=source;self.scopes=source.context.material.scopes;self.nodes=source.context.syntax.nodes
        self.resolutions={(r.scope,source.frontend.namespace.name(r.name)):i for i,r in enumerate(source.frontend.bindings.resolutions)}
        self._qualnames={};self._attributes=None
        self.annotations = source.scope_annotations.by_scope

    def access(self,interface,name,action):
        _require_class(name in _CLASS_NAMES and action in (P.BindingActionKind.READ,P.BindingActionKind.WRITE)
            and (action is P.BindingActionKind.WRITE or name=='__name__'),name,'unsupported implicit class name/access')
        return P.AccessClassName(interface,name,action,self.resolutions.get((interface.scope,name)))

    def protocol(self,interface,name):
        bindings=self.source.frontend.bindings.bindings
        found=[i for i in interface.protocol_bindings if self.source.frontend.namespace.name(bindings[i].name)==name]
        _require_class(len(found)<=1,name,'duplicate original class protocol binding')
        if not found:return None
        index=found[0]
        if name=='__conditional_annotations__':
            active=any(d.scope==interface.scope and d.rule=='CONDITIONAL_ANNOTATION_REQUIREMENT' for d in self.source.frontend.bindings.implicit)
        else:active=bindings[index].cell_required
        return index if active else None

    def _name_global(self,owner):
        kind=self.nodes[owner].constructor
        if kind not in ('ClassDef','FunctionDef','AsyncFunctionDef'):return False
        site=P.SlotRef(owner,'FIELD',self.source.roles.fields[kind]['name'])
        bindings = self.source.frontend.bindings
        for ordinal in self.source.role_ports(site):
            resolution = bindings.ports[ordinal].resolution
            if resolution is not None:
                return bindings.resolutions[resolution].classification == 'GLOBAL_EXPLICIT'
        return False

    def qualname(self,scope):
        if scope in self._qualnames:return self._qualnames[scope]
        current=self.scopes[scope]
        if current.parent is None:return ''
        if current.kind=='TYPE_PARAMETERS':return self.qualname(current.parent)
        owner=current.owner.occurrence;kind=self.nodes[owner].constructor
        _require_class(kind in ('FunctionDef','AsyncFunctionDef','ClassDef','Lambda'),scope,
            'class qualname ancestry must consist of original declaration scopes')
        name='<lambda>' if kind=='Lambda' else self.source.roles.field(owner,'name').scalar.to_python()
        parent=current.parent
        if self.scopes[parent].kind=='TYPE_PARAMETERS':parent=self.scopes[parent].parent
        prefix='' if self._name_global(owner) else self.qualname(parent)
        if prefix and self.scopes[parent].kind in ('FUNCTION','LAMBDA'):prefix+='.<locals>'
        result=prefix+'.'+name if prefix else name
        self._qualnames[scope]=result
        return result

    def first_line(self,scope):
        owner=self.scopes[scope].owner.occurrence
        decorators=self.source.roles.field(owner,'decorator_list').items
        node=self.nodes[decorators[0].reference if decorators else owner]
        schema=next(row for row in self.source.context.syntax.profile.ast_schema if row.name==node.constructor)
        return node.attributes[schema.attributes.index('lineno')].scalar.to_python()

    def attributes(self,scope):
        if self._attributes is None:
            always=defaultdict(set);asserted=defaultdict(set);roles=self.source.roles
            for owner,node in enumerate(self.nodes):
                if node.constructor!='Attribute' or roles.context(owner)!='Store':continue
                base=roles.field(owner,'value').reference
                if self.nodes[base].constructor!='Name' or roles.field(base,'id').scalar.to_python()!='self':continue
                context=self.source.context.entry_context(owner)
                if self.source.context.material.regions[context.region].eligibility!='ELIGIBLE':continue
                parent=roles.parents.get(owner)
                if parent is not None and (self.nodes[parent.occurrence].constructor=='AugAssign' or (
                    self.nodes[parent.occurrence].constructor=='AnnAssign' and roles.field(parent.occurrence,'value').kind!='NODE')):continue
                current=context.environment
                while self.source.frontend.topology.scopes[current].disposition=='INLINED':current=self.scopes[current].parent
                current=self.scopes[current].parent  # compiler stack excludes the current code unit
                while current is not None and self.scopes[current].kind!='CLASS':current=self.scopes[current].parent
                if current is None:continue
                conditional=False;original=owner
                while original in roles.parents:
                    original=roles.parents[original].occurrence
                    conditional |= self.nodes[original].constructor=='Assert'
                name=roles.field(owner,'attr').scalar.to_python()
                (asserted if conditional else always)[current].add(name)
            self._attributes={i:(tuple(sorted(always[i])),tuple(sorted(always[i]|asserted[i])))
                for i in self.source.declaration_scopes.classes.values()}
        return self._attributes[scope]

    def deferred_annotations(self,scope):
        if self.source.roots.future:return False
        return any(self.source.scope_annotations.publication(i, ExceptionABI.EXCEPTION_WORLD) is not None
            for i in self.annotations[scope])

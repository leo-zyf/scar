"""Build root and class namespace regions at their admitted source scopes."""
from hashlib import sha256
from pathlib import Path
from scar.analysis.source_callable_rules import build_request_body
from scar.analysis.source_expression_rules import SourceExpressionRules, _BOOL, _REF, _require
from scar.analysis.source_statement_rules import SourceStatementRules, _StatementBuilder
from scar.analysis.source_root_interfaces import FIELDS,clean_docstring
from scar.ir.program import source_productions as P
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.signature_patterns import CountBinding
from scar.ir.program.structured import PythonProtocolKind as K


class SourceRootRules(SourceStatementRules):
    def catalog(self):
        catalog=SourceExpressionRules.catalog(self)
        for row in catalog['rules']:
            row['status']='LOCAL_ROOT_RULE' if row['constructor'] in FIELDS else (
                'NON_EXECUTABLE_GRAMMAR' if row['constructor']=='FunctionType' else 'OUTSIDE_DOMAIN')
            row['purposes']=[P.ProductionPurpose.ROOT.value] if row['constructor'] in FIELDS else []
        return {**catalog,'schema':'scar.source-root-rule-catalog',
            'root_implementation_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'ADMITTED_SOURCE_ROOT_RECIPES_ONLY',
            'entry_contract':'GLOBALS_REFERENCE,LOCALS_REFERENCE,CURRENT_WORLD;NOT_EXTERNAL_ADMISSION_PROOF',
            'child_interfaces':'STATEMENT:(WORLD)->MAY_COMPLETIONS;VALUE:(WORLD)->(REFERENCE,WORLD)'}

    def descriptor_owner(self,occurrence):return None

    def build(self,occurrence=None,purpose=P.ProductionPurpose.ROOT,*,max_steps=100_000,max_regions=100_000,max_visits=1_000_000):
        syntax=self.source.context.syntax
        occurrence=syntax.root if occurrence is None else occurrence
        _require(type(occurrence) is int and occurrence==syntax.root,'SOURCE_REFERENCE',str(occurrence),
            'root rule requires the original F2 entry occurrence')
        _require(purpose is P.ProductionPurpose.ROOT,'PURPOSE_FRONTIER',str(occurrence),'root requires its entry interface')
        for name,value in (('max_steps',max_steps),('max_regions',max_regions),('max_visits',max_visits)):
            _require(type(value) is int and value>=0,'QUERY_ARGUMENT',name,'expected an exact nonnegative integer')
        _require(syntax.nodes[occurrence].constructor in FIELDS,'RULE_FRONTIER',str(occurrence),
            'type grammar has no executable root')
        result=_RootBuilder(self,occurrence,purpose,max_steps,max_regions).build()
        self.source.match(result,max_visits=max_visits)
        return result


def _publish_docstring(builder, region, site, text, namespaces, state, store):
    """Guard the namespace store with the compiler docstring policy."""
    enabled, = builder.emit(region, P.ReadCompilationOption(P.CompilationOption.DOCSTRINGS_ENABLED),
        (), (_BOOL,), may_raise=False)
    yes = builder.region(site, (_REF, _REF, builder.world), region)
    no = builder.region(site, (builder.world,), region)
    value, active = builder.literal(yes, text, builder.parameter(yes, 2), source=site)
    active = store(yes, value, active)
    active, = builder.protocol(yes, K.FRAME_RELEASE_TEMPORARY, (value,), active)
    builder.finish(yes, (active,), (builder.world,))
    builder.finish(no, (builder.parameter(no, 0),), (builder.world,))
    world, = builder.control(region, P.BranchControl(
        P.ProductionRegionCall(yes.id, (*namespaces, state)),
        P.ProductionRegionCall(no.id, (state,))), (enabled,), builder.union(builder.completions(yes), builder.completions(no)))
    return world


class _RootBuilder(_StatementBuilder):
    def store(self,region,name,value,world,*,fixed_name=False):
        kind=P.NamespaceActionKind.STORE_NAME if fixed_name else self.rules.source.roots.store_kind(name)
        namespace=self.parameter(region,1 if kind is P.NamespaceActionKind.STORE_NAME else 0)
        key,world=self.literal(region,name,world)
        world,=self.emit(region,P.ApplyNamespaceAction(kind,self.rules.exception_abi),(namespace,key,value,world),(self.world,))
        return world

    def build(self):
        source=P.ProductionOccurrence(self.occurrence);index=self.rules.source.roots
        interface=index.interface(self.world,self.rules.exception_abi)
        root=self.region(source,(_REF,_REF,self.world))
        world,=self.emit(root,P.EnterRootBody(interface),tuple(self.parameter(root,i) for i in range(3)),(self.world,),may_raise=False)
        for name in FIELDS[self.node.constructor]:self.mark(name,P.ProductionDispositionKind.METADATA)
        if self.node.constructor=='Expression':
            self.mark('body',P.ProductionDispositionKind.EVALUATED_CHILD)
            value,world=self.evaluate(root,self.site('body'),world)
            self.transfer(root,C.RETURN,(value,world),(_REF,self.world));return self.plan(root,source)
        # CPython installs the module annotate callable before user code,
        # including when user statements later raise and leave a partial module.
        if index.deferred and not index.future:
            request=self.rules.source.scope_annotations.interface(interface.scope)
            _require(request is not None,'SOURCE_ANNOTATION',str(source),'deferred root requires its annotation group')
            body=build_request_body(self,root,request)
            cells,world=self.closure_cells(root,request.captures,world,source=source)
            value,world=self.emit(root,P.BuildScopeAnnotations(interface,request,body.id),
                (self.parameter(root,0),*cells,world),(_REF,self.world))
            world=self.store(root,'__annotate__',value,world)
            world,=self.protocol(root,K.FRAME_RELEASE_TEMPORARY,(value,),world)
        if index.annotations:
            value,world=self.protocol(root,K.BUILD_SET_DISPLAY,(),world,(CountBinding('item_count',0),))
            world=self.store(root,'__conditional_annotations__',value,world,fixed_name=True)
            world,=self.protocol(root,K.FRAME_RELEASE_TEMPORARY,(value,),world)
        if index.annotations and index.future:
            world,=self.emit(root,P.ApplyNamespaceAction(P.NamespaceActionKind.ENSURE_ANNOTATIONS,self.rules.exception_abi),
                (self.parameter(root,1),world),(self.world,))
        docstring=self.rules.source.statements.declaration_docstring(self.occurrence)
        if docstring is not None:
            raw=self.value(docstring).scalar.to_python()
            try:cleaned=clean_docstring(raw)
            except UnicodeEncodeError as error:
                _require(False,'SOURCE_CODEGEN_CONFLICT',str(docstring),str(error))
            world = _publish_docstring(self, root, docstring, cleaned,
                (self.parameter(root, 0), self.parameter(root, 1)), world,
                lambda child, value, state: self.store(child, '__doc__', value, state))
            self.slots[docstring]=P.ProductionDispositionKind.METADATA
        self.mark('body',P.ProductionDispositionKind.EVALUATED_CHILD)
        body=self.region(self.site('body'),(self.world,),root)
        self.suite(body,'body',self.parameter(body,0),metadata_prefix=int(docstring is not None))
        result=self.control(root,P.ApplyRegion(P.ProductionRegionCall(body.id,(world,))),(),self.completions(body))
        if result:
            value,world=self.literal(root,None,result[0])
            self.transfer(root,C.RETURN,(value,world),(_REF,self.world))
        return self.plan(root,source)


def _class_access(b,region,interface,name,world,environment,namespace,value=None):
    action=P.BindingActionKind.READ if value is None else P.BindingActionKind.WRITE
    instruction=b.rules.source.class_names.access(interface,name,action)
    return b.emit(region,instruction,(environment,namespace,*(() if value is None else (value,)),world),
        (_REF,b.world) if value is None else (b.world,))


def _class_protocol(b,region,interface,binding,action,world,environment,value=None):
    return b.emit(region,P.AccessClassProtocol(interface,binding,action),
        (environment,*(() if value is None else (value,)),world),
        (_REF,b.world) if value is None else (b.world,),may_raise=action is P.ClassProtocolActionKind.WRITE_VALUE)


def _constant_store(b,region,interface,name,value,world,environment,namespace):
    reference,world=b.literal(region,value,world)
    world,=_class_access(b,region,interface,name,world,environment,namespace,reference)
    world,=b.protocol(region,K.FRAME_RELEASE_TEMPORARY,(reference,),world)
    return world


def initialize_class_namespace(b,body,interface,world,environment,namespace):
    facts=b.rules.source.class_names
    value,world=_class_access(b,body,interface,'__name__',world,environment,namespace)
    world,=_class_access(b,body,interface,'__module__',world,environment,namespace,value)
    world,=b.protocol(body,K.FRAME_RELEASE_TEMPORARY,(value,),world)
    world=_constant_store(b,body,interface,'__qualname__',facts.qualname(interface.scope),world,environment,namespace)
    world=_constant_store(b,body,interface,'__firstlineno__',facts.first_line(interface.scope),world,environment,namespace)
    if interface.type_parameters is not None:
        value,world=b.emit(body,P.ReadClassTypeParameters(interface),(environment,world),(_REF,b.world),may_raise=False)
        world,=_class_access(b,body,interface,'__type_params__',world,environment,namespace,value)
        world,=b.protocol(body,K.FRAME_RELEASE_TEMPORARY,(value,),world)
    dictionary=facts.protocol(interface,'__classdict__')
    if dictionary is not None:
        world,=_class_protocol(b,body,interface,dictionary,P.ClassProtocolActionKind.WRITE_VALUE,world,environment,namespace)
    conditional=facts.protocol(interface,'__conditional_annotations__')
    if conditional is not None:
        value,world=b.protocol(body,K.BUILD_SET_DISPLAY,(),world,(CountBinding('item_count',0),))
        world,=_class_protocol(b,body,interface,conditional,P.ClassProtocolActionKind.WRITE_VALUE,world,environment,value)
        world,=b.protocol(body,K.FRAME_RELEASE_TEMPORARY,(value,),world)
    if facts.annotations[interface.scope] and b.rules.source.roots.future:
        world,=b.emit(body,P.ApplyNamespaceAction(P.NamespaceActionKind.ENSURE_ANNOTATIONS,interface.exception_abi),
            (namespace,world),(b.world,))
    if interface.docstring is not None:
        def store_docstring(child, value, state):
            current, = _class_access(b, child, interface, '__doc__', state,
                b.parameter(child, 0), b.parameter(child, 1), value)
            return current
        text = clean_docstring(b.value(interface.docstring).scalar.to_python())
        world = _publish_docstring(b, body, interface.docstring, text, (environment, namespace), world, store_docstring)
    return world


def finalize_class_namespace(b,region,interface,world,environment,namespace):
    facts=b.rules.source.class_names
    if facts.deferred_annotations(interface.scope):
        request=b.rules.source.scope_annotations.interface(interface.scope)
        body=build_request_body(b,region,request);owner=P.ProductionOccurrence(facts.scopes[interface.scope].owner.occurrence)
        globals_ref,world=b.environment(region,P.GlobalsEnvironment(interface.globals_scope),world,source=owner)
        cells,world=b.closure_cells(region,request.captures,world,source=owner)
        value,world=b.emit(region,P.BuildClassAnnotations(interface,request,body.id),(globals_ref,*cells,world),(_REF,b.world))
        world,=_class_access(b,region,interface,'__annotate_func__',world,environment,namespace,value)
        world,=b.protocol(region,K.FRAME_RELEASE_TEMPORARY,(value,),world)
    plain,with_asserts=facts.attributes(interface.scope)
    if plain==with_asserts:
        world=_constant_store(b,region,interface,'__static_attributes__',plain,world,environment,namespace)
    else:
        enabled,=b.emit(region,P.ReadCompilationOption(P.CompilationOption.ASSERTIONS_ENABLED),(),(_BOOL,),may_raise=False)
        branches=[];regions=[]
        for values in (with_asserts,plain):
            child=b.region(region.source,(_REF,_REF,b.world),region)
            state=_constant_store(b,child,interface,'__static_attributes__',values,b.parameter(child,2),b.parameter(child,0),b.parameter(child,1))
            b.finish(child,(state,),(b.world,));regions.append(child)
            branches.append(P.ProductionRegionCall(child.id,(environment,namespace,world)))
        world,=b.control(region,P.BranchControl(*branches),(enabled,),b.union(*(b.completions(child) for child in regions)))
    dictionary=facts.protocol(interface,'__classdict__')
    if dictionary is not None:
        value,world=_class_protocol(b,region,interface,dictionary,P.ClassProtocolActionKind.CELL_REFERENCE,world,environment)
        world,=_class_access(b,region,interface,'__classdictcell__',world,environment,namespace,value)
        world,=b.protocol(region,K.FRAME_RELEASE_TEMPORARY,(value,),world)
    class_cell=facts.protocol(interface,'__class__')
    if class_cell is not None:
        returned,world=_class_protocol(b,region,interface,class_cell,P.ClassProtocolActionKind.CELL_REFERENCE,world,environment)
        world,=_class_access(b,region,interface,'__classcell__',world,environment,namespace,returned)
    else:returned,world=b.literal(region,None,world)
    b.transfer(region,C.RETURN,(returned,world),(_REF,b.world))

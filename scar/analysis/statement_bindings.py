"""Emit declarations, deferred bodies and namespace bindings."""

from scar.analysis.source_expression_rules import _BOOL, _REF, _require
from scar.ir.program import source_productions as P
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.structured import PythonProtocolKind as K
from scar.ir.program.signature_patterns import CountBinding


def prepare_defaults(b, root, state, env):
    """Definition-time values in source group order, not native frame order."""
    defaults, keywords = [], []
    b.mark('args', P.ProductionDispositionKind.METADATA)
    arguments = b.value(b.site('args')).reference
    roles = b.rules.source.roles
    for name in roles.fields['arguments']:
        b.auxiliary(arguments, name, P.ProductionDispositionKind.METADATA)
    for p in env.parameters:
        for name in roles.fields['arg']:
            b.auxiliary(p.source.occurrence, name, P.ProductionDispositionKind.ROLE_PORT
                if name == 'arg' else P.ProductionDispositionKind.METADATA)
    for name, output in (('defaults', defaults), ('kw_defaults', keywords)):
        site = b.auxiliary(arguments, name, P.ProductionDispositionKind.EVALUATED_CHILD)
        values = b.value(site)
        _require(values.kind == 'LIST', 'RULE_SHAPE', str(site), 'defaults require an original list')
        for ordinal, value in enumerate(values.items):
            if value.kind != 'NODE':
                _require(name == 'kw_defaults' and value.kind == 'SCALAR' and value.scalar.to_python() is None,
                    'RULE_SHAPE', str(site), 'only missing keyword defaults may be None holes')
                continue
            value, state = b.evaluate(root, P.SlotRef(site.occurrence, site.group, site.ordinal, (ordinal,)), state)
            output.append(value)
    return defaults, keywords, state


def complete_deferred_body(b, body, protected, arguments, env):
    """Implicit None return and compiler exception translation after the suite."""
    completions = b.completions(protected)
    routes, outputs = [], []
    for completion in completions:
        if completion.kind is C.FALLTHROUGH:
            handler = b.region(body.source, (b.world,), body, P.ProductionRegionKind.HANDLER)
            value, world = b.literal(handler,None,b.parameter(handler,0))
            b.transfer(handler,C.RETURN,(value,world),(_REF,b.world))
        elif completion.kind is C.RAISE and env.kind is not P.DeferredBodyKind.FUNCTION:
            handler = b.region(body.source,b.raised,body,P.ProductionRegionKind.HANDLER)
            payload = b.emit(handler,P.TranslateEscapingException(env.kind,b.rules.exception_abi),
                tuple(b.parameter(handler,i) for i in range(len(b.raised))),b.raised)
            b.transfer(handler,C.RAISE,payload,b.raised)
        else:
            outputs.append(completion)
            continue
        routes.append(P.CompletionRoute(completion.kind,completion.target,handler.id))
        outputs.extend(b.completions(handler))
    call = P.ProductionRegionCall(protected.id,arguments)
    instruction = P.HandleCompletions(call,tuple(routes)) if routes else P.ApplyRegion(call)
    b.control(body,instruction,(),b.union(tuple(outputs)))


def emit_callable(b, root, state):
    env = b.rules.source.deferred.interface(b.occurrence, b.world, b.rules.exception_abi)
    generator = b.node.constructor == 'GeneratorExp'
    field = 'elt' if generator else 'body'
    b.mark(field, P.ProductionDispositionKind.DEFERRED_CHILD)
    defaults, keywords = [], []
    if generator:
        from scar.analysis.static.lowering.source_comprehension import comprehension_clauses
        clauses = comprehension_clauses(b)
    else:
        defaults, keywords, state = prepare_defaults(b,root,state,env)
    body = b.region(b.site(field), (_REF, *(_REF for _ in env.parameters), b.world), root, P.ProductionRegionKind.DEFERRED)
    code, = b.emit(root, P.DeferRegion(body.id, env), (), (_REF,), may_raise=False)
    globals_ref, state = b.environment(root, P.GlobalsEnvironment(env.globals_scope), state)
    cells, state = b.closure_cells(root, env.captures, state)
    closure = CountBinding('closure_count', len(cells))
    if generator:
        *function_env, state = b.protocol(root, K.MAKE_GENEXPR_FUNCTION, (code, globals_ref, *cells), state, (closure,))
        iterable, state = b.evaluate(root, clauses[0][1], state)
        from scar.analysis.static.lowering.source_comprehension import get_iterator
        iterator, state = get_iterator(b, root, iterable, state, clauses[0][3])
        result = b.protocol(root, K.CREATE_GENEXPR, (*function_env, *cells, iterator), state, (closure,))
    else:
        result = b.protocol(root, K.MAKE_LAMBDA_FUNCTION, (code, globals_ref, *defaults, *keywords, *cells), state,
            (CountBinding('default_count', len(defaults)), CountBinding('keyword_default_count', len(keywords)), closure))
    b.finish(root, result, (_REF, b.world))
    current, = b.emit(body, P.EnterDeferredBody(env), tuple(b.parameter(body, i) for i in range(len(body.parameters))),
        (b.world,), may_raise=False)
    execution = body
    if env.kind is not P.DeferredBodyKind.FUNCTION:
        execution = b.region(body.source,(_REF,b.world) if generator else (b.world,),body)
        execute_world = b.parameter(execution,len(execution.parameters)-1)
    else:
        execute_world = current
    if generator:
        from scar.analysis.static.lowering.source_comprehension import emit_clause_loops
        execute_world = emit_clause_loops(b, execution, clauses, b.parameter(execution,0), execute_world,
            suspension=P.SuspensionKind.ASYNC_YIELD if env.kind is P.DeferredBodyKind.ASYNC_GENERATOR else P.SuspensionKind.YIELD)
        value, execute_world = b.literal(execution, None, execute_world)
    else:
        value, execute_world = b.evaluate(execution, b.site('body'), execute_world)
    b.transfer(execution, C.RETURN, (value, execute_world), (_REF, b.world))
    if execution is not body:
        complete_deferred_body(b,body,execution,(b.parameter(body,1),current) if generator else (current,),env)


def _value(b,region,site,world,*,unpack=False):
    value=b.value(site)
    if unpack:
        _require(b.rules.source.context.syntax.nodes[value.reference].constructor=='Starred',
            'RULE_SHAPE',str(site),'single-item unpack requires its original starred annotation/default')
        b.auxiliary(value.reference,'ctx',P.ProductionDispositionKind.MARKER)
        nested=b.auxiliary(value.reference,'value',P.ProductionDispositionKind.EVALUATED_CHILD)
        value,world=b.evaluate(region,nested,world)
        return b.protocol(region,K.UNPACK_FIXED,(value,),world,(CountBinding('child_count',1),),selectors=(site,))
    return b.evaluate(region,site,world)


def _scope_entry(b,region,entry,mapping,world):
    value,world=_value(b,region,entry.source,world)
    key,world=b.literal(region,entry.key,world,source=entry.name)
    world,=b.protocol(region,K.STORE_SUBSCRIPT,(mapping,key,value),world,selectors=(entry.source,entry.name))
    for reference in (key,value):
        world,=b.protocol(region,K.FRAME_RELEASE_TEMPORARY,(reference,),world)
    b.slots[entry.source]=P.ProductionDispositionKind.DEFERRED_CHILD
    return world


def _scope_variant(b,region,variant,environment,mapping,world):
    for entry in variant.entries:
        if entry.registration is None:
            world=_scope_entry(b,region,entry,mapping,world)
            continue
        access=b.rules.source.annotation_states.request_access(entry.registration.scope)
        registered,world=b.emit(region,P.ReadAnnotationRegistration(entry.registration,b.rules.exception_abi,access),
            (environment,world),(_BOOL,b.world),source=entry.source)
        yes=b.region(entry.source,(_REF,_REF,b.world),region)
        no=b.region(entry.source,(b.world,),region)
        current=_scope_entry(b,yes,entry,b.parameter(yes,1),b.parameter(yes,2))
        b.finish(yes,(current,),(b.world,));b.finish(no,(b.parameter(no,0),),(b.world,))
        world,=b.control(region,P.BranchControl(P.ProductionRegionCall(yes.id,(environment,mapping,world)),
            P.ProductionRegionCall(no.id,(world,))),(registered,),b.union(b.completions(yes),b.completions(no)))
    return world


def _scope_body(b,body,request,world):
    mapping,world=b.protocol(body,K.BUILD_MAP,(),world,(CountBinding('pair_count',0),))
    variants=request.variants
    if all(v.entries==variants[0].entries for v in variants):
        world=_scope_variant(b,body,variants[0],b.parameter(body,0),mapping,world)
        return mapping,world
    # Target options are symbolic compiler inputs, not the analyzer's flags.
    enabled,=b.emit(body,P.ReadCompilationOption(P.CompilationOption.ASSERTIONS_ENABLED),(),(_BOOL,),may_raise=False)
    yes=b.region(request.source,(_REF,_REF,b.world),body)
    no=b.region(request.source,(_REF,_REF,b.world),body)
    current=_scope_variant(b,yes,variants[0],b.parameter(yes,0),b.parameter(yes,1),b.parameter(yes,2))
    b.finish(yes,(current,),(b.world,))
    docs,=b.emit(no,P.ReadCompilationOption(P.CompilationOption.DOCSTRINGS_ENABLED),(),(_BOOL,),may_raise=False)
    children=[]
    for variant in variants[1:]:
        child=b.region(request.source,(_REF,_REF,b.world),no)
        current=_scope_variant(b,child,variant,b.parameter(child,0),b.parameter(child,1),b.parameter(child,2))
        b.finish(child,(current,),(b.world,));children.append(child)
    args=tuple(b.parameter(no,i) for i in range(3))
    current,=b.control(no,P.BranchControl(*(P.ProductionRegionCall(child.id,args) for child in children)),
        (docs,),b.union(*(b.completions(child) for child in children)))
    b.finish(no,(current,),(b.world,))
    args=(b.parameter(body,0),mapping,world)
    world,=b.control(body,P.BranchControl(P.ProductionRegionCall(yes.id,args),P.ProductionRegionCall(no.id,args)),
        (enabled,),b.union(b.completions(yes),b.completions(no)))
    return mapping,world


def build_request_body(b,parent,request):
    environment=b.rules.source.evaluations.environment(request,b.world,b.rules.exception_abi)
    body=b.region(environment.source,(_REF,_REF,b.world),parent,P.ProductionRegionKind.DEFERRED)
    world,=b.emit(body,P.EnterEvaluationRequest(environment),tuple(b.parameter(body,i) for i in range(3)),
        (b.world,),may_raise=False)
    maximum,world=b.literal(body,2,world)
    compared,world=b.emit(body,P.ApplyImplicitComparison(P.ImplicitComparisonKind.GREATER,b.rules.exception_abi),
        (b.parameter(body,1),maximum,world),(_REF,b.world))
    rejected,world=b.protocol(body,K.TRUTH,(compared,),world)
    world,=b.protocol(body,K.FRAME_RELEASE_TEMPORARY,(compared,),world)
    failure=b.region(environment.source,(b.world,),body)
    error,current=b.emit(failure,P.ReadCommonConstant(P.CommonConstantKind.NOT_IMPLEMENTED_ERROR),
        (b.parameter(failure,0),),(_REF,b.world),may_raise=False)
    payload=b.protocol(failure,K.RAISE_EXCEPTION,(error,),current)
    b.transfer(failure,C.RAISE,payload,b.raised)
    accepted=b.region(environment.source,(b.world,),body)
    b.finish(accepted,(b.parameter(accepted,0),),(b.world,))
    world,=b.control(body,P.BranchControl(P.ProductionRegionCall(failure.id,(world,)),
        P.ProductionRegionCall(accepted.id,(world,))),(rejected,),b.union(b.completions(failure),b.completions(accepted)))
    if type(request) is P.FunctionAnnotationInterface:
        pairs=[]
        for entry in request.entries:
            key,world=b.literal(body,entry.name,world,source=entry.parameter or entry.value)
            if request.stringized:
                value,world=b.emit(body,P.DefineAnnotationString(entry.value,b.rules.source.evaluations.stringize(entry.value)),
                    (world,),(_REF,b.world),may_raise=False,source=entry.value)
            else:
                selected=b.value(entry.value).reference
                unpack=b.rules.source.context.syntax.nodes[selected].constructor=='Starred'
                value,world=_value(b,body,entry.value,world,unpack=unpack)
            pairs.extend((key,value));b.slots[entry.value]=P.ProductionDispositionKind.DEFERRED_CHILD
        value,world=b.protocol(body,K.BUILD_MAP,tuple(pairs),world,
            (CountBinding('pair_count',len(request.entries)),),selectors=tuple(e.value for e in request.entries))
    elif type(request) is P.ScopeAnnotationInterface:
        value,world=_scope_body(b,body,request,world)
    else:
        value,world=_value(b,body,request.source,world,unpack=request.unpack_starred)
        b.slots[request.source]=P.ProductionDispositionKind.DEFERRED_CHILD
    b.transfer(body,C.RETURN,(value,world),(_REF,b.world))
    return body


# Generic factory activation and sequential type-parameter creation.
#
# Lazy expressions are captured as original evaluator interfaces; their bodies
# are not executed while creating TypeVar/TypeVarTuple/ParamSpec objects.
def lazy_factory(b,region,expression,world):
    owner=P.ProductionOccurrence(expression.source.occurrence)
    globals_ref,world=b.environment(region,P.GlobalsEnvironment(b.rules.source.context.material.root_context.environment),world,source=owner)
    captures,world=b.closure_cells(region,expression.captures,world,source=owner)
    b.slots[expression.source]=P.ProductionDispositionKind.DEFERRED_CHILD
    request=build_request_body(b,region,expression)
    return b.emit(region,P.BuildLazyExpression(expression,b.rules.exception_abi,request.id),
        (globals_ref,*captures,world),(_REF,b.world),source=expression.source)


def emit_type_parameters(b,region,world,environment):
    values=[]
    for parameter in environment.parameters:
        owner=P.ProductionOccurrence(parameter.name.occurrence)
        for name in b.role.fields[b.rules.source.context.syntax.nodes[parameter.name.occurrence].constructor]:
            b.auxiliary(parameter.name.occurrence,name,P.ProductionDispositionKind.METADATA)
        b.slots[parameter.name]=P.ProductionDispositionKind.ROLE_PORT
        bound=()
        if parameter.bound is not None:
            evaluator,world=lazy_factory(b,region,parameter.bound,world)
            bound=(evaluator,)
        value,world=b.emit(region,P.CreateTypeParameter(parameter,b.rules.exception_abi),(*bound,world),(_REF,b.world),source=owner)
        if parameter.default is not None:
            evaluator,world=lazy_factory(b,region,parameter.default,world)
            world,=b.emit(region,P.SetTypeParameterDefault(parameter,b.rules.exception_abi),
                (value,evaluator,world),(b.world,),source=owner)
        world,=b.emit(region,P.BindTypeParameter(parameter,b.rules.exception_abi),(value,world),(b.world,),source=owner)
        values.append(value)
    result,world=b.protocol(region,K.BUILD_CALL_POSITIONALS,tuple(values),world,
        (CountBinding('item_count',len(values)),),selectors=(b.site('type_params'),))
    return result,world


def enter_factory(b,root,world,inputs):
    environment=b.rules.source.type_parameters.environment(b.occurrence,b.world,b.rules.exception_abi)
    captures,world=b.closure_cells(root,environment.captures,world)
    arguments=(*captures,*inputs,world)
    region=b.region(b.site('type_params'),(*(_REF for _ in range(len(arguments)-1)),b.world),root,P.ProductionRegionKind.TYPE_PARAMETERS)
    result=b.emit(root,P.InvokeTypeParameterFactory(P.ProductionRegionCall(region.id,arguments),environment),(),(_REF,b.world))
    active,=b.emit(region,P.EnterTypeParameters(environment),tuple(b.parameter(region,i) for i in range(len(region.parameters))),
        (b.world,),may_raise=False)
    parameters,active=emit_type_parameters(b,region,active,environment)
    passed=tuple(b.parameter(region,len(captures)+i) for i in range(len(inputs)))
    b.mark('type_params',P.ProductionDispositionKind.DEFERRED_CHILD)
    return result,region,parameters,passed,active


# Class construction and lazy type-alias declarations over original scopes.
#
# Class callbacks are templates passed to the selected builder, not eager calls.
# Namespace setup/finalization remain typed language boundaries; their native
# implementation and the builder's call admission are separate obligations.
def _bind_definition(b,root,value,world):
    site=b.site('name')
    if b.node.constructor=='TypeAlias':
        b.mark('name',P.ProductionDispositionKind.TARGET_ADDRESS)
        world,=b.evaluate(root,site,world,P.ProductionPurpose.STORE,value)
    else:
        b.mark('name',P.ProductionDispositionKind.ROLE_PORT)
        ports=[p for p in b.rules.source.role_ports(site)
            if b.rules.source.frontend.bindings.ports[p].write_target is not None]
        _require(len(ports)==1,'BINDING_INTERFACE_FRONTIER',str(site),'class declaration needs one original name write port')
        world,=b.emit(root,P.AccessBinding(P.BindingActionKind.WRITE,ports[0]),(value,world),(b.world,),source=site)
    world,=b.protocol(root,K.FRAME_RELEASE_TEMPORARY,(value,),world)
    b.finish(root,(world,),(b.world,))


def emit_definition(b,root,world):
    if b.node.constructor=='TypeAlias':
        return _emit_alias(b,root,world)
    decorators=[]
    b.mark('decorator_list',P.ProductionDispositionKind.EVALUATED_CHILD)
    for ordinal in range(len(b.value(b.site('decorator_list')).items)):
        site=b.site('decorator_list',ordinal)
        b.slots[site]=P.ProductionDispositionKind.EVALUATED_CHILD
        value,world=b.evaluate(root,site,world)
        decorators.append((site,value))
    if b.value(b.site('type_params')).items:
        result,factory,parameters,_,active=enter_factory(b,root,world,())
        value,active=_build_class(b,factory,active,parameters)
        b.finish(factory,(value,active),(_REF,b.world))
        value,world=result
    else:
        value,world=_build_class(b,root,world,None)
    for site,decorator in reversed(decorators):
        value,world=b.emit(root,P.ApplyDecorator(site,b.rules.exception_abi),
            (decorator,value,world),(_REF,b.world),source=site)
    _bind_definition(b,root,value,world)


def _build_class(b,region,world,parameters):
    owner=P.ProductionOccurrence(b.occurrence)
    interface=b.rules.source.definitions.class_body(b.occurrence,b.world,b.rules.exception_abi)
    builder,world=b.emit(region,P.ReadBuildClass(b.rules.exception_abi),(world,),(_REF,b.world),source=owner)
    body=b.region(b.site('body'),(_REF,_REF,b.world),region,P.ProductionRegionKind.DEFERRED)
    code,=b.emit(region,P.DeferClassBody(body.id,interface),(),(_REF,),may_raise=False,source=owner)
    globals_ref,world=b.environment(region,P.GlobalsEnvironment(interface.globals_scope),world,source=owner)
    cells,world=b.closure_cells(region,interface.captures,world,source=owner)
    callback,world=b.emit(region,P.BuildClassBody(interface),
        (code,globals_ref,*cells,*(() if parameters is None else (parameters,)),world),(_REF,b.world),source=owner)
    name,world=b.emit(region,P.ConstructValue(P.ConstructionKind.LITERAL,(b.site('name'),)),
        (world,),(_REF,b.world),source=b.site('name'))
    suffix=()
    if parameters is not None:
        generic,world=b.emit(region,P.BuildGenericBase(b.rules.exception_abi),(parameters,world),(_REF,b.world),source=owner)
        suffix=(generic,)
    result=b.invoke_source_call(region,builder,world,argument_field='bases',prefix=(callback,name),suffix=suffix)

    env,namespace=b.parameter(body,0),b.parameter(body,1)
    active,=b.emit(body,P.EnterClassBody(interface),(env,namespace,b.parameter(body,2)),(b.world,),may_raise=False)
    from scar.analysis.source_root_rules import finalize_class_namespace, initialize_class_namespace
    active=initialize_class_namespace(b,body,interface,active,env,namespace)
    protected=b.region(body.source,(b.world,),body)
    b.suite(protected,'body',b.parameter(protected,0),metadata_prefix=int(interface.docstring is not None))
    b.mark('body',P.ProductionDispositionKind.DEFERRED_CHILD)
    if interface.docstring is not None:
        b.slots[interface.docstring]=P.ProductionDispositionKind.METADATA
    routes=[];outputs=[]
    for completion in b.completions(protected):
        if completion.kind is C.FALLTHROUGH:
            final=b.region(body.source,(b.world,_REF,_REF),body,P.ProductionRegionKind.HANDLER)
            finalize_class_namespace(b,final,interface,b.parameter(final,0),b.parameter(final,1),b.parameter(final,2))
            routes.append(P.CompletionRoute(C.FALLTHROUGH,None,final.id,(env,namespace)))
            outputs.extend(b.completions(final))
        else:
            _require(completion.kind is C.RAISE,'RULE_INTERFACE',str(owner),'class body cannot return or jump out of its scope')
            outputs.append(completion)
    call=P.ProductionRegionCall(protected.id,(active,))
    b.control(body,P.HandleCompletions(call,tuple(routes)) if routes else P.ApplyRegion(call),(),b.union(tuple(outputs)))
    return result


def _emit_alias(b,root,world):
    instruction=b.rules.source.definitions.alias(b.occurrence,b.rules.exception_abi)
    owner=P.ProductionOccurrence(b.occurrence)
    if instruction.type_parameters is not None:
        result,factory,parameters,_,active=enter_factory(b,root,world,())
        evaluator,active=lazy_factory(b,factory,instruction.expression,active)
        alias,active=b.emit(factory,instruction,(evaluator,parameters,active),(_REF,b.world),source=owner)
        b.finish(factory,(alias,active),(_REF,b.world))
        alias,world=result
    else:
        evaluator,world=lazy_factory(b,root,instruction.expression,world)
        alias,world=b.emit(root,instruction,(evaluator,world),(_REF,b.world),source=owner)
    _bind_definition(b,root,alias,world)


# Function creation schedules and separately admitted deferred bodies.
#
# Decorators/defaults run in the original creator; generic factory creation,
# annotation requests and body execution retain their distinct C1 scopes.
def emit_function_definition(b,root,state):
    env=b.rules.source.deferred.interface(b.occurrence,b.world,b.rules.exception_abi)
    annotations=b.rules.source.deferred.annotations(b.occurrence)
    docstring=b.rules.source.statements.function_docstring(b.occurrence)
    b.mark('name',P.ProductionDispositionKind.ROLE_PORT)
    b.mark('decorator_list',P.ProductionDispositionKind.EVALUATED_CHILD)
    decorators=[]
    for ordinal in range(len(b.value(b.site('decorator_list')).items)):
        site=b.site('decorator_list',ordinal)
        b.slots[site]=P.ProductionDispositionKind.EVALUATED_CHILD
        value,state=b.evaluate(root,site,state)
        decorators.append((site,value))
    defaults,keywords,state=prepare_defaults(b,root,state,env)
    if b.value(b.site('type_params')).items:
        result,factory,parameters,passed,active=enter_factory(b,root,state,(*defaults,*keywords))
        function,active=_build_function(b,factory,active,env,annotations,docstring,passed[:len(defaults)],passed[len(defaults):])
        active,=b.emit(factory,P.AttachTypeParameters(b.site('type_params'),b.rules.exception_abi),
            (function,parameters,active),(b.world,))
        b.finish(factory,(function,active),(_REF,b.world))
        function,state=result
    else:
        function,state=_build_function(b,root,state,env,annotations,docstring,defaults,keywords)
    for site,decorator in reversed(decorators):
        function,state=b.emit(root,P.ApplyDecorator(site,b.rules.exception_abi),
            (decorator,function,state),(_REF,b.world),source=site)
    ports=[p for p in b.rules.source.role_ports(b.site('name'))
        if b.rules.source.frontend.bindings.ports[p].write_target is not None]
    _require(len(ports)==1,'BINDING_INTERFACE_FRONTIER',str(b.site('name')),'function definition needs one original name write port')
    state,=b.emit(root,P.AccessBinding(P.BindingActionKind.WRITE,ports[0]),(function,state),(b.world,),source=b.site('name'))
    state,=b.protocol(root,K.FRAME_RELEASE_TEMPORARY,(function,),state)
    b.finish(root,(state,),(b.world,))


def _build_function(b,root,state,env,annotations,docstring,defaults,keywords):
    source=P.ProductionOccurrence(b.occurrence)
    globals_ref,state=b.environment(root,P.GlobalsEnvironment(env.globals_scope),state,source=source)
    annotate=None
    if annotations is not None:
        captures,state=b.closure_cells(root,annotations.captures,state,source=source)
        for entry in annotations.entries:
            b.slots[entry.value]=P.ProductionDispositionKind.DEFERRED_CHILD
        request=build_request_body(b,root,annotations)
        annotate,state=b.emit(root,P.BuildFunctionAnnotations(annotations,b.rules.exception_abi,request.id),
            (globals_ref,*captures,state),(_REF,b.world),source=source)
    b.mark('body',P.ProductionDispositionKind.DEFERRED_CHILD)
    body=b.region(b.site('body'),(_REF,*(_REF for _ in env.parameters),b.world),root,P.ProductionRegionKind.DEFERRED)
    code,=b.emit(root,P.DeferRegion(body.id,env),(),(_REF,),may_raise=False,source=source)
    cells,state=b.closure_cells(root,env.captures,state,source=source)
    arguments=(code,globals_ref,*defaults,*keywords,*cells,*(() if annotate is None else (annotate,)),state)
    function,state=b.emit(root,P.BuildFunction(env,annotations,docstring),arguments,(_REF,b.world),source=source)

    current,=b.emit(body,P.EnterDeferredBody(env),tuple(b.parameter(body,i) for i in range(len(body.parameters))),
        (b.world,),may_raise=False)
    protected=b.region(body.source,(b.world,),body)
    b.suite(protected,'body',b.parameter(protected,0),metadata_prefix=int(docstring is not None))
    b.mark('body',P.ProductionDispositionKind.DEFERRED_CHILD)
    if docstring is not None:
        b.slots[docstring]=P.ProductionDispositionKind.METADATA
    complete_deferred_body(b,body,protected,(current,),env)
    return function,state


# Import requests, member paths and namespace writes over original F2/F3.
#
# No imported code is executed by this producer. IMPORT_FROM retains its native
# fallback contract, and star import keeps dynamic namespace effects. Runtime
# importer/frame/registry expansion and exceptional ownership remain unproved.
def _fields(b, alias):
    _require(b.rules.source.context.syntax.nodes[alias].constructor == 'alias',
        'RULE_SHAPE', str(alias), 'import carrier must be an original alias')
    name = b.auxiliary(alias, 'name', P.ProductionDispositionKind.ROLE_PORT)
    renamed = b.auxiliary(alias, 'asname', P.ProductionDispositionKind.METADATA)
    if b.value(renamed).scalar.to_python() is not None:
        b.slots[renamed] = P.ProductionDispositionKind.ROLE_PORT
    return name, renamed


def _bind(b, root, alias, value, world):
    name, renamed = _fields(b, alias)
    site = name if b.value(renamed).scalar.to_python() is None else renamed
    ports = [p for p in b.rules.source.role_ports(site)
        if b.rules.source.frontend.bindings.ports[p].write_target is not None]
    _require(len(ports) == 1, 'BINDING_INTERFACE_FRONTIER', str(site), 'import requires one original namespace write target')
    world, = b.emit(root, P.AccessBinding(P.BindingActionKind.WRITE, ports[0]),
        (value, world), (b.world,), source=site)
    return world


def _release(b, root, value, world):
    return b.protocol(root, K.FRAME_RELEASE_TEMPORARY, (value,), world)[0]


def emit_import_statement(b, root, world):
    index = b.rules.source.statements
    aliases = index.node_items(b.occurrence, 'names')
    _require(bool(aliases), 'RULE_SHAPE', str(b.occurrence), 'import requires aliases')
    for alias in aliases:
        _fields(b, alias)
    if b.node.constructor == 'Import':
        for alias in aliases:
            source = P.ProductionOccurrence(alias)
            requested, world = b.emit(root, index.import_request(b.occurrence, b.rules.exception_abi, alias),
                (world,), (_REF, b.world), source=source)
            fields = b.role.field(alias, 'name').scalar.to_python().split('.')
            renamed = b.role.field(alias, 'asname').scalar.to_python() is not None
            if renamed and len(fields) > 1:
                owner = requested
                for component in range(1, len(fields)):
                    value, world = b.emit(root, index.import_member(alias, component, b.rules.exception_abi),
                        (owner, world), (_REF, b.world), source=source)
                    if component == len(fields)-1:
                        world = _bind(b, root, alias, value, world)
                        world = _release(b, root, value, world)
                    # Intermediate owners end before the next lookup; the last
                    # owner remains until the final binding has completed.
                    world = _release(b, root, owner, world)
                    owner = value
            else:
                world = _bind(b, root, alias, requested, world)
                world = _release(b, root, requested, world)
    else:
        b.mark('module', P.ProductionDispositionKind.ROLE_PORT)
        imported, world = b.emit(root, index.import_request(b.occurrence, b.rules.exception_abi),
            (world,), (_REF, b.world))
        if b.role.field(aliases[0], 'name').scalar.to_python() == '*':
            _require(len(aliases) == 1, 'RULE_SHAPE', str(b.occurrence), 'star import must be the only member')
            world, = b.emit(root, index.import_star(aliases[0], b.rules.exception_abi),
                (imported, world), (b.world,), source=P.ProductionOccurrence(aliases[0]))
        else:
            for alias in aliases:
                value, world = b.emit(root, index.import_member(alias, None, b.rules.exception_abi),
                    (imported, world), (_REF, b.world), source=P.ProductionOccurrence(alias))
                world = _bind(b, root, alias, value, world)
                world = _release(b, root, value, world)
        world = _release(b, root, imported, world)
    b.finish(root, (world,), (b.world,))

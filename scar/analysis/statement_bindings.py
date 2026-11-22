"""Statement bindings."""

from scar.analysis.source_expression_rules import _REF, _require
from scar.analysis.type_parameter_rules import enter_factory, lazy_factory
from scar.ir.program import source_productions as P
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.structured import PythonProtocolKind as K
from scar.analysis.source_callable_rules import build_request_body, complete_deferred_body, prepare_defaults


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
        from scar.analysis.type_parameter_rules import enter_factory
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

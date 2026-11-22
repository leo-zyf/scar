"""Source statement exceptions."""

from scar.analysis.source_expression_rules import _REF, _BOOL, _require
from scar.ir.program import source_productions as P
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.signature_patterns import CountBinding
from scar.ir.program.structured import PythonProtocolKind as K


# Ordered exception dispatch and one shared finally suite over original slots.
#
# Completion routing carries pending values explicitly, never an old world.
# Normal finally completion resumes the pending completion; abrupt completion
# replaces it. Exception installation/restoration remains visible in the recipe.
def _payload(b,region,count):
    return tuple(b.parameter(region,i) for i in range(count))


def _resume(b,region,completion,payload,previous=None):
    if completion.kind is C.RAISE:
        payload=b.emit(region,P.PrepareReraise(b.rules.exception_abi),(payload[0],payload[-1]),b.raised,may_raise=False)
    world=payload[-1]
    if previous is not None:
        world,=b.emit(region,P.RestoreHandledException(),(previous,world),(b.world,),may_raise=False)
    b.transfer(region,completion.kind,(*payload[:-1],world),completion.types,completion.target)


def _restore_failure(b,region,protected,arguments,previous):
    """Admission/type-test failures restore; successful selection stays handled."""
    completions=b.completions(protected)
    routes=[];outputs=list(completions)
    failure=next((c for c in completions if c.kind is C.RAISE),None)
    if failure is not None:
        failed=b.region(protected.source,(*b.raised,_REF),region,P.ProductionRegionKind.HANDLER)
        _resume(b,failed,failure,_payload(b,failed,len(b.raised)),b.parameter(failed,len(b.raised)))
        routes.append(P.CompletionRoute(C.RAISE,None,failed.id,(previous,)))
        outputs=[c for c in completions if c.kind is not C.RAISE]+list(b.completions(failed))
    call=P.ProductionRegionCall(protected.id,arguments)
    return b.control(region,P.HandleCompletions(call,tuple(routes)) if routes else P.ApplyRegion(call),(),b.union(tuple(outputs)))


def _after_cleanup(b,region,cleanup,arguments,pending,payload,previous=None):
    """Route a cleanup suite without copying it for each pending completion."""
    routes=[];outputs=[]
    for completion in b.completions(cleanup):
        normal=completion.kind is C.FALLTHROUGH
        _require(not normal or completion.types==(b.world,),'RULE_INTERFACE',str(cleanup.source),'cleanup normal completion must deliver current world')
        saved=payload[:-1] if normal else ()
        captures=(*saved,*(() if previous is None else (previous,)))
        handler=b.region(cleanup.source,(*completion.types,*(_REF for _ in captures)),region,P.ProductionRegionKind.HANDLER)
        delivered=_payload(b,handler,len(completion.types))
        prior=b.parameter(handler,len(handler.parameters)-1) if previous is not None else None
        if normal:
            carried=tuple(b.parameter(handler,len(completion.types)+i) for i in range(len(saved)))
            _resume(b,handler,pending,(*carried,delivered[-1]),prior)
        else:
            _resume(b,handler,completion,delivered,prior)
        routes.append(P.CompletionRoute(completion.kind,completion.target,handler.id,captures))
        outputs.extend(b.completions(handler))
    call=P.ProductionRegionCall(cleanup.id,arguments)
    b.control(region,P.HandleCompletions(call,tuple(routes)) if routes else P.ApplyRegion(call),(),b.union(tuple(outputs)))


def _clear_name(b,region,world,port,site):
    null,world=b.literal(region,None,world,source=site)
    world,=b.emit(region,P.AccessBinding(P.BindingActionKind.WRITE,port),(null,world),(b.world,),source=site)
    world,=b.emit(region,P.DeleteExceptionBinding(port,b.rules.exception_abi),(world,),(b.world,),source=site)
    return world


def _handler_body(b,region,owner):
    n=len(b.raised);packet=_payload(b,region,n);previous=b.parameter(region,n)
    name=P.SlotRef(owner,'FIELD',b.role.fields['ExceptHandler']['name'])
    body=P.SlotRef(owner,'FIELD',b.role.fields['ExceptHandler']['body'])
    port=b.rules.source.statements.exception_binding(owner)
    world=packet[-1]
    if port is not None:
        admission=b.region(name,(_REF,b.world),region)
        active,=b.emit(admission,P.AccessBinding(P.BindingActionKind.WRITE,port),
            (b.parameter(admission,0),b.parameter(admission,1)),(b.world,),source=name)
        b.finish(admission,(active,),(b.world,))
        world,=_restore_failure(b,region,admission,(packet[0],world),previous)
    protected=b.region(body,(b.world,),region)
    b.suite_at(protected,body,b.parameter(protected,0))
    routes=[];outputs=[]
    for completion in b.completions(protected):
        handler=b.region(body,(*completion.types,_REF),region,P.ProductionRegionKind.HANDLER)
        payload=_payload(b,handler,len(completion.types));prior=b.parameter(handler,len(completion.types))
        if port is None:
            _resume(b,handler,completion,payload,prior)
        elif completion.kind is C.RAISE:
            cleanup=b.region(name,(b.world,),handler)
            state=_clear_name(b,cleanup,b.parameter(cleanup,0),port,name)
            b.finish(cleanup,(state,),(b.world,))
            _after_cleanup(b,handler,cleanup,(payload[-1],),completion,payload,prior)
        else:
            # POP_EXCEPT precedes artificial clear on normal/return/loop exits.
            state,=b.emit(handler,P.RestoreHandledException(),(prior,payload[-1]),(b.world,),may_raise=False)
            state=_clear_name(b,handler,state,port,name)
            _resume(b,handler,completion,(*payload[:-1],state))
        routes.append(P.CompletionRoute(completion.kind,completion.target,handler.id,(previous,)))
        outputs.extend(b.completions(handler))
    result=b.control(region,P.HandleCompletions(P.ProductionRegionCall(protected.id,(world,)),tuple(routes)),(),b.union(tuple(outputs)))
    if result:b.finish(region,result,(b.world,))


def _dispatch(b,region,handlers,ordinal):
    n=len(b.raised);packet=_payload(b,region,n);previous=b.parameter(region,n)
    if ordinal==len(handlers):
        _resume(b,region,P.ProductionCompletion(C.RAISE,b.raised),packet,previous)
        return
    owner=handlers[ordinal]
    site=P.SlotRef(owner,'FIELD',b.role.fields['ExceptHandler']['type'])
    if b.value(site).kind!='NODE':
        _handler_body(b,region,owner)
        return
    matching=b.region(site,(_REF,b.world),region)
    expected,state=b.evaluate(matching,site,b.parameter(matching,1))
    yes,state=b.emit(matching,P.MatchExceptionType(site,b.rules.exception_abi),
        (b.parameter(matching,0),expected,state),(_BOOL,b.world),source=site)
    state,=b.protocol(matching,K.FRAME_RELEASE_TEMPORARY,(expected,),state)
    b.finish(matching,(yes,state),(_BOOL,b.world))
    decision,state=_restore_failure(b,region,matching,(packet[0],packet[-1]),previous)
    arguments=(*packet[:-1],state,previous)
    selected=b.region(b.site('handlers',ordinal),(*b.raised,_REF),region)
    otherwise=b.region(b.site('handlers',ordinal),(*b.raised,_REF),region)
    _handler_body(b,selected,owner)
    _dispatch(b,otherwise,handlers,ordinal+1)
    result=b.control(region,P.BranchControl(P.ProductionRegionCall(selected.id,arguments),
        P.ProductionRegionCall(otherwise.id,arguments)),(decision,),b.union(b.completions(selected),b.completions(otherwise)))
    if result:b.finish(region,result,(b.world,))


def _try_except(b,region,world,handlers):
    if not handlers:
        b.suite(region,'body',world)
        return
    body=b.region(b.site('body'),(b.world,),region)
    b.suite(body,'body',b.parameter(body,0))
    routes=[];outputs=[]
    for completion in b.completions(body):
        if completion.kind is C.FALLTHROUGH:
            handler=b.region(b.site('orelse'),(b.world,),region,P.ProductionRegionKind.HANDLER)
            b.suite(handler,'orelse',b.parameter(handler,0))
        elif completion.kind is C.RAISE:
            handler=b.region(b.site('handlers'),b.raised,region,P.ProductionRegionKind.HANDLER)
            packet=_payload(b,handler,len(b.raised))
            previous,state=b.emit(handler,P.EnterHandledException(b.rules.exception_abi),packet,(_REF,b.world),may_raise=False)
            if b.node.constructor=='TryStar':
                from scar.analysis.source_statement_exceptions import emit_group_dispatch
                from scar.analysis.source_statement_rules import restore_handled_completions
                dispatch=b.region(handler.source,(_REF,b.world),handler)
                emit_group_dispatch(b,dispatch,handlers)
                result=restore_handled_completions(b,handler,dispatch,(packet[0],state),previous)
            else:
                dispatch=b.region(handler.source,(*b.raised,_REF),handler)
                _dispatch(b,dispatch,handlers,0)
                result=b.control(handler,P.ApplyRegion(P.ProductionRegionCall(dispatch.id,(*packet[:-1],state,previous))),(),b.completions(dispatch))
            if result:b.finish(handler,result,(b.world,))
        else:
            outputs.append(completion)
            continue
        routes.append(P.CompletionRoute(completion.kind,completion.target,handler.id))
        outputs.extend(b.completions(handler))
    call=P.ProductionRegionCall(body.id,(world,))
    result=b.control(region,P.HandleCompletions(call,tuple(routes)) if routes else P.ApplyRegion(call),(),b.union(tuple(outputs)))
    if result:b.finish(region,result,(b.world,))


def emit_try_statement(b,root,world):
    handlers=b.rules.source.statements.node_items(b.occurrence,'handlers')
    final=bool(b.value(b.site('finalbody')).items)
    _require(bool(handlers) or final,'SOURCE_CODEGEN_CONFLICT',str(b.occurrence),'try requires handlers or finally')
    _require(bool(handlers) or not b.value(b.site('orelse')).items,'SOURCE_CODEGEN_CONFLICT',str(b.occurrence),'else requires handlers')
    b.mark('handlers',P.ProductionDispositionKind.EVALUATED_CHILD)
    for ordinal,owner in enumerate(handlers):
        _require(b.rules.source.context.syntax.nodes[owner].constructor=='ExceptHandler','RULE_SHAPE',str(owner),'try requires original exception-handler descriptors')
        b.slots[b.site('handlers',ordinal)]=P.ProductionDispositionKind.EVALUATED_CHILD
        typed=b.role.field(owner,'type').kind=='NODE'
        _require(typed or b.node.constructor=='Try' and ordinal==len(handlers)-1,
            'SOURCE_CODEGEN_CONFLICT',str(owner),'bare except must be last and cannot be except*')
        b.auxiliary(owner,'type',P.ProductionDispositionKind.EVALUATED_CHILD if typed else P.ProductionDispositionKind.METADATA)
        b.auxiliary(owner,'name',P.ProductionDispositionKind.ROLE_PORT if b.rules.source.statements.exception_binding(owner) is not None else P.ProductionDispositionKind.METADATA)
        b.auxiliary(owner,'body',P.ProductionDispositionKind.EVALUATED_CHILD)
    if not final:
        _try_except(b,root,world,handlers)
        return
    protected=b.region(P.ProductionOccurrence(b.occurrence),(b.world,),root)
    _try_except(b,protected,b.parameter(protected,0),handlers)
    cleanup=b.region(b.site('finalbody'),(b.world,),root,P.ProductionRegionKind.SHARED)
    b.suite(cleanup,'finalbody',b.parameter(cleanup,0))
    routes=[];outputs=[]
    for completion in b.completions(protected):
        handler=b.region(b.site('finalbody'),completion.types,root,P.ProductionRegionKind.HANDLER)
        payload=_payload(b,handler,len(completion.types));previous=None;state=payload[-1]
        if completion.kind is C.RAISE:
            previous,state=b.emit(handler,P.EnterHandledException(b.rules.exception_abi),payload,(_REF,b.world),may_raise=False)
        _after_cleanup(b,handler,cleanup,(state,),completion,payload,previous)
        routes.append(P.CompletionRoute(completion.kind,completion.target,handler.id))
        outputs.extend(b.completions(handler))
    call=P.ProductionRegionCall(protected.id,(world,))
    result=b.control(root,P.HandleCompletions(call,tuple(routes)) if routes else P.ApplyRegion(call),(),b.union(tuple(outputs)))
    if result:b.finish(root,result,(b.world,))


# Ordered except* partitions, accumulated failures and metadata-aware merge.
#
# The remaining group, original exception and accumulated list are distinct
# references. Split owns its conditional handled-exception update. Only body
# failures are accumulated; admission/cleanup failures leave the dispatcher.
def _group_body(b,region,owner):
    matched,accumulated,world=(b.parameter(region,i) for i in range(3))
    name=P.SlotRef(owner,'FIELD',b.role.fields['ExceptHandler']['name'])
    body=P.SlotRef(owner,'FIELD',b.role.fields['ExceptHandler']['body'])
    port=b.rules.source.statements.exception_binding(owner)
    if port is not None:
        # Binding occurs before the body-failure collection boundary.
        world,=b.emit(region,P.AccessBinding(P.BindingActionKind.WRITE,port),(matched,world),(b.world,),source=name)
    protected=b.region(body,(b.world,),region)
    b.suite_at(protected,body,b.parameter(protected,0))
    routes=[];outputs=[]
    for completion in b.completions(protected):
        _require(completion.kind in (C.FALLTHROUGH,C.RAISE),'SOURCE_CODEGEN_CONFLICT',str(body),
            'return/break/continue cannot escape an except* handler')
        raised=completion.kind is C.RAISE
        handler=b.region(body,(*completion.types,*((_REF,) if raised else ())),region,P.ProductionRegionKind.HANDLER)
        state=b.parameter(handler,len(completion.types)-1)
        if port is not None:state=_clear_name(b,handler,state,port,name)
        if raised:
            state,=b.protocol(handler,K.APPEND_DISPLAY,
                (b.parameter(handler,len(completion.types)),b.parameter(handler,0)),state,selectors=(body,))
        b.finish(handler,(state,),(b.world,))
        routes.append(P.CompletionRoute(completion.kind,completion.target,handler.id,(accumulated,) if raised else ()))
        outputs.extend(b.completions(handler))
    call=P.ProductionRegionCall(protected.id,(world,))
    result=b.control(region,P.HandleCompletions(call,tuple(routes)) if routes else P.ApplyRegion(call),(),b.union(tuple(outputs)))
    if result:b.finish(region,result,(b.world,))


def emit_group_dispatch(b,region,handlers):
    original,world=b.parameter(region,0),b.parameter(region,1)
    remaining=original
    accumulated,world=b.protocol(region,K.BUILD_DISPLAY_LIST,(),world,
        (CountBinding('item_count',0),),selectors=(b.site('handlers'),))
    for ordinal,owner in enumerate(handlers):
        site=P.SlotRef(owner,'FIELD',b.role.fields['ExceptHandler']['type'])
        # Do not short-circuit after the remaining group becomes None.
        expected,world=b.evaluate(region,site,world)
        remaining,matched,world=b.emit(region,P.SplitExceptionGroup(site,b.rules.exception_abi),
            (remaining,expected,world),(_REF,_REF,b.world),source=site)
        world,=b.protocol(region,K.FRAME_RELEASE_TEMPORARY,(expected,),world)
        absent,=b.emit(region,P.ReferenceIsNone(),(matched,),(_BOOL,),may_raise=False,source=site)
        skipped=b.region(b.site('handlers',ordinal),(b.world,),region)
        b.finish(skipped,(b.parameter(skipped,0),),(b.world,))
        selected=b.region(b.site('handlers',ordinal),(_REF,_REF,b.world),region)
        _group_body(b,selected,owner)
        world,=b.control(region,P.BranchControl(P.ProductionRegionCall(skipped.id,(world,)),
            P.ProductionRegionCall(selected.id,(matched,accumulated,world))),(absent,),
            b.union(b.completions(skipped),b.completions(selected)))
    world,=b.protocol(region,K.APPEND_DISPLAY,(accumulated,remaining),world,selectors=(b.site('handlers'),))
    merged,world=b.emit(region,P.MergeExceptionGroup(b.site('handlers'),b.rules.exception_abi),
        (original,accumulated,world),(_REF,b.world),source=b.site('handlers'))
    empty,=b.emit(region,P.ReferenceIsNone(),(merged,),(_BOOL,),may_raise=False)
    done=b.region(region.source,(b.world,),region)
    b.finish(done,(b.parameter(done,0),),(b.world,))
    failed=b.region(region.source,(_REF,b.world),region)
    packet=b.emit(failed,P.PrepareReraise(b.rules.exception_abi),
        (b.parameter(failed,0),b.parameter(failed,1)),b.raised,may_raise=False)
    b.transfer(failed,C.RAISE,packet,b.raised)
    result=b.control(region,P.BranchControl(P.ProductionRegionCall(done.id,(world,)),
        P.ProductionRegionCall(failed.id,(merged,world))),(empty,),b.union(b.completions(done),b.completions(failed)))
    if result:b.finish(region,result,(b.world,))

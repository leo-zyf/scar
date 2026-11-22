"""Source statement rules."""

from scar.analysis.source_expression_rules import (
    _REF,
    _BOOL,
    _require,
    SourceExpressionRules,
    _Builder,
)
from scar.ir.program import source_productions as P
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.signature_patterns import ChoiceBinding, CountBinding
from scar.ir.program.structured import PythonProtocolKind as K
from scar.analysis.await_protocol import emit_delegate_loop
from scar.analysis.source_pattern_interfaces import FIELDS
from hashlib import sha256
from pathlib import Path


# Assignment, assertion and raise schedules over original source ports.
#
# Address components are evaluated once; annotations retain their original C1
# phase. Symbolic compilation options are not observations of the target run.
# Native helper expansion and full reference/exception-context cleanup remain
# separate obligations, as for the other local source recipes.
def _release(b, root, value, state):
    return b.protocol(root, K.FRAME_RELEASE_TEMPORARY, (value,), state)[0]


def _binding(b, root, occurrence, action, state, value=None):
    site = b.auxiliary(occurrence, 'id', P.ProductionDispositionKind.ROLE_PORT)
    b.auxiliary(occurrence, 'ctx', P.ProductionDispositionKind.MARKER)
    field = 'read_plan' if action is P.BindingActionKind.READ else 'write_target'
    ports = [p for p in b.rules.source.role_ports(site) if getattr(b.rules.source.frontend.bindings.ports[p], field) is not None]
    _require(len(ports) == 1, 'BINDING_INTERFACE_FRONTIER', str(site), 'target requires one original namespace access')
    return b.emit(root, P.AccessBinding(action, ports[0]), (state,) if value is None else (value, state),
        (_REF, b.world) if value is None else (b.world,), source=P.ProductionOccurrence(occurrence))


def _inplace(b, root, state):
    b.mark('target', P.ProductionDispositionKind.TARGET_ADDRESS)
    b.mark('op', P.ProductionDispositionKind.MARKER)
    b.mark('value', P.ProductionDispositionKind.EVALUATED_CHILD)
    b.selector('BinOp', b.site('op'))
    target = b.value(b.site('target')).reference
    kind = b.rules.source.context.syntax.nodes[target].constructor
    _require(kind in ('Name', 'Attribute', 'Subscript') and b.role.context(target) == 'Store',
        'RULE_SHAPE', str(target), 'inplace target requires an original name/attribute/subscript Store')
    saved = ()
    sliced = False
    if kind == 'Name':
        old, state = _binding(b, root, target, P.BindingActionKind.READ, state)
    else:
        for name in b.role.fields[kind]:
            b.auxiliary(target, name, P.ProductionDispositionKind.METADATA)
        site = b.auxiliary(target, 'value', P.ProductionDispositionKind.EVALUATED_CHILD)
        receiver, state = b.evaluate(root, site, state)
        saved = (receiver,)
        if kind == 'Attribute':
            selector = b.auxiliary(target, 'attr', P.ProductionDispositionKind.ROLE_PORT)
            old, state = b.action(root, P.LanguageActionKind.ATTRIBUTE_GET, (selector,), (*saved, state))
        else:
            site = b.auxiliary(target, 'slice', P.ProductionDispositionKind.EVALUATED_CHILD)
            index = b.value(site).reference
            sliced = b.rules.source.statements.two_part_slice(index)
            if sliced:
                for name in ('lower', 'upper', 'step'):
                    b.auxiliary(index, name, P.ProductionDispositionKind.METADATA)
                for name in ('lower', 'upper'):
                    part = b.auxiliary(index, name, P.ProductionDispositionKind.EVALUATED_CHILD)
                    if b.value(part).kind == 'NODE':
                        bound, state = b.evaluate(root, part, state)
                    else:
                        bound, state = b.literal(root, None, state, source=part)
                    saved = (*saved, bound)
                old, state = b.protocol(root, K.BINARY_SLICE, saved, state)
            else:
                value, state = b.evaluate(root, site, state)
                saved = (*saved, value)
                old, state = b.action(root, P.LanguageActionKind.ITEM_GET, (), (*saved, state))
    right, state = b.evaluate(root, b.site('value'), state)
    result, state = b.emit(root, P.ApplyInplaceOperator(b.site('op'), b.rules.exception_abi),
        (old, right, state), (_REF, b.world))
    if kind == 'Name':
        state, = _binding(b, root, target, P.BindingActionKind.WRITE, state, result)
    elif kind == 'Attribute':
        state, = b.action(root, P.LanguageActionKind.ATTRIBUTE_SET, (selector,), (*saved, result, state), (b.world,))
    elif sliced:
        # Preserve the existing source-helper order: value, receiver, bounds.
        state, = b.protocol(root, K.STORE_SLICE, (result, *saved), state)
    else:
        state, = b.action(root, P.LanguageActionKind.ITEM_SET, (), (*saved, result, state), (b.world,))
    state = _release(b, root, result, state)
    b.finish(root, (state,), (b.world,))


def _annotation_address_checks(b, root, target, state):
    """Declaration-only attributes/subscripts evaluate addresses, never get/set.

    A declaration slice checks its components individually: constructing a
    tuple or slice object here would add work that CPython does not perform.
    """
    kind = b.rules.source.context.syntax.nodes[target].constructor
    for name in b.role.fields[kind]:
        b.auxiliary(target, name, P.ProductionDispositionKind.METADATA)
    if kind == 'Name':
        b.auxiliary(target, 'id', P.ProductionDispositionKind.ROLE_PORT)
        return state
    _require(kind in ('Attribute', 'Subscript'), 'RULE_SHAPE', str(target), 'invalid annotation target')
    fields = [b.auxiliary(target, 'value', P.ProductionDispositionKind.EVALUATED_CHILD)]
    if kind == 'Subscript':
        fields.append(b.auxiliary(target, 'slice', P.ProductionDispositionKind.EVALUATED_CHILD))
    pending = list(reversed(fields))
    while pending:
        site = pending.pop()
        node = b.value(site)
        _require(node.kind == 'NODE', 'RULE_SHAPE', str(site), 'annotation address requires an expression')
        constructor = b.rules.source.context.syntax.nodes[node.reference].constructor
        if constructor == 'Slice':
            children = []
            for name in ('lower', 'upper', 'step'):
                child = b.auxiliary(node.reference, name, P.ProductionDispositionKind.METADATA)
                if b.value(child).kind == 'NODE':
                    b.slots[child] = P.ProductionDispositionKind.EVALUATED_CHILD
                    children.append(child)
            pending.extend(reversed(children))
        elif constructor == 'Tuple':
            b.auxiliary(node.reference, 'ctx', P.ProductionDispositionKind.MARKER)
            field = b.auxiliary(node.reference, 'elts', P.ProductionDispositionKind.EVALUATED_CHILD)
            pending.extend(P.SlotRef(field.occurrence, field.group, field.ordinal, (i,))
                for i in reversed(range(len(b.value(field).items))))
        else:
            value, state = b.evaluate(root, site, state)
            state = _release(b, root, value, state)
    return state


def _annotated(b, root, state):
    b.mark('target', P.ProductionDispositionKind.TARGET_ADDRESS)
    b.mark('annotation', P.ProductionDispositionKind.SUPPRESSED)
    target = b.value(b.site('target')).reference
    if b.value(b.site('value')).kind == 'NODE':
        b.mark('value', P.ProductionDispositionKind.EVALUATED_CHILD)
        value, state = b.evaluate(root, b.site('value'), state)
        state, = b.evaluate(root, b.site('target'), state, P.ProductionPurpose.STORE, value)
        state = _release(b, root, value, state)
    else:
        state = _annotation_address_checks(b, root, target, state)
    publication = b.rules.source.statements.annotation_publication(b.occurrence, b.rules.exception_abi)
    if publication is not None:
        b.mark('annotation', P.ProductionDispositionKind.DEFERRED_CHILD
            if publication.publication is P.AnnotationPublicationKind.DEFERRED else P.ProductionDispositionKind.METADATA)
        from scar.analysis.annotation_state import publish
        state=publish(b,root,publication,state)
    b.finish(root, (state,), (b.world,))


def _raise(b, root, state):
    arguments = []
    for name in ('exc', 'cause'):
        site = b.site(name)
        value = b.value(site)
        if value.kind == 'NODE':
            _require(name != 'cause' or arguments, 'RULE_SHAPE', str(site), 'raise cause requires an exception expression')
            b.mark(name, P.ProductionDispositionKind.EVALUATED_CHILD)
            argument, state = b.evaluate(root, site, state)
            arguments.append(argument)
    protocol = (K.RERAISE, K.RAISE_EXCEPTION, K.RAISE_WITH_CAUSE)[len(arguments)]
    payload = b.protocol(root, protocol, tuple(arguments), state)
    b.transfer(root, C.RAISE, payload, b.raised)


def _assert(b, root, state):
    enabled, = b.emit(root, P.ReadCompilationOption(P.CompilationOption.ASSERTIONS_ENABLED), (), (_BOOL,), may_raise=False)
    checked = b.region(root.source, (b.world,), root)
    omitted = b.region(root.source, (b.world,), root)
    b.finish(omitted, (b.parameter(omitted, 0),), (b.world,))
    b.mark('test', P.ProductionDispositionKind.EVALUATED_CHILD)
    condition, current = b.condition(checked,b.site('test'),b.parameter(checked,0))
    decision_state = current
    passed = b.region(root.source, (b.world,), checked)
    b.finish(passed, (b.parameter(passed, 0),), (b.world,))
    failed = b.region(root.source, (b.world,), checked)
    current = b.parameter(failed, 0)
    message = ()
    if b.value(b.site('msg')).kind == 'NODE':
        b.mark('msg', P.ProductionDispositionKind.EVALUATED_CHILD)
        value, current = b.evaluate(failed, b.site('msg'), current)
        message = (value,)
    error, current = b.protocol(failed, K.MAKE_ASSERTION_ERROR, message, current,
        (ChoiceBinding('message', 'PRESENT' if message else 'ABSENT'),))
    payload = b.protocol(failed, K.RAISE_EXCEPTION, (error,), current)
    b.transfer(failed, C.RAISE, payload, b.raised)
    result = b.control(checked, P.BranchControl(P.ProductionRegionCall(passed.id, (decision_state,)),
        P.ProductionRegionCall(failed.id, (decision_state,))), (condition,), b.union(b.completions(passed), b.completions(failed)))
    b.finish(checked, result, (b.world,))
    result = b.control(root, P.BranchControl(P.ProductionRegionCall(checked.id, (state,)),
        P.ProductionRegionCall(omitted.id, (state,))), (enabled,), b.union(b.completions(checked), b.completions(omitted)))
    b.finish(root, result, (b.world,))


def emit_atomic_statement(b, root, state):
    {'AugAssign': _inplace, 'AnnAssign': _annotated, 'Assert': _assert, 'Raise': _raise}[b.node.constructor](b, root, state)


# Context-manager schedules and completion-preserving exception restoration.
#
# Each source withitem nests one protected template. Enter failures precede its
# own protection; target stores and later managers are inside it. No execution
# or assumption about the selected context-manager implementation is involved.
def restore_handled_completions(b,region,protected,arguments,previous):
    """Restore on every normal/abrupt completion, using the delivered world."""
    routes=[];outputs=[]
    for completion in b.completions(protected):
        handler=b.region(protected.source,(*completion.types,_REF),region,P.ProductionRegionKind.HANDLER)
        payload=tuple(b.parameter(handler,i) for i in range(len(completion.types)))
        world,=b.emit(handler,P.RestoreHandledException(),
            (b.parameter(handler,len(completion.types)),payload[-1]),(b.world,),may_raise=False)
        b.transfer(handler,completion.kind,(*payload[:-1],world),completion.types,completion.target)
        routes.append(P.CompletionRoute(completion.kind,completion.target,handler.id,(previous,)))
        outputs.extend(b.completions(handler))
    call=P.ProductionRegionCall(protected.id,arguments)
    return b.control(region,P.HandleCompletions(call,tuple(routes)) if routes else P.ApplyRegion(call),(),b.union(tuple(outputs)))


def _await(b,region,value,world,site,method):
    iterator,world=b.emit(region,P.ContextAwaitable(site,method,b.rules.exception_abi),
        (value,world),(_REF,b.world),source=site)
    return emit_delegate_loop(b,region,iterator,world,P.SuspensionKind.AWAIT)


def _invoke(b,region,callable,arguments,world,site,method):
    value,world=b.emit(region,P.InvokeContextMethod(site,method,b.rules.exception_abi),
        (callable,*arguments,world),(_REF,b.world),source=site)
    if method in (P.ContextMethodKind.ASYNC_ENTER,P.ContextMethodKind.ASYNC_EXIT):
        value,world=_await(b,region,value,world,site,method)
    return value,world


def _exceptional_exit(b,handler,site,method):
    payload=tuple(b.parameter(handler,i) for i in range(len(b.raised)))
    exit_method=b.parameter(handler,len(b.raised))
    previous,world=b.emit(handler,P.EnterHandledException(b.rules.exception_abi),payload,(_REF,b.world),may_raise=False)
    protected=b.region(handler.source,(*b.raised[:-1],_REF,b.world),handler)
    original=tuple(b.parameter(protected,i) for i in range(len(b.raised)-1))
    state=b.parameter(protected,len(b.raised))
    kind,value,trace,state=b.emit(protected,P.ContextExceptionArguments(b.rules.exception_abi),
        (*original,state),(_REF,_REF,_REF,b.world),may_raise=False)
    result,state=_invoke(b,protected,b.parameter(protected,len(b.raised)-1),(kind,value,trace),state,site,method)
    suppress,state=b.protocol(protected,K.TRUTH,(result,),state)
    state,=b.protocol(protected,K.FRAME_RELEASE_TEMPORARY,(result,),state)
    yes=b.region(handler.source,(b.world,),protected)
    b.finish(yes,(b.parameter(yes,0),),(b.world,))
    no=b.region(handler.source,b.raised,protected)
    fresh=b.emit(no,P.PrepareReraise(b.rules.exception_abi),
        (b.parameter(no,0),b.parameter(no,len(b.raised)-1)),b.raised,may_raise=False)
    b.transfer(no,C.RAISE,fresh,b.raised)
    b.control(protected,P.BranchControl(P.ProductionRegionCall(yes.id,(state,)),
        P.ProductionRegionCall(no.id,(*original,state))),(suppress,),b.union(b.completions(yes),b.completions(no)))
    result=restore_handled_completions(b,handler,protected,(*payload[:-1],exit_method,world),previous)
    if result:b.finish(handler,result,(b.world,))


def _normal_exit(b,handler,completion,site,method):
    payload=tuple(b.parameter(handler,i) for i in range(len(completion.types)))
    exit_method=b.parameter(handler,len(completion.types))
    null,world=b.literal(handler,None,payload[-1])
    result,world=_invoke(b,handler,exit_method,(null,null,null),world,site,method)
    # A normal exit's result is discarded, never truth-tested.
    world,=b.protocol(handler,K.FRAME_RELEASE_TEMPORARY,(result,),world)
    b.transfer(handler,completion.kind,(*payload[:-1],world),completion.types,completion.target)


def emit_context_statement(b,root,world):
    if b.node.constructor=='AsyncWith':
        owner=b.rules.source.context.entry_context(b.occurrence).owner
        _require(owner.kind=='FUNCTION_BODY'
            and b.rules.source.context.syntax.nodes[owner.occurrence].constructor=='AsyncFunctionDef',
            'SUSPENSION_CONTEXT_FRONTIER',str(b.occurrence),'async with requires its original resumable function owner')
    items=b.value(b.site('items'))
    _require(items.kind=='LIST' and bool(items.items),'RULE_SHAPE',str(b.occurrence),'with requires original manager descriptors')
    b.mark('items',P.ProductionDispositionKind.EVALUATED_CHILD)
    _manager(b,root,world,items.items,0)


def _manager(b,region,world,items,ordinal):
    item=items[ordinal]
    _require(item.kind=='NODE' and b.rules.source.context.syntax.nodes[item.reference].constructor=='withitem',
        'RULE_SHAPE',str(b.occurrence),'manager must retain its original withitem')
    b.slots[b.site('items',ordinal)]=P.ProductionDispositionKind.EVALUATED_CHILD
    site=b.auxiliary(item.reference,'context_expr',P.ProductionDispositionKind.EVALUATED_CHILD)
    target=b.auxiliary(item.reference,'optional_vars',P.ProductionDispositionKind.METADATA)
    asynchronous=b.node.constructor=='AsyncWith'
    enter_kind=P.ContextMethodKind.ASYNC_ENTER if asynchronous else P.ContextMethodKind.ENTER
    exit_kind=P.ContextMethodKind.ASYNC_EXIT if asynchronous else P.ContextMethodKind.EXIT
    manager,world=b.evaluate(region,site,world)
    exit_method,world=b.emit(region,P.ReadContextMethod(site,exit_kind,b.rules.exception_abi),
        (manager,world),(_REF,b.world),source=site)
    enter,world=b.emit(region,P.ReadContextMethod(site,enter_kind,b.rules.exception_abi),
        (manager,world),(_REF,b.world),source=site)
    entered,world=_invoke(b,region,enter,(),world,site,enter_kind)
    protected=b.region(b.site('items',ordinal),(_REF,b.world),region)
    value,state=b.parameter(protected,0),b.parameter(protected,1)
    if b.value(target).kind=='NODE':
        b.slots[target]=P.ProductionDispositionKind.TARGET_ADDRESS
        state,=b.evaluate(protected,target,state,P.ProductionPurpose.STORE,value)
    else:
        _require(b.value(target).kind=='SCALAR' and b.value(target).scalar.to_python() is None,
            'RULE_SHAPE',str(target),'missing as-target requires an explicit None field')
    state,=b.protocol(protected,K.FRAME_RELEASE_TEMPORARY,(value,),state)
    if ordinal+1<len(items):_manager(b,protected,state,items,ordinal+1)
    else:b.suite(protected,'body',state)
    routes=[];outputs=[]
    for completion in b.completions(protected):
        handler=b.region(protected.source,(*completion.types,_REF),region,P.ProductionRegionKind.HANDLER)
        if completion.kind is C.RAISE:_exceptional_exit(b,handler,site,exit_kind)
        else:_normal_exit(b,handler,completion,site,exit_kind)
        routes.append(P.CompletionRoute(completion.kind,completion.target,handler.id,(exit_method,)))
        outputs.extend(b.completions(handler))
    call=P.ProductionRegionCall(protected.id,(entered,world))
    result=b.control(region,P.HandleCompletions(call,tuple(routes)) if routes else P.ApplyRegion(call),(),b.union(tuple(outputs)))
    if result:b.finish(region,result,(b.world,))


# Expandable patterns with capture-bearing success and capture-free failure.
#
# Patterns do not publish namespace bindings. A case publishes only after its
# whole pattern succeeds; a false guard keeps those writes and tries the next
# case with the original subject reference and the delivered current world.
FAIL=P.PatternCompletionKind.FAIL
A=P.PatternActionKind


def _action(b,region,kind,source,arguments,world):
    instruction=b.rules.source.patterns.action(kind,source,b.rules.exception_abi)
    predicate=kind in (A.SEQUENCE_TEST,A.MAPPING_TEST,A.LENGTH_TEST)
    types=(_BOOL,b.world) if predicate else (*(_REF for _ in range(instruction.reference_results)),b.world)
    return b.emit(region,instruction,(*arguments,world),types,
        may_raise=kind not in (A.SEQUENCE_TEST,A.MAPPING_TEST,A.TUPLE_UNPACK),source=source)


def _guard(b,region,source,decision,world,*,fail_on_true=False):
    yes=b.region(source,(b.world,),region);no=b.region(source,(b.world,),region)
    b.finish(yes,(b.parameter(yes,0),),(b.world,))
    b.transfer(no,FAIL,(b.parameter(no,0),),(b.world,))
    branches=(no,yes) if fail_on_true else (yes,no)
    result=b.control(region,P.BranchControl(*(P.ProductionRegionCall(r.id,(world,)) for r in branches)),
        (decision,),b.union(b.completions(yes),b.completions(no)))
    return result[0]


def _present(b,region,source,value,world):
    absent,=b.emit(region,P.ReferenceIsNone(),(value,),(_BOOL,),may_raise=False,source=source)
    return _guard(b,region,source,absent,world,fail_on_true=True)


def _fields(b,owner):
    index=b.rules.source.patterns
    for name in FIELDS[index.nodes[owner].constructor]:
        b.slots[index.site(owner,name)]=P.ProductionDispositionKind.METADATA
    for name in ('name','rest'):
        if name in FIELDS[index.nodes[owner].constructor]:
            capture=index.capture(owner,name)
            if capture is not None:b.slots[capture.source]=P.ProductionDispositionKind.ROLE_PORT


def _metadata(b,owner):
    pending=[owner]
    while pending:
        current=pending.pop();_fields(b,current);pending.extend(b.rules.source.patterns.children(current))


def _child(b,region,site,subject,world):
    index=b.rules.source.patterns;owner=b.value(site).reference
    b.slots[site]=P.ProductionDispositionKind.EVALUATED_CHILD
    child=_pattern(b,region,owner)
    result=b.control(region,P.ApplyRegion(P.ProductionRegionCall(child.id,(subject,world))),(),b.completions(child))
    return result[:-1],result[-1]


def _pattern(b,parent,owner):
    index=b.rules.source.patterns;captures=index.captures(owner);kind=index.nodes[owner].constructor
    region=b.region(P.ProductionOccurrence(owner),(_REF,b.world),parent,P.ProductionRegionKind.PATTERN)
    _fields(b,owner);subject,world=b.parameter(region,0),b.parameter(region,1);values=[]
    source=P.ProductionOccurrence(owner)
    if kind in ('MatchAs','MatchStar'):
        if kind=='MatchAs' and index.children(owner):
            inner,world=_child(b,region,index.site(owner,'pattern'),subject,world);values.extend(inner)
        if index.capture(owner,'name') is not None:values.append(subject)
    elif kind=='MatchValue':
        site=index.site(owner,'value');b.slots[site]=P.ProductionDispositionKind.EVALUATED_CHILD
        expected,world=b.evaluate(region,site,world)
        compared,world=_action(b,region,A.EQUAL,source,(subject,expected),world)
        matched,world=b.protocol(region,K.TRUTH,(compared,),world)
        world,=b.protocol(region,K.FRAME_RELEASE_TEMPORARY,(compared,),world)
        world=_guard(b,region,source,matched,world)
    elif kind=='MatchSingleton':
        literal=index.roles.field(owner,'value').scalar.to_python()
        expected,world=b.literal(region,literal,world,source=index.site(owner,'value'))
        matched,=b.emit(region,P.ReferenceIdentical(),(subject,expected),(_BOOL,),may_raise=False)
        world=_guard(b,region,source,matched,world)
    elif kind=='MatchSequence':
        matched,world=_action(b,region,A.SEQUENCE_TEST,source,(subject,),world)
        world=_guard(b,region,source,matched,world)
        children,star,wild,only=index.sequence(owner)
        if star is None or len(children)>1:
            matched,world=_action(b,region,A.LENGTH_TEST,source,(subject,),world)
            world=_guard(b,region,source,matched,world)
        b.slots[index.site(owner,'patterns')]=P.ProductionDispositionKind.EVALUATED_CHILD
        extracted=()
        if not only and not wild:
            *extracted,world=_action(b,region,A.SEQUENCE_UNPACK,source,(subject,),world)
        for ordinal,child in enumerate(children):
            site=index.site(owner,'patterns',ordinal)
            if only or wild and (ordinal==star or index.wildcard(child)):
                b.slots[site]=P.ProductionDispositionKind.METADATA;_metadata(b,child);continue
            if wild:value,world=_action(b,region,A.SEQUENCE_ITEM,site,(subject,),world)
            else:value=extracted[ordinal]
            inner,world=_child(b,region,site,value,world);values.extend(inner)
    elif kind in ('MatchClass','MatchMapping'):
        if kind=='MatchClass':
            site=index.site(owner,'cls');b.slots[site]=P.ProductionDispositionKind.EVALUATED_CHILD
            cls,world=b.evaluate(region,site,world)
            unpacked,world=_action(b,region,A.CLASS_EXTRACT,source,(subject,cls),world)
            fields=('patterns','kwd_patterns')
            for ordinal in range(len(index.roles.field(owner,'kwd_attrs').items)):
                b.slots[index.site(owner,'kwd_attrs',ordinal)]=P.ProductionDispositionKind.ROLE_PORT
        else:
            matched,world=_action(b,region,A.MAPPING_TEST,source,(subject,),world)
            world=_guard(b,region,source,matched,world)
            keys=index.items(owner,'keys');rest=index.capture(owner,'rest')
            if not keys and rest is None:
                b.finish(region,(world,),(b.world,));return region
            if keys:
                matched,world=_action(b,region,A.LENGTH_TEST,source,(subject,),world)
                world=_guard(b,region,source,matched,world)
            b.slots[index.site(owner,'keys')]=P.ProductionDispositionKind.EVALUATED_CHILD;key_values=[]
            for ordinal in range(len(keys)):
                site=index.site(owner,'keys',ordinal);b.slots[site]=P.ProductionDispositionKind.EVALUATED_CHILD
                value,world=b.evaluate(region,site,world);key_values.append(value)
            packed,world=b.protocol(region,K.BUILD_CALL_POSITIONALS,tuple(key_values),world,
                (CountBinding('item_count',len(keys)),),selectors=(index.site(owner,'keys'),))
            unpacked,world=_action(b,region,A.KEYS_EXTRACT,source,(subject,packed),world)
            fields=('patterns',)
        world=_present(b,region,source,unpacked,world)
        *attributes,world=_action(b,region,A.TUPLE_UNPACK,source,(unpacked,),world)
        ordinal=0
        for name in fields:
            b.slots[index.site(owner,name)]=P.ProductionDispositionKind.EVALUATED_CHILD
            for item in range(len(index.items(owner,name))):
                inner,world=_child(b,region,index.site(owner,name,item),attributes[ordinal],world)
                values.extend(inner);ordinal+=1
        if kind=='MatchMapping' and rest is not None:
            copied,world=b.protocol(region,K.BUILD_MAP,(),world,(CountBinding('pair_count',0),),selectors=(rest.source,))
            world,=b.protocol(region,K.DICT_UPDATE,(copied,subject),world,selectors=(rest.source,))
            for ordinal,key in enumerate(key_values):
                world,=b.action(region,P.LanguageActionKind.ITEM_DELETE,(index.site(owner,'keys',ordinal),),(copied,key,world),(b.world,))
            values.append(copied)
    elif kind=='MatchOr':
        _or_pattern(b,region,owner);return region
    _require(len(values)==len(captures),'RULE_INTERFACE',str(owner),'pattern results differ from original capture interface')
    b.finish(region,(*values,world),(*(_REF for _ in captures),b.world))
    return region


def _or_pattern(b,region,owner):
    index=b.rules.source.patterns;alternatives=index.children(owner);control=index.captures(owner)
    b.slots[index.site(owner,'patterns')]=P.ProductionDispositionKind.EVALUATED_CHILD
    blocks=[b.region(index.site(owner,'patterns',i),(_REF,b.world),region,P.ProductionRegionKind.SHARED) for i in range(len(alternatives))]
    for ordinal in reversed(range(len(blocks))):
        block=blocks[ordinal];subject,world=b.parameter(block,0),b.parameter(block,1)
        child=_pattern(b,block,alternatives[ordinal]);child_captures=index.captures(alternatives[ordinal])
        routes=[];outputs=[]
        for completion in b.completions(child):
            if completion.kind is C.RAISE:outputs.append(completion);continue
            failed=completion.kind is FAIL
            has_next=failed and ordinal+1<len(blocks)
            handler=b.region(child.source,(*completion.types,*((_REF,) if has_next else ())),block,P.ProductionRegionKind.HANDLER)
            state=b.parameter(handler,len(completion.types)-1)
            if not failed:
                positions={capture.name:i for i,capture in enumerate(child_captures)}
                b.finish(handler,(*(b.parameter(handler,positions[capture.name]) for capture in control),state),(*(_REF for _ in control),b.world))
            elif has_next:
                next_block=blocks[ordinal+1]
                result=b.control(handler,P.ApplyRegion(P.ProductionRegionCall(next_block.id,(b.parameter(handler,len(completion.types)),state))),(),b.completions(next_block))
                if result:b.finish(handler,result,(*(_REF for _ in control),b.world))
            else:b.transfer(handler,FAIL,(state,),(b.world,))
            routes.append(P.CompletionRoute(completion.kind,completion.target,handler.id,(subject,) if has_next else ()))
            outputs.extend(b.completions(handler))
        result=b.control(block,P.HandleCompletions(P.ProductionRegionCall(child.id,(subject,world)),tuple(routes)),(),b.union(tuple(outputs)))
        if result:b.finish(block,result,(*(_REF for _ in control),b.world))
    result=b.control(region,P.ApplyRegion(P.ProductionRegionCall(blocks[0].id,(b.parameter(region,0),b.parameter(region,1)))),(),b.completions(blocks[0]))
    if result:b.finish(region,result,(*(_REF for _ in control),b.world))


def _next_case(b,region,following,subject,world):
    if following is None:b.finish(region,(world,),(b.world,));return
    result=b.control(region,P.ApplyRegion(P.ProductionRegionCall(following.id,(subject,world))),(),b.completions(following))
    if result:b.finish(region,result,(b.world,))


def emit_match_statement(b,root,world):
    index=b.rules.source.patterns;cases=b.rules.source.statements.node_items(b.occurrence,'cases')
    _require(bool(cases),'RULE_SHAPE',str(b.occurrence),'match requires cases')
    b.mark('subject',P.ProductionDispositionKind.EVALUATED_CHILD)
    subject,world=b.evaluate(root,b.site('subject'),world)
    b.mark('cases',P.ProductionDispositionKind.EVALUATED_CHILD)
    blocks=[b.region(b.site('cases',i),(_REF,b.world),root,P.ProductionRegionKind.SHARED) for i in range(len(cases))]
    for ordinal in reversed(range(len(cases))):
        case=cases[ordinal];block=blocks[ordinal];following=blocks[ordinal+1] if ordinal+1<len(cases) else None
        b.slots[b.site('cases',ordinal)]=P.ProductionDispositionKind.EVALUATED_CHILD
        pattern=b.auxiliary(case,'pattern',P.ProductionDispositionKind.EVALUATED_CHILD)
        guard=b.auxiliary(case,'guard',P.ProductionDispositionKind.METADATA)
        body=b.auxiliary(case,'body',P.ProductionDispositionKind.EVALUATED_CHILD)
        owner=b.value(pattern).reference;captures=index.captures(owner);guarded=b.value(guard).kind=='NODE'
        _require(not index.irrefutable(owner) or guarded or following is None,'SOURCE_CODEGEN_CONFLICT',str(owner),'unguarded irrefutable case must be last')
        matcher=_pattern(b,block,owner);routes=[];outputs=[]
        for completion in b.completions(matcher):
            if completion.kind is C.RAISE:outputs.append(completion);continue
            failed=completion.kind is FAIL
            needs_subject=following is not None and (failed or guarded)
            handler=b.region(pattern,(*completion.types,*((_REF,) if needs_subject else ())),block,P.ProductionRegionKind.HANDLER)
            current=b.parameter(handler,len(completion.types)-1)
            retained=b.parameter(handler,len(completion.types)) if needs_subject else None
            if failed:_next_case(b,handler,following,retained,current)
            else:
                for position,capture in enumerate(captures):
                    current,=b.emit(handler,P.AccessBinding(P.BindingActionKind.WRITE,capture.port),
                        (b.parameter(handler,position),current),(b.world,),source=capture.source)
                if guarded:
                    b.slots[guard]=P.ProductionDispositionKind.EVALUATED_CHILD
                    decision,current=b.condition(handler,guard,current)
                    accepted=b.region(body,(b.world,),handler);b.suite_at(accepted,body,b.parameter(accepted,0))
                    rejected=b.region(guard,(_REF,b.world) if needs_subject else (b.world,),handler)
                    _next_case(b,rejected,following,b.parameter(rejected,0) if needs_subject else None,
                        b.parameter(rejected,1 if needs_subject else 0))
                    result=b.control(handler,P.BranchControl(P.ProductionRegionCall(accepted.id,(current,)),
                        P.ProductionRegionCall(rejected.id,(retained,current) if needs_subject else (current,))),
                        (decision,),b.union(b.completions(accepted),b.completions(rejected)))
                    if result:b.finish(handler,result,(b.world,))
                else:b.suite_at(handler,body,current)
            routes.append(P.CompletionRoute(completion.kind,completion.target,handler.id,(b.parameter(block,0),) if needs_subject else ()))
            outputs.extend(b.completions(handler))
        result=b.control(block,P.HandleCompletions(P.ProductionRegionCall(matcher.id,(b.parameter(block,0),b.parameter(block,1))),tuple(routes)),(),b.union(tuple(outputs)))
        if result:b.finish(block,result,(b.world,))
    result=b.control(root,P.ApplyRegion(P.ProductionRegionCall(blocks[0].id,(subject,world))),(),b.completions(blocks[0]))
    if result:b.finish(root,result,(b.world,))


# Structured statement schedules over the original F2/F3 recipe grammar.
#
# Expression interfaces remain child obligations. Reference release is explicit:
# discarding a result is not evidence that its computation/effects are dead.
# Suites preserve completion ports; loops are shared templates. Definitions
# and cleanup forms still retain local production frontiers.
_FIELDS = {'Expr': ('value',), 'Pass': (), 'Assign': ('targets', 'value', 'type_comment'),
    'Delete': ('targets',), 'Return': ('value',), 'Global': ('names',), 'Nonlocal': ('names',),
    'If': ('test', 'body', 'orelse'), 'While': ('test', 'body', 'orelse'),
    'For': ('target', 'iter', 'body', 'orelse', 'type_comment'),
    'AsyncFor': ('target', 'iter', 'body', 'orelse', 'type_comment'), 'Break': (), 'Continue': (),
    'AugAssign': ('target', 'op', 'value'), 'AnnAssign': ('target', 'annotation', 'value', 'simple'),
    'Assert': ('test', 'msg'), 'Raise': ('exc', 'cause'),
    'Import': ('names',), 'ImportFrom': ('module', 'names', 'level'),
    'FunctionDef': ('name','args','body','decorator_list','returns','type_comment','type_params'),
    'AsyncFunctionDef': ('name','args','body','decorator_list','returns','type_comment','type_params'),
    'ClassDef': ('name','bases','keywords','body','decorator_list','type_params'),
    'TypeAlias': ('name','type_params','value'),
    'With': ('items','body','type_comment'), 'AsyncWith': ('items','body','type_comment'),
    'Try': ('body','handlers','orelse','finalbody'), 'TryStar': ('body','handlers','orelse','finalbody'),
    'Match': ('subject','cases')}


class SourceStatementRules(SourceExpressionRules):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for constructor, expected in _FIELDS.items():
            _require(tuple(self.source.roles.fields[constructor]) == expected,
                'SOURCE_SCHEMA', constructor, 'original statement field grammar differs from the rule')

    def catalog(self):
        catalog = super().catalog()
        statements = {row.name for row in self.source.context.syntax.profile.ast_schema if 'stmt' in row.bases}
        for row in catalog['rules']:
            row['status'] = ('LOCAL_STATEMENT_RULE' if row['constructor'] in _FIELDS else
                'RULE_FRONTIER' if row['constructor'] in statements else 'OUTSIDE_DOMAIN')
            row['purposes'] = [P.ProductionPurpose.STATEMENT.value] if row['constructor'] in _FIELDS else []
        return {**catalog, 'schema': 'scar.source-statement-rule-catalog',
            'statement_implementation_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope': 'STRUCTURED_STATEMENT_SOURCE_RECIPES_ONLY',
            'statement_interface_sha256': sha256(Path(__file__).with_name('source_statement_interfaces.py').read_bytes()).hexdigest(),
            'statement_completion_constructors': list(self.source.statements.constructors),
            'child_completions': 'CONSERVATIVE_SOURCE_OBLIGATIONS_NOT_CHILD_IMPLEMENTATION_PROOF'}

    def descriptor_owner(self, occurrence):
        return self.source.statements.docstring_owner(occurrence) if self.source.context.syntax.nodes[occurrence].constructor == 'Expr' else None

    def build(self, occurrence, purpose=P.ProductionPurpose.STATEMENT, *, max_steps=100_000, max_regions=100_000, max_visits=1_000_000):
        syntax = self.source.context.syntax
        _require(type(occurrence) is int and 0 <= occurrence < len(syntax.nodes),
            'SOURCE_REFERENCE', 'occurrence', 'expected an original F2 statement occurrence')
        _require(purpose is P.ProductionPurpose.STATEMENT, 'PURPOSE_FRONTIER', str(occurrence),
            'statement rules require the statement interface')
        for name, count in (('max_steps', max_steps), ('max_regions', max_regions), ('max_visits', max_visits)):
            _require(type(count) is int and count >= 0, 'QUERY_ARGUMENT', name, 'expected an exact nonnegative integer')
        constructor = syntax.nodes[occurrence].constructor
        _require(constructor in _FIELDS, 'RULE_FRONTIER', str(occurrence),
            'no local statement rule implemented for ' + constructor)
        _require(self.descriptor_owner(occurrence) is None, 'PARENT_OWNED_SOURCE', str(occurrence),
            'docstring belongs to its parent declaration schedule')
        context = self.source.context.entry_context(occurrence)
        _require(self.source.context.material.regions[context.region].eligibility == 'ELIGIBLE',
            'SUPPRESSED_SOURCE', str(occurrence), 'C1 retains this statement without execution')
        result = _StatementBuilder(self, occurrence, purpose, max_steps, max_regions).build()
        self.source.match(result, max_visits=max_visits)
        return result


class _StatementBuilder(_Builder):
    def build(self):
        source = P.ProductionOccurrence(self.occurrence)
        root = self.region(source, (self.world,))
        current = self.parameter(root, 0)
        kind = self.node.constructor
        for name in _FIELDS[kind]:
            self.mark(name, P.ProductionDispositionKind.METADATA)
        if kind == 'Expr':
            if not self.rules.source.statements.ignored_constant_expression(self.occurrence):
                self.mark('value', P.ProductionDispositionKind.EVALUATED_CHILD)
                value, current = self.evaluate(root, self.site('value'), current)
                if self.rules.source.roots.interactive_expression(self.occurrence):
                    current,=self.emit(root,P.DisplayExpression(self.rules.exception_abi),(value,current),(self.world,))
                current, = self.protocol(root, K.FRAME_RELEASE_TEMPORARY, (value,), current)
        elif kind in ('Assign', 'Delete'):
            self.mark('targets', P.ProductionDispositionKind.TARGET_ADDRESS)
            targets = self.value(self.site('targets'))
            _require(targets.kind == 'LIST' and bool(targets.items), 'RULE_SHAPE', str(source), 'targets must be nonempty')
            value = None
            if kind == 'Assign':
                self.mark('value', P.ProductionDispositionKind.EVALUATED_CHILD)
                value, current = self.evaluate(root, self.site('value'), current)
            for ordinal in range(len(targets.items)):
                site = self.site('targets', ordinal)
                current, = self.evaluate(root, site, current,
                    P.ProductionPurpose.STORE if kind == 'Assign' else P.ProductionPurpose.DELETE, value)
            if kind == 'Assign':
                current, = self.protocol(root, K.FRAME_RELEASE_TEMPORARY, (value,), current)
        elif kind == 'Return':
            _require(self.rules.source.context.entry_context(self.occurrence).owner.kind == 'FUNCTION_BODY',
                'STATEMENT_CONTEXT_FRONTIER', str(source), 'return requires an original function execution owner')
            field = self.value(self.site('value'))
            if field.kind == 'NODE':
                self.mark('value', P.ProductionDispositionKind.EVALUATED_CHILD)
                value, current = self.evaluate(root, self.site('value'), current)
            else:
                _require(field.kind == 'SCALAR' and field.scalar.to_python() is None,
                    'RULE_SHAPE', str(source), 'bare return requires an explicit None field')
                value, current = self.literal(root, None, current)
            self.transfer(root, C.RETURN, (value, current), (_REF, self.world))
            return self.plan(root, source)
        elif kind in ('Global', 'Nonlocal'):
            values = self.value(self.site('names'))
            _require(values.kind == 'LIST' and bool(values.items), 'RULE_SHAPE', str(source), 'directive requires names')
            for ordinal in range(len(values.items)):
                self.slots[self.site('names', ordinal)] = P.ProductionDispositionKind.ROLE_PORT
        elif kind in ('Break', 'Continue'):
            target = self.rules.source.statements.active_loop(source)
            _require(target is not None, 'STATEMENT_CONTEXT_FRONTIER', str(source), 'jump requires an active original loop')
            self.transfer(root, C.BREAK if kind == 'Break' else C.CONTINUE, (current,), (self.world,), target)
            return self.plan(root, source)
        elif kind == 'If':
            self.conditional_statement(root, current)
            return self.plan(root, source)
        elif kind in ('While', 'For', 'AsyncFor'):
            self.loop_statement(root, current)
            return self.plan(root, source)
        elif kind in ('AugAssign', 'AnnAssign', 'Assert', 'Raise'):
            from scar.analysis.source_statement_rules import emit_atomic_statement
            emit_atomic_statement(self, root, current)
            return self.plan(root, source)
        elif kind in ('Import', 'ImportFrom'):
            from scar.analysis.statement_bindings import emit_import_statement
            emit_import_statement(self, root, current)
            return self.plan(root, source)
        elif kind in ('FunctionDef','AsyncFunctionDef'):
            from scar.analysis.statement_bindings import emit_function_definition
            emit_function_definition(self, root, current)
            return self.plan(root, source)
        elif kind in ('ClassDef','TypeAlias'):
            from scar.analysis.statement_bindings import emit_definition
            emit_definition(self,root,current)
            return self.plan(root,source)
        elif kind in ('With','AsyncWith'):
            from scar.analysis.source_statement_rules import emit_context_statement
            emit_context_statement(self,root,current)
            return self.plan(root,source)
        elif kind in ('Try','TryStar'):
            from scar.analysis.source_statement_exceptions import emit_try_statement
            emit_try_statement(self,root,current)
            return self.plan(root,source)
        elif kind == 'Match':
            from scar.analysis.source_statement_rules import emit_match_statement
            emit_match_statement(self,root,current)
            return self.plan(root,source)
        self.finish(root, (current,), (self.world,))
        return self.plan(root, source)

    def interface(self, occurrence):
        return tuple(P.ProductionCompletion(kind,
            self.raised if kind is C.RAISE else (_REF, self.world) if kind is C.RETURN else (self.world,), target)
            for kind, target in self.rules.source.statements.completions(occurrence))

    def suite(self, region, name, current, *, metadata_prefix=0):
        self.suite_at(region,self.site(name),current,metadata_prefix=metadata_prefix)

    def suite_at(self,region,site,current,*,metadata_prefix=0):
        self.slots[site]=P.ProductionDispositionKind.EVALUATED_CHILD
        values = self.value(site)
        _require(values.kind == 'LIST', 'RULE_SHAPE', str(site), 'suite requires an original list field')
        for ordinal, value in enumerate(values.items):
            child = P.SlotRef(site.occurrence,site.group,site.ordinal,(*site.item_path,ordinal))
            if ordinal < metadata_prefix:
                self.slots[child] = P.ProductionDispositionKind.METADATA
                continue
            if current is None:
                self.slots[child] = P.ProductionDispositionKind.NO_NORMAL_PREDECESSOR
                continue
            _require(value.kind == 'NODE', 'RULE_SHAPE', str(child), 'suite item must be an original statement')
            self.slots[child] = P.ProductionDispositionKind.EVALUATED_CHILD
            results = self.control(region, P.EvaluateChild(child, P.ProductionPurpose.STATEMENT),
                (current,), self.interface(value.reference), source=child)
            current = results[0] if results else None
        if current is not None:
            self.finish(region, (current,), (self.world,))

    def conditional_statement(self, root, current):
        self.mark('test', P.ProductionDispositionKind.EVALUATED_CHILD)
        decision, current = self.condition(root,self.site('test'),current)
        alternatives = []
        for name in ('body', 'orelse'):
            child = self.region(self.site(name), (self.world,), root)
            self.suite(child, name, self.parameter(child, 0))
            alternatives.append(child)
        result = self.control(root, P.BranchControl(*(P.ProductionRegionCall(child.id, (current,)) for child in alternatives)),
            (decision,), self.union(*(self.completions(child) for child in alternatives)))
        if result:
            self.finish(root, result, (self.world,))

    def loop_statement(self, root, current):
        source = P.ProductionOccurrence(self.occurrence)
        target = P.SourceLoopTarget(self.occurrence)
        iterator = None
        if self.node.constructor != 'While':
            from scar.analysis.static.lowering.source_comprehension import get_iterator
            self.mark('iter', P.ProductionDispositionKind.EVALUATED_CHILD)
            self.mark('target', P.ProductionDispositionKind.TARGET_ADDRESS)
            iterable, current = self.evaluate(root, self.site('iter'), current)
            iterator, current = get_iterator(self, root, iterable, current, self.node.constructor == 'AsyncFor')
        invariant = 0 if iterator is None else 1
        loop = self.region(source, (self.world,) if iterator is None else (_REF, self.world), root, P.ProductionRegionKind.LOOP)
        condition = self.region(self.site('test' if iterator is None else 'iter'), loop.parameters,
            loop, P.ProductionRegionKind.LOOP_CONDITION)
        state = self.parameter(condition, invariant)
        if iterator is None:
            self.mark('test', P.ProductionDispositionKind.EVALUATED_CHILD)
            decision, state = self.condition(condition,self.site('test'),state)
            self.finish(condition, (decision, state), (_BOOL, self.world))
            control = condition.id
        else:
            asynchronous = self.node.constructor == 'AsyncFor'
            if asynchronous:
                from scar.analysis.await_protocol import emit_delegate_loop
                qualified, state = self.delegation(condition, P.DelegationProtocolKind.GET_ASYNC_NEXT,
                    (self.parameter(condition, 0),), state)
                advanced = emit_delegate_loop(self, condition, qualified, state, P.SuspensionKind.AWAIT)
            else:
                advanced = self.protocol(condition, K.COMPREHENSION_NEXT, (self.parameter(condition, 0),), state,
                    selectors=(self.site('iter'),))
            self.finish(condition, advanced, (_REF, self.world))
            exhausted = self.region(self.site('iter'), self.raised, loop, P.ProductionRegionKind.HANDLER)
            args = (self.parameter(exhausted, 0),)
            state = self.parameter(exhausted, len(self.raised)-1)
            stopped, = (self.delegation(exhausted, P.DelegationProtocolKind.MATCH_STOP_ASYNC_ITERATION, args, state)
                if asynchronous else self.protocol(exhausted, K.MATCH_STOP_ITERATION, args, state))
            self.finish(exhausted, (stopped,), (_BOOL,))
            control = P.IterationCondition(P.ProductionRegionCall(condition.id,
                tuple(self.parameter(loop, i) for i in range(len(loop.parameters)))), exhausted.id, invariant)
        body = self.region(self.site('body'), loop.parameters if iterator is None else (*loop.parameters, _REF),
            loop, P.ProductionRegionKind.LOOP_BODY)
        state = self.parameter(body, invariant)
        if iterator is not None:
            state, = self.evaluate(body, self.site('target'), state, P.ProductionPurpose.STORE, self.parameter(body, invariant+1))
        self.suite(body, 'body', state)
        otherwise = self.region(self.site('orelse'), loop.parameters, loop)
        self.suite(otherwise, 'orelse', self.parameter(otherwise, invariant))
        consumed = {(C.FALLTHROUGH, None), (C.BREAK, target), (C.CONTINUE, target)}
        escaped = tuple(c for c in self.completions(body) if (c.kind, c.target) not in consumed)
        exits = self.union(escaped, self.completions(otherwise), (P.ProductionCompletion(C.RAISE, self.raised),),
            (self.normal((self.world,)),) if any(c.kind is C.BREAK and c.target == target for c in self.completions(body)) else ())
        self.control(loop, P.LoopControl(control, body.id, otherwise.id, invariant, target),
            tuple(self.parameter(loop, i) for i in range(len(loop.parameters))), exits)
        inputs = (current,) if iterator is None else (iterator, current)
        result = self.control(root, P.ApplyRegion(P.ProductionRegionCall(loop.id, inputs)), (), exits)
        if result:
            self.finish(root, result, (self.world,))

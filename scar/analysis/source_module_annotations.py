"""Source module annotations."""

from __future__ import annotations

import ast
import __future__
import dis
from dataclasses import dataclass, replace
from scar.analysis.static.compilation import StaticCodeEvidence
from scar.ir.program.entry_namespaces import (
    ModuleAnnotationCompilerWitness,
    ModuleAnnotationMode as Mode,
    FrameBodyProtocolKind as K,
)
from scar.ir.program.structured import PythonProtocolKind as P
from scar.ir.program.model import (
    Behavior,
    ExitKind,
    LiteralOrigin,
    StateVersionValue,
    ReferenceValue,
)
from scar.ir.program.source import GapKind
from scar.ir.program.state_addresses import (
    InvokeStateOperand,
    ReferenceOwner,
    AddressCoverage,
    Element,
    WholeOwner,
)


# Exact CPython 3.14 module annotation stages; no annotation is executed.
#
# The generated annotation child remains the compiler prelude's deferred child.
# Its captured globals reference does not establish later namespace contents.
# Implicit compiler operands use the original AnnAssign source, never fake ASTs.
def _span(node):
    return node.lineno,node.end_lineno,node.col_offset,node.end_col_offset


@dataclass(frozen=True, slots=True)
class ModuleAnnotationPlan:
    witness: ModuleAnnotationCompilerWitness
    store: dis.Instruction | None
    lookup: dis.Instruction
    value: dis.Instruction
    update: dis.Instruction
    key: dis.Instruction | None = None
    discard: dis.Instruction | None = None
    empty_dict: dis.Instruction | None = None


def module_annotation_plan(context,evidence,node,*,profile):
    """Recognize a unique original statement window, not an evaluation plan."""
    from scar.analysis.source_module_lowering import (
        module_unit,require_root_evidence,_mapping_source_instructions,_module_statement_ref,
    )
    from scar.analysis.source_module_compiler_lowering import compiler_prelude_plan
    if (type(evidence) is not StaticCodeEvidence or not isinstance(node,ast.AnnAssign)
            or node.simple!=1 or not isinstance(node.target,ast.Name)):
        raise ValueError('module annotations currently require an original simple-name AnnAssign')
    require_root_evidence(module_unit(context),evidence,allow_compiler_locals=True)
    pending=list(context.tree.body);owned=False
    while pending:
        current=pending.pop()
        if current is node:owned=True;break
        if not isinstance(current,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Lambda)):
            pending.extend(ast.iter_child_nodes(current))
    if not owned:raise ValueError('annotation statement is not owned by the original module scope')
    for source in (node,node.target,node.annotation):_module_statement_ref(context,source)
    prelude=compiler_prelude_plan(evidence.code,context.tree)
    rows=tuple(row for row in _mapping_source_instructions(context,evidence,profile=profile)
               if row.opname!='EXTENDED_ARG')
    if not rows or rows[0].offset<prelude.first_source_offset:
        raise ValueError('module annotation requires its independently modeled compiler prelude')
    at={row.offset:index for index,row in enumerate(rows)}
    own=[row for row in rows if tuple(row.positions)==_span(node)]
    stores=[row for row in rows if row.opname in ('STORE_NAME','STORE_GLOBAL')
            and row.argval==node.target.id and tuple(row.positions)==_span(node.target)]
    if len(stores)!=(node.value is not None):
        raise ValueError('annotation RHS has no unique original namespace store')
    store=stores[0] if stores else None
    future=bool(evidence.code.co_flags & __future__.annotations.compiler_flag)
    key=discard=None
    if future:
        loads=[row for row in rows if row.opname=='LOAD_CONST' and type(row.argval) is str
               and tuple(row.positions)==_span(node.annotation)]
        if len(loads)!=1:raise ValueError('future annotation lacks its exact compiler string slot')
        value=loads[0]
        if len(own)<3:raise ValueError('future annotation update sequence is incomplete')
        lookup,key,update=own[:3];stages=[value,lookup,key,update]
        if (lookup.opname!='LOAD_NAME' or lookup.argval!='__annotations__'
                or key.opname!='LOAD_CONST' or type(key.argval) is not str or key.argval!=node.target.id
                or update.opname!='STORE_SUBSCR'):
            raise ValueError('future annotation changed its original owner/key/update sequence')
        mode=Mode.FUTURE_STRING;tail=own[3:]
    else:
        if len(own)<4:raise ValueError('lazy annotation registration sequence is incomplete')
        lookup,value,update,discard=own[:4];stages=[lookup,value,update,discard]
        if (lookup.opname!='LOAD_NAME' or lookup.argval!='__conditional_annotations__'
                or value.opname not in ('LOAD_SMALL_INT','LOAD_CONST')
                or type(value.argval) is not int or value.argval<0
                or update.opname!='SET_ADD' or update.arg!=1 or discard.opname!='POP_TOP'):
            raise ValueError('lazy annotation changed its original lookup/index/set/discard sequence')
        if len(prelude.children)!=1:
            raise ValueError('lazy annotation lacks its exact deferred compiler child')
        child=evidence.code.co_consts[prelude.children[0].code_path[0]]
        child_rows=tuple(dis.get_instructions(child,adaptive=False,show_caches=False))
        registrations=[window for index in range(len(child_rows)-2)
            if len(window:=child_rows[index:index+3])==3
            and window[0].opname in ('LOAD_SMALL_INT','LOAD_CONST')
            and type(window[0].argval) is int and window[0].argval==value.argval
            and window[1].opname=='LOAD_GLOBAL' and window[1].argval=='__conditional_annotations__'
            and window[2].opname=='CONTAINS_OP' and window[2].arg==0
            and all(tuple(row.positions)==_span(node) for row in window)]
        if len(registrations)!=1:
            raise ValueError('lazy annotation index does not match this deferred child source test')
        mode=Mode.LAZY_SET;tail=own[4:]
    if tail and not (len(tail)==2 and tail[0].opname=='LOAD_CONST' and tail[0].argval is None
            and tail[1].opname=='RETURN_VALUE'):
        raise ValueError('annotation source span contains extra compiler stages')
    if any(at[right.offset]!=at[left.offset]+1 for left,right in zip(stages,stages[1:])):
        raise ValueError('annotation compiler stages are not one original contiguous window')
    if store is not None and at[stages[0].offset]!=at[store.offset]+1:
        raise ValueError('annotation update must immediately follow its successful RHS store')
    empty_dict=None
    if isinstance(node.value,ast.Dict):
        if node.value.keys or node.value.values:
            raise ValueError('module annotation dictionary RHS currently requires an empty display')
        builds=[row for row in rows if row.opname=='BUILD_MAP' and row.arg==0
                and tuple(row.positions)==_span(node.value)]
        if len(builds)!=1 or store is None or at[store.offset]!=at[builds[0].offset]+1:
            raise ValueError('empty dict RHS has no exact BUILD_MAP 0 to namespace-store sequence')
        empty_dict=builds[0]
    instructions=(([store] if store is not None else [])+stages)
    witness=ModuleAnnotationCompilerWitness(mode,evidence.root_code_layout_key,
        tuple(row.offset for row in instructions),empty_dict.offset if empty_dict else None)
    return ModuleAnnotationPlan(witness,store,lookup,value,update,key,discard,empty_dict)


def lower_module_annotation(builder,node,cursor,plan):
    """Connect original compiler operands and both completion worlds."""
    from scar.analysis.source_call_lowering import _constant
    from scar.ir.program.state_addresses import (
        AddressEffect,ReferenceOwner,NamespaceKey,Element,WholeOwner,
    )
    from scar.ir.program.model import AccessMode
    baseline=len(builder.active_temporaries)

    def hold(value,role):
        builder.frame_transition(cursor,node,((value,node,role),),P.FRAME_HOLD_TEMPORARY)
        builder.active_temporaries.append(value)

    def lookup():
        nonlocal cursor
        key=builder.literal(node,plan.lookup.argval)
        references=builder.entry_references
        cursor,owner,identity=builder.invoke(cursor,node,tuple((reference,node,role,index,None)
            for index,(reference,role) in enumerate(zip((*references,key),
                ('locals-owner','globals-owner','saved-builtins-owner','namespace-key')))))
        builder.protocol(identity,K.LOAD_NAME_314,node)
        operation=builder.operation_records[identity]
        builder._addresses(identity,tuple(AddressEffect(ReferenceOwner(operation.uses[index].id),
            (NamespaceKey(operation.uses[3].id),),AccessMode.READ) for index in range(3)))
        return owner

    def update(inputs,kind,consumed=()):
        nonlocal cursor
        def failed(failure,payload):
            saved=tuple(builder.active_temporaries)
            try:
                if consumed:
                    if tuple(builder.active_temporaries[-len(consumed):])!=consumed:
                        raise ValueError('annotation update lost its consumed stack references')
                    del builder.active_temporaries[-len(consumed):]
                builder.finish_exit(failure,node,payload,ExitKind.RAISE)
            finally:builder.active_temporaries[:]=saved
        cursor,_,identity=builder.invoke(cursor,node,inputs,result_type=None,failure_handler=failed)
        builder.protocol(identity,kind,node)
        original=builder.operation_records[identity]
        marked=replace(original,behaviors=original.behaviors+(Behavior.MEMORY,))
        builder.operations[builder.operations.index(original)]=marked;builder.operation_records[identity]=marked
        if consumed:del builder.active_temporaries[-len(consumed):]
        return marked

    try:
        if node.value is not None:
            if plan.empty_dict is not None:
                cursor,value,identity=builder.invoke(cursor,node.value,())
                builder.protocol(identity,P.BUILD_MAP,node.value)
                original=builder.operation_records[identity]
                marked=replace(original,behaviors=original.behaviors+(Behavior.MEMORY,))
                builder.operations[builder.operations.index(original)]=marked;builder.operation_records[identity]=marked
            else:cursor,value=builder.expr(node.value,cursor)
            cursor=builder.store(node.target,value,cursor)
        if plan.witness.mode is Mode.LAZY_SET:
            owner=lookup();hold(owner,'annotation-set-owner')
            index=(_constant(builder,node,cursor,plan.value) if plan.value.opname=='LOAD_CONST'
                   else builder.literal(node,plan.value.argval))
            operation=update(((owner,node,'annotation-set-owner',0,None),
                (index,node,'annotation-index',0,None)),K.SET_ADD_314)
            builder._addresses(operation.id,(AddressEffect(ReferenceOwner(operation.uses[0].id),
                (WholeOwner(),),AccessMode.READ_WRITE),))
            builder.frame_transition(cursor,node,((owner,node,'annotation-set-owner'),),P.FRAME_RELEASE_TEMPORARY)
            builder.active_temporaries.pop()
        else:
            annotation=_constant(builder,node.annotation,cursor,plan.value)
            hold(annotation,'annotation-string')
            owner=lookup();hold(owner,'annotation-mapping-owner')
            key=_constant(builder,node,cursor,plan.key)
            operation=update(((owner,node,'annotation-mapping-owner',0,None),
                (key,node,'annotation-key',0,None),(annotation,node.annotation,'annotation-string',0,None)),
                P.STORE_SUBSCRIPT,(annotation,owner))
            builder._addresses(operation.id,(AddressEffect(ReferenceOwner(operation.uses[0].id),
                (Element(operation.uses[1].id),),AccessMode.WRITE),))
        builder.gap(GapKind.EFFECT_SUMMARY,node,
            'annotation registration uses actual namespace lookup results; lookup/update/allocation/cleanup '
            'effects and later annotation reads remain open; generated annotation child remains deferred')
        return cursor
    finally:del builder.active_temporaries[baseline:]


# Independent port/order/lifetime checks for source-compiled annotations.
def _need(condition,detail):
    if not condition:raise ValueError('module annotation: '+detail)


def empty_dict_result(node,plan,*,source_reference,overlay,protocols,operations,invoke,ref_type,state_type):
    source=source_reference(node)
    matches=[operations[p.operation] for p in overlay.protocols if p.kind is P.BUILD_MAP and p.source==source]
    _need(plan.empty_dict is not None and len(matches)==1,'empty RHS must have its exact BUILD_MAP 0 allocation')
    operation=matches[0]
    _need(len(operation.uses)==1 and operation.callee is None,'empty dict cannot use children or a fixed target')
    normal,_=invoke(operation,(ref_type,state_type),memory=True)
    return normal.arguments[0]


def validate_module_annotation(node,expansion,plan,*,overlay,blocks,operations,values,protocols,
        expression_value,invoke,source_reference,normal_path,exception_holds,namespace_refs,
        ref_type,state_type,refinements):
    """Replay only the original annotation stages over canonical graph ports."""
    _need(expansion.compiler_annotation==plan.witness,'source/compiler statement receipt differs')
    path=normal_path(expansion,blocks,operations)
    _need(bool(path),'annotation has no normal corridor')
    use_links={(row.operation,row.use):row for row in overlay.use_links}
    local_refs=[];failure_checks=[]
    def kind(op):
        marker=protocols.get(op.id)
        return marker.kind if marker else None
    def literal(value,expected):
        record=values[value]
        _need(type(record.origin) is LiteralOrigin and type(record.origin.value) is type(expected)
              and record.origin.value==expected,'compiler literal/index changed')
    def consumed_on_failure(block,expected):
        failure_checks.append((block,tuple(expected)))
    def transition(op,before):
        _need(op.terminator is None and len(op.uses)==2 and len(op.accesses)==1
              and set(op.behaviors)=={Behavior.READ,Behavior.WRITE,Behavior.MEMORY}
              and not op.effects_complete and op.callee is None
              and op.uses[-1].value==before and op.accesses[0].before==op.uses[-1].id
              and op.accesses[0].mode.value=='READ_WRITE' and not op.accesses[0].full_overwrite
              and op.accesses[0].after is not None and op.results==(op.accesses[0].after,)
              and type(values[op.results[0]]) is StateVersionValue,
              'temporary transition changed its incoming/post world')
        return op.results[0]
    # Recover the current state at this exact statement entry, including
    # preceding frame transitions in the same canonical block.
    state=blocks[expansion.entry].arguments[-1]
    for identity in blocks[expansion.entry].operations:
        if identity==path[0].id:break
        for result in operations[identity].results:
            if type(values[result]) is StateVersionValue:state=result
    for operation in path:
        worlds=[use.value for use in operation.uses if type(values[use.value]) is StateVersionValue]
        _need(worlds and all(world==state for world in worlds),'normal corridor bypassed its preceding world')
        if operation.terminator is not None:
            edge=next(edge for edge in operation.terminator.successors if edge.kind is ExitKind.NORMAL)
            state=blocks[edge.block].arguments[-1]
        else:
            states=[result for result in operation.results if type(values[result]) is StateVersionValue]
            if states:
                _need(len(states)==1,'annotation normal action has multiple unmodeled post-worlds')
                state=states[0]
    start=0
    if node.value is not None:
        stores=[op for op in path if kind(op) in (K.STORE_NAME_314,P.STORE_GLOBAL)
                and protocols[op.id].source==source_reference(node.target)]
        _need(len(stores)==1,'RHS must feed exactly its original namespace store')
        store=stores[0]
        _need(store.uses[2].value==expression_value(node.value),'RHS did not feed its namespace store')
        start=path.index(store)+1
    macro=path[start:];position=0;held=[]
    def take(expected):
        nonlocal position
        _need(position<len(macro),'annotation stages ended early')
        operation=macro[position];position+=1
        _need(kind(operation) is expected,'annotation stages are reordered or contain an extra action')
        return operation
    def constant(instruction,source):
        operation=take(P.READ_CODE_CONSTANT)
        constants=[row for row in overlay.constants if row.operation==operation.id]
        _need(len(constants)==1 and constants[0].source==source
              and constants[0].constant_path==(instruction.arg,)
              and constants[0].code_key==plan.witness.root_code_layout_key,
              'annotation reference lost its exact original code constant slot')
        _need(len(operation.results)==1,'compiler constant has no unique reference result')
        _need(operation.terminator is None and operation.callee is None and not operation.effects_complete
              and set(operation.behaviors)=={Behavior.READ} and len(operation.uses)==1
              and len(operation.accesses)==1 and operation.accesses[0].before==operation.uses[0].id
              and operation.accesses[0].mode.value=='READ' and operation.accesses[0].after is None
              and type(values[operation.results[0]]) is ReferenceValue,
              'annotation constant must read the original code-slot reference without fabricating an invoke')
        return operation.results[0]
    def hold(reference,role):
        operation=take(P.FRAME_HOLD_TEMPORARY)
        row=use_links.get((operation.id,operation.uses[0].id)) if operation.uses else None
        _need(operation.uses[0].value==reference and protocols[operation.id].source==expansion.source
              and row is not None and row.role==role,'annotation temporary owner/role changed')
        transition(operation,operation.uses[-1].value);held.append(reference)
    def lookup(expected_holds):
        operation=take(K.LOAD_NAME_314)
        _need(protocols[operation.id].source==expansion.source and len(operation.uses)==5
              and tuple(use.value for use in operation.uses[:3])==tuple(namespace_refs),
              'implicit annotation lookup lost actual L/G/savedB ports')
        literal(operation.uses[3].value,plan.lookup.argval)
        normal,failure=invoke(operation,(ref_type,state_type))
        consumed_on_failure(failure.id,expected_holds)
        reference=normal.arguments[0];local_refs.append(reference)
        return reference
    def update(operation,expected_inputs,expected_holds):
        _need(protocols[operation.id].source==expansion.source
              and tuple(use.value for use in operation.uses[:-1])==tuple(expected_inputs)
              and operation.callee is None,'annotation update changed its actual operand references')
        _,failure=invoke(operation,(state_type,),memory=True)
        consumed_on_failure(failure.id,expected_holds)
        refinement=refinements.get(operation.id)
        _need(refinement is not None and type(refinement.site) is InvokeStateOperand
              and refinement.site.state_use==operation.uses[-1].id
              and refinement.coverage is AddressCoverage.PARTIAL and len(refinement.effects)==1,
              'annotation update needs its partial attempted address on the actual invoke state')
        effect=refinement.effects[0]
        _need(type(effect.owner) is ReferenceOwner and effect.owner.use==operation.uses[0].id
              and len(effect.path)==1,'annotation update changed its actual attempted owner')
        return effect
    if plan.witness.mode is Mode.LAZY_SET:
        owner=lookup(())
        hold(owner,'annotation-set-owner')
        if plan.value.opname=='LOAD_CONST':index=constant(plan.value,expansion.source)
        else:
            operation=macro[position]
            _need(kind(operation) is K.SET_ADD_314 and len(operation.uses)==3,
                  'annotation integer must feed its original SET_ADD')
            index=operation.uses[1].value;literal(index,plan.value.argval)
        operation=take(K.SET_ADD_314)
        effect=update(operation,(owner,index),(owner,))
        _need(type(effect.path[0]) is WholeOwner and effect.mode.value=='READ_WRITE',
              'set insertion is a whole-set partial update, not indexed storage')
        release=take(P.FRAME_RELEASE_TEMPORARY)
        row=use_links.get((release.id,release.uses[0].id)) if release.uses else None
        _need(release.uses[0].value==owner and row is not None and row.role=='annotation-set-owner'
              and protocols[release.id].source==expansion.source,'annotation POP_TOP lost its retained set owner')
        transition(release,release.uses[-1].value);held.pop()
    else:
        annotation=constant(plan.value,source_reference(node.annotation));local_refs.append(annotation)
        hold(annotation,'annotation-string')
        owner=lookup((annotation,));hold(owner,'annotation-mapping-owner')
        key=constant(plan.key,expansion.source)
        operation=take(P.STORE_SUBSCRIPT)
        effect=update(operation,(owner,key,annotation),())
        _need(type(effect.path[0]) is Element and effect.path[0].index==operation.uses[1].id
              and effect.mode.value=='WRITE','annotation STORE_SUBSCR key/write address changed')
        held.clear()  # The original STORE_SUBSCR consumes all three inputs on both outcomes.
    _need(position==len(macro) and not held,'annotation stages leaked a temporary or added a runtime action')
    for block,expected in failure_checks:
        actual=exception_holds(block,operations,protocols,blocks,use_links)
        _need(tuple(reference for reference in actual if reference in local_refs)==expected,
              'annotation exceptional handoff changed pending/consumed reference cleanup')

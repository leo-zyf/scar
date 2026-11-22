"""Source comprehension."""

from __future__ import annotations

import ast
import dis
from dataclasses import replace
from scar.analysis.program_lowering import _Unsupported, _REF, _BOOL
from scar.analysis.static.lowering.source_heap_lowering import (
    _source_definition,
    _associate,
    _hold,
    _normal_producer,
)
from scar.analysis.structured_lowering import _transfer
from scar.analysis.static.lowering.source_slice import _invoke
from scar.ir.program.bindings import BindingRead
from scar.ir.program.ids import OperationID, ValueID
from scar.ir.program.model import (
    AccessMode,
    Behavior,
    DataValue,
    ExitKind,
    Operation,
    OperationResult,
    ReferenceValue,
    StateAccess,
    TerminatorKind,
    BlockArgument,
    CompletionResult,
    LocalBindingValue,
    StateVersionValue,
    value_type,
)
from scar.ir.program.source import GapKind, SourceRole
from scar.ir.program.structured import PythonProtocolKind as K
from scar.ir.semantics_v2 import Opcode
from scar.ir.v2.semantic import OperationKind, SemanticRelation
from scar.analysis.source_expression_rules import _require
from scar.ir.program import source_productions as P
from scar.ir.program.signature_patterns import CountBinding, ChoiceBinding
from scar.analysis.source_language import function_exception_types, require_source_tree
from scar.ir.program.source_language import SourceLanguageCapability as C


# Eager source comprehensions in the existing function CFG and world.
#
# Private SSA keys name original comprehension slots; they are not Python local
# names, new source slots, child frames, or extra target-program operations.
# Outer bindings remain held and the scoped target is released on scope exit.
VERSION = 'PYTHON_COMPREHENSION_314_V1'
KINDS = frozenset((K.COMPREHENSION_BUILD, K.COMPREHENSION_NEXT,
                   K.COMPREHENSION_ADD, K.COMPREHENSION_LEAVE))


def target_leaves(target):
    """Original left-to-right occurrences; repeated names remain repeated."""
    if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
        return (target,)
    if isinstance(target, (ast.Tuple, ast.List)) and isinstance(target.ctx, ast.Store):
        return tuple(leaf for child in target.elts for leaf in target_leaves(child))
    raise ValueError('comprehension target requires recursive Tuple/List -> Store Name')


def target_shape(target):
    try:
        leaves=target_leaves(target)
    except (ValueError, RecursionError):
        return False
    return bool(leaves)


def shape(node, *, recursive_targets=False):
    return (isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp))
        and len(node.generators) == 1 and not node.generators[0].is_async
        and (target_shape(node.generators[0].target) if recursive_targets else isinstance(node.generators[0].target, ast.Name))
        and not node.generators[0].ifs
        and not any(isinstance(n, (ast.NamedExpr, ast.Yield, ast.YieldFrom, ast.Await)) for n in ast.walk(node))
        and not any(isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
                    for child in ast.iter_child_nodes(node) for n in ast.walk(child)))


def compiler_scope(node, code, *, recursive_targets=False):
    """Witness the inlined scope's original build/update/compiler target, not execution."""
    if recursive_targets:
        return _compiler_targets(node, code)
    span=lambda n:(n.lineno,n.end_lineno,n.col_offset,n.end_col_offset)
    positions=span(node);all_rows=tuple(dis.get_instructions(code))
    def within(row):
        p=row.positions
        return (p.lineno is not None and (node.lineno,node.col_offset)<=(p.lineno,p.col_offset)
                and (p.end_lineno,p.end_col_offset)<=(node.end_lineno,node.end_col_offset))
    rows=[r for r in all_rows if within(r)]
    build,update={ast.ListComp:('BUILD_LIST','LIST_APPEND'),ast.SetComp:('BUILD_SET','SET_ADD'),ast.DictComp:('BUILD_MAP','MAP_ADD')}[type(node)]
    target=node.generators[0].target.id
    builds=[r for r in rows if r.opname==build and r.arg==0 and tuple(r.positions)==positions]
    updates=[r for r in rows if r.opname==update]
    clears=[r for r in all_rows if r.opname=='LOAD_FAST_AND_CLEAR' and r.argval==target and tuple(r.positions)==positions]
    if (len(builds)!=1 or len(updates)!=1 or len(clears)!=1 or target in code.co_cellvars
            or not clears[0].offset<builds[0].offset<updates[0].offset):
        raise _Unsupported(GapKind.CODE_VERSION,node,'COMPREHENSION_INLINED_BUILD_UPDATE_TARGET_INTERFACE_UNRECOVERED')


def _fast_store_names(code, instruction):
    """Decode only the fixed 3.14.7 local-store encodings used by targets.

    Like the existing dual-borrow load decoder, check both nibble indexes
    against dis's decoded names and the original code's local table. The
    fused instruction keeps only its first source position; it does not
    establish a second bytecode source occurrence.
    """
    if instruction.opname=='STORE_FAST':
        indexes=(instruction.arg,);names=(instruction.argval,)
    elif instruction.opname=='STORE_FAST_STORE_FAST':
        packed=instruction.arg;names=instruction.argval
        if (type(packed)is not int or not 0<=packed<=0xff
                or type(names)is not tuple or len(names)!=2):
            return None
        indexes=(packed>>4,packed&0xf)
    else:return None
    if any(type(index)is not int or not 0<=index<code.co_nlocals
           or index>=len(code.co_varnames) or type(name)is not str
           or code.co_varnames[index]!=name for index,name in zip(indexes,names)):
        return None
    return names


def _compiler_targets(node, code):
    """Exact clear/store/restore corridors; no guessed physical name ordering."""
    span=lambda n:(n.lineno,n.end_lineno,n.col_offset,n.end_col_offset)
    position=span(node);rows=tuple(dis.get_instructions(code))
    def inside(row):
        p=row.positions
        return (p.lineno is not None and p.col_offset is not None and p.end_lineno is not None
                and p.end_col_offset is not None and (node.lineno,node.col_offset)<=(p.lineno,p.col_offset)
                and (p.end_lineno,p.end_col_offset)<=(node.end_lineno,node.end_col_offset))
    build_name,update_name={ast.ListComp:('BUILD_LIST','LIST_APPEND'),ast.SetComp:('BUILD_SET','SET_ADD'),ast.DictComp:('BUILD_MAP','MAP_ADD')}[type(node)]
    builds=[r for r in rows if r.opname==build_name and r.arg==0 and tuple(r.positions)==position]
    updates=[r for r in rows if r.opname==update_name and inside(r)]
    leaves=target_leaves(node.generators[0].target);names=set(n.id for n in leaves)
    clears=[r for r in rows if r.opname=='LOAD_FAST_AND_CLEAR' and tuple(r.positions)==position]
    def reject(detail):
        raise _Unsupported(GapKind.CODE_VERSION,node,'COMPREHENSION_TARGET_COMPILER_'+detail)
    if (len(builds)!=1 or len(updates)!=1 or len(clears)!=len(names)
            or {r.argval for r in clears}!=names or any(name in code.co_cellvars for name in names)
            or any(r.offset>=builds[0].offset for r in clears) or builds[0].offset>=updates[0].offset):
        reject('CLEAR_BUILD_UPDATE_UNRECOVERED')
    expected=[]
    def target(n):
        if isinstance(n,ast.Name):expected.append(('STORE_FAST',n.id,span(n)))
        else:
            expected.append(('UNPACK_SEQUENCE',len(n.elts),span(n)))
            for child in n.elts:target(child)
    target(node.generators[0].target)
    positions={item[2] for item in expected}
    actual=[r for r in rows if tuple(r.positions) in positions
            and r.opname in ('STORE_FAST','STORE_FAST_STORE_FAST','UNPACK_SEQUENCE','UNPACK_EX')
            and builds[0].offset<r.offset<updates[0].offset]
    index=0
    for row in actual:
        if row.opname in ('STORE_FAST','STORE_FAST_STORE_FAST'):
            names_in_order=_fast_store_names(code,row)
            if names_in_order is None:reject('FAST_STORE_OPERANDS_DIFFER')
            original=expected[index:index+len(names_in_order)]
            # A fused pair consumes exactly the next two original Name
            # leaves, not a set of names. Its retained position must be the
            # first leaf's; the second leaf retains its AST SourceReference,
            # without inventing a missing second instruction/position.
            if (len(original)!=len(names_in_order)
                    or any(item[0]!='STORE_FAST'for item in original)
                    or tuple(item[1]for item in original)!=names_in_order
                    or tuple(row.positions)!=original[0][2]):
                reject('ORDERED_UNPACK_STORES_UNRECOVERED')
            index+=len(names_in_order)
        else:
            if index>=len(expected) or (row.opname,row.arg,tuple(row.positions))!=expected[index]:
                reject('ORDERED_UNPACK_STORES_UNRECOVERED')
            index+=1
    if index!=len(expected):reject('ORDERED_UNPACK_STORES_UNRECOVERED')
    entries=[e for e in dis.Bytecode(code).exception_entries
             if e.start<=builds[0].offset<e.end and updates[0].offset<e.end]
    if len(entries)!=1:reject('EXCEPTION_CORRIDOR_UNRECOVERED')
    handler=entries[0];tail=[r for r in rows if r.offset>=handler.target]
    stop=next((i for i,r in enumerate(tail)if r.opname=='RERAISE'),None)
    if stop is None:reject('RESTORE_RERAISE_MISSING')
    tail=tail[:stop+1]
    if (any(r.opname not in ('SWAP','POP_TOP','STORE_FAST','NOP','RERAISE')for r in tail)
            or tail[-1].arg!=0):reject('RESTORE_RERAISE_PROTOCOL_DIFFERS')
    error=[r.argval for r in tail if r.opname=='STORE_FAST' and tuple(r.positions)==position]
    normal=[r.argval for r in rows if r.opname=='STORE_FAST' and tuple(r.positions)==position
            and updates[0].offset<r.offset<handler.target]
    if (len(normal)!=len(names) or set(normal)!=names or len(error)!=len(names) or set(error)!=names):
        reject('NORMAL_EXCEPTION_RESTORE_NAMES_DIFFER')
    return {'clear_order':tuple(r.argval for r in clears),'normal_order':tuple(normal),
            'failure_order':tuple(error),'handler':handler.target}


def _target_read(builder, node, cursor, old=None):
    scope = getattr(builder, 'comprehension_scope', None)
    if isinstance(scope,_ScopedTargets):
        alias=scope.aliases.get(node.id)
        if alias is None:return NotImplemented
    elif scope is not None and node.id==scope[0]:
        alias=scope[1]
    else:
        return NotImplemented
    binding = cursor.env[alias]
    identity = builder.fresh(OperationID, 'comprehension-read')
    use = builder.use(identity, binding, node, 'local-binding', 0)
    value = ReferenceValue(builder.fresh(ValueID, 'comprehension-reference'), builder.function_context,
                           OperationResult(identity, 0))
    builder.add_value(value); builder.link_value(value, node, SourceRole.EXPRESSION)
    builder.emit(cursor, Operation(identity, 'read-local', (Behavior.COMPUTE,), (use,), (value.id,),
                                  effects_complete=True), node, SourceRole.EXPRESSION, old=old)
    builder.binding_reads.append(BindingRead(identity, use.id, value.id))
    return cursor, value.id


def expression(builder, node, cursor):
    """Original SG children absent from the old opaque comprehension overlay."""
    definition = _source_definition(builder, node)
    if (builder.has_language('SOURCE_REFERENCE_LIFETIME_314_V1') and isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Load) and not isinstance(node.slice, (ast.Slice, ast.Tuple))):
        from scar.analysis.static.lowering.source_slice import lower_subscript
        return lower_subscript(builder, node, cursor, source_expression=True)
    if isinstance(node, ast.Name):
        first = len(builder.operations)
        cursor, value = builder.read_source_name(node, cursor, None)
        if cursor is not None:
            _associate(builder, _normal_producer(builder, node, value, first), node, definition, Opcode.READ)
        return cursor, value
    if isinstance(node, ast.Constant):
        if builder.has_language('SOURCE_EXPRESSIONS_314_V1'):
            from scar.analysis.static.lowering.source_expression_lowering import lower_source_literal
            return lower_source_literal(builder, node, cursor, definition=definition)
        return builder.expr(node, cursor)
    if (builder.has_language('SOURCE_EXPRESSIONS_314_V1')
            and isinstance(node, (ast.BinOp, ast.Compare))):
        from scar.analysis.static.lowering.source_expression_lowering import lower_source_expression
        return lower_source_expression(builder, node, cursor, definition=definition,
            evaluate=lambda child, at: expression(builder, child, at))
    if isinstance(node, ast.JoinedStr):
        from scar.analysis.static.lowering.source_string import lower_joined_string
        return lower_joined_string(builder, node, cursor, recover_missing_children=True)
    if isinstance(node, ast.Attribute):
        children = ((node.value, 'receiver', 0),); opcode = Opcode.ATTRIBUTE
    elif isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Slice):
        children = ((node.value, 'receiver', 0), (node.slice, 'index', 1)); opcode = Opcode.INDEX
    elif isinstance(node, ast.Call) and not any(isinstance(a, ast.Starred) for a in node.args) and all(k.arg is not None for k in node.keywords):
        children = ((node.func, 'callee', 0),) + tuple((c, 'argument', i) for i,c in enumerate(node.args)) + tuple((k.value, 'keyword:'+k.arg, i) for i,k in enumerate(node.keywords))
        opcode = Opcode.CALL
    else:
        raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX, node,
                           'COMPREHENSION_EXPRESSION_INTERFACE_UNIMPLEMENTED: '+type(node).__name__)
    base = len(builder.active_temporaries); held=[]; inputs=[]
    try:
        for child, role, position in children:
            cursor, value = expression(builder, child, cursor)
            if cursor is None: return None, None
            _hold(builder, cursor, child, value); held.append((child,value))
            inputs.append((value,child,role,position,None))
        cursor, value, identity = _invoke(builder,cursor,node,inputs,tuple(held))
        _associate(builder,identity,node,definition,opcode)
        if opcode is Opcode.ATTRIBUTE:
            from scar.analysis.source_attribute_lookup import register_attribute_lookup
            register_attribute_lookup(builder,node,identity)
        if opcode is Opcode.INDEX:
            from scar.ir.program.state_addresses import AddressEffect, Element, ReferenceOwner
            uses=builder.operation_records[identity].uses
            builder._addresses(identity,(AddressEffect(ReferenceOwner(uses[0].id),(Element(uses[1].id),),AccessMode.READ),))
        builder.link_value(builder.values_by_id[value],node,SourceRole.EXPRESSION)
        if opcode is Opcode.CALL: builder.pending_targets.append((identity,inputs[0][0],None,node))
        return cursor,value
    finally: del builder.active_temporaries[base:]


def _update(builder,node,cursor,container,children):
    base=len(builder.active_temporaries);held=[];inputs=[(container,node,'container',0,None)]
    try:
        for i,child in enumerate(children):
            cursor,value=expression(builder,child,cursor)
            if cursor is None:return None
            _hold(builder,cursor,child,value);held.append((child,value));inputs.append((value,child,'element',i,None))
        def failed(at,exception):
            pending=tuple(builder.active_temporaries)
            try:
                _transfer(builder,tuple(held),node)
                builder.finish_exit(at,node,exception,ExitKind.RAISE)
            finally:builder.active_temporaries[:]=pending
        cursor,_,identity=builder.invoke_results(cursor,node,tuple(inputs),result_types=(),failure_handler=failed)
        builder.protocol(identity,K.COMPREHENSION_ADD,node)
        _transfer(builder,tuple(held),node)
        return cursor
    finally:del builder.active_temporaries[base:]


class _ScopedTargets:
    """Private construction context; all binding endpoints are original slots."""
    def __init__(self,builder,node,definition,plan,baseline,iterator):
        self.builder,self.node,self.definition,self.plan=builder,node,definition,plan
        self.baseline,self.iterator,self.container=baseline,iterator,None
        self.leaves=target_leaves(node.generators[0].target)
        self.first={}
        for leaf in self.leaves:self.first.setdefault(leaf.id,leaf)
        self.aliases={name:'\x00comprehension:'+definition.id.wire+':'+name for name in plan['clear_order']}
        self.slots={}
        for name in self.aliases:
            slots=[s.slot_id for s in builder.semantic.slots.values()if s.owner==definition.id
                   and s.name==name and s.metadata.get('scope_kind')=='comprehension']
            if len(slots)!=1:
                raise _Unsupported(GapKind.SOURCE_MAPPING,self.first[name],'COMPREHENSION_ORIGINAL_TARGET_SLOT_REQUIRED')
            self.slots[name]=slots[0]
        self.sink=self.failed

    def bind(self,target,value,cursor):
        if not isinstance(target,ast.Name) or target.id not in self.aliases:
            raise _Unsupported(GapKind.SOURCE_MAPPING,target,'COMPREHENSION_TARGET_OCCURRENCE_UNRECOVERED')
        alias=self.aliases[target.id]
        self.builder.frame_transition(cursor,target,
            ((cursor.env[alias],target,'previous-binding'),(value,target,'assigned-reference')),K.FRAME_BIND)
        self.builder.define_binding(alias,cursor,target,value)
        return cursor

    def leave(self,cursor,*,failure):
        """One versioned scope protocol; no claim of actual deallocation.

        The compiler exception corridor drops iterator then result-container
        before restoring scoped names. Normal exhaustion forwards the result.
        Existing outer-name binding tokens are carried unchanged, not rebound.
        """
        order=self.plan['failure_order'if failure else'normal_order']
        refs=[(self.iterator,self.node,'comprehension-exception-iterator'if failure else'exhausted-iterator')]
        if failure and self.container is not None:
            refs.append((self.container,self.node,'comprehension-exception-container'))
        refs.extend((cursor.env[self.aliases[name]],self.first[name],
                     'comprehension-exception-binding'if failure else'scoped-binding')for name in order)
        self.builder.frame_transition(cursor,self.node,tuple(refs),K.COMPREHENSION_LEAVE)
        for alias in self.aliases.values():cursor.env.pop(alias)

    def failed(self,cursor,payload):
        builder=self.builder;saved=tuple(builder.active_temporaries)
        if not builder.failure_sinks or builder.failure_sinks[-1] is not self.sink:
            raise _Unsupported(GapKind.SOURCE_MAPPING,self.node,'COMPREHENSION_FAILURE_SINK_OWNERSHIP_DIFFERS')
        reserved=(self.iterator,)if self.container is None else(self.iterator,self.container)
        owned=tuple(builder.active_temporaries[self.baseline:])
        if owned[:len(reserved)]!=reserved:
            raise _Unsupported(GapKind.SOURCE_MAPPING,self.node,'COMPREHENSION_PENDING_PREFIX_DIFFERS')
        builder.failure_sinks.pop()
        try:
            # Helpers already removed interpreter-consumed arguments. Remaining
            # unpack siblings/child operands unwind in reverse retained order.
            for value in reversed(owned[len(reserved):]):
                builder.frame_transition(cursor,self.node,((value,self.node,'function-exception-unwind'),),K.FRAME_RELEASE_TEMPORARY)
            self.leave(cursor,failure=True)
            del builder.active_temporaries[self.baseline:]
            # Preserve this INVOKE's exception/traceback pair and latest W;
            # the original outer C0/C1 sink remains the sole pending authority.
            builder.finish_exit(cursor,self.node,payload,ExitKind.RAISE)
        finally:
            builder.active_temporaries[:]=saved
            builder.failure_sinks.append(self.sink)


def scoped_target(builder,target):
    """Resolve only a current original comprehension Store-Name occurrence."""
    scope=getattr(builder,'comprehension_scope',None)
    if not isinstance(scope,_ScopedTargets) or not isinstance(target,ast.Name):return None
    reference=builder.source(target)
    if not any(builder.source(leaf)==reference for leaf in scope.leaves):return None
    return scope


def scoped_target_slots(semantic,source_tree,language):
    """Exact leaf SourceReference -> original comp slot for shared validators."""
    from scar.ir.program.source_language import SourceLanguageCapability as C
    if C.COMPREHENSION_TARGETS_314_V1 not in language.capabilities:return {}
    atoms={(a.kind,a.reference.start_line,a.reference.end_line,a.reference.start_column,a.reference.end_column):a
           for a in semantic.source_atoms.values()if a.reference.path==language.source.path
           and a.reference.fingerprint==language.source.fingerprint}
    result={}
    for node in ast.walk(source_tree):
        if not shape(node,recursive_targets=True):continue
        key=(type(node).__name__,node.lineno,node.end_lineno,node.col_offset,node.end_col_offset)
        atom=atoms.get(key)
        if atom is None:continue
        owners=[d.id for d in semantic.definitions.values()if d.kind is OperationKind.OPAQUE
                and d.code_id==atom.id.wire and d.metadata.get('ast_type')==type(node).__name__]
        if len(owners)!=1:continue
        for leaf in target_leaves(node.generators[0].target):
            leaf_atom=atoms.get(('Name',leaf.lineno,leaf.end_lineno,leaf.col_offset,leaf.end_col_offset))
            slots=[s.slot_id for s in semantic.slots.values()if s.owner==owners[0]
                   and s.name==leaf.id and s.metadata.get('scope_kind')=='comprehension']
            if leaf_atom is not None and len(slots)==1:result[leaf_atom.reference]=slots[0]
    return result


def _lower_targets(builder,node,cursor):
    if not shape(node,recursive_targets=True):
        raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX,node,'COMPREHENSION_RECURSIVE_TARGET_SHAPE_UNIMPLEMENTED')
    if (not builder.has_language('FUNCTION_TRY_314_V1') or not builder.has_language('FIXED_UNPACK')
            or not hasattr(builder,'failure_sinks')):
        raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX,node,'COMPREHENSION_TARGETS_REQUIRE_EXISTING_PAIRED_PROPAGATION_AND_UNPACK')
    evidence=builder.language_options.code_evidence
    from scar.analysis.source_language import supports_cpython_compiler
    if not supports_cpython_compiler(evidence):
        raise _Unsupported(GapKind.CODE_VERSION,node,'COMPREHENSION_TARGETS_REQUIRE_CPYTHON_3147_COMPILER')
    plan=compiler_scope(node,evidence.code,recursive_targets=True)
    definition=_source_definition(builder,node)
    if definition.kind is not OperationKind.OPAQUE or definition.metadata.get('ast_type')!=type(node).__name__:
        raise _Unsupported(GapKind.SOURCE_MAPPING,node,'COMPREHENSION_ORIGINAL_SOURCE_SCOPE_REQUIRED')
    baseline=len(builder.active_temporaries);clause=node.generators[0]
    prior=getattr(builder,'comprehension_scope',None);original_slots=builder.slots
    scope=None
    try:
        # Outermost iterable and GET_ITER happen before target save/clear.
        cursor,iterable=expression(builder,clause.iter,cursor)
        if cursor is None:return None,None
        _hold(builder,cursor,clause.iter,iterable)
        def get_failed(at,payload):
            saved=tuple(builder.active_temporaries)
            try:
                _transfer(builder,((clause.iter,iterable),),node)
                builder.finish_exit(at,node,payload,ExitKind.RAISE)
            finally:builder.active_temporaries[:]=saved
        cursor,iterator,_=builder.invoke_protocol(cursor,clause.iter,(iterable,),K.GET_ITERATOR,failure_handler=get_failed)
        _transfer(builder,((clause.iter,iterable),),node);_hold(builder,cursor,node,iterator)
        scope=_ScopedTargets(builder,node,definition,plan,baseline,iterator)
        builder.slots={**builder.slots,**{(builder.function_definition.id,scope.aliases[name]):slot
                                       for name,slot in scope.slots.items()}}
        for name in plan['clear_order']:
            builder.define_binding(scope.aliases[name],cursor,scope.first[name])
        builder.comprehension_scope=scope
        builder.failure_sinks.append(scope.sink)
        cursor,container,identity=builder.invoke(cursor,node,())
        scope.container=container
        builder.protocol(identity,K.COMPREHENSION_BUILD,node)
        _associate(builder,identity,node,definition,Opcode.OPAQUE)
        _hold(builder,cursor,node,container)
        names=tuple(sorted(cursor.env));header=builder.new_block(node,named=names)
        builder.jump(cursor,header,names,node)
        natural=builder.new_block(node)
        def exhausted(at,payload):
            identity=builder.fresh(OperationID,'comprehension-stop-match')
            use=builder.use(identity,builder.exception_reference(payload),node,'exception',0)
            before=builder.use(identity,at.state,node,'exception-type-state',0)
            boolean=DataValue(builder.fresh(ValueID,'comprehension-stop'),builder.function_context,OperationResult(identity,0),'builtins.bool')
            builder.add_value(boolean);builder.link_value(boolean,node,SourceRole.TRUTH)
            builder.emit(at,Operation(identity,'match-stop-iteration',(Behavior.COMPUTE,Behavior.READ),(use,before),(boolean.id,),
                (StateAccess(builder.world,before.id,AccessMode.READ),),effects_complete=True),node,SourceRole.TRUTH)
            builder.protocol(identity,K.MATCH_STOP_ITERATION,node)
            raised=builder.new_block(node);builder.branch(at,boolean.id,natural,raised,node)
            builder.frame_transition(natural,node,tuple((v,node,'consumed-exhaustion')for v in builder.failure_references(payload)),K.FRAME_DISCARD)
            builder.finish_exit(raised,node,payload,ExitKind.RAISE)
        body,item,_=builder.invoke_protocol(header,node,(iterator,),K.COMPREHENSION_NEXT,failure_handler=exhausted)
        if isinstance(clause.target,ast.Name):
            body=scope.bind(clause.target,item,body)
        else:
            from scar.analysis.static.lowering.source_unpack_lowering import lower_unpack_target
            body=lower_unpack_target(builder,clause.target,item,body)
        children=(node.key,node.value)if isinstance(node,ast.DictComp)else(node.elt,)
        body=_update(builder,node,body,container,children)
        if body is not None:builder.jump(body,header,names,node)
        scope.leave(natural,failure=False)
        from scar.analysis.static.lowering.source_expression_lowering import forward_source_reference
        forward_source_reference(builder,natural,node,container)
        _transfer(builder,((node,iterator),(node,container)),node)
        builder.link_value(builder.values_by_id[container],node,SourceRole.EXPRESSION)
        builder.gap(GapKind.EFFECT_SUMMARY,node,'recursive comp target/partial-unpack/paired cleanup interfaces are explicit; object aliasing, contents, finalizers, iteration count and full effects remain open')
        return natural,container
    finally:
        if scope is not None and builder.failure_sinks and builder.failure_sinks[-1] is scope.sink:
            builder.failure_sinks.pop()
        builder.slots=original_slots;builder.comprehension_scope=prior
        del builder.active_temporaries[baseline:]


def lower_comprehension(builder,node,cursor):
    if builder.has_language('COMPREHENSION_TARGETS_314_V1'):
        return _lower_targets(builder,node,cursor)
    if not shape(node):
        raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX,node,'COMPREHENSION_ASYNC_MULTI_CLAUSE_NESTED_OR_TARGET_INTERFACE_UNIMPLEMENTED')
    if getattr(builder,'try_depth',0) or builder.has_language('FUNCTION_TRY_314_V1'):
        raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX,node,'COMPREHENSION_PARENT_CLEANUP_INTERFACE_UNIMPLEMENTED')
    evidence=builder.language_options.code_evidence
    if evidence is None:raise _Unsupported(GapKind.CODE_VERSION,node,'COMPREHENSION_READONLY_CODE_WITNESS_REQUIRED')
    compiler_scope(node,evidence.code)
    definition=_source_definition(builder,node);clause=node.generators[0];target=clause.target
    if definition.kind is not OperationKind.OPAQUE or definition.metadata.get('ast_type')!=type(node).__name__:
        raise _Unsupported(GapKind.SOURCE_MAPPING,node,'COMPREHENSION_ORIGINAL_SOURCE_SCOPE_REQUIRED')
    slots=[s.slot_id for s in builder.semantic.slots.values() if s.owner==definition.id and s.name==target.id and s.metadata.get('scope_kind')=='comprehension']
    if len(slots)!=1:raise _Unsupported(GapKind.SOURCE_MAPPING,target,'COMPREHENSION_ORIGINAL_TARGET_SLOT_REQUIRED')
    baseline=len(builder.active_temporaries);prior=getattr(builder,'comprehension_scope',None)
    alias='\x00comprehension:'+definition.id.wire+':'+target.id
    original_slots=builder.slots;builder.slots={**builder.slots,(builder.function_definition.id,alias):slots[0]}
    try:
        cursor,iterable=expression(builder,clause.iter,cursor)
        if cursor is None:return None,None
        _hold(builder,cursor,clause.iter,iterable)
        def get_failed(at,exception):
            held=tuple(builder.active_temporaries)
            try:
                _transfer(builder,((clause.iter,iterable),),node)
                builder.finish_exit(at,node,exception,ExitKind.RAISE)
            finally:builder.active_temporaries[:]=held
        cursor,iterator,_=builder.invoke_protocol(cursor,clause.iter,(iterable,),K.GET_ITERATOR,failure_handler=get_failed)
        _transfer(builder,((clause.iter,iterable),),node);_hold(builder,cursor,node,iterator)
        cursor,container,identity=builder.invoke(cursor,node,())
        builder.protocol(identity,K.COMPREHENSION_BUILD,node)
        _associate(builder,identity,node,definition,Opcode.OPAQUE)
        _hold(builder,cursor,node,container)
        builder.define_binding(alias,cursor,target)
        names=tuple(sorted(cursor.env));header=builder.new_block(node,named=names)
        builder.jump(cursor,header,names,node)
        natural=builder.new_block(node)
        def exhausted(at,exception):
            identity=builder.fresh(OperationID,'comprehension-stop-match')
            use=builder.use(identity,builder.exception_reference(exception),node,'exception',0)
            before=builder.use(identity,at.state,node,'exception-type-state',0)
            boolean=DataValue(builder.fresh(ValueID,'comprehension-stop'),builder.function_context,OperationResult(identity,0),'builtins.bool')
            builder.add_value(boolean);builder.link_value(boolean,node,SourceRole.TRUTH)
            builder.emit(at,Operation(identity,'match-stop-iteration',(Behavior.COMPUTE,Behavior.READ),(use,before),(boolean.id,),
                (StateAccess(builder.world,before.id,AccessMode.READ),),effects_complete=True),node,SourceRole.TRUTH)
            builder.protocol(identity,K.MATCH_STOP_ITERATION,node)
            raised=builder.new_block(node);builder.branch(at,boolean.id,natural,raised,node)
            builder.frame_transition(natural,node,tuple((v,node,'consumed-exhaustion')for v in builder.failure_references(exception)),K.FRAME_DISCARD)
            builder.finish_exit(raised,node,exception,ExitKind.RAISE)
        body,item,identity=builder.invoke_protocol(header,node,(iterator,),K.COMPREHENSION_NEXT,failure_handler=exhausted)
        builder.frame_transition(body,target,((body.env[alias],target,'previous-binding'),(item,target,'assigned-reference')),K.FRAME_BIND)
        builder.define_binding(alias,body,target,item)
        builder.comprehension_scope=(target.id,alias)
        if body is not None:
            children=(node.key,node.value) if isinstance(node,ast.DictComp) else(node.elt,)
            body=_update(builder,node,body,container,children)
        if body is not None:builder.jump(body,header,names,node)
        # Clear this transient binding while the result/container and enclosing
        # bindings remain held. The physical save/restore staging is folded.
        builder.frame_transition(natural,node,((natural.env[alias],target,'scoped-binding'),(iterator,node,'exhausted-iterator')),K.COMPREHENSION_LEAVE)
        natural.env.pop(alias)
        from scar.analysis.static.lowering.source_expression_lowering import forward_source_reference
        forward_source_reference(builder,natural,node,container)
        _transfer(builder,((node,iterator),(node,container)),node)
        builder.link_value(builder.values_by_id[container],node,SourceRole.EXPRESSION)
        builder.gap(GapKind.EFFECT_SUMMARY,node,'comprehension scoped iteration/container dependencies are explicit; dispatch, contents, identity, iteration count and complete effects remain unproved')
        return natural,container
    finally:
        builder.slots=original_slots;builder.comprehension_scope=prior
        del builder.active_temporaries[baseline:]


# Scoped comprehensions using shared synchronous/asynchronous loop templates.
#
# Reference-valued invariant ports do not certify invariant referent contents.
# The changing namespace/container/iterator state remains in the carried world.
# Physical inlining, frame cleanup and escaped-cell lifetime are not proved here.
def comprehension_clauses(b):
    generators = b.value(b.site('generators'))
    _require(generators.kind == 'LIST' and bool(generators.items), 'RULE_SHAPE', str(b.occurrence),
             'comprehension requires at least one original clause')
    b.mark('generators', P.ProductionDispositionKind.EVALUATED_CHILD)
    clauses = []
    for item in generators.items:
        _require(item.kind == 'NODE' and b.rules.source.context.syntax.nodes[item.reference].constructor == 'comprehension',
                 'RULE_SHAPE', str(b.occurrence), 'expected an original comprehension descriptor')
        target = b.auxiliary(item.reference, 'target', P.ProductionDispositionKind.TARGET_ADDRESS)
        iterable = b.auxiliary(item.reference, 'iter', P.ProductionDispositionKind.EVALUATED_CHILD)
        filters = b.auxiliary(item.reference, 'ifs', P.ProductionDispositionKind.EVALUATED_CHILD)
        asynchronous = b.auxiliary(item.reference, 'is_async', P.ProductionDispositionKind.METADATA)
        raw = b.value(asynchronous)
        _require(raw.kind == 'SCALAR' and type(raw.scalar.to_python()) is int and raw.scalar.to_python() in (0, 1),
                 'RULE_SHAPE', str(asynchronous), 'async marker must be an explicit compiler flag')
        _require(b.value(filters).kind == 'LIST', 'RULE_SHAPE', str(filters), 'filters require an original expression list')
        clauses.append((target, iterable, filters, bool(raw.scalar.to_python())))
    return clauses


def get_iterator(b, region, value, world, asynchronous):
    if asynchronous:
        return b.delegation(region, P.DelegationProtocolKind.GET_ASYNC_ITERATOR, (value,), world)
    return b.protocol(region, K.GET_ITERATOR, (value,), world)


def emit_comprehension(b, root, state):
    clauses = comprehension_clauses(b)
    output_names = ('key', 'value') if b.node.constructor == 'DictComp' else ('elt',)
    for name in output_names:
        b.mark(name, P.ProductionDispositionKind.EVALUATED_CHILD)
    # Only the outermost iterable is evaluated in the enclosing environment.
    iterable, state = b.evaluate(root, clauses[0][1], state)
    iterator, state = get_iterator(b, root, iterable, state, clauses[0][3])
    scope = b.region(b.site(output_names[0]), (_REF, b.world), root, P.ProductionRegionKind.COMPREHENSION)
    result = b.emit(root, P.ApplyRegion(P.ProductionRegionCall(scope.id, (iterator, state))), (), (_REF, b.world))
    b.finish(root, result, (_REF, b.world))

    world = b.parameter(scope, 1)
    kind = {'ListComp':'LIST', 'SetComp':'SET', 'DictComp':'DICT'}[b.node.constructor]
    build = {'LIST':K.BUILD_DISPLAY_LIST, 'SET':K.BUILD_SET_DISPLAY, 'DICT':K.BUILD_MAP}[kind]
    count = 'pair_count' if kind == 'DICT' else 'item_count'
    container, world = b.protocol(scope, build, (), world, (CountBinding(count, 0),), (b.site(output_names[0]),))
    world = emit_clause_loops(b, scope, clauses, b.parameter(scope, 0), world, container)
    b.finish(scope, (container, world), (_REF, b.world))


def emit_clause_loops(b, scope, clauses, initial, world, container=None, *, suspension=P.SuspensionKind.YIELD):
    """One loop per clause; generator bodies have no synthetic container."""
    pending = []
    invariant_types = (_REF,) if container is None else (_REF, _REF)
    world_index = len(invariant_types)

    def inputs(iterator, container):
        return (iterator,) if container is None else (iterator, container)

    def unpack(region):
        return (b.parameter(region, 0), None if container is None else b.parameter(region, 1),
            b.parameter(region, world_index), b.parameter(region, world_index+1))

    def launch(parent, index, iterator, container, world):
        loop = b.region(clauses[index][0], (*invariant_types, b.world), parent, P.ProductionRegionKind.LOOP)
        pending.append((loop, index))
        world, = b.emit(parent, P.ApplyRegion(P.ProductionRegionCall(loop.id, (*inputs(iterator, container), world))), (), (b.world,))
        return world

    result = launch(scope, 0, initial, container, world)
    while pending:
        loop, index = pending.pop()
        target, iterable, filters, asynchronous = clauses[index]
        advance = b.region(target, (_REF, b.world), loop, P.ProductionRegionKind.LOOP_CONDITION)
        if asynchronous:
            from scar.analysis.await_protocol import emit_delegate_loop
            qualified, current = b.delegation(advance, P.DelegationProtocolKind.GET_ASYNC_NEXT,
                (b.parameter(advance, 0),), b.parameter(advance, 1))
            next_item = emit_delegate_loop(b, advance, qualified, current, P.SuspensionKind.AWAIT)
        else:
            next_item = b.protocol(advance, K.COMPREHENSION_NEXT,
                (b.parameter(advance, 0),), b.parameter(advance, 1), selectors=(iterable,))
        b.finish(advance, next_item, (_REF, b.world))
        exhausted = b.region(target, b.raised, loop, P.ProductionRegionKind.HANDLER)
        if asynchronous:
            stopped, = b.delegation(exhausted, P.DelegationProtocolKind.MATCH_STOP_ASYNC_ITERATION,
                (b.parameter(exhausted, 0),), b.parameter(exhausted, len(b.raised)-1))
        else:
            stopped, = b.protocol(exhausted, K.MATCH_STOP_ITERATION,
                (b.parameter(exhausted, 0),), b.parameter(exhausted, len(b.raised)-1), selectors=(iterable,))
        b.finish(exhausted, (stopped,), (_BOOL,))
        body = b.region(target, (*loop.parameters, _REF), loop, P.ProductionRegionKind.LOOP_BODY)
        condition = P.IterationCondition(P.ProductionRegionCall(advance.id,
            (b.parameter(loop, 0), b.parameter(loop, world_index))), exhausted.id, world_index)
        b.emit(loop, P.LoopControl(condition, body.id, invariant_count=world_index),
            tuple(b.parameter(loop, i) for i in range(len(loop.parameters))), (b.world,))
        iterator, current_container, world, item = unpack(body)
        world, = b.evaluate(body, target, world, P.ProductionPurpose.STORE, item)
        for ordinal in range(len(b.value(filters).items)):
            site = P.SlotRef(filters.occurrence, filters.group, filters.ordinal, (ordinal,))
            test, world = b.condition(body,site,world)
            skip = b.region(site, (b.world,), body)
            b.finish(skip, (b.parameter(skip, 0),), (b.world,))
            follow = b.region(site, body.parameters, body)
            b.emit(body, P.BranchControl(P.ProductionRegionCall(follow.id, (*inputs(iterator, current_container), world, item)),
                P.ProductionRegionCall(skip.id, (world,))), (test,), (b.world,))
            body = follow
            iterator, current_container, world, item = unpack(body)
        if index+1 < len(clauses):
            value, world = b.evaluate(body, clauses[index+1][1], world)
            nested, world = get_iterator(b, body, value, world, clauses[index+1][3])
            world = launch(body, index+1, nested, current_container, world)
        else:
            values = []
            output_names = ('key', 'value') if b.node.constructor == 'DictComp' else ('elt',)
            for name in output_names:
                value, world = b.evaluate(body, b.site(name), world)
                values.append(value)
            if current_container is None:
                from scar.analysis.await_protocol import emit_suspension
                _, world = emit_suspension(b, body, suspension, values[0], world)
            else:
                kind = {'ListComp':'LIST', 'SetComp':'SET', 'DictComp':'DICT'}[b.node.constructor]
                world, = b.protocol(body, K.COMPREHENSION_ADD, (current_container, *values), world,
                    (ChoiceBinding('comprehension_kind', kind),), tuple(b.site(name) for name in output_names))
        b.finish(body, (world,), (b.world,))
    return result


# Source-owned eager comprehension ports over ordinary core SSA/world.
def validate_comprehensions(model,graph,interface,metadata,semantic,language,*,source_tree=None,options=None,structural_only=False):
    from scar.analysis.static.lowering.source_comprehension import KINDS, shape, compiler_scope
    selected={p.operation:p for p in model.protocols if p.kind in KINDS}
    if not selected:return set()
    def need(ok,detail):
        if not ok:raise ValueError('source comprehension: '+detail)
    need(C.COMPREHENSION_314_V1 in language.capabilities and isinstance(source_tree,ast.Module),'explicit capability and retained original AST required')
    if C.COMPREHENSION_TARGETS_314_V1 in language.capabilities:
        _validate_targets(model,graph,interface,metadata,semantic,language,source_tree,options,structural_only)
        return set(selected)
    ops={o.id:o for o in graph.operations};values={v.id:v for v in graph.values}
    blocks={b.id:b for r in graph.regions for b in r.blocks};owner={o:b for b in blocks.values()for o in b.operations}
    links={r.operation:r for r in metadata.operation_links};use_links={(r.operation,r.use):r for r in metadata.use_links}
    protocols={p.operation:p for p in model.protocols};world=value_type(values[interface.world])
    reference=lambda value:type(values[value])is ReferenceValue
    def invoke(identity,count,result_types):
        op=ops[identity]
        need(op.terminator is not None and op.terminator.kind is TerminatorKind.INVOKE and op.callee is None and not op.effects_complete,
             'primitive must keep incomplete ordinary normal/raise invoke')
        need(len(op.uses)==count+1 and all(reference(u.value)for u in op.uses[:-1])
             and type(values[op.uses[-1].value])is StateVersionValue and value_type(values[op.uses[-1].value])==world
             and op.invoke_arguments==tuple(u.id for u in op.uses),'ordered primitive reference/world Uses differ')
        need(tuple((c.kind,c.types)for c in op.completions)==((ExitKind.NORMAL,(*result_types,world)),
             (ExitKind.RAISE,function_exception_types(language,_REF,world)))
             and tuple(e.kind for e in op.terminator.successors)==(ExitKind.NORMAL,ExitKind.RAISE)
             and all(e.arguments==tuple(CompletionResult(i)for i in range(len(c.types)))for e,c in zip(op.terminator.successors,op.completions)),
             'primitive must preserve its own payload/world completion order')
        return op,blocks[op.terminator.successors[0].block],blocks[op.terminator.successors[1].block]
    grouped={}
    for identity,p in selected.items():grouped.setdefault(p.source,{}).setdefault(p.kind,[]).append(identity)
    for source,kinds in grouped.items():
        need(set(kinds)=={K.COMPREHENSION_BUILD,K.COMPREHENSION_NEXT,K.COMPREHENSION_ADD,K.COMPREHENSION_LEAVE}
             and all(len(v)==1 for v in kinds.values()),'one original comprehension needs exactly build/next/add/leave')
        matches=[n for n in require_source_tree(source_tree,language.source,source)if isinstance(n,(ast.ListComp,ast.SetComp,ast.DictComp))]
        need(len(matches)==1 and shape(matches[0]),'original AST scope/strict supported clause differs')
        node=matches[0];clause=node.generators[0]
        build,next_,add,leave=(kinds[k][0]for k in(K.COMPREHENSION_BUILD,K.COMPREHENSION_NEXT,K.COMPREHENSION_ADD,K.COMPREHENSION_LEAVE))
        definition=semantic.definitions.get(links[build].semantic_operation)
        need(definition is not None and definition.kind is OperationKind.OPAQUE and definition.metadata.get('ast_type')==type(node).__name__
             and definition.code_id==source.atom_id.wire and len(definition.output_slots)==1 and links[build].opcode is Opcode.OPAQUE,
             'allocation lost the original source comprehension/output')
        slots=[s.slot_id for s in semantic.slots.values()if s.owner==definition.id and s.name==clause.target.id and s.metadata.get('scope_kind')=='comprehension']
        need(len(slots)==1,'target needs one original comprehension slot')
        slot=slots[0]
        op,allocated,_=invoke(build,0,(_REF,));container=allocated.arguments[0]
        nxt,body,failure=invoke(next_,1,(_REF,));iterator=nxt.uses[0].value
        producers=[o for o in ops.values()if protocols.get(o.id) is not None and protocols[o.id].kind is K.GET_ITERATOR
            and any(e.kind is ExitKind.NORMAL and blocks[e.block].arguments[0]==iterator for e in o.terminator.successors)]
        need(len(producers)==1,'loop iterator is not the original GET_ITERATOR normal result')
        get_source=links[producers[0].id].source
        need((get_source.start_line,get_source.end_line,get_source.start_column,get_source.end_column)==
             (clause.iter.lineno,clause.iter.end_lineno,clause.iter.col_offset,clause.iter.end_col_offset),'eager iterable source differs')
        # The exhaustion matcher consumes this NEXT's original failure; a body
        # exception cannot be mistaken for natural exhaustion.
        branch=ops[failure.operations[-1]]
        need(branch.terminator is not None and branch.terminator.kind is TerminatorKind.BRANCH,'next failure lacks exhaustion split')
        match=ops[failure.operations[-2]]
        need(protocols.get(match.id) is not None and protocols[match.id].kind is K.MATCH_STOP_ITERATION
             and match.uses[0].value==failure.arguments[0] and branch.uses[0].value==match.results[0],
             'exhaustion test must use this NEXT failure/boolean')
        natural=blocks[branch.terminator.successors[0].block]
        need(owner[leave].id==natural.id,'scope cleanup is not on this natural exhaustion edge')
        # An original target binding must receive precisely this iteration's
        # item before elt evaluation; its real source slot never aliases the
        # enclosing name merely because the spellings coincide.
        bind=ops[body.operations[0]];definition_op=ops[body.operations[1]]
        need(protocols.get(bind.id) is not None and protocols[bind.id].kind is K.FRAME_BIND
             and protocols[bind.id].source==links[definition_op.id].source
             and bind.uses[1].value==body.arguments[0] and len(definition_op.results)==1
             and type(values[definition_op.results[0]])is LocalBindingValue and values[definition_op.results[0]].slot==slot
             and definition_op.uses[0].value==body.arguments[0],'target did not retain the original per-item normal reference/slot')
        target_source=links[definition_op.id].source
        need((target_source.start_line,target_source.end_line,target_source.start_column,target_source.end_column)==
             (clause.target.lineno,clause.target.end_lineno,clause.target.col_offset,clause.target.end_col_offset),'target original AST differs')
        update,after,_=invoke(add,3 if isinstance(node,ast.DictComp)else 2,())
        need(update.uses[0].value==container,'container update changed the allocated receiver')
        children=(node.key,node.value)if isinstance(node,ast.DictComp)else(node.elt,)
        for i,(child,use)in enumerate(zip(children,update.uses[1:-1])):
            row=use_links[(add,use.id)]
            need(row.role=='element' and row.index==i and(row.source.start_line,row.source.end_line,row.source.start_column,row.source.end_column)==
                 (child.lineno,child.end_lineno,child.col_offset,child.end_col_offset),'key/value or element source evaluation order differs')
        jump=ops[after.operations[-1]]
        need(jump.terminator is not None and jump.terminator.kind is TerminatorKind.JUMP
             and jump.terminator.successors[0].block==owner[next_].id,'successful update must repeat original iterator header')
        cleanup=ops[leave]
        need(cleanup.terminator is None and cleanup.callee is None and not cleanup.effects_complete
             and len(cleanup.uses)==3 and cleanup.uses[1].value==iterator and type(values[cleanup.uses[0].value])is LocalBindingValue
             and values[cleanup.uses[0].value].slot==slot and len(cleanup.accesses)==1,'scope leave must release this transient binding through the current world')
        for read in graph.binding_reads:
            use=next(u for u in ops[read.operation].uses if u.id==read.binding)
            if values[use.value].slot!=slot:continue
            link=links[read.operation];d=semantic.definitions.get(link.semantic_operation)
            need(d is not None and link.opcode is Opcode.READ and d.code_id==link.source.atom_id.wire
                 and any(e.source.id==d.id and e.target.id==slot and e.relation is SemanticRelation.READS_SLOT for e in semantic.edges.values()),
                 'scoped read does not name its exact original SG binding slot')
        if not structural_only:
            need(options is not None and options.code_evidence is not None,'independent readonly CodeWitness required')
            options.code_evidence.require_definition(metadata.source,interface.definition,model.source)
            compiler_scope(node,options.code_evidence.code)
    return set(selected)


def _validate_targets(model,graph,interface,metadata,semantic,language,source_tree,options,structural_only):
    """Validate the explicit recursive-target/paired-scope capability.

    UNPACK_FIXED's existing checker owns arity and leaf-result correspondence;
    this checker owns the surrounding iteration, current scoped bindings and
    local cleanup before the original C0/C1 propagation corridor.
    """
    from scar.analysis.static.lowering.source_comprehension import KINDS,shape,target_leaves,scoped_target_slots,compiler_scope
    from scar.ir.program.model import OperationResult
    def need(ok,detail):
        if not ok:raise ValueError('source comprehension targets: '+detail)
    need(C.FUNCTION_TRY_314_V1 in language.capabilities and C.FIXED_UNPACK in language.capabilities,
         'recursive targets require the existing paired propagation and fixed-unpack capabilities')
    ops={o.id:o for o in graph.operations};values={v.id:v for v in graph.values}
    blocks={b.id:b for r in graph.regions for b in r.blocks}
    positions={o:(b.id,i)for b in blocks.values()for i,o in enumerate(b.operations)}
    protocols={p.operation:p for p in model.protocols}
    links={r.operation:r for r in metadata.operation_links}
    use_links={(r.operation,r.use):r for r in metadata.use_links}
    world=value_type(values[interface.world]);scoped_slots=scoped_target_slots(semantic,source_tree,language)
    incoming={key:[]for key in blocks}
    for op in ops.values():
        term=op.terminator
        for edge in getattr(term,'successors',()):incoming[edge.block].append(positions[op.id])
    memo={}
    def latest(slot,block_id,stop,visiting=frozenset()):
        key=(slot,block_id,stop)
        if key in memo:return memo[key]
        need(key not in visiting,'binding cycle lacks its original slot block argument')
        block=blocks[block_id]
        for identity in reversed(block.operations[:stop]):
            found=[v for v in ops[identity].results if type(values[v])is LocalBindingValue and values[v].slot==slot]
            if found:
                need(len(found)==1,'one statement produced ambiguous scoped binding versions')
                memo[key]=frozenset(found);return memo[key]
        arguments=[v for v in block.arguments if type(values[v])is LocalBindingValue and values[v].slot==slot]
        if arguments:
            need(len(arguments)==1,'scope header duplicates its slot argument')
            memo[key]=frozenset(arguments);return memo[key]
        result=set()
        for parent,index in incoming[block_id]:
            result.update(latest(slot,parent,index+1,visiting|{key}))
        need(bool(result),'scoped binding has no preceding original definition/header')
        memo[key]=frozenset(result);return memo[key]
    def kind(identity):
        row=protocols.get(identity)
        return None if row is None else row.kind
    def invoke(identity,count,result_types):
        op=ops[identity]
        need(op.terminator is not None and op.terminator.kind is TerminatorKind.INVOKE
             and op.callee is None and not op.effects_complete,'primitive lost its normal/raise boundary')
        need(len(op.uses)==count+1 and all(type(values[u.value])is ReferenceValue for u in op.uses[:-1])
             and type(values[op.uses[-1].value])is StateVersionValue
             and value_type(values[op.uses[-1].value])==world
             and op.invoke_arguments==tuple(u.id for u in op.uses),'ordered reference/world ports differ')
        need(tuple((c.kind,c.types)for c in op.completions)==((ExitKind.NORMAL,(*result_types,world)),
             (ExitKind.RAISE,function_exception_types(language,_REF,world)))
             and tuple(e.kind for e in op.terminator.successors)==(ExitKind.NORMAL,ExitKind.RAISE)
             and all(e.arguments==tuple(CompletionResult(i)for i in range(len(c.types)))
                     for e,c in zip(op.terminator.successors,op.completions)),
             'primitive dropped/changed the original paired completion or post-world')
        return op,blocks[op.terminator.successors[0].block],blocks[op.terminator.successors[1].block]
    packet_ops={values[packet.value].origin.operation:packet for packet in graph.exception_packets
                if isinstance(values[packet.value].origin,OperationResult)}
    def pair_handoff(block_id,start,payload):
        cursor=(block_id,start);seen=set();matched=False
        while cursor not in seen:
            seen.add(cursor);block=blocks[cursor[0]]
            need(cursor[1]<len(block.operations),'scope failure falls out before original propagation handoff')
            op=ops[block.operations[cursor[1]]]
            packet=packet_ops.get(op.id)
            if packet is not None:
                own={use.id:use for use in op.uses}
                need((own[packet.exception].value,own[packet.traceback].value)==payload,
                     'scope cleanup replaced the original exception/propagation traceback pair')
                matched=True
            term=op.terminator
            if term is None:cursor=(block.id,cursor[1]+1)
            elif term.kind in (TerminatorKind.PENDING_DISPATCH,TerminatorKind.CLEANUP_ROUTE):
                need(matched,'scope error reached pending routing without its original pair packet');return
            elif term.kind is TerminatorKind.JUMP and len(term.successors)==1:
                edge=term.successors[0]
                if not matched:
                    own={use.id:use for use in op.uses};target=blocks[edge.block]
                    forwarded=[]
                    for reference in payload:
                        indices=[i for i,argument in enumerate(edge.arguments)
                                 if argument in own and own[argument].value==reference]
                        need(len(indices)==1 and type(values[target.arguments[indices[0]]])is ReferenceValue,
                             'scope error failed to forward its exact original pair into the outer C0/C1 gate')
                        forwarded.append(target.arguments[indices[0]])
                    payload=tuple(forwarded)
                cursor=(edge.block,0)
            else:
                need(False,'scope error skipped its canonical C0/C1 pair handoff')
        need(False,'scope error cycles before its pair handoff')
    grouped={}
    for identity,p in protocols.items():
        if p.kind in KINDS:grouped.setdefault(p.source,{}).setdefault(p.kind,[]).append(identity)
    for source,kinds in grouped.items():
        need(set(kinds)==KINDS and all(len(kinds[k])==1 for k in(K.COMPREHENSION_BUILD,K.COMPREHENSION_NEXT,K.COMPREHENSION_ADD))
             and bool(kinds[K.COMPREHENSION_LEAVE]),'build/next/add/leave scope indices differ')
        matches=[n for n in require_source_tree(source_tree,language.source,source)
                 if isinstance(n,(ast.ListComp,ast.SetComp,ast.DictComp))]
        need(len(matches)==1 and shape(matches[0],recursive_targets=True),'original recursive target/clause differs')
        node=matches[0];clause=node.generators[0];leaves=target_leaves(clause.target)
        build,next_,add=(kinds[k][0]for k in(K.COMPREHENSION_BUILD,K.COMPREHENSION_NEXT,K.COMPREHENSION_ADD))
        definition=semantic.definitions.get(links[build].semantic_operation)
        need(definition is not None and definition.kind is OperationKind.OPAQUE
             and definition.code_id==source.atom_id.wire and definition.metadata.get('ast_type')==type(node).__name__
             and len(definition.output_slots)==1 and links[build].opcode is Opcode.OPAQUE,
             'comprehension lost its original source/output')
        refs={}
        for leaf in leaves:
            atoms=[a for a in semantic.source_atoms.values()if a.kind=='Name'
                   and a.reference.path==language.source.path and a.reference.fingerprint==language.source.fingerprint
                   and(a.reference.start_line,a.reference.end_line,a.reference.start_column,a.reference.end_column)==
                   (leaf.lineno,leaf.end_lineno,leaf.col_offset,leaf.end_col_offset)]
            need(len(atoms)==1 and atoms[0].reference in scoped_slots,'target leaf lacks its original source slot')
            refs[(leaf.lineno,leaf.end_lineno,leaf.col_offset,leaf.end_col_offset)]=atoms[0].reference
        slots={leaf.id:scoped_slots[refs[(leaf.lineno,leaf.end_lineno,leaf.col_offset,leaf.end_col_offset)]]for leaf in leaves}
        need(all(semantic.slots[slot].owner==definition.id for slot in slots.values()),'target was rebound into outer function scope')
        _,allocated,build_failure=invoke(build,0,(_REF,));container=allocated.arguments[0]
        nxt,body,next_failure=invoke(next_,1,(_REF,));iterator=nxt.uses[0].value
        gets=[o for o in ops.values()if kind(o.id)is K.GET_ITERATOR and o.terminator is not None
              and any(e.kind is ExitKind.NORMAL and blocks[e.block].arguments[0]==iterator for e in o.terminator.successors)]
        need(len(gets)==1,'loop iterator lost original eager GET_ITERATOR')
        get_source=links[gets[0].id].source
        need((get_source.start_line,get_source.end_line,get_source.start_column,get_source.end_column)==
             (clause.iter.lineno,clause.iter.end_lineno,clause.iter.col_offset,clause.iter.end_col_offset),
             'outer iterable source differs')
        plan=None
        if not structural_only:
            need(options is not None and options.code_evidence is not None,'independent static compiler input required')
            from scar.analysis.source_language import supports_cpython_compiler
            need(supports_cpython_compiler(options.code_evidence),'recursive target compiler protocol requires CPython 3.14.7')
            options.code_evidence.require_definition(metadata.source,interface.definition,model.source)
            plan=compiler_scope(node,options.code_evidence.code,recursive_targets=True)
        # Distinct unique names, not leaf occurrences, own saved bindings.
        need(len(set(slots.values()))==len(slots),'different names share an invented scoped slot')
        if isinstance(clause.target,(ast.Tuple,ast.List)):
            need(len(body.operations)>=2 and kind(body.operations[0])is K.FRAME_HOLD_TEMPORARY
                 and ops[body.operations[0]].uses[0].value==body.arguments[0]
                 and kind(body.operations[1])is K.UNPACK_FIXED,'NEXT item did not enter the original root unpack')
            root_unpack=ops[body.operations[1]]
            need(root_unpack.uses[0].value==body.arguments[0],'root unpack changed the exact NEXT result')
        # Ordered target binding occurrences (including repeated names). Shared
        # unpack validators separately check every N result -> child mapping.
        cursor=(body.id,0);seen=set();stored=[];unpack_pending={}
        pending_results=(body.arguments[0],)if isinstance(clause.target,(ast.Tuple,ast.List))else()
        while len(stored)<len(leaves):
            need(cursor not in seen,'target stores cycle before all original leaves');seen.add(cursor)
            block=blocks[cursor[0]];need(cursor[1]<len(block.operations),'target binding is missing')
            identity=block.operations[cursor[1]];op=ops[identity]
            if kind(identity)is K.FRAME_BIND:
                leaf=leaves[len(stored)];reference=refs[(leaf.lineno,leaf.end_lineno,leaf.col_offset,leaf.end_col_offset)]
                need(protocols[identity].source==reference and len(op.uses)==3
                     and type(values[op.uses[0].value])is LocalBindingValue and values[op.uses[0].value].slot==slots[leaf.id],
                     'leaf occurrence bound a different name/source/previous slot')
                need(latest(slots[leaf.id],block.id,cursor[1])==frozenset((op.uses[0].value,)),
                     'repeated/scoped target did not replace the latest binding')
                need(cursor[1]+1<len(block.operations),'target binding definition missing')
                defined=ops[block.operations[cursor[1]+1]]
                need(len(defined.results)==1 and type(values[defined.results[0]])is LocalBindingValue
                     and values[defined.results[0]].slot==slots[leaf.id] and len(defined.uses)==1
                     and defined.uses[0].value==op.uses[1].value,'target did not retain its ordered assigned reference')
                if isinstance(clause.target,ast.Name):need(op.uses[1].value==body.arguments[0],'Name target lost NEXT payload')
                else:
                    need(pending_results and pending_results[-1]==op.uses[1].value,
                         'leaf store changed the next published unpack result')
                    pending_results=pending_results[:-1]
                stored.append(identity);cursor=(block.id,cursor[1]+2);continue
            term=op.terminator
            if term is None:cursor=(block.id,cursor[1]+1)
            elif term.kind is TerminatorKind.INVOKE and kind(identity)is K.UNPACK_FIXED:
                need(pending_results and op.uses[0].value==pending_results[-1],
                     'nested unpack did not consume its exact next published reference')
                # UNPACK_FIXED consumes its input in either completion. On
                # failure only already-published outer siblings remain held.
                unpack_pending[identity]=tuple(reversed(pending_results[:-1]))
                normal=blocks[term.successors[0].block]
                pending_results=pending_results[:-1]+tuple(reversed(normal.arguments[:-1]))
                cursor=(normal.id,0)
            else:need(False,'target store path executes another expression before binding leaves')
        need(not pending_results,'target completed while a published unpack sibling remained held')
        update,after,_=invoke(add,3 if isinstance(node,ast.DictComp)else 2,())
        need(update.uses[0].value==container,'update changed original container')
        children=(node.key,node.value)if isinstance(node,ast.DictComp)else(node.elt,)
        for i,(child,use)in enumerate(zip(children,update.uses[1:-1])):
            row=use_links[(add,use.id)]
            need(row.role=='element' and row.index==i and(row.source.start_line,row.source.end_line,row.source.start_column,row.source.end_column)==
                 (child.lineno,child.end_lineno,child.col_offset,child.end_col_offset),'key/value/elt order differs')
        jump=ops[after.operations[-1]]
        need(jump.terminator is not None and jump.terminator.kind is TerminatorKind.JUMP
             and jump.terminator.successors[0].block==positions[next_][0],'update lost original loop backedge')
        need(len(next_failure.operations)==2,'NEXT failure acquired another error producer before exhaustion test')
        match,branch=(ops[i]for i in next_failure.operations)
        need(kind(match.id)is K.MATCH_STOP_ITERATION and match.uses[0].value==next_failure.arguments[0]
             and branch.terminator is not None and branch.terminator.kind is TerminatorKind.BRANCH
             and branch.uses[0].value==match.results[0],'only NEXT failure may select natural exhaustion')
        natural=blocks[branch.terminator.successors[0].block]
        false=blocks[branch.terminator.successors[1].block]
        # Traverse all successful scope paths; every potentially raising child
        # must pass a local leave before its pre-existing propagation corridor.
        pending=[(allocated.id,0),(natural.id,0)];visited=set();active={build};normal_leaves=set()
        while pending:
            point=pending.pop()
            if point in visited:continue
            visited.add(point);block=blocks[point[0]]
            need(point[1]<len(block.operations),'scope normal path falls through without leave')
            op=ops[block.operations[point[1]]]
            if kind(op.id)is K.COMPREHENSION_LEAVE:
                normal_leaves.add(op.id);continue
            term=op.terminator
            if term is None:pending.append((block.id,point[1]+1))
            elif term.kind is TerminatorKind.INVOKE:
                active.add(op.id);pending.extend((e.block,0)for e in term.successors if e.kind is ExitKind.NORMAL)
            elif term.kind in(TerminatorKind.JUMP,TerminatorKind.BRANCH):pending.extend((e.block,0)for e in term.successors)
            else:need(False,'scope normal path escapes before leave')
        need(len(normal_leaves)==1 and positions[next_][0] in {b for b,_ in visited},'normal exhaustion must have exactly one scope leave')
        normal_leave=next(iter(normal_leaves))
        need(positions[normal_leave][0]==natural.id,'normal scope leave is not on NEXT natural exhaustion')
        consumed=[ops[i].uses[0].value for i in natural.operations[:positions[normal_leave][1]]if kind(i)is K.FRAME_DISCARD]
        need(consumed==list(next_failure.arguments[:-1]),'natural exhaustion dropped/replaced exception or propagation traceback')
        def validate_leave(identity,failure,has_container):
            op=ops[identity];refs_used=op.uses[:-1]
            resources=[iterator]+([container]if failure and has_container else[])
            roles=['comprehension-exception-iterator'if failure else'exhausted-iterator']
            if failure and has_container:roles.append('comprehension-exception-container')
            need(op.terminator is None and not op.effects_complete and len(op.accesses)==1
                 and len(refs_used)==len(resources)+len(slots)
                 and [u.value for u in refs_used[:len(resources)]]==resources,
                 'scope cleanup changed its iterator/container or dropped binding tokens')
            names=[];seen_slots=set()
            for use,role in zip(refs_used[:len(resources)],roles):
                need(use_links[(identity,use.id)].role==role,'scope cleanup completion role differs')
            for use in refs_used[len(resources):]:
                token=values[use.value]
                need(type(token)is LocalBindingValue and token.slot in slots.values()and token.slot not in seen_slots,
                     'scope cleanup duplicates or misses a unique target binding')
                seen_slots.add(token.slot);names.append(next(name for name,slot in slots.items()if slot==token.slot))
                block,index=positions[identity]
                need(latest(token.slot,block,index)==frozenset((token.id,)),
                     'scope cleanup rolled back a partially assigned/repeated binding')
                need(use_links[(identity,use.id)].role==('comprehension-exception-binding'if failure else'scoped-binding'),
                     'scope binding cleanup role differs')
            if plan is not None:need(tuple(names)==plan['failure_order'if failure else'normal_order'],'compiler name restore order differs')
            access=op.accesses[0]
            need(access.location==values[interface.world].location and access.before==op.uses[-1].id
                 and access.after==op.results[0]and values[op.uses[-1].value].location==access.location
                 and type(values[access.after])is StateVersionValue,'scope cleanup lost current-world state chain')
        validate_leave(normal_leave,False,True)
        checked={normal_leave}
        for identity in active:
            op=ops[identity]
            failure=blocks[op.terminator.successors[1].block]
            payload=tuple(failure.arguments[:2]);start=false.id if identity==next_ else failure.id
            point=(start,0);seen=set();leave=None;released=[]
            while point not in seen:
                seen.add(point);block=blocks[point[0]]
                need(point[1]<len(block.operations),'scope error lost local cleanup')
                current=ops[block.operations[point[1]]]
                if kind(current.id)is K.COMPREHENSION_LEAVE:
                    leave=current.id;break
                need(kind(current.id)is K.FRAME_RELEASE_TEMPORARY
                     and use_links[(current.id,current.uses[0].id)].role=='function-exception-unwind',
                     'scope error ran another producer or outer cleanup before local leave')
                released.append(current.uses[0].value)
                point=(block.id,point[1]+1)
            need(leave is not None,'scope failure cycles without local leave')
            if identity in unpack_pending or identity in(build,next_,add):
                need(tuple(released)==unpack_pending.get(identity,()),
                     'failed unpack changed the remaining sibling release order or released an assigned binding again')
            validate_leave(leave,True,identity!=build);checked.add(leave)
            block,index=positions[leave]
            pair_handoff(block,index+1,payload)
        need(checked==set(kinds[K.COMPREHENSION_LEAVE]),'unowned or unreachable scope leave was inserted')
        for read in graph.binding_reads:
            use=next(u for u in ops[read.operation].uses if u.id==read.binding)
            token=values[use.value]
            if type(token)is not LocalBindingValue or token.slot not in slots.values():continue
            link=links[read.operation];definition_read=semantic.definitions.get(link.semantic_operation)
            need(definition_read is not None and link.opcode is Opcode.READ
                 and definition_read.code_id==link.source.atom_id.wire
                 and any(e.source.id==definition_read.id and e.target.id==token.slot and e.relation is SemanticRelation.READS_SLOT
                         for e in semantic.edges.values()),'scoped read escaped its original target slot')

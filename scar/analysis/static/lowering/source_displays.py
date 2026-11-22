"""Source displays."""

from __future__ import annotations

import ast
import dis
from scar.analysis.program_lowering import _Unsupported
from scar.analysis.static.lowering.source_call_lowering import _invoke, _instructions
from scar.analysis.static.lowering.source_heap_lowering import _source_definition, _associate
from scar.analysis.static.lowering.source_mapping_lowering import _transfer
from scar.ir.program.source import GapKind, SourceRole
from scar.ir.program.structured import PythonProtocolKind as K, PythonProtocolKind as Kind
from scar.ir.semantics_v2 import Opcode
from scar.analysis.source_expression_rules import _REF, _require
from scar.ir.program import source_productions as P
from scar.ir.program.signature_patterns import CountBinding
from collections import deque
from scar.analysis.source_language import (
    SourceLanguageOptions,
    require_source_tree,
    function_exception_types,
)
from scar.ir.program.model import (
    Behavior,
    BlockArgument,
    CompletionResult,
    ExitKind,
    OperationResult,
    ReferenceValue,
    StateVersionValue,
    TerminatorKind,
    ValueKind,
    ValueType,
    value_type,
)
from scar.ir.v2.semantic import OperationKind


# Ordered starred Tuple/List displays, using existing list primitive ports.
#
# A display has no callee, kwargs, parameter binding or implicit function frame.
# Its temporary list survives append/extend; element references are consumed on
# both completions, while initial BUILD_LIST steals only on success. Tuple
# conversion consumes the temporary list; a List display forwards that list.
KINDS=frozenset((K.BUILD_DISPLAY_LIST,K.EXTEND_DISPLAY,K.APPEND_DISPLAY,K.DISPLAY_LIST_TO_TUPLE))


def _shape(node):
    return (isinstance(node,(ast.Tuple,ast.List)) and isinstance(node.ctx,ast.Load)
        and any(isinstance(n,ast.Starred)for n in node.elts) and len(node.elts)<=30)


def _expected(node):
    if not _shape(node):raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX,node,'STARRED_DISPLAY_SHAPE_OR_LARGE_INCREMENTAL_INTERFACE_UNIMPLEMENTED')
    first=next(i for i,n in enumerate(node.elts)if isinstance(n,ast.Starred))
    stages=[(K.BUILD_DISPLAY_LIST,first,None)]
    stages.extend((K.EXTEND_DISPLAY if isinstance(n,ast.Starred)else K.APPEND_DISPLAY,1,i)for i,n in enumerate(node.elts[first:],first))
    if isinstance(node,ast.Tuple):stages.append((K.DISPLAY_LIST_TO_TUPLE,None,None))
    return stages


def _compiler(node,rows):
    expected=_expected(node);span=(node.lineno,node.end_lineno,node.col_offset,node.end_col_offset)
    names={'BUILD_LIST','BUILD_TUPLE','LIST_APPEND','LIST_EXTEND','CALL_INTRINSIC_1'}
    actual=[r for r in rows if tuple(r.positions)==span and r.opname in names]
    if len(actual)!=len(expected):raise _Unsupported(GapKind.CODE_VERSION,node,'STARRED_DISPLAY_EXACT_COMPILER_SEQUENCE_UNAVAILABLE')
    for row,(kind,arg,_)in zip(actual,expected):
        if kind is K.DISPLAY_LIST_TO_TUPLE:
            ok=row.opname=='CALL_INTRINSIC_1' and dis._intrinsic_1_descs[row.arg]=='INTRINSIC_LIST_TO_TUPLE'
        else:
            name={K.BUILD_DISPLAY_LIST:'BUILD_LIST',K.EXTEND_DISPLAY:'LIST_EXTEND',K.APPEND_DISPLAY:'LIST_APPEND'}[kind]
            ok=row.opname==name and row.arg==arg
        if not ok:raise _Unsupported(GapKind.CODE_VERSION,node,'STARRED_DISPLAY_EXACT_COMPILER_SEQUENCE_UNAVAILABLE')


def _hold(builder,cursor,node,reference,role):
    # Same ordinary lifetime transition as call expansion, with display-owned
    # roles: the call validator must not mistake an inner display for its own
    # callee/argument-packing stack.
    builder.frame_transition(cursor,node,((reference,node,'display-held-'+role),),K.FRAME_HOLD_TEMPORARY)
    builder.active_temporaries.append(reference)


def lower_starred_display(builder,node,cursor):
    stages=_expected(node);_compiler(node,_instructions(builder,node))
    definition=_source_definition(builder,node)
    if definition.metadata.get('ast_type')!=type(node).__name__ or len(definition.output_slots)!=1:
        raise _Unsupported(GapKind.SOURCE_MAPPING,node,'STARRED_DISPLAY_ORIGINAL_CONTAINER_SOURCE_REQUIRED')
    baseline=len(builder.active_temporaries);held=[];inputs=[]
    try:
        first=stages[0][1]
        for index,child in enumerate(node.elts[:first]):
            cursor,value=builder.expr(child,cursor)
            if cursor is None:return None,None
            _hold(builder,cursor,child,value,'element');held.append((child,value));inputs.append((value,child,'element',index,None))
        cursor,container,build=_invoke(builder,cursor,node,inputs,K.BUILD_DISPLAY_LIST,tuple(held),raise_consumed=(),result=True)
        _hold(builder,cursor,node,container,'container')
        for index,argument in enumerate(node.elts[first:],first):
            child=argument.value if isinstance(argument,ast.Starred)else argument
            cursor,value=builder.expr(child,cursor)
            if cursor is None:return None,None
            _hold(builder,cursor,child,value,'element')
            cursor,_,_=_invoke(builder,cursor,node,
                ((container,node,'display-container',0,None),(value,child,'starred-element'if isinstance(argument,ast.Starred)else'element',index,None)),
                K.EXTEND_DISPLAY if isinstance(argument,ast.Starred)else K.APPEND_DISPLAY,((child,value),))
        if isinstance(node,ast.Tuple):
            cursor,result,identity=_invoke(builder,cursor,node,((container,node,'display-container',0,None),),
                K.DISPLAY_LIST_TO_TUPLE,((node,container),),result=True)
            _associate(builder,identity,node,definition,Opcode.BUILD_TUPLE)
        else:
            from scar.analysis.static.lowering.source_expression_lowering import forward_source_reference
            forward_source_reference(builder,cursor,node,container)
            result=container;_transfer(builder,((node,container),),node)
            _associate(builder,build,node,definition,Opcode.BUILD_LIST)
        builder.link_value(builder.values_by_id[result],node,SourceRole.EXPRESSION)
        builder.gap(GapKind.EFFECT_SUMMARY,node,'ordered starred display construction is connected; iteration dispatch, partial mutation, finalizers, object identity and contents remain unproved inside each current-world interface')
        return cursor,result
    finally:del builder.active_temporaries[baseline:]


def validate_starred_displays(model,graph,interface,metadata,semantic,language,*,source_tree=None,options=None,structural_only=False):
    from scar.analysis.program_lowering import _REF
    from scar.analysis.source_language import require_source_tree,function_exception_types
    from scar.analysis.static.lowering.source_call_lowering import _pop_suffix, _check_failure_pending
    from scar.analysis.static.lowering.source_string import _ordered_next
    from scar.ir.program.model import Behavior,CompletionResult,ExitKind,ReferenceValue,StateVersionValue,TerminatorKind,value_type
    from scar.ir.program.source_language import SourceLanguageCapability as C
    selected={p.operation:p for p in model.protocols if p.kind in KINDS}
    if not selected:return set()
    def need(ok,detail):
        if not ok:raise ValueError('starred display: '+detail)
    need(C.STARRED_DISPLAY_314_V1 in language.capabilities and isinstance(source_tree,ast.Module),'capability and exact retained AST required')
    operations={o.id:o for o in graph.operations};values={v.id:v for v in graph.values};blocks={b.id:b for r in graph.regions for b in r.blocks}
    protocols={p.operation:p for p in model.protocols};links={r.operation:r for r in metadata.operation_links};uses={(r.operation,r.use):r for r in metadata.use_links}
    positions={o:(b.id,i)for b in blocks.values()for i,o in enumerate(b.operations)};world=value_type(values[interface.world])
    grouped={}
    for identity,p in selected.items():grouped.setdefault(p.source,[]).append(identity)
    for source,group in grouped.items():
        matches=[n for n in require_source_tree(source_tree,language.source,source)if isinstance(n,(ast.Tuple,ast.List))]
        need(len(matches)==1 and _shape(matches[0]),'original display shape differs')
        node=matches[0];expected=_expected(node)
        need(tuple(protocols[i].kind for i in group)==tuple(k for k,_,_ in expected),'ordered display stages differ from original AST')
        container=None;owned=set();stage_set=set(group)
        for n,(identity,(kind,count,index))in enumerate(zip(group,expected)):
            op=operations[identity];normal=blocks[op.terminator.successors[0].block]if op.terminator else None
            results=(_REF,)if kind in(K.BUILD_DISPLAY_LIST,K.DISPLAY_LIST_TO_TUPLE)else()
            arity=count if kind is K.BUILD_DISPLAY_LIST else 1 if kind is K.DISPLAY_LIST_TO_TUPLE else 2
            need(op.terminator is not None and op.terminator.kind is TerminatorKind.INVOKE and op.callee is None and not op.effects_complete
                 and Behavior.MEMORY in op.behaviors and len(op.uses)==arity+1 and op.invoke_arguments==tuple(u.id for u in op.uses)
                 and all(type(values[u.value])is ReferenceValue for u in op.uses[:-1])and type(values[op.uses[-1].value])is StateVersionValue,'display primitive operands/world/shape differ')
            need(tuple((c.kind,c.types)for c in op.completions)==((ExitKind.NORMAL,results+(world,)),(ExitKind.RAISE,function_exception_types(language,_REF,world)))
                 and tuple(e.kind for e in op.terminator.successors)==(ExitKind.NORMAL,ExitKind.RAISE)
                 and all(e.arguments==tuple(CompletionResult(i)for i in range(len(c.types)))for e,c in zip(op.terminator.successors,op.completions)), 'normal/raise completion ports differ')
            if kind is K.BUILD_DISPLAY_LIST:container=normal.arguments[0];children=node.elts[:count];operands=op.uses[:-1];indexes=range(count)
            elif kind is K.DISPLAY_LIST_TO_TUPLE:
                need(op.uses[0].value==container,'tuple conversion must consume its own temporary list');children=();operands=();indexes=()
            else:
                need(op.uses[0].value==container,'append/extend changed display receiver');argument=node.elts[index]
                children=(argument.value if isinstance(argument,ast.Starred)else argument,);operands=(op.uses[1],);indexes=(index,)
            for child,use,position in zip(children,operands,indexes):
                row=uses[identity,use.id];ref=row.source
                need(row.index==position and row.role==('starred-element'if kind is K.EXTEND_DISPLAY else'element')
                     and(ref.start_line,ref.end_line,ref.start_column,ref.end_column)==(child.lineno,child.end_lineno,child.col_offset,child.end_col_offset),'element source or evaluation order differs')
            owned.update(u.value for u in op.uses[:-1]);owned.update(normal.arguments[:-1])
            if n:_ordered_next(group[n-1],identity,stage_set,operations,blocks,positions)
            link=links[identity];association=kind is(K.DISPLAY_LIST_TO_TUPLE if isinstance(node,ast.Tuple)else K.BUILD_DISPLAY_LIST)
            if association:
                d=semantic.definitions.get(link.semantic_operation)
                need(d is not None and d.code_id==source.atom_id.wire and d.metadata.get('ast_type')==type(node).__name__
                     and len(d.output_slots)==1 and link.opcode is(Opcode.BUILD_TUPLE if isinstance(node,ast.Tuple)else Opcode.BUILD_LIST),'final container lost original source association')
            else:need(link.semantic_operation is None,'intermediate list stage invented a source operation')
        # Reuse the existing call-expansion suffix/paired-handoff checks. Display
        # holds are separate; no callee or call/binding qualification is added.
        ordered=list(operations);end=ordered.index(group[-1]);starts=[]
        for i,identity in enumerate(ordered[:ordered.index(group[0])+1]):
            p=protocols.get(identity);op=operations[identity]
            if p is not None and p.kind is K.FRAME_HOLD_TEMPORARY and len(op.uses)==2:
                row=uses.get((identity,op.uses[0].id))
                if row is not None and row.role=='display-held-element'and op.uses[0].value in owned:starts.append(i)
        begin=min(starts)if starts else ordered.index(group[0]);stack=();prefix=None;deferred=[]
        for identity in ordered[begin:end+1]:
            op=operations[identity];p=protocols.get(identity);kind=p.kind if p else None
            if kind is K.FRAME_HOLD_TEMPORARY and len(op.uses)==2:
                row=uses.get((identity,op.uses[0].id))
                if row is not None and op.uses[0].value in owned and (row.role=='display-held-element'or row.role=='display-held-container'and row.source==source):stack+=((identity,op.uses[0].value),)
            if op.terminator is None or op.terminator.kind is not TerminatorKind.INVOKE:continue
            normal_stack=failure_stack=stack
            if identity in stage_set:
                consumed=tuple(u.value for u in op.uses[:-1])if kind in(K.BUILD_DISPLAY_LIST,K.DISPLAY_LIST_TO_TUPLE)else(op.uses[-2].value,)
                normal_stack=_pop_suffix(stack,consumed)
                if kind is not K.BUILD_DISPLAY_LIST:failure_stack=normal_stack
            failure=blocks[op.terminator.successors[1].block]
            if C.FUNCTION_TRY_314_V1 in language.capabilities:
                from scar.analysis.static.lowering.source_function_try import exception_handoff_holds
                actual=exception_handoff_holds(failure.id,operations,protocols,blocks,uses)
            else:
                cleanup=operations[failure.operations[0]]
                need(protocols[cleanup.id].kind is K.FRAME_EXIT,'display failure lost frame cleanup')
                actual=tuple(u.value for u in cleanup.uses if uses[cleanup.id,u.id].role=='held-temporary')
            if identity in stage_set:prefix=_check_failure_pending(actual,failure_stack,prefix)
            else:deferred.append((actual,failure_stack))
            stack=normal_stack
        need(prefix is not None and (not stack if isinstance(node,ast.Tuple)else tuple(v for _,v in stack)==(container,)),'display did not transfer its result/temporary list correctly')
        for actual,local in deferred:_check_failure_pending(actual,local,prefix,exact=False)
        if not structural_only:
            need(options is not None and options.code_evidence is not None,'independent readonly CodeWitness required')
            options.code_evidence.require_definition(metadata.source,interface.definition,model.source)
            _compiler(node,tuple(dis.get_instructions(options.code_evidence.code)))
    return set(selected)


# Container display/target productions; original fields, not AST visitation.
def emit_display(b, region, state):
    if b.node.constructor == 'Dict':
        return _mapping(b, region, state)
    kind = b.node.constructor
    if kind != 'Set':
        b.mark('ctx', P.ProductionDispositionKind.MARKER)
        marker = b.value(b.site('ctx'))
        expected = {P.ProductionPurpose.VALUE:'Load', P.ProductionPurpose.STORE:'Store',
                    P.ProductionPurpose.DELETE:'Del'}[b.purpose]
        _require(marker.kind == 'NODE' and b.rules.source.context.syntax.nodes[marker.reference].constructor == expected,
                 'PURPOSE_SOURCE_MISMATCH', str(b.occurrence), 'display target context differs from its purpose')
    b.mark('elts', P.ProductionDispositionKind.EVALUATED_CHILD)
    elements = b.value(b.site('elts'))
    _require(elements.kind == 'LIST', 'RULE_SHAPE', str(b.occurrence), 'display requires an original element list')
    items = []
    for ordinal, item in enumerate(elements.items):
        _require(item.kind == 'NODE', 'RULE_SHAPE', str(b.occurrence), 'display element must be an expression')
        node = b.rules.source.context.syntax.nodes[item.reference]
        site, starred = b.site('elts', ordinal), node.constructor == 'Starred'
        if starred:
            _require(b.purpose is not P.ProductionPurpose.DELETE, 'SOURCE_CODEGEN_CONFLICT', str(site), 'starred deletion is invalid')
            context = b.auxiliary(item.reference, 'ctx', P.ProductionDispositionKind.MARKER)
            marker = b.value(context)
            expected = 'Store' if b.purpose is P.ProductionPurpose.STORE else 'Load'
            _require(marker.kind == 'NODE' and b.rules.source.context.syntax.nodes[marker.reference].constructor == expected,
                     'RULE_SHAPE', str(context), 'starred display context differs from its parent')
            site = b.auxiliary(item.reference, 'value', P.ProductionDispositionKind.TARGET_ADDRESS
                if b.purpose is P.ProductionPurpose.STORE else P.ProductionDispositionKind.EVALUATED_CHILD)
        items.append((site, starred))
    if b.purpose is P.ProductionPurpose.STORE:
        return _store(b, region, state, items)
    if b.purpose is P.ProductionPurpose.DELETE:
        for site, _ in items:
            state, = b.evaluate(region, site, state, P.ProductionPurpose.DELETE)
        return b.finish(region, (state,), (b.world,))

    sequence, held = None, []
    build_kind = K.BUILD_SET_DISPLAY if kind == 'Set' else K.BUILD_DISPLAY_LIST
    if len(items) > b.rules.call_profile['stack_guideline']:
        sequence, state = b.protocol(region, build_kind, (), state, (CountBinding('item_count', 0),), (b.site('elts'),))
    for site, starred in items:
        if starred and sequence is None:
            sequence, state = b.protocol(region, build_kind, tuple(held), state,
                (CountBinding('item_count', len(held)),), (b.site('elts'),))
        value, state = b.evaluate(region, site, state)
        if sequence is None:
            held.append(value)
        elif kind == 'Set':
            state, = b.action(region, P.LanguageActionKind.SET_UPDATE if starred else P.LanguageActionKind.SET_INSERT,
                             (site,), (sequence, value, state), (b.world,))
        else:
            state, = b.protocol(region, K.EXTEND_DISPLAY if starred else K.APPEND_DISPLAY,
                                 (sequence, value), state, selectors=(site,))
    if sequence is None:
        if kind == 'Tuple':
            if not held:
                sequence, state = b.literal(region, (), state, source=b.site('elts'))
            else:
                sequence, state = b.emit(region, P.ConstructValue(P.ConstructionKind.TUPLE, (b.site('elts'), b.site('ctx')),
                    P.TupleConstructionLayout.ITEMS),
                                         (*held, state), (_REF, b.world))
        else:
            sequence, state = b.protocol(region, build_kind, tuple(held), state,
                (CountBinding('item_count', len(held)),), (b.site('elts'),))
    elif kind == 'Tuple':
        sequence, state = b.protocol(region, K.DISPLAY_LIST_TO_TUPLE, (sequence,), state, selectors=(b.site('elts'),))
    b.finish(region, (sequence, state), (_REF, b.world))


def _store(b, region, state, items):
    stars = [i for i, (_, star) in enumerate(items) if star]
    _require(len(stars) <= 1, 'SOURCE_CODEGEN_CONFLICT', str(b.occurrence), 'assignment permits at most one starred target')
    assigned = b.parameter(region, 0)
    if stars:
        index = stars[0]
        _require(index < 256 and len(items)-index-1 < ((1 << 31)-1) >> 8,
                 'SOURCE_CODEGEN_CONFLICT', str(b.occurrence), 'extended target counts exceed the compiler encoding')
        unpacked = b.action(region, P.LanguageActionKind.UNPACK_EXTENDED,
            (b.site('elts'), b.site('elts', index)), (assigned, state), (*((_REF,)*len(items)), b.world))
    else:
        unpacked = b.protocol(region, K.UNPACK_FIXED, (assigned,), state,
            (CountBinding('child_count', len(items)),), (b.site('elts'),))
    state = unpacked[-1]
    for (site, _), value in zip(items, unpacked[:-1]):
        state, = b.evaluate(region, site, state, P.ProductionPurpose.STORE, value)
    b.finish(region, (state,), (b.world,))


def _mapping(b, region, state):
    b.mark('keys', P.ProductionDispositionKind.EVALUATED_CHILD)
    b.mark('values', P.ProductionDispositionKind.EVALUATED_CHILD)
    keys, values = b.value(b.site('keys')), b.value(b.site('values'))
    _require(keys.kind == values.kind == 'LIST' and len(keys.items) == len(values.items),
             'RULE_SHAPE', str(b.occurrence), 'dict key/value source lists differ')
    mapping, elements = None, 0

    def commit(update):
        nonlocal mapping, state
        if mapping is None:
            mapping = update
        else:
            state, = b.protocol(region, K.DICT_UPDATE, (mapping, update), state, selectors=(b.site('keys'), b.site('values')))

    def subdict(begin, end):
        nonlocal state
        result, pairs = None, []
        if (end-begin)*2 > b.rules.call_profile['stack_guideline']:
            result, state = b.protocol(region, K.BUILD_MAP, (), state, (CountBinding('pair_count', 0),), (b.site('keys'),))
        for ordinal in range(begin, end):
            key, state = b.evaluate(region, b.site('keys', ordinal), state)
            value, state = b.evaluate(region, b.site('values', ordinal), state)
            if result is None:
                pairs.extend((key, value))
            else:
                state, = b.protocol(region, K.MAP_ADD, (result, key, value), state,
                    selectors=(b.site('keys', ordinal), b.site('values', ordinal)))
        if result is None:
            result, state = b.protocol(region, K.BUILD_MAP, tuple(pairs), state,
                (CountBinding('pair_count', end-begin),), (b.site('keys'),))
        return result

    for ordinal, key in enumerate(keys.items):
        hole = key.kind == 'SCALAR' and key.scalar.to_python() is None
        _require(hole or key.kind == 'NODE', 'RULE_SHAPE', str(b.site('keys', ordinal)), 'dict key must be an expression or an unpack hole')
        if hole:
            b.slots[b.site('keys', ordinal)] = P.ProductionDispositionKind.METADATA
            if elements:
                commit(subdict(ordinal-elements, ordinal))
                elements = 0
            if mapping is None:
                mapping, state = b.protocol(region, K.BUILD_MAP, (), state, (CountBinding('pair_count', 0),), (b.site('keys'),))
            update, state = b.evaluate(region, b.site('values', ordinal), state)
            state, = b.protocol(region, K.DICT_UPDATE, (mapping, update), state, selectors=(b.site('values', ordinal),))
        elif elements*2 > b.rules.call_profile['stack_guideline']:
            commit(subdict(ordinal-elements, ordinal+1))
            elements = 0
        else:
            elements += 1
    if elements:
        commit(subdict(len(keys.items)-elements, len(keys.items)))
    if mapping is None:
        mapping, state = b.protocol(region, K.BUILD_MAP, (), state, (CountBinding('pair_count', 0),), (b.site('keys'),))
    b.finish(region, (mapping, state), (_REF, b.world))


# Independent, bounded checks of tuple-key ports and modeled pending refs.
#
# This is a query/validator over the existing graph, not a second lifetime IR.
# An unknown intermediate consumption rule retains a frontier; it never grants
# purity, runtime code-instance identity, or whole-frame lifetime correctness.
def _need(condition, detail):
    if not condition:
        raise ValueError("tuple key ownership: " + detail)


def validate_tuple_key_ownership(model, graph, interface, metadata, semantic, *,
                                 source_tree=None, options=None, structural_only=False, language=None):
    """Check exact tuple-key sources/ports and success-only input transfer.

    Held instances retain their FRAME_HOLD operation identity and order, even
    when several holds refer to the same ValueID. The local BUILD_TUPLE failure
    suffix is mandatory. Full outer-prefix/normal-continuation claims require
    an independently recoverable pending path, otherwise a frontier is returned.
    """
    _need(type(structural_only) is bool, "verification mode must be explicit")
    if not structural_only:
        _need(isinstance(source_tree, ast.Module) and type(options) is SourceLanguageOptions
              and options.code_evidence is not None,
              "source/code checking requires independent exact SourceLanguageOptions")
        options.code_evidence.require_definition(metadata.source, interface.definition, model.source)
    operations = {row.id: row for row in graph.operations}
    values = {row.id: row for row in graph.values}
    blocks = {block.id: block for region in graph.regions for block in region.blocks}
    protocols = {row.operation: row for row in model.protocols}
    links = {row.operation: row for row in metadata.operation_links}
    use_links = {(row.operation, row.use): row for row in metadata.use_links}
    tuples = {key for key, row in links.items() if row.opcode is Opcode.BUILD_TUPLE}
    normal_producers = {}
    for operation in graph.operations:
        if operation.terminator is not None and operation.terminator.kind is TerminatorKind.INVOKE:
            for edge in operation.terminator.successors:
                if edge.kind is ExitKind.NORMAL:
                    normal_producers.setdefault(edge.block, []).append(operation.id)

    def producer(identity):
        origin = values[identity].origin
        if type(origin) is OperationResult:
            return origin.operation
        if type(origin) is BlockArgument:
            choices = normal_producers.get(origin.block, ())
            return choices[0] if origin.index == 0 and len(choices) == 1 else None
        return None

    selected, pending = set(), []
    for operation in graph.operations:
        link, protocol = links[operation.id], protocols.get(operation.id)
        key_role = ("index" if link.opcode is Opcode.INDEX else
                    "element-key" if protocol is not None and protocol.kind is Kind.STORE_SUBSCRIPT else None)
        if key_role is not None:
            for use in operation.uses:
                source_use = use_links.get((operation.id, use.id))
                if source_use is not None and source_use.role == key_role:
                    key = producer(use.value)
                    if key in tuples:
                        pending.append(key)
    while pending:
        identity = pending.pop()
        if identity in selected:
            continue
        selected.add(identity)
        for use in operations[identity].uses:
            if type(values[use.value]) is ReferenceValue:
                child = producer(use.value)
                if child in tuples:
                    pending.append(child)
    if not selected:
        return {"tuple_keys": 0, "records": [], "frontiers": [],
                "modeled_pending_refs_verified": True, "whole_lifetime_verified": False}

    records, operands_by_id = {}, {}
    state_type = value_type(values[interface.world])
    ref_type = ValueType(ValueKind.REFERENCE)
    from scar.ir.program.source_language import SourceLanguageCapability
    paired = language is not None and SourceLanguageCapability.FUNCTION_TRY_314_V1 in language.capabilities
    raise_types = function_exception_types(language, ref_type, state_type)
    failure_holds = {}
    if paired:
        from scar.analysis.static.lowering.source_function_try import exception_handoff_holds, validate_function_try_protocols
        # The public query must not infer packet/cleanup ownership merely
        # from a three-port shape. Reuse the original paired dataflow and
        # C0/C1 source checks before following its actual unwind corridor.
        validate_function_try_protocols(model, graph, interface, metadata, semantic, language,
                                        source_tree=source_tree)
    for identity in sorted(selected, key=lambda key: key.wire):
        operation, link = operations[identity], links[identity]
        definition = semantic.definitions.get(link.semantic_operation)
        atom = semantic.source_atoms.get(link.source.atom_id)
        _need(definition is not None and definition.kind is OperationKind.OPERATOR
              and definition.metadata.get("ast_type") == "Tuple" and len(definition.output_slots) == 1
              and definition.code_id == link.source.atom_id.wire and link.source.atom_id in definition.source_atoms
              and atom is not None and atom.reference == link.source and atom.kind == "Tuple",
              "tuple lost its unique original SG/source/output-slot correspondence")
        _need(operation.terminator is not None and operation.terminator.kind is TerminatorKind.INVOKE
              and operation.callee is None and not operation.effects_complete
              and Behavior.MEMORY in operation.behaviors
              and tuple(use.id for use in operation.uses) == operation.invoke_arguments,
              "BUILD_TUPLE must retain its incomplete allocation invoke and exact Uses")
        operands = operation.uses[:-1]
        _need(all(type(values[use.value]) is ReferenceValue for use in operands)
              and type(values[operation.uses[-1].value]) is StateVersionValue
              and values[operation.uses[-1].value].location == values[interface.world].location,
              "tuple references/world do not retain their original typed ports")
        expected = ((ExitKind.NORMAL, (ref_type, state_type)), (ExitKind.RAISE, raise_types))
        _need(tuple((row.kind, row.types) for row in operation.completions) == expected
              and tuple(edge.kind for edge in operation.terminator.successors) == (ExitKind.NORMAL, ExitKind.RAISE)
              and all(edge.arguments == tuple(CompletionResult(index) for index in range(len(types)))
                      and tuple(value_type(values[value]) for value in blocks[edge.block].arguments) == types
                      for edge, (_, types) in zip(operation.terminator.successors, expected)),
              "tuple normal/raise payloads must be their own ordered CompletionResults")
        child_sources = []
        for index, use in enumerate(operands):
            row = use_links.get((identity, use.id))
            child = semantic.source_atoms.get(row.source.atom_id) if row is not None else None
            _need(row is not None and row.role == "element" and row.index == index
                  and child is not None and child.reference == row.source,
                  "tuple elements lost their original ordered child Uses")
            child_sources.append(row.source)
        if not structural_only:
            _need(source_tree is not None and options is not None and options.code_evidence is not None,
                  "tuple construction requires independent exact source and CodeWitness inputs")
            nodes = [node for node in require_source_tree(source_tree, metadata.source, link.source)
                     if isinstance(node, ast.Tuple)]
            _need(len(nodes) == 1 and isinstance(nodes[0].ctx, ast.Load)
                  and not any(isinstance(child, ast.Starred) for child in nodes[0].elts),
                  "tuple construction is not one exact fixed-arity Load expression")
            node, = nodes
            _need(tuple((child.lineno, child.end_lineno, child.col_offset, child.end_col_offset)
                        for child in node.elts)
                  == tuple((row.start_line, row.end_line, row.start_column, row.end_column)
                           for row in child_sources), "tuple child sequence differs from exact source")
            instructions = [row for row in dis.get_instructions(options.code_evidence.code)
                if tuple(row.positions) == (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset)]
            _need(not any(row.opname == "LOAD_CONST" and type(row.argval) is tuple for row in instructions),
                  "compiler constant tuple cannot be modeled as a new BUILD_TUPLE allocation")
            _need(sum(row.opname == "BUILD_TUPLE" and row.arg == len(operands) for row in instructions) == 1,
                  "tuple has no unique exact BUILD_TUPLE arity/position witness")
        operands_by_id[identity] = tuple((use.value, source) for use, source in zip(operands, child_sources))
        failure = blocks[operation.terminator.successors[1].block]
        if paired:
            held = exception_handoff_holds(failure.id, operations, protocols, blocks, use_links)
        else:
            _need(len(failure.operations) == 2 and protocols.get(failure.operations[0]) is not None
                  and protocols[failure.operations[0]].kind is Kind.FRAME_EXIT
                  and operations[failure.operations[1]].terminator.kind is TerminatorKind.RAISE,
                  "tuple failure must retain the original no-handler FrameExit/RAISE path")
            cleanup = operations[failure.operations[0]]
            held = tuple(use.value for use in cleanup.uses
                         if (row := use_links.get((cleanup.id, use.id))) is not None and row.role == "held-temporary")
        failure_holds[identity] = held
        parts = tuple(use.value for use in operands)
        _need(not parts or len(held) >= len(parts) and held[-len(parts):] == parts,
              "BUILD_TUPLE failure dropped/reordered pending element references before ERROR_NO_POP unwind")
        records[identity] = {"operation": identity.wire, "ordered_source_ports_verified": True,
            "code_witness_verified": not structural_only, "raise_element_retention_verified": True,
            "outer_pending_verified": False, "normal_pending_verified": True,
            "normal_cleanup_sites": set(), "frontiers": []}

    # Abstract pending states are derived from graph/control/source records,
    # never a lowerer-reported consumed set. Unknown contracts remain explicit.
    frontiers, frontier_seen = [], set()
    def frontier(kind, operation, affected):
        key = kind, operation.wire
        if key not in frontier_seen:
            frontier_seen.add(key); frontiers.append({"kind": kind, "operation": operation.wire})
        for identity in affected:
            row = records[identity]
            if key not in row["frontiers"]:
                row["frontiers"].append(key)
            row["normal_pending_verified"] = False

    region = next(row for row in graph.regions if row.id == graph.root)
    # hold tuple=(original FRAME_HOLD operation, original reference, source).
    states, queue = {region.entry: ((), frozenset(), frozenset())}, deque((region.entry,))
    preserving = frozenset((Kind.LOOKUP_GLOBAL, Kind.READ_FREE_CELL, Kind.READ_LOCAL_CELL,
        Kind.MAKE_CELL, Kind.GET_ITERATOR, Kind.ADVANCE_ITERATOR, Kind.TRUTH,
        Kind.MAKE_UNBOUND_LOCAL_ERROR, Kind.MATCH_STOP_ITERATION, Kind.READ_CODE_CONSTANT))
    maximum = max(128, len(blocks) * (len(selected) + 4))
    visits = 0
    def send(block, stack, unknown, affected):
        incoming = stack, frozenset(unknown), frozenset(affected)
        old = states.get(block)
        if old is not None:
            if old[0] != stack:
                common = []
                for left, right in zip(old[0], stack):
                    if left != right:
                        break
                    common.append(left)
                stack = tuple(common)
                token = ("UNRESOLVED_PENDING_JOIN", blocks[block].operations[0])
                unknown = frozenset(unknown) | old[1] | frozenset((token,))
                frontier(*token, old[2] | frozenset(affected))
            incoming = stack, old[1] | frozenset(unknown), old[2] | frozenset(affected)
        if old != incoming:
            states[block] = incoming; queue.append(block)

    while queue:
        visits += 1
        if visits > maximum:
            frontier("PENDING_DATAFLOW_BUDGET", next(iter(selected)), selected)
            break
        block_id = queue.popleft()
        stack, unknown, affected = states[block_id]
        for identity in blocks[block_id].operations:
            operation, protocol = operations[identity], protocols.get(identity)
            kind = protocol.kind if protocol is not None else None
            refs = tuple(use.value for use in operation.uses if type(values[use.value]) is ReferenceValue)
            if kind is Kind.FRAME_HOLD_TEMPORARY:
                _need(len(refs) == 1, "temporary hold has no unique reference")
                stack += ((identity, refs[0], protocol.source),)
            elif kind is Kind.FRAME_RELEASE_TEMPORARY:
                _need(len(refs) == 1, "temporary release has no unique reference")
                if stack and stack[-1][1] == refs[0]:
                    stack = stack[:-1]
                else:
                    token = ("UNRECOVERED_TEMPORARY_RELEASE", identity)
                    unknown |= frozenset((token,)); frontier(*token, affected)
            elif kind is Kind.FRAME_EXIT or paired and kind is Kind.FUNCTION_FRAME_EXIT_EXCEPTION_PACKET:
                held = tuple(use.value for use in operation.uses
                    if (row := use_links.get((identity, use.id))) is not None and row.role == "held-temporary")
                if affected and not unknown:
                    _need(held == tuple(item[1] for item in stack),
                          "normal/raise FrameExit lost, duplicated or reordered a modeled pending hold")
                    for key in affected:
                        records[key]["normal_cleanup_sites"].add(identity.wire)
            terminator = operation.terminator
            if terminator is None:
                continue
            if terminator.kind is TerminatorKind.INVOKE:
                normal_stack, failure_stack = stack, stack
                normal_affected, failure_affected = affected, affected
                if identity in selected:
                    expected = operands_by_id[identity]
                    parts = tuple(item[1:] for item in stack[-len(expected):]) if expected else ()
                    _need(unknown or parts == expected, "tuple inputs are not their ordered pending hold instances")
                    row = records[identity]
                    row["outer_pending_verified"] = not unknown
                    for token in unknown:
                        frontier(*token, (identity,))
                    normal_stack = stack[:-len(expected)] if expected else stack
                    normal_affected |= frozenset((identity,))
                    if not unknown:
                        held = failure_holds[identity]
                        _need(held == tuple(item[1] for item in stack),
                              "tuple failure changed its pending outer prefix or element hold instances")
                elif stack and kind not in preserving:
                    # Existing slice/index/map adapters have an input-transfer
                    # convention but are not independently proven here. Apply
                    # their structural suffix only to continue the query,
                    # retaining UNKNOWN instead of certifying that convention.
                    token = ("INTERMEDIATE_PRIMITIVE_CONSUMPTION_UNVERIFIED", identity)
                    unknown |= frozenset((token,)); frontier(*token, affected)
                    consumed = ()
                    if kind is Kind.BUILD_SLICE or links[identity].opcode is Opcode.INDEX:
                        consumed = refs
                    elif kind is Kind.UNPACK_FIXED:
                        consumed = refs[:1]
                    elif kind is Kind.BUILD_MAP:
                        consumed = refs
                    elif kind is Kind.MAP_ADD:
                        consumed = refs[-2:]
                    elif kind is Kind.DICT_UPDATE:
                        consumed = refs[-1:]
                    if consumed and tuple(item[1] for item in stack[-len(consumed):]) == consumed:
                        normal_stack = failure_stack = stack[:-len(consumed)]
                send(terminator.successors[0].block, normal_stack, unknown, normal_affected)
                send(terminator.successors[1].block, failure_stack, unknown, failure_affected)
            elif terminator.kind in (TerminatorKind.JUMP, TerminatorKind.BRANCH):
                for edge in terminator.successors:
                    send(edge.block, stack, unknown, affected)
            elif terminator.kind not in (TerminatorKind.RETURN, TerminatorKind.RAISE):
                frontier("UNSUPPORTED_PENDING_CONTROL", identity, affected)
            break
    rows = list(records.values())
    for row in rows:
        row["normal_cleanup_sites"] = sorted(row["normal_cleanup_sites"])
        row["normal_pending_verified"] &= bool(row["normal_cleanup_sites"])
        row["frontiers"] = [{"kind": kind, "operation": operation} for kind, operation in row["frontiers"]]
    return {"tuple_keys": len(rows), "records": rows, "frontiers": frontiers,
            "modeled_pending_refs_verified": not frontiers and all(
                row["outer_pending_verified"] and row["normal_pending_verified"] for row in rows),
            "whole_lifetime_verified": False, "runtime_code_instance_established": False,
            "physical_allocation_events_established": False}


__all__ = [
    'validate_tuple_key_ownership',
    'KINDS',
    'lower_starred_display',
    'validate_starred_displays',
    'emit_display',
]

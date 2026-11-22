"""Source class body."""

from __future__ import annotations

import ast
import dis
from dataclasses import replace
from types import CodeType
from scar.analysis.source_class_lowering import (
    _resolve_class,
    _class_header_inputs,
    _class_scope_statements,
    _compiler_cells,
    _frontiers,
    _stable_id,
    _node_reference,
)
from scar.ir.program.class_entries import (
    ClassSuiteOverlay,
    ClassSuiteProtocol,
    ClassSuiteInstruction,
    ClassSuiteCell,
    ClassSuiteConstant,
    ClassSuiteFactory,
    ClassSuiteResidual,
    ClassEntryResult,
    ClassSuiteMethodOverlay,
    ClassSuiteMethodLookup,
    ClassSuiteMethodCall,
    ClassFrontierKind,
    compiler_cell_protocol,
    ClassNormalPayloadKind,
    validate_class_suite_core,
)
from scar.ir.program.ids import (
    BlockID,
    OperationID,
    RegionID,
    ValueID,
    UseID,
    LocationID,
    ControlScopeID,
)
from scar.ir.program.model import (
    AccessMode,
    Behavior,
    Block,
    BlockArgument,
    CompletionResult,
    CompletionSignature,
    CompletionSuccessor,
    DataValue,
    ExitKind,
    ExitPort,
    LiteralOrigin,
    Operation,
    OperationResult,
    ProgramGraph,
    ReferenceValue,
    Region,
    Stability,
    StateAccess,
    StateLocation,
    StateVersionValue,
    Terminator,
    TerminatorKind,
    Use,
    UseRole,
    ValueKind,
    ValueType,
)
from scar.ir.program.source import (
    GapKind,
    LoweringGap,
    LoweringMetadata,
    OperationSource,
    UseSource,
    ValueSource,
    SourceRole,
)
from scar.ir.program.source_subjects import (
    ClassFrameABI,
    MappingFramePorts,
    SourceCompilationFrameRoot,
    FrameKind,
)
from scar.ir.program.control import ControlScope, ControlScopeKind
from scar.ir.semantics_v2 import Opcode
from scar.ir.v2.semantic import OperationKind
from scar.ir.program.state_addresses import (
    AddressCoverage,
    AddressEffect,
    CellContent,
    ExistingAccess,
    InvokeStateOperand,
    NamespaceKey,
    ObjectField,
    ReferenceOwner,
    StateAddressRefinement,
)
from scar.analysis.static_compilation import compile_source_unit
from scar.analysis.source_class_validation import (
    _source_semantics,
    _expected_header_inputs,
    _expected_body_statements,
)
from scar.ir.program.validation import require_valid


# Static class-suite construction from its exact source/compiler interface.
#
# The compiler stack is construction-only. Copies/borrows forward existing SSA
# references; actual cell, namespace, value, factory and state operations become
# canonical nodes. Unsupported control starts an exact compiler continuation,
# never a replay of already evaluated source or an invented function frame.
REF=ValueType(ValueKind.REFERENCE)
BOOL=ValueType(ValueKind.DATA,data_type='builtins.bool')
SUPPORTED=frozenset(('RESUME','COPY_FREE_VARS','MAKE_CELL','LOAD_CONST','LOAD_SMALL_INT',
    'LOAD_NAME','LOAD_GLOBAL','LOAD_LOCALS','STORE_NAME','DELETE_NAME','STORE_GLOBAL',
    'LOAD_DEREF','STORE_DEREF','LOAD_FROM_DICT_OR_DEREF','LOAD_FAST_BORROW',
    'BUILD_TUPLE','BUILD_LIST','BUILD_SET','BUILD_MAP','MAKE_FUNCTION','SET_FUNCTION_ATTRIBUTE',
    'COPY','SWAP','POP_TOP','PUSH_NULL','CALL','CALL_KW','LOAD_ATTR','BINARY_OP',
    'UNARY_NEGATIVE','UNARY_INVERT','UNARY_NOT','RETURN_VALUE','NOP','EXTENDED_ARG'))


class ClassSuiteBuilder:
    def __init__(self,prepared,definition,source,evidence,limit,*,profile='STATIC_CLASS_SUITE_314_V1'):
        if profile not in ('STATIC_CLASS_SUITE_314_V1','STATIC_CLASS_SUITE_METHOD_314_V1'):
            raise ValueError('class suite requires an explicit supported source profile')
        self.method_profile=profile=='STATIC_CLASS_SUITE_METHOD_314_V1'
        self.method_lookups=[];self.method_calls=[];self.method_carriers={}
        self.prepared,self.definition,self.source,self.evidence=prepared,definition,source,evidence
        self.snapshot,self.semantic_definition,self.node=_resolve_class(prepared,definition,source)
        evidence.require_definition(self.snapshot,definition,source)
        self.code=evidence.code;self.limit=limit;self.serial=0
        self.region=self.fresh(RegionID,'class-suite');self.scope=self.fresh(ControlScopeID,'class-frame')
        self.world=self.fresh(LocationID,'class-world');self.state_type=ValueType(ValueKind.STATE,location=self.world)
        self.ops=[];self.values=[];self.blocks=[];self.block_ops={};self.args={};self.records={}
        self.op_links=[];self.value_links=[];self.use_links=[];self.addresses=[]
        self.attribute_lookups=[]
        self.protocols=[];self.witnesses=[];self.constants=[];self.factories=[];self.cells=[]
        self.cell_refs={};self.stack=[];self.literal_values={};self.code_refs={};self.tuple_items={};self.function_codes={};self.returned=False;self.remaining=None
        from scar.analysis.static_compilation import compile_source_unit
        self.compilation_index=compile_source_unit(prepared)
        self.instructions=tuple(dis.get_instructions(self.code,adaptive=False,show_caches=False))
        self.normal_mode=compiler_cell_protocol(self.code)[1]
        self.owner_nodes=[n for n in ast.walk(self.node) if isinstance(n,(ast.expr,ast.stmt))]
        self.decorator_offsets=set();self.unclassified_applications=set()
        for n in self.owner_nodes:
            if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
            source_n=_node_reference(prepared,n)
            defs=[d for d in prepared.semantic.definitions.values()if d.code_id==source_n.atom_id.wire and d.metadata.get('phase')=='call_time']
            paths=[]
            for d in defs:
                try:proof=self.compilation_index.for_definition(d.id,source_n)
                except ValueError:continue
                if proof.code_path[:-1]==evidence.code_path:paths.append(proof.code_path)
            if len(paths)!=1:continue
            loads=[r for r in self.instructions if r.opname=='LOAD_CONST'and r.arg==paths[0][-1]and type(r.argval)is CodeType]
            binding=n.name
            if binding.startswith('__')and not binding.endswith('__')and self.node.name.lstrip('_'):
                binding='_'+self.node.name.lstrip('_')+binding
            stores=[r for r in self.instructions if r.opname=='STORE_NAME'and r.argval==binding and tuple(r.positions)==
                (n.lineno,n.end_lineno,n.col_offset,n.end_col_offset)and loads and r.offset>loads[0].offset]
            if len(loads)!=1 or len(stores)!=1:continue
            calls=[r for r in self.instructions if loads[0].offset<r.offset<stores[0].offset and r.opname in ('CALL','CALL_KW','CALL_FUNCTION_EX')]
            expected=[(d.lineno,d.end_lineno,d.col_offset,d.end_col_offset)for d in reversed(n.decorator_list)]
            if len(calls)==len(expected)and all(r.opname=='CALL'and r.arg==0 and tuple(r.positions)==s for r,s in zip(calls,expected)):
                self.decorator_offsets.update(r.offset for r in calls)
            else:self.unclassified_applications.update(r.offset for r in calls)
        refs=3+len(self.code.co_freevars)
        self.current=self.new_block((REF,)*refs+(self.state_type,));self.entry=self.current
        arguments=self.args[self.entry]
        self.namespaces=arguments[:3];self.free=dict(zip(self.code.co_freevars,arguments[3:-1]));self.state=arguments[-1]
        self.interface=ClassFrameABI(MappingFramePorts(*self.namespaces,tuple(arguments[3:-1]),self.state),
            tuple(self.code.co_freevars),tuple(self.code.co_cellvars),_compiler_cells(self.code))
        self.transition('FRAME_ENTER',tuple(arguments[:-1]),source,-1)

    def fresh(self,kind,role):
        self.serial+=1
        return kind(_stable_id(kind,'class-suite',self.definition,self.source.fingerprint,role,self.serial))

    def new_block(self,types):
        block=self.fresh(BlockID,'block');values=[]
        for index,t in enumerate(types):
            identity=self.fresh(ValueID,'argument')
            value=(ReferenceValue(identity,self.region,BlockArgument(block,index)) if t.kind is ValueKind.REFERENCE else
                   DataValue(identity,self.region,BlockArgument(block,index),t.data_type) if t.kind is ValueKind.DATA else
                   StateVersionValue(identity,self.region,BlockArgument(block,index),self.world))
            self.add_value(value,self.source,SourceRole.PARAMETER);values.append(identity)
        self.blocks.append(block);self.args[block]=tuple(values);self.block_ops[block]=[]
        return block

    def add_value(self,value,source,role=SourceRole.EXPRESSION):
        self.values.append(value);self.records[value.id]=value;self.value_links.append(ValueSource(value.id,source,role))

    def uses(self,op,values,source,roles=None,sources=None):
        result=[]
        for index,value in enumerate(values):
            identity=self.fresh(UseID,'use');record=self.records[value]
            result.append(Use(identity,value,UseRole.STATE_CONTENT if type(record)is StateVersionValue else
                              UseRole.DATA if type(record)is DataValue else UseRole.REFERENCE))
            self.use_links.append(UseSource(op,identity,sources[index] if sources else source,roles[index] if roles else 'operand',index))
        return tuple(result)

    def emit(self,op,source,role=SourceRole.EXPRESSION,definition=None):
        if len(self.ops)>=self.limit:raise ValueError('class suite operation budget exceeded')
        self.ops.append(op);self.block_ops[self.current].append(op.id)
        self.op_links.append(OperationSource(op.id,source,role,definition))

    def source_for(self,instruction):
        p=tuple(instruction.positions)
        exact=[n for n in self.owner_nodes if (n.lineno,n.end_lineno,n.col_offset,n.end_col_offset)==p]
        # Prefer an expression/statement's actual occurrence, never a made-up
        # AST for generated code. The containing ClassDef is the fallback.
        if exact:
            exact.sort(key=lambda n:(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)),isinstance(n,ast.expr)),reverse=True)
            try:return _node_reference(self.prepared,exact[0])
            except ValueError:pass
        return self.source

    def literal(self,value,source):
        identity=self.fresh(ValueID,'literal')
        self.add_value(ReferenceValue(identity,self.region,LiteralOrigin(value),targets=()),source)
        self.literal_values[identity]=value
        return identity

    def transition(self,kind,refs,source,offset):
        identity=self.fresh(OperationID,kind);uses=self.uses(identity,(*refs,self.state),source)
        after=self.fresh(ValueID,'world');self.add_value(StateVersionValue(after,self.region,OperationResult(identity,0),self.world),source,SourceRole.STATE)
        op=Operation(identity,kind,(Behavior.READ,Behavior.WRITE,Behavior.MEMORY),uses,(after,),
                     (StateAccess(self.world,uses[-1].id,AccessMode.READ_WRITE,after),),effects_complete=False)
        self.emit(op,source,SourceRole.STATE);self.protocols.append(ClassSuiteProtocol(identity,kind,source,offset));self.state=after
        return identity

    def live(self,extra=()):
        refs=(*self.namespaces,*self.free.values(),*self.cell_refs.values(),*(x for x in self.stack if x is not None),*extra)
        return tuple(dict.fromkeys(v for v in refs if type(self.records[v])is ReferenceValue))

    def invoke(self,kind,operands,source,offset,results=1,address=None,definition=None,roles=None,use_sources=None,
               result_types=None,consumed_on_raise=False):
        identity=self.fresh(OperationID,kind);uses=self.uses(identity,(*operands,self.state),source,
            (*roles,'world') if roles is not None else None,(*use_sources,source) if use_sources is not None else None)
        normal_types=(tuple(result_types)if result_types is not None else(REF,)*results)+(self.state_type,)
        normal=self.new_block(normal_types);failure=self.new_block((REF,REF,self.state_type))
        signatures=(CompletionSignature(ExitKind.NORMAL,normal_types),CompletionSignature(ExitKind.RAISE,(REF,REF,self.state_type)))
        op=Operation(identity,kind,(Behavior.CONTROL,Behavior.READ,Behavior.WRITE,Behavior.MEMORY),uses,
            effects_complete=False,terminator=Terminator(TerminatorKind.INVOKE,tuple(
                CompletionSuccessor(s.kind,b,tuple(CompletionResult(i)for i in range(len(s.types))))
                for s,b in zip(signatures,(normal,failure)))),completions=signatures,invoke_arguments=tuple(u.id for u in uses))
        self.emit(op,source,definition=definition);self.protocols.append(ClassSuiteProtocol(identity,kind,source,offset))
        if address:
            effects=tuple(AddressEffect(ReferenceOwner(uses[owner].id),
                (NamespaceKey(uses[key].id) if type(key)is int else key,),mode)for owner,key,mode in address)
            self.addresses.append(StateAddressRefinement(InvokeStateOperand(identity,uses[-1].id),effects,AddressCoverage.PARTIAL))
        # Frame cleanup receives the original pair and post-operation world.
        # Historical compounds borrow their inputs. The method-capable
        # profile explicitly consumes lookup owners/call inputs internally;
        # those must not be kept alive solely by exceptional frame cleanup.
        prior_current,prior_state=self.current,self.state
        self.current=failure;exception,traceback,self.state=self.args[failure]
        self.transition('FRAME_EXIT_EXCEPTION',(*self.live(()if consumed_on_raise else operands),exception,traceback),source,offset)
        self.exit((exception,traceback),ExitKind.RAISE,source)
        self.current=normal;self.state=self.args[normal][-1]
        return tuple(self.args[normal][:-1]),identity

    def exit(self,payload,kind,source):
        identity=self.fresh(OperationID,'exit');uses=self.uses(identity,(*payload,self.state),source)
        self.emit(Operation(identity,'class-body-'+kind.value.lower(),(Behavior.CONTROL,),uses,
            terminator=Terminator(TerminatorKind.RETURN if kind is ExitKind.NORMAL else TerminatorKind.RAISE,
                exit=kind.value.lower(),values=tuple(u.id for u in uses))),source,
            SourceRole.RETURN if kind is ExitKind.NORMAL else SourceRole.EXCEPTION)

    def pop(self):
        if not self.stack:raise ValueError('class compiler operand stack underflow')
        return self.stack.pop()

    def handle(self,i):
        source=self.source_for(i);name=i.opname;arg=i.argval;identity=None
        if name in ('RESUME','COPY_FREE_VARS','NOP','EXTENDED_ARG'):return None
        if name=='PUSH_NULL':self.stack.append(None)
        elif name=='COPY':self.stack.append(self.stack[-i.arg])
        elif name=='SWAP':self.stack[-1],self.stack[-i.arg]=self.stack[-i.arg],self.stack[-1]
        elif name=='LOAD_FAST_BORROW':
            if arg not in self.cell_refs and arg not in self.free:raise ValueError('class borrow has no original owned/free cell')
            self.stack.append(self.cell_refs.get(arg,self.free.get(arg)))
        elif name=='LOAD_LOCALS':self.stack.append(self.namespaces[0])
        elif name=='LOAD_SMALL_INT':self.stack.append(self.literal(arg,source))
        elif name=='LOAD_CONST':
            if type(i.arg)is not int or self.code.co_consts[i.arg] is not arg:raise ValueError('class constant slot differs')
            identity=self.fresh(OperationID,'class-code-constant');before=self.uses(identity,(self.state,),source)
            value=self.fresh(ValueID,'code-reference');self.add_value(ReferenceValue(value,self.region,OperationResult(identity,0)),source)
            self.emit(Operation(identity,'READ_CODE_CONSTANT',(Behavior.READ,),before,(value,),
                (StateAccess(self.world,before[0].id,AccessMode.READ),),effects_complete=False),source)
            self.protocols.append(ClassSuiteProtocol(identity,'READ_CODE_CONSTANT',source,i.offset))
            self.constants.append(ClassSuiteConstant(identity,value,(i.arg,)));self.stack.append(value)
            if type(arg)is CodeType:self.code_refs[value]=(i.arg,arg)
            else:self.literal_values[value]=arg
        elif name=='MAKE_CELL':
            if arg not in self.code.co_cellvars or arg in self.cell_refs:raise ValueError('class own cell allocation differs')
            result,identity=self.invoke('MAKE_CELL',(),source,i.offset)
            self.cell_refs[arg]=result[0];self.cells.append(ClassSuiteCell(arg,self.code.co_cellvars.index(arg),result[0],identity))
        elif name in ('LOAD_DEREF','LOAD_FROM_DICT_OR_DEREF'):
            cell=self.cell_refs.get(arg,self.free.get(arg))
            if cell is None:raise ValueError('class deref has no declared cell')
            refs=(cell,) if name=='LOAD_DEREF' else (self.pop(),self.literal(arg,source),cell)
            addresses=((0,CellContent(),AccessMode.READ),) if name=='LOAD_DEREF' else ((0,1,AccessMode.READ),(2,CellContent(),AccessMode.READ))
            result,identity=self.invoke(name,refs,source,i.offset,address=addresses)
            self.stack.append(result[0])
        elif name=='STORE_DEREF':
            cell=self.cell_refs.get(arg,self.free.get(arg));value=self.pop()
            if cell is None:raise ValueError('class deref write has no declared cell')
            _,identity=self.invoke('STORE_CELL',(cell,value),source,i.offset,results=0,address=((0,CellContent(),AccessMode.WRITE),))
        elif name in ('LOAD_NAME','LOAD_GLOBAL'):
            key=self.literal(arg,source);refs=self.namespaces if name=='LOAD_NAME' else self.namespaces[1:]
            result,identity=self.invoke('LOAD_NAME_314' if name=='LOAD_NAME' else 'LOAD_GLOBAL',(*refs,key),source,i.offset,
                address=tuple((n,len(refs),AccessMode.READ)for n in range(len(refs))))
            self.stack.append(result[0])
            if name=='LOAD_GLOBAL' and i.arg&1:self.stack.append(None)
        elif name in ('STORE_NAME','STORE_GLOBAL','DELETE_NAME'):
            key=self.literal(arg,source);owner=self.namespaces[1] if name=='STORE_GLOBAL' else self.namespaces[0]
            refs=(owner,key) if name=='DELETE_NAME' else (owner,key,self.pop())
            _,identity=self.invoke('STORE_NAME_314' if name=='STORE_NAME' else name,refs,source,i.offset,results=0,
                address=((0,1,AccessMode.WRITE),),roles=('locals-owner','namespace-key','assigned-reference') if len(refs)==3 else None)
        elif name in ('BUILD_TUPLE','BUILD_LIST','BUILD_SET','BUILD_MAP'):
            count=i.arg*(2 if name=='BUILD_MAP' else 1);refs=tuple(self.pop()for _ in range(count))[::-1]
            result,identity=self.invoke(name,refs,source,i.offset);self.stack.append(result[0])
            if name=='BUILD_TUPLE':self.tuple_items[result[0]]=refs
        elif name=='MAKE_FUNCTION':
            code=self.pop()
            if code not in self.code_refs:raise ValueError('class factory lacks its actual child code reference')
            index,child=self.code_refs[code]
            result,identity=self.invoke('MAKE_FUNCTION_314',(code,self.namespaces[1]),source,i.offset,
                roles=('function-code','function-globals'))
            self.stack.append(result[0]);self.function_codes[result[0]]=child;definition=None
            candidates=[d for d in self.prepared.semantic.definitions.values() if d.code_id==source.atom_id.wire
                and d.metadata.get('phase')=='call_time']
            matched=[]
            for candidate in candidates:
                try:proof=self.compilation_index.for_definition(candidate.id,source)
                except ValueError:continue
                if proof.code_path==self.evidence.code_path+(index,):matched.append(candidate.id)
            if len(matched)==1:
                definition=matched[0]
                declarations=[d for d in self.prepared.semantic.definitions.values() if d.code_id==source.atom_id.wire
                    and d.metadata.get('phase')=='definition_time']
                if len(declarations)==1:
                    link_index=next(n for n,link in enumerate(self.op_links)if link.operation==identity)
                    self.op_links[link_index]=replace(self.op_links[link_index],semantic_operation=declarations[0].id)
            self.factories.append(ClassSuiteFactory(identity,result[0],code,(index,),definition,source,tuple(child.co_freevars)))
        elif name=='SET_FUNCTION_ATTRIBUTE':
            function=self.pop();value=self.pop()
            field={1:'__defaults__',2:'__kwdefaults__',4:'__annotations__',8:'__closure__',16:'__annotate__'}.get(i.arg)
            if field is None:raise ValueError('class function attribute flag unsupported')
            if function not in self.function_codes:raise ValueError('class attribute initialization lost its fresh function result')
            if field=='__closure__':
                expected=tuple(self.cell_refs.get(n,self.free.get(n))for n in self.function_codes[function].co_freevars)
                if None in expected or self.tuple_items.get(value)!=expected:
                    raise ValueError('class function closure did not capture original ordered cell references')
            _,identity=self.invoke('SET_FUNCTION_ATTRIBUTE',(function,value),source,i.offset,results=0,
                address=((0,ObjectField(field),AccessMode.WRITE),));self.stack.append(function)
        elif name=='LOAD_ATTR':
            owner=self.pop();key=self.literal(arg,source)
            if i.arg&1:
                result,identity=self.invoke('LOAD_METHOD_314',(owner,key),source,i.offset,
                    result_types=(REF,REF,BOOL),consumed_on_raise=True,
                    address=((0,ObjectField(arg),AccessMode.READ),),roles=('method-owner','attribute-key'))
                op=next(o for o in self.ops if o.id==identity)
                record=ClassSuiteMethodLookup(identity,op.uses[0].id,op.uses[1].id,*result)
                self.method_lookups.append(record);self.method_carriers[result[1]]=record
                self.stack.extend(result[:2])
            else:
                result,identity=self.invoke('LOAD_ATTRIBUTE',(owner,key),source,i.offset,address=((0,ObjectField(arg),AccessMode.READ),))
                self.stack.append(result[0])
            from scar.analysis.source_attribute_lookup import register_attribute_lookup
            from scar.ir.program.attribute_lookups import AttributeLookupForm
            nodes=[n for n in self.owner_nodes if isinstance(n,ast.Attribute) and isinstance(n.ctx,(ast.Load,ast.Store))
                and (n.lineno,n.end_lineno,n.col_offset,n.end_col_offset)==tuple(i.positions)]
            if len(nodes)!=1:raise ValueError('compiler attribute lookup lacks its exact original source occurrence')
            register_attribute_lookup(self,nodes[0],identity,source=source,
                form=AttributeLookupForm.CPYTHON_LOAD_METHOD if i.arg&1 else AttributeLookupForm.CPYTHON_LOAD_ATTR,
                key_use=next(o for o in self.ops if o.id==identity).uses[1].id)
        elif name in ('CALL','CALL_KW'):
            keyword=self.pop() if name=='CALL_KW' else None
            args=tuple(self.pop()for _ in range(i.arg))[::-1];receiver=self.pop();callee=self.pop()
            if callee is None:raise ValueError('class CALL lacks a callable reference')
            method=self.method_carriers.get(receiver)if self.method_profile else None
            if receiver is not None and method is None:args=(receiver,*args)
            operands=((callee,receiver,method.has_self,*args,*((keyword,)if keyword is not None else ()))if method is not None
                      else(callee,*args,*((keyword,)if keyword is not None else ())))
            application=i.offset in self.decorator_offsets
            call_nodes=[n for n in self.owner_nodes if isinstance(n,ast.Call)and tuple(i.positions)==
                (n.lineno,n.end_lineno,n.col_offset,n.end_col_offset)]
            actual_call=call_nodes[0]if len(call_nodes)==1 and not application and i.offset not in self.unclassified_applications else None
            operand_sources=None;semantic=None
            roles=(('callee','method-self-or-padding','method-has-self',*('argument'for _ in args),*(('keyword-names',)if keyword is not None else()))if method is not None
                   else('callee',*('argument'for _ in operands[1:])))
            if actual_call is not None and(receiver is None or method is not None)and not any(isinstance(a,ast.Starred)for a in actual_call.args):
                exprs=(actual_call.func,*actual_call.args,*(k.value for k in actual_call.keywords))
                if len(exprs)==1+len(args):
                    expression_sources=tuple(_node_reference(self.prepared,n)for n in exprs)
                    operand_sources=((expression_sources[0],source,source,*expression_sources[1:])if method is not None else expression_sources)+((source,)if keyword is not None else ())
                    definitions=[d for d in self.prepared.semantic.definitions.values()if d.code_id==source.atom_id.wire
                        and d.output_slots and d.metadata.get('role')!='binding'
                        and (d.metadata.get('ast_type')=='Call' or d.id in self.prepared.semantics.operations
                             and self.prepared.semantics.operations[d.id].opcode is Opcode.CALL)]
                    if len(definitions)==1:semantic=definitions[0].id
            kind='CALL_METHOD_314'if method is not None else 'APPLY_DECORATOR_314'if application else 'CALL'
            result,identity=self.invoke(kind,operands,source,i.offset,roles=roles,use_sources=operand_sources,definition=semantic,
                                        consumed_on_raise=self.method_profile)
            if method is not None:
                op=next(o for o in self.ops if o.id==identity)
                self.method_calls.append(ClassSuiteMethodCall(identity,method.operation,
                    op.uses[0].id,op.uses[1].id,op.uses[2].id,
                    tuple(u.id for u in op.uses[3:3+len(args)]),op.uses[-2].id if keyword is not None else None))
            if actual_call is not None and operand_sources is not None:
                link_index=next(n for n,link in enumerate(self.op_links)if link.operation==identity)
                self.op_links[link_index]=replace(self.op_links[link_index],opcode=Opcode.CALL)
            self.stack.append(result[0])
        elif name in ('BINARY_OP','UNARY_NEGATIVE','UNARY_INVERT','UNARY_NOT'):
            refs=tuple(self.pop()for _ in range(2 if name=='BINARY_OP' else 1))[::-1]
            result,identity=self.invoke(name,refs,source,i.offset);self.stack.append(result[0])
        elif name=='POP_TOP':
            value=self.pop();identity=self.transition('FRAME_DISCARD',(value,),source,i.offset)
        elif name=='RETURN_VALUE':
            value=self.pop()
            if self.stack:raise ValueError('class normal return has leftover compiler stack')
            identity=self.transition('FRAME_EXIT_NORMAL',(*self.live(),value),source,i.offset)
            self.exit((value,),ExitKind.NORMAL,source);self.returned=True
        else:raise ValueError('unsupported class compiler instruction '+name)
        return identity

    def preflight(self,i):
        """Admission is read-only; a debt starts with the unmodified stack."""
        name,arg=i.opname,i.argval
        if name not in SUPPORTED:return 'class compiler/control protocol requires expansion'
        if name=='LOAD_ATTR'and i.arg&1 and not self.method_profile:return 'method-lookup two-result ABI requires expansion'
        if name=='LOAD_FAST_BORROW'and arg not in self.cell_refs and arg not in self.free:
            return 'class compiler local borrow requires its scoped local interface'
        if name in ('LOAD_DEREF','LOAD_FROM_DICT_OR_DEREF','STORE_DEREF')and arg not in self.cell_refs and arg not in self.free:
            return 'class cell access lacks its original declared cell interface'
        if name=='MAKE_FUNCTION'and(not self.stack or self.stack[-1]not in self.code_refs):
            return 'function creation requires original compiler code-slot provenance'
        if name=='SET_FUNCTION_ATTRIBUTE':
            if len(self.stack)<2 or self.stack[-1]not in self.function_codes or i.arg not in (1,2,4,8,16):
                return 'function attribute requires its fresh factory/attribute protocol'
            if i.arg==8:
                expected=tuple(self.cell_refs.get(n,self.free.get(n))for n in self.function_codes[self.stack[-1]].co_freevars)
                if None in expected or self.tuple_items.get(self.stack[-2])!=expected:
                    return 'function closure requires exact ordered compiler cell provenance'
        if name in ('CALL','CALL_KW'):
            start=len(self.stack)-i.arg-(1 if name=='CALL_KW'else 0)-2
            if start<0 or self.stack[start]is None:return 'class CALL requires its exact callable/receiver ABI'
            if self.method_profile and self.stack[start+1]in self.method_carriers:
                carrier=self.method_carriers[self.stack[start+1]]
                if self.stack[start]!=carrier.callable:return 'method CALL requires its matching lookup callable/self carrier'
        return None

    def run(self):
        # This version is a linear compiler/source profile. Branches and
        # handlers stay a precise continuation instead of being executed.
        for i in self.instructions:
            before=tuple(self.stack)
            unsupported=self.preflight(i)
            if unsupported is not None:
                self.continuation(i,unsupported);break
            identity=self.handle(i)
            self.witnesses.append(ClassSuiteInstruction(i.offset,i.opname,i.arg,tuple(i.positions),
                self.source_for(i),identity,before,tuple(self.stack)))
            if self.returned:break
        if not self.returned:raise ValueError('class suite has no final continuation or normal return')
        graph=ProgramGraph(self.region,(Region(self.region,'static class suite',self.entry,
            tuple(Block(b,self.args[b],tuple(self.block_ops[b]))for b in self.blocks),
            (ExitPort('normal',ExitKind.NORMAL,(REF,self.state_type)),ExitPort('raise',ExitKind.RAISE,(REF,REF,self.state_type)))),),
            tuple(self.ops),tuple(self.values),(StateLocation(self.world,'coarse Python/external world',stability=Stability.UNKNOWN),),
            control_scopes=(ControlScope(self.scope,self.region,None,ControlScopeKind.FRAME,self.source,tuple(self.blocks)),),
            frame_roots=(SourceCompilationFrameRoot(self.scope,FrameKind.CLASS,self.evidence.to_reference(),self.interface),),
            state_address_refinements=tuple(self.addresses),
            attribute_lookups=tuple(self.attribute_lookups) if self.attribute_lookups else None)
        frontiers=tuple(f for f in _frontiers(self.source,has_freevars=bool(self.free),has_class_cell='__class__'in self.cell_refs,
            has_decorators=bool(self.node.decorator_list)) if f.kind is not ClassFrontierKind.BODY_REQUIRED_EXPANSION)
        if self.remaining is not None:
            from scar.ir.program.class_entries import ClassEntryFrontier
            frontiers+=(ClassEntryFrontier(ClassFrontierKind.BODY_REQUIRED_EXPANSION,self.remaining.reason,self.remaining.statements[0].source if self.remaining.statements else self.source),)
        gaps=tuple(LoweringGap(GapKind.EFFECT_SUMMARY,f.source or self.source,f.kind.value+': '+f.detail)for f in frontiers)
        metadata=LoweringMetadata(self.definition.wire,self.snapshot,tuple(self.op_links),tuple(self.value_links),
                                 gaps=gaps,use_links=tuple(self.use_links))
        overlay_type=ClassSuiteMethodOverlay if self.method_profile else ClassSuiteOverlay
        extra={'method_lookups':tuple(self.method_lookups),'method_calls':tuple(self.method_calls)}if self.method_profile else{}
        overlay=overlay_type(self.definition,self.semantic_definition.parent_id,self.source,self.evidence.to_reference(),
            self.scope,self.interface,_class_header_inputs(self.prepared,self.node),self.normal_mode,
            _class_scope_statements(self.prepared,self.node),tuple(self.witnesses),tuple(self.protocols),tuple(self.cells),
            tuple(self.constants),tuple(self.factories),self.remaining,frontiers,len(self.ops),**extra)
        return ClassEntryResult(graph,metadata,overlay,len(self.ops))

    def continuation(self,i,reason):
        selectors=tuple(dict.fromkeys(self.method_carriers[v].has_self for v in self.stack if v in self.method_carriers))
        operands=(*self.namespaces,*self.free.values(),*self.cell_refs.values(),*(v for v in self.stack if v is not None),*selectors)
        results,identity=self.invoke('REQUIRED_CLASS_CONTINUATION',operands,self.source_for(i),i.offset,
            consumed_on_raise=self.method_profile)
        statements=_class_scope_statements(self.prepared,self.node)
        position=i.positions.lineno or self.node.lineno
        remaining=tuple(replace(row,order=n)for n,row in enumerate(
            row for row in statements if row.source.end_line>=position))
        self.remaining=ClassSuiteResidual(identity,i.offset,remaining,tuple(self.stack),reason)
        self.transition('FRAME_EXIT_NORMAL',(*self.live(),results[0]),self.source,i.offset)
        self.exit((results[0],),ExitKind.NORMAL,self.source);self.returned=True


def lower_class_suite(prepared,definition,source,*,code_evidence,max_operations=10000,profile='STATIC_CLASS_SUITE_314_V1'):
    from scar.analysis.source_language import supports_cpython_compiler
    if not supports_cpython_compiler(code_evidence):raise ValueError('class suite requires exact CPython 3.14.7 source compilation')
    result=ClassSuiteBuilder(prepared,definition,source,code_evidence,max_operations,profile=profile).run()
    from scar.analysis.source_class_body import require_valid_class_suite
    require_valid_class_suite(prepared,definition,source,code_evidence=code_evidence,result=result)
    return result


# Independent compiler-stack and canonical-port checks for class suites.
def require_valid_class_suite(prepared,definition,source,*,code_evidence,result):
    def need(ok,why):
        if not ok:raise ValueError('class suite source: '+why)
    snapshot,semantic,node=_source_semantics(prepared,definition,source)
    independently_compiled=compile_source_unit(prepared)
    evidence=independently_compiled.for_definition(definition,source)
    need(evidence.to_reference()==code_evidence.to_reference(),'independent no-exec compilation differs')
    graph,metadata,overlay=result.graph,result.metadata,result.overlay
    require_valid(graph);validate_class_suite_core(overlay,graph,metadata)
    need(overlay.parent_definition==semantic.parent_id and overlay.compilation==evidence.to_reference(),'original lexical/compile context differs')
    need(overlay.inputs==_expected_header_inputs(prepared,node),'original class-header input order differs')
    need(overlay.statements==_expected_body_statements(prepared,node),'class source statements omitted/reordered')
    for rows in (metadata.operation_links,metadata.value_links,metadata.use_links):
        for link in rows:
            atom=prepared.semantic.source_atoms.get(link.source.atom_id)
            need(atom is not None and atom.reference==link.source,'invented/changed source atom')
    for link in metadata.operation_links:
        if link.semantic_operation is not None:
            original=prepared.semantic.definitions.get(link.semantic_operation)
            need(original is not None and original.code_id==link.source.atom_id.wire,'original semantic correspondence differs')
    operations={o.id:o for o in graph.operations};values={v.id:v for v in graph.values}
    blocks={b.id:b for r in graph.regions for b in r.blocks}
    protocols={p.operation:p for p in overlay.protocols};mapping=overlay.interface.mapping
    namespace=(mapping.locals_namespace,mapping.globals_namespace,mapping.builtins_namespace)
    free=dict(zip(overlay.interface.co_freevars,mapping.closure_cells));cells={}
    constants={c.operation:c for c in overlay.constants};constant_values={};code_values={};tuples={};functions={}
    factories={f.operation:f for f in overlay.factories};stack=[]
    method_profile=type(overlay)is ClassSuiteMethodOverlay
    method_lookups={x.operation:x for x in overlay.method_lookups}if method_profile else{}
    method_calls={x.operation:x for x in overlay.method_calls}if method_profile else{}
    method_carriers={}
    rows=tuple(dis.get_instructions(evidence.code,adaptive=False,show_caches=False))
    applications=set()
    from scar.analysis.source_class_validation import _source_reference
    for n in ast.walk(node):
        if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
        ref_n=_source_reference(prepared,n)
        definitions=[d for d in prepared.semantic.definitions.values()if d.code_id==ref_n.atom_id.wire and d.metadata.get('phase')=='call_time']
        for d in definitions:
            try:proof=independently_compiled.for_definition(d.id,ref_n)
            except ValueError:continue
            if proof.code_path[:-1]!=evidence.code_path:continue
            loads=[r for r in rows if r.opname=='LOAD_CONST'and r.arg==proof.code_path[-1]and type(r.argval)is CodeType]
            binding=n.name
            if binding.startswith('__')and not binding.endswith('__')and node.name.lstrip('_'):
                binding='_'+node.name.lstrip('_')+binding
            stores=[r for r in rows if r.opname=='STORE_NAME'and r.argval==binding and tuple(r.positions)==(n.lineno,n.end_lineno,n.col_offset,n.end_col_offset)
                    and loads and r.offset>loads[0].offset]
            if len(loads)!=1 or len(stores)!=1:continue
            calls=[r for r in rows if loads[0].offset<r.offset<stores[0].offset and r.opname in ('CALL','CALL_KW','CALL_FUNCTION_EX')]
            expected=[(x.lineno,x.end_lineno,x.col_offset,x.end_col_offset)for x in reversed(n.decorator_list)]
            if len(calls)==len(expected)and all(r.opname=='CALL'and r.arg==0 and tuple(r.positions)==s for r,s in zip(calls,expected)):
                applications.update(r.offset for r in calls)
    need(len(overlay.instructions)<=len(rows),'extra compiler witnesses')
    def normal(op):
        need(op.terminator is not None and op.terminator.kind is TerminatorKind.INVOKE,'missing original invoke')
        edge=next(e for e in op.terminator.successors if e.kind is ExitKind.NORMAL)
        return blocks[edge.block].arguments[:-1]
    def operands(op):return tuple(u.value for u in op.uses[:-1])
    def literal(v,expected):
        origin=values[v].origin
        need(type(origin)is LiteralOrigin and type(origin.value)is type(expected) and origin.value==expected,'literal operand differs')
    def consumed_cleanup(op):
        # Independently replay the frame's still-owned references after the
        # original compiler pops. Consumed lookup/call inputs must not be
        # retained until frame cleanup, unless another actual stack/frame
        # binding independently owns that reference.
        refs=(*namespace,*free.values(),*cells.values(),*(v for v in stack if v is not None))
        expected=tuple(dict.fromkeys(v for v in refs if type(values[v])is ReferenceValue))
        failure=next(e for e in op.terminator.successors if e.kind is ExitKind.RAISE)
        cleanup=operations[blocks[failure.block].operations[0]]
        need(tuple(u.value for u in cleanup.uses[:-3])==expected,
             'consumed lookup/call input retained past its primitive or frame references lost')
    for row,witness in zip(rows,overlay.instructions):
        need((row.offset,row.opname,row.arg,tuple(row.positions))==
             (witness.offset,witness.opcode,witness.argument,witness.positions),'compiler opcode/order/position changed')
        need(tuple(stack)==witness.stack_before,'original compiler stack input differs')
        op=operations.get(witness.operation);p=protocols.get(witness.operation)
        if op is not None:need(p is not None and p.offset==row.offset and p.source==witness.source,'compiler/core witness endpoint differs')
        name,arg=row.opname,row.argval
        kind={'LOAD_CONST':'READ_CODE_CONSTANT','LOAD_NAME':'LOAD_NAME_314','STORE_NAME':'STORE_NAME_314',
              'STORE_DEREF':'STORE_CELL','MAKE_FUNCTION':'MAKE_FUNCTION_314','LOAD_ATTR':'LOAD_ATTRIBUTE',
              'RETURN_VALUE':'FRAME_EXIT_NORMAL','POP_TOP':'FRAME_DISCARD'}.get(name,name)
        if name=='LOAD_ATTR'and row.arg&1:
            need(method_profile,'historical suite admitted method-mode LOAD_ATTR')
            kind='LOAD_METHOD_314'
        if name in ('CALL','CALL_KW'):
            receiver_index=len(stack)-row.arg-(1 if name=='CALL_KW'else 0)-1
            method=method_carriers.get(stack[receiver_index])if method_profile else None
            kind='CALL_METHOD_314'if method is not None else 'APPLY_DECORATOR_314'if row.offset in applications else 'CALL'
        if op is not None:need(p.kind==kind,'compiler/source protocol classification differs')
        if name in ('RESUME','COPY_FREE_VARS','NOP','EXTENDED_ARG'):
            need(op is None,'encoding-only instruction became a program operation')
        elif name=='PUSH_NULL':stack.append(None)
        elif name=='COPY':stack.append(stack[-row.arg])
        elif name=='SWAP':stack[-1],stack[-row.arg]=stack[-row.arg],stack[-1]
        elif name=='LOAD_FAST_BORROW':
            need(arg in cells or arg in free,'borrow is not an original compiler cell reference')
            stack.append(cells.get(arg,free.get(arg)))
        elif name=='LOAD_LOCALS':stack.append(namespace[0])
        elif name=='LOAD_SMALL_INT':
            need(len(witness.stack_after)==len(stack)+1,'small literal result missing')
            value=witness.stack_after[-1];literal(value,arg);stack.append(value)
        elif name=='LOAD_CONST':
            record=constants.get(op.id)
            need(record is not None and record.constant_path==(row.arg,) and op.results==(record.reference,),
                 'original constant table slot/result differs')
            if type(arg)is CodeType:code_values[record.reference]=(row.arg,arg)
            else:constant_values[record.reference]=arg
            stack.append(record.reference)
        elif name=='MAKE_CELL':
            declared=[c for c in overlay.cells if c.operation==op.id]
            need(len(declared)==1 and declared[0].name==arg and arg not in cells,'cell allocation source differs')
            cells[arg]=normal(op)[0]
        elif name in ('LOAD_NAME','LOAD_GLOBAL'):
            expected=namespace if name=='LOAD_NAME' else namespace[1:]
            actual=operands(op);need(actual[:-1]==expected,'class L/G/saved-B lookup order differs');literal(actual[-1],arg)
            stack.append(normal(op)[0])
            if name=='LOAD_GLOBAL' and row.arg&1:stack.append(None)
        elif name in ('STORE_NAME','STORE_GLOBAL','DELETE_NAME'):
            actual=operands(op);owner=namespace[1] if name=='STORE_GLOBAL' else namespace[0]
            need(actual[0]==owner,'class mapping write targets the wrong namespace');literal(actual[1],arg)
            if name!='DELETE_NAME':need(actual[2:]==(stack.pop(),),'mapping write changed its original stack value')
        elif name in ('LOAD_DEREF','LOAD_FROM_DICT_OR_DEREF'):
            cell=cells.get(arg,free.get(arg));actual=operands(op)
            if name=='LOAD_DEREF':need(actual==(cell,),'deref lost its original cell')
            else:
                owner=stack.pop();need(actual[0]==owner and actual[2:]==(cell,),'class-deref mapping/cell priority differs');literal(actual[1],arg)
            stack.append(normal(op)[0])
        elif name=='STORE_DEREF':
            value=stack.pop();need(operands(op)==(cells.get(arg,free.get(arg)),value),'cell write dependency differs')
        elif name in ('BUILD_TUPLE','BUILD_LIST','BUILD_SET','BUILD_MAP'):
            count=row.arg*(2 if name=='BUILD_MAP' else 1);items=tuple(stack.pop()for _ in range(count))[::-1]
            need(operands(op)==items,'container packing input order differs')
            value=normal(op)[0];stack.append(value)
            if name=='BUILD_TUPLE':tuples[value]=items
        elif name=='MAKE_FUNCTION':
            code=stack.pop();need(code in code_values,'factory code was not read from its original slot')
            index,child=code_values[code];factory=factories.get(op.id)
            need(factory is not None and factory.code_reference==code and factory.constant_path==(index,)
                 and factory.free_names==tuple(child.co_freevars),'factory code/closure descriptor differs')
            need(operands(op)==(code,namespace[1]),'class local mapping was used as method globals')
            if factory.definition is not None:
                proof=independently_compiled.for_definition(factory.definition,factory.source)
                need(proof.code_path==evidence.code_path+(index,),'method body is not the original child definition')
            functions[factory.reference]=child;stack.append(normal(op)[0])
        elif name=='SET_FUNCTION_ATTRIBUTE':
            function=stack.pop();value=stack.pop();need(operands(op)==(function,value) and function in functions,'function attribute lost fresh factory identity')
            if row.arg==8:
                expected=tuple(cells.get(n,free.get(n))for n in functions[function].co_freevars)
                need(None not in expected and tuples.get(value)==expected,'closure field captures wrong/order cell references')
            stack.append(function)
        elif name=='LOAD_ATTR':
            owner=stack.pop();actual=operands(op)
            need(len(actual)==2 and actual[0]==owner,'attribute owner differs');literal(actual[1],arg)
            if row.arg&1:
                lookup=method_lookups.get(op.id);returned=normal(op)
                need(lookup is not None and returned==(lookup.callable,lookup.optional_self,lookup.has_self),
                     'method lookup lost its callable/self/boolean completion interface')
                consumed_cleanup(op)
                method_carriers[lookup.optional_self]=lookup
                stack.extend(returned[:2])
            else:stack.append(normal(op)[0])
        elif name in ('CALL','CALL_KW'):
            keyword=stack.pop()if name=='CALL_KW'else None
            args=tuple(stack.pop()for _ in range(row.arg))[::-1];receiver=stack.pop();callee=stack.pop()
            method=method_carriers.get(receiver)if method_profile else None
            if method is not None:
                record=method_calls.get(op.id)
                need(record is not None and record.lookup==method.operation and callee==method.callable,
                     'method CALL lost its original lookup/callable association')
                expected=(callee,receiver,method.has_self,*args,*((keyword,)if keyword is not None else ()))
            else:
                if receiver is not None:args=(receiver,*args)
                expected=(callee,*args,*((keyword,)if keyword is not None else ()))
            need(operands(op)==expected,'callable/receiver/arguments/selector were reordered')
            if method_profile:consumed_cleanup(op)
            link=next(x for x in metadata.operation_links if x.operation==op.id)
            from scar.ir.semantics_v2 import Opcode
            if row.offset in applications:
                need(link.opcode is None and link.semantic_operation is None,
                     'decorator application cannot masquerade as the original AST CALL')
            if link.opcode is Opcode.CALL:
                matches=[n for n in ast.walk(node)if isinstance(n,ast.Call)and
                    (n.lineno,n.end_lineno,n.col_offset,n.end_col_offset)==tuple(row.positions)]
                need(len(matches)==1,'source CALL lost its original AST occurrence')
                call=matches[0];sources=[x for x in metadata.use_links if x.operation==op.id]
                callee_sources=[x for x in sources if x.use==op.uses[0].id]
                need(len(callee_sources)==1 and callee_sources[0].source==_source_reference(prepared,call.func),
                     'source CALL callee Use has wrong original source')
            stack.append(normal(op)[0])
        elif name in ('BINARY_OP','UNARY_NEGATIVE','UNARY_INVERT','UNARY_NOT'):
            args=tuple(stack.pop()for _ in range(2 if name=='BINARY_OP'else 1))[::-1]
            need(operands(op)==args,'value-operation inputs differ');stack.append(normal(op)[0])
        elif name=='POP_TOP':
            value=stack.pop();need(op.uses[0].value==value,'discard changed its reference')
        elif name=='RETURN_VALUE':
            value=stack.pop();need(not stack and op.uses[-2].value==value,'normal completion must preserve its actual payload')
            if overlay.normal_payload is ClassNormalPayloadKind.CLASS_CELL:
                need(value==cells['__class__'],'class suite returned a value instead of the compiler class cell')
            else:need(value in constant_values and constant_values[value]is None,'class suite None return lost its original code-slot None')
        else:raise ValueError('class suite source: unadmitted compiler witness '+name)
        need(tuple(stack)==witness.stack_after,'original compiler result stack differs')
    if overlay.remaining is None:
        need(len(overlay.instructions)==len(rows) and rows[-1].opname=='RETURN_VALUE','suite silently dropped source/compiler suffix')
    else:
        need(len(overlay.instructions)<len(rows) and rows[len(overlay.instructions)].offset==overlay.remaining.first_offset,
             'remaining compiler continuation starts at wrong offset')
        need(tuple(stack)==overlay.remaining.stack_inputs,'remaining compiler continuation lost partial-evaluation inputs')
        selectors=tuple(dict.fromkeys(method_carriers[v].has_self for v in stack if v in method_carriers))
        expected=(*namespace,*free.values(),*cells.values(),*(v for v in stack if v is not None),*selectors)
        need(operands(operations[overlay.remaining.operation])==expected,
             'remaining continuation lost its exact frame/stack/method-selector inputs')
    need(set(cells)==set(overlay.interface.co_cellvars),'compiler-owned cells not allocated before suite completion')
    if graph.attribute_lookups is not None:
        from scar.analysis.source_attribute_lookup import validate_source_attribute_lookups
        validate_source_attribute_lookups(graph, metadata, source_text=prepared.text,
            semantic=prepared.semantic, compiler_overlay=overlay, compiler_evidence=evidence)
    return {'valid':True,'source_compilation_replayed':True,'runtime_target_executed':False,
            'suite_complete':overlay.remaining is None,'effects_complete':False}

"""Source lambda."""

from __future__ import annotations

import ast
import dis
from dataclasses import replace
from types import CodeType
from scar.analysis.program_lowering import _Unsupported, _REF, _key
from scar.analysis.environment_lowering import captured_cell_slot, _PREFIX
from scar.analysis.static.lowering.source_generator_lowering import _Nested
from scar.analysis.static.lowering.source_function_definitions import _code_reference, _hold
from scar.analysis.static.lowering.source_heap_lowering import (
    _source_definition,
    _associate,
    _hold as hold_expression,
)
from scar.analysis.static.lowering.source_slice import _invoke
from scar.analysis.static.lowering.source_mapping_lowering import _transfer
from scar.analysis.structured_lowering import StructuredBuilder
from scar.analysis.function_interfaces import formal_nodes
from scar.ir.program.environments import (
    EnvironmentInterface,
    EnvironmentMember,
    EnvironmentRole,
    FunctionEntryLayout,
)
from scar.ir.program.functions import BodyEntryInterface, BodyLoweringProfile, FormalPort
from scar.ir.program.ids import EnvironmentInterfaceID, OperationID, RegionID, ValueID
from scar.ir.program.function_sites import (
    LambdaBodyDefinition,
    LambdaCapture,
    LambdaEnvironmentField,
)
from scar.ir.program.model import (
    Behavior,
    Block,
    BlockArgument,
    ExitKind,
    ExitPort,
    LiteralOrigin,
    Operation,
    ReferenceValue,
    Region,
    Terminator,
    TerminatorKind,
    CompletionResult,
    OperationResult,
    StateVersionValue,
    value_type,
)
from scar.ir.program.source import GapKind, SourceRole
from scar.ir.program.structured import PythonProtocolKind as K, StructuredBodyModel
from scar.ir.semantics_v2 import Opcode
from scar.ir.v2.semantic import OperationKind, SemanticRelation
from scar.analysis.source_language import function_exception_types, require_source_tree
from scar.ir.program.source_language import SourceLanguageCapability as C


# Lambda creation plus its original, deferred expression-body Region.
#
# One creation interface references original child code/defaults/cells. A
# separate ordinary Region evaluates the original Lambda.body and returns its
# reference/world; creation never CALLs that Region. No synthesized source AST,
# captured CodeID, runtime callable, frozen cell contents or execution is added.
VERSION = 'PYTHON_LAMBDA_EXPRESSION_314_V1'


def span(node):
    return node.lineno, node.end_lineno, node.col_offset, node.end_col_offset


def compiler_lambda(node, code):
    rows = tuple(dis.get_instructions(code))
    loads = [r for r in rows if r.opname == 'LOAD_CONST' and type(r.argval) is CodeType
             and r.argval.co_name == '<lambda>' and tuple(r.positions) == span(node)]
    if not loads or len({r.arg for r in loads}) != 1:
        raise _Unsupported(GapKind.CODE_VERSION,node,'LAMBDA_ORIGINAL_CONSTANT_SLOT_AMBIGUOUS')
    index = loads[0].arg; child = code.co_consts[index]
    names = tuple(a.arg for a, _ in formal_nodes(node))
    parameters = (*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs)
    if (any(r.argval is not child for r in loads) or child.co_firstlineno != node.lineno
            or child.co_qualname != code.co_qualname + '.<locals>.<lambda>'
            or child.co_argcount != len(node.args.posonlyargs)+len(node.args.args)
            or child.co_posonlyargcount != len(node.args.posonlyargs)
            or child.co_kwonlyargcount != len(node.args.kwonlyargs)
            or bool(child.co_flags & 4) != (node.args.vararg is not None)
            or bool(child.co_flags & 8) != (node.args.kwarg is not None)
            or set(child.co_varnames[:len(names)]) != set(names)):
        raise _Unsupported(GapKind.CODE_VERSION,node,'LAMBDA_ORIGINAL_PARAMETER_CODE_INTERFACE_DIFFERS')
    for load in loads:
        at=rows.index(load)
        if at+1>=len(rows) or rows[at+1].opname!='MAKE_FUNCTION':
            raise _Unsupported(GapKind.CODE_VERSION,node,'LAMBDA_ORIGINAL_FACTORY_INTERFACE_UNRECOVERED')
        if child.co_freevars:
            prefix=rows[at-len(child.co_freevars)-1:at]
            if (len(prefix)!=len(child.co_freevars)+1 or prefix[-1].opname!='BUILD_TUPLE'
                    or prefix[-1].arg!=len(child.co_freevars)
                    or tuple(r.argval for r in prefix[:-1])!=child.co_freevars
                    or any(r.opname not in ('LOAD_FAST','LOAD_FAST_BORROW')for r in prefix[:-1])
                    or rows[at+2].opname!='SET_FUNCTION_ATTRIBUTE' or rows[at+2].arg!=8):
                raise _Unsupported(GapKind.CODE_VERSION,node,'LAMBDA_ORIGINAL_CELL_CAPTURE_INTERFACE_DIFFERS')
    return loads[0], child


class _ExpressionBody(_Nested):
    """Original expression adapter with ordinary function ABI, not a driver."""
    def fresh(self, cls, tag):
        return self.owner.fresh(cls,'lambda-'+tag)

    def emit(self, cursor, operation, node, role, old=None):
        # Bypass a parent's active Try scope: the deferred child frame owns
        # these blocks, and cannot accidentally enter a creating frame's except.
        # Buffers and the construction budget belong to the creating builder;
        # the child owns only its Region. Bypass Try.emit without copying the
        # integer budget counter into the child.
        StructuredBuilder.emit(self.owner,cursor,operation,node,role,old=old)

    def expr(self,node,cursor):
        definition=_source_definition(self,node)
        if isinstance(node,ast.Name):
            cursor,value=self.read_source_name(node,cursor,None)
            if cursor is not None:
                from scar.analysis.static.lowering.source_heap_lowering import _normal_producer
                record=self.values_by_id[value]
                if isinstance(record.origin,BlockArgument):
                    producers=[o.id for o in self.operations if o.terminator is not None
                        and o.terminator.kind is TerminatorKind.INVOKE
                        and any(e.kind is ExitKind.NORMAL and e.block==record.origin.block for e in o.terminator.successors)]
                    identity,=producers
                else:identity=record.origin.operation
                _associate(self,identity,node,definition,Opcode.READ)
            return cursor,value
        if isinstance(node,ast.Constant):
            try:origin=LiteralOrigin(node.value)
            except (TypeError,ValueError) as error:
                raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX,node,'LAMBDA_LITERAL_INTERFACE_UNIMPLEMENTED: '+str(error))
            value=ReferenceValue(self.fresh(ValueID,'literal'),self.function_context,origin,targets=())
            self.add_value(value);self.link_value(value,node,SourceRole.EXPRESSION)
            return cursor,value.id
        if isinstance(node,ast.Dict):
            if len(node.keys)>15 or any(k is None for k in node.keys):
                raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX,node,'LAMBDA_DICT_EXPANSION_OR_INCREMENTAL_INTERFACE_UNIMPLEMENTED')
            from scar.analysis.static.lowering.source_mapping_lowering import lower_dictionary
            return lower_dictionary(self,node,cursor)
        if isinstance(node,ast.Call) and not any(isinstance(a,ast.Starred)for a in node.args) and all(k.arg is not None for k in node.keywords):
            children=[(node.func,'callee',0),*((n,'argument',i)for i,n in enumerate(node.args)),
                      *((k.value,'keyword:'+k.arg,i)for i,k in enumerate(node.keywords))]
            opcode=Opcode.CALL
        elif isinstance(node,ast.Attribute) and not (node.attr.startswith('__')and not node.attr.endswith('__')):
            children=[(node.value,'receiver',0)];opcode=Opcode.ATTRIBUTE
        elif isinstance(node,ast.Subscript) and not isinstance(node.slice,(ast.Slice,ast.Tuple)):
            children=[(node.value,'receiver',0),(node.slice,'index',1)];opcode=Opcode.INDEX
        else:
            raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX,node,'LAMBDA_EXPRESSION_INTERFACE_UNIMPLEMENTED: '+type(node).__name__)
        baseline=len(self.active_temporaries);held=[];inputs=[]
        try:
            for child,role,index in children:
                cursor,value=self.expr(child,cursor)
                if cursor is None:return None,None
                hold_expression(self,cursor,child,value);held.append((child,value));inputs.append((value,child,role,index,None))
            cursor,result,identity=_invoke(self,cursor,node,inputs,tuple(held))
            _associate(self,identity,node,definition,opcode)
            if opcode is Opcode.ATTRIBUTE:
                from scar.analysis.source_attribute_lookup import register_attribute_lookup
                register_attribute_lookup(self,node,identity)
            self.link_value(self.values_by_id[result],node,SourceRole.EXPRESSION)
            if opcode is Opcode.CALL:
                self.pending_targets.append((identity,inputs[0][0],None,node))
            if opcode is Opcode.INDEX:
                from scar.ir.program.state_addresses import AddressEffect,Element,ReferenceOwner
                from scar.ir.program.model import AccessMode
                uses=self.operation_records[identity].uses
                self._addresses(identity,(AddressEffect(ReferenceOwner(uses[0].id),(Element(uses[1].id),),AccessMode.READ),))
            return cursor,result
        finally:del self.active_temporaries[baseline:]

    def terminal(self,cursor,results,kind=ExitKind.NORMAL):
        identity=self.fresh(OperationID,'return'if kind is ExitKind.NORMAL else'raise')
        uses=tuple(self.use(identity,value,self.node.body,'result',index)for index,value in enumerate((*results,cursor.state)))
        self.emit(cursor,Operation(identity,kind.value,(Behavior.CONTROL,),uses,
            terminator=Terminator(TerminatorKind.RETURN if kind is ExitKind.NORMAL else TerminatorKind.RAISE,
                exit=kind.value.lower(),values=tuple(u.id for u in uses))),self.node.body,SourceRole.RETURN)
        return identity


def _expression_body(owner,node,definition,code,index):
    if code.co_cellvars:
        raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX,node,'LAMBDA_EXPRESSION_OWN_CELL_ALLOCATION_UNIMPLEMENTED')
    region=RegionID(_key('lambda-expression-body',definition.id.wire,owner.language_options.code_evidence.code_layout_key,index,VERSION))
    b=_ExpressionBody(owner,node,region,definition);b.language_options=None
    parameters=tuple(a for a,_ in formal_nodes(node));b.locals={a.arg for a in parameters}
    b.free_cells=frozenset(code.co_freevars);b.global_names=set(code.co_names)
    cursor,refs=b.ports((_REF,)*(len(_PREFIX)+len(code.co_freevars)+len(parameters)))
    env_refs=refs[:len(_PREFIX)+len(code.co_freevars)];actual=refs[len(env_refs):]
    members=[EnvironmentMember(role,value,(b.world,))for role,value in zip(_PREFIX,env_refs)]
    for position,name in enumerate(code.co_freevars):
        members.append(EnvironmentMember(EnvironmentRole.CAPTURED_CELL,env_refs[len(_PREFIX)+position],(b.world,),
            captured_cell_slot(definition.id,name),position))
    b.environment_references=dict(zip(_PREFIX,env_refs));b.cell_references=dict(zip(code.co_freevars,env_refs[len(_PREFIX):]))
    ports=tuple(FormalPort(b.slots[definition.id,arg.arg],value)for arg,value in zip(parameters,actual))
    for parameter,port in zip(parameters,ports):
        b.link_value(b.values_by_id[port.value],parameter,SourceRole.PARAMETER,slot=port.slot)
    interface=BodyEntryInterface(definition.id,region,ports,cursor.state,BodyLoweringProfile.STRUCTURED_ENV)
    b.interface=interface
    environment=EnvironmentInterface(EnvironmentInterfaceID(_key('lambda-environment',region.wire)),region,definition.id,tuple(members))
    entry=cursor.block
    b.frame_transition(cursor,node,tuple((v,node,'environment-reference')for v in env_refs)
        +tuple((v,a,'formal-reference')for a,v in zip(parameters,actual)),K.FRAME_ENTER)
    for parameter,value in zip(parameters,actual):b.define_binding(parameter.arg,cursor,parameter,value)
    cursor,result=b.expr(node.body,cursor)
    if cursor is not None:b.finish_exit(cursor,node.body,result,ExitKind.NORMAL)
    blocks=tuple(Block(key,b.block_arguments[key],tuple(b.block_operations[key]))for key in b.function_blocks)
    # Buffers are shared with the creating frame and sibling bodies. Only
    # this child's actual terminal sites can declare its exit ports: a literal
    # or bound-local expression need not have a RAISE site at all.
    completed={operation.terminator.kind for block in blocks for identity in block.operations
        if (operation:=b.operation_records[identity]).terminator is not None}
    exits=tuple(ExitPort(kind.value.lower(),kind,(_REF,b.state_type))for kind,terminal in
        ((ExitKind.NORMAL,TerminatorKind.RETURN),(ExitKind.RAISE,TerminatorKind.RAISE))if terminal in completed)
    b.regions.append(Region(region,'lambda expression',entry,blocks,exits,parent=owner.function_context))
    owner.lambda_environments.append(environment);owner.lambda_entries.append(FunctionEntryLayout(environment.id,ports))
    return interface,environment


def lower_lambda(builder,node,cursor):
    evidence=builder.language_options.code_evidence
    if evidence is None:
        raise _Unsupported(GapKind.CODE_VERSION,node,'LAMBDA_READONLY_PARENT_CODE_REQUIRED')
    load,code=compiler_lambda(node,evidence.code)
    declaration=_source_definition(builder,node)
    bodies=[d for d in builder.by_atom.get(builder.source(node).atom_id.wire,())
            if d.metadata.get('phase')=='call_time'and d.kind in(OperationKind.FUNCTION,OperationKind.METHOD)]
    if declaration.metadata.get('phase')!='definition_time' or len(bodies)!=1:
        raise _Unsupported(GapKind.SOURCE_MAPPING,node,'LAMBDA_ORIGINAL_DECLARATION_OR_BODY_MISSING')
    if any(n not in builder.cell_references for n in code.co_freevars):
        raise _Unsupported(GapKind.SOURCE_MAPPING,node,'LAMBDA_CAPTURE_HAS_NO_ACTUAL_PARENT_CELL')
    if not hasattr(builder,'lambda_bodies'):
        builder.lambda_bodies=[];builder.lambda_environments=[];builder.lambda_entries=[]
    body,environment=_expression_body(builder,node,bodies[0],code,load.arg)
    baseline=len(builder.active_temporaries);defaults=[];inputs=[]
    try:
        from scar.analysis.static.lowering.source_heap_lowering import expression_for_heap_target
        for index,source in enumerate(node.args.defaults):
            cursor,value=expression_for_heap_target(builder,source,cursor)
            if cursor is None:return None,None
            _hold(builder,cursor,source,value,'default');defaults.append((source,value))
            inputs.append((value,source,'lambda-default',index,None))
        for index,(argument,source)in enumerate(zip(node.args.kwonlyargs,node.args.kw_defaults)):
            if source is None:continue
            cursor,value=expression_for_heap_target(builder,source,cursor)
            if cursor is None:return None,None
            _hold(builder,cursor,source,value,'keyword-default');defaults.append((source,value))
            inputs.append((value,source,'lambda-keyword-default:'+argument.arg,index,None))
        code_ref=_code_reference(builder,node,cursor,load,'lambda code')
        read=builder.values_by_id[code_ref].origin.operation
        prefix=[(code_ref,node,'lambda-code',0,None),
                (builder.environment_references[EnvironmentRole.GLOBALS_NAMESPACE],node,'lambda-globals',0,None)]
        inputs=prefix+inputs+[(builder.cell_references[name],node,'lambda-closure-cell',i,None)for i,name in enumerate(code.co_freevars)]
        def failed(failure,error):
            pending=tuple(builder.active_temporaries)
            try:
                _transfer(builder,tuple(defaults),node);builder.finish_exit(failure,node,error,ExitKind.RAISE)
            finally:builder.active_temporaries[:]=pending
        cursor,results,identity=builder.invoke_results(cursor,node,tuple(inputs),result_types=(_REF,),failure_handler=failed)
        builder.protocol(identity,K.MAKE_LAMBDA_FUNCTION,node)
        _associate(builder,identity,node,declaration,None);_transfer(builder,tuple(defaults),node)
        uses=builder.operation_records[identity].uses
        original=builder.operation_records[identity]
        marked=replace(original,behaviors=original.behaviors+(Behavior.MEMORY,))
        builder.operations[builder.operations.index(original)]=marked;builder.operation_records[identity]=marked
        function=results[0]
        # A constructor's initialized fields are a conditional state summary,
        # not extra runtime returns and not analyzer-invented target reads.
        captures=(LambdaCapture(environment.members[0].reference,normal_result=0),
                  LambdaCapture(environment.members[1].reference,source_use=uses[1].id))
        captures+=tuple(LambdaCapture(member.reference,created_field=field)
                       for member,field in zip(environment.members[2:5],LambdaEnvironmentField))
        closure=[u for u in uses if any(s.operation==identity and s.use==u.id and s.role=='lambda-closure-cell'for s in builder.use_links)]
        captures+=tuple(LambdaCapture(member.reference,source_use=use.id)for member,use in zip(environment.members[5:],closure))
        builder.lambda_bodies.append(LambdaBodyDefinition(identity,builder.source(node),read,body,environment.id,captures))
        builder.link_value(builder.values_by_id[function],node,SourceRole.EXPRESSION)
        builder.gap(GapKind.CODE_VERSION,node,'Lambda creation and expression body use the original child code/source; runtime callable/code identity and execution remain unproved')
        builder.gap(GapKind.EFFECT_SUMMARY,node,'Lambda defaults are evaluated only at creation; cells remain references read at invocation, and dynamic dispatch/finalizers/contents remain incomplete')
        return cursor,function
    finally:del builder.active_temporaries[baseline:]


def assemble_lambda_bodies(builder,graph):
    return replace(graph,environments=(*graph.environments,*builder.lambda_environments),
        function_entries=(*graph.function_entries,*builder.lambda_entries))


# Closed Lambda creation, captures and original expression-body validation.
def need(condition,detail):
    if not condition:raise ValueError('Lambda expression: '+detail)


def refspan(ref):return ref.start_line,ref.end_line,ref.start_column,ref.end_column


def region_projection(model,graph,metadata,region,*,strip_parent=False):
    """Ephemeral verifier view, never a replacement artifact or source fact."""
    ids={i for b in region.blocks for i in b.operations};values={v.id for v in graph.values if v.region==region.id}
    environment=[e for e in graph.environments if e.region==region.id]
    env_ids={e.id for e in environment}
    projected=replace(graph,root=region.id,regions=(replace(region,parent=None)if strip_parent else region,),
        operations=tuple(o for o in graph.operations if o.id in ids),values=tuple(v for v in graph.values if v.id in values),
        environments=tuple(environment),function_entries=tuple(e for e in graph.function_entries if e.environment in env_ids),
        binding_definitions=tuple(r for r in graph.binding_definitions if r.value in values),
        binding_reads=tuple(r for r in graph.binding_reads if r.operation in ids),
        state_address_refinements=tuple(r for r in graph.state_address_refinements if r.site.operation in ids),
        control_scopes=(),control_ports=(),exception_packets=(),pending_completions=(),
        handled_context_domains=(),cleanup_interfaces=(),cleanup_enters=(),cleanup_resumes=(),cleanup_replacements=(),handled_context_reads=(),
        resumable_bodies=(),resumable_creates=(),frame_context_initials=(),frame_context_attaches=(),frame_context_detaches=(),
        driver_interfaces=(),raw_driver_requests=(),final_body_gates=(),body_drives=(),driver_lifecycles=(),body_outcome_commits=())
    projected_model=replace(model,region=region.id,protocols=tuple(p for p in model.protocols if p.operation in ids),
        loops=tuple(l for l in model.loops if l.region==region.id))
    projected_metadata=replace(metadata,operation_links=tuple(r for r in metadata.operation_links if r.operation in ids),
        value_links=tuple(r for r in metadata.value_links if r.value in values),
        use_links=tuple(r for r in metadata.use_links if r.operation in ids),call_targets=tuple(r for r in metadata.call_targets if r.operation in ids))
    return projected_model,projected,projected_metadata


def _check_expression(node,result,model,graph,metadata,semantic,interface):
    """Actual expression result Uses, not a source-body label or old CFG claim."""
    operations={r.id:r for r in graph.operations};values={r.id:r for r in graph.values}
    protocols={r.operation:r for r in model.protocols};links={r.operation:r for r in metadata.operation_links}
    sources={(r.operation,r.use):r for r in metadata.use_links};by_value={}
    addresses={r.site.operation:r for r in graph.state_address_refinements}
    need(len(addresses)==len(graph.state_address_refinements),'duplicate child state address refinement')
    for op in operations.values():
        if op.terminator is not None and op.terminator.kind is TerminatorKind.INVOKE:
            for edge in op.terminator.successors:
                if edge.kind is ExitKind.NORMAL:by_value[edge.block]=op
    verified=set();index_addresses=set()
    def expression(n,value):
        original=values[value]
        need(type(original)is ReferenceValue,'expression result is not a Python reference')
        if isinstance(n,ast.Constant):
            need(type(original.origin)is LiteralOrigin and type(original.origin.value)is type(n.value)
                 and original.origin.value==n.value,'literal changed its exact original value/type')
            return
        if type(original.origin)is OperationResult:op=operations[original.origin.operation]
        else:
            need(type(original.origin)is BlockArgument and original.origin.index==0,'expression lost its exact NORMAL result')
            op=by_value[original.origin.block]
        link=links[op.id];verified.add(op.id)
        need(refspan(link.source)==span(n),'expression producer lost its exact original AST occurrence')
        if isinstance(n,ast.Name):
            need(link.opcode is Opcode.READ,'Name is not its original read')
            return  # Existing binding/environment validation owns exact lexical/cell/global addressing.
        if isinstance(n,ast.Dict):
            need(protocols[op.id].kind is K.BUILD_MAP and len(n.keys)<=15 and all(k is not None for k in n.keys),
                 'Dict expression lost its original ordinary mapping construction')
            children=[(v,role,i)for i,(key,val)in enumerate(zip(n.keys,n.values))for v,role in((key,'mapping-key'),(val,'mapping-value'))]
        elif isinstance(n,ast.Call):
            need(link.opcode is Opcode.CALL,'Call lost its exact source call interface')
            children=[(n.func,'callee',0),*((a,'argument',i)for i,a in enumerate(n.args)),
                      *((k.value,'keyword:'+k.arg,i)for i,k in enumerate(n.keywords))]
        elif isinstance(n,ast.Attribute):
            need(link.opcode is Opcode.ATTRIBUTE,'Attribute lost its source interface');children=[(n.value,'receiver',0)]
        elif isinstance(n,ast.Subscript):
            need(link.opcode is Opcode.INDEX and isinstance(n.ctx,ast.Load),'Subscript lost its source read interface')
            children=[(n.value,'receiver',0),(n.slice,'index',1)]
            from scar.ir.program.state_addresses import AddressCoverage,AddressEffect,Element,InvokeStateOperand,ReferenceOwner
            from scar.ir.program.state_addresses import AccessMode
            definition=semantic.definitions.get(link.semantic_operation)
            atom=semantic.source_atoms.get(link.source.atom_id)
            need(definition is not None and definition.kind is OperationKind.INDEX
                 and definition.code_id==link.source.atom_id.wire and link.source.atom_id in definition.source_atoms
                 and atom is not None and atom.kind=='Subscript' and atom.reference==link.source,
                 'Subscript address lost its original SG/source ownership')
            need(op.terminator is not None and op.terminator.kind is TerminatorKind.INVOKE
                 and op.callee is None and not op.effects_complete and len(op.uses)==3
                 and op.invoke_arguments==tuple(u.id for u in op.uses)
                 and type(values[op.uses[-1].value])is StateVersionValue
                 and values[op.uses[-1].value].location==values[interface.world].location,
                 'Subscript address lost its exact receiver/key/world invoke')
            row=addresses.get(op.id)
            expected=AddressEffect(ReferenceOwner(op.uses[0].id),(Element(op.uses[1].id),),AccessMode.READ)
            need(row is not None and row.site==InvokeStateOperand(op.id,op.uses[-1].id)
                 and row.coverage is AddressCoverage.PARTIAL and row.protocol is None and row.effects==(expected,),
                 'Subscript partial address differs from its original receiver/key/world Uses')
            index_addresses.add(op.id)
        else:raise ValueError('Lambda expression: unsupported original expression entered the body')
        operands=op.uses[:-1]
        need(len(operands)==len(children),'expression operand arity differs from its original AST')
        for use,(child,role,index)in zip(operands,children):
            source=sources.get((op.id,use.id))
            need(source is not None and source.role==role and source.index==index and refspan(source.source)==span(child),
                 'expression original operand Use/order/source differs')
            expression(child,use.value)
    expression(node,result)
    bindings={values[r.value].origin.operation for r in graph.binding_definitions}
    for op in graph.operations:
        if op.id in verified or op.id in bindings:continue
        marker=protocols.get(op.id)
        need(op.terminator is not None and op.terminator.kind in(TerminatorKind.RETURN,TerminatorKind.RAISE)
             or marker is not None and marker.kind in(K.FRAME_ENTER,K.FRAME_EXIT,K.FRAME_HOLD_TEMPORARY,K.FRAME_RELEASE_TEMPORARY),
             'unclaimed executable operation in Lambda expression body')
    return index_addresses


def validate_lambda_regions(model,graph,interface,metadata,semantic,control,language,*,
                            source_tree=None,options=None,structural_only=False,sealed_source=None):
    records={r.creation:r for r in language.lambda_bodies}
    makes={r.operation for r in model.protocols if r.kind is K.MAKE_LAMBDA_FUNCTION}
    need(len(records)==len(language.lambda_bodies) and set(records)==makes,'Lambda creation/body coverage differs')
    if not records:return set()
    need(C.LAMBDA_EXPRESSION_314_V1 in language.capabilities and isinstance(source_tree,ast.Module),
         'explicit expression-body capability and exact original AST required')
    operations={r.id:r for r in graph.operations};values={r.id:r for r in graph.values}
    regions={r.id:r for r in graph.regions};blocks={b.id:b for r in graph.regions for b in r.blocks}
    owners={i:r.id for r in graph.regions for b in r.blocks for i in b.operations}
    protocols={r.operation:r for r in model.protocols};links={r.operation:r for r in metadata.operation_links}
    sources={(r.operation,r.use):r for r in metadata.use_links};constants={r.operation:r for r in language.constants}
    environments={e.id:e for e in graph.environments};layouts={e.environment:e for e in graph.function_entries}
    root_env=next(e for e in graph.environments if e.region==graph.root)
    parent_cells={m.reference for m in root_env.members if m.role is EnvironmentRole.CAPTURED_CELL}
    own_cells=[blocks[o.terminator.successors[0].block].arguments[0]for p in model.protocols
        if p.kind is K.MAKE_CELL and owners[p.operation]==graph.root and(o:=operations[p.operation])is not None]
    parent_cells.update(own_cells);world=value_type(values[interface.world]);delegated=set();body_regions=set()
    for identity,record in records.items():
        need(owners[identity]==graph.root,'only the original creating frame owns the creation interface')
        nodes=[n for n in require_source_tree(source_tree,language.source,record.source)if isinstance(n,ast.Lambda)]
        need(len(nodes)==1,'original Lambda AST/source is missing or ambiguous');node,=nodes
        bodies=[d for d in semantic.definitions.values()if d.code_id==record.source.atom_id.wire
                and record.source.atom_id in d.source_atoms and d.metadata.get('phase')=='call_time'
                and d.kind in(OperationKind.FUNCTION,OperationKind.METHOD)]
        declarations=[d for d in semantic.definitions.values()if d.code_id==record.source.atom_id.wire
                and record.source.atom_id in d.source_atoms and d.metadata.get('phase')=='definition_time']
        need(len(bodies)==len(declarations)==1 and record.body.definition==bodies[0].id
             and links[identity].semantic_operation==declarations[0].id and links[identity].opcode is None,
             'creation/body must reference original SG phases/IDs, never a fabricated callable definition')
        need(len(bodies[0].output_slots)==1 and semantic.slots[bodies[0].output_slots[0]].name=='$return'
             and semantic.slots[bodies[0].output_slots[0]].owner==bodies[0].id,
             'Lambda body lost its original semantic return slot')
        region=regions[record.body.region];body_regions.add(region.id)
        environment=environments[record.environment];layout=layouts[record.environment]
        need(region.id!=graph.root and region.parent==graph.root and environment.region==region.id
             and environment.code_definition==record.body.definition and layout.parameters==record.body.parameters,
             'deferred body/entry/environment ownership differs')
        op=operations[identity]
        need(op.callee is None and not op.effects_complete and op.terminator is not None
             and op.terminator.kind is TerminatorKind.INVOKE
             and op.invoke_arguments==tuple(u.id for u in op.uses)
             and all(type(values[u.value])is ReferenceValue for u in op.uses[:-1])
             and tuple((c.kind,c.types)for c in op.completions)==((ExitKind.NORMAL,(_REF,world)),
                 (ExitKind.RAISE,function_exception_types(language,_REF,world))),
             'creation has only its function payload/world; it cannot execute child or assert purity')
        read=operations[record.code_constant];constant=constants[read.id]
        need(protocols[read.id].kind is K.READ_CODE_CONSTANT and constant.source==record.source
             and len(constant.constant_path)==1 and len(read.results)==1 and read.terminator is None
             and read.behaviors==(Behavior.READ,) and not read.effects_complete,
             'child code must remain an original nonexecuting code-table reference')
        args=op.uses[:-1];roles=[sources[identity,u.id]for u in args]
        need(len(args)>=2 and [(r.role,r.index)for r in roles[:2]]==[('lambda-code',0),('lambda-globals',0)]
             and args[0].value==read.results[0] and args[1].value==root_env.members[1].reference,
             'Lambda factory changed original code/globals inputs')
        cursor=2
        for i,default in enumerate(node.args.defaults):
            need(cursor<len(args)and(roles[cursor].role,roles[cursor].index)==('lambda-default',i)
                 and refspan(roles[cursor].source)==span(default),'positional default evaluation/source order differs');cursor+=1
        for i,(parameter,default)in enumerate(zip(node.args.kwonlyargs,node.args.kw_defaults)):
            if default is None:continue
            need(cursor<len(args)and(roles[cursor].role,roles[cursor].index)==('lambda-keyword-default:'+parameter.arg,i)
                 and refspan(roles[cursor].source)==span(default),'keyword default evaluation/source order differs');cursor+=1
        closure=args[cursor:]
        need(len(closure)==len(environment.members)-5 and all(u.value in parent_cells for u in closure)
             and [(r.role,r.index)for r in roles[cursor:]]==[('lambda-closure-cell',i)for i in range(len(closure))],
             'capture substituted contents/foreign cells or changed order')
        from scar.ir.program.function_sites import LambdaEnvironmentField
        need(tuple(c.formal for c in record.captures)==tuple(m.reference for m in environment.members)
             and record.captures[0].normal_result==0 and record.captures[1].source_use==args[1].id
             and tuple(c.created_field for c in record.captures[2:5])==tuple(LambdaEnvironmentField)
             and tuple(c.source_use for c in record.captures[5:])==tuple(u.id for u in closure)
             and all(c.normal_result is None for c in record.captures[5:]),'entry captures differ from creation NORMAL ports/original Uses')
        if not structural_only:
            from scar.analysis.static.lowering.source_lambda import compiler_lambda
            from scar.analysis.environment_lowering import captured_cell_slot
            evidence=options.code_evidence
            need(evidence is not None,'independent readonly parent CodeWitness required')
            load,child=compiler_lambda(node,evidence.code)
            need(constant.code_key==evidence.code_layout_key and constant.constant_path==(load.arg,),
                 'Lambda code constant differs from original parent slot')
            transient={r.name for r in language.compiler_locals if r.category=='COMPREHENSION_TRANSIENT'}
            names=tuple(n for n in evidence.code.co_cellvars if n not in transient)
            need(len(names)==len(own_cells),'parent cell layout differs')
            by_name=dict(zip(names,own_cells));by_name.update(zip(evidence.code.co_freevars,(m.reference for m in root_env.members[5:])))
            need(tuple(u.value for u in closure)==tuple(by_name[n]for n in child.co_freevars),
                 'Lambda did not capture exact original compiler-ordered cell identities')
            need(tuple(m.cell_slot for m in environment.members[5:])==tuple(captured_cell_slot(record.body.definition,n)for n in child.co_freevars),
                 'child free-cell interface differs from original code/definition')
        child_model,child_graph,child_metadata=region_projection(model,graph,metadata,region,strip_parent=True)
        child_model=replace(child_model,definition=record.body.definition,source=record.source)
        normals=[o for o in child_graph.operations if o.terminator is not None and o.terminator.kind is TerminatorKind.RETURN]
        need(len(normals)==1,'ordinary expression body must retain one actual NORMAL return site')
        terminal=normals[0];own={u.id:u for u in terminal.uses}
        result=own[terminal.terminator.values[0]].value
        need(len(terminal.terminator.values)==2 and type(values[result])is ReferenceValue
             and type(values[own[terminal.terminator.values[1]].value])is StateVersionValue,
             'Lambda normal exit must separate its Python payload from world')
        index_addresses=_check_expression(node.body,result,child_model,child_graph,child_metadata,semantic,record.body)
        # The expression adapter above owns these exact INDEX refinements.
        # Hand only those verified rows out of the ordinary environment view;
        # every other row remains subject to its exact coverage check. This
        # private projection never changes the original canonical graph.
        ordinary=replace(child_graph,state_address_refinements=tuple(
            row for row in child_graph.state_address_refinements if row.site.operation not in index_addresses))
        from scar.analysis.structured_lowering import _require_valid_structured_body
        _require_valid_structured_body(child_model,ordinary,record.body,child_metadata,
            semantic=semantic,control=control,sealed_source=sealed_source,language=None,source_tree=source_tree,
            structural_only=True)
        atoms={(type(n).__name__,span(n))for n in ast.walk(node.body)if hasattr(n,'end_lineno')and n.end_lineno is not None}
        need(not any(owners[l.operation]==graph.root and(semantic.source_atoms[l.source.atom_id].kind,refspan(l.source))in atoms
                     for l in metadata.operation_links),'creating frame eagerly executed the deferred body expression')
        delegated.update((identity,read.id))
    need(len(body_regions)==len(records),'Lambda body Region reused by different source creations')
    return delegated


def root_projection(model,graph,metadata,language,delegated):
    removed={r.body.region for r in language.lambda_bodies}
    kept={i for r in graph.regions if r.id not in removed for b in r.blocks for i in b.operations}
    values={v.id for v in graph.values if v.region not in removed}
    environments=tuple(e for e in graph.environments if e.region not in removed);envs={e.id for e in environments}
    graph=replace(graph,regions=tuple(r for r in graph.regions if r.id not in removed),
        operations=tuple(o for o in graph.operations if o.id in kept),values=tuple(v for v in graph.values if v.id in values),
        environments=environments,function_entries=tuple(e for e in graph.function_entries if e.environment in envs),
        binding_definitions=tuple(r for r in graph.binding_definitions if r.value in values),
        binding_reads=tuple(r for r in graph.binding_reads if r.operation in kept),
        state_address_refinements=tuple(r for r in graph.state_address_refinements if r.site.operation in kept))
    model=replace(model,protocols=tuple(p for p in model.protocols if p.operation in kept and p.operation not in delegated))
    metadata=replace(metadata,operation_links=tuple(r for r in metadata.operation_links if r.operation in kept),
        value_links=tuple(r for r in metadata.value_links if r.value in values),
        use_links=tuple(r for r in metadata.use_links if r.operation in kept),call_targets=tuple(r for r in metadata.call_targets if r.operation in kept))
    language=replace(language,constants=tuple(r for r in language.constants if r.operation not in delegated),lambda_bodies=())
    return model,graph,metadata,language


__all__ = [
    'lower_lambda',
    'assemble_lambda_bodies',
    'compiler_lambda',
    'validate_lambda_regions',
    'root_projection',
    'VERSION',
    'span',
    'need',
    'refspan',
    'region_projection',
]

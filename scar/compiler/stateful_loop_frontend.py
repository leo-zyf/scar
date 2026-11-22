"""Original-F4 proof of a shared range loop over one stateful CALL.

The loop carries contents, not a Python result assignment. This adapter
reuses the formal loop/profile interpreters; no AST or function-name rule
decides that a loop is replaceable. Live identities remain obligations.
"""
from collections import Counter
from dataclasses import dataclass,asdict
from hashlib import sha256
import json
from pathlib import Path

from scar.analysis.source_body_interfaces import BodyInterfaceResolver
from scar.analysis.source_call_view import source_call_site,function_interface_fact
from scar.analysis.source_reference_slice import source_port
from scar.compiler.caller_pipeline import _abi,_global_name,_reference,_signature
from scar.compiler.graph_frontend import _origin,_reject,_ENV,_WORLD,_Scalar
from scar.compiler.loop_frontend import _LoopLowerer,_Parameter,_Range,_Iterator,_RANGE,_ITEM
from scar.compiler.model import SourceRef
from scar.compiler.stateful_model import StatefulRegion,stateful_digest,stateful_from_dict,document as step_document
from scar.ir.execution_recipe import artifact_reference,read_artifact
from scar.ir.program import source_productions as P, normal_graph as F, normal_graph as G
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.record_codec import encode


SCHEMA='scar.stateful-loop-proof'
_NONE=object()


def _canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def _digest(value):return sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class StatefulLoopProof:
    id:str
    source:SourceRef
    parameters:tuple[str,...]
    step:StatefulRegion
    step_reference:dict
    trip_count_parameter:str
    parameter_map:dict[str,str]
    step_global_name:str
    caller_unit:str
    function_ordinal:int
    interface_ordinal:int
    call_evidence:dict
    origins:tuple[dict,...]
    producer_pins:dict


@dataclass(frozen=True)
class _Step:
    name:str


class _StatefulLoopLowerer(_LoopLowerer):
    def __init__(self,source,function,resolver,global_name,selected,*,syntax):
        super().__init__(source,function,resolver,{global_name:(_Step(global_name),)},syntax=syntax)
        self.selected=selected;self.global_name=global_name;self.call_arguments=None

    def action(self,context,step,args,origin):
        if step.instruction.action is not P.LanguageActionKind.CALL:
            _reject('STATE_LOOP_EXTRA_ACTION','only original range and selected stateful CALL admitted',origin)
        fact=self.call_facts.get(context.result(step).port)
        if (fact is None or fact.positional_pack is not None or fact.keyword_pack is not None
                or args[-1] is not _WORLD):
            _reject('STATE_LOOP_CALL_PACK','explicit original CALL operands required',origin)
        operands=(fact.callee,*fact.positional,*(index for _,index in fact.keywords),fact.world)
        if (tuple(context.value(v).port for v in step.arguments)!=tuple(source_port(self.facts,i).port for i in operands)
                or context.result(step,1).port!=source_port(self.facts,fact.result_world).port):
            _reject('STATE_LOOP_CALL_ABI','original CALL operands/results/world differ',origin)
        if args[0] is _RANGE:
            if(self.phase!='PRE' or self.range_calls or fact.keywords or len(args)!=3
                    or type(args[1])is not _Parameter):
                _reject('STATE_LOOP_RANGE','one range(original exact-int parameter) required',origin)
            self.range_calls+=1
            self.origins.append(dict(origin,role='original_builtin_range',trip_parameter_index=args[1].index))
            return (_Range(args[1].index),_WORLD)
        if (self.phase!='BODY' or self.body_calls or type(args[0])is not _Step
                or context.result(step)!=self.selected.link.result
                or any(type(value)is not _Parameter for value in args[1:-1])):
            _reject('STATE_LOOP_BODY_CALL','one selected stateful CALL on fixed outer parameters required',origin)
        self.body_calls+=1;self.selected_body=args[0]
        self.call_arguments=dict(positional=tuple(value.index for value in args[1:1+len(fact.positional)]),
            keywords=tuple((name,value.index)for (name,_),value in zip(fact.keywords,args[1+len(fact.positional):-1],strict=True)))
        self.origins.append(dict(origin,role='original_stateful_step',world_before=encode(context.worlds(step)[0]),
            world_after=encode(context.worlds(step)[1]),source_result_is_none=True))
        return (_NONE,_WORLD)

    def control(self,context,step,args,origin):
        ins=step.instruction
        if (self.phase!='PRE' or self.loops or type(ins.condition)is not P.IterationCondition
                or ins.invariant_count!=1 or ins.condition.world_parameter!=1
                or len(args)!=2 or type(args[0])is not _Iterator or args[1]is not _WORLD):
            _reject('STATE_LOOP_CONTROL','one shared iterator/world loop required',origin)
        self.loops+=1;self.roles=(None,args[0].range.trip)
        if ins.condition.advance.arguments!=tuple(P.ProductionParameter(step.region,i)for i in range(2)):
            _reject('STATE_LOOP_ADVANCE','original iterator/world parameters required',origin)
        exhausted=context.regions[ins.condition.exhausted];items=[context.steps[s]for s in exhausted.steps]
        if(len(items)!=2 or type(items[0].instruction)is not P.ApplyPythonProtocol
                or items[0].instruction.protocol.value!='MATCH_STOP_ITERATION'
                or items[0].arguments!=tuple(P.ProductionParameter(exhausted.id,i)for i in range(2))
                or type(items[1].instruction)is not P.TransferCompletion or items[1].instruction.kind is not C.FALLTHROUGH
                or items[1].arguments!=(P.ProductionResult(items[0].id,0),)):
            _reject('STATE_LOOP_EXHAUSTION','original StopIteration classification required',origin)
        if ins.otherwise is not None:
            branch=context.regions[ins.otherwise];items=[context.steps[s]for s in branch.steps]
            if(len(items)!=1 or type(items[0].instruction)is not P.TransferCompletion
                    or items[0].instruction.kind is not C.FALLTHROUGH
                    or items[0].arguments!=(P.ProductionParameter(branch.id,1),)):
                _reject('STATE_LOOP_ELSE','nonempty loop else is outside this profile',origin)
        self.phase='CONDITION';kind,result=self.run(context,ins.condition.advance.region,args)
        if kind is not C.FALLTHROUGH or result!=(_ITEM,_WORLD) or self.next_calls!=1:
            _reject('STATE_LOOP_NEXT','advance must only fetch the next original range item',origin)
        self.phase='BODY';kind,result=self.run(context,ins.body,(*args,_ITEM))
        if kind is not C.FALLTHROUGH or result!=(_WORLD,) or self.body_calls!=1 or self.item_writes!=1:
            _reject('STATE_LOOP_BODY','one ignored loop item and one complete step required',origin)
        self.phase='POST'
        self.origins.append(dict(origin,role='shared_stateful_loop',header_region=step.region.wire,
            body_region=ins.body.wire,iteration_condition=encode(ins.condition),
            iterator=encode(context.gp(context.emitter.port(step.region,G.PortKind.PARAMETER,0))),
            backedge=dict(from_region=ins.body.wire,to_region=step.region.wire,world_parameter_index=1),
            source_body_copies=1,iterations_unrolled=False))
        return (_WORLD,)

    def run(self,context,region_id,parameters):
        identity=context.key,region_id
        if identity in self.active:_reject('STATE_LOOP_RECURSIVE_RECIPE','recursive control recipe unsupported')
        if context.owner(region_id).owner!=self.function.body_scope.owner:
            _reject('STATE_LOOP_CODE_SCOPE','loop crossed its original code scope')
        region=context.regions[region_id]
        if len(parameters)!=len(region.parameters):_reject('STATE_LOOP_REGION_ARITY','original region ABI differs')
        self.active.add(identity);values={P.ProductionParameter(region_id,i):value for i,value in enumerate(parameters)}
        try:
            for sid in region.steps:
                step=context.steps[sid];ins=step.instruction;args=tuple(values[v]for v in step.arguments)
                origin=_origin(context,region_id,step,arguments=[encode(context.value(v))for v in step.arguments])
                self.origins.append(origin)
                if type(ins)is P.TransferCompletion:
                    if ins.kind not in(C.FALLTHROUGH,C.RETURN)or ins.kind is C.RETURN and self.phase!='POST':
                        _reject('STATE_LOOP_COMPLETION','break/continue/early return unsupported',origin)
                    return ins.kind,args
                if type(ins)is P.EvaluateChild:
                    link=self.source.material.interfaces['links'][context.key,step.id]
                    child=self.resolver.plan(self.source.unit,link.callee);kind,result=self.run(child,child.plan.root,args)
                    if kind is not C.FALLTHROUGH:return kind,result
                elif type(ins)is P.ApplyRegion:
                    kind,result=self.run(context,ins.invocation.region,tuple(values[v]for v in ins.invocation.arguments))
                    if kind is not C.FALLTHROUGH:return kind,result
                elif type(ins)is P.HandleCompletions:
                    if(len(ins.routes)!=1 or ins.routes[0].kind is not C.FALLTHROUGH
                            or ins.routes[0].target is not None or ins.routes[0].captures):
                        _reject('STATE_LOOP_HANDLER','only normal implicit-None return adapter admitted',origin)
                    route=ins.routes[0];handler=context.regions[route.handler];items=[context.steps[s]for s in handler.steps]
                    if(len(handler.parameters)!=1 or len(items)!=2 or type(items[0].instruction)is not P.DefineLiteral
                            or items[0].instruction.value.to_python()is not None
                            or type(items[1].instruction)is not P.TransferCompletion or items[1].instruction.kind is not C.RETURN):
                        _reject('STATE_LOOP_HANDLER_EFFECT','handler must return the original literal None',origin)
                    kind,result=self.run(context,ins.protected.region,tuple(values[v]for v in ins.protected.arguments))
                    if kind is C.FALLTHROUGH:kind,result=self.run(context,route.handler,result)
                    if kind is not C.FALLTHROUGH:return kind,result
                elif type(ins)is P.DefineLiteral and ins.value.to_python()is None:
                    result=(_NONE,*args)
                elif type(ins)is P.ConstructValue:
                    value=super().construct(context,step,args,{},origin)
                    result=tuple(_NONE if type(v)is _Scalar and v.value is None else v for v in value)
                elif type(ins)is P.EnterDeferredBody:result=(_WORLD,)
                elif type(ins)is P.AccessBinding:result=self.binding(context,step,args)
                elif type(ins)is P.ApplyLanguageAction:result=self.action(context,step,args,origin)
                elif type(ins)is P.ApplyPythonProtocol:result=self.protocol(context,step,args,origin)
                elif type(ins)is P.LoopControl:result=self.control(context,step,args,origin)
                else:_reject('STATE_LOOP_OPEN_INSTRUCTION',type(ins).__name__+' has unclosed interstep semantics',origin)
                normal=next((c for c in step.completions if c.kind is C.FALLTHROUGH),None)
                if normal is None or len(result)!=len(normal.types):_reject('STATE_LOOP_RESULT_ABI','fallthrough ABI differs',origin)
                values.update((P.ProductionResult(step.id,i),value)for i,value in enumerate(result))
                if type(ins)is P.LoopControl:
                    if region.kind is not P.ProductionRegionKind.LOOP or region.steps!=(step.id,):
                        _reject('STATE_LOOP_REGION','one LoopControl must own its dedicated region',origin)
                    if normal not in region.completions:
                        _reject('STATE_LOOP_REGION_COMPLETION','original loop completion ABI differs',origin)
                    return C.FALLTHROUGH,result
            _reject('STATE_LOOP_TERMINATOR','source region has no original terminator')
        finally:self.active.remove(identity)

    def lower(self):
        function=self.function;_signature(function)
        context=self.resolver.plan(self.source.unit,function.template)
        if any(type(step.instruction)is P.ApplyDecorator for step in context.plan.steps):
            _reject('STATE_LOOP_DECORATOR','decorated loop entry unsupported')
        for i,_ in enumerate(function.parameter_names):
            formal=source_port(self.facts,function.formals[i+1]);names=self.names.get((formal.port,True),())
            if len(names)!=1 or names[0].kind is not F.NamespaceKind.LEXICAL:
                _reject('STATE_LOOP_FORMAL','original lexical parameter binding required')
            self.parameter_namespaces[i]=names[0];self.locals[names[0]]=_Parameter(i)
        kind,result=self.run(context,function.body,(_ENV,*(_Parameter(i)for i in range(len(function.parameter_names))),_WORLD))
        if(kind is not C.RETURN or result!=(_NONE,_WORLD)or self.loops!=1 or self.range_calls!=1
                or self.iterator_gets!=1 or self.body_calls!=1 or self.call_arguments is None):
            _reject('STATE_LOOP_RETURN','one complete shared range loop followed by None return required')
        expected=Counter({('PRE',self.parameter_namespaces[self.roles[1]]):1})
        actuals=(*self.call_arguments['positional'],*(i for _,i in self.call_arguments['keywords']))
        expected.update(('BODY',self.parameter_namespaces[i])for i in actuals)
        if self.reads!=expected or set(actuals)|{self.roles[1]}!=set(self.parameter_namespaces):
            _reject('STATE_LOOP_EXTRA_LOCAL_USE','only trip count and fixed original CALL argument reads admitted')
        return self.roles[1],self.call_arguments,tuple(self.origins)


def _pins():
    from scar.compiler import loop_frontend,stateful_frontend,caller_pipeline,graph_frontend
    return {name:sha256(Path(module.__file__).read_bytes()).hexdigest()for name,module in(
        ('loop_proof',__import__(__name__,fromlist=['__file__'])),('loop_control',loop_frontend),
        ('stateful_source',stateful_frontend),('call_abi',caller_pipeline),('source_values',graph_frontend))}


def document(proof):
    value=dict(schema=SCHEMA,schema_version=1,**asdict(proof))
    value['step']=step_document(proof.step)
    return value


def build_stateful_loop(view,provider,*,caller_unit,function_ordinal,interface_ordinal,step_reference):
    from scar.compiler.stateful_frontend import discover_stateful_regions
    if step_reference.get('schema')!='scar.stateful-array-region':raise ValueError('retained stateful step required')
    step=stateful_from_dict(read_artifact(step_reference));matches,_=discover_stateful_regions(view,source=step.source.path)
    if len([r for r in matches if stateful_digest(r)==stateful_digest(step)])!=1:
        raise ValueError('stateful loop step differs from complete original source discovery')
    source=view.unit(caller_unit);function=source.facts.functions[function_ordinal];_signature(function)
    material=provider.store.read_unit(caller_unit);interface=material.interfaces[interface_ordinal]
    target=view.unit(interface.link.body.unit)
    targets=[f for f in target.facts.functions if f.template==interface.link.body.template and f.body==interface.link.body.region]
    if len(targets)!=1:raise ValueError('selected step interface lacks one original function body')
    target_function=targets[0]
    target_identity=_reference(view,target.unit,target_function,(),{})
    if any(getattr(target_identity,k)!=getattr(step.source,k)for k in('path','sha256','code_key','qualname')):
        raise ValueError('selected source CALL does not target the admitted stateful step')
    routes,formals=_abi(source,function,interface,target_function)
    name=_global_name(source.facts,interface.link.callee)
    resolver=BodyInterfaceResolver(view);resolver.prepare(interface.link.body)
    identities={};_reference(view,caller_unit,function,(),identities)
    lowerer=_StatefulLoopLowerer(source,function,resolver,name,interface,syntax=identities[caller_unit][1])
    trip,args,origins=lowerer.lower();reference=_reference(view,caller_unit,function,origins,identities)
    fact=source.facts.calls[interface.query]
    positions={source_port(source.facts,i):function.parameter_names[p]
        for i,p in zip(fact.positional,args['positional'],strict=True)}
    positions.update((source_port(source.facts,i),function.parameter_names[p])
        for (_,i),(_,p)in zip(fact.keywords,args['keywords'],strict=True))
    mapping={name:positions[routes[formal].actuals[0]]for name,formal in zip(step.parameter_names,formals[1:-1],strict=True)}
    provider.attach(caller_unit,queries=(interface.query,));frame=provider.frames[caller_unit,interface_ordinal]
    operation,_,_=source_call_site(view,interface.link)
    evidence=dict(frame=encode(frame),interface=encode(function_interface_fact(provider.store,caller_unit,interface_ordinal)),
        original_query=interface.query,original_operation=encode(operation),parameters=encode(interface.link.parameters),
        original_call=encode(interface.call),returns=encode(interface.link.returns),result=encode(interface.link.result),
        world_before=encode(source_port(source.facts,fact.world)),world_after=encode(source_port(source.facts,fact.result_world)),
        positional_parameters=[function.parameter_names[i]for i in args['positional']],
        keyword_parameters={name:function.parameter_names[i]for name,i in args['keywords']},
        snapshot_guards_are_not_runtime_dispatch_proof=True)
    proof=StatefulLoopProof('',reference,function.parameter_names,step,step_reference,function.parameter_names[trip],mapping,
        name,caller_unit,function_ordinal,interface_ordinal,evidence,origins,_pins())
    provider.store.validate();source.check();read_artifact(step_reference,json_document=False)
    from dataclasses import replace
    return replace(proof,id='stateful-loop-proof:'+_digest(document(proof)))


def validate_stateful_loop_proof(proof,*,view,provider):
    if type(proof)is not StatefulLoopProof or proof.producer_pins!=_pins():raise ValueError('current source-proven stateful loop required')
    replay=build_stateful_loop(view,provider,caller_unit=proof.caller_unit,function_ordinal=proof.function_ordinal,
        interface_ordinal=proof.interface_ordinal,step_reference=proof.step_reference)
    if _canonical(document(proof))!=_canonical(document(replay)):raise ValueError('stateful loop differs from full F4/step/ABI replay')
    return replay


def write_stateful_loop_proof(proof,path):
    path=Path(path).resolve()
    with path.open('xb')as stream:stream.write(_canonical(document(proof))+b'\n')
    return artifact_reference(path,SCHEMA)


def read_stateful_loop_proof(path,*,view,provider):
    from scar.compiler.model import SourceRef
    raw=json.loads(Path(path).read_text())
    if raw.pop('schema',None)!=SCHEMA or raw.pop('schema_version',None)!=1:raise ValueError('stateful loop proof schema required')
    source=dict(raw['source']);source['origins']=tuple(source.get('origins',()))
    raw['source']=SourceRef(**source);raw['step']=stateful_from_dict(raw['step'])
    for key in('parameters','origins'):raw[key]=tuple(raw[key])
    return validate_stateful_loop_proof(StatefulLoopProof(**raw),view=view,provider=provider)

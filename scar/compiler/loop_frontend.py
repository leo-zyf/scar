"""Extract one original range loop and lexical array carry without unrolling."""
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from scar.analysis.program_view import ProgramView
from scar.analysis.source_body_interfaces import BodyInterfaceResolver
from scar.analysis.source_references import source_port
from scar.compiler.graph_frontend import (_GraphLowerer,_source_identity,_metadata,_origin,
    GraphRegionRejected,_reject,_ENV,_WORLD)
from scar.compiler.pipeline_frontend import discover_graph_pipelines, _global_functions, _Callee
from scar.compiler.pipeline_frontend import CalleeGuard
from scar.compiler.loop_admission import ArrayLoopPlan, validate_loop
from scar.compiler.model import SourceRef
from scar.ir.program import source_productions as P, normal_graph as F, normal_graph as G
from scar.ir.program.normal_graph import Body
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.record_codec import encode
from scar.trace.code_snapshot import code_key


@dataclass(frozen=True)
class _Parameter:
    index: int


@dataclass(frozen=True)
class _Range:
    trip: int


@dataclass(frozen=True)
class _Iterator:
    range: _Range


@dataclass(frozen=True)
class _Carry:
    parameter: int
    updated: bool=False


_RANGE=object()
_ITEM=object()


class _LoopLowerer(_GraphLowerer):
    def __init__(self,source,function,resolver,globals,*,syntax=None):
        super().__init__(source,function,resolver,syntax=syntax)
        self.globals=globals;self.phase='PRE';self.loops=0;self.range_calls=0;self.iterator_gets=0
        self.body_calls=0;self.carry_writes=0;self.item_writes=0;self.next_calls=0
        self.roles=None;self.selected_body=None;self.parameter_namespaces={};self.reads=Counter()
        self.call_facts={source_port(self.facts,row.result).port:row for row in self.facts.calls}

    def binding(self,context,step,args):
        action=step.instruction.action;write=action is P.BindingActionKind.WRITE
        if action not in (P.BindingActionKind.READ,P.BindingActionKind.WRITE):
            _reject('LOOP_BINDING_DELETE','binding deletion is outside the single-carry profile')
        port=context.value(step.arguments[0]).port if write else context.result(step).port
        names=tuple(dict.fromkeys(self.names.get((port,write),())))
        if names and all(ns.kind is F.NamespaceKind.LEXICAL for ns in names):
            if len(names)!=1:_reject('LOOP_BINDING_AMBIGUOUS','one original lexical binding required')
            ns=names[0]
            if not write:
                if ns not in self.locals:_reject('LOOP_UNBOUND_LOCAL','local read lacks its original parameter/write')
                value=self.locals[ns]
                if value is _ITEM:_reject('LOOP_ITERATION_ITEM_USED','iteration item must remain unused')
                self.reads[self.phase,ns]+=1
                return (value,_WORLD)
            if self.phase!='BODY':_reject('LOOP_EXTRA_ASSIGNMENT','only carry and ignored iteration-target writes are supported')
            value=args[0]
            if value is _ITEM:
                if ns in self.parameter_namespaces.values():
                    _reject('LOOP_TARGET_CLOBBERS_PARAMETER','iteration target cannot overwrite a parameter')
                self.item_writes+=1;self.locals[ns]=value
            elif isinstance(value,_Carry) and value.updated and ns==self.parameter_namespaces[self.roles[0]]:
                self.carry_writes+=1;self.locals[ns]=value
                self.origins.append(dict(_origin(context,step.region,step),role='lexical_carry_write',binding=encode(ns)))
            else:_reject('LOOP_EXTRA_ASSIGNMENT','body writes must update the original carry from its pipeline result')
            return (_WORLD,)
        if write:_reject('LOOP_EXTERNAL_WRITE','external namespace writes are unsupported')
        if (not names or any(ns.kind not in (F.NamespaceKind.MODULE,F.NamespaceKind.BUILTINS) for ns in names)
                or len({ns.name for ns in names})!=1 or not any(ns.kind is F.NamespaceKind.MODULE for ns in names)):
            _reject('LOOP_DYNAMIC_LOOKUP','direct module/builtin lookup required')
        name=names[0].name
        if name=='range':return (_RANGE,_WORLD)
        candidates=self.globals.get(name,())
        if len(candidates)!=1:_reject('LOOP_BODY_TARGET_OPEN','global has no unique admitted pipeline body')
        return (candidates[0],_WORLD)

    def action(self,context,step,args,origin):
        if step.instruction.action is not P.LanguageActionKind.CALL:
            _reject('LOOP_EXTRA_ACTION','only builtin range and the original pipeline call are supported',origin)
        fact=self.call_facts.get(context.result(step).port)
        if (fact is None or fact.keywords or fact.positional_pack is not None or fact.keyword_pack is not None
                or len(fact.positional)!=1 or len(args)!=3):
            _reject('LOOP_CALL_ARGUMENTS','one fixed positional operand is required',origin)
        actual=tuple(context.value(v).port for v in step.arguments)
        expected=tuple(source_port(self.facts,i).port for i in (fact.callee,*fact.positional,fact.world))
        if actual!=expected or context.result(step,1).port!=source_port(self.facts,fact.result_world).port:
            _reject('LOOP_CALL_ABI','CALL facts differ from original F4 arguments/results',origin)
        if args[0] is _RANGE:
            if self.phase!='PRE' or self.range_calls or type(args[1])is not _Parameter:
                _reject('LOOP_RANGE_FORM','only one range(original trip parameter) is supported',origin)
            self.range_calls+=1
            self.origins.append(dict(origin,role='builtin_range',trip_parameter_index=args[1].index,
                runtime_obligation='GLOBAL_THEN_BUILTINS_LOOKUP_IS_EXACT_BUILTIN_RANGE'))
            return (_Range(args[1].index),_WORLD)
        if (self.phase!='BODY' or not isinstance(args[0],_Callee) or not isinstance(args[1],_Carry)
                or args[1].updated or args[1].parameter!=self.roles[0] or self.body_calls):
            _reject('LOOP_BODY_CALL','each iteration must call the admitted pipeline exactly once on current carry',origin)
        body=args[0].plan
        if len(body.parameters)!=1:_reject('LOOP_BODY_ARITY','body pipeline must have one array parameter',origin)
        self.body_calls+=1;self.selected_body=args[0]
        self.origins.append(dict(origin,role='body_pipeline_call',body_pipeline_id=body.id,
            call_inputs=[encode(source_port(self.facts,i)) for i in fact.positional]))
        return (_Carry(self.roles[0],True),_WORLD)

    def protocol(self,context,step,args,origin):
        kind=step.instruction.protocol.value
        if kind=='GET_ITERATOR' and self.phase=='PRE' and len(args)==2 and type(args[0])is _Range:
            self.iterator_gets+=1;return (_Iterator(args[0]),_WORLD)
        if kind=='COMPREHENSION_NEXT' and self.phase=='CONDITION' and len(args)==2 and type(args[0])is _Iterator:
            self.next_calls+=1;return (_ITEM,_WORLD)
        if kind=='FRAME_RELEASE_TEMPORARY':return (_WORLD,)
        _reject('LOOP_PROTOCOL','unadmitted protocol '+kind,origin)

    def control(self,context,step,args,origin):
        ins=step.instruction
        if (self.phase!='PRE' or self.loops or type(ins.condition)is not P.IterationCondition
                or ins.invariant_count!=1 or ins.condition.world_parameter!=1
                or len(args)!=2 or type(args[0])is not _Iterator or args[1]is not _WORLD):
            _reject('LOOP_CONTROL_FORM','one iterator/world range loop required',origin)
        self.loops+=1;self.roles=(1-args[0].range.trip,args[0].range.trip)
        if ins.condition.advance.arguments!=tuple(P.ProductionParameter(step.region,i) for i in range(2)):
            _reject('LOOP_ADVANCE_BINDING','advance must consume the original iterator/world parameters',origin)
        exhausted=context.regions[ins.condition.exhausted];exhausted_steps=[context.steps[s] for s in exhausted.steps]
        if (len(exhausted_steps)!=2 or type(exhausted_steps[0].instruction)is not P.ApplyPythonProtocol
                or exhausted_steps[0].instruction.protocol.value!='MATCH_STOP_ITERATION'
                or exhausted_steps[0].arguments!=tuple(P.ProductionParameter(exhausted.id,i) for i in range(2))
                or type(exhausted_steps[1].instruction)is not P.TransferCompletion
                or exhausted_steps[1].instruction.kind is not C.FALLTHROUGH
                or exhausted_steps[1].arguments!=(P.ProductionResult(exhausted_steps[0].id,0),)):
            _reject('LOOP_EXHAUSTION_HANDLER','only original StopIteration classification is supported',origin)
        if ins.otherwise is not None:
            otherwise=context.regions[ins.otherwise];items=[context.steps[s] for s in otherwise.steps]
            if (len(items)!=1 or type(items[0].instruction)is not P.TransferCompletion
                    or items[0].instruction.kind is not C.FALLTHROUGH
                    or items[0].arguments!=(P.ProductionParameter(otherwise.id,1),)):
                _reject('LOOP_ELSE','nonempty loop else is outside this profile',origin)
        self.phase='CONDITION';kind,condition=self.run(context,ins.condition.advance.region,args)
        if kind is not C.FALLTHROUGH or condition!=(_ITEM,_WORLD) or self.next_calls!=1:
            _reject('LOOP_CONDITION','advance must only fetch the next original range item',origin)
        state_namespace=self.parameter_namespaces[self.roles[0]]
        self.locals[state_namespace]=_Carry(self.roles[0])
        self.phase='BODY';kind,result=self.run(context,ins.body,(*args,_ITEM))
        if (kind is not C.FALLTHROUGH or result!=(_WORLD,) or self.body_calls!=1
                or self.carry_writes!=1 or self.item_writes!=1
                or self.locals[state_namespace]!=_Carry(self.roles[0],True)):
            _reject('LOOP_CARRY','one ignored item and one pipeline-result carry write required',origin)
        self.phase='POST'
        self.origins.append(dict(origin,role='single_shared_loop',iterator_parameter=encode(context.gp(
            context.emitter.port(step.region,G.PortKind.PARAMETER,0))),
            iteration_condition=encode(ins.condition),body_region=ins.body.wire,
            lexical_carry_binding=encode(state_namespace),backedge=dict(from_region=ins.body.wire,
                to_region=step.region.wire,world_parameter_index=1,explicit_array_phi=False),
            body_materializations=1,loop_unrolled=False))
        return (_WORLD,)

    def run(self,context,region_id,parameters):
        identity=context.key,region_id
        if identity in self.active:_reject('LOOP_RECURSIVE_RECIPE','recursive recipe is outside this profile')
        if context.owner(region_id).owner!=self.function.body_scope.owner:
            _reject('LOOP_CODE_SCOPE','loop crossed the original code owner')
        region=context.regions[region_id]
        if len(parameters)!=len(region.parameters):_reject('LOOP_REGION_ARITY','original region arity differs')
        self.active.add(identity);values={P.ProductionParameter(region_id,i):v for i,v in enumerate(parameters)}
        try:
            for sid in region.steps:
                step=context.steps[sid];ins=step.instruction;args=tuple(values[v] for v in step.arguments)
                origin=_origin(context,region_id,step,arguments=[encode(context.value(v)) for v in step.arguments])
                self.origins.append(origin)
                if type(ins)is P.TransferCompletion:
                    if ins.kind not in (C.FALLTHROUGH,C.RETURN) or ins.kind is C.RETURN and self.phase!='POST':
                        _reject('LOOP_COMPLETION','break/continue/early return is unsupported',origin)
                    return ins.kind,args
                if type(ins)is P.EvaluateChild:
                    link=self.source.material.interfaces['links'][context.key,step.id]
                    child=self.resolver.plan(self.source.unit,link.callee);kind,result=self.run(child,child.plan.root,args)
                    if kind is not C.FALLTHROUGH:return kind,result
                elif type(ins)is P.ApplyRegion:
                    kind,result=self.run(context,ins.invocation.region,tuple(values[v] for v in ins.invocation.arguments))
                    if kind is not C.FALLTHROUGH:return kind,result
                elif type(ins)is P.EnterDeferredBody:result=(_WORLD,)
                elif type(ins)is P.AccessBinding:result=self.binding(context,step,args)
                elif type(ins)is P.ApplyLanguageAction:result=self.action(context,step,args,origin)
                elif type(ins)is P.ApplyPythonProtocol:result=self.protocol(context,step,args,origin)
                elif type(ins)is P.LoopControl:result=self.control(context,step,args,origin)
                else:_reject('LOOP_UNSUPPORTED_INSTRUCTION',type(ins).__name__+' is outside single-carry control',origin)
                normal=next((c for c in step.completions if c.kind is C.FALLTHROUGH),None)
                if normal is None or len(result)!=len(normal.types):_reject('LOOP_RESULT_ARITY','original fallthrough ABI differs',origin)
                values.update((P.ProductionResult(step.id,i),v) for i,v in enumerate(result))
                if type(ins)is P.LoopControl:
                    # F4's dedicated LOOP region is terminated by its control
                    # operation: its completion is the region completion ABI.
                    if region.kind is not P.ProductionRegionKind.LOOP or region.steps!=(step.id,):
                        _reject('LOOP_CONTROL_REGION','one original LoopControl must own its loop region',origin)
                    if normal not in region.completions:
                        _reject('LOOP_CONTROL_COMPLETION','loop completion differs from its original region ABI',origin)
                    return C.FALLTHROUGH,result
            _reject('LOOP_MISSING_TRANSFER','region lacks original completion transfer')
        finally:self.active.remove(identity)

    def lower(self):
        function=self.function
        if (len(function.parameter_names)!=2 or function.body_scope is None
                or function.environment.kind is not P.DeferredBodyKind.FUNCTION
                or function.environment.captures or function.environment.owned_cells
                or any(p.kind in (P.DeferredParameterKind.VAR_POSITIONAL,P.DeferredParameterKind.VAR_KEYWORD)
                    for p in function.environment.parameters)):
            _reject('LOOP_SIGNATURE','ordinary fixed two-parameter function without captures required')
        context=self.resolver.plan(self.source.unit,function.template)
        if any(type(s.instruction)is P.ApplyDecorator for s in context.plan.steps):
            _reject('LOOP_DECORATOR','decorated loop entry is unsupported')
        for i,name in enumerate(function.parameter_names):
            formal=source_port(self.facts,function.formals[i+1]);names=self.names.get((formal.port,True),())
            if len(names)!=1 or names[0].kind is not F.NamespaceKind.LEXICAL:
                _reject('LOOP_PARAMETER_BINDING','formal lacks one original lexical binding')
            self.parameter_namespaces[i]=names[0];self.locals[names[0]]=_Parameter(i)
        kind,result=self.run(context,function.body,(_ENV,_Parameter(0),_Parameter(1),_WORLD))
        if (self.loops!=1 or self.range_calls!=1 or self.iterator_gets!=1 or self.selected_body is None
                or kind is not C.RETURN or result!=(_Carry(self.roles[0],True),_WORLD)):
            _reject('LOOP_FINAL_RETURN','one range loop followed by return of its original lexical carry required')
        state,trip=self.roles
        if (self.reads['PRE',self.parameter_namespaces[trip]]!=1
                or self.reads['BODY',self.parameter_namespaces[state]]!=1
                or self.reads['POST',self.parameter_namespaces[state]]!=1
                or sum(self.reads.values())!=3):
            _reject('LOOP_EXTRA_LOCAL_USE','only trip lookup, body carry and final carry reads are supported')
        self.origins.append(dict(kind='loop_runtime_obligations',builtin_range='EXACT_BUILTIN_RANGE',
            trip_count='EXACT_INT_POSITIVE_FOR_NATIVE',nonpositive='ORIGINAL_ALIAS_RETURN_VIA_RUNTIME_FALLBACK',
            original_body_and_inner_callee_guards_retained=True,inner_callee_namespace='BODY_FUNCTION_GLOBALS'))
        body=self.selected_body.plan
        return function.parameter_names[state],function.parameter_names[trip],body,CalleeGuard(body.source,(self.selected_body.name,)),tuple(self.origins)


def discover_graph_loops(catalog,source=None):
    view=catalog if isinstance(catalog,ProgramView) else ProgramView(catalog)
    pipelines,_=discover_graph_pipelines(view,source);bodies={}
    for body in pipelines:
        identity=next(row for row in body.source.origins if row.get('kind')=='source_identity')
        bodies[identity['unit'],identity['occurrence']]=body
    units=tuple(view.catalog.fact_store.rows) if source is None else tuple(sorted(
        view.catalog.units_for_source(str(Path(source).absolute())) or ()))
    if source is not None and not units:return [],[dict(path=str(Path(source).absolute()),code='SOURCE_NOT_IN_CATALOG')]
    loops=[];rejected=[];resolver=BodyInterfaceResolver(view)
    for unit in units:
        original=view.unit(unit);functions=original.facts.functions
        resolver.prepare_many(Body(unit,fn.template,fn.body) for fn in functions)
        model,syntax,codes,recipe,digest=_source_identity(view.catalog,unit)
        selected={fn.template:bodies[unit,fn.template.occurrence] for fn in functions if (unit,fn.template.occurrence) in bodies}
        globals=_global_functions(original.facts,selected)
        for ordinal,function in enumerate(functions):
            if len(function.parameter_names)!=2:continue
            name,span,lines=_metadata(syntax,function.template.occurrence)
            candidates=[c for c in codes if c.co_name==name and c.co_firstlineno in lines
                and c.co_varnames[:c.co_argcount+c.co_kwonlyargcount]==function.parameter_names]
            try:
                if len(candidates)!=1:_reject('LOOP_CODE_IDENTITY','unique original loop CodeID unavailable')
                state,trip,body,guard,origins=_LoopLowerer(original,function,resolver,globals,syntax=syntax).lower()
                code=candidates[0];key=code_key(code)
                identity=dict(kind='source_identity',method='COMPILE_ONLY_RETAINED_F0',unit=unit,
                    occurrence=function.template.occurrence,span=span,recipe=recipe,code_key=key,
                    semantic_authority='ORIGINAL_F4_PRODUCTION_PLANS_AND_UNIT_FACTS')
                reference=SourceRef(model['path'],digest,code.co_qualname,span['line'],span['column'],key,(identity,))
                plan=ArrayLoopPlan('graph-array-loop:'+sha256((unit+'\0'+function.body.wire+'\0'+key).encode()).hexdigest()[:24],
                    reference,function.parameter_names,state,trip,body,guard,origins)
                loops.append(validate_loop(plan))
            except GraphRegionRejected as error:
                rejected.append(dict(unit=unit,function=ordinal,path=model['path'],qualname=candidates[0].co_qualname
                    if len(candidates)==1 else name,code=error.code,reason=error.code,detail=error.detail,origin=error.origin))
    return loops,rejected


__all__=['discover_graph_loops']

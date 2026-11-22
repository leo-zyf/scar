"""Loop admission."""

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import sys
import re
from pathlib import Path
from scar.compiler.model import (
    SourceRef,
    document as plan_document,
    plan_from_dict,
    read_plan,
    ArrayRegion,
    region_from_dict,
)
from scar.compiler.pipeline_frontend import (
    PipelinePlan,
    CalleeGuard,
    validate_pipeline,
    document as pipeline_document,
    pipeline_from_dict,
    _global_functions,
    _Callee,
    _source_occurrence,
    discover_graph_pipelines,
)
from scar.compiler.model import validate_plan
from collections import Counter
from scar.analysis.program_view import ProgramView
from scar.analysis.source_references import source_port
from scar.compiler.graph_frontend import (
    discover_graph_regions,
    _discovery_for,
    _origin,
    GraphRegionRejected,
    _reject,
    _ENV,
    _WORLD,
    _GraphLowerer,
)
from scar.ir.program import source_productions as P, normal_graph as F, normal_graph as G
from scar.ir.program.normal_graph import Body
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.record_codec import encode


# Typed loop plans and replay of their body, source and placement contracts.
# One guarded array carry and one exact-integer trip-count boundary.
SCHEMA='scar.array-loop-plan'
PROFILE='cpu-array-loop-v1'


@dataclass(frozen=True)
class ArrayLoopPlan:
    id: str
    source: SourceRef
    parameters: tuple[str,...]
    state_parameter: str
    trip_count_parameter: str
    body: PipelinePlan
    body_callee: CalleeGuard
    origins: tuple[dict,...]=()
    schema_version: int=1
    profile: str=PROFILE


def validate_loop(loop):
    if type(loop)is not ArrayLoopPlan or loop.schema_version!=1 or loop.profile!=PROFILE:
        raise ValueError('supported typed array loop required')
    if (type(loop.id)is not str or not loop.id or type(loop.source)is not SourceRef
            or len(loop.parameters)!=2 or len(set(loop.parameters))!=2
            or any(type(name)is not str or not name for name in loop.parameters)
            or loop.state_parameter==loop.trip_count_parameter
            or {loop.state_parameter,loop.trip_count_parameter}!=set(loop.parameters)):
        raise ValueError('loop requires exactly one named array carry and one exact-integer parameter')
    validate_pipeline(loop.body)
    if len(loop.body.parameters)!=1:
        raise ValueError('loop body must accept exactly one array carry')
    if (type(loop.body_callee)is not CalleeGuard or loop.body_callee.source!=loop.body.source
            or loop.body_callee.lookup_root!='PIPELINE_FUNCTION_GLOBALS'
            or len(loop.body_callee.lookup_path)!=1
            or any(type(name)is not str or not name for name in loop.body_callee.lookup_path)):
        raise ValueError('loop body needs its original direct global function guard')
    return loop


def document(loop):
    validate_loop(loop)
    return dict(schema=SCHEMA,schema_version=loop.schema_version,profile=loop.profile,id=loop.id,
        source=asdict(loop.source),parameters=list(loop.parameters),state_parameter=loop.state_parameter,
        trip_count_parameter=loop.trip_count_parameter,body=pipeline_document(loop.body),
        body_callee=dict(lookup_root=loop.body_callee.lookup_root,lookup_path=list(loop.body_callee.lookup_path)),
        origins=list(loop.origins))


def loop_from_dict(value):
    value=dict(value)
    if value.pop('schema',None)!=SCHEMA:raise ValueError('unsupported loop schema')
    source=dict(value['source']);source['origins']=tuple(source.get('origins',()))
    value['source']=SourceRef(**source);value['parameters']=tuple(value['parameters'])
    value['body']=pipeline_from_dict(value['body']);guard=dict(value['body_callee'])
    if set(guard)!={'lookup_root','lookup_path'}:raise ValueError('body guard must reference its shared body source')
    guard['lookup_path']=tuple(guard['lookup_path']);guard['source']=value['body'].source
    value['body_callee']=CalleeGuard(**guard);value['origins']=tuple(value.get('origins',()))
    return validate_loop(ArrayLoopPlan(**value))


def loop_digest(loop):
    return sha256(json.dumps(document(loop),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


digest=loop_digest
def read_loop(path):return loop_from_dict(json.loads(Path(path).read_text()))
def write_loop(loop,path):Path(path).write_text(json.dumps(document(loop),indent=2,allow_nan=False)+'\n')


# Shared legality for the existing synchronous CPU temporal-loop profile.
#
# This admission owns source/body/control/shape/payload and retained placement
# checks. It does not generate C++, select instructions, compile, or execute the
# application. Live array, CPU, fenv and callable guards remain execution duties.
def _require(condition, message):
    if not condition:
        raise ValueError(message)


def check_loop_sources(loop):
    """Verify original loop/body/leaf source snapshots without importing them."""
    seen = set()
    for source in (loop.source, loop.body.source, *(call.callee.source for call in loop.body.calls)):
        key = (source.path, source.sha256, source.code_key)
        if key in seen:
            continue
        seen.add(key)
        _require(bool(source.code_key) and Path(source.path).is_absolute(),
                 "original loop/body/leaf source versions required")
        _require(sha256(Path(source.path).read_bytes()).hexdigest() == source.sha256,
                 "loop or nested callee source snapshot differs")


def _component(plan):
    # Shared structural validation covers schema/profile/ABI, effects, exact
    # schedule, attributes, constants, DAG closure, resource and payload fields.
    validate_plan(plan)
    if plan.resource_binding is not None:
        _require(sys.platform == "linux", "C++ CPU affinity binding requires Linux")
        _require(all(cpu < 2**20 for cpu in plan.resource_binding.cpu_ids),
                 "C++ CPU affinity binding supports Linux CPU IDs below 2**20")
    scalar = {}
    for node in plan.region.nodes:
        if node.op in ("input", "slice"):
            scalar[node.id] = False
        elif node.op == "const":
            scalar[node.id] = True
        else:
            scalar[node.id] = all(scalar[operand] for operand in node.inputs)
            _require(not scalar[node.id], "scalar-only operations must be folded before C++ lowering")


def admit_loop(loop, strategy, execution_plan):
    """Return the replayed strategy and exact bound body plan for lowering.

    The body is one shared one-array DAG, not an ordinary one-step substitute
    for the original loop. Source-file I/O is separate in check_loop_sources.
    """
    from scar.optimization.hardware import bind_cpu_placement
    from scar.optimization.loop_strategy import replay_loop_strategy

    strategy, expected = replay_loop_strategy(loop, strategy)
    _require(strategy["selected"]["kind"] == "TEMPORAL_FUSION" and
             strategy["capability"] == "EXACT_ELEMENTWISE_TEMPORAL_FUSION",
             "loop backend requires the selected exact elementwise temporal profile")
    actual = (read_plan(execution_plan) if isinstance(execution_plan, (str, Path))
              else plan_from_dict(plan_document(execution_plan)))
    _require(actual.resource_binding is not None, "loop execution requires an explicit recorded CPU binding")
    _require(sys.platform == "linux", "CPU-bound loop execution requires Linux")
    _component(expected)
    _component(actual)
    _require(plan_document(actual) == plan_document(bind_cpu_placement(expected, actual.resource_binding)),
             "loop body plan differs from exact replay or admitted CPU binding")
    _require(actual.region.source == loop.body.source and actual.region.parameters == loop.body.parameters
             and len(actual.region.parameters) == 1 and
             all(node.op in ("input", "const", "add", "sub", "mul", "div", "neg")
                 for node in actual.region.nodes),
             "loop body must be the shared one-array elementwise DAG")
    count = strategy["trip_count"]
    _require(type(count) is int and 0 < count <= 2**63 - 1,
             "loop count must be an exact positive int64 specialization")
    memory = strategy["selected_memory_plan"]
    shape = memory["input_shapes"][loop.state_parameter]
    _require(set(memory["input_shapes"]) == {loop.state_parameter} and len(shape) == 2 and
             all(type(n) is int and 0 <= n <= sys.maxsize // 8 for n in shape),
             "loop state shape exceeds host bounds")
    size = shape[0] * shape[1] * 8
    budget = actual.cost.get("memory_budget_bytes")
    _require(memory["output_shape"] == shape and memory["final_bytes"] == size <= sys.maxsize and
             memory["total_workspace_bytes"] == 0 and memory["peak_allocation_bytes"] == size and
             type(budget) is int and size <= budget <= memory["memory_budget_bytes"] <= sys.maxsize,
             "loop requires a shape-preserving fresh final within the narrower payload budget")
    return strategy, actual


# Extract one original range loop and lexical array carry without unrolling.
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


def discover_graph_loops(catalog,source=None,*,_discovery=None):
    discovery = _discovery_for(catalog, _discovery)
    view, resolver = discovery.view, discovery.resolver
    pipelines, _ = discover_graph_pipelines(view, source, _discovery=discovery)
    bodies = {_source_occurrence(body.source): body for body in pipelines}
    units=tuple(view.catalog.fact_store.rows) if source is None else tuple(sorted(
        view.catalog.units_for_source(str(Path(source).absolute())) or ()))
    if source is not None and not units:return [],[dict(path=str(Path(source).absolute()),code='SOURCE_NOT_IN_CATALOG')]
    loops, rejected = [], []
    for unit in units:
        original=view.unit(unit);functions=original.facts.functions
        resolver.prepare_many(Body(unit,fn.template,fn.body) for fn in functions)
        model,syntax,_,recipe,digest=discovery.identity(unit)
        selected={fn.template:bodies[unit,fn.template.occurrence] for fn in functions if (unit,fn.template.occurrence) in bodies}
        globals=_global_functions(original.facts,selected)
        for ordinal,function in enumerate(functions):
            if len(function.parameter_names)!=2:continue
            location = discovery.function_identity(unit, function)
            code, span = location.code, location.span
            try:
                if code is None:_reject('LOOP_CODE_IDENTITY','unique original loop CodeID unavailable')
                state,trip,body,guard,origins=_LoopLowerer(original,function,resolver,globals,syntax=syntax).lower()
                key = location.key
                identity=dict(kind='source_identity',method='COMPILE_ONLY_RETAINED_F0',unit=unit,
                    occurrence=function.template.occurrence,span=span,recipe=recipe,code_key=key,
                    semantic_authority='ORIGINAL_F4_PRODUCTION_PLANS_AND_UNIT_FACTS')
                reference=SourceRef(model['path'],digest,code.co_qualname,span['line'],span['column'],key,(identity,))
                plan=ArrayLoopPlan('graph-array-loop:'+sha256((unit+'\0'+function.body.wire+'\0'+key).encode()).hexdigest()[:24],
                    reference,function.parameter_names,state,trip,body,guard,origins)
                loops.append(validate_loop(plan))
            except GraphRegionRejected as error:
                rejected.append(dict(unit=unit,function=ordinal,path=model['path'],qualname=code.co_qualname
                    if code is not None else location.name,code=error.code,reason=error.code,detail=error.detail,origin=error.origin))
    return loops,rejected


def read_array_loop_proof(reference, *, catalog):
    """Independently recover the original F4 loop and all its component calls."""
    from scar.ir.execution_recipe import read_artifact
    from scar.optimization.resource_solver import semantic_sha256

    _require(reference['schema'] == SCHEMA, 'exact original array loop proof required')
    saved = read_artifact(reference)
    retained = loop_from_dict(saved)
    check_loop_sources(retained)
    discovery = _discovery_for(catalog, None)
    candidates, _ = discover_graph_loops(catalog, source=retained.source.path, _discovery=discovery)
    discovery.validate()
    matches = [loop for loop in candidates if loop.id == retained.id]
    _require(len(matches) == 1 and loop_digest(matches[0]) == semantic_sha256(saved),
             'array loop differs from original F4 loop, pipeline and callee replay')
    read_artifact(reference, json_document=False)
    return matches[0]


# Original-F4 proof of a single array carry with invariant external inputs.
#
# The carry role comes from the original lexical write target, not its name or
# argument position. The leaf's own value DAG remains the existing ArrayRegion.
# This source proof does not discharge ndarray, global identity or concurrency
# guards and is not an arbitrary boolean credential for loop lowering.
@dataclass(frozen=True)
class InvariantLoopProof:
    id: str
    source: SourceRef
    parameters: tuple[str, ...]
    body: ArrayRegion
    carry_parameter: str
    trip_count_parameter: str
    parameter_map: dict[str, str]
    leaf_global_name: str
    origins: tuple[dict, ...]

    @property
    def carry_outer_parameter(self):
        return self.parameter_map[self.carry_parameter]


@dataclass(frozen=True)
class _PendingLeafResult:
    outer_parameters: tuple[int, ...]


class _InvariantLoopLowerer(_LoopLowerer):
    def binding(self, context, step, args):
        if (step.instruction.action is P.BindingActionKind.WRITE and args
                and type(args[0]) is _PendingLeafResult):
            if self.phase != 'BODY' or self.carry_writes:
                _reject('INVARIANT_LOOP_EXTRA_CARRY_WRITE', 'one body result carry write required')
            port = context.value(step.arguments[0]).port
            namespaces = tuple(dict.fromkeys(self.names.get((port, True), ())))
            if len(namespaces) != 1 or namespaces[0].kind is not F.NamespaceKind.LEXICAL:
                _reject('INVARIANT_LOOP_EXTERNAL_WRITE', 'body result must update one original lexical binding')
            target = namespaces[0]
            positions = [i for i, namespace in self.parameter_namespaces.items() if namespace == target]
            if len(positions) != 1 or positions[0] == self.roles[1]:
                _reject('INVARIANT_LOOP_CARRY_TARGET', 'body result must update an original non-trip parameter')
            carry = positions[0]; actuals = args[0].outer_parameters
            if actuals.count(carry) != 1:
                _reject('INVARIANT_LOOP_CARRY_MULTIPLICITY', 'exactly one leaf formal must receive the carried parameter')
            self.roles = carry, self.roles[1]
            self.parameter_map_indices = actuals
            self.carry_formal_index = actuals.index(carry)
            self.carry_writes += 1; self.locals[target] = _Carry(carry, True)
            self.origins.append(dict(_origin(context, step.region, step), role='lexical_carry_write',
                binding=encode(target), outer_parameter_index=carry, leaf_formal_index=self.carry_formal_index))
            return (_WORLD,)
        return super().binding(context, step, args)

    def action(self, context, step, args, origin):
        if step.instruction.action is not P.LanguageActionKind.CALL:
            _reject('INVARIANT_LOOP_EXTRA_ACTION', 'only exact range and admitted array leaf CALL supported', origin)
        fact = self.call_facts.get(context.result(step).port)
        if (fact is None or fact.keywords or fact.positional_pack is not None or fact.keyword_pack is not None
                or len(args) != len(fact.positional) + 2 or args[-1] is not _WORLD):
            _reject('INVARIANT_LOOP_CALL_ARGUMENTS', 'original fixed positional CALL arguments required', origin)
        actual = tuple(context.value(value).port for value in step.arguments)
        expected = tuple(source_port(self.facts, i).port for i in (fact.callee, *fact.positional, fact.world))
        if (actual != expected or context.result(step, 1).port != source_port(self.facts, fact.result_world).port):
            _reject('INVARIANT_LOOP_CALL_ABI', 'callee/arguments/results/world differ from original F4', origin)
        if args[0] is _RANGE:
            if self.phase != 'PRE' or self.range_calls or len(args) != 3 or type(args[1]) is not _Parameter:
                _reject('INVARIANT_LOOP_RANGE', 'one range(original trip parameter) required', origin)
            self.range_calls += 1
            self.origins.append(dict(origin, role='builtin_range', trip_parameter_index=args[1].index,
                runtime_obligation='GLOBAL_THEN_BUILTINS_LOOKUP_IS_EXACT_BUILTIN_RANGE'))
            return (_Range(args[1].index), _WORLD)
        if self.phase != 'BODY' or self.body_calls or type(args[0]) is not _Callee:
            _reject('INVARIANT_LOOP_BODY_CALL', 'one original direct global array leaf CALL per iteration required', origin)
        callee = args[0]; body = callee.plan; actuals = args[1:-1]
        if (type(body) is not ArrayRegion or len(body.parameters) < 2 or len(actuals) != len(body.parameters)
                or any(type(value) is not _Parameter or value.index == self.roles[1] for value in actuals)):
            _reject('INVARIANT_LOOP_ARRAY_ARGUMENTS', 'leaf arrays must come directly from non-trip outer parameters', origin)
        signature = callee.function.environment.parameters
        if (len(signature) != len(body.parameters) or any(parameter.kind not in
                (P.DeferredParameterKind.POSITIONAL_ONLY, P.DeferredParameterKind.POSITIONAL_OR_KEYWORD)
                or parameter.default_source is not None for parameter in signature)):
            _reject('INVARIANT_LOOP_LEAF_SIGNATURE', 'leaf has unsupported keyword/default/variadic parameters', origin)
        leaf_context = self.resolver.plan(self.source.unit, callee.function.template)
        if any(type(row.instruction) is P.ApplyDecorator for row in leaf_context.plan.steps):
            _reject('INVARIANT_LOOP_LEAF_DECORATOR', 'decorated array leaf is unsupported', origin)
        self.body_calls += 1; self.selected_body = callee
        self.origins.append(dict(origin, role='invariant_array_leaf_call', leaf_region_id=body.id,
            call_callee=encode(source_port(self.facts, fact.callee)),
            call_result=encode(source_port(self.facts, fact.result)),
            call_inputs=[encode(source_port(self.facts, i)) for i in fact.positional],
            outer_parameter_indices=[value.index for value in actuals]))
        return (_PendingLeafResult(tuple(value.index for value in actuals)), _WORLD)

    def control(self, context, step, args, origin):
        ins = step.instruction
        if (self.phase != 'PRE' or self.loops or type(ins.condition) is not P.IterationCondition
                or ins.invariant_count != 1 or ins.condition.world_parameter != 1
                or len(args) != 2 or type(args[0]) is not _Iterator or args[1] is not _WORLD):
            _reject('INVARIANT_LOOP_CONTROL', 'one original iterator/world range loop required', origin)
        self.loops += 1; self.roles = None, args[0].range.trip
        if ins.condition.advance.arguments != tuple(P.ProductionParameter(step.region, i) for i in range(2)):
            _reject('INVARIANT_LOOP_ADVANCE', 'advance must use original iterator/world parameters', origin)
        exhausted = context.regions[ins.condition.exhausted]
        steps = [context.steps[s] for s in exhausted.steps]
        if (len(steps) != 2 or type(steps[0].instruction) is not P.ApplyPythonProtocol
                or steps[0].instruction.protocol.value != 'MATCH_STOP_ITERATION'
                or steps[0].arguments != tuple(P.ProductionParameter(exhausted.id, i) for i in range(2))
                or type(steps[1].instruction) is not P.TransferCompletion
                or steps[1].instruction.kind is not C.FALLTHROUGH
                or steps[1].arguments != (P.ProductionResult(steps[0].id, 0),)):
            _reject('INVARIANT_LOOP_EXHAUSTION', 'original StopIteration classifier required', origin)
        if ins.otherwise is not None:
            otherwise = context.regions[ins.otherwise]; items = [context.steps[s] for s in otherwise.steps]
            if (len(items) != 1 or type(items[0].instruction) is not P.TransferCompletion
                    or items[0].instruction.kind is not C.FALLTHROUGH
                    or items[0].arguments != (P.ProductionParameter(otherwise.id, 1),)):
                _reject('INVARIANT_LOOP_ELSE', 'nonempty loop else is unsupported', origin)
        self.phase = 'CONDITION'; kind, condition = self.run(context, ins.condition.advance.region, args)
        if kind is not C.FALLTHROUGH or condition != (_ITEM, _WORLD) or self.next_calls != 1:
            _reject('INVARIANT_LOOP_CONDITION', 'advance must fetch only the original next range item', origin)
        self.phase = 'BODY'; kind, result = self.run(context, ins.body, (*args, _ITEM))
        carry = self.roles[0]
        if (carry is None or kind is not C.FALLTHROUGH or result != (_WORLD,)
                or self.body_calls != 1 or self.carry_writes != 1 or self.item_writes != 1
                or self.locals[self.parameter_namespaces[carry]] != _Carry(carry, True)):
            _reject('INVARIANT_LOOP_CARRY', 'one ignored loop item and one original leaf-result carry write required', origin)
        self.phase = 'POST'
        self.origins.append(dict(origin, role='single_shared_invariant_loop',
            iterator_parameter=encode(context.gp(context.emitter.port(step.region, G.PortKind.PARAMETER, 0))),
            iteration_condition=encode(ins.condition), body_region=ins.body.wire,
            lexical_carry_binding=encode(self.parameter_namespaces[carry]),
            backedge=dict(from_region=ins.body.wire, to_region=step.region.wire,
                world_parameter_index=1, explicit_array_phi=False),
            body_materializations=1, loop_unrolled=False))
        return (_WORLD,)

    def lower(self):
        function = self.function
        if (len(function.parameter_names) < 3 or function.body_scope is None
                or function.environment.kind is not P.DeferredBodyKind.FUNCTION
                or function.environment.captures or function.environment.owned_cells
                or len(function.environment.parameters) != len(function.parameter_names)
                or any(parameter.kind not in (P.DeferredParameterKind.POSITIONAL_ONLY,
                    P.DeferredParameterKind.POSITIONAL_OR_KEYWORD) or parameter.default_source is not None
                    for parameter in function.environment.parameters)):
            _reject('INVARIANT_LOOP_SIGNATURE', 'ordinary fixed explicit parameters without captures/defaults required')
        context = self.resolver.plan(self.source.unit, function.template)
        if any(type(step.instruction) is P.ApplyDecorator for step in context.plan.steps):
            _reject('INVARIANT_LOOP_DECORATOR', 'decorated outer loop function unsupported')
        for index, _ in enumerate(function.parameter_names):
            formal = source_port(self.facts, function.formals[index+1]); names = self.names.get((formal.port, True), ())
            if len(names) != 1 or names[0].kind is not F.NamespaceKind.LEXICAL:
                _reject('INVARIANT_LOOP_PARAMETER_BINDING', 'formal lacks one original lexical binding')
            self.parameter_namespaces[index] = names[0]; self.locals[names[0]] = _Parameter(index)
        kind, result = self.run(context, function.body,
            (_ENV, *(_Parameter(i) for i in range(len(function.parameter_names))), _WORLD))
        if (self.loops != 1 or self.range_calls != 1 or self.iterator_gets != 1 or self.selected_body is None
                or self.roles[0] is None or kind is not C.RETURN or result != (_Carry(self.roles[0], True), _WORLD)):
            _reject('INVARIANT_LOOP_RETURN', 'one shared range loop followed by original carry return required')
        carry, trip = self.roles
        expected = Counter({('PRE', self.parameter_namespaces[trip]): 1,
            ('POST', self.parameter_namespaces[carry]): 1})
        expected.update(('BODY', self.parameter_namespaces[i]) for i in self.parameter_map_indices)
        if self.reads != expected:
            _reject('INVARIANT_LOOP_EXTRA_READ', 'only original trip, leaf actuals and final carry reads supported')
        self.origins.append(dict(kind='invariant_loop_runtime_obligations',
            builtin_range='EXACT_BUILTIN_RANGE', trip_count='EXACT_POSITIVE_INT_OR_ORIGINAL_FALLBACK',
            leaf_lookup='OUTER_FUNCTION_GLOBALS_EXACT_SOURCE_CODE_AND_SIGNATURE',
            arrays='EXACT_FINITE_FLOAT64_NDARRAYS_NO_FOREIGN_WRITERS',
            reference_stability_does_not_prove_content_invariance=True))
        body = self.selected_body.plan
        mapping = dict(zip(body.parameters, (function.parameter_names[i] for i in self.parameter_map_indices)))
        return body, body.parameters[self.carry_formal_index], function.parameter_names[trip], mapping, self.selected_body.name, tuple(self.origins)


def discover_invariant_loops(catalog, source=None, *, _discovery=None):
    """Discover original module-level range/carry/leaf bindings without execution."""
    discovery = _discovery_for(catalog, _discovery)
    view, resolver = discovery.view, discovery.resolver
    leaves, _ = discover_graph_regions(view, source, _discovery=discovery)
    by_source = {_source_occurrence(leaf.source): leaf for leaf in leaves}
    units = tuple(view.catalog.fact_store.rows) if source is None else tuple(sorted(
        view.catalog.units_for_source(str(Path(source).absolute())) or ()))
    if source is not None and not units:
        return [], [dict(path=str(Path(source).absolute()), code='SOURCE_NOT_IN_CATALOG')]
    loops, rejected = [], []
    for unit in units:
        original = view.unit(unit); functions = original.facts.functions
        resolver.prepare_many(Body(unit, fn.template, fn.body) for fn in functions)
        try: model, syntax, _, recipe, digest = discovery.identity(unit)
        except (ValueError, KeyError, SyntaxError, OSError) as error:
            rejected.append(dict(unit=unit, code='INVARIANT_LOOP_SOURCE_IDENTITY', detail=str(error))); continue
        selected = {fn.template:by_source[unit, fn.template.occurrence] for fn in functions
            if (unit, fn.template.occurrence) in by_source}
        globals = _global_functions(original.facts, selected)
        module_functions = {candidate.function.template for candidates in
            _global_functions(original.facts, {fn.template:fn for fn in functions}).values() for candidate in candidates}
        for ordinal, function in enumerate(functions):
            if function.template not in module_functions or function.template in selected: continue
            location = discovery.function_identity(unit, function)
            code, span = location.code, location.span
            try:
                if code is None: _reject('INVARIANT_LOOP_CODE_IDENTITY', 'unique original CodeID unavailable')
                body, carry, trip, mapping, global_name, origins = _InvariantLoopLowerer(
                    original, function, resolver, globals, syntax=syntax).lower()
                key = location.key
                identity = dict(kind='source_identity', method='COMPILE_ONLY_RETAINED_F0', unit=unit,
                    occurrence=function.template.occurrence, span=span, recipe=recipe, code_key=key,
                    semantic_authority='ORIGINAL_F4_PRODUCTION_PLANS_AND_UNIT_FACTS')
                reference = SourceRef(model['path'], digest, code.co_qualname, span['line'], span['column'], key, (identity,))
                identifier = 'invariant-array-loop:' + sha256((unit+'\0'+function.body.wire+'\0'+key).encode()).hexdigest()[:24]
                loops.append(InvariantLoopProof(identifier, reference, function.parameter_names,
                    body, carry, trip, mapping, global_name, origins))
            except GraphRegionRejected as error:
                rejected.append(dict(unit=unit, function=ordinal, path=model['path'],
                    qualname=code.co_qualname if code is not None else location.name,
                    code=error.code, detail=error.detail, origin=error.origin))
    return loops, rejected


PROOF_SCHEMA = 'scar.invariant-loop-proof'


def validate_invariant_loop_proof(proof):
    """Validate the retained shape/bytes, never replace original F4 replay."""
    from scar.compiler.model import validate_region
    if type(proof) is not InvariantLoopProof:
        raise ValueError('typed original invariant loop proof required')
    validate_region(proof.body)
    if (type(proof.id) is not str or not proof.id.startswith('invariant-array-loop:')
            or type(proof.parameters) is not tuple or len(proof.parameters) < 3
            or len(set(proof.parameters)) != len(proof.parameters)
            or any(type(name) is not str or not name.isidentifier() for name in proof.parameters)
            or proof.trip_count_parameter not in proof.parameters
            or proof.carry_parameter not in proof.body.parameters
            or type(proof.parameter_map) is not dict or set(proof.parameter_map) != set(proof.body.parameters)
            or any(value not in proof.parameters or value == proof.trip_count_parameter for value in proof.parameter_map.values())
            or type(proof.leaf_global_name) is not str or not proof.leaf_global_name.isidentifier()):
        raise ValueError('original outer/leaf parameter mapping required')
    if list(proof.parameter_map.values()).count(proof.carry_outer_parameter) != 1:
        raise ValueError('carried outer parameter must map to exactly one leaf formal')
    identities=[]
    for source in (proof.source, proof.body.source):
        if (type(source) is not SourceRef or not Path(source.path).is_absolute()
                or re.fullmatch('[0-9a-f]{64}', source.sha256) is None
                or re.fullmatch('pycode-v2:[0-9a-f]{64}', source.code_key) is None
                or sha256(Path(source.path).read_bytes()).hexdigest() != source.sha256):
            raise ValueError('original source bytes/code identity changed')
        candidates=[row for row in source.origins if row.get('kind')=='source_identity'
            and row.get('method')=='COMPILE_ONLY_RETAINED_F0'
            and row.get('semantic_authority')=='ORIGINAL_F4_PRODUCTION_PLANS_AND_UNIT_FACTS'
            and row.get('code_key')==source.code_key]
        if len(candidates)!=1:raise ValueError('original F0 identity with F4 semantic provenance required')
        identities.append(candidates[0])
    if (proof.source.path,proof.source.sha256,identities[0]['unit']) != (
            proof.body.source.path,proof.body.source.sha256,identities[1]['unit']):
        raise ValueError('loop and direct global leaf must share the original unit')
    witnesses={}
    for role in ('builtin_range','lexical_carry_write','invariant_array_leaf_call','single_shared_invariant_loop'):
        candidates=[row for row in proof.origins if row.get('role')==role
            and row.get('kind')=='original_f4' and row.get('unit')==identities[0]['unit']]
        if len(candidates)!=1:raise ValueError('missing unique original F4 loop witness: '+role)
        witnesses[role]=candidates[0]
    carry=proof.parameters.index(proof.carry_outer_parameter)
    actuals=[proof.parameters.index(proof.parameter_map[name]) for name in proof.body.parameters]
    if (witnesses['builtin_range']['trip_parameter_index']!=proof.parameters.index(proof.trip_count_parameter)
            or witnesses['lexical_carry_write']['outer_parameter_index']!=carry
            or witnesses['lexical_carry_write']['leaf_formal_index']!=proof.body.parameters.index(proof.carry_parameter)
            or witnesses['invariant_array_leaf_call']['outer_parameter_indices']!=actuals
            or witnesses['invariant_array_leaf_call']['leaf_region_id']!=proof.body.id
            or len(witnesses['invariant_array_leaf_call']['call_inputs'])!=len(proof.body.parameters)
            or witnesses['single_shared_invariant_loop']['loop_unrolled'] is not False):
        raise ValueError('retained F4 witnesses differ from the declared loop interface')
    return proof


def invariant_loop_proof_document(proof):
    return dict(schema=PROOF_SCHEMA,schema_version=1,**asdict(validate_invariant_loop_proof(proof)))


def write_invariant_loop_proof(proof,path):
    """Store the original body once; the file itself grants no permission."""
    path=Path(path).absolute();document=invariant_loop_proof_document(proof)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(document,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    from scar.ir.execution_recipe import artifact_reference
    return artifact_reference(path,PROOF_SCHEMA)


def _replay_invariant_loop_proof(path,*,catalog):
    from scar.ir.execution_recipe import open_artifact,read_artifact
    reference,document=open_artifact(path,PROOF_SCHEMA)
    fields={'schema','schema_version',*InvariantLoopProof.__dataclass_fields__}
    if set(document)!=fields or document['schema_version']!=1 or type(document['schema_version']) is not int:
        raise ValueError('unsupported original invariant loop proof fields/version')
    data={name:document[name] for name in InvariantLoopProof.__dataclass_fields__}
    source=dict(data['source']);source['origins']=tuple(source.get('origins',()))
    data.update(source=SourceRef(**source),body=region_from_dict(data['body']),
        parameters=tuple(data['parameters']),origins=tuple(data['origins']))
    proof=validate_invariant_loop_proof(InvariantLoopProof(**data))
    original,_=discover_invariant_loops(catalog,source=proof.source.path)
    canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    matches=[value for value in original if value.id==proof.id and
        canonical(invariant_loop_proof_document(value))==canonical(document)]
    if len(matches)!=1:raise ValueError('invariant loop proof differs from original F4 rediscovery')
    read_artifact(reference,json_document=False)
    return matches[0],reference


def read_invariant_loop_proof(path,*,catalog):
    """Replay the original loop through the admitted F4 catalog and its source bytes."""
    proof,_=_replay_invariant_loop_proof(path,catalog=catalog)
    return proof


def canonical_invariant_motion_interface(proof):
    """Canonical source boundary, excluding lowering/runtime specializations.

    This is only a projection of a typed proof. Its caller still owns original
    F4 rediscovery; a serialized proof or this dictionary grants no authority.
    """
    if type(proof) is not InvariantLoopProof:
        raise TypeError('original typed invariant loop proof required')
    return dict(carry_parameter=proof.carry_parameter,
        readonly_parameters=[name for name in proof.body.parameters if name!=proof.carry_parameter],
        original_f4_proof=asdict(proof))


def read_invariant_loop_motion(path,*,catalog):
    """Join motion facts to one independently replayed original source loop.

    The ArrayRegion query alone has no source authority.  This adapter first
    uses the existing full F4 rediscovery, then retains the original CALL,
    lexical write and loop-control witnesses.  Array contents and live
    callable dispatch remain runtime obligations; a stable binding does not
    establish an immutable referent.
    """
    from scar.analysis.array_motion_legality import derive_array_motion_and_legality
    from scar.analysis.source_motion_window import project_loop_source_motion
    from scar.optimization.resource_solver import semantic_sha256

    view=catalog if isinstance(catalog,ProgramView) else ProgramView(catalog)
    proof,reference=_replay_invariant_loop_proof(path,catalog=view)
    interface=canonical_invariant_motion_interface(proof)
    roles={row['role']:row for row in proof.origins if row.get('kind')=='original_f4'
        and row.get('role') in ('builtin_range','lexical_carry_write',
            'invariant_array_leaf_call','single_shared_invariant_loop')}
    window,routes=project_loop_source_motion(view,proof,
        roles['single_shared_invariant_loop'],roles['invariant_array_leaf_call'])
    motion,legality=derive_array_motion_and_legality(proof.body,carry_parameter=proof.carry_parameter,
        loop_interface=interface,source_window=window)
    report=dict(schema='scar.source-loop-motion',schema_version=2,
        proof=reference,proof_id=proof.id,
        proof_sha256=semantic_sha256(asdict(proof)),body_sha256=motion['body_sha256'],
        source_identity=dict(path=proof.source.path,sha256=proof.source.sha256,
            code_key=proof.source.code_key),
        array_motion=motion,
        source_window=window,movement_legality=legality,
        canonical_interface_sha256=semantic_sha256(interface),
        parameter_routes=routes,
        source_witnesses=roles,
        source_facts=dict(authority='FULL_ORIGINAL_F4_REDISCOVERY',
            body_profile='ONE_RANGE_BODY_ONE_ADMITTED_ARRAY_CALL_ONE_LEXICAL_CARRY_WRITE',
            source_actions_enumerated=True,opaque_source_actions=False,
            input_writes='NONE_WITHIN_REPLAYED_ARRAY_PROFILE',
            lexical_carry_write_is_input_storage_write=False,
            positive_iteration_call='ONE_PER_ENTERED_BODY',
            zero_trip='NO_BODY_EVALUATION_AND_ORIGINAL_CARRY_ALIAS',
            bodies_shared=True,iterations_unrolled=False),
        runtime_obligations=motion['conditional_obligations'],
        runtime_obligations_discharged=False,source_replacement_authorized=False,
        cost_qualified=False,selected_for_use=False)
    report['id']='source-loop-motion:'+semantic_sha256(report)
    return proof,report


def verify_invariant_loop_motion(report,*,catalog):
    """Cold source replay; no target execution or trusted proof markers."""
    from scar.ir.execution_recipe import read_artifact
    from scar.optimization.resource_solver import semantic_sha256

    if (type(report) is not dict or report.get('schema')!='scar.source-loop-motion'
            or type(report.get('schema_version')) is not int or report['schema_version']!=2):
        raise ValueError('current source loop motion report required')
    read_artifact(report['proof'])
    proof,expected=read_invariant_loop_motion(report['proof']['uri'],catalog=catalog)
    if semantic_sha256(expected)!=semantic_sha256(report):
        raise ValueError('source loop motion differs from original F4 replay')
    return proof,expected


def loop_dependency_witnesses(proof):
    """Return the original control and CALL witnesses for an admitted loop."""
    if type(proof) is ArrayLoopPlan:
        control = next(row for row in proof.origins if row.get('role') == 'single_shared_loop')
        call = next(row for row in proof.origins if row.get('role') == 'body_pipeline_call')
        result, = (value for value in call['results'] if value['port']['completion'] == 'FALLTHROUGH'
                   and value['port']['ordinal'] == 0)
        return control, dict(call, call_callee=call['arguments'][0], call_result=result)
    control = next(row for row in proof.origins if row.get('role') == 'single_shared_invariant_loop')
    call = next(row for row in proof.origins if row.get('role') == 'invariant_array_leaf_call')
    return control, call


def validate_array_loop_links(proof, spec, loopsref):
    _require(type(proof) is ArrayLoopPlan, 'original F4 array loop required')
    return _validate_loop_links(proof, spec, loopsref)


def validate_invariant_loop_links(proof, spec, loopsref):
    _require(type(proof) is InvariantLoopProof, 'original F4 invariant loop proof required')
    return _validate_loop_links(proof, spec, loopsref)


def _validate_loop_links(proof,spec,loopsref):
    """Join the retained loop/CALL facts to an already F4-replayed proof.

    No catalog is opened and no source graph is rebuilt here. Both FactRefs
    must point into the same byte-checked loops publication and their original
    ports must equal the proof's control/CALL witnesses, including world ABI.
    """
    from scar.ir.execution_recipe import read_artifact
    if spec.get('id')!=proof.id:
        raise ValueError('loop specification differs from its original proof ID')
    if spec.get('source_code_key')!=proof.source.code_key or spec.get('loop_link_frontier'):
        raise ValueError('original source code identity and closed loop links required')
    document=read_artifact(loopsref)
    if document.get('schema')!='scar.source-loop-dependencies' or document.get('schema_version')!=1:
        raise ValueError('original source loop dependency publication required')
    links=spec.get('loop_links')
    if type(links) is not list or len(links)!=1:
        raise ValueError('one original F4 loop/CALL relationship required')
    link=links[0]
    if (set(link)!={'kind','loop','call'} or link['kind']!='EXACT_ORIGINAL_F4_LOOP_HEADER_AND_CALL_RESULT'
            or any(type(link[name]) is not dict or set(link[name])!={'artifact','locator'}
                or link[name]['artifact']!=loopsref for name in ('loop','call'))):
        raise ValueError('loop and CALL FactRefs must use the retained parent artifact')
    location=link['loop']['locator'];call_location=link['call']['locator']
    match=re.fullmatch(r'#/functions/(0|[1-9][0-9]*)/loops/(0|[1-9][0-9]*)',location)
    call_match=re.fullmatch(re.escape(location)+r'/body_calls/(0|[1-9][0-9]*)',call_location)
    if match is None or call_match is None:raise ValueError('canonical original loop and enclosed CALL locators required')
    try:
        function=document['functions'][int(match[1])]
        loop=function['loops'][int(match[2])];call=loop['body_calls'][int(call_match[1])]
        control, original = loop_dependency_witnesses(proof)
        identity=next(row for row in proof.source.origins if row.get('kind')=='source_identity')
        header=loop['header'];port=header['port']
        normal_world=[value for value in original['results'] if value['port']['completion']=='FALLTHROUGH'
            and value['port']['ordinal']==1]
        if (function['body']['unit']!=control['unit'] or function['body']['template']['occurrence']!=identity['occurrence']
                or header['unit']!=control['unit'] or port['template']!=control['template']
                or port['owner']['wire']!=control['step'] or port['kind']!='ENTER'
                or loop['body']['unit']!=control['unit'] or loop['body']['template']!=control['template']
                or loop['body']['region']['wire']!=control['body_region']
                or call['result']!=original['call_result'] or call['callee']!=original['call_callee']
                or call['positional']!=original['call_inputs'] or call['keywords']
                or call['positional_pack'] is not None or call['keyword_pack'] is not None
                or call['world_before']!=original['arguments'][-1]
                or len(normal_world)!=1 or call['world_after']!=normal_world[0]
                or len(loop['slots'])!=2 or loop['slots'][0]['formal']!=control['iterator_parameter']
                or [slot['role'] for slot in loop['slots']]!=['INVARIANT_PREFIX','CARRIED_SUFFIX']
                or not loop['control_backedges'] or not loop['slots'][1]['backedges']):
            raise ValueError('retained loop/CALL facts differ from original F4 control, operands or world ports')
    except (KeyError,IndexError,StopIteration,TypeError) as error:
        raise ValueError('original loop/CALL FactRef locator or witness is absent') from error
    return links


__all__ = [
    'ArrayLoopPlan',
    'validate_loop',
    'document',
    'loop_from_dict',
    'loop_digest',
    'digest',
    'read_loop',
    'write_loop',
    'read_array_loop_proof',
    'validate_array_loop_links',
    'loop_dependency_witnesses',
    'admit_loop',
    'check_loop_sources',
    'SCHEMA',
    'PROFILE',
    'InvariantLoopProof',
    'discover_invariant_loops',
    'validate_invariant_loop_proof',
    'invariant_loop_proof_document',
    'write_invariant_loop_proof',
    'read_invariant_loop_proof',
    'validate_invariant_loop_links',
    'PROOF_SCHEMA',
    'discover_graph_loops',
    'canonical_invariant_motion_interface',
    'read_invariant_loop_motion',
    'verify_invariant_loop_motion',
]

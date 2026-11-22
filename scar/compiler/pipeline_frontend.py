"""Pipeline frontend."""

from collections import defaultdict
from dataclasses import dataclass, asdict, replace
from hashlib import sha256
from pathlib import Path
from scar.analysis.program_view import ProgramView
from scar.analysis.source_body_interfaces import BodyInterfaceResolver
from scar.analysis.source_references import source_port
from scar.compiler.graph_frontend import (
    discover_graph_regions,
    _GraphLowerer,
    GraphRegionRejected,
    _reject,
    _origin,
    _source_identity,
    _metadata,
    _WORLD,
    _ENV,
)
from scar.compiler.model import (
    SourceRef,
    ExecutionPlan,
    document as leaf_document,
    plan_from_dict,
)
from scar.ir.program import source_productions as P, normal_graph as F
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.normal_graph import Body
from scar.ir.record_codec import encode
from scar.optimization.optimizer import optimize_region
from scar.trace.code_snapshot import code_key
from copy import deepcopy
import json


# A guarded, straight-line composition of shared original array-region plans.
SCHEMA='scar.array-pipeline-plan'
PROFILE='cpu-array-pipeline-v1'
LOWERING_CONTEXT='PIPELINE_NATIVE_HOST_COMPONENT'


@dataclass(frozen=True)
class CalleeGuard:
    source: SourceRef
    lookup_path: tuple[str,...]
    lookup_root: str='PIPELINE_FUNCTION_GLOBALS'


@dataclass(frozen=True)
class RegionCall:
    id: str
    plan: ExecutionPlan
    inputs: tuple[str,...]
    output: str
    callee: CalleeGuard
    origins: tuple[dict,...]=()


@dataclass(frozen=True)
class PipelinePlan:
    id: str
    source: SourceRef
    parameters: tuple[str,...]
    input_values: tuple[str,...]
    calls: tuple[RegionCall,...]
    output: str
    schema_version: int=1
    profile: str=PROFILE
    assumptions: tuple[str,...]=('EXACT_NUMPY_NDARRAY_FLOAT64_CPU_2D',
        'ACTUAL_GLOBAL_CALLEE_CODE_IDS_MATCH','FRESH_LEAF_OUTPUTS','NO_INPUT_WRITES')
    frontiers: tuple[str,...]=()


def prepare_pipeline_leaf(plan):
    """Admit a callable component without claiming standalone profitability."""
    from scar.compiler.model import validate_plan
    validate_plan(plan)
    if (plan.resource_binding is not None or plan.proof.get('supported')is not True
            or plan.proof.get('legal')is not True or plan.proof.get('frontiers')
            or plan.region.frontiers or plan.buffer_policy!='fresh_output'):
        raise ValueError('pipeline component requires an unbound supported/legal closed-profile leaf')
    budget=plan.cost.get('memory_budget_bytes')
    if budget is not None and (type(budget)is not int or budget<8):
        raise ValueError('pipeline component cannot bypass its output memory budget')
    prepared=plan.proof.get('lowering_context')==LOWERING_CONTEXT
    eligible=plan.proof.get('standalone_lowering_eligible') if prepared else plan.proof.get('lowering_eligible')
    decision=plan.proof.get('standalone_decision') if prepared else plan.proof.get('decision',plan.cost.get('decision'))
    if type(eligible)is not bool:
        raise ValueError('standalone lowering eligibility must be explicit')
    if not eligible and (plan.cost.get('reason')!=['no_full_array_intermediate_to_eliminate']
            or decision!='not_profitable' or plan.cost.get('decision')!='not_profitable'):
        raise ValueError('pipeline component cannot bypass unknown or rejected standalone lowering reasons')
    if prepared:
        if plan.proof.get('lowering_eligible')is not True or plan.proof.get('selected')is not False:
            raise ValueError('pipeline component remains an eligible, unselected candidate')
        return plan
    proof=deepcopy(plan.proof)
    proof.update(lowering_context=LOWERING_CONTEXT,standalone_lowering_eligible=eligible,
        standalone_decision=decision,lowering_eligible=True,selected=False,
        pipeline_component_claim='Callable component only; no standalone or pipeline speedup is claimed.')
    return replace(plan,proof=proof)


def validate_pipeline(pipeline):
    from scar.compiler.model import validate_plan
    if type(pipeline)is not PipelinePlan or pipeline.schema_version!=1 or pipeline.profile!=PROFILE:
        raise ValueError('supported typed array pipeline required')
    if (type(pipeline.id)is not str or not pipeline.id or pipeline.frontiers
            or any(type(x)is not str or not x for x in (*pipeline.parameters,*pipeline.input_values,pipeline.output))
            or len(pipeline.parameters)!=len(pipeline.input_values)
            or len(set(pipeline.parameters))!=len(pipeline.parameters)
            or len(set(pipeline.input_values))!=len(pipeline.input_values) or not pipeline.calls):
        raise ValueError('pipeline needs unique parameters/input values and at least one call')
    available=set(pipeline.input_values);calls=set();guards={};checked=set()
    for call in pipeline.calls:
        if type(call)is not RegionCall or type(call.plan)is not ExecutionPlan:
            raise ValueError('pipeline calls require admitted leaf execution plans')
        if (call.plan.resource_binding is not None or call.plan.device!='cpu'
                or call.plan.buffer_policy!='fresh_output'):
            raise ValueError('pipeline composition requires unbound fresh-output CPU leaf plans')
        if id(call.plan) not in checked:
            validate_plan(call.plan)
            if call.plan.proof.get('legal')is not True or call.plan.proof.get('lowering_eligible')is not True:
                raise ValueError('every pipeline leaf must have a legal, lowering-eligible proof')
            if call.plan.proof.get('frontiers'):raise ValueError('leaf proof frontiers remain open')
            if call.plan.proof.get('lowering_context')==LOWERING_CONTEXT:prepare_pipeline_leaf(call.plan)
            checked.add(id(call.plan))
        if (type(call.id)is not str or not call.id or type(call.output)is not str or not call.output
                or any(type(x)is not str or not x for x in call.inputs)
                or call.id in calls or call.output in available
                or len(call.inputs)!=len(call.plan.region.parameters) or not set(call.inputs)<=available):
            raise ValueError('call order, unique outputs and leaf parameter routing must agree')
        if (type(call.callee)is not CalleeGuard or call.callee.source!=call.plan.region.source
                or call.callee.lookup_root!='PIPELINE_FUNCTION_GLOBALS'
                or len(call.callee.lookup_path)!=1 or not all(type(x)is str and x for x in call.callee.lookup_path)):
            raise ValueError('each leaf needs its original source and direct global function guard')
        key=call.callee.lookup_root,call.callee.lookup_path
        if key in guards and guards[key]!=call.callee.source:
            raise ValueError('one global lookup cannot require different source/code identities')
        guards[key]=call.callee.source
        calls.add(call.id);available.add(call.output)
    if pipeline.output not in {call.output for call in pipeline.calls}:
        raise ValueError('pipeline output must be one fresh leaf result')
    return pipeline


def compose_pipeline(source,parameters,calls,output,*,input_values=None,id=None):
    """Compose already admitted calls; this API does not select Python callees."""
    parameters=tuple(parameters);calls=tuple(calls)
    values=tuple('arg:'+str(i) for i in range(len(parameters))) if input_values is None else tuple(input_values)
    identity=id or 'array-pipeline:'+sha256((source.path+'\0'+source.code_key+'\0'+repr(tuple(c.id for c in calls))).encode()).hexdigest()[:24]
    return validate_pipeline(PipelinePlan(identity,source,parameters,values,calls,output))


def document(pipeline):
    validate_pipeline(pipeline);plans={};keys={};calls=[]
    for call in pipeline.calls:
        leaf=leaf_document(call.plan);canonical=json.dumps(leaf,sort_keys=True,separators=(',',':'),allow_nan=False)
        if canonical not in keys:
            keys[canonical]='plan:'+str(len(keys));plans[keys[canonical]]=leaf
        calls.append(dict(id=call.id,plan_id=keys[canonical],inputs=list(call.inputs),output=call.output,
            callee=dict(lookup_path=list(call.callee.lookup_path),lookup_root=call.callee.lookup_root),origins=list(call.origins)))
    return dict(schema=SCHEMA,schema_version=pipeline.schema_version,profile=pipeline.profile,
        id=pipeline.id,source=asdict(pipeline.source),parameters=list(pipeline.parameters),
        input_values=list(pipeline.input_values),leaf_plans=plans,calls=calls,output=pipeline.output,
        assumptions=list(pipeline.assumptions),frontiers=list(pipeline.frontiers))


def pipeline_from_dict(value):
    value=dict(value)
    if value.pop('schema',None)!=SCHEMA:raise ValueError('unsupported pipeline schema')
    def source(row):
        row=dict(row);row['origins']=tuple(row.get('origins',()));return SourceRef(**row)
    plans={key:plan_from_dict(plan) for key,plan in value.pop('leaf_plans').items()}
    calls=[];used=set()
    for row in value.pop('calls'):
        if set(row)!={'id','plan_id','inputs','output','callee','origins'}:
            raise ValueError('unknown or missing pipeline call wire fields')
        if set(row['callee'])!={'lookup_root','lookup_path'}:
            raise ValueError('callee wire must reference its shared leaf source')
        row=dict(row);guard=dict(row['callee']);guard['source']=plans[row['plan_id']].region.source
        used.add(row['plan_id'])
        guard['lookup_path']=tuple(guard['lookup_path'])
        calls.append(RegionCall(row['id'],plans[row['plan_id']],tuple(row['inputs']),row['output'],
            CalleeGuard(**guard),tuple(row.get('origins',()))))
    if set(plans)!=used:raise ValueError('unused leaf table entries are outside the pipeline profile')
    value['source']=source(value['source']);value['calls']=tuple(calls)
    for name in ('parameters','input_values','assumptions','frontiers'):value[name]=tuple(value[name])
    return validate_pipeline(PipelinePlan(**value))


def pipeline_digest(pipeline):
    return sha256(json.dumps(document(pipeline),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_pipeline(pipeline,path):Path(path).write_text(json.dumps(document(pipeline),indent=2,allow_nan=False)+'\n')
def read_pipeline(path):return pipeline_from_dict(json.loads(Path(path).read_text()))


# Discover guarded straight-line leaf compositions from original F4 fields.
#
# No AST expression interpretation or target import is used. Shared production
# plans supply action order and bindings; UnitFacts supply CALL argument ports
# and module declarations. Static global targets remain runtime CodeID guards.
@dataclass(frozen=True)
class _Value:
    id: str


@dataclass(frozen=True)
class _Callee:
    name: str
    function: object
    plan: object


def _global_functions(facts,leaf_plans):
    """Original module writes and reference transports, never name heuristics."""
    outgoing=defaultdict(list)
    for left,right in facts.aliases:outgoing[left].append(right)
    result=defaultdict(list)
    for function in facts.functions:
        plan=leaf_plans.get(function.template)
        if plan is None:continue
        pending=[function.value];reachable=set()
        while pending:
            value=pending.pop()
            if value in reachable:continue
            reachable.add(value);pending.extend(outgoing.get(value,()))
        for row in facts.names:
            if row.write and row.namespace.kind is F.NamespaceKind.MODULE and row.value in reachable:
                candidate=_Callee(row.namespace.name,function,plan)
                if candidate not in result[row.namespace.name]:result[row.namespace.name].append(candidate)
    return result


class _PipelineLowerer(_GraphLowerer):
    def __init__(self,source,function,resolver,globals):
        super().__init__(source,function,resolver)
        self.globals=globals;self.calls=[]
        self.call_facts={source_port(self.facts,row.result).port:row for row in self.facts.calls}

    def binding(self,context,step,args):
        write=step.instruction.action is P.BindingActionKind.WRITE
        if step.instruction.action not in (P.BindingActionKind.READ,P.BindingActionKind.WRITE):
            _reject('PIPELINE_BINDING_DELETE','binding deletion is outside the straight-line pipeline profile')
        port=context.value(step.arguments[0]).port if write else context.result(step).port
        namespaces=tuple(dict.fromkeys(self.names.get((port,write),())))
        if namespaces and all(ns.kind is F.NamespaceKind.LEXICAL for ns in namespaces):
            if write:
                if not isinstance(args[0],(_Value,_Callee)):
                    _reject('PIPELINE_LOCAL_VALUE','locals must alias array values or guarded functions')
                for ns in namespaces:self.locals[ns]=args[0]
                return (_WORLD,)
            if len(namespaces)!=1 or namespaces[0] not in self.locals:
                _reject('PIPELINE_UNBOUND_LOCAL','local has no preceding original binding')
            return (self.locals[namespaces[0]],_WORLD)
        if write:_reject('PIPELINE_EXTERNAL_WRITE','writes outside original lexical locals are unsupported')
        if (not namespaces or any(ns.kind not in (F.NamespaceKind.MODULE,F.NamespaceKind.BUILTINS) for ns in namespaces)
                or len({ns.name for ns in namespaces})!=1 or not any(ns.kind is F.NamespaceKind.MODULE for ns in namespaces)):
            _reject('PIPELINE_CALLEE_LOOKUP','only an original direct module-global function read is supported')
        candidates=self.globals.get(namespaces[0].name,())
        if len(candidates)!=1:
            _reject('PIPELINE_GLOBAL_TARGET_OPEN','global has no unique admitted leaf declaration')
        return (candidates[0],_WORLD)

    def action(self,context,step,args,origin):
        if step.instruction.action is not P.LanguageActionKind.CALL:
            _reject('PIPELINE_NONCALL_ACTION','pipeline expressions must be admitted leaf calls',origin)
        fact=self.call_facts.get(context.result(step).port)
        if fact is None or fact.positional_pack is not None or fact.keyword_pack is not None:
            _reject('PIPELINE_CALL_FACT','call requires original fixed positional/keyword ports',origin)
        ports=tuple(context.value(value).port for value in step.arguments)
        expected=(source_port(self.facts,fact.callee).port,
            *(source_port(self.facts,i).port for i in fact.positional),
            *(source_port(self.facts,i).port for _,i in fact.keywords),source_port(self.facts,fact.world).port)
        if ports!=expected:
            _reject('PIPELINE_CALL_ABI','original CALL operands differ from the F4 action',origin)
        normal=next(completion for completion in step.completions if completion.kind is C.FALLTHROUGH)
        if context.result(step,len(normal.types)-1).port!=source_port(self.facts,fact.result_world).port:
            _reject('PIPELINE_CALL_ABI','original CALL post-world differs from its F4 result',origin)
        if not isinstance(args[0],_Callee) or not all(isinstance(v,_Value) for v in args[1:-1]):
            _reject('PIPELINE_DYNAMIC_CALL','callee and array arguments lack admitted bindings',origin)
        callee=args[0];names=callee.plan.region.parameters
        kinds=tuple(parameter.kind for parameter in callee.function.environment.parameters)
        if len(kinds)!=len(names):_reject('PIPELINE_CALL_ARGUMENTS','leaf signature metadata differs from its parameters',origin)
        actuals=args[1:-1];n=len(fact.positional);bound={}
        if n>len(names) or any(kind is P.DeferredParameterKind.KEYWORD_ONLY for kind in kinds[:n]):
            _reject('PIPELINE_CALL_ARGUMENTS','positional arguments do not match leaf parameters',origin)
        bound.update(zip(names[:n],actuals[:n]))
        for (name,_),value in zip(fact.keywords,actuals[n:]):
            if (name not in names or name in bound
                    or kinds[names.index(name)] is P.DeferredParameterKind.POSITIONAL_ONLY):
                _reject('PIPELINE_CALL_ARGUMENTS','keyword argument binding is not direct and unique',origin)
            bound[name]=value
        if set(bound)!=set(names):
            _reject('PIPELINE_CALL_DEFAULTS','all original leaf array parameters must be explicitly bound',origin)
        index=len(self.calls);output='call:'+str(index)+':output'
        call=RegionCall('call:'+str(index),callee.plan,tuple(bound[name].id for name in names),output,
            CalleeGuard(callee.plan.region.source,(callee.name,)),
            (dict(origin,call_index=index,call_fact=dict(result=encode(source_port(self.facts,fact.result)),
                callee=encode(source_port(self.facts,fact.callee)),positional=[encode(source_port(self.facts,i)) for i in fact.positional],
                keywords=[dict(name=name,value=encode(source_port(self.facts,i))) for name,i in fact.keywords])),))
        self.calls.append(call)
        return (_Value(output),_WORLD)

    def lower(self):
        function=self.function
        if (function.body_scope is None or function.environment.kind is not P.DeferredBodyKind.FUNCTION
                or function.environment.captures or function.environment.owned_cells):
            _reject('PIPELINE_ENVIRONMENT','only ordinary functions without captures are supported')
        if any(p.kind in (P.DeferredParameterKind.VAR_POSITIONAL,P.DeferredParameterKind.VAR_KEYWORD)
                for p in function.environment.parameters):
            _reject('PIPELINE_VARIADIC_SIGNATURE','pipeline parameters must have a fixed original signature')
        context=self.resolver.plan(self.source.unit,function.template)
        if any(type(step.instruction)is P.ApplyDecorator for step in context.plan.steps):
            _reject('PIPELINE_DECORATOR','decorated pipeline functions are unsupported')
        parameters=[_ENV]
        for index,name in enumerate(function.parameter_names):
            value=_Value('arg:'+str(index));parameters.append(value)
            formal=source_port(self.facts,function.formals[index+1])
            namespaces=self.names.get((formal.port,True),())
            if len(namespaces)!=1 or namespaces[0].kind is not F.NamespaceKind.LEXICAL:
                _reject('PIPELINE_PARAMETER_BINDING','formal lacks one original lexical binding')
            self.locals[namespaces[0]]=value
            self.origins.append(_origin(context,function.body,value=encode(formal),pipeline_value=value.id))
        parameters.append(_WORLD)
        kind,result=self.run(context,function.body,tuple(parameters))
        if kind is not C.RETURN or len(result)!=2 or not isinstance(result[0],_Value):
            _reject('PIPELINE_RETURN','pipeline must return one fresh leaf ndarray result')
        if len(self.calls)<2:_reject('NOT_MULTI_REGION_PIPELINE','at least two original leaf calls required')
        if result[0].id not in {call.output for call in self.calls}:
            _reject('PIPELINE_ALIASED_OUTPUT','returning a pipeline input violates fresh-output semantics')
        return tuple(self.calls),result[0].id,tuple(self.origins)


def _source_occurrence(source):
    identity = next(row for row in source.origins if row.get('kind') == 'source_identity')
    return identity['unit'], identity['occurrence']


def _prepare_leaf_plans(leaves, units):
    """Prepare one component per original admitted function occurrence."""
    selected = set(units)
    regions = {}
    for leaf in leaves:
        site = _source_occurrence(leaf.source)
        if site[0] in selected:
            regions[site] = leaf
    return {site: prepare_pipeline_leaf(optimize_region(region)) for site, region in regions.items()}


def discover_graph_pipelines(catalog,source=None):
    """Return automatically routed pipelines plus explicit profile rejections."""
    view=catalog if isinstance(catalog,ProgramView) else ProgramView(catalog)
    units=tuple(view.catalog.fact_store.rows) if source is None else tuple(sorted(
        view.catalog.units_for_source(str(Path(source).absolute())) or ()))
    if source is not None and not units:
        return [],[dict(path=str(Path(source).absolute()),code='SOURCE_NOT_IN_CATALOG')]
    leaves, _ = discover_graph_regions(view, source)
    return _pipelines_from_leaf_plans(view, units, _prepare_leaf_plans(leaves, units))


def _pipelines_from_leaf_plans(view, units, plans):
    result=[];rejected=[];resolver=BodyInterfaceResolver(view)
    for unit in units:
        original=view.unit(unit);functions=original.facts.functions
        resolver.prepare_many(Body(unit,fn.template,fn.body) for fn in functions)
        try:model,syntax,codes,recipe,digest=_source_identity(view.catalog,unit)
        except (ValueError,KeyError,SyntaxError,OSError) as error:
            rejected.append(dict(unit=unit,code='PIPELINE_SOURCE_IDENTITY_UNAVAILABLE',detail=str(error)));continue
        local_plans={fn.template:plans[unit,fn.template.occurrence] for fn in functions
            if (unit,fn.template.occurrence) in plans}
        globals=_global_functions(original.facts,local_plans)
        for ordinal,function in enumerate(functions):
            if function.template in local_plans:continue
            name,span,lines=_metadata(syntax,function.template.occurrence)
            candidates=[code for code in codes if code.co_name==name and code.co_firstlineno in lines
                and code.co_varnames[:code.co_argcount+code.co_kwonlyargcount]==function.parameter_names]
            try:
                if len(candidates)!=1:_reject('PIPELINE_CODE_IDENTITY','no unique complete code identity')
                calls,output,origins=_PipelineLowerer(original,function,resolver,globals).lower()
                code=candidates[0];key=code_key(code)
                identity=dict(kind='source_identity',method='COMPILE_ONLY_RETAINED_F0',
                    semantic_authority='ORIGINAL_F4_PRODUCTION_PLANS_AND_UNIT_FACTS',unit=unit,
                    f0_sha256=model['artifacts']['source_bytes']['decoded_sha256'],
                    f2_sha256=model['artifacts']['source_syntax']['decoded_sha256'],
                    occurrence=function.template.occurrence,span=span,recipe=recipe,code_key=key)
                reference=SourceRef(model['path'],digest,code.co_qualname,span['line'],span['column'],key,(identity,*origins))
                plan_id='graph-array-pipeline:'+sha256((unit+'\0'+function.body.wire+'\0'+key).encode()).hexdigest()[:24]
                result.append(compose_pipeline(reference,function.parameter_names,calls,output,id=plan_id))
            except GraphRegionRejected as error:
                rejected.append(dict(unit=unit,function=ordinal,path=model['path'],qualname=candidates[0].co_qualname
                    if len(candidates)==1 else name,code=error.code,reason=error.code,detail=error.detail,origin=error.origin))
    return result,rejected


discover_pipelines=discover_graph_pipelines


__all__ = [
    'discover_graph_pipelines',
    'discover_pipelines',
    'CalleeGuard',
    'RegionCall',
    'PipelinePlan',
    'prepare_pipeline_leaf',
    'compose_pipeline',
    'validate_pipeline',
    'document',
    'pipeline_from_dict',
    'pipeline_digest',
    'write_pipeline',
    'read_pipeline',
    'SCHEMA',
    'PROFILE',
    'LOWERING_CONTEXT',
]

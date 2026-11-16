"""Replay caller-matched source CALL interfaces into the existing array host.

The graph selects original shared bodies and conditional ABI routes. Existing
F4 lowerers establish the bounded pure-array profile and complete caller order.
This proof is conditional on live source/callable/input contracts; an observed
target does not prove execution coverage, purity, defaults or value identity.
"""
from collections import defaultdict
from dataclasses import dataclass,replace
from hashlib import sha256
import json
from pathlib import Path

from scar.analysis.program_view import ProgramView
from scar.analysis.source_body_interfaces import BodyInterfaceResolver
from scar.analysis.source_call_view import SourceFunctionCallProvider,function_interface_fact
from scar.analysis.source_reference_slice import source_port
from scar.compiler.graph_frontend import (_GraphLowerer,GraphRegionRejected,_reject,
    _source_identity,_metadata)
from scar.compiler.model import ArrayRegion,SourceRef
from scar.compiler.pipeline_frontend import _PipelineLowerer,_Callee,_Value
from scar.compiler.pipeline_model import (PipelinePlan,compose_pipeline,prepare_pipeline_leaf,
    document as pipeline_document,pipeline_from_dict,write_pipeline)
from scar.ir.program import normal_targets as T,source_productions as P,normal_links as F
from scar.ir.program.dependencies import DependencyKind as K
from scar.ir.record_codec import encode
from scar.optimization.optimizer import optimize_region
from scar.trace.code_snapshot import code_key


SCHEMA='scar.caller-matched-array-pipeline-proof'
PROFILE='ORIGINAL_F4_CALLER_MATCHED_PURE_ARRAY_PIPELINE_V1'
_RUNTIME=('EXACT_SOURCE_AND_CODE_IDS_AT_EVERY_CALL',
    'PLAIN_FUNCTION_GLOBAL_LOOKUPS_AND_FUNCTION_DEFAULT_STATES_UNCHANGED',
    'NO_CLOSURES_DEFAULT_PARAMETERS_CUSTOM_SIGNATURES_OR_OBSERVERS',
    'EXACT_NUMPY_NDARRAY_FLOAT64_CPU_RANK2_MODELED_SHAPES_AND_LAYOUT',
    'EXCLUSIVE_SYNCHRONOUS_INPUTS_AND_EXISTING_PIPELINE_FP_FALLBACK_CONTRACT',
    'VALIDATE_ENTIRE_ORIGINAL_CALLER_BEFORE_ANY_DEPLOYMENT_OR_PROFITABILITY_CLAIM')


def _canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def _digest(value):return sha256(_canonical(value).encode()).hexdigest()


def _pins():
    root=Path(__file__).resolve().parents[1]
    names=('compiler/caller_pipeline.py','compiler/pipeline_frontend.py','compiler/graph_frontend.py',
        'compiler/pipeline_model.py','analysis/source_call_view.py','analysis/source_body_interfaces.py',
        'ir/program/normal_targets.py','optimization/optimizer.py')
    return {name:sha256((root/name).read_bytes()).hexdigest()for name in names}


@dataclass(frozen=True)
class CallerPipelineProof:
    id:str
    caller_unit:str
    function_ordinal:int
    interface_ordinals:tuple[int,...]
    pipeline:PipelinePlan
    call_evidence:tuple[dict,...]
    catalog:dict
    interface_manifest:dict
    caller_origin:dict
    producer_pins:dict
    profile:str=PROFILE
    required_runtime_guards:tuple[str,...]=_RUNTIME
    semantic_scope:str='CLOSED_EXISTING_ARRAY_HOST_PROFILE_UNDER_RUNTIME_CONTRACTS_NOT_GENERAL_PYTHON_PURITY'
    runtime_dispatch_proven:bool=False
    observed_invocation_or_completion_proven:bool=False
    transformation_authorized:bool=False


def document(proof):
    if type(proof)is not CallerPipelineProof:raise TypeError('typed caller pipeline proof required')
    return dict(schema=SCHEMA,schema_version=1,id=proof.id,caller_unit=proof.caller_unit,
        function_ordinal=proof.function_ordinal,interface_ordinals=list(proof.interface_ordinals),
        pipeline=pipeline_document(proof.pipeline),call_evidence=list(proof.call_evidence),
        catalog=proof.catalog,interface_manifest=proof.interface_manifest,caller_origin=proof.caller_origin,
        producer_pins=proof.producer_pins,profile=proof.profile,required_runtime_guards=list(proof.required_runtime_guards),
        semantic_scope=proof.semantic_scope,runtime_dispatch_proven=proof.runtime_dispatch_proven,
        observed_invocation_or_completion_proven=proof.observed_invocation_or_completion_proven,
        transformation_authorized=proof.transformation_authorized)


def _signature(function):
    if (function.body_scope is None or function.environment.kind is not P.DeferredBodyKind.FUNCTION
            or function.environment.captures or function.environment.owned_cells
            or len(function.environment.parameters)!=len(function.parameter_names)
            or any(parameter.default_source is not None or parameter.kind not in
                (P.DeferredParameterKind.POSITIONAL_ONLY,P.DeferredParameterKind.POSITIONAL_OR_KEYWORD,
                    P.DeferredParameterKind.KEYWORD_ONLY)for parameter in function.environment.parameters)
            or any(value is not None for value in function.defaults)):
        _reject('CALLER_PIPELINE_SIGNATURE','ordinary fixed array parameters without defaults/captures required')


def _reference(view,unit,function,origins,identities):
    if unit not in identities:identities[unit]=_source_identity(view.catalog,unit)
    model,syntax,codes,recipe,digest=identities[unit]
    name,span,lines=_metadata(syntax,function.template.occurrence)
    codes=[code for code in codes if code.co_name==name and code.co_firstlineno in lines
        and code.co_varnames[:code.co_argcount+code.co_kwonlyargcount]==function.parameter_names]
    if len(codes)!=1:_reject('CALLER_PIPELINE_CODE_IDENTITY','complete original function CodeID is not unique')
    code=codes[0];key=code_key(code)
    identity=dict(kind='source_identity',method='COMPILE_ONLY_RETAINED_F0',
        semantic_authority='ORIGINAL_F4_PRODUCTION_PLANS_AND_UNIT_FACTS',unit=unit,
        f0_sha256=model['artifacts']['source_bytes']['decoded_sha256'],
        f2_sha256=model['artifacts']['source_syntax']['decoded_sha256'],occurrence=function.template.occurrence,
        span=span,recipe=recipe,code_key=key)
    return SourceRef(model['path'],digest,code.co_qualname,span['line'],span['column'],key,(identity,*origins))


def _leaf(view,body,source_callable,resolver,identities):
    source=view.unit(body.unit)
    rows=[function for function in source.facts.functions if function.template==body.template
        and function.body==body.region and source_port(source.facts,function.value)==source_callable]
    if len(rows)!=1:_reject('CALLER_PIPELINE_BODY_IDENTITY','selected body/callable lacks one original function')
    function=rows[0];_signature(function);resolver.prepare(body)
    if body.unit not in identities:identities[body.unit]=_source_identity(view.catalog,body.unit)
    nodes,output,origins=_GraphLowerer(source,function,resolver,syntax=identities[body.unit][1]).lower()
    reference=_reference(view,body.unit,function,origins,identities)
    identifier='graph-array-region:'+sha256((body.unit+'\0'+function.body.wire+'\0'+reference.code_key).encode()).hexdigest()[:24]
    region=ArrayRegion(identifier,reference,function.parameter_names,nodes,output)
    if any(node.op=='copy'for node in nodes):
        from scar.compiler.array_copy_contract import COPY_REQUIRED_GUARDS
        region=replace(region,assumptions=tuple(dict.fromkeys((*region.assumptions,*COPY_REQUIRED_GUARDS))))
    return function,prepare_pipeline_leaf(optimize_region(region))


def _global_name(facts,callee):
    index=next((i for i in range(len(facts.ports))if source_port(facts,i).port==callee.port),None)
    if index is None:_reject('CALLER_PIPELINE_CALLEE_PORT','selected callable is not an original caller port')
    reverse=defaultdict(list)
    for left,right in facts.aliases:reverse[right].append(left)
    pending=[index];seen=set()
    while pending:
        value=pending.pop()
        if value not in seen:seen.add(value);pending.extend(reverse.get(value,()))
    names={row.namespace.name for row in facts.names if not row.write and row.value in seen
        and row.namespace.kind is F.NamespaceKind.MODULE}
    if len(names)!=1:_reject('CALLER_PIPELINE_CALLEE_LOOKUP','one direct caller-global callable read required')
    return next(iter(names))


def _abi(caller,function,interface,leaf_function):
    link=interface.link;fact=caller.facts.calls[interface.query]
    if (interface.activation is not T.FunctionActivation.NORMAL or link.binding is not T.ArgumentBinding.DIRECT
            or link.input_envelope is not None or link.argument_packs is not None or link.projections
            or fact.positional_pack is not None or fact.keyword_pack is not None
            or link.result!=source_port(caller.facts,fact.result)
            or link.callee!=source_port(caller.facts,fact.callee)):
        _reject('CALLER_PIPELINE_OPEN_ABI','only original normal DIRECT argument ABI is supported')
    target_facts=caller.material.catalog.fact_store.load(link.body.unit)
    formals=tuple(source_port(target_facts,index)for index in leaf_function.formals)
    if len(formals)!=len(leaf_function.parameter_names)+2:
        _reject('CALLER_PIPELINE_FORMALS','selected leaf environment/data/world formals differ')
    routes={route.formal:route for route in link.parameters}
    if len(routes)!=len(link.parameters)or set(routes)!=set(formals):
        _reject('CALLER_PIPELINE_ROUTE_COVERAGE','selected ABI must cover each exact original formal once')
    if (routes[formals[0]].kind is not T.RouteKind.FUNCTION_ENVIRONMENT
            or routes[formals[0]].actuals!=(link.callee,)
            or routes[formals[-1]].kind is not T.RouteKind.STATE
            or routes[formals[-1]].actuals!=(source_port(caller.facts,fact.world),)):
        _reject('CALLER_PIPELINE_WORLD_ENVIRONMENT','original callable environment and incoming world are required')
    actuals={source_port(caller.facts,index)for index in (*fact.positional,*(index for _,index in fact.keywords))}
    for formal in formals[1:-1]:
        route=routes[formal]
        if (route.kind is not T.RouteKind.ARGUMENT or len(route.actuals)!=1
                or type(route.actuals[0])is not T.GlobalPort or route.actuals[0]not in actuals or route.keyword_names):
            _reject('CALLER_PIPELINE_ARGUMENT_ROUTE','direct explicit data operand required; defaults/receiver/pack remain open')
    if (leaf_function.returns is None or leaf_function.return_world is None
            or link.returns!=((source_port(target_facts,leaf_function.returns),source_port(caller.facts,fact.result)),
                (source_port(target_facts,leaf_function.return_world),source_port(caller.facts,fact.result_world)))):
        _reject('CALLER_PIPELINE_RETURN_WORLD','both original normal data and world return routes are required')
    return routes,formals


class _MatchedLowerer(_PipelineLowerer):
    def __init__(self,*args,selected,provider,**kwargs):
        super().__init__(*args,**kwargs);self.selected=selected;self.provider=provider;self.evidence=[]
        self.query_by_result={source_port(self.facts,row.result).port:i for i,row in enumerate(self.facts.calls)}

    def action(self,context,step,args,origin):
        query=self.query_by_result.get(context.result(step).port)
        if query not in self.selected:_reject('CALLER_PIPELINE_CALL_NOT_SELECTED','every executed original CALL needs its selected interface',origin)
        ordinal,interface,function,plan,routes,formals=self.selected[query]
        if not args or type(args[0])is not _Callee or args[0].plan is not plan:
            _reject('CALLER_PIPELINE_CALLEE_FLOW','original global/lexical callable flow differs from selected body',origin)
        result=super().action(context,step,args,origin);call=self.calls[-1]
        values={context.value(argument):value for argument,value in zip(step.arguments[1:-1],args[1:-1])}
        routed=[]
        for formal in formals[1:-1]:
            actual=values.get(routes[formal].actuals[0])
            if type(actual)is not _Value:_reject('CALLER_PIPELINE_VALUE_ROUTE','selected actual has no original caller dataflow value',origin)
            routed.append(actual.id)
        if tuple(routed)!=call.inputs:_reject('CALLER_PIPELINE_SIGNATURE_ROUTE','signature binding and selected source ABI disagree',origin)
        key=self.source.unit,ordinal;frame=self.provider.frames.get(key)
        if frame is None:_reject('CALLER_PIPELINE_FRAME','selected ordinary caller frame is absent',origin)
        relations=[row for row in self.provider.relations if row.interface==key]
        required={K.CALL_TARGET,K.ENVIRONMENT,K.ARGUMENT,K.STATE_INTERFACE,K.COMPLETION}
        if not required<={row.kind for row in relations}or any(row.frame!=frame for row in relations):
            _reject('CALLER_PIPELINE_MATCHED_RELATIONS','caller-matched call/argument/environment/world/return relations are incomplete',origin)
        def endpoint(point):
            if point.namespace==frame.caller:namespace='ORIGINAL_CALLER_FRAME_NAMESPACE'
            elif point.namespace==frame.target:namespace='ORIGINAL_TARGET_FRAME_NAMESPACE'
            else:_reject('CALLER_PIPELINE_RELATION_NAMESPACE','DIRECT route left its original caller/target namespaces',origin)
            return dict(namespace=namespace,node=point.node.wire)
        getter=getattr(self.provider.store,'interface_evidence',None)
        evidence=(frame.selection,)if getter is None else getter(*key)
        fact=self.facts.calls[query]
        self.evidence.append(dict(pipeline_call=call.id,original_query=query,interface_ordinal=ordinal,
            frame=encode(frame),interface_fact=encode(function_interface_fact(self.provider.store,*key)),
            original_binding_evidence=[encode(value)for value in evidence],original_call=encode(interface.call),
            original_definition=encode(interface.definition),original_reference=encode(interface.reference),
            body=encode(interface.link.body),parameters=encode(interface.link.parameters),returns=encode(interface.link.returns),
            world_before=encode(source_port(self.facts,fact.world)),world_after=encode(source_port(self.facts,fact.result_world)),
            result=encode(interface.link.result),conditions=list(interface.link.conditions),
            matched_relations=[dict(source=endpoint(row.source),target=endpoint(row.target),flow=row.flow.value,
                kind=row.kind.value,meaning=row.meaning.value,route=row.route,guards=list(row.guards))for row in relations],
            runtime_conditions_discharged=False))
        return result


def build_caller_pipeline(view,provider,*,caller_unit,function_ordinal,interface_ordinals=None):
    """Lower selected original interfaces; derive every pure leaf again from F4."""
    from scar.ir.execution_recipe import artifact_reference
    if type(view)is not ProgramView or type(provider)is not SourceFunctionCallProvider or provider.view is not view:
        raise TypeError('one shared ProgramView and its original source-call provider required')
    source=view.unit(caller_unit)
    if type(function_ordinal)is not int or not 0<=function_ordinal<len(source.facts.functions):
        raise ValueError('original caller function ordinal required')
    function=source.facts.functions[function_ordinal];_signature(function)
    material=provider.store.read_unit(caller_unit)
    queries={row.query:row for row in material.queries if function.body_scope.owner in row.owners}
    if interface_ordinals is None:
        indexes=[]
        for qi in sorted(queries):
            choices=[i for i,row in enumerate(material.interfaces)if row.query==qi]
            if len(choices)!=1:_reject('CALLER_PIPELINE_TARGET_OPEN','each caller CALL needs one explicit retained interface selection')
            indexes.extend(choices)
        interface_ordinals=tuple(indexes)
    if (type(interface_ordinals)is not tuple or any(type(i)is not int or not 0<=i<len(material.interfaces)
            for i in interface_ordinals)or len(set(interface_ordinals))!=len(interface_ordinals)):
        raise ValueError('unique original interface ordinals required')
    resolver=BodyInterfaceResolver(view);identities={};selected={};globals=defaultdict(list);leaves={}
    for ordinal in sorted(interface_ordinals):
        interface=material.interfaces[ordinal];link=interface.link
        if interface.query not in queries:_reject('CALLER_PIPELINE_CALLER_SCOPE','selected interface is not owned by this caller')
        if interface.query in selected:_reject('CALLER_PIPELINE_MULTI_TARGET','one selected body per original CALL required')
        if link.body is None or link.source_callable is None:_reject('CALLER_PIPELINE_SOURCE_BODY','selected CALL lacks a retained source body')
        body_key=link.body,link.source_callable
        if body_key not in leaves:leaves[body_key]=_leaf(view,link.body,link.source_callable,resolver,identities)
        target,plan=leaves[body_key];routes,formals=_abi(source,function,interface,target)
        name=_global_name(source.facts,link.callee);candidate=_Callee(name,target,plan)
        if candidate not in globals[name]:globals[name].append(candidate)
        selected[interface.query]=(ordinal,interface,target,plan,routes,formals)
    if len(selected)<2:_reject('NOT_MULTI_REGION_PIPELINE','at least two selected original pure-array calls required')
    provider.attach(caller_unit,queries=tuple(sorted(selected)))
    resolver.prepare(T.Body(caller_unit,function.template,function.body))
    lowerer=_MatchedLowerer(source,function,resolver,globals,selected=selected,provider=provider)
    calls,output,origins=lowerer.lower()
    if {row['original_query']for row in lowerer.evidence}!=set(selected):
        _reject('CALLER_PIPELINE_UNUSED_SELECTION','a selected CALL was not visited by the complete straight-line F4 body')
    reference=_reference(view,caller_unit,function,origins,identities)
    pipeline=compose_pipeline(reference,function.parameter_names,calls,output,
        id='caller-matched-array-pipeline:'+_digest((caller_unit,function_ordinal,reference.code_key,tuple(sorted(interface_ordinals))))[:24])
    proof=CallerPipelineProof('',caller_unit,function_ordinal,tuple(sorted(interface_ordinals)),pipeline,
        tuple(lowerer.evidence),artifact_reference(view.catalog.path/'result.json',view.catalog.report['schema']),
        encode(provider.manifest),dict(body=encode(T.Body(caller_unit,function.template,function.body)),
            scope=encode(function.body_scope),world_order='COMPLETE_ORIGINAL_F4_CALLER_STEP_ORDER_NOT_CALL_QUERY_ORDINAL_ORDER',
            caller_parameters=[encode(source_port(source.facts,index))for index in function.formals],
            caller_return=encode(source_port(source.facts,function.returns)),
            caller_return_world=encode(source_port(source.facts,function.return_world))),_pins())
    provider.store.validate();source.check()
    return replace(proof,id='caller-pipeline-proof:'+_digest(document(proof)))


def validate_caller_pipeline_proof(proof,*,view,provider):
    if (type(proof)is not CallerPipelineProof or proof.profile!=PROFILE or proof.required_runtime_guards!=_RUNTIME
            or proof.transformation_authorized is not False or proof.runtime_dispatch_proven is not False
            or proof.observed_invocation_or_completion_proven is not False or proof.producer_pins!=_pins()):
        raise ValueError('original caller-matched proof with undischarged live guards required')
    replay=build_caller_pipeline(view,provider,caller_unit=proof.caller_unit,function_ordinal=proof.function_ordinal,
        interface_ordinals=proof.interface_ordinals)
    if _canonical(document(replay))!=_canonical(document(proof)):
        raise ValueError('caller-matched pipeline differs from original F4/frame/ABI/leaf-profile replay')
    return replay


def write_caller_pipeline_proof(proof,path):
    from scar.ir.execution_recipe import artifact_reference
    path=Path(path).resolve()
    with path.open('x')as stream:json.dump(document(proof),stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    return artifact_reference(path,SCHEMA)


def read_caller_pipeline_proof(path,*,view,provider):
    document_=json.loads(Path(path).read_bytes())
    expected={'schema','schema_version',*CallerPipelineProof.__dataclass_fields__}
    if set(document_)!=expected or(document_['schema'],document_['schema_version'])!=(SCHEMA,1):
        raise ValueError('supported caller pipeline proof fields/version required')
    value={key:item for key,item in document_.items()if key not in('schema','schema_version')}
    value['pipeline']=pipeline_from_dict(value['pipeline'])
    for key in('interface_ordinals','call_evidence','required_runtime_guards'):value[key]=tuple(value[key])
    return validate_caller_pipeline_proof(CallerPipelineProof(**value),view=view,provider=provider)


def lower_caller_pipeline(proof,out,*,view,provider,input_shapes,target,cpu_threads,memory_budget_bytes,
                          strategy_mode='pool'):
    """Replay before invoking the existing C++20 pipeline compiler unchanged."""
    from scar.compiler.compile_pipeline import compile_pipelines
    from scar.ir.execution_recipe import artifact_reference
    proof=validate_caller_pipeline_proof(proof,view=view,provider=provider)
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh caller pipeline lowering output required')
    out.mkdir(parents=True);reference=write_caller_pipeline_proof(proof,out/'proof.json')
    write_pipeline(proof.pipeline,out/'pipeline.json')
    result=compile_pipelines(out/'compiled',pipeline_path=out/'pipeline.json',input_shapes=input_shapes,
        target=target,cpu_threads=cpu_threads,memory_budget_bytes=memory_budget_bytes,strategy_mode=strategy_mode)
    report=dict(schema='scar.caller-matched-pipeline-lowering',schema_version=1,proof=reference,
        original_f4_and_matched_interface_replayed_before_lowering=True,
        pipeline=artifact_reference(out/'pipeline.json','scar.array-pipeline-plan'),
        compiler=artifact_reference(out/'compiled/result.json',result['schema']),
        cpp_artifacts_compiled=sum(row['status']=='COMPILED_UNMEASURED'for row in result['pipelines']),
        new_numerical_backend=False,selected_for_use=False,transformation_authorized=False,
        preview_only=True,automatic_deployment_authorized=False,
        caller_matched_live_runtime_guard_adapter_implemented=False,
        required_runtime_guards=list(proof.required_runtime_guards),whole_program_compiled=False)
    with(out/'result.json').open('x')as stream:json.dump(report,stream,sort_keys=True,indent=2);stream.write('\n')
    return report


__all__=['CallerPipelineProof','build_caller_pipeline','validate_caller_pipeline_proof',
    'write_caller_pipeline_proof','read_caller_pipeline_proof','lower_caller_pipeline','document']

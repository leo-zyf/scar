"""A guarded, straight-line composition of shared original array-region plans."""
from dataclasses import dataclass,asdict,replace
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from scar.compiler.model import SourceRef,ExecutionPlan,document as leaf_document,plan_from_dict

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
    from scar.compiler.validation import validate_plan
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
    from scar.compiler.validation import validate_plan
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


__all__=['CalleeGuard','RegionCall','PipelinePlan','prepare_pipeline_leaf','compose_pipeline','validate_pipeline',
    'document','pipeline_from_dict','pipeline_digest','write_pipeline','read_pipeline']

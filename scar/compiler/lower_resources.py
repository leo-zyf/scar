"""Lower an admitted resource solver result through the C++ backend."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shlex

from scar.codegen import build
from .model import read_plan
from scar.optimization.schedule import schedule_from_dict
from scar.optimization.resource_solver import semantic_sha256

def lower_resources(plan_path,resources_path,planning_path,out):
    out=Path(out).absolute()
    if out.exists():raise ValueError('fresh lowered artifact output required')
    raw=Path(planning_path).read_bytes();report=json.loads(raw)
    if (report.get('schema'),report.get('schema_version'))!=('scar.resource-planning-report',1):
        raise ValueError('versioned resource planning report required')
    resource_bytes=Path(resources_path).read_bytes()
    if report.get('source_schedule',{}).get('sha256')!=sha256(resource_bytes).hexdigest():
        raise ValueError('planning report and resource schedule bytes differ')
    plan=read_plan(plan_path);schedule=schedule_from_dict(json.loads(resource_bytes))
    prediction=report.get('cost_prediction')
    if prediction is not None and prediction.get('prediction_available') is not True:
        raise ValueError('cost model has unresolved applicability or coefficient fields')
    profile=None
    if prediction is not None:
        from scar.optimization.cost_model import predict_region_costs
        receipt=report.get('cost_profile',{});profile_bytes=Path(receipt['path']).read_bytes()
        profile=json.loads(profile_bytes)
        if (sha256(profile_bytes).hexdigest()!=receipt.get('sha256')
                or semantic_sha256(profile)!=prediction.get('profile_sha256')
                or semantic_sha256(profile)!=semantic_sha256(prediction.get('profile'))):
            raise ValueError('cost profile receipt or embedded semantic document differs')
        model=report['solver']['model'];input_shapes=model['instantiation']['input_shapes']
        checked=predict_region_costs(plan,schedule,profile,input_shapes=input_shapes)
        if checked['prediction_available'] is not True:
            raise ValueError('cost profile is no longer applicable to this resource plan')
        for key in ('task_costs','work_demand','instantiation','schedule_sha256'):
            if semantic_sha256(checked[key])!=semantic_sha256(prediction.get(key)):
                raise ValueError('stored cost prediction differs from sourced model: '+key)
        durations={key:value['duration_ticks'] for key,value in checked['task_costs'].items()}
        if durations!=model['task_durations_ticks']:
            raise ValueError('solver task durations differ from the calibrated cost prediction')
        if shlex.split(os.environ.get('CXX','g++'))!=profile['backend']['compiler']['argv']:
            raise ValueError('configured compiler prefix differs from the calibrated cost context')
    artifact=build(plan,out,resource_schedule=schedule,timed_plan=report['solver'])
    if prediction is not None:
        compiler=profile['backend']['compiler']
        if (artifact.report['compiler_version']!=compiler['identity']
                or artifact.report['compiler_options']!=compiler['options']
                or artifact.report['compiler']!=compiler.get('argv')
                or artifact.report.get('runtime_implementation_sha256')!=profile['backend'].get('runtime_implementation_sha256')):
            raise ValueError('lowered compiler/runtime context differs from cost profile')
    result=dict(schema='scar.lowered-resource-plan',schema_version=1,artifact=artifact.report,
        planning_report=dict(path=str(Path(planning_path).absolute()),sha256=sha256(raw).hexdigest()),
        selected_for_use=False,deployment_authorized=False,whole_program_compiled=False)
    if prediction is not None:
        result['cost_profile']=report['cost_profile']
        result['cost_scope']=dict(physical_optimality_proven=False,
            assumptions=prediction['model_assumptions'],context=prediction['context'],
            backend_context_at_lowering='MATCHED_COMPILER_AND_RUNTIME_IMPLEMENTATION',
            cache_residency_verified=False)
    (out/'lowering.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--resources',required=True)
    p.add_argument('--planning',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    result=lower_resources(a.plan,a.resources,a.planning,a.out)
    print(json.dumps(dict(artifact=a.out,execution_linked=result['artifact'].get('execution_linked'),
        deployment_authorized=False)))

if __name__=='__main__':main()

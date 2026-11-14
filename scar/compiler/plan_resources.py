"""Solve a declared hardware-task model without executing program variants."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from scar.optimization.resource_solver import hardware_model_skeleton,solve_resource_schedule
from scar.optimization.schedule import schedule_from_dict
from .model import read_plan

def plan_resources(schedule_path,out,*,model_path=None,plan_path=None,input_shapes=None,
                   profile_path=None,backend_path=None,metadata_path=None,search_budget=100000):
    schedule_path=Path(schedule_path).absolute();out=Path(out).absolute()
    if out.exists():raise ValueError('fresh resource-planning output required')
    raw=schedule_path.read_bytes();schedule=schedule_from_dict(json.loads(raw))
    supplied={} if model_path is None else json.loads(Path(model_path).read_bytes())
    if supplied and (supplied.get('schema'),supplied.get('schema_version'))!=('scar.resource-cost-model',1):
        raise ValueError('explicit versioned resource cost model required')
    plan=None if plan_path is None else read_plan(plan_path)
    shape_evidence=None
    if metadata_path is not None:
        if plan is None or input_shapes is not None:raise ValueError('observed metadata needs a plan and no declared shape override')
        from .input_metadata import shapes_from_metadata
        input_shapes,shape_evidence=shapes_from_metadata(metadata_path,plan.region.source,plan.region.parameters)
    skeleton=(None if plan is None else hardware_model_skeleton(plan,schedule,input_shapes=input_shapes))
    model={} if skeleton is None else dict(skeleton)
    prediction=None
    if profile_path is not None:
        from scar.codegen import load
        from scar.optimization.cost_model import predict_region_costs
        if plan is None:raise ValueError('cost profile needs the original execution plan')
        if 'task_costs' in supplied:raise ValueError('use either calibrated profile costs or explicit task costs')
        context=None if backend_path is None else load(backend_path).report
        profile=json.loads(Path(profile_path).read_bytes())
        prediction=predict_region_costs(plan,schedule,profile,input_shapes=input_shapes,backend_context=context)
        model.update(prediction['solver_model'])
    if skeleton is not None:
        # Facts already derived from the admitted graph cannot be relaxed by
        # a cost file. A missing budget can be declared for model exploration,
        # but it still does not authorize native deployment.
        for key in ('task_placements','resource_capacities','materialization_sizes','memory_locations','model_instantiation'):
            if key in supplied and supplied[key]!=skeleton[key]:
                raise ValueError('cost model contradicts derived hardware/shape constraint: '+key)
        if skeleton['memory_capacities'] and 'memory_capacities' in supplied and supplied['memory_capacities']!=skeleton['memory_capacities']:
            raise ValueError('cost model cannot relax the plan memory budget')
    model.update(supplied)
    fields=('task_costs','task_placements','resource_capacities','memory_capacities',
        'materialization_sizes','memory_locations','model_instantiation')
    result=solve_resource_schedule(schedule,search_budget=search_budget,
        **{key:model.get(key) for key in fields})
    report=dict(schema='scar.resource-planning-report',schema_version=1,solver=result,
        source_schedule=dict(path=str(schedule_path),sha256=sha256(raw).hexdigest()),
        derived_constraints=skeleton)
    if shape_evidence is not None:report['input_shape_evidence']=shape_evidence
    if prediction is not None:
        report['cost_prediction']=prediction
        profile_raw=Path(profile_path).read_bytes()
        report['cost_profile']=dict(path=str(Path(profile_path).absolute()),sha256=sha256(profile_raw).hexdigest())
    report['cost_model']=(None if model_path is None else
        dict(path=str(Path(model_path).absolute()),sha256=sha256(Path(model_path).read_bytes()).hexdigest(),
            basis=model.get('basis','CALLER_DECLARED_NOT_INDEPENDENTLY_CALIBRATED')))
    report['timed_plan_backend_implemented']='SYNCHRONOUS_CPU_PROFILE_ONLY'
    report['deployment_authorized']=False
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('schedule');p.add_argument('--model');p.add_argument('--out',required=True)
    p.add_argument('--plan',help='execution plan providing original CPU, memory and expression constraints')
    p.add_argument('--shape',type=int,nargs=2,help='declared common input shape; no program execution')
    p.add_argument('--inputs',help='observed function-boundary metadata instead of a declared shape')
    p.add_argument('--profile',help='sourced hardware/backend cost profile; derive task durations from static work')
    p.add_argument('--backend',help='optional existing native artifact to check compiler context before lowering')
    p.add_argument('--search-budget',type=int,default=100000);a=p.parse_args()
    report=plan_resources(a.schedule,a.out,model_path=a.model,plan_path=a.plan,
        input_shapes=a.shape,profile_path=a.profile,backend_path=a.backend,metadata_path=a.inputs,search_budget=a.search_budget)
    result=report['solver']
    print(json.dumps(dict(status=result['status'],optimality=result['optimality'],
        lower_bound_ticks=result['lower_bound_ticks'],unresolved_fields=len(result['unresolved']),out=a.out)))

if __name__=='__main__':main()

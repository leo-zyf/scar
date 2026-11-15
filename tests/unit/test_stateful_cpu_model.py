"""Resource facts are source-independent; complete profiles bind the source call."""
from copy import deepcopy
import json

from test_cpu_hardware_placement import hardware
from test_stateful_recipe import setup_stateful_recipe
from scar.analysis.target_graph import build_target_graph
from scar.ir.execution_recipe import read_artifact
from scar.optimization.stateful_cpu_choices import build_stateful_cpu_domain,plan_stateful_cpu_choices
from scar.optimization.stateful_cpu_model import (stateful_cpu_demand,stateful_resource_context,
    predict_stateful_cpu_model)


def test_resource_reuse_complete_scope_and_missing_component_frontier(hardware,tmp_path):
    inventory=json.loads((tmp_path/'inventory.json').read_bytes())
    inventory['platform']=dict(machine='x86_64',system='linux')
    inv=tmp_path/'platform.json';inv.write_text(json.dumps(inventory))
    target=tmp_path/'model-target';build_target_graph(target,inventory=inv)
    _,region,_=setup_stateful_recipe(target,tmp_path)
    domain=build_stateful_cpu_domain(region,target,input_shapes={'left':[3,5],'right':[3,5]},
        memory_budget_bytes=4096,max_threads=1,out=tmp_path/'domain')
    candidate=domain['candidates'][0]['id'];demand=stateful_cpu_demand(domain,candidate)
    other=deepcopy(domain);other['region_sha256']='f'*64;other['domain_id']='a-different-program-domain'
    assert stateful_resource_context(other,candidate)==demand['resource_context']
    assert not {'region_sha256','domain_id','source_stage_index'}&demand['resource_context'].keys()
    assert all(not {'source_stage_index','dag_sha256'}&row['parameters'].keys() for row in demand['terms'])
    terms={row['key']:row for row in demand['terms']}
    records=[dict(component=row['component'],parameters=row['parameters'],median_ns=float(i+1),
        basis='DECLARED_COMPONENT_MODEL',scope='SYNTHETIC_UNIT_MODEL_NOT_HARDWARE_MEASUREMENTS')
        for i,row in enumerate(terms.values())]
    source=dict(schema='scar.stateful-resource-calibration',schema_version=1,
        resource_context=demand['resource_context'],components=records)
    partial=tmp_path/'partial.json';partial.write_text(json.dumps({**source,'components':records[:-1]}))
    missing=predict_stateful_cpu_model(domain,candidate,[partial],out=tmp_path/'missing-model')
    assert not missing['complete'] and missing['duration_ticks'] is None and missing['profile'] is None
    assert missing['unresolved'] and missing['known_component_sum_ns']>0
    rejected=tmp_path/'rejected.json'
    rejected.write_text(json.dumps({**source,'components':[],
        'proposed_estimates':[dict(component=records[-1]['component'],parameters=records[-1]['parameters'],
            selectable=False,point_estimate_ns=1.0,rejection_reason='independent validation error exceeds declared policy')]}))
    diagnostic=predict_stateful_cpu_model(domain,candidate,[partial,rejected],out=tmp_path/'rejected-model')
    assert diagnostic['profile'] is None and not diagnostic['complete']
    assert diagnostic['known_component_sum_ns']==missing['known_component_sum_ns']
    assert diagnostic['unresolved'][0]['rejected_estimates'][0]['point_estimate_ns']==1.0
    complete=tmp_path/'complete.json';complete.write_text(json.dumps(source))
    model=predict_stateful_cpu_model(domain,candidate,[complete],out=tmp_path/'complete-model')
    assert model['complete'] and not model['unresolved']
    profile=json.loads(open(model['profile']).read())
    assert profile['derivation']['origin']=='DECLARED_FORWARD_MODEL'
    assert 'region_sha256' in profile['context']
    choices=plan_stateful_cpu_choices(domain,cost_profiles={candidate:model['profile']},out=tmp_path/'choices')
    row=next(row for row in choices['variants'] if row['id']==candidate)
    assert row['cost_status']=='KNOWN' and choices['status']=='UNRESOLVED'
    assert choices['selected_id'] is None  # Other masks and the original NO_OP remain unknown.
    assert not model['target_workload_executed'] and not model['physical_optimality_proven']

"""One recipe preserves state epochs, storage, cost evidence and backend input."""
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from test_cpu_hardware_placement import hardware
from scar.compiler.model import Expr,SourceRef
from scar.compiler.stateful_model import (StatefulInput,StatefulAction,StatefulProvider,
    StatefulRegion,REQUIRED_GUARDS,POWER2_SEMANTICS,stateful_digest)
from scar.ir.execution_recipe import artifact_reference,recipe_id
from scar.optimization.hardware import propose_cpu_allocations
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.schedule import schedule_document,schedule_from_dict
from scar.optimization.stateful_schedule import plan_stateful_region
from scar.optimization.stateful_recipe import (make_stateful_recipe,verify_stateful_recipe,
    unpack_stateful_recipe,estimate_stateful_recipe)


def setup_stateful_recipe(hardware,tmp_path):
    source=tmp_path/'original.py';source.write_text('def step(state):\n    return None\n')
    ref=SourceRef(str(source),sha256(source.read_bytes()).hexdigest(),'step',1,0,'pycode-v2:'+'1'*64)
    inputs=(StatefulInput('left',('state','left'),'array'),StatefulInput('right',('state','right'),'array'))
    nodes=(Expr('a0','input',attrs=dict(name='left',kind='array',epoch=0)),
        Expr('b0','input',attrs=dict(name='right',kind='array',epoch=0)),
        Expr('powered','power2',('b0',),dict(provider='power')),
        Expr('captured','mul',('a0','powered')),
        Expr('a1','input',attrs=dict(name='left',kind='array',epoch=1)),
        Expr('last','add',('a1','captured')))
    actions=(StatefulAction('eval:power','evaluate','powered'),
        StatefulAction('eval:captured','evaluate','captured'),
        StatefulAction('write:left','write','captured','left',None,0,1,'add',
            dict(port='world:0'),dict(port='world:1')),
        StatefulAction('eval:last','evaluate','last'),
        StatefulAction('write:right','write','last','right',None,0,1,'assign',
            dict(port='world:1'),dict(port='world:2')),
        StatefulAction('write:left:again','write','last','left',None,1,2,'sub',
            dict(port='world:2'),dict(port='world:3')))
    region=StatefulRegion('stateful:recipe-test',ref,('state',),inputs,nodes,actions,
        providers=(StatefulProvider('power','numpy','power',('np','power'),POWER2_SEMANTICS),),guards=REQUIRED_GUARDS)
    binding=next(row for row in propose_cpu_allocations(hardware,max_threads=2,memory_budget_bytes=4096)
                 if len(row.cpu_ids)==2)
    plan=plan_stateful_region(region,input_shapes={'left':[3,5],'right':[3,5]},cpu_threads=2,
        memory_budget_bytes=4096,resource_binding=binding)
    return make_stateful_recipe(region,plan,out=tmp_path/'recipe'),region,plan


def test_versioned_recipe_cost_and_lowering_share_exact_transaction(hardware,tmp_path,monkeypatch):
    recipe,region,plan=setup_stateful_recipe(hardware,tmp_path)
    assert verify_stateful_recipe(tmp_path/'recipe/recipe.json')==recipe
    assert unpack_stateful_recipe(recipe)==(region,plan)
    graph=recipe['task_graph']
    assert graph['schema_version']==2 and schedule_document(schedule_from_dict(graph))==graph
    assert len(recipe['artifacts']['region']['sha256'])==64
    assert 'origins' not in recipe['semantic_boundary']['source']['source']
    assert all('original_expression_refs' in row['attrs'] for row in graph['tasks'] if row['id'].startswith('stage:'))
    fence=next(row for row in graph['tasks'] if row['kind']=='SUCCESS_FENCE')
    assert set(fence['validates'])=={row['id'] for row in graph['tasks']
        if row['kind'] in ('ALLOCATE','SNAPSHOT','COMPUTE','STATE_WRITE')}
    assert all(fence['id'] in row['depends_on'] for row in graph['tasks'] if row['kind']=='COMMIT')
    assert all(row['attrs']['epoch_scope']=='DEFERRED_COMMIT_OCCURRENCE_NOT_ORIGINAL_INTERMEDIATE_STATE'
        for row in graph['values'] if row['id'].startswith('committed-state:'))
    materials={row['id']:row for row in graph['materializations']}
    values={row['id']:row for row in graph['values']}
    final_copies=[row for row in graph['tasks'] if row['attrs'].get('logical_identity')=='VALUE_PRESERVING_FINAL_ROOT_COMMIT']
    assert len(final_copies)==2
    for commit in final_copies:
        destination=materials[commit['writes'][0]];source=materials[commit['attrs']['identity_source_material']]
        assert destination['value']==source['value'] and destination['ready_task']==commit['id']
        assert values[destination['value']]['producer']==source['ready_task']!=commit['id']
    earlier=[row for row in graph['tasks'] if row['kind']=='COMMIT' and row not in final_copies]
    assert len(earlier)==1 and values[materials[earlier[0]['writes'][0]]['value']]['producer']==earlier[0]['id']
    assert all(row['attrs'].get('establishes_value_ready') is False for row in graph['tasks'] if row['kind']=='ALLOCATE')
    external={row['id'] for row in graph['storage_slots'] if row['ownership']=='EXTERNAL'}
    assert len(external)==2
    assert plan['total_region_array_payload_bytes']==sum(row['capacity_bytes'] for row in graph['storage_slots'])
    unknown=estimate_stateful_recipe(recipe)
    assert unknown['timing']['status']=='UNKNOWN' and unknown['timing']['duration_ticks'] is None
    assert all(row['duration_ticks'] is None for row in unknown['timing']['task_costs'].values())
    assert unknown['payload']['total_region_array_payload_bytes']==plan['total_region_array_payload_bytes']

    facts=dict(schema='test.direct-task-costs',context=unknown['cost_context'],
        tick_seconds=dict(value=1e-9,unit='seconds/tick',basis='DECLARED_MODEL'),
        tasks={row['id']:dict(task_id=row['id'],duration_ticks=i+1,unit='ticks',basis='DECLARED_MODEL')
               for i,row in enumerate(graph['tasks'])})
    path=tmp_path/'declared-costs.json';path.write_text(json.dumps(facts))
    artifact=artifact_reference(path,facts['schema'])
    def fact(pointer): return dict(artifact=artifact,locator=pointer)
    def pointer(key): return key.replace('~','~0').replace('/','~1')
    profile=dict(schema='scar.recipe-task-costs',schema_version=1,context=facts['context'],
        context_evidence=[fact('#/context')],
        tick_seconds={**facts['tick_seconds'],'evidence':[fact('#/tick_seconds')]},
        task_costs={key:{**row,'evidence':[fact('#/tasks/'+pointer(key))]} for key,row in facts['tasks'].items()},
        model_assumptions=['EXPLICIT_SYNTHETIC_FIXED_ORDER_TASK_COSTS_NOT_HARDWARE_MEASUREMENTS'])
    estimated=estimate_stateful_recipe(recipe,profile)
    assert estimated['status']=='FIXED_ORDER_COST_MODEL_REPLAYED' and not estimated['unresolved']
    assert estimated['timing']['duration_ticks']==sum(row['duration_ticks'] for row in facts['tasks'].values())
    assert not estimated['makespan_optimality_proven'] and not estimated['whole_call_observation_split']
    wrong=deepcopy(profile);key=next(iter(wrong['task_costs']));wrong['task_costs'][key]['duration_ticks']+=1
    rejected=estimate_stateful_recipe(recipe,wrong)
    assert rejected['timing']['status']=='UNKNOWN' and any('source quantity' in row['reason'] for row in rejected['unresolved'])

    from scar.codegen import stateful as backend
    from scar.codegen.recipe import build_recipe
    calls=[]
    def build(actual,directory,**kwargs):
        calls.append((actual,kwargs));assert actual==region and kwargs==dict(plan=plan)
        directory.mkdir();(directory/'fixture.so').write_bytes(b'not-an-ELF-no-target-compiled')
        cap=recipe['implementation']['backend_capability_record']
        metadata=dict(schema=backend.SCHEMA,schema_version=1,stateful_abi_version=1,build_id='b'*64,
            region_sha256=stateful_digest(region),plan_sha256=semantic_sha256(plan),contract_sha256='c'*64,
            implementation_sha256=cap['implementation_sha256'],compiler=cap['compiler'],provider_runtime=cap['provider_runtime'],
            files=dict(library='fixture.so'),cpu_ids=recipe['resources']['selected_cpu']['cpu_ids'],
            **{key:plan[key] for key in ('cpu_threads','memory_budget_bytes','workspace_payload_bytes',
                'external_array_bytes','total_region_array_payload_bytes')})
        (directory/'artifact.json').write_text(json.dumps(metadata))
        return SimpleNamespace(region=region,plan=plan,directory=directory,report=metadata,run=lambda arrays,scalars:None)
    monkeypatch.setattr(backend,'build',build)
    lowered=build_recipe(recipe,tmp_path/'lowered')
    assert len(calls)==1 and lowered.region==region and lowered.plan==plan
    assert lowered.report['recipe_id']==estimated['recipe_id']==recipe['recipe_id']
    assert lowered.receipt['task_graph_sha256']==semantic_sha256(graph)
    assert not lowered.receipt['deployment_authorized']
    bad=deepcopy(recipe);bad['control']['fixed_source_order']=False;bad['recipe_id']=recipe_id(bad)
    with pytest.raises(ValueError,match='complete source/epoch/slot/transaction replay'):
        build_recipe(bad,tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists() and len(calls)==1
    source=tmp_path/'original.py';source.write_bytes(source.read_bytes()+b'# changed\n')
    with pytest.raises(ValueError,match='recipe component byte identity differs'): verify_stateful_recipe(recipe)

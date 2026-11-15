"""State epochs, native-provider operands and transaction buffers stay distinct."""
from copy import deepcopy

import pytest

from test_cpu_hardware_placement import hardware
from scar.compiler.model import Expr,SourceRef
from scar.compiler.stateful_model import (StatefulInput,StatefulProvider,StatefulAction,
    StatefulRegion,REQUIRED_GUARDS,POWER2_SEMANTICS)
from scar.optimization.hardware import propose_cpu_allocations
from scar.optimization.stateful_schedule import plan_stateful_region,verify_stateful_plan


def test_cross_barrier_payloads_provider_inputs_and_rhs_are_ready_before_shadow_write(hardware):
    inputs=(StatefulInput('left',('obj','first'),'array'),StatefulInput('right',('obj','second'),'array'),
            StatefulInput('scale',('obj','scale'),'scalar'))
    bounds=[[1,-1],[1,-1]]
    nodes=(Expr('a','input',attrs=dict(name='left',kind='array',epoch=0)),
        Expr('b','input',attrs=dict(name='right',kind='array',epoch=0)),
        Expr('s','input',attrs=dict(name='scale',kind='scalar',epoch=None)),
        Expr('av','slice',('a',),dict(bounds=bounds,semantics='ALIAS_DESCRIPTOR')),
        Expr('bv','slice',('b',),dict(bounds=bounds,semantics='ALIAS_DESCRIPTOR')),
        Expr('added','add',('bv','bv')),Expr('powered','power2',('added',),dict(provider='power')),
        Expr('captured','mul',('av','powered')),
        Expr('neighbor','slice',('a',),dict(bounds=[[2,None],[1,-1]],semantics='ALIAS_DESCRIPTOR')),
        Expr('delta','sub',('neighbor','captured')),Expr('factor','neg',('s',)),
        Expr('first_rhs','mul',('delta','factor')),Expr('second_delta','add',('bv','captured')),
        Expr('second_rhs','mul',('second_delta','factor')))
    evaluations=['added','powered','captured','delta','factor','first_rhs']
    actions=tuple(StatefulAction('eval:'+name,'evaluate',name) for name in evaluations)+(
        StatefulAction('write:first','write','first_rhs','left',tuple(tuple(x) for x in bounds),0,1,'add',
                       world_before={'port':'before:first'},world_after={'port':'after:first'}),
        StatefulAction('eval:second_delta','evaluate','second_delta'),
        StatefulAction('eval:second_rhs','evaluate','second_rhs'),
        StatefulAction('write:second','write','second_rhs','right',tuple(tuple(x) for x in bounds),0,1,'add',
                       world_before={'port':'before:second'},world_after={'port':'after:second'}))
    region=StatefulRegion('stateful:test',SourceRef('/tmp/stateful.py','0'*64,'step',1,0,'pycode-v2:'+'1'*64),
        ('obj',),inputs,nodes,actions,providers=(StatefulProvider('power','numpy','power',('np','power'),POWER2_SEMANTICS),),
        guards=REQUIRED_GUARDS)
    def binding(budget):
        return next(row for row in propose_cpu_allocations(hardware,max_threads=2,memory_budget_bytes=budget)
                    if len(row.cpu_ids)==2)
    plan=plan_stateful_region(region,input_shapes={'left':[5,5],'right':[5,5]},cpu_threads=2,
        memory_budget_bytes=4096,resource_binding=binding(4096))
    assert plan['status']=='READY' and plan['budget_satisfied']
    assert verify_stateful_plan(region,plan)==plan
    assert {'added','powered','captured','first_rhs','second_rhs'}<=plan['materialized_exprs'].keys()
    provider=next(row for row in plan['stages'] if row['kind']=='PROVIDER')
    assert plan['materialized_exprs']['added'] in provider['input_payloads']
    first_write=next(row for row in plan['stages'] if row['kind']=='WRITE_SHADOW')
    captured=next(row for row in plan['values'] if row['id']==plan['materialized_exprs']['captured'])
    assert captured['producer_index']<first_write['index']<captured['last_use']
    values={row['id']:row for row in plan['values']}
    for stage in plan['stages']:
        if stage['kind']=='WRITE_SHADOW':
            rhs=values[stage['input_payloads'][0]]
            assert rhs['producer_stage'] in stage['depends_on'] and rhs['producer_index']<stage['index']
    for slot in plan['workspace_slots']:
        records=sorted((values[name] for name in slot['values']),key=lambda row:row['producer_index'])
        assert all(a['last_use']<b['producer_index'] for a,b in zip(records,records[1:]))
    assert plan['transaction']['external_writes_before_success'] is False
    assert all(stage['kind']=='COMMIT' for stage in plan['stages'][-len(plan['commit_rows']):])
    assert 'NO_FOREIGN_CONCURRENT_STATE_READERS_OR_WRITERS' in plan['external_assumptions']
    assert plan['total_region_array_payload_bytes']==plan['external_array_bytes']+plan['workspace_payload_bytes']
    small=plan['total_region_array_payload_bytes']-1
    rejected=plan_stateful_region(region,input_shapes=plan['input_shapes'],cpu_threads=2,
        memory_budget_bytes=small,resource_binding=binding(small))
    assert rejected['status']=='INFEASIBLE_WITHIN_FIXED_POOL_MODEL' and not rejected['budget_satisfied']
    bad=deepcopy(plan);bad['materialized_exprs'].pop('captured')
    with pytest.raises(ValueError,match='barrier/epoch/lifetime replay'): verify_stateful_plan(region,bad)

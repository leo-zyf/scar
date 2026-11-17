"""Ordered logical epochs do not establish physical alias independence."""
from copy import deepcopy
from dataclasses import replace

import pytest

from scar.analysis.stateful_region_effects import derive_stateful_region_effects, verify_stateful_region_effects
from scar.compiler.model import Expr, SourceRef
from scar.compiler.stateful_model import StatefulInput, StatefulAction, StatefulRegion, REQUIRED_GUARDS


def test_epoch_reads_immutable_values_and_logical_views_preserve_physical_may_hazards():
    left = ((None,None),(0,2)); right = ((None,None),(2,4)); overlap = ((None,None),(1,3))
    source = SourceRef('/original/region.py','0'*64,'Region.update',1,0,'pycode-v2:'+'1'*64)
    nodes = (
        Expr('a0','input',attrs=dict(name='a',kind='array',epoch=0)),
        Expr('b0','input',attrs=dict(name='b',kind='array',epoch=0)),
        Expr('s','input',attrs=dict(name='s',kind='scalar',epoch=None)),
        Expr('left0','slice',('a0',),dict(bounds=left,semantics='ALIAS_DESCRIPTOR')),
        Expr('overlap0','slice',('a0',),dict(bounds=overlap,semantics='ALIAS_DESCRIPTOR')),
        Expr('bview','slice',('b0',),dict(bounds=left,semantics='ALIAS_DESCRIPTOR')),
        Expr('saved','add',('left0','bview')),
        Expr('seen','mul',('overlap0','s')),
        Expr('a1','input',attrs=dict(name='a',kind='array',epoch=1)),
        Expr('left1','slice',('a1',),dict(bounds=left,semantics='ALIAS_DESCRIPTOR')),
        Expr('fresh','add',('left1','s')),
        Expr('after_b','add',('bview','s')))
    actions = (
        StatefulAction('eval:saved','evaluate','saved'),
        StatefulAction('eval:seen','evaluate','seen'),
        StatefulAction('write:left','write','saved','a',left,0,1,'add'),
        StatefulAction('eval:fresh','evaluate','fresh'),
        StatefulAction('eval:after_b','evaluate','after_b'),
        StatefulAction('write:right','write','saved','a',right,1,2,'assign'),
        StatefulAction('write:b','write','seen','b',left,0,1,'assign'))
    region = StatefulRegion('source-region',source,('obj',),(
        StatefulInput('a',('obj','a'),'array'), StatefulInput('b',('obj','b'),'array'),
        StatefulInput('s',('obj','s'),'scalar')),nodes,actions,guards=REQUIRED_GUARDS)
    report = derive_stateful_region_effects(region,input_shapes={'a':[4,4],'b':[4,4]})
    assert verify_stateful_region_effects(region,report)==report
    expressions = {row['id']:row for row in report['expressions']}
    projected = {row['id']:row for row in report['actions']}
    accesses = report['accesses']; edges = report['dependencies']
    reads = [accesses[index] for index in projected['eval:fresh']['reads']]
    assert [(row['input_id'],row['epoch'],row['bounds']) for row in reads]==[('a',1,[[0,4],[0,2]])]
    assert any(row['kind']=='RAW' and row['before']=='write:left' and row['after']=='eval:fresh'
        and row['logical_overlap']=='OVERLAPPING_LOGICAL_COORDINATES' for row in edges)
    assert any(row['kind']=='WAR' and row['before']=='eval:seen' and row['after']=='write:left'
        and row['logical_overlap']=='OVERLAPPING_LOGICAL_COORDINATES' for row in edges)
    assert any(row['kind']=='RAW' and row['before']=='write:left' and row['after']=='eval:after_b'
        and row['physical_overlap']=='UNKNOWN_DISTINCT_INPUT_STORAGE_ALIAS' for row in edges)
    assert any(row['kind']=='WAW' and row['before']=='write:left' and row['after']=='write:right'
        and row['logical_overlap']=='DISJOINT_LOGICAL_COORDINATES'
        and row['physical_overlap']=='UNKNOWN_ROOT_LAYOUT_INJECTIVITY' for row in edges)
    assert sum(row['kind']=='ORIGINAL_ACTION_ORDER' for row in edges)==len(actions)-1
    assert not projected['write:right']['reads']  # saved value is not loaded from a0 again
    assert projected['write:right']['immutable_value_dependencies']==[{'expression':'saved','producer_action':'eval:saved'}]
    assert projected['write:right']['state_preservation']['unchanged_complement']
    assert not projected['write:right']['state_preservation']['complement_is_emitted_payload_read']
    assert {row['epoch'] for row in expressions['saved']['input_provenance']}=={0}
    assert not expressions['saved']['provenance_is_runtime_reload']
    assert sum(row['role']=='READ_MODIFY_WRITE_DESTINATION' for row in accesses)==1
    assert next(row for row in report['array_root_aliases'] if row['left']=='a' and row['right']=='b')['physical_alias']=='UNKNOWN'
    assert all(row['status']=='UNDISCHARGED_PROFILE_REQUIREMENT' for row in report['contracts'])
    assert report['source_total_order_retained'] and not report['runtime_guards_discharged']
    assert not report['whole_python_function_purity_proven'] and not report['parallelization_authorized']
    assert report['unknown_effects']['hooks_and_callbacks']['status']=='UNKNOWN'
    forged = deepcopy(report); forged['accesses'][0]['epoch']=99
    with pytest.raises(ValueError,match='action/epoch/shape replay'):
        verify_stateful_region_effects(region,forged)
    stale = replace(region,nodes=tuple(replace(node,inputs=('left0','s')) if node.id=='fresh' else node for node in nodes))
    with pytest.raises(ValueError,match='current ordered storage epoch'):
        derive_stateful_region_effects(stale,input_shapes={'a':[4,4],'b':[4,4]})

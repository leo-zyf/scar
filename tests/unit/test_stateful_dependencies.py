"""State/value dependencies survive writes without inventing loop recurrence."""
from scar.analysis.stateful_dependencies import project_stateful_dependencies, trace_dependencies
from scar.compiler.model import Expr, SourceRef
from scar.compiler.stateful_model import StatefulInput, StatefulAction, StatefulRegion, REQUIRED_GUARDS


def test_original_alias_reads_snapshots_partial_write_and_full_overwrite():
    source = SourceRef('/original/solver.py', '0'*64, 'Solver.step', 1, 0,
        'pycode-v2:'+'1'*64, origins=({'expression':'sum', 'callsite':{'original_port':'callee'}},))
    inputs = (StatefulInput('left', ('obj','left'), 'array'),
        StatefulInput('right', ('obj','right'), 'array'), StatefulInput('scale', ('obj','scale'), 'scalar'))
    bounds = ((1,-1),(1,-1))
    nodes = (Expr('a','input',attrs=dict(name='left',kind='array',epoch=0)),
        Expr('b','input',attrs=dict(name='right',kind='array',epoch=0)),
        Expr('s','input',attrs=dict(name='scale',kind='scalar',epoch=None)),
        Expr('av','slice',('a',),dict(bounds=bounds,semantics='ALIAS_DESCRIPTOR')),
        Expr('bv','slice',('b',),dict(bounds=bounds,semantics='ALIAS_DESCRIPTOR')),
        Expr('sum','add',('av','bv')), Expr('saved','mul',('sum','s')),
        Expr('after','input',attrs=dict(name='left',kind='array',epoch=1)),
        Expr('next','mul',('after','s')), Expr('zero','const',attrs=dict(value=0)))
    actions = (StatefulAction('sum','evaluate','sum'), StatefulAction('saved','evaluate','saved'),
        StatefulAction('write1','write','saved','left',None,0,1,'assign'),
        StatefulAction('next','evaluate','next'),
        StatefulAction('write2','write','saved','right',bounds,0,1,'assign'),
        StatefulAction('clear','write','zero','left',None,1,2,'assign'))
    region = StatefulRegion('original:region',source,('obj',),inputs,nodes,actions,guards=REQUIRED_GUARDS)
    projection = project_stateful_dependencies(region)
    keys = {record.get('original_id'):record['key'] for record in projection['nodes']
        if record['kind'] not in ('EVALUATE','ORDERED_ARRAY_WRITE') and 'original_id' in record}
    writes = projection['writes']; assert len(writes)==3
    first = trace_dependencies(projection,writes[0]['after'])
    root_pairs = {(row.get('input_id'),row.get('epoch')) for row in first['roots']
        if row['kind']=='ORDERED_ARRAY_CONTENT_VERSION'}
    assert root_pairs=={('left',0),('right',0)}
    second = trace_dependencies(projection,writes[1]['after'])
    assert writes[0]['after'] not in second['visited_keys']  # saved is immutable across the write
    assert any(edge['kind']=='PRESERVES_UNWRITTEN_REGION' for edge in projection['edges'])
    updated = trace_dependencies(projection,keys['next'])
    assert writes[0]['after'] in updated['visited_keys']  # new descriptor reads the new content epoch
    cleared = trace_dependencies(projection,projection['exit_states']['left'])
    assert all(row['kind']!='ORDERED_ARRAY_CONTENT_VERSION' for row in cleared['roots'])
    assert keys['zero'] in {row['key'] for row in cleared['roots']}
    ordered = trace_dependencies(projection,projection['exit_states']['left'],include_ordering=True)
    assert len(ordered['visited_keys'])>len(cleared['visited_keys'])
    assert projection['transfer']['loop_recurrence_proven'] is False
    assert projection['transfer']['iterations_unrolled'] is False
    assert projection['frontiers'][0]['kind']=='CALLER_LOOP_CONNECTION'
    assert projection['transformation_authorized'] is False
    assert next(row for row in projection['nodes'] if row['key']==keys['sum'])['origins']==['#/source/origins/0']

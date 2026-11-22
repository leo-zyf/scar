from dataclasses import replace

import pytest

from scar.compiler.model import ArrayRegion,Expr,SourceRef
from scar.compiler.pipeline_frontend import CalleeGuard, RegionCall, compose_pipeline
from scar.ir.versioned_schedule import validate_versioned_schedule
from scar.optimization.optimizer import optimize_region
from scar.optimization.pipeline_memory import plan_pipeline_memory
from scar.optimization.pipeline_schedule import _dependencies, _waves, _task_graph


def _leaf(arity):
    names=tuple('x'+str(i)for i in range(arity));source=SourceRef('/original/waves.py','a'*64,
        'leaf'+str(arity),1,0,'pycode-v2:'+str(arity)*64)
    nodes=tuple(Expr(name,'input',attrs={'name':name})for name in names)
    expr=Expr('value','add',names)if arity==2 else Expr('value','neg',names)
    return optimize_region(ArrayRegion('leaf:'+str(arity),source,names,(*nodes,expr,Expr('out','neg',('value',))),'out'))


def _pipeline(spec):
    unary,binary=_leaf(1),_leaf(2);calls=[]
    for name,inputs in spec:
        plan=unary if len(inputs)==1 else binary
        calls.append(RegionCall(name,plan,tuple(inputs),name,CalleeGuard(plan.region.source,(plan.region.source.qualname,))))
    source=SourceRef('/original/waves.py','a'*64,'entry',1,0,'pycode-v2:'+'f'*64)
    return compose_pipeline(source,('x',),calls,calls[-1].output)


def test_fork_wave_has_true_raw_edges_and_two_disjoint_cpu_assignments():
    pipeline=_pipeline([('a',('arg:0',)),('b',('arg:0',)),('c',('a','b'))])
    memory=plan_pipeline_memory(pipeline,input_shapes=[2,3],memory_budget_bytes=1024)
    predecessors,_=_dependencies(pipeline,memory);waves,cpus=_waves(pipeline,predecessors,[4,9])
    assert predecessors=={'a':[],'b':[],'c':['a','b']}
    assert waves==[['a','b'],['c']]and cpus=={'a':4,'b':9,'c':4}
    graph=_task_graph(pipeline,memory,predecessors,waves,cpus,({'kind':'STRUCTURAL_TEST_ONLY'},))
    assert validate_versioned_schedule(graph)['overwrite_happens_before_verified']is True
    assert graph.proof['external_input_aliasing']=='MAY_ALIAS_READ_ONLY_NO_DISTINCT_ALLOCATION_INFERRED'
    assert graph.budget_bytes==memory['memory_budget_bytes']+48


def test_pool_war_waits_for_every_previous_consumer_and_validator_rejects_missing_happens_before():
    pipeline=_pipeline([('a',('arg:0',)),('b',('a',)),('c',('a',)),('d',('b','c')),('e',('arg:0',)),('f',('d','e'))])
    memory=plan_pipeline_memory(pipeline,input_shapes=[2,3],memory_budget_bytes=4096)
    predecessors,reasons=_dependencies(pipeline,memory);waves,cpus=_waves(pipeline,predecessors,[4,9])
    wars=[r for r in reasons if r['kind']=='WAR_SLOT_REUSE'and r['previous_value']=='a']
    assert wars and all({'b','c'}<=set(predecessors[row['after']])for row in wars)
    assert 'd'in predecessors['e']and pipeline.calls[4].inputs==('arg:0',)
    graph=_task_graph(pipeline,memory,predecessors,waves,cpus,({'kind':'STRUCTURAL_TEST_ONLY'},))
    validate_versioned_schedule(graph)
    # e has no RAW predecessor; retain its reused slot's old producer b (WAW)
    # but drop old reader d (WAR). Tuple/source order must not mask the hazard.
    broken=replace(graph,tasks=tuple(replace(task,depends_on=tuple(d for d in task.depends_on
        if d.startswith('allocate:'))+('compute:1',))if task.id=='compute:4'else task for task in graph.tasks))
    with pytest.raises(ValueError,match='old-version consumer'):
        validate_versioned_schedule(broken)

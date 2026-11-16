from dataclasses import replace

from scar.compiler.array_copy_contract import COPY_REQUIRED_GUARDS,copy_attributes
from scar.compiler.model import ArrayRegion,ExecutionPlan,Expr,SourceRef
from scar.compiler.pipeline_model import CalleeGuard,RegionCall,compose_pipeline,prepare_pipeline_leaf
from scar.optimization.graph import ordered_nodes,simplify
from scar.optimization.optimizer import optimize_region
from scar.optimization.pipeline_fusion import fuse_pipeline_region,verify_pipeline_fusion
from scar.optimization.resource_solver import _shape_instantiation


SOURCE=SourceRef('/original/copy_leaf.py','a'*64,'copy_leaf',1,0,'pycode-v2:'+'b'*64)
ASSUMPTIONS=('EXACT_NUMPY_NDARRAY_FLOAT64_CPU_2D','FRESH_OUTPUT','NO_INPUT_WRITES',*COPY_REQUIRED_GUARDS)


def _region(arithmetic=True):
    nodes=[Expr('x','input',attrs={'name':'x'}),Expr('first','copy',('x',),copy_attributes()),
        Expr('second','copy',('first',),copy_attributes())]
    if arithmetic:nodes.append(Expr('negative','neg',('second',)))
    nodes.append(Expr('out','copy',('negative' if arithmetic else 'second',),copy_attributes()))
    return ArrayRegion('copy-leaf',SOURCE,('x',),tuple(nodes),'out',ASSUMPTIONS)


def test_copy_value_forwarding_keeps_final_fresh_boundary_and_explicit_work():
    raw=_region();plan=optimize_region(raw,memory_budget_bytes=1024)
    assert plan.proof['legal']is True
    assert [(n.id,n.op,n.inputs)for n in plan.region.nodes]==[
        ('x','input',()),('negative','neg',('x',)),('out','copy',('negative',))]
    assert plan.proof['source_operation_map']['first']['reason']=='exact_copy_value_forwarding'
    assert plan.proof['source_operation_map']['second']['optimized_node']=='x'
    assert plan.buffer_policy=='fresh_output' and plan.region.output=='out'
    proof=plan.proof['copy_value_forwarding']
    assert proof['final_copy_boundary_preserved']and not proof['source_object_or_storage_identity_reused']
    assert not proof['original_internal_allocation_failure_sequence_preserved']
    assert set(COPY_REQUIRED_GUARDS)<=set(plan.region.assumptions)&set(plan.proof['runtime_guards'])
    assert plan.cost['baseline_copy_materializations']==3 and plan.cost['baseline_array_allocations']==4
    assert plan.cost['baseline_copy_logical_reads_per_output_element']==3
    assert all(row['physical_transfer']=='UNKNOWN_NOT_RUNTIME_OBSERVED'for row in plan.cost['copy_work'])
    assert _shape_instantiation(ExecutionPlan(raw),{'x':[3,5]})[1]==(3,5)
    assert sum(n.op=='copy'for n in simplify(ordered_nodes(raw),raw.output).nodes)==3
    for rejected in (replace(raw,assumptions=ASSUMPTIONS[:-1]),replace(raw,frontiers=('unknown_state_write',)),
        replace(raw,nodes=(raw.nodes[0],replace(raw.nodes[1],attrs={**copy_attributes(),'order':'F'}),*raw.nodes[2:]))):
        result=optimize_region(rejected)
        assert result.region is rejected and result.proof['supported']is False


def test_pipeline_copy_calls_keep_source_work_and_guard_obligations():
    leaf=prepare_pipeline_leaf(optimize_region(_region(False)))
    guard=CalleeGuard(SOURCE,('copy_leaf',))
    source=replace(SOURCE,qualname='entry',code_key='pycode-v2:'+'c'*64)
    calls=(RegionCall('a',leaf,('arg:0',),'a',guard),RegionCall('b',leaf,('a',),'b',guard))
    pipeline=compose_pipeline(source,('x',),calls,'b')
    merged,certificate=fuse_pipeline_region(pipeline)
    assert merged is not None and certificate['copy_work']['original_materialization_count']==6
    assert certificate['copy_work']['represented_copy_expression_count']==2
    assert set(COPY_REQUIRED_GUARDS)<=set(merged.assumptions)
    fused=optimize_region(merged)
    assert sum(n.op=='copy'for n in fused.region.nodes)==1 and fused.buffer_policy=='fresh_output'
    assert verify_pipeline_fusion(pipeline,merged,certificate)is True
    dead=compose_pipeline(source,('x',),(calls[0],replace(calls[1],inputs=('arg:0',))),'b')
    region,refusal=fuse_pipeline_region(dead)
    assert region is None and any(row['reason']=='ORIGINAL_CALL_MATERIALIZATION_OUTSIDE_FINAL_DEPENDENCY'
        for row in refusal['frontiers'])

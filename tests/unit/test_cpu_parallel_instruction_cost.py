import json

from scar.codegen import cpu_parallel_instruction_cost as P
from scar.codegen import cpu_instruction_cost as S
from scar.codegen import cpu_fused_cost_v2 as V2
from scar.codegen import cpu_fused_cost_v3 as V3


def test_parallel_partition_fresh_design_and_one_stage_barrier():
    context={'worker_cpu_ids':[0,1,2,3]};configs=P._configurations(context)
    assert len(configs)==123 and sum(r['holdout'] for r in configs)==32
    for row in configs:
        assert sum(row['logical_worker_rows'])==row['rows']
        assert row['elements']==max(row['logical_worker_rows'])*row['cols']
    assert next(r['logical_worker_rows'] for r in configs if r['rows']==439)==[110,110,110,109]
    assert next(r['logical_worker_rows'] for r in configs if r['rows']==618)==[155,155,154,154]
    shape=lambda r:json.dumps({'graph':r['graph'],'output':r['output']},sort_keys=True)
    seen={shape(r) for registry in (S._registry(),V2.canonical_registry(),V3._registry())
        for r in registry if r['holdout']}
    assert not seen&{shape(r) for r in P._registry() if r['holdout']}
    source=P._source(context)
    start=source.index('void parallel_stage_barrier_floor_g1(')
    end=source.index('void parallel_stage_barrier_floor_g2(')
    assert source[start:end].count('#pragma omp barrier')==1
    assert '#pragma omp for' not in source[start:end]
    analysis={'vector':{'block_throughput_cycles':4},'scalar':{'cycles_per_element':3}}
    coefficient={'barrier_floor_ns':11,'effective_ns_per_mca_cycle':2,'memory':{'1':{
        '2':[{'elements':1,'proxy_ns_per_element':1},{'elements':100,'proxy_ns_per_element':1}]}}}
    cost=P._predict({'rows':7,'cols':5,'workers':4,'elements':10,'groups':1,'array_loads':2},analysis,coefficient)
    assert cost['duration_ns']==11+max(10,2*2*(2*4+3))
    assert cost['completion_barriers_charged']==1 and cost['team_lifecycle_charged_here'] is False
    assert 'dispatch_ns' not in cost
    assembly='''.string "keep#literal"
.L1:
 movsd (%rsi,%rax,8), %xmm2
 movapd %xmm2, %xmm1 # MEM[(double *)ptr] describes a source value
 addsd %xmm1, %xmm0
 movsd %xmm0, (%rdx,%rax,8)
 addq $1, %rax
 cmpq %rdi, %rax
 jne .L1
'''
    lexed=P._lex_assembly(assembly)
    assert '.string "keep#literal"' in lexed
    assert len(assembly.splitlines())==len(lexed.splitlines())
    assert S._natural_loops(lexed)[0]['width']==1

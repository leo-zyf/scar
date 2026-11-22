"""Serial and OpenMP instruction-cost modeling and evidence replay."""

import json

import pytest

from scar.codegen.cpu_fused_cost_models import registry_v2, registry_v3
from scar.codegen.cpu_fused_cost import _receipt
from scar.codegen.cpu_instruction_cost import PARALLEL, SERIAL, _cycle_demand, _lex_assembly, _natural_loops
from scar.codegen.cpu_instruction_replay import _replay_analysis


@pytest.fixture
def instruction_analysis():
    return {
        'vector': {'block_throughput_cycles': 4},
        'scalar': {'cycles_per_element': 3},
    }


def test_instruction_structure_and_serial_boundary_are_explicit(instruction_analysis):
    text='''f:
# /tmp/canonical.cpp:10: for (...)
.L1:
    movupd (%rsi,%rax), %xmm0
    addpd %xmm1, %xmm0
    movups %xmm0, (%rbp,%rax)
    addq $16, %rax
    cmpq %rbx, %rax
    jne .L1
.L2:
    movsd (%rsi,%rax,8), %xmm0
    addsd %xmm1, %xmm0
    movsd %xmm0, (%rdx,%rax,8)
    addq $1, %rax
    cmpq %rbx, %rax
    jne .L2
'''
    loops=_natural_loops(text,source_range=(10,12))
    assert [r['width'] for r in loops]==[2,1]
    assert _natural_loops(text,source_range=(20,30))==[]
    # An out-of-line label can follow another scope/header in GCC verbose asm.
    # Its own explicit source marks identify this loop, not that stale prefix.
    prefixed=text.replace('f:\n# /tmp/canonical.cpp:10: for (...)',
        'f:\n# /tmp/canonical.cpp:999: previous_scope')
    prefixed=prefixed.replace('.L1:\n','.L1:\n# /tmp/canonical.cpp:10: value\n# /usr/include/header.h:888: inline\n')
    assert _natural_loops(prefixed,source_range=(10,12),source_path='/tmp/canonical.cpp')[0]['width']==2
    assert _cycle_demand(instruction_analysis,3,5,1)==3*(2*4+3)
    source=SERIAL.source({'worker_cpu_ids':[0]})
    assert '#ifdef _OPENMP\n#include <omp.h>\n#endif' in source
    assert 'probe_team()' in source and 'probe_tid()' in source
    registry=SERIAL.registry();configs=SERIAL.configurations({'worker_cpu_ids':[0]})
    assert all(r['role'] in ('DISPATCH','MEMORY','MATH','FRESH_HOLDOUT') for r in registry)
    assert {r['rows'] for r in configs if r['holdout']}=={432,593}
    assert not {r['rows'] for r in configs if r['holdout']}&{r['rows'] for r in configs if not r['holdout']}
    # Synthetic unit values exercise the forward formula, never hardware claims.
    coefficients={'dispatch_ns':7,'effective_ns_per_mca_cycle':2,'memory':{'1':{
        '2':[{'elements':1,'proxy_ns_per_element':1},{'elements':100,'proxy_ns_per_element':1}]}}}
    cost=SERIAL.predict({'rows':3,'cols':5,'workers':1,'elements':15,'groups':1,'array_loads':2},
                  instruction_analysis,coefficients)
    assert cost['duration_ns']==7+max(15,2*33)


def test_parallel_partition_fresh_design_and_one_stage_barrier(instruction_analysis):
    context={'worker_cpu_ids':[0,1,2,3]};configs=PARALLEL.configurations(context)
    assert len(configs)==123 and sum(r['holdout'] for r in configs)==32
    for row in configs:
        assert sum(row['logical_worker_rows'])==row['rows']
        assert row['elements']==max(row['logical_worker_rows'])*row['cols']
    assert next(r['logical_worker_rows'] for r in configs if r['rows']==439)==[110,110,110,109]
    assert next(r['logical_worker_rows'] for r in configs if r['rows']==618)==[155,155,154,154]
    shape=lambda r:json.dumps({'graph':r['graph'],'output':r['output']},sort_keys=True)
    seen={shape(r) for registry in (SERIAL.registry(),registry_v2(),registry_v3())
        for r in registry if r['holdout']}
    assert not seen&{shape(r) for r in PARALLEL.registry() if r['holdout']}
    source=PARALLEL.source(context)
    start=source.index('void parallel_stage_barrier_floor_g1(')
    end=source.index('void parallel_stage_barrier_floor_g2(')
    assert source[start:end].count('#pragma omp barrier')==1
    assert '#pragma omp for' not in source[start:end]
    coefficient={'barrier_floor_ns':11,'effective_ns_per_mca_cycle':2,'memory':{'1':{
        '2':[{'elements':1,'proxy_ns_per_element':1},{'elements':100,'proxy_ns_per_element':1}]}}}
    cost=PARALLEL.predict({'rows':7,'cols':5,'workers':4,'elements':10,'groups':1,'array_loads':2},instruction_analysis,coefficient)
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
    lexed=_lex_assembly(assembly)
    assert '.string "keep#literal"' in lexed
    assert len(assembly.splitlines())==len(lexed.splitlines())
    assert _natural_loops(lexed)[0]['width']==1


def test_rehashed_instruction_mutation_is_rejected_against_compiler_loop(tmp_path):
    assembly='''function:
.L1:
 movupd (%rsi,%rax), %xmm0
 addpd %xmm1, %xmm0
 movups %xmm0, (%rdx,%rax)
 addq $16, %rax
 cmpq %rdi, %rax
 jne .L1
'''
    path=tmp_path/'loop.s'
    path.write_text(assembly[assembly.index('.L1:'):].replace('addpd','mulpd'))
    output=tmp_path/'result.json';output.write_text(json.dumps({}))
    # Both forged file receipts are internally valid. The original compiler
    # instruction stream still proves that the substituted operation is wrong.
    analysis={'loops':[{'loop_source':_receipt(path),'result':_receipt(output)}]}
    with pytest.raises(ValueError,match='MCA instruction input differs'):
        _replay_analysis(assembly,analysis,{},function=None)

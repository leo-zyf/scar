from scar.codegen.cpu_instruction_cost import (_natural_loops,_cycle_demand,_source,
    _registry,_configurations,_predict)


def test_instruction_structure_and_serial_boundary_are_explicit():
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
    analysis={'vector':{'block_throughput_cycles':4},'scalar':{'cycles_per_element':3}}
    assert _cycle_demand(analysis,3,5,1)==3*(2*4+3)
    source=_source({'worker_cpu_ids':[0]})
    assert '#ifdef _OPENMP\n#include <omp.h>\n#endif' in source
    assert 'probe_team()' in source and 'probe_tid()' in source
    registry=_registry();configs=_configurations({'worker_cpu_ids':[0]})
    assert all(r['role'] in ('DISPATCH','MEMORY','MATH','FRESH_HOLDOUT') for r in registry)
    assert {r['rows'] for r in configs if r['holdout']}=={432,593}
    assert not {r['rows'] for r in configs if r['holdout']}&{r['rows'] for r in configs if not r['holdout']}
    # Synthetic unit values exercise the forward formula, never hardware claims.
    coefficients={'dispatch_ns':7,'effective_ns_per_mca_cycle':2,'memory':{'1':{
        '2':[{'elements':1,'proxy_ns_per_element':1},{'elements':100,'proxy_ns_per_element':1}]}}}
    cost=_predict({'rows':3,'cols':5,'workers':1,'elements':15,'groups':1,'array_loads':2},analysis,coefficients)
    assert cost['duration_ns']==7+max(15,2*33)

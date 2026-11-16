import ctypes
import json

from scar.codegen import cpu_batch_instruction_cost as B
from scar.codegen import cpu_instruction_cost as S
from scar.codegen import cpu_parallel_instruction_cost as P
from scar.codegen import cpu_fused_cost_v2 as V2
from scar.codegen import cpu_fused_cost_v3 as V3
from scar.codegen.cpu_batch_cost_diagnostics import _repeat_diagnostics


def test_frozen_batch_design_and_nonauthorizing_repeat_diagnostics():
    shape=lambda row:json.dumps({'graph':row['graph'],'output':row['output']},sort_keys=True)
    old={shape(row) for registry in (S._registry(),P._registry(),V2.canonical_registry(),V3._registry())
        for row in registry if row['holdout']}
    fresh=[row for row in B._registry() if row['holdout']]
    assert len(fresh)==8 and not old & {shape(row) for row in fresh}
    configurations=B._configurations({'worker_cpu_ids':[14,15]})
    assert len(configurations)==123 and sum(row['holdout'] for row in configurations)==32
    assert {(row['rows'],row['cols']) for row in configurations if row['holdout']}=={(427,587),(622,764)}
    assert ctypes.sizeof(B._Mask)==16 and B._Mask.cpu_ids.offset==8
    first='.L119:\n movsd (%rax), %xmm0\n addsd %xmm1, %xmm0\n jne .L119\n'
    other=first.replace('.L119','.L43')
    assert B._normalized_block(first)==B._normalized_block(other)
    assert B._normalized_block(first)!=B._normalized_block(first.replace('%rax','%rbx'))
    assert B._normalized_block(first)!=B._normalized_block(first.replace('(%rax)','8(%rax)'))
    result=_repeat_diagnostics([99.,100.,101.],70.)
    assert result['prediction_outside_observed_repeat_range']
    assert result['closest_repeat_relative_error']>.20
    assert result['relative_repeat_span']==.02
    assert result['observed_scatter_only_not_theoretical_or_statistical_bound'] is True

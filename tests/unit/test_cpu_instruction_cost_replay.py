import json

import pytest

from scar.codegen.cpu_fused_cost import _receipt
from scar.codegen.cpu_instruction_cost_replay import _replay_analysis


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

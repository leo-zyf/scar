"""Whole-module identities retain future/annotations and conditional stores."""
from copy import deepcopy
import dis
from hashlib import sha256
import sys
from types import CodeType

import pytest

from scar.compiler.entry_attachment import plan_entry_attachment,verify_entry_attachment
from scar.runtime.loop_contract import ASSUMPTIONS,canonical_sha256,validate_contract
from scar.trace.code_snapshot import code_key


def test_whole_module_original_store_anchors_are_candidates_not_execution(tmp_path):
    path=tmp_path/'anchored.py'
    text=('from __future__ import annotations\nglobal evolve\n'
        'def leaf(x: MissingType):\n    return x\n'
        'def body(x):\n    return leaf(x)\n'
        'if choose_branch:\n    def evolve(state,steps):\n        return body(state)\n'
        'else:\n    def evolve(state,steps):\n        return body(state)+0\n'
        'raise RuntimeError("this module must not execute")\n')
    path.write_text(text);whole=compile(path.read_bytes(),str(path),'exec',flags=0,dont_inherit=True,optimize=sys.flags.optimize)
    codes={}
    for value in whole.co_consts:
        if type(value)is CodeType:codes.setdefault(value.co_name,value)
    source=lambda name:dict(path=str(path),sha256=sha256(path.read_bytes()).hexdigest(),qualname=name,code_key=code_key(codes[name]))
    ids=dict(build_id='1'*64,loop_sha256='2'*64,strategy_sha256='3'*64,execution_plan_sha256='4'*64)
    native=dict(schema='scar.native-loop-contract',schema_version=1,loop_sha256=ids['loop_sha256'],strategy_sha256=ids['strategy_sha256'],
        execution_plan_sha256=ids['execution_plan_sha256'],trip_count=2,state_shape=[2,2],output_shape=[2,2],final_bytes=32,memory_budget_bytes=32)
    ids['contract_sha256']=canonical_sha256(native);assumptions=sorted(ASSUMPTIONS)
    contract=validate_contract(dict(schema='scar.loop-runtime-contract',schema_version=1,profile='cpu-f64-rank2-loop-v1',
        native=dict(library='kernel.so',library_sha256='5'*64,loop_abi_version=1,ids=ids,contract=native),
        source=source('evolve'),parameters=['state','steps'],state_parameter='state',trip_count_parameter='steps',
        callee_guards=[dict(environment_path=['body'],source=source('body')),dict(environment_path=['body','leaf'],source=source('leaf'))],
        array=dict(shape=[2,2],dtype='float64',device='cpu',layout='C',alignment_bytes=8,final_bytes=32),trip_count=2,memory_budget_bytes=32,
        cpu=dict(ids=[0],threads=1,architecture='x86_64',os='linux',pointer_bits=64,byteorder='little',required_isa='BASELINE_X86_64'),
        external_assumptions=assumptions,floating_environment_policy='PRESERVE_CALLER_AND_WORKER_ENVIRONMENT_NOT_NUMPY_STATUS_FLAG_EQUIVALENCE',
        selection=dict(decision='TRANSFORM_UNDER_CONTRACT',selected=True,artifact={k:ids[k] for k in ('build_id','loop_sha256','strategy_sha256')},external_assumptions=assumptions)))
    plan=plan_entry_attachment(path,contract);module=plan['modules'][0]
    assert module['module_code_key']==code_key(whole) and module['mode']=='ENTRY_MAIN_LOCAL_POST_STORE_CANDIDATES'
    assert module['compile_recipe']==dict(filename=str(path),mode='exec',flags=0,dont_inherit=True,optimize=sys.flags.optimize)
    instructions=tuple(dis.get_instructions(whole,adaptive=False,show_caches=False));by_offset={i.offset:j for j,i in enumerate(instructions)}
    assert set(module['required_global_names'])=={'evolve','body','leaf'} and len(module['code_matches'])==3
    assert {a['store_opname'] for a in module['anchors']}=={'STORE_NAME','STORE_GLOBAL'}
    assert any(a['after_is_jump_target'] for a in module['anchors'])
    for anchor in module['anchors']:
        index=by_offset[anchor['store_offset']]
        assert instructions[index].argval==anchor['binding'] and instructions[index+1].offset==anchor['after_offset']
        assert not anchor['predecessor_execution_proven'] and not anchor['guard_readiness_proven']
    assert verify_entry_attachment(path,contract,plan) and not plan['attachment_before_first_invocation_proven']
    changed=deepcopy(plan);changed['modules'][0]['anchors'][0]['after_offset']+=2
    with pytest.raises(ValueError,match='original module replay'):verify_entry_attachment(path,contract,changed)
    launcher=tmp_path/'launch.py';launcher.write_text('from anchored import evolve\n')
    imported=plan_entry_attachment(launcher,contract)
    owner=next(m for m in imported['modules'] if m['attachment_target'])
    assert owner['mode']=='IMPORTED_SOURCE_POST_EXEC_CANDIDATE' and owner['module_code_key']==module['module_code_key']
    assert not imported['target_executed'] and not imported['source_rewritten']

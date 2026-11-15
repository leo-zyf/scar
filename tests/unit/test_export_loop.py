"""Export delegates full admission and retains lightweight hierarchical guards."""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json,platform,struct,sys
from types import SimpleNamespace as NS

import pytest

from scar.codegen import loop as complete
from scar.compiler.export_loop import export_loop
from scar.compiler.model import SourceRef
from scar.runtime.loop_contract import ASSUMPTIONS,TRUST_POLICY,canonical_sha256,validate_contract,validate_admission


def test_export_delegates_complete_loader_projects_guards_and_rejects_wrong_selection(tmp_path,monkeypatch):
    if sys.platform!='linux' or platform.machine().lower() not in ('x86_64','amd64') or struct.calcsize('P')!=8:
        pytest.skip('export profile is Linux x86_64')
    original=tmp_path/'full-artifact';original.mkdir();(original/'artifact.json').write_text('{}\n')
    files={'library':'kernel.so','loop':'loop.json','strategy':'strategy.json','execution_plan':'execution-plan.json',
        'source':'loop.cpp','loop_abi_header':'loop.h','array_abi_header':'array.h'}
    for key,name in files.items():(original/name).write_bytes(b'original-library' if key=='library' else b'ORIGINAL_IR_MARKER'*1000)
    source_path=tmp_path/'application.py';source_path.write_text('# original application source\n')
    digest=sha256(source_path.read_bytes()).hexdigest()
    def source(name,key):return SourceRef(str(source_path),digest,name,1,0,'pycode-v2:'+key*64,({'omitted_original_IR':'ORIGINAL_IR_MARKER'},))
    entry=source('entry','a');body=source('body','b');leaf=source('leaf','c')
    guard=lambda name,ref:NS(lookup_path=(name,),source=ref)
    loop=NS(source=entry,parameters=('steps','state'),state_parameter='state',trip_count_parameter='steps',
        body_callee=guard('shared',body),body=NS(source=body,calls=(NS(callee=guard('operation',leaf)),NS(callee=guard('operation',leaf)))))
    assumptions=sorted(ASSUMPTIONS);ids=dict(build_id='1'*64,loop_sha256='2'*64,strategy_sha256='3'*64,execution_plan_sha256='4'*64)
    native=dict(schema='scar.native-loop-contract',schema_version=1,loop_sha256=ids['loop_sha256'],strategy_sha256=ids['strategy_sha256'],
        execution_plan_sha256=ids['execution_plan_sha256'],trip_count=3,state_shape=[2,3],output_shape=[2,3],final_bytes=48,memory_budget_bytes=48)
    ids['contract_sha256']=canonical_sha256(native)
    report=dict(**ids,files=files,sha256={k:sha256((original/n).read_bytes()).hexdigest() for k,n in files.items()},
        external_assumptions=assumptions,loop_abi_version=1,state_shape=[2,3],final_bytes=48,trip_count=3,memory_budget_bytes=48,cpu_ids=[0],cpu_threads=1,
        floating_environment_policy='PRESERVE_CALLER_AND_WORKER_ENVIRONMENT_NOT_NUMPY_STATUS_FLAG_EQUIVALENCE')
    @dataclass
    class Binding:
        evidence: tuple=()
    artifact=NS(directory=original,loop=loop,execution_plan=NS(resource_binding=Binding()),strategy={'external_assumptions':assumptions},layout=None,report=report)
    calls=[]
    def admitted(path):calls.append(path);return artifact
    monkeypatch.setattr(complete,'load',admitted);monkeypatch.setattr(complete,'_contract',lambda *args:native)
    selection=dict(schema='scar.loop-measured-selection',schema_version=1,status='MEASURED',decision='TRANSFORM_UNDER_CONTRACT',selected=True,reasons=[],
        artifact={k:ids[k] for k in ('build_id','loop_sha256','strategy_sha256')},external_assumptions=assumptions,
        baseline={'count':3,'median':2.0},native={'count':3,'median':1.0},speedup=2.0,gain_fraction=0.5,minimum_gain=0.05,scope='EXACT_LOOP')
    selection_path=tmp_path/'selection.json';selection_path.write_text(json.dumps({'selection':selection,'IR_not_exported':'ORIGINAL_IR_MARKER'}))
    output=tmp_path/'runtime';result=export_loop(original,output,selection_path=selection_path)
    assert calls==[original] and result['full_original_loader_admission'] and not result['native_recompiled']
    runtime=validate_contract(json.loads((output/'runtime.json').read_bytes()));admission=validate_admission(json.loads((output/'admission.json').read_bytes()))
    assert runtime['parameters']==['steps','state'] and runtime['selection']==selection
    assert [g['environment_path'] for g in runtime['callee_guards']]==[['shared'],['shared','operation']]
    assert all(set(g['source'])=={'path','sha256','code_key','qualname'} for g in runtime['callee_guards'])
    assert set(p.name for p in output.iterdir())=={'runtime.json','admission.json','export-report.json','kernel.so'}
    assert (output/'kernel.so').read_bytes()==(original/'kernel.so').read_bytes()
    assert 'ORIGINAL_IR_MARKER' not in (output/'runtime.json').read_text()+(output/'admission.json').read_text()
    assert admission['runtime_contract_sha256']==sha256((output/'runtime.json').read_bytes()).hexdigest()
    assert result['expected_admission_sha256']==sha256((output/'admission.json').read_bytes()).hexdigest()
    assert admission['trust_policy']==TRUST_POLICY and any(r['role']=='selection:#/selection' for r in admission['evidence'])
    for index,field in enumerate(('artifact','external_assumptions')):
        wrong=deepcopy(selection)
        if field=='artifact':wrong[field]['build_id']='f'*64
        else:wrong[field]=[]
        selection_path.write_text(json.dumps(wrong));rejected=tmp_path/('rejected'+str(index))
        with pytest.raises(ValueError,match='selection'):export_loop(original,rejected,selection_path=selection_path)
        assert not rejected.exists()
    assert len(calls)==3

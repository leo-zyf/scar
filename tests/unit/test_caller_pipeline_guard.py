"""Guard dispatch only; the real ABI/export path has separate bounded evidence."""
from hashlib import sha256
import importlib.util
import os
import sys

import numpy as np

from scar.runtime.caller_pipeline_guard import (NativeCallerPipeline,guarded_caller_pipeline,GuardFailure,SCHEMA)
from scar.trace.code_snapshot import code_key


def test_plain_caller_guards_recompute_or_fall_back_once(tmp_path):
    path=tmp_path/'original.py'
    path.write_text('def first(x):\n    return x*x\ndef second(x,y):\n    return x+y\ndef caller(x,y):\n    return second(x,first(y))\ndef replacement(x):\n    return x+x\n')
    spec=importlib.util.spec_from_file_location('ordinary_caller_guard_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=lambda fn:dict(path=str(path),sha256=sha256(path.read_bytes()).hexdigest(),
        code_key=code_key(fn.__code__),qualname=fn.__qualname__)
    array=dict(shape=[2,3],nbytes=48,dtype='float64',layout='C_CONTIGUOUS')
    library=dict(library='test.so',sha256='0'*64,bytes=64)
    contract=dict(schema=SCHEMA,schema_version=1,source=source(module.caller),parameters=['x','y'],
        callees=[dict(global_name='first',source=source(module.first)),dict(global_name='second',source=source(module.second))],
        inputs=dict(x=array,y=array),output=array,native=dict(host=library,host_abi_version=1,operator_abi_version=1,
            pipeline_sha256='1'*64,memory_plan_sha256='2'*64,
            operators=[dict(**library,plan_sha256='3'*64,input_count=1),dict(**library,plan_sha256='4'*64,input_count=2)]),
        cpu=dict(ids=[min(os.sched_getaffinity(0))],os='linux',architecture='x86_64',pointer_bits=64,byteorder='little',required_isa='BASELINE_X86_64'),
        memory=dict(policy='UNBOUND',scope='INCREMENTAL_PAYLOAD_WORKSPACE_CAPACITIES_PLUS_FRESH_FINAL',
            workspace_bytes=48,final_bytes=48,peak_bytes=96,budget_bytes=96),
        cost_status='UNKNOWN',automatic_deployment_authorized=False,
        execution_policy='NO_OP_UNLESS_EXPLICIT_CANDIDATE_MEASUREMENT',defaults_policy='NO_DEFAULT_PARAMETERS',
        input_writes=False,fresh_output=True)
    # Deliberately controlled admission for a guard unit test, not an ELF or
    # source-purity certificate. No mathematical backend is built here.
    artifact=NativeCallerPipeline(tmp_path,contract,{},None,(),None,None);dispatches=[]
    def native(x,y):dispatches.append(1);return x+y*y
    artifact.run=native
    x=np.ones((2,3));y=np.full((2,3),.25)
    default=guarded_caller_pipeline(module.caller,artifact,exclusive_inputs=True)
    assert np.array_equal(default(x,y),module.caller(x,y))and not dispatches
    guarded=guarded_caller_pipeline(module.caller,artifact,exclusive_inputs=True,measure_candidate=True)
    assert np.array_equal(guarded(x,y),module.caller(x,y))and len(dispatches)==1
    y[:]+=.125
    assert np.array_equal(guarded(x,y),module.caller(x,y))and len(dispatches)==2
    def fallback_case(invoke,expected):
        before=guarded.scar_stats['fallback_calls'];native_before=len(dispatches)
        assert np.array_equal(invoke(),expected)
        assert guarded.scar_stats['fallback_calls']==before+1 and len(dispatches)==native_before
    saved=module.second;original_calls=[]
    def changed_second(a,b):original_calls.append(1);return a+b+.5
    module.second=changed_second
    fallback_case(lambda:guarded(x,y),x+y*y+.5)
    assert len(original_calls)==1
    module.second=saved
    module.first.__defaults__=(None,)
    fallback_case(lambda:guarded(x,y),x+y*y)
    module.first.__defaults__=None
    code=module.first.__code__;module.first.__code__=module.replacement.__code__
    fallback_case(lambda:guarded(x,y),x+y+y)
    module.first.__code__=code
    module.first.__signature__=object()
    fallback_case(lambda:guarded(x,y),x+y*y)
    del module.first.__signature__
    fallback_case(lambda:guarded(x.T,y.T),x.T+y.T*y.T)
    old_profile=sys.getprofile();sys.setprofile(lambda *args:None)
    try:fallback_case(lambda:guarded(x,y),x+y*y)
    finally:sys.setprofile(old_profile)
    monitoring=getattr(sys,'monitoring',None)
    if monitoring is not None:
        tool=next(i for i in range(6)if monitoring.get_tool(i)is None)
        monitoring.use_tool_id(tool,'scar_guard_test')
        try:
            monitoring.set_local_events(tool,module.caller.__code__,monitoring.events.LINE)
            fallback_case(lambda:guarded(x,y),x+y*y)
        finally:
            monitoring.set_local_events(tool,module.caller.__code__,0);monitoring.free_tool_id(tool)
    def decline(*args):raise GuardFailure('private output declined')
    artifact.run=decline
    fallback_case(lambda:guarded(x,y),x+y*y)

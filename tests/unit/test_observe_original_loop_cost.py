"""One original call is timed; later calls and all array values stay unrecorded."""
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from scar.compiler.observe_original_loop_cost import _digest,_source
from scar.compiler.model import SourceRef
from scar.runtime.compiled_deployment import entry_context
from scar.trace.code_snapshot import code_key


def test_worker_only_records_one_original_dispatch_and_metadata(tmp_path):
    source=tmp_path/'case.py';source.write_text('def step(state, fixed):\n    return state + fixed * fixed\n\ndef simulate(state, fixed, count):\n    for unused in range(count):\n        state = step(state, fixed)\n    return state\n')
    entry=tmp_path/'entry.py';entry.write_text('import numpy as np\nfrom case import simulate\nx=np.ones((2,3)); fixed=np.full((2,3),.25)\nx=simulate(x,fixed,3)\nx=simulate(x,fixed,3)\nprint(float(x.sum()))\n')
    spec=importlib.util.spec_from_file_location('fixture',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def source_ref(function):
        return _source(SourceRef(str(source),sha256(source.read_bytes()).hexdigest(),function.__qualname__,
            function.__code__.co_firstlineno,0,code_key(function.__code__)))
    env=dict(os.environ)
    for name in ('SCAR_CALL_BINDING_PLAN','SCAR_REGION_METADATA_PLAN','SCAR_TRACE_DIR','SCAR_OPTIMIZE'):env.pop(name,None)
    design=dict(phase='MEASUREMENT',index=0,environment_sha256=_digest(env),entry=str(entry),argv=[],
        entry_context=entry_context(entry,[]),sources=[source_ref(module.simulate),source_ref(module.step)],
        source=source_ref(module.simulate),leaf=dict(global_name='step',source=source_ref(module.step)),
        trip_count_parameter='count',trip_count=3,parameter_map=dict(state='state',fixed='fixed'),
        input_shapes=dict(state=[2,3],fixed=[2,3]))
    path=tmp_path/'design.json';path.write_text(json.dumps(design));sample=tmp_path/'sample.json'
    result=subprocess.run([sys.executable,'-m','scar.compiler.observe_original_loop_cost','--worker',str(path),
        '--sample',str(sample)],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert result.stdout.strip()=='8.25'
    record=json.loads(sample.read_text())
    assert record['recorded_calls']==1 and record['duration_ns']>0
    assert record['status']=='OBSERVED_ORIGINAL_RETURN' and not record['tensor_values_recorded']
    assert not record['function_internals_instrumented'] and not record['clean_whole_entry_benchmark']
    assert record['metadata']['trip_count']==3
    assert record['metadata']['arrays']['state']==dict(exact_ndarray=True,shape=[2,3],dtype='<f8',
        strides=[24,8],nbytes=48,c_contiguous=True,aligned=True)
    assert record['basis']=='OBSERVED_ORIGINAL_CALL_WITH_WRAPPER_DISPATCH_COST_MODEL'

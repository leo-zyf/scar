"""Lean delivery uses exact ELF bytes and no graph/compiler imports."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import os

import pytest

from scar.compiler.model import ArrayRegion,Expr,SourceRef
from scar.compiler.temporal_invariant_loop import build_temporal_invariant_loop
from scar.trace.code_snapshot import code_key


@pytest.mark.parametrize('parallel,composite',[(False,False),(True,False),(False,True)])
def test_light_runtime_pinned_bundle_imports_and_live_callable_guards(tmp_path,parallel,composite):
    source=tmp_path/'application.py';source.write_text('def step(state, fixed):\n    return state + fixed * fixed\n\ndef solve(state, fixed, count):\n    for _ in range(count):\n        state = step(state, fixed)\n    return state\n')
    if composite:
        source.write_text('def fixed_part(fixed):\n    return fixed * fixed\n\ndef step(state, fixed):\n    return state + fixed_part(fixed)\n\ndef relay(state, fixed):\n    return step(state, fixed)\n\ndef solve(state, fixed, count):\n    for _ in range(count):\n        state = relay(state, fixed)\n    return state\n')
    spec=importlib.util.spec_from_file_location('runtime_fixture',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    digest=sha256(source.read_bytes()).hexdigest()
    def ref(function):
        return SourceRef(str(source),digest,function.__qualname__,function.__code__.co_firstlineno,0,code_key(function.__code__))
    body=ArrayRegion('test:runtime-temporal',ref(module.step),('state','fixed'),
        (Expr('s','input',attrs=dict(name='state')),Expr('x','input',attrs=dict(name='fixed')),
         Expr('xx','mul',('x','x')),Expr('out','add',('s','xx'))),'out')
    cpus=[min(os.sched_getaffinity(0))]
    if parallel:
        from scar.analysis.target_graph import build_target_graph
        from scar.optimization.hardware import propose_cpu_allocations
        from scar.compiler.temporal_invariant_loop import build_parallel_temporal_invariant_loop
        if len(os.sched_getaffinity(0))<2:pytest.skip('two actual CPU affinity slots required')
        cpus=sorted(os.sched_getaffinity(0))[:2]
        inventory=dict(schema='scar.target-inventory',schema_version=1,sources=[],diagnostics=[],
            dram=dict(memory_total_bytes=2**30),
            cpu=dict(processors=[dict(cpu_id=cpu,package_id=0,core_id=i) for i,cpu in enumerate(cpus)],
                online_cpu_ids=cpus,model_names=['actual affinity unit fixture'],
                numa_nodes=[dict(node_id=0,cpu_ids=cpus,memory_total_bytes=2**30)]),
            gpus=[],paths=[],storage=dict(disks=[],block_devices=[]))
        path=tmp_path/'inventory.json';path.write_text(json.dumps(inventory))
        target=tmp_path/'hardware';build_target_graph(target,inventory=path)
        binding=next(p for p in propose_cpu_allocations(target,max_threads=2,memory_budget_bytes=4096)
            if len(p.cpu_ids)==2)
        artifact=build_parallel_temporal_invariant_loop(body,tmp_path/'native',carry_parameter='state',
            loop_interface=dict(carry_parameter='state',readonly_parameters=['fixed']),
            memory_budget_bytes=4096,resource_binding=binding)
    else:
        artifact=build_temporal_invariant_loop(body,tmp_path/'native',carry_parameter='state',
            loop_interface=dict(carry_parameter='state',readonly_parameters=['fixed']),memory_budget_bytes=4096)
    # This unit fixture checks runtime delivery independently of exporter/F4
    # admission; the real export path separately requires catalog replay.
    bundle=tmp_path/'bundle';bundle.mkdir();library=(artifact.directory/'temporal.so').read_bytes()
    (bundle/'temporal.so').write_bytes(library)
    small_source=lambda f:{key:getattr(ref(f),key) for key in ('path','sha256','code_key','qualname')}
    contract=dict(schema='scar.temporal-invariant-runtime-contract',schema_version=1,
        source=small_source(module.solve),leaf=dict(global_name='step',source=small_source(module.step)),
        native=dict(library='temporal.so',library_sha256=sha256(library).hexdigest(),
            operator_descriptor_abi_version=1,temporal_control_abi_version=1),
        parameters=['state','fixed','count'],parameter_map=dict(state='state',fixed='fixed'),
        input_order=['state','fixed'],carry_parameter='state',trip_count_parameter='count',trip_count=7,
        arrays={name:dict(shape=[3,5],dtype='float64',layout='C_CONTIGUOUS') for name in ('state','fixed')},
        memory_budget_bytes=4096,cpu=dict(ids=cpus,os='linux',
            architecture='x86_64',pointer_bits=64,byteorder='little',required_isa='BASELINE_X86_64'),
        profitability_status='UNPROVEN_NO_SOURCED_TIMING_MODEL',deployment_authorized=False)
    if parallel:
        contract['cpu']['threads']=2
        contract['native']['control_family']='CPP_OPENMP_EXPLICIT_ROW_PARTITION_TEMPORAL'
    if composite:
        contract.update(proof_kind='COMPOSITE',body_identity='DERIVED_FUSED_DAG_NOT_A_NEW_PYTHON_CALLABLE',
            root_callee_source=small_source(module.relay),
            leaf=dict(global_name='relay',source=small_source(module.relay)),
            source_proof=dict(schema='scar.composite-invariant-loop-proof'))
        contract['nested_callee_guards']=[dict(owner_source=small_source(owner),lookup_name=path[-1],
            target_source=small_source(target),global_path=path,lookup_semantics='FUNCTION_GLOBALS_AT_EACH_HOP')
            for owner,target,path in ((module.solve,module.relay,['relay']),
                (module.relay,module.step,['relay','step']),
                (module.step,module.fixed_part,['relay','step','fixed_part']))]
    raw=json.dumps(contract).encode();(bundle/'runtime.json').write_bytes(raw)
    admission=dict(schema='scar.temporal-invariant-runtime-admission',schema_version=1,
        runtime_sha256=sha256(raw).hexdigest(),library_sha256=sha256(library).hexdigest(),
        trust_policy='EXPLICIT_EXPECTED_ADMISSION_SHA256_FROM_EXPORT_PRODUCER_NOT_A_SIGNATURE')
    raw=json.dumps(admission).encode();(bundle/'admission.json').write_bytes(raw);pin=sha256(raw).hexdigest()
    script=r'''
import builtins,importlib.abc,importlib.util,json,pathlib,sys
forbidden=('scar.analysis','scar.compiler','scar.codegen','scar.optimization','scar.ir')
class Stop(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if any(fullname==p or fullname.startswith(p+'.') for p in forbidden):
            raise AssertionError('forbidden runtime import: '+fullname)
sys.meta_path.insert(0,Stop())
from scar.runtime.native_temporal_invariant import load,guarded_temporal_invariant
assert 'numpy' not in sys.modules
artifact=load(sys.argv[1],expected_admission_sha256=sys.argv[2])
assert 'numpy' not in sys.modules
source=pathlib.Path(sys.argv[3]);spec=importlib.util.spec_from_file_location('case',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
wrapped=guarded_temporal_invariant(module.solve,artifact,exclusive_inputs=True,trial=True)
import numpy as np
state=np.ones((3,5));fixed=np.full((3,5),0.125)
result=wrapped(state,fixed,7)
assert np.array_equal(result,module.solve(state,fixed,7))
assert wrapped.scar_stats['native_calls']==1
assert wrapped(state,object(),0) is state
class Foreign:
    @property
    def shape(self):raise AssertionError('foreign shape effect executed')
try:wrapped(state,Foreign(),7)
except TypeError:pass
else:raise AssertionError('original unsupported arithmetic must raise')
saved=builtins.range
try:
    builtins.range=lambda n:saved(0)
    assert wrapped(state,fixed,7) is state
finally:builtins.range=saved
assert wrapped.scar_stats['native_calls']==1
if artifact.contract.get('proof_kind')=='COMPOSITE':
    previous=module.fixed_part
    module.fixed_part=lambda fixed:fixed*0
    before=wrapped.scar_stats['fallback_calls']
    assert np.array_equal(wrapped(state,fixed,7),module.solve(state,fixed,7))
    assert wrapped.scar_stats['fallback_calls']==before+1
    module.fixed_part=previous
    module.fixed_part.__defaults__=(None,)
    before=wrapped.scar_stats['fallback_calls']
    assert np.array_equal(wrapped(state,fixed,7),module.solve(state,fixed,7))
    assert wrapped.scar_stats['fallback_calls']==before+1
    module.fixed_part.__defaults__=None
module.step=lambda a,b:a-b
assert np.array_equal(wrapped(state,fixed,7),module.solve(state,fixed,7))
assert not any(any(n==p or n.startswith(p+'.') for p in forbidden) for n in sys.modules)
print(json.dumps(dict(no_forbidden_imports=True,numpy_lazy=True,native_calls=wrapped.scar_stats['native_calls'])))
'''
    result=subprocess.run([sys.executable,'-c',script,str(bundle),pin,str(source)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads(result.stdout)
    assert report==dict(no_forbidden_imports=True,numpy_lazy=True,native_calls=1)

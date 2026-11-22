"""Actual CPU-team placement, strict recurrence and private-output fallback."""
from dataclasses import replace
from hashlib import sha256
import json
import os

import numpy as np
import pytest

from scar.analysis.target_graph import build_target_graph
from scar.compiler.model import ArrayRegion,Expr,SourceRef
from scar.compiler.temporal_invariant_loop import (
    build_parallel_temporal_invariant_loop, load_temporal_invariant_loop as load_parallel_temporal_invariant_loop)
from scar.optimization.hardware import propose_cpu_allocations


def test_parallel_temporal_workers_restore_actual_cpu_and_fenv(tmp_path):
    allowed=sorted(os.sched_getaffinity(0))
    if len(allowed)<2:pytest.skip('two actual CPU affinity slots required')
    cpus=allowed[:2]
    inventory=dict(schema='scar.target-inventory',schema_version=1,sources=[],diagnostics=[],
        dram=dict(memory_total_bytes=2**30),
        cpu=dict(processors=[dict(cpu_id=cpu,package_id=0,core_id=i) for i,cpu in enumerate(cpus)],
            online_cpu_ids=cpus,model_names=['declared actual-affinity test CPUs'],
            numa_nodes=[dict(node_id=0,cpu_ids=cpus,memory_total_bytes=2**30)]),
        gpus=[],paths=[],storage=dict(disks=[],block_devices=[]))
    path=tmp_path/'inventory.json';path.write_text(json.dumps(inventory))
    target=tmp_path/'hardware';build_target_graph(target,inventory=path)
    placement=next(p for p in propose_cpu_allocations(target,max_threads=2,memory_budget_bytes=4096)
        if len(p.cpu_ids)==2)
    source=tmp_path/'original.py';source.write_text('def step(state, fixed):\n    return state + fixed * fixed\n')
    ref=SourceRef(str(source),sha256(source.read_bytes()).hexdigest(),'step',1,0,'pycode-v2:'+'6'*64)
    body=ArrayRegion('test:parallel-temporal',ref,('state','fixed'),
        (Expr('s','input',attrs=dict(name='state')),Expr('x','input',attrs=dict(name='fixed')),
         Expr('xx','mul',('x','x')),Expr('out','add',('s','xx'))),'out')
    interface=dict(carry_parameter='state',readonly_parameters=['fixed'])
    artifact=build_parallel_temporal_invariant_loop(body,tmp_path/'native',carry_parameter='state',
        loop_interface=interface,memory_budget_bytes=4096,resource_binding=placement)
    state=np.ones((7,5));fixed=np.full((7,5),0.125);expected=state.copy()
    for _ in range(11):expected=expected+fixed*fixed
    result=artifact.run(dict(state=state,fixed=fixed),11)
    assert np.array_equal(result,expected) and np.array_equal(state,np.ones((7,5)))
    observed=artifact.report['last_execution'];assert observed['element_preheaders']==35
    assert observed['element_iterations']==385 and observed['status']==0
    assert len(observed['workers'])==2
    assert [(w['row_begin'],w['row_end']) for w in observed['workers']]==[(0,4),(4,7)]
    assert [w['observed_cpu'] for w in observed['workers']]==list(placement.cpu_ids)
    assert all(w['binding_kept'] and w['original_affinity_restored'] and w['original_fenv_restored'] for w in observed['workers'])
    assert sorted(os.sched_getaffinity(0))==allowed
    assert artifact.run(dict(state=state),0) is state
    artifact=load_parallel_temporal_invariant_loop(tmp_path/'native')
    fixed[:]=1e200;fallback=[]
    def original():
        fallback.append(1)
        with np.errstate(over='ignore'):return state+fixed*fixed
    result=artifact.run(dict(state=state,fixed=fixed),11,original_loop=original)
    assert len(fallback)==1 and np.isinf(result).all()
    assert artifact.report['last_execution']['fallback'] and sorted(os.sched_getaffinity(0))==allowed
    with pytest.raises(ValueError):
        build_parallel_temporal_invariant_loop(body,tmp_path/'invalid',carry_parameter='state',
            loop_interface=interface,memory_budget_bytes=4096,
            resource_binding=replace(placement,cpu_ids=(max(allowed)+1,max(allowed)+2)))

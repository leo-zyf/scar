"""Native repeated-call control preserves commit progress across guard failure."""
from hashlib import sha256

import numpy as np
import pytest

from test_cpu_hardware_placement import hardware
from scar.compiler.model import Expr,SourceRef
from scar.compiler.stateful_model import StatefulInput,StatefulAction,StatefulRegion,REQUIRED_GUARDS
from scar.compiler.stateful_repeat import build_repeat,load_repeat,RepeatGuardFailure
from scar.codegen.stateful import build
from scar.optimization.hardware import propose_cpu_allocations
from scar.optimization.stateful_schedule import plan_stateful_region


def test_repeat_uses_original_guards_and_resumes_only_uncommitted_suffix(hardware,tmp_path):
    source=tmp_path/'step.py';source.write_text('def step(state, factor):\n    state *= factor\n')
    ref=SourceRef(str(source),sha256(source.read_bytes()).hexdigest(),'step',1,0,'pycode-v2:'+'d'*64)
    region=StatefulRegion('test:repeated-stateful',ref,('state','factor'),
        (StatefulInput('state',('state',),'array'),StatefulInput('factor',('factor',),'scalar')),
        (Expr('x','input',attrs=dict(name='state',kind='array',epoch=0)),
         Expr('f','input',attrs=dict(name='factor',kind='scalar',epoch=None)),
         Expr('y','mul',('x','f'))),
        (StatefulAction('eval','evaluate','y'),StatefulAction('write','write','y','state',None,0,1,'assign')),
        guards=REQUIRED_GUARDS)
    binding=propose_cpu_allocations(hardware,max_threads=1,memory_budget_bytes=4096)[0]
    plan=plan_stateful_region(region,input_shapes={'state':[2,3]},cpu_threads=1,
        memory_budget_bytes=4096,resource_binding=binding)
    base=build(region,tmp_path/'native',plan=plan)
    repeated=build_repeat(base.directory,tmp_path/'repeat')
    assert not repeated.report['native_math_changed']
    assert not repeated.report['target_loop_replacement_authorized']
    assert not repeated.report['per_iteration_python_callbacks']
    x=np.ones((2,3));result=repeated.run({'state':x},{'factor':2.0},5)
    assert result==dict(requested=5,completed=5,native_calls=5) and np.array_equal(x,np.full((2,3),32.0))
    assert load_repeat(tmp_path/'repeat').report==repeated.report
    assert repeated.run({}, {},0)['completed']==0
    with pytest.raises(ValueError,match='exact nonnegative'):repeated.run({}, {},True)
    with pytest.raises(RepeatGuardFailure) as rejected:repeated.run({'state':x[:,:2]},{'factor':2.0},4)
    assert rejected.value.completed==0
    x[:]=1e100
    with pytest.raises(RepeatGuardFailure) as partial:repeated.run({'state':x},{'factor':1e100},4)
    assert partial.value.completed==2 and partial.value.status==3
    assert np.array_equal(x,np.full((2,3),(1e100*1e100)*1e100))
    x[:]=1e100;calls=[]
    def fallback():
        calls.append(1)
        with np.errstate(over='ignore'):x.__imul__(1e100)
    resumed=repeated.run_with_fallback({'state':x},{'factor':1e100},4,fallback)
    assert resumed['native_calls']==2 and resumed['fallback_calls']==2 and len(calls)==2
    assert np.isinf(x).all()

"""Static ownership follows existing local import candidates, never execution."""
from hashlib import sha256
import json,sys

from scar.compiler.runtime_entry import inspect_entry,main
from scar.runtime.loop_contract import ASSUMPTIONS,canonical_sha256,validate_contract


def test_static_entry_package_chain_source_entry_budgets_and_version_frontiers(tmp_path):
    entry=tmp_path/'run.py';package=tmp_path/'unitpkg';package.mkdir()
    entry.write_text("raise RuntimeError('must not execute')\nfrom unitpkg.bridge import route\nimport unavailable_external\n"
        "def deferred(name):\n    return __import__(name)\n")
    (package/'__init__.py').write_text('from .bridge import route\n')
    (package/'bridge.py').write_text('from .kernel import evolve as route\n')
    kernel=package/'kernel.py';kernel.write_text('from .ops import leaf\ndef evolve(state,steps):\n    return leaf(state)\n')
    ops=package/'ops.py';ops.write_text('def leaf(state):\n    return state\n')
    source=lambda p,name,key:dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest(),qualname=name,code_key='pycode-v2:'+key*64)
    ids=dict(build_id='1'*64,loop_sha256='2'*64,strategy_sha256='3'*64,execution_plan_sha256='4'*64)
    native=dict(schema='scar.native-loop-contract',schema_version=1,loop_sha256=ids['loop_sha256'],strategy_sha256=ids['strategy_sha256'],
        execution_plan_sha256=ids['execution_plan_sha256'],trip_count=2,state_shape=[2,2],output_shape=[2,2],final_bytes=32,memory_budget_bytes=32)
    ids['contract_sha256']=canonical_sha256(native);assumptions=sorted(ASSUMPTIONS)
    contract=validate_contract(dict(schema='scar.loop-runtime-contract',schema_version=1,profile='cpu-f64-rank2-loop-v1',
        native=dict(library='kernel.so',library_sha256='5'*64,loop_abi_version=1,ids=ids,contract=native),
        source=source(kernel,'evolve','a'),parameters=['state','steps'],state_parameter='state',trip_count_parameter='steps',
        callee_guards=[dict(environment_path=['leaf'],source=source(ops,'leaf','b'))],array=dict(shape=[2,2],dtype='float64',device='cpu',layout='C',alignment_bytes=8,final_bytes=32),
        trip_count=2,memory_budget_bytes=32,cpu=dict(ids=[0],threads=1,architecture='x86_64',os='linux',pointer_bits=64,byteorder='little',required_isa='BASELINE_X86_64'),
        external_assumptions=assumptions,floating_environment_policy='PRESERVE_CALLER_AND_WORKER_ENVIRONMENT_NOT_NUMPY_STATUS_FLAG_EQUIVALENCE',
        selection=dict(decision='TRANSFORM_UNDER_CONTRACT',selected=True,artifact={k:ids[k] for k in ('build_id','loop_sha256','strategy_sha256')},external_assumptions=assumptions)))
    assert 'unitpkg' not in sys.modules
    report=inspect_entry(entry,contract)
    assert report['status']=='SOURCE_OWNERS_IDENTIFIED' and report['counts']['files']==5
    bindings={row['path']:row for row in report['source_bindings']}
    assert [r['name'] for r in bindings[str(kernel)]['module_candidates']]==['unitpkg.kernel']
    assert [r['name'] for r in bindings[str(ops)]['module_candidates']]==['unitpkg.ops']
    assert all(not row['code_identity_verified'] and not row['runtime_module_selected'] for row in bindings.values())
    assert 'DYNAMIC_IMPORT' in {row['reason'] for row in report['unresolved']}
    assert any(row['status']=='EXTERNAL_OR_UNAVAILABLE' for row in report['imports'])
    assert 'unitpkg' not in sys.modules and not report['application_imported'] and not report['loops_unrolled']
    own=inspect_entry(kernel,contract)
    binding=next(row for row in own['source_bindings'] if row['path']==str(kernel))
    assert binding['status']=='ENTRY_SOURCE' and binding['module_candidates'][0]['name']=='__main__'
    bounded=inspect_entry(entry,contract,max_files=1)
    assert bounded['counts']['files']==1
    assert all(row['status']=='UNRESOLVED_SOURCE_OWNER' for row in bounded['source_bindings'])
    assert any(row['reason']=='FILE_BUDGET' for row in bounded['unresolved'])
    kernel.write_text(kernel.read_text()+'# changed version\n')
    changed=inspect_entry(entry,contract)
    assert next(row for row in changed['source_bindings'] if row['path']==str(kernel))['status']=='SOURCE_VERSION_MISMATCH'
    contract_path=tmp_path/'runtime.json';contract_path.write_text(json.dumps(contract));out=tmp_path/'inspection.json'
    assert main(['--entry',str(entry),'--contract',str(contract_path),'--out',str(out)])==0
    assert json.loads(out.read_text())['status']=='UNRESOLVED_SOURCE_OWNERS'

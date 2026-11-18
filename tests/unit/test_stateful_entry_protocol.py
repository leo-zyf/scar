"""Entry templates bind metadata and loaded implementation code, never execution."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from test_cpu_hardware_placement import hardware
from test_stateful_stage_dag import _inputs, _region, _render
from scar.analysis.stateful_entry_protocol import (derive_stateful_entry_protocol,
    verify_stateful_entry_protocol, _owner)
from scar.codegen import stateful
from scar.compiler.model import Expr, SourceRef
from scar.compiler.stateful_model import document, stateful_digest


def test_protocol_owner_flags_selection_and_code_identity_are_metadata_only(hardware,tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('protocol derivation must not compile, load ELF/provider or execute target code')
    for owner,name in ((stateful,'load'),(stateful,'build'),(stateful,'_providers'),(stateful,'_admit')):
        monkeypatch.setattr(owner,name,forbidden)
    monkeypatch.setattr('subprocess.Popen',forbidden)
    monkeypatch.setattr('ctypes.CDLL',forbidden)
    original=tmp_path/'original.py'
    original.write_text('class Ordinary:\n    def step(state):\n        state.left[:] = state.left + state.right\n')
    source=SourceRef(str(original),sha256(original.read_bytes()).hexdigest(),'Ordinary.step',2,4,'pycode-v2:'+'1'*64)
    region=replace(_region((*_inputs(),Expr('sum','add',('x','y'))),('sum','WRITE')),source=source)
    plan,_,binding,_=_render(region,hardware)
    provider=dict(numpy_version='2.5.2',ndarray_c_abi=33554432,capsule_abi='numpy_1.24_ufunc_call_info',
        loaded_binary=dict(path=str(tmp_path/'unloaded-provider.so'),sha256='0'*64,bytes=1),
        dtype_signature=['float64']*3,fixed_strides=[8,0,8],requires_pyapi=False,
        API_stability='EXPERIMENTAL_EXACT_NUMPY_VERSION_AND_BINARY_PIN',
        execution='ORIGINAL_NUMPY_UFUNC_STRIDED_LOOP_SERIAL_PROVIDER_STAGE')
    core=dict(region_sha256=stateful_digest(region),plan_sha256=stateful._digest(plan),provider_runtime=provider)
    files=dict(region='region.json',plan='plan.json',source='stateful.cpp',library='unloaded.so',
        **{'stateful_abi_v1.h':'stateful_abi_v1.h','scar_abi_v1.h':'scar_abi_v1.h'})
    data=dict(region=json.dumps(document(region)).encode(),plan=json.dumps(plan).encode(),
        source=stateful._source(region,plan,binding,core).encode(),library=b'UNEXECUTED_METADATA_FIXTURE')
    for name in ('stateful_abi_v1.h','scar_abi_v1.h'):data[name]=(stateful._HERE/name).read_bytes()
    for key,name in files.items():(tmp_path/name).write_bytes(data[key])
    hashes={key:sha256(value).hexdigest()for key,value in data.items()}
    compiler=dict(argv=['metadata-only-no-compiler'],identity='NOT_QUERIED',options=['-std=c++20'])
    implementation=stateful.implementation_sha256()
    build=stateful._digest(dict(source=hashes['source'],headers={name:hashes[name]for name in ('stateful_abi_v1.h','scar_abi_v1.h')},
        compiler=compiler['argv'],identity=compiler['identity'],options=compiler['options'],
        implementation_sha256=implementation,contract=core))
    manifest=dict(schema=stateful.SCHEMA,schema_version=1,stateful_abi_version=1,files=files,sha256=hashes,
        **core,contract_sha256=stateful._digest(core),build_id=build,implementation_sha256=implementation,
        compiler=compiler,cpu_ids=list(binding.cpu_ids),external_assumptions=plan['external_assumptions'],
        **{key:plan[key]for key in ('cpu_threads','external_array_bytes','workspace_payload_bytes',
            'total_region_array_payload_bytes','memory_budget_bytes')})
    path=tmp_path/'artifact.json';path.write_text(json.dumps(manifest))
    report=derive_stateful_entry_protocol(path,trial=True,exclusive_inputs=True)
    assert verify_stateful_entry_protocol(report,path,trial=True,exclusive_inputs=True)==report
    assert report['owner']['policy']=='EXACT_RECEIVER_CLASS_PER_INSTALLATION'
    assert report['owner']['symbolic_source_owner']['qualname']=='Ordinary'
    assert report['normal_success_only'] and report['fallback_cost_ns']is None
    assert report['required_sample_guard_delta']==dict(native_calls=1,fallback_calls=0)
    assert not report['compact_artifact_protocol_supported'] and not report['source_F4_admitted']
    assert not report['native_loaded'] and not report['cost_qualified'] and not report['deployment_authorized']
    assert report['shared_call_code_key'].startswith('pycode-v2:')
    assert {'adapter_init','adapter_run','artifact_run','shared_guard_call','shared_guard_factory'}<=set(report['implementation_functions'])
    forged=deepcopy(report);forged['configuration']['trial']=False
    with pytest.raises(ValueError,match='protocol differs'):
        verify_stateful_entry_protocol(forged,path,trial=True,exclusive_inputs=True)
    with pytest.raises(ValueError,match='real retained measured selection'):
        derive_stateful_entry_protocol(path,trial=False,exclusive_inputs=True)
    with pytest.raises(ValueError,match='retained validation evidence'):
        derive_stateful_entry_protocol(path,trial=False,exclusive_inputs=True,
            selection=dict(decision='TRANSFORM_UNDER_CONTRACT',selected=True,artifact=dict(build_id=build)))
    changed=derive_stateful_entry_protocol(path,trial=True,exclusive_inputs=False)
    assert changed['id']!=report['id'] and not changed['native_success_branch_possible_under_runtime_guards']
    for text,qualname,line,policy in (
            ('def step(state):\n    pass\n','step',1,'NO_RECEIVER_CLASS_CHECK'),
            ('class Ordinary:\n    @staticmethod\n    def step(state):\n        pass\n','Ordinary.step',2,'NO_RECEIVER_CLASS_CHECK')):
        original.write_text(text)
        ref=replace(source,sha256=sha256(original.read_bytes()).hexdigest(),qualname=qualname,line=line)
        assert _owner(ref)['policy']==policy
    original.write_text('class Ordinary:\n    @classmethod\n    def step(cls):\n        pass\n')
    with pytest.raises(ValueError,match='classmethods'):
        _owner(replace(source,sha256=sha256(original.read_bytes()).hexdigest(),line=2))
    with pytest.raises(ValueError,match='closures'):
        _owner(replace(source,sha256=sha256(original.read_bytes()).hexdigest(),qualname='outer.<locals>.step'))

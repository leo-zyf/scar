"""Local entry metadata never installs a global function-call hook."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from types import CodeType

import pytest

from scar.compiler.model import SourceRef,ArrayRegion,Expr,ExecutionPlan,write_plan
from scar.trace.code_snapshot import code_key


def _plan(tmp_path,*,mismatch=False):
    source=tmp_path/'ordinary.py'
    source.write_text('def untouched(x):\n    return x\n\ndef chosen(x, *, scale=1):\n    return -x * scale\n')
    root=compile(source.read_bytes(),str(source),'exec')
    code=next(value for value in root.co_consts if type(value)is CodeType and value.co_name=='chosen')
    ref=SourceRef(str(source),sha256(source.read_bytes()).hexdigest(),code.co_qualname,
        code.co_firstlineno,0,'pycode-v2:'+'0'*64 if mismatch else code_key(code))
    region=ArrayRegion('region',ref,('x','scale'),(Expr('x','input',attrs={'parameter':'x'}),Expr('y','neg',('x',))),'y')
    path=tmp_path/'plan.json';write_plan(ExecutionPlan(region),path)
    return source,path


def _run(script,tmp_path,source,plan):
    environment=dict(os.environ)
    for name in ('SCAR_REGION_METADATA_PLAN','SCAR_REGION_METADATA_OUT','SCAR_TRACE_DIR','SCAR_OPTIMIZE'):
        environment.pop(name,None)
    result=subprocess.run([sys.executable,'-c',script,str(source),str(plan),str(tmp_path/'out')],
        cwd=Path(__file__).resolve().parents[2],env=environment,text=True,capture_output=True,timeout=20)
    assert result.returncode==0,result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(not hasattr(sys,'monitoring'),reason='local monitoring requires Python 3.12+')
def test_exec_selected_py_start_reads_bound_parameters_and_flushes_two_call_window(tmp_path):
    source,plan=_plan(tmp_path)
    report=_run('''
import json,os,sys
from pathlib import Path
import numpy as np
from scar.trace.region_metadata import install
source,plan,out=sys.argv[1:];os.environ['SCAR_REGION_METADATA_PLAN']=plan;os.environ['SCAR_REGION_METADATA_OUT']=out
before=sys.getprofile();collector=install();namespace={}
exec(compile(Path(source).read_bytes(),source,'exec'),namespace)
target=namespace['chosen'];other=namespace['untouched'];monitor=sys.monitoring
assert monitor.get_events(collector.tool_id)==0
assert monitor.get_local_events(collector.tool_id,other.__code__)==0
assert monitor.get_local_events(collector.tool_id,target.__code__)==monitor.events.PY_START
value=np.ones((5,7),dtype=np.float64)
for _ in range(3):assert np.array_equal(target(value,scale=2),-2*value)
assert sys.getprofile()is before and monitor.get_events(collector.tool_id)==0
assert monitor.get_local_events(collector.tool_id,target.__code__)==0
path=Path(out)/('metadata.'+str(os.getpid())+'.json')
report=json.loads(path.read_text())
assert report['status']=='COMPLETE' and report['collector']['window_flushed']
assert report['collector']['callbacks']==2 and len(report['records'])==2
assert report['records'][0]['inputs']['x']['shape']==[5,7]
assert report['records'][0]['inputs']['scale']['kind']=='UNSUPPORTED_METADATA_TYPE'
assert not report['tensor_contents_recorded']
collector.close();print(json.dumps(report))
''',tmp_path,source,plan)
    assert report['matched']==1 and report['errors']==[]


@pytest.mark.skipif(not hasattr(sys,'monitoring'),reason='local monitoring requires Python 3.12+')
def test_code_mismatch_and_missing_monitoring_are_explicit_without_breaking_target(tmp_path):
    source,plan=_plan(tmp_path,mismatch=True)
    report=_run('''
import json,os,sys
from pathlib import Path
from scar.trace.region_metadata import install
source,plan,out=sys.argv[1:];collector=install(plan,out);namespace={}
exec(compile(Path(source).read_bytes(),source,'exec'),namespace)
assert namespace['chosen'](3,scale=2)==-6
assert sys.monitoring.get_local_events(collector.tool_id,namespace['chosen'].__code__)==0
collector.close();report=json.loads((Path(out)/('metadata.'+str(os.getpid())+'.json')).read_text())
assert report['matched']==0 and not report['records']
assert any(error['error']=='CODE_ID_MISMATCH' for error in report['errors'])
monitor=sys.monitoring;sys.monitoring=None
unsupported=install(plan,str(Path(out)/'unsupported'));unsupported.close();sys.monitoring=monitor
missing=json.loads((Path(out)/'unsupported'/('metadata.'+str(os.getpid())+'.json')).read_text())
assert missing['status']=='UNAVAILABLE' and not missing['records']
assert any(error['error']=='SYS_MONITORING_UNAVAILABLE' for error in missing['errors'])
assert namespace['chosen'](4)==-4
# A collector exception at entry also leaves the original function intact.
from scar.trace.code_snapshot import code_key
from scar.compiler.input_metadata import InputMetadataObserver
document=json.loads(Path(plan).read_text());document['region']['source']['code_key']=code_key(namespace['chosen'].__code__)
correct=Path(out).parent/'correct.json';correct.write_text(json.dumps(document))
broken=install(correct,str(Path(out)/'broken'))
class BrokenObserver(InputMetadataObserver):
    def __call__(self,*args):raise RuntimeError('collector only')
broken.observer=BrokenObserver();exec(compile(Path(source).read_bytes(),source,'exec'),namespace)
assert namespace['chosen'](5)==-5
assert broken.status=='ERROR' and broken.callbacks==1
assert sys.monitoring.get_local_events(broken.tool_id,namespace['chosen'].__code__)==0
broken.close()
# External control signals must retain their normal exception semantics.
interrupted=install(correct,str(Path(out)/'interrupted'))
class InterruptObserver(InputMetadataObserver):
    def __call__(self,*args):raise KeyboardInterrupt
interrupted.observer=InterruptObserver();exec(compile(Path(source).read_bytes(),source,'exec'),namespace)
try:namespace['chosen'](6)
except KeyboardInterrupt:pass
else:raise AssertionError('metadata callback swallowed the external interrupt')
interrupted.close()
print(json.dumps(report))
''',tmp_path,source,plan)
    assert report['collector']['callbacks']==0

"""Run an original program with bounded, code-selected input metadata capture."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .input_metadata import read_input_boundary

def observe_inputs(plan_path,command,out,*,max_seconds=None,stop_when_collected=False):
    plan_path=Path(plan_path).absolute();read_input_boundary(plan_path)
    out=Path(out).absolute()
    if out.exists():raise ValueError('fresh observation output required')
    if not command:raise ValueError('original program command required')
    if max_seconds is not None and max_seconds<=0:raise ValueError('positive observation time budget required')
    out.mkdir(parents=True)
    env=dict(os.environ)
    env['SCAR_REGION_METADATA_PLAN']=str(plan_path)
    env['SCAR_REGION_METADATA_OUT']=str(out)
    env.pop('SCAR_OPTIMIZE',None)
    root=str(Path(__file__).resolve().parents[2])
    env['PYTHONPATH']=root+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    started=time.monotonic();reason=None
    process=subprocess.Popen(command,env=env,start_new_session=True)
    while process.poll() is None:
        if max_seconds is not None and time.monotonic()-started>=max_seconds:
            reason='TIME_BUDGET_PREFIX_STOP';break
        if stop_when_collected:
            for path in out.glob('metadata.*.json'):
                try:document=json.loads(path.read_bytes())
                except (ValueError,OSError):continue
                if document.get('collector',{}).get('status')=='COMPLETE':
                    reason='METADATA_WINDOW_PREFIX_STOP';break
            if reason is not None:break
        time.sleep(0.05)
    if reason is not None and process.poll() is None:
        # Only the explicitly launched child session is interrupted. The
        # resulting records describe a prefix, never a full-episode result.
        for signum,timeout in ((signal.SIGINT,5),(signal.SIGTERM,2),(signal.SIGKILL,2)):
            try:os.killpg(process.pid,signum)
            except ProcessLookupError:break
            try:process.wait(timeout=timeout);break
            except subprocess.TimeoutExpired:continue
    returncode=process.wait()
    records=[];collectors=[]
    for path in sorted(out.glob('metadata.*.json')):
        document=json.loads(path.read_bytes())
        if (document.get('schema'),document.get('schema_version'))!=('scar.region-input-metadata',1):
            raise ValueError('unexpected child metadata schema')
        records.extend(document['records']);collectors.append(dict(path=str(path),status=document.get('collector')))
    merged=dict(schema='scar.region-input-metadata',schema_version=1,records=records,
        max_calls_per_code=2,tensor_contents_recorded=False,
        observation_ids_are_trace_invocation_keys=False,logical_versions_inferred=False)
    (out/'inputs.json').write_text(json.dumps(merged,indent=2,allow_nan=False)+'\n')
    report=dict(schema='scar.input-metadata-run',schema_version=1,command=command,
        child_returncode=returncode,stop_reason=reason,full_run_completed=reason is None and returncode==0,
        seconds=time.monotonic()-started,observations=len(records),collectors=collectors,
        metadata=str(out/'inputs.json'),transformations_disabled=True,performance_measurement=False)
    report['collection_status']='OBSERVED_INPUT_METADATA' if records else 'NO_INPUT_OBSERVATIONS'
    (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--out',required=True)
    p.add_argument('--max-seconds',type=float);p.add_argument('--stop-when-collected',action='store_true')
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    result=observe_inputs(a.plan,command,a.out,max_seconds=a.max_seconds,stop_when_collected=a.stop_when_collected)
    print(json.dumps(result))
    if result['child_returncode'] and result['stop_reason'] is None:raise SystemExit(result['child_returncode'])
    if not result['observations']:raise SystemExit(2)

if __name__=='__main__':main()

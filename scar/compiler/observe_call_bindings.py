"""Fill original static CALL target fields with a bounded local-event window."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .call_binding import (plan_call_bindings, plan_program_call_bindings, read_call_binding_plan,
                          write_call_binding_plan, join_call_bindings)


def observe_call_bindings(command, out, *, plan_path=None, source=None, catalog=None,
                          entry=None, max_targets_per_site=1, max_seconds=60,
                          all_sources=False,include_modules=False,environment=None,cwd=None,
                          stdin=None,stdout=None,stderr=None):
    if plan_path is not None:
        if source is not None or catalog is not None or all_sources or include_modules:
            raise ValueError('read a plan or select original source calls')
        plan=read_call_binding_plan(plan_path)
    elif all_sources or include_modules or isinstance(source,(list,tuple)) and len(source)>1:
        if catalog is None or entry is not None:
            raise ValueError('program-wide CALL selection requires a catalog and no declaration filter')
        sources=None if all_sources else ([source] if isinstance(source,(str,Path)) else source)
        plan=plan_program_call_bindings(catalog,sources,include_modules=all_sources or include_modules,
            max_targets_per_site=max_targets_per_site)
    else:
        if isinstance(source,(list,tuple)):source=source[0] if source else None
        if source is None or catalog is None: raise ValueError('original source and F4 catalog required')
        plan=plan_call_bindings(catalog,source,entry=entry,max_targets_per_site=max_targets_per_site)
    if not command or max_seconds <= 0: raise ValueError('original command and positive time budget required')
    out=Path(out).absolute()
    if out.exists(): raise ValueError('fresh CALL observation output required')
    out.mkdir(parents=True);path=out/'plan.json';write_call_binding_plan(plan,path)
    env=dict(os.environ if environment is None else environment)
    for name in ('SCAR_TRACE_DIR','SCAR_OPTIMIZE','SCAR_REGION_METADATA_PLAN','SCAR_REGION_METADATA_OUT'):
        env.pop(name,None)
    env['SCAR_CALL_BINDING_PLAN']=str(path);env['SCAR_CALL_BINDING_OUT']=str(out)
    if environment is None:
        root=str(Path(__file__).resolve().parents[2])
        env['PYTHONPATH']=root+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    started=time.monotonic();reason=None
    process=subprocess.Popen(command,env=env,cwd=cwd,stdin=stdin,stdout=stdout,stderr=stderr,start_new_session=True)
    try: process.wait(timeout=max_seconds)
    except subprocess.TimeoutExpired:
        reason='TIME_BUDGET_PREFIX_STOP'
        for signum,timeout in ((signal.SIGINT,5),(signal.SIGTERM,2),(signal.SIGKILL,2)):
            try: os.killpg(process.pid,signum)
            except ProcessLookupError: break
            try: process.wait(timeout=timeout);break
            except subprocess.TimeoutExpired: pass
    code=process.wait()
    observations=sorted(out.glob('call-bindings.*.json'))
    joined=join_call_bindings(plan,observations)
    (out/'join.json').write_text(json.dumps(joined,indent=2,allow_nan=False)+'\n')
    report=dict(schema='scar.call-binding-run',schema_version=1,command=command,
        plan=str(path),join=str(out/'join.json'),seconds=time.monotonic()-started,
        child_returncode=code,stop_reason=reason,full_run_completed=code==0 and reason is None,
        planned_sites=joined['planned_sites'],observed_sites=joined['observed_sites'],
        attempts=joined['attempts'],raw_observations=[str(p) for p in observations],
        transformations_disabled=True,performance_measurement=False,first_stage_complete=False,
        global_profile_hook=False,tensor_contents_recorded=False)
    (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    origin=parser.add_mutually_exclusive_group(required=True)
    origin.add_argument('--plan');origin.add_argument('--source',action='append')
    origin.add_argument('--all-sources',action='store_true',help='select functions and modules from every admitted catalog unit in one bounded run')
    parser.add_argument('--include-modules',action='store_true',help='include module initialization CALLs in the selected original sources')
    parser.add_argument('--catalog');parser.add_argument('--entry');parser.add_argument('--out',required=True)
    parser.add_argument('--max-targets-per-site',type=int,choices=(1,2),default=1)
    parser.add_argument('--max-seconds',type=float,default=60)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();command=args.command[1:] if args.command[:1]==['--'] else args.command
    result=observe_call_bindings(command,args.out,plan_path=args.plan,source=args.source,catalog=args.catalog,
        entry=args.entry,max_targets_per_site=args.max_targets_per_site,max_seconds=args.max_seconds,
        all_sources=args.all_sources,include_modules=args.include_modules)
    print(json.dumps(result))
    if result['child_returncode'] and result['stop_reason'] is None: raise SystemExit(result['child_returncode'])
    if not result['attempts']: raise SystemExit(2)


if __name__=='__main__': main()

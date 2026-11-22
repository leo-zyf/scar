"""Observe one unchanged stateful method call per independent entry process.

Original F4/source/domain admission happens in the controller. Children receive
only frozen guard metadata and references, not a numeric DAG or array contents.
The original dispatch-to-return interval excludes checks, imports and binding;
it is not a clean whole-entry benchmark or a native variant measurement.
"""
import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import secrets
import statistics
import subprocess
import sys

from scar.ir.execution_recipe import artifact_reference,read_artifact

_BASIS='OBSERVED_ORIGINAL_CALL_WITH_WRAPPER_DISPATCH_COST_MODEL'
_DESIGN='scar.original-stateful-observation-design'
_SAMPLE='scar.original-stateful-call-sample'
_OBSERVATION='scar.original-stateful-call-observation'
_RESULT='scar.original-stateful-cost-result'
_DISABLE=('SCAR_CALL_BINDING_PLAN','SCAR_REGION_METADATA_PLAN','SCAR_TRACE_DIR','SCAR_OPTIMIZE')


def _json(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def _digest(value):return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def _receipt(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return dict(path=str(path),sha256=sha256(raw).hexdigest(),bytes=len(raw))
def _source(source):return {key:getattr(source,key)for key in ('path','sha256','code_key','qualname')}
def _save(path,document):
    with Path(path).open('xb')as stream:stream.write(_json(document))
    return artifact_reference(path,document['schema'])


def _tool_receipts():
    package=Path(__file__).resolve().parents[1]
    names=('compiler/observe_original_stateful_cost.py','compiler/original_stateful_worker.py',
        'runtime/entry_runner.py','runtime/stateful_guard.py','runtime/compiled_deployment.py',
        'trace/code_snapshot.py')
    return [_receipt(package/name)for name in names]


def _validate_sample(document,spec):
    """Accept a complete ORIGINAL return in precisely the frozen child scope."""
    if (type(document)is not dict or type(document.get('schema_version'))is not int
            or (document.get('schema'),document.get('schema_version'))!=(_SAMPLE,1)):
        raise ValueError('versioned original stateful worker sample required')
    for key in ('phase','index','worker_nonce','frozen_design','entry_context'):
        if document.get(key)!=spec[key]:raise ValueError('original worker sample differs: '+key)
    if (document.get('basis')!=_BASIS or document.get('status')!='OBSERVED_ORIGINAL_RETURN'
            or type(document.get('recorded_calls'))is not int or document['recorded_calls']!=1
            or type(document.get('process_id'))is not int or document['process_id']<=0
            or type(document.get('duration_ns'))is not int or document['duration_ns']<=0):
        raise ValueError('one positive-duration original return required; missing/failed timing is not zero')
    for key in ('function_internals_instrumented','global_trace_or_profile_installed','tensor_values_recorded',
            'target_variants_executed','target_variants_ranked'):
        if document.get(key)is not False:raise ValueError('original-only metadata observation required: '+key)
    if (type(document.get('native_calls'))is not int or document['native_calls']!=0
            or document.get('target_workload_executed')is not True):
        raise ValueError('sample must execute original dispatch and no native replacement')
    if (document.get('observed_caller_affinity_cpu_ids')!=spec['entry_context']['process_affinity_cpu_ids']
            or document.get('observed_native_thread_environment')!=spec['entry_context']['native_thread_environment']
            or document.get('actual_full_thread_environment')!=spec['entry_context']['native_thread_environment']):
        raise ValueError('observed original affinity/full thread environment differs')
    return document['duration_ns']


def _admit(domain_path,program,entry,argv):
    from scar.analysis.build import BuildSession
    from scar.compiler.frontend.stateful import discover_stateful_regions
    from scar.compiler.stateful_model import stateful_from_dict,stateful_digest
    from scar.optimization.stateful_cpu_model import _domain, stateful_cpu_cost_context
    from scar.runtime.compiled_deployment import entry_context,thread_environment
    domain_path=Path(domain_path).resolve()
    if domain_path.is_dir():domain_path/='domain.json'
    domain_ref=artifact_reference(domain_path,'scar.stateful-cpu-domain');domain=_domain(read_artifact(domain_ref))
    region_ref=domain['region'];region=stateful_from_dict(read_artifact(region_ref))
    program=Path(program).resolve();program=program/'manifest.json'if program.is_dir()else program
    parent=artifact_reference(program,'scar.program-build');manifest=read_artifact(parent)
    hardware=manifest.get('target');snapshot=domain['target_snapshot']
    if (manifest.get('backend')!='F4' or hardware is None
            or (snapshot['uri'],snapshot['sha256'],snapshot['size_bytes'])!=
                (hardware['path'],hardware['sha256'],hardware['bytes'])):
        raise ValueError('stateful observation program/domain original hardware differs')
    with BuildSession(program.parent)as session:
        found,_=discover_stateful_regions(session.view,source=region.source.path)
        matches=[value for value in found if value.id==region.id and stateful_digest(value)==stateful_digest(region)]
        if len(matches)!=1:raise ValueError('stateful region differs from original F4/source rediscovery')
    if (Path(region.source.path)==entry or any(not part.isidentifier()for part in region.source.qualname.split('.'))):
        raise ValueError('original observation requires an imported ordinary module/class boundary')
    current=entry_context(entry,argv)
    if (current['process_affinity_cpu_ids']!=domain['live_constraints']['process_affinity_cpu_ids']
            or thread_environment()!=domain['live_constraints']['native_thread_environment']):
        raise ValueError('original observation affinity/full thread environment differs from domain')
    context=stateful_cpu_cost_context(domain,'NO_OP')
    sources={}
    for value in (region.source,*(row.source for row in region.callees)):
        if _receipt(value.path)['sha256']!=value.sha256:raise ValueError('original source version differs')
        previous=sources.setdefault(value.path,_source(value))
        if previous['sha256']!=value.sha256:raise ValueError('original source has conflicting versions')
    arrays=domain['request']['input_shapes'];written={row.target for row in region.actions if row.kind=='write'}
    inputs=[dict(id=row.id,path=list(row.path),kind=row.kind,
        shape=arrays[row.id]if row.kind=='array'else None,
        nbytes=math.prod(arrays[row.id])*8 if row.kind=='array'else None,written=row.id in written)
        for row in region.inputs]
    return domain,dict(domain=domain_ref,program=parent,proof=region_ref,context=context,
        region_id=region.id,region_sha256=stateful_digest(region),source=_source(region.source),
        sources=list(sources.values()),parameters=list(region.parameter_names),inputs=inputs,
        input_shapes=arrays,callee_guards=[dict(path=list(row.path),kind=row.kind,source=_source(row.source))for row in region.callees],
        providers=[dict(id=row.id,module=row.module,symbol=row.symbol,lookup_path=list(row.lookup_path),semantics=row.semantics)
            for row in region.providers],provider_runtime=domain['backend_capabilities']['serial']['provider_runtime'],
        entry=str(entry),argv=argv,entry_context=current,source_proof_basis='EXACT_ORIGINAL_F4_STATEFUL_REDISCOVERY',
        full_f4_replays=1,worker_replays_f4=False)


def observe_original_stateful_cost(domain_path,program,entry,argv,out,*,warmup=1,repetitions=3,max_seconds=600):
    """Freeze a guard-only design, observe originals, then publish a NO_OP profile."""
    from scar.optimization.stateful_cpu_model import wrap_original_call_cost
    from scar.runtime.compiled_deployment import entry_context
    if (type(warmup)is not int or warmup<0 or type(repetitions)is not int or repetitions<3
            or type(max_seconds)not in (int,float)or not math.isfinite(max_seconds)or max_seconds<=0):
        raise ValueError('nonnegative excluded warmup, >=3 independent samples and positive finite timeout required')
    entry=Path(entry).resolve();argv=list(argv);out=Path(out).resolve()
    if out.exists():raise ValueError('fresh original stateful cost output required')
    if not entry.is_file()or entry.suffix!='.py'or any(type(arg)is not str for arg in argv):
        raise ValueError('original Python entry and exact string argv required')
    domain,admitted=_admit(domain_path,program,entry,argv)
    environment=dict(os.environ);removed=[name for name in _DISABLE if name in environment]
    for name in _DISABLE:environment.pop(name,None)
    tools=_tool_receipts()
    design=dict(schema=_DESIGN,schema_version=1,**admitted,warmup=warmup,repetitions=repetitions,
        max_seconds=max_seconds,environment_sha256=_digest(environment),
        environment_variable_names=sorted(environment),removed_instrumentation_variables=removed,
        tool_receipts=tools,timing_basis=_BASIS,frozen_before_measurement=True,
        measurement_scope='ONE_ORIGINAL_METHOD_CALL_SITE',
        independent_process_per_sample=True,first_call_only=True,
        warmup_scope='EXCLUDED_FRESH_PROCESSES_NOT_CACHE_WARMING_OF_MEASUREMENT_PROCESSES',
        clock_includes=['ORIGINAL_ARGUMENT_DISPATCH_BODY_AND_RETURN'],
        clock_excludes=['IMPORT_AND_LOAD','METADATA_SOURCE_PROVIDER_INPUT_GUARDS','SIGNATURE_BINDING','REPORT_IO'],
        function_internals_instrumented=False,tensor_values_recorded=False,
        target_variants_executed=False,target_variants_ranked=False,clean_whole_entry_benchmark=False)
    out.mkdir(parents=True);design_ref=_save(out/'design.json',design);samples=[];failure=None
    def recheck():
        if entry_context(entry,argv)!=design['entry_context']or _digest(environment)!=design['environment_sha256']:
            raise ValueError('frozen original execution context changed')
        for receipt in tools:
            if _receipt(receipt['path'])!=receipt:raise ValueError('frozen observer/runtime implementation changed')
        for ref in (design_ref,admitted['domain'],admitted['program'],admitted['proof']):read_artifact(ref)
        for source in design['sources']:
            if _receipt(source['path'])['sha256']!=source['sha256']:raise ValueError('frozen original source changed')
    for phase,count in (('WARMUP',warmup),('MEASUREMENT',repetitions)):
        for index in range(count):
            recheck();prefix=phase.lower()+'-'+str(index)
            spec={**design,**dict(phase=phase,index=index,worker_nonce=secrets.token_hex(16),frozen_design=design_ref)}
            spec_path=out/(prefix+'-design.json');spec_ref=_save(spec_path,spec)
            sample_path=out/(prefix+'.json');stdout_path=out/(prefix+'.stdout');stderr_path=out/(prefix+'.stderr')
            returncode=None;errors=[]
            with stdout_path.open('xb')as stdout,stderr_path.open('xb')as stderr:
                try:
                    process=subprocess.run([sys.executable,'-m','scar.compiler.original_stateful_worker',
                        '--spec',str(spec_path),'--sample',str(sample_path)],env=environment,
                        stdout=stdout,stderr=stderr,timeout=max_seconds)
                    returncode=process.returncode
                except subprocess.TimeoutExpired:errors.append('ORIGINAL_WORKER_TIMEOUT')
            if returncode!=0:errors.append('ORIGINAL_WORKER_DID_NOT_EXIT_SUCCESSFULLY')
            sample_ref=None;duration=None;pid=None
            try:
                if not sample_path.is_file():raise ValueError('original worker sample missing')
                sample_ref=artifact_reference(sample_path,_SAMPLE);sample=read_artifact(sample_ref)
                duration=_validate_sample(sample,spec);pid=sample['process_id']
            except (OSError,ValueError,TypeError,KeyError)as error:errors.append(str(error))
            samples.append(dict(phase=phase,index=index,spec=spec_ref,sample=sample_ref,
                process_id=pid,duration_ns=duration if not errors else None,returncode=returncode,
                stdout=_receipt(stdout_path),stderr=_receipt(stderr_path),validation_errors=errors))
            recheck()
            if errors:failure=dict(phase=phase,index=index,reasons=errors);break
        if failure is not None:break
    if failure is not None:
        result=dict(schema=_RESULT,schema_version=1,status='REJECTED_OBSERVATION',design=design_ref,
            samples=samples,failure=failure,profile=None,observations=None,duration_ns=None,
            target_workload_executed=True,target_variants_ranked=False)
        _save(out/'result.json',result);return result
    measured=[row['duration_ns']for row in samples if row['phase']=='MEASUREMENT']
    observation=dict(schema=_OBSERVATION,schema_version=1,context=admitted['context'],design=design_ref,
        measurement_scope='ONE_ORIGINAL_METHOD_CALL_SITE',basis=_BASIS,samples_ns=measured,
        median_ns=statistics.median(measured),min_ns=min(measured),max_ns=max(measured),stdev_ns=statistics.stdev(measured),
        warmup=warmup,repetitions=repetitions,samples=samples,
        observed_caller_affinity_cpu_ids=design['entry_context']['process_affinity_cpu_ids'],
        observed_native_thread_environment=design['entry_context']['native_thread_environment'],
        independent_process_per_sample=True,first_call_only=True,whole_episode_averaged=False,
        clean_whole_entry_benchmark=False,function_internals_instrumented=False,tensor_values_recorded=False,
        target_workload_executed=True,target_variants_executed=False,target_variants_ranked=False,native_calls=0)
    observed_ref=_save(out/'observations.json',observation)
    # The existing public profile producer intentionally performs its final
    # domain replay. Workers never load an analysis graph or a cost model.
    profile=wrap_original_call_cost(domain,out/'observations.json',out=out/'forward-cost')
    recheck()
    result=dict(schema=_RESULT,schema_version=1,status='OBSERVED_ORIGINAL_ONLY',design=design_ref,
        observations=observed_ref,profile=profile,duration_ns=observation['median_ns'],
        warmup=warmup,repetitions=repetitions,samples=samples,basis=_BASIS,
        target_workload_executed=True,target_variants_executed=False,target_variants_ranked=False,
        clean_whole_entry_benchmark=False,native_calls=0)
    _save(out/'result.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain',required=True);parser.add_argument('--build',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--warmup',type=int,default=1);parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--max-seconds',type=float,default=600);parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv);command=args.command[1:]if args.command[:1]==['--']else args.command
    if not command:parser.error('original Python entry required after --')
    result=observe_original_stateful_cost(args.domain,args.build,command[0],command[1:],args.out,
        warmup=args.warmup,repetitions=args.repetitions,max_seconds=args.max_seconds)
    print(json.dumps(result,sort_keys=True,indent=2));return 0 if result['profile']is not None else 2


if __name__=='__main__':raise SystemExit(main())

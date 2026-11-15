"""Bounded original-call timing for a retained F4 loop and CPU cost domain.

Only the first original boundary call in each fresh process is observed.
Metadata/source/applicability checks stay outside the clock. This is a
wrapper-dispatch cost observation, never a clean whole-entry benchmark.
"""
import argparse
from functools import wraps
from hashlib import sha256
import importlib.abc
from importlib.machinery import PathFinder,SourceFileLoader
import inspect
import json
import os
from pathlib import Path
import runpy
import statistics
import subprocess
import sys
import time
from types import FunctionType

_RANGE=range
_BASIS='OBSERVED_ORIGINAL_CALL_WITH_WRAPPER_DISPATCH_COST_MODEL'
_DISABLE=('SCAR_CALL_BINDING_PLAN','SCAR_REGION_METADATA_PLAN','SCAR_TRACE_DIR','SCAR_OPTIMIZE')


def _json(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def _digest(value):return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def _receipt(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return dict(path=str(path),sha256=sha256(raw).hexdigest(),bytes=len(raw))
def _source(source):return {key:getattr(source,key) for key in ('path','sha256','code_key','qualname')}


def _worker(spec_path,sample_path):
    """Run one untouched entry with one timed ORIGINAL dispatch, no F4 load."""
    from scar.runtime.entry_runner import _Loader,_file
    from scar.runtime.loop_guard import _state,_same_state,_range
    from scar.runtime.compiled_deployment import entry_context
    from scar.trace.code_snapshot import code_key
    spec=json.loads(Path(spec_path).read_bytes());sample_path=Path(sample_path)
    observed=dict(schema='scar.original-loop-call-sample',schema_version=1,
        phase=spec['phase'],index=spec['index'],basis=_BASIS,status='NOT_OBSERVED',
        recorded_calls=0,total_calls_collected=False,duration_ns=None,tensor_values_recorded=False,
        target_workload_executed=True,target_variants_executed=False,target_variants_ranked=False)
    if _digest(dict(os.environ))!=spec['environment_sha256']:
        raise ValueError('original observation child environment differs from frozen effective environment')
    if entry_context(spec['entry'],spec['argv'])!=spec['entry_context']:
        raise ValueError('original observation child execution context differs')
    for source in spec['sources']:
        if _receipt(source['path'])['sha256']!=source['sha256']:raise ValueError('original source changed before observation')
    sources={s['path']:s['sha256'] for s in spec['sources']}
    bindings=[];seen=set();captured=[False];observer_pid=os.getpid()
    def ordinary(value,source):
        return (type(value) is FunctionType and not value.__closure__
            and type(value.__globals__) is dict and type(value.__builtins__) is dict
            and (value.__defaults__ is None or type(value.__defaults__) is tuple)
            and (value.__kwdefaults__ is None or type(value.__kwdefaults__) is dict)
            and code_key(value.__code__)==source['code_key'])
    class Scope:
        frontier=[]
        def attach(self,module,name,*,natural):
            if id(module) in seen:return
            seen.add(id(module))
            if _file(module)!=Path(spec['source']['path']):return
            original=vars(module).get(spec['source']['qualname'])
            if not ordinary(original,spec['source']):raise ValueError('original loop does not match admitted ordinary CodeID')
            leaf=original.__globals__.get(spec['leaf']['global_name'])
            if not ordinary(leaf,spec['leaf']['source']):raise ValueError('original leaf does not match admitted CodeID')
            entry_state=_state(original);leaf_state=_state(leaf)
            shadow=FunctionType(original.__code__,original.__globals__,argdefs=original.__defaults__)
            shadow.__kwdefaults__=original.__kwdefaults__;signature=inspect.signature(shadow,follow_wrapped=False)
            @wraps(original)
            def wrapped(*args,**kwargs):
                if os.getpid()!=observer_pid:return original(*args,**kwargs)
                if captured[0]:return original(*args,**kwargs)
                captured[0]=True;observed['recorded_calls']=1
                metadata={};reason=None
                if (not ordinary(original,spec['source']) or not ordinary(leaf,spec['leaf']['source'])
                        or not _same_state(original,entry_state) or not _same_state(leaf,leaf_state)
                        or original.__globals__.get(spec['leaf']['global_name']) is not leaf
                        or _range(original) is not _RANGE):
                    reason='ORIGINAL_CALLABLE_DEFAULTS_RANGE_OR_GLOBAL_BINDING_CHANGED'
                if any(_receipt(path)['sha256']!=digest for path,digest in sources.items()):
                    reason='ORIGINAL_SOURCE_VERSION_CHANGED'
                try:bound=signature.bind(*args,**kwargs);bound.apply_defaults()
                except TypeError:bound=None;reason='ORIGINAL_ARGUMENT_BINDING_FAILED'
                if bound is not None:
                    count=bound.arguments[spec['trip_count_parameter']]
                    metadata['trip_count']=count if type(count) is int else None
                    metadata['trip_count_exact_int']=type(count) is int
                    if type(count) is not int or count!=spec['trip_count']:reason='COUNT_OUTSIDE_COST_CONTEXT'
                    import numpy as np
                    metadata['numpy_version']=np.__version__;metadata['arrays']={}
                    for leaf_name,outer in spec['parameter_map'].items():
                        value=bound.arguments[outer]
                        if type(value) is not np.ndarray:
                            metadata['arrays'][leaf_name]=dict(exact_ndarray=False)
                            reason='INPUT_NOT_EXACT_NDARRAY';continue
                        row=dict(exact_ndarray=True,shape=list(value.shape),dtype=value.dtype.str,
                            strides=list(value.strides),nbytes=value.nbytes,c_contiguous=bool(value.flags.c_contiguous),
                            aligned=bool(value.flags.aligned))
                        metadata['arrays'][leaf_name]=row
                        if (value.dtype!=np.dtype('float64') or not value.dtype.isnative or value.ndim!=2
                                or row['shape']!=spec['input_shapes'][leaf_name] or not value.flags.c_contiguous
                                or not value.flags.aligned):reason='ARRAY_METADATA_OUTSIDE_COST_CONTEXT'
                if entry_context(spec['entry'],spec['argv'])!=spec['entry_context']:
                    reason='EXECUTION_CONTEXT_CHANGED_BEFORE_ORIGINAL_CALL'
                if sys.gettrace() is not None or sys.getprofile() is not None:
                    reason='EXISTING_GLOBAL_OBSERVER_OUTSIDE_COST_CONTEXT'
                observed['metadata']=metadata
                if reason is not None:
                    observed.update(status='OUTSIDE_FROZEN_CONTEXT',reason=reason)
                    return original(*args,**kwargs)
                # All metadata checks above are excluded. Dispatch, original
                # execution and return/exception dispatch are inside the clock.
                start=time.perf_counter_ns()
                try:result=original(*args,**kwargs)
                except BaseException as error:
                    end=time.perf_counter_ns()
                    observed.update(status='ORIGINAL_RAISED',duration_ns=end-start,exception_type=type(error).__name__)
                    raise
                end=time.perf_counter_ns()
                observed.update(status='OBSERVED_ORIGINAL_RETURN',duration_ns=end-start)
                return result
            setattr(module,spec['source']['qualname'],wrapped)
            bindings.append((module,spec['source']['qualname'],original,wrapped))
    scope=Scope()
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            candidate=PathFinder.find_spec(fullname,path,target)
            if (candidate is not None and type(candidate.origin) is str
                    and Path(candidate.origin).resolve()==Path(spec['source']['path'])
                    and type(candidate.loader) is SourceFileLoader):
                candidate.loader=_Loader(candidate.loader,scope,fullname)
            return candidate
    entry=Path(spec['entry']);old_argv,old_path,old_meta=sys.argv,sys.path,sys.meta_path
    sys.argv=[str(entry),*spec['argv']];sys.path=[str(entry.parent),*old_path[1:]]
    finder=Finder();sys.meta_path=[finder if item is PathFinder else item for item in old_meta]
    try:
        try:runpy.run_path(str(entry),run_name='__main__')
        except SystemExit as error:
            if error.code is not None and error.code!=0:raise
    finally:
        for module,name,original,wrapped in reversed(bindings):
            if vars(module).get(name) is wrapped:setattr(module,name,original)
        sys.argv,sys.path,sys.meta_path=old_argv,old_path,old_meta
        observed.update(scope='ONE_ORIGINAL_CALL_DISPATCH_THROUGH_RETURN',
            function_internals_instrumented=False,global_trace_or_profile_installed=False,
            input_content_identity_proven=False,clock_includes_wrapper_dispatch=True,
            clean_whole_entry_benchmark=False,observer=_receipt(__file__))
        sample_path.write_bytes(_json(observed))
    return observed


def observe_original_loop_cost(domain_path,program,proof_reference,entry,argv,out,*,warmup=1,repetitions=3,max_seconds=600):
    """Observe original calls only, after original F4 and domain admission."""
    from dataclasses import asdict
    from scar.analysis.build import BuildSession
    from scar.compiler.invariant_loop_frontend import read_invariant_loop_proof
    from scar.ir.execution_recipe import artifact_reference,read_artifact
    from scar.optimization.temporal_cpu_choices import verify_temporal_cpu_domain,temporal_cpu_cost_context
    from scar.runtime.compiled_deployment import entry_context
    if type(warmup) is not int or warmup<0 or type(repetitions) is not int or repetitions<3 or max_seconds<=0:
        raise ValueError('nonnegative warmup, at least three original repetitions and positive time budget required')
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh original-loop observation output required')
    entry=Path(entry).resolve();argv=list(argv)
    if not entry.is_file() or any(type(a) is not str for a in argv):raise ValueError('original script and exact argv strings required')
    domain_ref=artifact_reference(domain_path,'scar.temporal-cpu-domain');domain=read_artifact(domain_ref)
    if isinstance(proof_reference,(str,Path)):
        proof_reference=artifact_reference(proof_reference,'scar.invariant-loop-proof')
    read_artifact(proof_reference)
    with BuildSession(program) as session:
        proof=read_invariant_loop_proof(proof_reference['uri'],catalog=session.view)
        verify_temporal_cpu_domain(proof,domain['target_path'],domain)
        if _digest(asdict(proof))!=domain['proof_sha256']:raise ValueError('original cost proof differs from domain')
    if not proof.source.qualname.isidentifier() or Path(proof.source.path)==entry:
        raise ValueError('original cost observation requires an ordinary imported module-level function boundary')
    context=temporal_cpu_cost_context(domain,'NO_OP')
    current=entry_context(entry,argv)
    if (current['process_affinity_cpu_ids']!=domain['live_constraints']['process_affinity_cpu_ids']
            or any(os.environ.get(key)!=value for key,value in domain['live_constraints']['native_thread_environment'].items())):
        raise ValueError('original observation affinity/thread environment differs from domain')
    env=dict(os.environ);removed=[name for name in _DISABLE if name in env]
    for name in _DISABLE:env.pop(name,None)
    specification=dict(schema='scar.original-loop-observation-design',schema_version=1,
        domain=domain_ref,program=_receipt(Path(program)/'manifest.json'),proof=proof_reference,
        context=context,entry=str(entry),argv=argv,entry_context=current,
        source=_source(proof.source),leaf=dict(global_name=proof.leaf_global_name,source=_source(proof.body.source)),
        sources=[_source(proof.source),_source(proof.body.source)],parameter_map=proof.parameter_map,
        trip_count_parameter=proof.trip_count_parameter,input_shapes=domain['input_shapes'],trip_count=domain['trip_count'],
        warmup=warmup,repetitions=repetitions,max_seconds=max_seconds,environment_sha256=_digest(env),
        environment_variable_names=sorted(env),removed_instrumentation_variables=removed,
        observer=_receipt(__file__),tensor_values_recorded=False,target_variants_ranked=False,
        timing_basis=_BASIS,clean_whole_entry_benchmark=False,frozen_before_measurement=True)
    out.mkdir(parents=True);design=out/'design.json';design.write_bytes(_json(specification))
    design_ref=artifact_reference(design,specification['schema']);samples=[]
    for phase,total in (('WARMUP',warmup),('MEASUREMENT',repetitions)):
        for index in range(total):
            if (entry_context(entry,argv)!=current or _receipt(__file__)!=specification['observer']
                    or _receipt(Path(program)/'manifest.json')!=specification['program']):
                raise ValueError('frozen original observation inputs or observer changed')
            read_artifact(domain_ref);read_artifact(proof_reference)
            row={**specification,'phase':phase,'index':index}
            prefix=phase.lower()+'-'+str(index);worker_spec=out/(prefix+'-design.json')
            worker_spec.write_bytes(_json(row));sample=out/(prefix+'.json')
            with (out/(prefix+'.stdout')).open('xb') as stdout,(out/(prefix+'.stderr')).open('xb') as stderr:
                process=subprocess.run([sys.executable,'-m','scar.compiler.observe_original_loop_cost',
                    '--worker',str(worker_spec),'--sample',str(sample)],env=env,stdout=stdout,stderr=stderr,timeout=max_seconds)
            if process.returncode or not sample.is_file():raise ValueError('original entry observation process failed: '+prefix)
            document=json.loads(sample.read_bytes())
            if document['status']!='OBSERVED_ORIGINAL_RETURN' or document['recorded_calls']!=1:
                raise ValueError('one complete original call in the exact cost context was not observed: '+prefix)
            samples.append(dict(phase=phase,index=index,sample=artifact_reference(sample,document['schema']),
                duration_ns=document['duration_ns'],stdout=_receipt(out/(prefix+'.stdout')),stderr=_receipt(out/(prefix+'.stderr'))))
    measured=[r['duration_ns'] for r in samples if r['phase']=='MEASUREMENT'];duration=statistics.median(measured)
    facts=dict(schema='scar.original-loop-cost-observations',schema_version=1,context=context,
        duration_ns=duration,samples=samples,design=design_ref,basis=_BASIS,
        warmup=warmup,repetitions=repetitions,min_ns=min(measured),max_ns=max(measured),
        stdev_ns=statistics.stdev(measured),target_workload_executed=True,target_variants_executed=False,
        target_variants_ranked=False,tensor_values_recorded=False,clean_whole_entry_benchmark=False)
    path=out/'observations.json';path.write_bytes(_json(facts));reference=artifact_reference(path,facts['schema'])
    profile=dict(schema='scar.temporal-cpu-cost-profile',schema_version=1,context=context,duration_ns=duration,
        context_evidence=[dict(artifact=reference,locator='#/context')],cost_evidence=[dict(artifact=reference,locator='#/duration_ns')],
        basis=_BASIS,observation_design=design_ref,target_workload_executed=True,target_variants_executed=False,
        target_variants_ranked=False,clean_whole_entry_benchmark=False)
    path=out/'profile.json';path.write_bytes(_json(profile))
    result=dict(schema='scar.original-loop-cost-result',schema_version=1,profile=str(path),observations=reference,
        duration_ns=duration,warmup=warmup,repetitions=repetitions,basis=_BASIS,
        target_workload_executed=True,target_variants_ranked=False,clean_whole_entry_benchmark=False)
    (out/'result.json').write_bytes(_json(result));return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain');parser.add_argument('--build');parser.add_argument('--proof')
    parser.add_argument('--out');parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--repetitions',type=int,default=3);parser.add_argument('--max-seconds',type=float,default=600)
    parser.add_argument('--worker');parser.add_argument('--sample');parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv)
    if args.worker:
        if not args.sample:parser.error('--sample required for original worker')
        _worker(args.worker,args.sample);return 0
    if not all((args.domain,args.build,args.proof,args.out)):parser.error('--domain --build --proof --out required')
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:parser.error('original Python script required after --')
    print(json.dumps(observe_original_loop_cost(args.domain,args.build,args.proof,command[0],command[1:],args.out,
        warmup=args.warmup,repetitions=args.repetitions,max_seconds=args.max_seconds),indent=2));return 0


if __name__=='__main__':raise SystemExit(main())

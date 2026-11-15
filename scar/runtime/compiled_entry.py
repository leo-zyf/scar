"""Execute an unchanged entry with model-selected loop implementations.

Original F4 proofs are replayed before execution. Python keeps compatibility
work and external effects. Candidate validation is distinct from deployment
profitability; this entry does not authorize an unmeasured deployment.
"""
import argparse
from hashlib import sha256
from functools import wraps
import importlib.abc
from importlib.machinery import PathFinder,SourceFileLoader
import json
from pathlib import Path
import runpy
import sys

from scar.runtime.entry_runner import _Scope,_Loader,_file,run_original_script


class _LoopScope(_Scope):
    def __init__(self,artifact,proof,spec):
        self.artifact,self.proof,self.spec=artifact,proof,spec
        self.source=Path(proof.source.path).resolve();self.name=proof.source.qualname
        self.active=True;self.bindings=[];self.seen=set();self.frontier=[]

    def attach(self,module,name,*,natural):
        if not self.active or id(module) in self.seen:return
        self.seen.add(id(module))
        if _file(module)!=self.source or not self.name.isidentifier():
            self.frontier.append('unsupported_original_function_boundary');return
        original=vars(module).get(self.name)
        try:
            if self.proof is None:
                from scar.runtime.native_temporal_invariant import guarded_temporal_invariant,SourceMismatch
                guarded=guarded_temporal_invariant(original,self.artifact,exclusive_inputs=True,
                    trial=self.trial,selection=self.selection)
            else:
                from scar.compiler.invariant_loop_runtime import guarded_invariant_loop
                from scar.compiler.runtime import SourceMismatch
                guarded=guarded_invariant_loop(original,self.artifact,self.proof,exclusive_inputs=True,trial=True)
        except (SourceMismatch,ValueError) as error:
            self.frontier.append('original_boundary_admission_rejected:'+str(error));return
        scope=self
        @wraps(original)
        def binding(*args,**kwargs):
            return guarded(*args,**kwargs) if scope.active else original(*args,**kwargs)
        binding.scar_stats=guarded.scar_stats;binding.scar_artifact=self.artifact
        setattr(module,self.name,binding)
        self.bindings.append(dict(module=module,module_name=name,original=original,owner=module,
            attribute=self.name,descriptor=original,installed=binding,binding=binding,
            stats=guarded.scar_stats,natural_import=natural))


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self,scopes):self.scopes=scopes
    def find_spec(self,fullname,path=None,target=None):
        spec=PathFinder.find_spec(fullname,path,target)
        if spec is not None and type(spec.origin) is str:
            selected=[scope for scope in self.scopes if Path(spec.origin).resolve()==scope.source]
            if selected and type(spec.loader) is SourceFileLoader:
                # Each loader completes once, then attaches all selected
                # boundaries in this original module before from-import.
                class Group:
                    frontier=[]
                    def attach(self,module,name,*,natural):
                        for scope in selected:scope.attach(module,name,natural=natural)
                spec.loader=_Loader(spec.loader,Group(),fullname)
        return spec


def artifact_reference(path,schema):
    path=Path(path).resolve();raw=path.read_bytes()
    return dict(uri=str(path),sha256=sha256(raw).hexdigest(),size_bytes=len(raw),schema=schema)


def read_artifact(reference):
    if artifact_reference(reference['uri'],reference['schema'])!=reference:
        raise ValueError('compiled entry receipt differs')
    value=json.loads(Path(reference['uri']).read_bytes())
    if value.get('schema')!=reference['schema']:raise ValueError('compiled entry schema differs')
    return value


def run_build(pipeline,entry,argv=(),*,exclusive_inputs=False,measure_candidate=False,deployment=None):
    pipeline=Path(pipeline).resolve()
    if pipeline.is_dir():pipeline=pipeline/'result.json'
    reference=artifact_reference(pipeline,'scar.program-compiler-pipeline');report=read_artifact(reference)
    from scar.runtime.compiled_deployment import deployment_gate
    gate=(dict(decision='MEASURE_SELECTED_MODEL_IMPLEMENTATION',deployment_authorized=False)
        if measure_candidate else deployment_gate(deployment,reference,entry,argv))
    if not exclusive_inputs or gate['decision'] not in ('MEASURE_SELECTED_MODEL_IMPLEMENTATION','LOAD_MODEL_SELECTED_NATIVE'):
        return dict(mode='ORIGINAL_EXECUTION',reason='candidate_measurement_and_exclusive_inputs_required',
            execution=run_original_script(entry,argv),pipeline=reference,deployment_gate=gate)
    rows=[r for r in report.get('invariant_loops',()) if r['status']=='MODEL_SELECTED_LOOP_COMPILED']
    if not rows:return dict(mode='ORIGINAL_EXECUTION',execution=run_original_script(entry,argv),pipeline=reference)
    scopes=[]
    if all(row.get('bundle_export') for row in rows):
        from scar.runtime.native_temporal_invariant import load
        for row in rows:
            exported=read_artifact(row['bundle_export'])
            if exported['bundle']!=row['bundle']:raise ValueError('bundle directory differs from retained export')
            artifact=load(row['bundle'],expected_admission_sha256=exported['expected_admission_sha256'])
            contract=artifact.contract;schedule=read_artifact(row['schedule'])
            receipt=contract['resource_schedule']['receipt']
            if (contract['source_proof']!=row['proof'] or receipt['path']!=row['schedule']['uri']
                    or receipt['sha256']!=row['schedule']['sha256'] or receipt['bytes']!=row['schedule']['size_bytes']
                    or contract['resource_schedule']['id']!=schedule['id']):
                raise ValueError('bundle proof/schedule differs from compiled pipeline')
            scope=_LoopScope.__new__(_LoopScope)
            scope.artifact,scope.proof,scope.spec=artifact,None,row
            scope.source=Path(contract['source']['path']).resolve();scope.name=contract['source']['qualname']
            scope.active=True;scope.bindings=[];scope.seen=set();scope.frontier=[]
            scope.trial=measure_candidate
            scope.selection={**gate,'native_admission_sha256':artifact.report['admission_sha256']}
            scopes.append(scope)
    else:
        from scar.analysis.build import BuildSession
        with BuildSession(Path(report['program']['uri']).parent) as session:
            scopes.extend(_legacy_scopes(rows,session,report))
    return _execute_scopes(scopes,entry,argv,reference,gate)


def _legacy_scopes(rows,session,report):
    from scar.compiler.invariant_loop_frontend import read_invariant_loop_proof
    from scar.compiler.temporal_invariant_loop import load_temporal_invariant_loop
    from scar.optimization.invariant_loop_schedule import verify_invariant_loop_schedule
    scopes=[]
    for row in rows:
        read_artifact(row['proof']);read_artifact(row['native_artifact'])
        proof=read_invariant_loop_proof(row['proof']['uri'],catalog=session.view)
        if artifact_reference(Path(row['native_directory'])/'artifact.json',row['native_artifact']['schema'])!=row['native_artifact']:
            raise ValueError('native directory differs from its retained artifact')
        artifact=load_temporal_invariant_loop(row['native_directory'])
        schedule=read_artifact(row['schedule'])
        verify_invariant_loop_schedule(proof,Path(report['hardware']['path']).parent,schedule)
        specialization=artifact.plan['loop_interface']['execution_specialization']
        if (specialization['input_shapes']!=schedule['input_shapes'] or specialization['trip_count']!=schedule['trip_count']
                or specialization.get('input_layout')!='C_CONTIGUOUS'
                or artifact.plan['memory_budget_bytes']!=schedule['effective_incremental_budget_bytes']
                or artifact.plan['execution_plan']['resource_binding']!=schedule['resource_binding']):
            raise ValueError('native specialization/placement differs from selected resource schedule')
        scopes.append(_LoopScope(artifact,proof,row))
    return scopes


def _execute_scopes(scopes,entry,argv,reference,gate):
    entry=Path(entry).resolve();old_argv,old_path,old_meta=sys.argv,sys.path,sys.meta_path
    argv_values,path_values,meta_values=list(old_argv),list(old_path),list(old_meta)
    sys.argv=[str(entry),*argv];sys.path=[str(entry.parent),*old_path[1:]]
    finder=_Finder(scopes)
    sys.meta_path=[finder if item is PathFinder else item for item in old_meta]
    try:
        for name,module in tuple(sys.modules.items()):
            for scope in scopes:
                if _file(module)==scope.source:scope.attach(module,name,natural=False)
        try:runpy.run_path(str(entry),run_name='__main__')
        except SystemExit as error:
            if error.code is not None and error.code!=0:raise
    finally:
        try:
            for scope in scopes:scope.finish()
        finally:
            old_argv[:]=argv_values;old_path[:]=path_values;old_meta[:]=meta_values
            sys.argv,sys.path,sys.meta_path=old_argv,old_path,old_meta
    return dict(schema='scar.compiled-program-execution',schema_version=1,pipeline=reference,
        mode=gate['decision'],deployment_gate=gate,entry=str(entry),argv=list(argv),
        attachments=[dict(source=str(s.source),qualname=s.name,bindings=len(s.bindings),
            statistics=[r['stats'] for r in s.bindings],native_execution=s.artifact.report.get('last_execution'),
            frontier=s.frontier) for s in scopes],full_program_cpp=False,
        deployment_authorized=gate['deployment_authorized'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline',required=True);parser.add_argument('--exclusive-inputs',action='store_true')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--measure-candidate',action='store_true');modes.add_argument('--deployment')
    parser.add_argument('--report')
    parser.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args(argv)
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:parser.error('original Python script required after --')
    path=None if args.report is None else Path(args.report)
    if path is not None and path.exists():parser.error('fresh report path required')
    result=run_build(args.pipeline,command[0],command[1:],exclusive_inputs=args.exclusive_inputs,
        measure_candidate=args.measure_candidate,deployment=args.deployment)
    if path is not None:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    return 0


if __name__=='__main__':raise SystemExit(main())

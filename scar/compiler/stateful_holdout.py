"""Precommitted whole-method holdouts over existing stateful implementations.

Preparation and cold verification do not import the target or open its ELF.
Execution preserves the original driver and only temporarily selects the
original method or the existing guarded explicit-trial callable.
"""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from copy import deepcopy
from functools import wraps
from hashlib import sha256
import importlib
import inspect
import io
import json
import os
from pathlib import Path
import statistics
import struct
import sys
import time
import uuid
from types import FunctionType, ModuleType

DESIGN = 'scar.stateful-method-holdout-design'
POLICY = 'scar.stateful-method-holdout-policy'
REPORT = 'scar.stateful-method-holdout'
TIMER = dict(clock='perf_counter_ns', span='SELECTED_CALLABLE_DISPATCH_THROUGH_RETURN',
    includes=['PYTHON_CALL_DISPATCH','IMPLEMENTATION_GUARDS','ABI_PACKING',
        'IMPLEMENTATION_ALLOCATIONS','NATIVE_WORK','RETURN'],
    excludes=['CALLER_ATTRIBUTE_LOOKUP','ALREADY_EVALUATED_ARGUMENT_EXPRESSIONS',
        'INPUT_METADATA_OBSERVATION','POST_ENTRY_STATE_AND_FILE_COMPARISONS',
        'ORIGINAL_CALLER_CONTROL_AND_IO','IMPORT_LOAD_INSTALL_ONE_TIME_ADMISSION'])


def _need(value, message):
    if not value:raise ValueError(message)


def _ref(path, schema=None):
    path=Path(path).resolve();raw=path.read_bytes()
    if schema is None:schema=json.loads(raw)['schema']
    return dict(uri=str(path),schema=schema,sha256=sha256(raw).hexdigest(),size_bytes=len(raw))


def _read(reference):
    from scar.ir.execution_recipe import read_artifact
    _need(type(reference)is dict and set(reference)=={'uri','schema','sha256','size_bytes'},'complete ArtifactRef required')
    _need(_ref(reference['uri'],reference['schema'])==reference,'artifact bytes or identity differ')
    return read_artifact(reference)


def _material(value):
    ref=deepcopy(value) if type(value)is dict else _ref(value)
    return ref,_read(ref)


def _write(path, document):
    with path.open('x')as stream:
        json.dump(document,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    return _ref(path)


def stateful_holdout_policy(*,max_relative_prediction_error,minimum_observed_gain):
    """Return an explicit policy for callers to review and persist before use."""
    result=dict(schema=POLICY,schema_version=1,timer=deepcopy(TIMER),warmup_pairs=2,
        measurement_pairs=7,ordering='ALTERNATING_AB_BA',
        max_relative_prediction_error=max_relative_prediction_error,minimum_observed_gain=minimum_observed_gain,
        relative_error_denominator='OBSERVED_NATIVE_MEDIAN_OF_EPISODE_MEAN_METHOD_DURATIONS',
        fit_or_threshold_adjustment_from_holdout=False,automatic_deployment_authorized=False)
    _policy(result);return result


def _policy(document):
    _need(type(document.get('schema'))is str and type(document.get('schema_version'))is int,'versioned numeric policy required')
    _need('timer'not in document or document['timer']==TIMER,'explicit policy timer/includes/excludes are incompatible')
    _need((document.get('warmup_pairs',2),document.get('measurement_pairs',document.get('implementation_pairs',7)),document.get('ordering','ALTERNATING_AB_BA'))
        ==(2,7,'ALTERNATING_AB_BA'),'fixed two warmup pairs and seven alternating AB/BA pairs required')
    _need(document.get('fit_or_threshold_adjustment_from_holdout',False)is False
        and document.get('automatic_deployment_authorized',False)is False,'holdout cannot authorize fitting or deployment')
    _need(document.get('relative_error_denominator','OBSERVED_NATIVE_MEDIAN_OF_EPISODE_MEAN_METHOD_DURATIONS')in(
        'OBSERVED_NATIVE_MEDIAN_OF_EPISODE_MEAN_METHOD_DURATIONS','OBSERVED_NATIVE_MEDIAN_OF_SEVEN_EPISODE_MEAN_METHOD_DURATIONS'),
        'explicit observed-native error denominator required')
    for key in ('max_relative_prediction_error','minimum_observed_gain'):
        value=document.get(key)
        _need(type(value)in(int,float) and 0<=value<1,'explicit finite policy threshold in [0,1) required')


def _freeze_policy(reference,document,capture_epoch):
    _policy(document)
    return dict(schema=POLICY,schema_version=2,purpose='FOR_FUTURE_CAPTURE',
        input_policy=reference,input_policy_document=deepcopy(document),timer=deepcopy(TIMER),
        capture_epoch=capture_epoch,harness=_ref(__file__,'scar.implementation-source'),
        implementation_sources=_pins(),warmup_pairs=2,measurement_pairs=7,ordering='ALTERNATING_AB_BA',
        max_relative_prediction_error=document['max_relative_prediction_error'],minimum_observed_gain=document['minimum_observed_gain'],
        relative_error_denominator='OBSERVED_NATIVE_MEDIAN_OF_SEVEN_EPISODE_MEAN_METHOD_DURATIONS',
        fit_or_threshold_adjustment_from_holdout=False,automatic_deployment_authorized=False,
        prior_samples_authorized=False)


def _pins():
    root=Path(__file__).resolve().parents[1]
    names=('compiler/stateful_holdout.py','compiler/stateful_runtime.py','compiler/export_stateful.py',
        'compiler/stateful_model.py','runtime/stateful_guard.py','runtime/stateful_contract.py',
        'runtime/native_stateful.py','runtime/entry_runner.py','trace/code_snapshot.py',
        'optimization/stateful_entry_cost.py',
        'codegen/stateful.py','codegen/cpp.py','codegen/stateful_v1.cpp.in',
        'codegen/stateful_abi_v1.h','codegen/scar_abi_v1.h','codegen/loop_v1.cpp.in')
    return {name:_ref(root/name,'scar.implementation-source')for name in names}


def _derive(request, frozen_policy):
    from scar.optimization.stateful_cpu_choices import _domain,stateful_cpu_cost_context
    from scar.compiler.stateful_model import read_stateful,stateful_digest
    _need(type(request)is dict and set(request)=={'model','policy','specialization','prior_correctness','entry',
        'argv','output_token','caller_cpu_ids','output_directory','capture_epoch'},'complete known holdout request fields required')
    model=_read(request['model']);policy=_read(request['policy']);_policy(policy)
    frozen=_read(frozen_policy)
    epoch=request['capture_epoch']
    _need(type(epoch)is str and len(epoch)==32 and all(c in '0123456789abcdef'for c in epoch),'prepared capture epoch required')
    _need(frozen==_freeze_policy(request['policy'],policy,epoch),'generated future-capture policy differs')
    _need((model.get('schema'),model.get('schema_version'))==('scar.stateful-activity-entry-model',1),
        'source-accounted stateful activity entry model required')
    spec=_read(request['specialization']);candidate=_read(spec['candidate'])
    _need((spec.get('schema'),spec.get('schema_version'))==('scar.observed-stateful-specialization',1)
        and(candidate.get('schema'),candidate.get('schema_version'))==('scar.stateful-validation-candidate',1),
        'original observed specialization and F4 candidate required')
    domain=_domain(_read(spec['domain']));context=stateful_cpu_cost_context(domain,candidate['candidate_id'])
    facts=_read(model['facts']);activities=_read(model['activity_costs']);program=_read(spec['program'])
    _need(facts.get('context')==context and activities.get('candidate_id')==candidate['candidate_id']
        and activities.get('domain')==spec['domain']
        and facts.get('derivation',{}).get('activity_cost_receipt')==model['activity_costs'],
        'model context, candidate or original activity domain differs')
    _need(program.get('entry')==request['entry']['uri'] and candidate.get('program')==spec['program'],
        'actual entry and specialization program differ')
    _need(_ref(request['entry']['uri'],'scar.python-source')==request['entry'],'actual entry source changed')
    native=_read(candidate['native_artifact']);directory=Path(candidate['native_directory']).resolve()
    _need((native.get('schema'),native.get('schema_version'),native.get('stateful_abi_version'))
        ==('scar.native-stateful-artifact',1,1),'existing stateful ABI1 backend required')
    _need(all(type(name)is str and Path(name).name==name for name in native['files'].values()),'native members must be local files')
    _need(_ref(directory/'artifact.json',native['schema'])==candidate['native_artifact'],'actual native manifest differs')
    members={key:_ref(directory/name,'scar.native-member')for key,name in native['files'].items()}
    _need(all(members[key]['sha256']==native['sha256'][key]for key in members),'native member bytes differ')
    region=read_stateful(directory/native['files']['region'])
    _need(stateful_digest(region)==spec['region_sha256'],'actual region differs from specialization')
    sources={path:_ref(path,'scar.python-source')for path in sorted({region.source.path,*(c.source.path for c in region.callees)})}
    _need(all(sources[row.path]['sha256']==row.sha256 for row in (region.source,*(c.source for c in region.callees))),
        'original source snapshots changed')
    prior=_read(request['prior_correctness'])
    prior_harness=prior.get('harness',{})
    _need(type(prior_harness)is dict and set(prior_harness)=={'uri','schema','sha256','size_bytes'}
        and _ref(prior_harness['uri'],prior_harness['schema'])==prior_harness,'independent correctness harness source differs')
    comparisons=prior.get('comparison',{})
    _need(prior.get('status')=='PASSED_EXPLICIT_NATIVE_TRIAL' and prior.get('entry')==request['entry']
        and prior.get('specialization')==request['specialization']
        and prior.get('native_artifact')==candidate['native_artifact']
        and prior.get('native_trial',{}).get('native_calls',0)>0
        and prior.get('native_trial',{}).get('fallback_calls')==0
        and all(comparisons.get(key)is True for key in ('final_declared_state_bytes_and_types_equal',
            'per_boundary_pre_post_state_bytes_and_types_equal','source_sha256_before_after_equal')),
        'independent source-bound full-trajectory correctness receipt required')
    argv=request['argv'];token=request['output_token']
    _need(type(argv)is list and all(type(value)is str for value in argv),'explicit string argv list required')
    _need(token is None or type(token)is str and token and argv.count(token)==1,
        'output token must occur once as a complete argv token')
    _need(request['caller_cpu_ids']==sorted(os.sched_getaffinity(0)),'prepared default caller mask changed')
    _need(type(model.get('duration_ns'))is int and model['duration_ns']>=0,'frozen nonnegative native prediction required')
    episodes=[]
    for phase,count in (('warmup',2),('measurement',7)):
        for index in range(count):
            for label in (('A','B')if index%2==0 else('B','A')):
                directory=Path(request['output_directory'])/'run'/(phase+'-'+str(index)+'-'+label)
                output=None if token is None else str(directory/'program-output')
                episodes.append(dict(phase=phase,index=index,label=label,directory=str(directory),capture_epoch=epoch,
                    program_output_directory=output,argv=[output if token is not None and value==token else value for value in argv]))
    return dict(schema=DESIGN,schema_version=2,request=deepcopy(request),frozen_policy=frozen_policy,
        policy_contract=frozen,timer=deepcopy(TIMER),context=context,
        candidate=spec['candidate'],domain=spec['domain'],program=spec['program'],native_artifact=candidate['native_artifact'],
        native_members=members,model_facts=model['facts'],activity_costs=model['activity_costs'],
        source_refs=sources,implementation_sources=_pins(),
        predicted_native_method_ns=model['duration_ns'],purpose='FOR_FUTURE_CAPTURE',
        episodes=episodes,
        python=dict(executable=sys.executable,version=sys.version),
        preparation_target_executed=False,native_loaded_during_preparation=False,
        default_backend='NO_OP',deployment_authorized=False,model_qualification_promoted=False)


def prepare_stateful_holdout(model,policy,specialization,entry,argv,out,*,output_token=None,prior_correctness):
    """Persist the complete design before any target import or native load."""
    model_ref,_=_material(model);policy_ref,policy_doc=_material(policy);_policy(policy_doc)
    spec_ref,_=_material(specialization);prior_ref,_=_material(prior_correctness)
    out=Path(out).resolve();_need(not out.exists(),'fresh holdout design directory required')
    request=dict(model=model_ref,policy=policy_ref,specialization=spec_ref,prior_correctness=prior_ref,
        entry=_ref(entry,'scar.python-source'),argv=deepcopy(argv),output_token=output_token,
        caller_cpu_ids=sorted(os.sched_getaffinity(0)),output_directory=str(out),capture_epoch=uuid.uuid4().hex)
    out.mkdir(parents=True)
    frozen=_write(out/'frozen-policy.json',_freeze_policy(policy_ref,policy_doc,request['capture_epoch']))
    document=_derive(request,frozen);reference=_write(out/'design.json',document)
    verify_stateful_holdout_design(reference['uri'],expected_sha256=reference['sha256'])
    return reference


def verify_stateful_holdout_design(design,*,expected_sha256):
    reference=_ref(design,DESIGN)
    _need(reference['sha256']==expected_sha256,'holdout design SHA256 differs')
    document=_read(reference)
    _need(document.get('schema_version')==2,'future design version 2 required')
    _need(document==_derive(document['request'],document['frozen_policy']),'complete holdout design cold replay differs')
    return deepcopy(document)


def _owner(module,region):
    owner=module;parts=region.source.qualname.split('.')
    _need(all(part.isidentifier()for part in parts),'ordinary module/class boundary required')
    for part in parts[:-1]:
        _need(type(owner)in(ModuleType,type),'ordinary source owner required');owner=vars(owner)[part]
    _need(type(owner)in(ModuleType,type),'ordinary source owner required')
    return owner,parts[-1]


def _scalar(value):
    import numpy as np
    if type(value)is int:return dict(type='builtins.int',representation=str(value))
    if type(value)is float:return dict(type='builtins.float',representation=value.hex(),bits=struct.pack('=d',value).hex())
    if type(value)is np.float64:return dict(type='numpy.float64',representation=float(value).hex(),bits=value.tobytes().hex())
    raise ValueError('declared scalar is outside exact numeric profile')


def _state(region,bound,*,contents):
    import numpy as np
    from scar.runtime.stateful_guard import _input
    state={}
    for row in region.inputs:
        value=_input(row.path,bound)
        if row.kind=='array':
            _need(type(value)is np.ndarray,'exact declared ndarray required')
            item=dict(kind='array',path=list(row.path),dtype=value.dtype.str,shape=list(value.shape),strides=list(value.strides))
            item.update(device='cpu',c_contiguous=bool(value.flags.c_contiguous),aligned=bool(value.flags.aligned),tensor_contents_read=contents)
            if contents:item['payload']=value.tobytes(order='C')
        else:item=dict(kind='scalar',path=list(row.path),**_scalar(value))
        state[row.id]=item
    return state


def _summary(state):
    result=deepcopy(state)
    for row in result.values():
        if 'payload'in row:
            raw=row.pop('payload');row.update(payload_bytes=len(raw),payload_sha256=sha256(raw).hexdigest())
    return result


def _stats(guard):
    value=guard.scar_stats
    return dict(native_calls=value['native_calls'],fallback_calls=value['fallback_calls'],reasons=dict(value['reasons']))


def _environment(design):
    _need(sorted(os.sched_getaffinity(0))==design['request']['caller_cpu_ids'],'caller affinity changed; repinning forbidden')
    _need(sys.gettrace()is None and sys.getprofile()is None,'holdout requires no tracing/profiling')
    monitor=getattr(sys,'monitoring',None)
    _need(monitor is None or all(monitor.get_tool(i)is None for i in range(6)),'holdout requires no monitoring tools')


@contextmanager
def _timed_boundary(module,region,signature,selected,guard,design):
    owner,name=_owner(module,region);saved=vars(owner)[name]
    observed=dict(last_bound=None,steps=[])
    @wraps(selected)
    def invoke(*args,**kwargs):
        bound=signature.bind(*args,**kwargs);bound.apply_defaults();observed['last_bound']=dict(bound.arguments)
        context=_state(region,bound.arguments,contents=False);_environment(design);before=_stats(guard)
        start=time.perf_counter_ns()
        result=selected(*args,**kwargs)
        stop=time.perf_counter_ns()
        after=_stats(guard);_environment(design)
        observed['steps'].append(dict(ordinal=len(observed['steps']),elapsed_ns=stop-start,inputs=context,
            capture_epoch=design['request']['capture_epoch'],
            native_calls=after['native_calls']-before['native_calls'],fallback_calls=after['fallback_calls']-before['fallback_calls']))
        return result
    installed=staticmethod(invoke)if type(saved)is staticmethod else invoke;setattr(owner,name,installed)
    try:yield observed
    finally:
        _need(vars(owner).get(name)is installed,'entry replaced the selected boundary')
        setattr(owner,name,saved)


def _episode(entry,module,region,signature,selected,guard,design,planned):
    from scar.runtime.entry_runner import run_original_script
    directory=Path(planned['directory']);directory.mkdir();argv=list(planned['argv'])
    output=None if planned['program_output_directory']is None else Path(planned['program_output_directory'])
    stdout,stderr=io.StringIO(),io.StringIO()
    with _timed_boundary(module,region,signature,selected,guard,design)as observed:
        with redirect_stdout(stdout),redirect_stderr(stderr):execution=run_original_script(entry,argv)
        _need(observed['last_bound']is not None,'entry never reached the declared source boundary')
        final=_state(region,observed['last_bound'],contents=True)
    files={str(path.relative_to(output)):path.read_bytes()for path in sorted(output.rglob('*'))if path.is_file()}if output is not None else {}
    return dict(steps=observed['steps'],final=final,files=files,stdout=stdout.getvalue(),stderr=stderr.getvalue(),execution=execution,argv=argv)


def _distribution(samples):
    return dict(samples_ns=samples,count=len(samples),median_ns=statistics.median(samples),mean_ns=statistics.mean(samples),
        min_ns=min(samples),max_ns=max(samples),sample_variance_ns2=statistics.variance(samples)if len(samples)>1 else None)


def run_stateful_holdout(design,*,expected_sha256):
    """Run exactly the persisted design; no fitting, automatic retry or deploy."""
    document=verify_stateful_holdout_design(design,expected_sha256=expected_sha256)
    design_ref=_ref(design,DESIGN);request=document['request'];_environment(document)
    directory=Path(request['output_directory'])/'run';directory.mkdir(exist_ok=False)
    # All admission above, including policy and implementation fingerprints,
    # precedes the first target import and native ELF load.
    from scar.codegen.stateful import load
    from scar.compiler.stateful_runtime import install_stateful_regions
    from scar.runtime.stateful_guard import _function
    candidate=_read(document['candidate']);artifact=load(candidate['native_directory']);region=artifact.region
    previous_path=list(sys.path);previous_bytecode=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    sys.path.insert(0,str(Path(region.source.path).parent));pairs=[]
    try:
        module=importlib.import_module(Path(region.source.path).stem)
        _need(Path(vars(module)['__file__']).resolve()==Path(region.source.path).resolve(),'ordinary import selected another source')
        owner,name=_owner(module,region);saved=vars(owner)[name]
        original=saved.__func__ if type(saved)is staticmethod else saved;_function(region.source,original)
        _need(type(original)is FunctionType,'ordinary original function required')
        signature=inspect.signature(original);_need(tuple(signature.parameters)==region.parameter_names,'original parameter layout differs')
        with install_stateful_regions(module,[artifact],trial=True,exclusive_inputs=True)as guards:
            guard=guards[0]
            for phase,count in (('warmup',2),('measurement',7)):
                for index in range(count):
                    order=('A','B')if index%2==0 else('B','A');runs={}
                    for label in order:
                        planned=next(row for row in document['episodes']if(row['phase'],row['index'],row['label'])==(phase,index,label))
                        runs[label]=_episode(request['entry']['uri'],module,region,signature,
                            original if label=='A'else guard,guard,document,planned)
                    a,b=runs['A'],runs['B'];native=sum(s['native_calls']for s in b['steps']);fallback=sum(s['fallback_calls']for s in b['steps'])
                    checks=dict(final_state_equal=a['final']==b['final'],stdout_equal=a['stdout']==b['stdout'],stderr_equal=a['stderr']==b['stderr'],
                        configured_output_files_equal=a['files']==b['files'],context_trajectory_equal=[s['inputs']for s in a['steps']]==[s['inputs']for s in b['steps']],
                        native_success=native==len(b['steps'])and native>0 and fallback==0)
                    public={label:dict(steps=row['steps'],final_state=_summary(row['final']),argv=row['argv'],execution=row['execution'],
                        stdout=row['stdout'],stderr=row['stderr'],files={name:dict(size_bytes=len(raw),sha256=sha256(raw).hexdigest())for name,raw in row['files'].items()})for label,row in runs.items()}
                    pairs.append(dict(phase=phase,index=index,order=list(order),checks=checks,implementations=public))
                    _need(all(checks.values()),'paired correctness/native success failed; no automatic retry')
            total_stats=_stats(guard)
        _need(vars(owner)[name]is saved,'original source descriptor was not restored')
    finally:sys.path[:]=previous_path;sys.dont_write_bytecode=previous_bytecode
    after_design=verify_stateful_holdout_design(design,expected_sha256=expected_sha256)
    measured=[p for p in pairs if p['phase']=='measurement'];summaries={}
    for label in ('A','B'):
        means=[statistics.mean(s['elapsed_ns']for s in p['implementations'][label]['steps'])for p in measured]
        summaries[label]=dict(episode_means=_distribution(means),method_calls=_distribution(
            [s['elapsed_ns']for p in measured for s in p['implementations'][label]['steps']]))
    original_ns=summaries['A']['episode_means']['median_ns'];native_ns=summaries['B']['episode_means']['median_ns']
    _need(original_ns>0 and native_ns>0,'positive observed duration required')
    relative_error=abs(document['predicted_native_method_ns']-native_ns)/native_ns
    gain=(original_ns-native_ns)/original_ns;policy=document['policy_contract']
    report=dict(schema=REPORT,schema_version=2,status='OBSERVED_FIXED_IMPLEMENTATIONS',design=design_ref,
        capture_epoch=request['capture_epoch'],observation_origin='COLLECTOR_TARGET_EXECUTION',
        frozen_policy=document['frozen_policy'],frozen_model=request['model'],timer=document['timer'],
        harness=document['implementation_sources']['compiler/stateful_holdout.py'],
        implementation_sources=document['implementation_sources'],
        original_source_refs_before={**document['source_refs'],request['entry']['uri']:request['entry']},
        original_source_refs_after={**after_design['source_refs'],request['entry']['uri']:request['entry']},
        policy_frozen_before_target=True,design_readback_before_target=True,design_readback_after_target=True,
        caller_affinity_changed_by_harness=False,original_descriptors_restored=True,
        pairs=pairs,summaries=summaries,guard_statistics_including_warmups=total_stats,
        frozen_policy_evaluation=dict(predicted_native_method_ns=document['predicted_native_method_ns'],
            relative_prediction_error=relative_error,observed_method_gain=gain,
            prediction_error_threshold_passed=relative_error<=policy['max_relative_prediction_error'],
            observed_gain_threshold_passed=gain>=policy['minimum_observed_gain']),
        per_step_array_content_reads_by_observer=0,final_state_reads_per_episode=1,
        independent_prior_correctness=request['prior_correctness'],performance_measured=True,
        model_fitted_from_holdout=False,model_qualification_promoted=False,deployment_authorized=False,
        whole_program_speedup_measured=False,output_comparison_scope='CONFIGURED_OUTPUT_DIRECTORY_ONLY',
        noise_limitations=['Method samples are clustered within seven episodes, not independent observations.',
            'Clock/dispatch overhead is included without subtraction; scheduling, frequency and shared resources are uncontrolled.',
            'Caller control/I/O and lightweight observations outside each timer can affect inter-call state.',
            'No confidence bounds, physical-resource qualification or whole-program gain follows from these samples.'])
    return _write(directory/'result.json',report)


__all__=['TIMER','stateful_holdout_policy','prepare_stateful_holdout','verify_stateful_holdout_design','run_stateful_holdout']


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    prepare=commands.add_parser('prepare',help='persist a future design without target import or native load')
    for name in ('model','policy','specialization','entry','prior-correctness','out'):
        prepare.add_argument('--'+name,required=True)
    prepare.add_argument('--argv-json',required=True,help='explicit JSON list of original entry argument strings; [] is valid')
    prepare.add_argument('--output-token',help='optional complete argv token replaced by each fresh program output directory')
    for command in ('check','run'):
        child=commands.add_parser(command,help='cold-check the design'if command=='check'else'explicitly execute the frozen target capture')
        child.add_argument('--design',required=True);child.add_argument('--sha256',required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':
        result=prepare_stateful_holdout(args.model,args.policy,args.specialization,args.entry,json.loads(args.argv_json),
            args.out,output_token=args.output_token,prior_correctness=args.prior_correctness)
    elif args.command=='check':
        document=verify_stateful_holdout_design(args.design,expected_sha256=args.sha256)
        result=dict(status='VERIFIED_FUTURE_DESIGN',design=_ref(args.design,DESIGN),
            capture_epoch=document['request']['capture_epoch'],target_executed=False,native_loaded=False)
    else:result=run_stateful_holdout(args.design,expected_sha256=args.sha256)
    print(json.dumps(result,sort_keys=True));return 0


if __name__=='__main__':
    raise SystemExit(main())

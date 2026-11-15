"""Bootstrap a declared normal-entry cost model from retained fresh evidence.

The application entry never runs here. Independent fresh children measure only
cold gate import and metadata replay. Candidate totals remain candidate-policy
observations; adding that direct component is explicitly a cross-policy model.
"""
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys


_ASSUMPTIONS = [
    'STATIONARY_IO_AND_IMPORT_COSTS',
    'STATIONARY_GATE_METADATA_REPLAY_COSTS',
    'SAMPLED_MAX_NOT_UPPER_BOUND',
    'CANDIDATE_PLUS_DIRECT_GATE_COMPONENT_IS_MODEL_NOT_OBSERVED_COST_GATED_TOTAL',
    'ORIGINAL_DIRECT_ENTRY_TOTAL_IS_CONSERVATIVE_REFERENCE_NOT_OBSERVED_COST_GATED_TOTAL',
]

_CHILD = r'''
import json,os,sys,time
from pathlib import Path
request=json.loads(Path(sys.argv[1]).read_bytes())
from scar.runtime.bundle_metadata import inspect_bundle
from scar.runtime.entry_cli import entry_scope
metadata=inspect_bundle(request['bundle'],expected_admission_sha256=request['pin'])
contract=metadata['contract']; scope=entry_scope(request['entry'],request['argv'])
assert scope==request['entry_scope']
assert 'scar.optimization.deployment_gate' not in sys.modules
assert 'scar.optimization.deployment_cost' not in sys.modules
start=time.perf_counter_ns()
from scar.optimization.deployment_gate import deployment_gate
gate=deployment_gate(request['seed_plan'],bundle_ids=contract['native']['ids'],
    admission_sha256=request['pin'],entry_scope=scope,reuse_count=request['reuse_count'],
    available_cpu_ids=sorted(os.sched_getaffinity(0)),memory_budget_bytes=request['memory_budget_bytes'],
    required_cpu_ids=contract['cpu']['ids'],required_memory_bytes=request['required_memory_bytes'],
    memory_scope=request['memory_scope'])
elapsed=(time.perf_counter_ns()-start)/1e9
forbidden=[name for name in sys.modules if name=='numpy' or name.startswith(
    ('scar.compiler','scar.codegen','scar.analysis','scar.ir'))]
result=dict(schema='scar.entry-gate-component-observation',schema_version=1,
    seconds=elapsed,unit='seconds',basis='OBSERVED',measurement_scope='DIRECT_COMPONENT_TIMING',
    component='COLD_IMPORT_DEPLOYMENT_GATE_AND_EXACT_METADATA_GATE_REPLAY',
    prefix_excluded=['inspect_bundle','entry_scope','process_startup'],
    original_entry_executed=False,native_library_loaded=False,forbidden_imports=forbidden,
    seed_plan=request['seed_plan'],seed_plan_sha256=request['seed_plan_sha256'],
    entry_scope=scope,bundle_ids=contract['native']['ids'],admission_sha256=request['pin'],
    gate=gate,pid=os.getpid())
Path(sys.argv[2]).write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
assert not forbidden
assert not gate['frontier'],gate
assert elapsed>0
print(json.dumps(dict(seconds=elapsed,decision=gate['decision'],target_executed=False)))
'''


def _write(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


def _receipt(path):
    path=Path(path).resolve(); raw=path.read_bytes()
    return dict(path=str(path),sha256=sha256(raw).hexdigest(),bytes=len(raw))


def _reference(path,pointer):
    row=_receipt(path); document=json.loads(Path(path).read_bytes())
    return dict(artifact=dict(uri=row['path'],sha256=row['sha256'],size_bytes=row['bytes'],schema=document['schema']),locator=pointer)


def _observation(value,reference,*,variant,scope,bundle,pin,reuse_count,budget,required,memory_scope,cpu_ids,bundle_ids):
    from statistics import median
    from scar.optimization.deployment_cost import _fact, _number
    if is_dataclass(reference): reference=asdict(reference)
    if type(reference) is not dict: raise ValueError('original timing ProviderFactReference required')
    recorded,_=_fact(reference)
    key='baseline' if variant=='ORIGINAL' else 'native'
    if (reference['locator']!='#/aggregates/'+key+'/median' or _number(value) is None
            or value<=0 or _number(recorded)!=_number(value)):
        raise ValueError('timing differs from the requested generic aggregate FactReference')
    path=Path(reference['artifact']['uri']); raw=path.read_bytes()
    if sha256(raw).hexdigest()!=reference['artifact']['sha256'] or len(raw)!=reference['artifact']['size_bytes']:
        raise ValueError('entry benchmark changed during admission')
    document=json.loads(raw)
    fields={'schema','schema_version','boundary','entry_scope','native_launch_policy',
            'runtime_bundle','reuse_count','memory','cpu_ids','observations','aggregates'}
    if (set(document)!=fields or document.get('schema')!='scar.entry-benchmark'
            or type(document.get('schema_version')) is not int or document['schema_version']!=1
            or document['boundary']!='FRESH_PROCESS' or document['entry_scope']!=scope
            or document['native_launch_policy']!='MEASURE_ADMITTED_CANDIDATE'
            or document['runtime_bundle']!=dict(ids=bundle_ids,pin=pin)
            or type(document['reuse_count']) is not int or document['reuse_count']!=reuse_count
            or document['memory']!=dict(budget_bytes=budget,required_bytes=required,scope=memory_scope)
            or type(document['cpu_ids']) is not list or any(type(cpu) is not int for cpu in document['cpu_ids'])
            or document['cpu_ids']!=cpu_ids):
        raise ValueError('complete generic fresh-entry benchmark context differs')
    observations=document['observations']; aggregates=document['aggregates']
    if (type(observations) is not list or not observations or type(aggregates) is not dict
            or set(aggregates)!={'baseline','native'}):
        raise ValueError('two complete benchmark sample populations required')
    values={'ORIGINAL':[],'NATIVE':[]}; retained=[]
    for row in observations:
        if (type(row) is not dict or set(row)!={'variant','seconds','command','execution','source_receipts'}
                or row['variant'] not in values or type(row['seconds']) not in (float,int)
                or not math.isfinite(row['seconds']) or row['seconds']<=0
                or type(row['command']) is not list or any(type(arg) is not str for arg in row['command'])
                or type(row['source_receipts']) is not list or not row['source_receipts']):
            raise ValueError('complete finite observation, exact command and source receipts required')
        values[row['variant']].append(row['seconds'])
        for source in row['source_receipts']:
            if type(source) is not dict: raise ValueError('typed source receipt required')
            if set(source)=={'artifact','locator'}:
                _fact(source)  # Opaque original source; no testcase schema is interpreted.
            elif set(source)=={'path','sha256','bytes'}:
                if (type(source['path']) is not str or not Path(source['path']).is_absolute()
                        or _receipt(source['path'])!=source):
                    raise ValueError('original observation source byte identity differs')
            else: raise ValueError('ProviderFactReference or exact Path/SHA/bytes receipt required')
            retained.append(deepcopy(source))
        command=row['command']
        if row['variant']=='ORIGINAL':
            if command!=[sys.executable,'-B',scope['entry_path'],*scope['argv']]:
                raise ValueError('original direct-entry command differs')
            if row['execution'] is not None and type(row['execution']) is not dict:
                raise ValueError('invalid optional original execution record')
            continue
        try:
            split=command.index('--')
            def option(name): return command[command.index(name)+1]
            same=(command[:5]==[sys.executable,'-B','-m','scar','run']
                  and command[split+1:]==[scope['entry_path'],*scope['argv']]
                  and Path(option('--bundle')).resolve()==bundle and option('--admission-sha256')==pin
                  and int(option('--reuse-count'))==reuse_count and int(option('--memory-budget-bytes'))==budget
                  and '--exclusive-inputs' in command and '--measure-candidate' in command
                  and '--deployment-plan' not in command)
        except (ValueError,IndexError): same=False
        execution=row['execution']
        if (not same or type(execution) is not dict or execution.get('schema')!='scar.automatic-entry-execution'
                or execution.get('entry_scope')!=scope or execution.get('mode')!='MEASUREMENT_ONLY'
                or execution.get('expected_reuse_count')!=reuse_count or execution.get('required_memory_bytes')!=required
                or execution.get('memory_scope')!=memory_scope or execution.get('memory_budget_bytes')!=budget):
            raise ValueError('generic candidate launch/runtime context differs')
        statistics=execution.get('execution',{}).get('statistics',{})
        if (type(statistics.get('native_calls')) is not int or statistics['native_calls']!=reuse_count
                or type(statistics.get('fallback_calls')) is not int or statistics['fallback_calls']!=0):
            raise ValueError('generic candidate installed-boundary execution differs')
    if not values['ORIGINAL'] or len(values['ORIGINAL'])!=len(values['NATIVE']):
        raise ValueError('balanced nonempty original/native sample populations required')
    for name,role in (('baseline','ORIGINAL'),('native','NATIVE')):
        aggregate=aggregates[name]
        if (type(aggregate) is not dict or set(aggregate)!={'samples','median'}
                or type(aggregate['samples']) is not list
                or any(type(sample) not in (int,float) or not math.isfinite(sample) or sample<=0 for sample in aggregate['samples'])
                or aggregate['samples']!=values[role] or type(aggregate['median']) not in (float,int)
                or _number(aggregate['median'])!=_number(median(values[role]))):
            raise ValueError('generic benchmark aggregate differs from its complete observation population')
    return dict(value=value,unit='seconds',basis='OBSERVED',source=deepcopy(reference),
        original_boundary=document['boundary'],
        original_launch_policy='DIRECT_ENTRY' if variant=='ORIGINAL' else document['native_launch_policy'],
        entry_scope=scope,source_receipts=retained,
        target_policy_association='CROSS_POLICY_MODEL_INPUT_NOT_OBSERVED_NORMAL_TOTAL')


def _plan_files(directory,context,original,native,count,available,budget,observations,*,seed):
    from scar.optimization.deployment_cost import plan_deployment, verify_deployment_plan
    directory.mkdir()
    scope='UNCALIBRATED_BOOTSTRAP_ARITHMETIC_SEED_NEVER_EXECUTED' if seed else 'DECLARED_CROSS_POLICY_COMPLETE_SCOPE_MODEL'
    facts=dict(schema='scar.entry-gate-model-facts',schema_version=1,context=context,
        model_kind=scope,original_observations=observations,
        costs={key:dict(value=value,unit='seconds',basis='DECLARED_MODEL',measurement_scope=scope,reuse_count=count)
               for key,value in (('original_total',original),('native_total',native))})
    _write(directory/'facts.json',facts)
    profile=dict(schema='scar.deployment-cost-profile',schema_version=1,mode='COMPLETE_SCOPE',
        context=context,context_evidence=[_reference(directory/'facts.json','#/context')],
        observed_reuse_count=count,model_assumptions=_ASSUMPTIONS+(['UNCALIBRATED_BOOTSTRAP_SEED_NEVER_FOR_TARGET_EXECUTION'] if seed else []),
        costs={key:dict(value=value,unit='seconds',basis='DECLARED_MODEL',evidence=[_reference(directory/'facts.json','#/costs/'+key)])
               for key,value in (('original_total',original),('native_total',native))},
        provenance=dict(model_kind=scope,source_observations_are_not_cost_gated_observations=True,target_executed=False))
    _write(directory/'profile.json',profile)
    plan=plan_deployment(directory/'profile.json',context=context,reuse_count=count,
        available_cpu_ids=available,memory_budget_bytes=budget)
    verify_deployment_plan(plan); _write(directory/'plan.json',plan)
    return plan


def calibrate_entry_gate(bundle,pin,entry,argv,*,reuse_count,memory_budget_bytes,
                         baseline_seconds,native_candidate_seconds,out,
                         baseline_evidence=None,candidate_evidence=None,samples=3):
    """Return a sourced DECLARED_MODEL plan; run no application or native kernel.

    Evidence is a numeric FactReference into a generic scar.entry-benchmark
    record. Complete observation populations and their supplied receipts bind
    the source scope; no external testcase fields or filenames are inferred.
    A LOAD_NATIVE_CANDIDATE result is a prior model, never a deployment result.
    """
    from scar.runtime.bundle_metadata import inspect_bundle
    from scar.runtime.entry_cli import entry_scope
    from scar.optimization.deployment_cost import verify_deployment_plan
    if type(samples) is not int or not 1<=samples<=3: raise ValueError('one to three independent component samples required')
    if type(reuse_count) is not int or reuse_count<1: raise ValueError('exact positive reuse count required')
    if type(memory_budget_bytes) is not int or memory_budget_bytes<0: raise ValueError('explicit memory budget required')
    if any(type(value) not in (int,float) or not math.isfinite(value) or value<=0 for value in (baseline_seconds,native_candidate_seconds)):
        raise ValueError('positive finite fresh observations required')
    bundle=Path(bundle).resolve(); scope=entry_scope(entry,list(argv)); directory=Path(out).resolve()
    if directory.exists(): raise ValueError('fresh bootstrap output directory required')
    metadata=inspect_bundle(bundle,expected_admission_sha256=pin); contract=metadata['contract']
    stateful=contract['schema']=='scar.stateful-runtime-contract'
    required=contract['memory']['total_region_array_payload_bytes'] if stateful else contract['array']['final_bytes']
    memory_scope=contract['memory']['scope'] if stateful else 'REGION_INCREMENTAL_FRESH_FINAL_PAYLOAD_EXCLUDES_EXTERNAL_INPUTS_AND_PROCESS_RSS'
    cpus=contract['cpu']['ids']; available=sorted(os.sched_getaffinity(0))
    if cpus!=sorted(set(cpus)) or not set(cpus)<=set(available) or required>memory_budget_bytes:
        raise ValueError('actual CPU allocation or complete region payload unavailable')
    kwargs=dict(scope=scope,bundle=bundle,pin=pin,reuse_count=reuse_count,budget=memory_budget_bytes,required=required,memory_scope=memory_scope,cpu_ids=cpus,bundle_ids=contract['native']['ids'])
    observations=dict(baseline=_observation(baseline_seconds,baseline_evidence,variant='ORIGINAL',**kwargs),
                      candidate=_observation(native_candidate_seconds,candidate_evidence,variant='NATIVE',**kwargs))
    context=dict(boundary='FRESH_PROCESS',launch_policy='COST_GATED_DEPLOYMENT',
        original_launch_policy='DIRECT_ENTRY',native_launch_policy='COST_GATED_DEPLOYMENT',
        workload_id='entry:'+scope['entry_sha256'],artifact_id=contract['native']['ids']['build_id'],
        runtime_bundle=dict(ids=contract['native']['ids'],admission_sha256=pin),entry_scope=scope,
        cpu_ids=cpus,required_memory_bytes=required,memory_scope=memory_scope,
        external_assumptions=contract['external_assumptions'],runtime_guards=contract.get('guards',[]))
    directory.mkdir(parents=True)
    seed=_plan_files(directory/'seed',context,baseline_seconds,native_candidate_seconds,reuse_count,
                     available,memory_budget_bytes,observations,seed=True)
    request=dict(bundle=str(bundle),pin=pin,entry=scope['entry_path'],argv=scope['argv'],entry_scope=scope,
        seed_plan=str(directory/'seed/plan.json'),seed_plan_sha256=_receipt(directory/'seed/plan.json')['sha256'],
        reuse_count=reuse_count,memory_budget_bytes=memory_budget_bytes,required_memory_bytes=required,memory_scope=memory_scope)
    _write(directory/'component-request.json',request); measured=[]
    for index in range(samples):
        path=directory/('gate-component-'+str(index)+'.json')
        process=subprocess.run([sys.executable,'-B','-c',_CHILD,str(directory/'component-request.json'),str(path)],
            capture_output=True,text=True,timeout=60)
        (directory/('gate-component-'+str(index)+'.stdout')).write_text(process.stdout)
        (directory/('gate-component-'+str(index)+'.stderr')).write_text(process.stderr)
        if process.returncode or not path.exists():
            result=dict(schema='scar.entry-gate-calibration',schema_version=1,status='UNKNOWN',decision='NO_OP',
                reason='independent component measurement failed',child_returncode=process.returncode,
                target_executed=False,unknown_component_is_not_zero=True)
            _write(directory/'result.json',result); return result
        sample=json.loads(path.read_bytes())
        if (sample['forbidden_imports'] or sample['native_library_loaded'] or sample['original_entry_executed']
                or sample['gate']['frontier'] or sample['seed_plan_sha256']!=request['seed_plan_sha256']):
            raise ValueError('component scope or seed replay differs')
        measured.append(sample['seconds'])
    maximum=max(measured)
    component=dict(schema='scar.entry-gate-component-summary',schema_version=1,
        samples_seconds=measured,aggregation='SAMPLED_MAX_NOT_UPPER_BOUND',sampled_max_seconds=maximum,
        measurement_scope='DIRECT_COMPONENT_TIMING',basis='OBSERVED',unit='seconds',
        evidence=[_reference(directory/('gate-component-'+str(i)+'.json'),'#/seconds') for i in range(samples)],
        target_executed=False,native_library_loaded=False)
    _write(directory/'gate-component-summary.json',component)
    model_sources=deepcopy(observations)
    model_sources['direct_gate_component']=dict(value=maximum,unit='seconds',basis='OBSERVED',
        source=_reference(directory/'gate-component-summary.json','#/sampled_max_seconds'),
        formula='SAMPLED_MAX_OF_INDEPENDENT_COLD_GATE_IMPORT_AND_METADATA_REPLAY')
    plan=_plan_files(directory/'model',context,baseline_seconds,native_candidate_seconds+maximum,reuse_count,
                     available,memory_budget_bytes,model_sources,seed=False)
    verify_deployment_plan(directory/'model/plan.json')
    from scar.optimization.deployment_gate import deployment_gate
    gate=deployment_gate(directory/'model/plan.json',bundle_ids=contract['native']['ids'],admission_sha256=pin,
        entry_scope=scope,reuse_count=reuse_count,available_cpu_ids=available,memory_budget_bytes=memory_budget_bytes,
        required_cpu_ids=cpus,required_memory_bytes=required,memory_scope=memory_scope)
    _write(directory/'gate.json',gate)
    result=dict(schema='scar.entry-gate-calibration',schema_version=1,status='DECLARED_MODEL_BOOTSTRAPPED',
        decision=gate['decision'],context=context,baseline_direct_reference_seconds=baseline_seconds,
        native_candidate_observed_seconds=native_candidate_seconds,gate_component_sampled_max_seconds=maximum,
        native_model_seconds=native_candidate_seconds+maximum,model_assumptions=_ASSUMPTIONS,
        model_basis='DECLARED_MODEL',whole_scope_cost_observed=False,normal_total_measured=False,
        seed_only_for_offline_metadata_replay=True,seed_plan=str(directory/'seed/plan.json'),
        profile=str(directory/'model/profile.json'),deployment_plan=str(directory/'model/plan.json'),gate=str(directory/'gate.json'),
        source_observations=observations,component_summary=_receipt(directory/'gate-component-summary.json'),
        target_executed=False,native_library_loaded=False,native_compiled=False,deployment_authorized=False)
    _write(directory/'result.json',result)
    return result


__all__=['calibrate_entry_gate']

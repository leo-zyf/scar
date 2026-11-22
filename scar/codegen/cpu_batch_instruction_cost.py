"""Shared immutable canonical kernels, independently sampled CPU contexts.

Serial and OpenMP use separate compiler envelopes. CPU masks are arguments of
the measurement harness, never timing extrapolation parameters. LLVM-MCA may
share an exactly identical instruction block; measured nanoseconds never move
between masks. A fresh fixed batch holdout is frozen before pilot sampling.
"""
import argparse
from copy import deepcopy
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re
import resource
import subprocess
import sys
import time

from . import cpu_instruction_cost as C
from .cpu_fused_cost import (_HERE, _Layout, _dag_layouts, _hash, _json, _probe_template,
                             _receipt, _render_dag_probe, _require, _save)
from .cpu_fused_cost_models import _dag_case
from scar.ir.execution_recipe import artifact_reference,read_artifact

PACK='scar.cpu-batch-instruction-pack'
MODEL='scar.cpu-batch-instruction-context-model'


class _Mask(ctypes.Structure):
    _fields_=[('workers',ctypes.c_int32),('reserved',ctypes.c_int32),('cpu_ids',ctypes.POINTER(ctypes.c_int32))]


def _registry():
    result=[deepcopy(r) for r in C._registry() if not r['holdout']]
    result[0]['id']='batch_stage_floor'
    for i,(family,n,extra) in enumerate((('SERIAL_FANIN',3,6),('SERIAL_FANIN',5,12),
        ('BALANCED_FANIN',7,7),('BALANCED_FANIN',3,7),('PARALLEL_BRANCHES',3,6),
        ('PARALLEL_BRANCHES',7,8),('SHARED_BRANCHES',5,8),('SHARED_BRANCHES',7,12))):
        case=_dag_case(family,n,extra,seed=907+i,holdout=True);pool=5 if i%2==0 else 8
        for node in case['graph']:
            node['inputs']=['s'+str(int(v[1:])%pool) if v.startswith('s') else v for v in node['inputs']]
        case.update(id='batch_fresh_holdout_'+str(i),role='FRESH_HOLDOUT',operand_pool=pool)
        result.append(case)
    return result


def _configurations(context):
    result=[];workers=len(context['worker_cpu_ids'])
    for i,case in enumerate(_registry()):
        shapes=((427,587),(622,764)) if case['holdout'] else ((256,320),(512,512),(768,832))
        groups=(1,2) if case['role'] in ('MEMORY','FRESH_HOLDOUT') else (1,)
        if case['role']=='DISPATCH':shapes=((1,1),)
        for group in groups:
            if group==2 and case['array_loads']==1:continue
            for r,c in shapes:
                result.append(dict(id='batch-configuration-'+str(len(result)),case_index=i,kernel_index=i*2+group-1,
                    groups=group,rows=r,cols=c,workers=workers,elements=((r+workers-1)//workers)*c,
                    array_loads=case['array_loads'],layouts=_dag_layouts(case['array_loads'],c,group),
                    role=case['role'],holdout=case['holdout']))
    random.Random(1001831).shuffle(result);return result


def _instruction_tool(tool):
    return {k:tool[k] for k in ('executable','version','cpu_model','iterations','noalias')}


def _compatible(pack,context):
    for key,value in pack['resource_template'].items():
        if key not in ('worker_cpu_ids','executor_id'):
            _require(context[key]==value,'batch shared compiler/hardware context differs: '+key)
    profile = C.instruction_cost_profile(pack['mode'])
    profile.require_workers(len(context['worker_cpu_ids']))
    _require(set(context['worker_cpu_ids'])<=set(pack['available_cpu_ids']),'mask outside frozen hardware availability')
    _require(sorted(os.sched_getaffinity(0))==context['caller_affinity_cpu_ids']
        and all(os.environ.get(k)==v for k,v in context['native_thread_environment'].items()),'current caller/thread environment differs')
    tool=profile.tool_contract(context)
    _require(_instruction_tool(tool)==_instruction_tool(pack['tool']),'shared instruction tool/ISA differs')
    if pack['mode']=='parallel':
        _require(tool['openmp_runtime']==pack['tool']['openmp_runtime']
            and tool['compiler_openmp_runtime']==pack['tool']['compiler_openmp_runtime'],'parallel runtime library differs')
    return tool



def _source(available):
    template = _probe_template('cpu_batch_instruction_v1.cpp.in',
                               CPU_SLOTS=str(max(available) + 1), MAX_WORKERS=str(len(available)))
    boundary = '#ifdef _OPENMP\n#pragma omp barrier\n#else\nasm volatile("" : : "g"(&a) : "memory");\n#endif\n'
    return _render_dag_probe(template, _registry(), boundary_bodies={'DISPATCH': boundary})


def _normalized_block(text):
    definitions=re.findall(r'(?m)^\s*(\.L\d+):',text);mapping={key:'.Lblock'+str(i) for i,key in enumerate(definitions)}
    def rename(match):return mapping.get(match[0],match[0])
    return '\n'.join(re.sub(r'\.L\d+\b',rename,line.strip()) for line in text.splitlines() if line.strip())+'\n'


def _block(text,tool,cache):
    normalized=_normalized_block(text);identity=dict(instruction_text=normalized,tool=_instruction_tool(tool))
    key=_hash(_json(identity));directory=Path(cache)/key;path=directory/'record.json'
    if path.exists():
        reference=artifact_reference(path,'scar.mca-shared-instruction-block');record=read_artifact(reference)
        _require(record['identity']==identity,'shared instruction block key collision')
        _require(_receipt(Path(record['input']['path']))==record['input']
            and Path(record['input']['path']).read_text()==normalized
            and _receipt(Path(record['result']['path']))==record['result'],'shared MCA block receipt differs')
        return reference,record,True
    directory.mkdir(parents=True);source=directory/'block.s';source.write_text(normalized)
    command=[tool['executable']['path'],'-mcpu='+tool['cpu_model'],'-iterations='+str(tool['iterations']),
        '--noalias=true','--json',str(source)]
    p=subprocess.run(command,capture_output=True,text=True);(directory/'stderr.txt').write_text(p.stderr)
    _require(p.returncode==0,'shared MCA block rejected: '+p.stderr[-500:]);result=json.loads(p.stdout)
    region=result['CodeRegions'][0];summary=region['SummaryView']
    _require(result['TargetInfo']['CPUName']==tool['cpu_model'] and not any(
        r['hasUnmodeledSideEffects'] for r in region['InstructionInfoView']['InstructionList']),
        'unsupported shared instruction model effect')
    output=directory/'mca.json';output.write_text(p.stdout)
    record=dict(schema='scar.mca-shared-instruction-block',schema_version=1,key=key,identity=identity,
        input=_receipt(source),result=_receipt(output),command=command,summary=summary,
        label_alpha_normalization_only=True,registers_and_memory_operands_preserved=True,
        timing_fact=False)
    _save(path,record);return artifact_reference(path,record['schema']),record,False


def _analyze(assembly,tool,cache,*,function=None,source_range=None,source_path=None):
    loops=C._natural_loops(C._lex_assembly(assembly),function,source_range,source_path)
    _require(bool(loops),'no supported store-bearing natural loop');rows=[];hits=0
    for loop in loops:
        _require(not re.search(r'\bcall\w*\b',loop['text']),'opaque call in shared instruction block')
        reference,record,hit=_block(loop['text'],tool,cache);hits+=hit;summary=record['summary']
        rows.append(dict(**{k:v for k,v in loop.items() if k!='text'},block=reference,
            original_instruction_sha256=_hash(loop['text'].encode()),
            block_throughput_cycles=summary['BlockRThroughput'],cycles_per_element=summary['BlockRThroughput']/loop['width'],
            runtime_path_observed=False))
    vectors=[r for r in rows if r['width']==2];scalars=[r for r in rows if r['width']==1]
    return dict(loops=rows,vector=max(vectors,key=lambda r:r['cycles_per_element']) if vectors else None,
        scalar=max(scalars,key=lambda r:r['cycles_per_element']) if scalars else None,
        lexer_rule='INSTRUCTION_OPERANDS_EXCLUDE_TRAILING_COMMENTS_SOURCE_MARKS_PRESERVED',
        original_assembly_sha256=_hash(assembly.encode()),static_cache_hits=hits)


def _numeric_model(profile):
    return dict(module=_receipt(Path(C.__file__)), derive=profile.derive.__name__,
        predict=profile.predict.__name__, policy=profile.precision_policy)


def _pack_implementation():
    files = ['cpu_batch_instruction_cost.py', 'cpu_batch_instruction_v1.cpp.in', *C._implementation()]
    return [_receipt(_HERE / name) for name in files]


def build_instruction_pack(domain,mode,out,*,cache):
    from scar.optimization.stateful_cpu_model import _domain
    from scar.optimization.stateful_cpu_model import stateful_resource_context
    domain=_domain(domain);profile=C.instruction_cost_profile(mode)
    out=Path(out).resolve();_require(not out.exists(),'fresh kernel pack required')
    candidates=[r for r in domain['candidates'] if (r['worker_count']==1)==(mode=='serial')]
    _require(bool(candidates),'no matching backend mode');context=stateful_resource_context(domain,candidates[0]['id'])
    available = context['caller_affinity_cpu_ids']
    tool = profile.tool_contract(context)
    started = time.perf_counter()
    native = C._compile_probe(context, out, _source(available))
    native['vectorization'] = native.pop('vectorization_remarks')
    analyses = {}
    asm = Path(native['assembly']['path']).read_text()
    for index,case in enumerate(_registry()):
        if case['role']=='DISPATCH':continue
        for group in (1,2):
            if group==2 and case['array_loads']==1:continue
            analyses[str(index*2+group-1)]=_analyze(asm,tool,cache,function=case['id']+'_g'+str(group))
    _save(out/'instruction-analysis.json',dict(schema='scar.cpu-batch-shared-instruction-analysis',schema_version=1,analyses=analyses))
    parent=artifact_reference(Path(domain['base_stage_plan']['uri']).parent/'domain.json',domain['schema'])
    pack=dict(schema=PACK,schema_version=1,mode=mode,domain=parent,resource_template=context,
        available_cpu_ids=available,tool=tool,registry=_registry(),cache_directory=str(Path(cache).resolve()),
        **native,
        instruction_analysis=artifact_reference(out/'instruction-analysis.json','scar.cpu-batch-shared-instruction-analysis'),
        numeric_model=_numeric_model(profile),
        implementation_files=_pack_implementation(),
        build_seconds=time.perf_counter()-started,target_workload_executed=False,target_dag_provided_to_probe=False,
        cpu_mask_is_runtime_measurement_parameter=True,cost_times_shared_between_masks=False,
        maximum_holdout_relative_error=.20)
    _save(out/'pack.json',pack);return dict(pack=artifact_reference(out/'pack.json',PACK),directory=str(out),mode=mode)


def _read_pack(value):
    if isinstance(value,(str,Path)):
        path=Path(value);path=path/'pack.json' if path.is_dir() else path;reference=artifact_reference(path,PACK)
    else:reference=value
    pack=read_artifact(reference);_require(pack['schema_version']==1 and pack['registry']==_registry(),'batch pack registry differs')
    for key in ('source','assembly','library','vectorization'):
        receipt=pack[key];_require(_receipt(Path(receipt['path']))==receipt,'kernel pack receipt differs')
    _require(Path(pack['source']['path']).read_text()==_source(pack['available_cpu_ids']),'shared native source differs')
    _require(pack['implementation_files'] == _pack_implementation(),'batch producer/numeric source differs')
    profile = C.instruction_cost_profile(pack['mode'])
    _require(pack['numeric_model'] == _numeric_model(profile),'frozen numerical model differs')
    return reference,pack


def _handle(pack):
    handle=ctypes.CDLL(pack['library']['path']);function=handle.scar_batch_instruction_probe_v1
    function.argtypes=[ctypes.c_int32,ctypes.POINTER(_Layout),ctypes.POINTER(_Mask),ctypes.c_uint64,ctypes.c_uint64,
        ctypes.c_uint64,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_int32)]
    function.restype=ctypes.c_int32;return handle,function


def sample_instruction_context(pack_value,context,out,*,repetitions=3,warmup=1,batch=4):
    reference,pack=_read_pack(pack_value);out=Path(out).resolve();_require(not out.exists(),'fresh context observations required')
    _require(repetitions==3 and warmup==1 and batch==4,'frozen batch sampling policy is3/1/4')
    tool=_compatible(pack,context);handle,function=_handle(pack);mode=pack['mode']
    profile=C.instruction_cost_profile(mode)
    if profile.parallel:_require(C._loaded_openmp(handle)==tool['openmp_runtime'],'actual shared probe runtime differs')
    cpulist=context['worker_cpu_ids'];cpu_array=(ctypes.c_int32*len(cpulist))(*cpulist);mask=_Mask(len(cpulist),0,cpu_array)
    configs=_configurations(context);out.mkdir(parents=True);started=time.perf_counter();load_before=list(os.getloadavg())
    design=dict(schema='scar.cpu-batch-context-design',schema_version=1,pack=reference,resource_context=context,
        tool=tool,configurations=configs,repetitions=repetitions,warmup=warmup,batch=batch,
        new_context_samples_not_old_mask_times=True,holdout_used_for_fitting=False,holdout_used_for_model_choice=False)
    _save(out/'design.json',design);design_ref=artifact_reference(out/'design.json',design['schema'])
    observations=[];journal=out/'samples.jsonl'
    for row in configs:
        layout=C._probe_layout(row)
        samples=(ctypes.c_double*repetitions)();sink=ctypes.c_uint64();cpus=(ctypes.c_int32*(2+len(cpulist)))()
        status=function(row['kernel_index'],ctypes.byref(layout),ctypes.byref(mask),repetitions,warmup,batch,samples,ctypes.byref(sink),cpus)
        if status!=0:
            _save(out/'partial-observations.json',dict(schema='scar.cpu-batch-partial-observations',schema_version=1,
                frozen_design=design_ref,completed_observations=observations,failed_configuration=row,
                native_status=status,target_workload_executed=False))
            raise ValueError('shared independent primitive FP/CPU/allocation guard: '+str(status))
        _require(list(cpus)[2:]==cpulist,'actual worker placement differs')
        observations.append(dict(**row, **C._sample_summary(samples),
            checksum=sink.value,actual_cpu_ids=list(cpus)[2:],caller_cpu_before=cpus[0],caller_cpu_after=cpus[1]))
        with journal.open('a') as stream:stream.write(json.dumps(observations[-1],sort_keys=True)+'\n')
    _require(sorted(os.sched_getaffinity(0))==context['caller_affinity_cpu_ids'],'caller affinity not restored')
    raw=dict(schema='scar.cpu-batch-context-observations',schema_version=1,frozen_design=design_ref,pack=reference,
        observations=observations,observed_seconds=time.perf_counter()-started,
        host_load=dict(before=load_before,after=list(os.getloadavg()),isolated=False),pid=os.getpid(),
        target_workload_executed=False,target_variants_ranked=False)
    _save(out/'raw.json',raw);analyses=read_artifact(pack['instruction_analysis'])['analyses'];coefficients=profile.derive(observations,analyses)
    model=dict(schema=MODEL,schema_version=1,mode=mode,pack=reference,resource_context=context,tool=tool,frozen_design=design_ref,
        raw_observations=artifact_reference(out/'raw.json',raw['schema']),coefficients=coefficients,
        local_models=C._qualify(observations,analyses,coefficients,profile),precision_policy=profile.precision_policy,
        cache_geometry=C._llc(context)[1],assumptions=profile.assumptions,
        observed_seconds=raw['observed_seconds'],peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        target_workload_executed=False,target_variants_ranked=False,serial_duration_divided_by_workers=False,
        other_mask_observations_used=False,physical_optimality_proven=False,deployment_authorized=False)
    _save(out/'profile.json',model);return dict(profile=artifact_reference(out/'profile.json',MODEL),mode=mode,
        admitted={g:r['admitted'] for g,r in model['local_models'].items()},
        maximum_errors={g:r['maximum_holdout_relative_error'] for g,r in model['local_models'].items()},
        seconds=model['observed_seconds'],peak_rss_kib=model['peak_rss_kib'])


def read_context_model(path):
    reference=artifact_reference(path,MODEL);model=read_artifact(reference);pack_ref,pack=_read_pack(model['pack'])
    _require(model['schema_version']==1 and model['mode']==pack['mode'],'batch context model schema/mode differs')
    tool=_compatible(pack,model['resource_context']);_require(tool==model['tool'],'exact context tool/runtime differs')
    design=read_artifact(model['frozen_design']);raw=read_artifact(model['raw_observations'])
    _require(design['pack']==raw['pack']==pack_ref and raw['frozen_design']==model['frozen_design']
        and design['resource_context']==model['resource_context'] and design['tool']==tool
        and (design['repetitions'],design['warmup'],design['batch'])==(3,1,4), 'raw context scope differs')
    configs=_configurations(model['resource_context'])
    _require(design['configurations']==configs,'complete frozen context design required')
    C._verify_observations(raw['observations'],configs,design['repetitions'])
    cpus=model['resource_context']['worker_cpu_ids']
    _require(all(row['actual_cpu_ids']==cpus for row in raw['observations']),'independent context observations differ')
    analyses=read_artifact(pack['instruction_analysis'])['analyses']
    profile=C.instruction_cost_profile(pack['mode'])
    _require(profile.derive(raw['observations'],analyses)==model['coefficients']
        and C._qualify(raw['observations'],analyses,model['coefficients'],profile)==model['local_models']
        and model['precision_policy']==profile.precision_policy and model['target_workload_executed'] is False
        and model['target_variants_ranked'] is False and model['other_mask_observations_used'] is False,
        'per-mask model coefficient/holdout/independence differs')
    return reference,model,pack_ref,pack,analyses


def project_instruction_context(domain,candidate_id,model_path,out):
    """Statically lower original code; never execute a target implementation."""
    from scar.optimization.stateful_cpu_model import stateful_cpu_demand
    # The batch parent already admitted this immutable domain. This operation
    # only compiles static evidence; final public replay re-admits the domain.
    if isinstance(domain,(str,Path)):domain=read_artifact(artifact_reference(domain,'scar.stateful-cpu-domain'))
    reference,model,pack_ref,pack,calibrated=read_context_model(model_path)
    demand=stateful_cpu_demand(domain,candidate_id)
    _require(demand['resource_context']==model['resource_context'],'target/resource context differs')
    out=Path(out).resolve();_require(not out.exists(),'fresh target instruction projection required');out.mkdir(parents=True)
    region,plan,source,assembly,lowering=C._target_static(domain,candidate_id,demand['resource_context'],out/'static')
    profile=C.instruction_cost_profile(pack['mode'])
    lowering_ref=artifact_reference(out/'static/lowering.json',lowering['schema'])
    order={n.id:i for i,n in enumerate(region.nodes)};components=[];unsupported=[];proposals=[]
    capacity=model['cache_geometry']['reported_capacity_bytes'];stack_max=max(r['stack_memory_references'] for a in calibrated.values() for r in a['loops'])
    for term in demand['terms']:
        if term['component']!='FUSED_SCALAR_DAG_ROW_STAGE':continue
        proposal=None
        try:
            p=term['parameters']
            _require(p['workers']==len(model['resource_context']['worker_cpu_ids']),'exact target worker mask differs')
            _,span,witness=C._projection_stage(plan,term,order,source,capacity,profile)
            analysis=_analyze(assembly,model['tool'],pack['cache_directory'],source_range=span,source_path=lowering['source']['path'])
            cost,qualified=C._projection_cost(term,analysis,model,profile,stack_max)
            proposal=dict(component=term['component'],parameters=p,prediction=cost,point_estimate_ns=cost['duration_ns'],
                selectable=False,maximum_holdout_relative_error=qualified['maximum_holdout_relative_error'])
            _require(qualified['admitted'],'fixed fresh batch holdout maximum relative error '
                +str(qualified['maximum_holdout_relative_error'])+' exceeds frozen0.20 policy')
            component=C._projection_component(term,analysis,cost,qualified,profile,reference,lowering_ref,witness)
            component['source_pack']=pack_ref
            components.append(component)
        except (ValueError,StopIteration) as error:
            unsupported.append(dict(component=term['component'],parameters=term['parameters'],reason=str(error)))
            if proposal is not None:proposals.append(dict(**proposal,rejection_reason=str(error)))
    value=dict(schema='scar.stateful-resource-calibration',schema_version=1,resource_context=demand['resource_context'],
        components=components,unsupported_components=unsupported,proposed_estimates=proposals,source_model=reference,shared_pack=pack_ref,
        static_lowering=lowering_ref,all_fused_components_covered=not unsupported,
        target_dag_compiled_for_static_analysis=True,target_workload_executed=False,target_variants_ranked=False,
        complete_method_cost_measured=False,physical_optimality_proven=False)
    _save(out/'observations.json',value);return artifact_reference(out/'observations.json',value['schema'])


def make_calibration_context(domain,candidate_id):
    from scar.optimization.stateful_cpu_model import stateful_resource_context, stateful_cpu_demand
    from scar.optimization.stateful_cpu_model import stateful_cpu_cost_context
    row=next(r for r in domain['candidates'] if r['id']==candidate_id)
    parent=artifact_reference(Path(domain['base_stage_plan']['uri']).parent/'domain.json',domain['schema'])
    return dict(schema='scar.cpu-batch-calibration-context',schema_version=1,domain=parent,candidate_id=candidate_id,
        resource_context=stateful_resource_context(domain,candidate_id),binding=row['binding'],
        cost_context=stateful_cpu_cost_context(domain,candidate_id),demand=stateful_cpu_demand(domain,candidate_id),
        source_region=domain['region'],base_stage_plan=domain['base_stage_plan'])


def run_context_worker(context_path,fused_pack,aux_pack,out):
    """One candidate per fresh process; all timing precedes static projection."""
    from scar.codegen.cpu_batch_boundary_resource import sample_aux_pack
    from scar.optimization.stateful_cpu_model import _predict_stateful_cpu_model
    context_ref=artifact_reference(context_path,'scar.cpu-batch-calibration-context');context=read_artifact(context_ref)
    domain=read_artifact(context['domain']);_require(make_calibration_context(domain,context['candidate_id'])==context,
        'frozen source/hardware/candidate context differs')
    out=Path(out).resolve();_require(not out.exists(),'fresh batch worker result required');out.mkdir(parents=True)
    start=time.perf_counter();sample=sample_instruction_context(fused_pack,context['resource_context'],out/'fused')
    auxiliary=sample_aux_pack(aux_pack,context,out/'aux',repetitions=3,warmup=1)
    fused_model=read_artifact(sample['profile']);runtime=read_artifact(auxiliary['runtime_witness'])
    if sample['mode']=='parallel':
        _require(runtime['openmp']['actual_loader']==fused_model['tool']['openmp_runtime']
            and runtime['openmp']['compiler_selected']==fused_model['tool']['compiler_openmp_runtime'],
            'fused and auxiliary probes resolved different actual/compiler OpenMP libraries')
    # This fence is a process-phase boundary: no GCC/MCA jobs run during either
    # the fused or auxiliary physical measurements.
    timing_finished=time.perf_counter();fused=project_instruction_context(domain,context['candidate_id'],sample['profile']['uri'],out/'projection')
    model=_predict_stateful_cpu_model(domain,context['candidate_id'],[auxiliary['resource_calibration']['uri'],
        auxiliary['boundary_calibration']['uri'],fused['uri']],out=out/'complete-model')
    result=dict(schema='scar.cpu-batch-context-result',schema_version=1,context=context_ref,candidate_id=context['candidate_id'],
        fused_pack=fused_pack if isinstance(fused_pack,dict) else artifact_reference(fused_pack,PACK),
        aux_pack=aux_pack if isinstance(aux_pack,dict) else artifact_reference(aux_pack,'scar.cpu-batch-boundary-resource-pack'),
        fused_model=sample['profile'],fused_projection=fused,auxiliary=auxiliary,
        complete_model=artifact_reference(out/'complete-model/model.json',model['schema']),
        cost_profile=artifact_reference(model['profile'],'scar.stateful-cpu-cost-profile') if model['profile'] else None,
        complete=model['complete'],unresolved=model['unresolved'],duration_ticks=model['duration_ticks'],
        holdout_maximum_errors=sample['maximum_errors'],observed_calibration_seconds=timing_finished-start,
        total_worker_seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        pid=os.getpid(),fresh_process_per_mask=True,owned_static_jobs_during_timing=False,
        target_workload_executed=False,target_variants_ranked=False,global_optimality_proven=False)
    _save(out/'result.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('build-pack');c.add_argument('--domain',required=True);c.add_argument('--mode',choices=('serial','parallel'),required=True)
    c.add_argument('--out',required=True);c.add_argument('--cache',required=True)
    c=sub.add_parser('worker');c.add_argument('--context',required=True);c.add_argument('--fused-pack',required=True)
    c.add_argument('--aux-pack',required=True);c.add_argument('--out',required=True);a=p.parse_args(argv)
    if a.command=='build-pack':value=build_instruction_pack(a.domain,a.mode,a.out,cache=a.cache)
    else:
        try:result=run_context_worker(a.context,a.fused_pack,a.aux_pack,a.out)
        except Exception as error:
            directory=Path(a.out).resolve();directory.mkdir(parents=True,exist_ok=True)
            _save(directory/'failure.json',dict(schema='scar.cpu-batch-context-failure',schema_version=1,
                context=str(Path(a.context).resolve()),error_type=type(error).__name__,reason=str(error),
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,pid=os.getpid(),
                target_workload_executed=False,unknown_cost_is_zero=False))
            raise
        value={k:result[k] for k in ('candidate_id','complete','duration_ticks','holdout_maximum_errors','total_worker_seconds','peak_rss_kib')}
    print(json.dumps(value,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())

"""Fixed canonical layout diagnostics; never a target variant or cost authority.

Retain allocation offsets and every worker's elapsed time, which the earlier
calibration reduced to a single maximum. Addresses identify one probe call's
temporary allocations only. These exploratory observations are not fresh
holdouts and cannot authorize a transformed program.
"""
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import random
import resource
import statistics
import subprocess
import time

from . import cpu_batch_instruction_cost as B
from . import cpu_parallel_instruction_cost as P
from .cpu_fused_cost import _HERE,_Layout,_receipt,_require,_save
from scar.ir.execution_recipe import artifact_reference,read_artifact

PACK='scar.cpu-layout-diagnostic-pack'


def _registry():
    cases=[dict(id='layout_empty',role='DISPATCH',array_loads=0,graph=[],output=None)]
    for loads in range(1,9):
        case=B.S.V2._case('BALANCED_FANIN',loads,0,seed=801301+loads)
        case.update(id='layout_memory_'+str(loads),role='MEMORY');cases.append(case)
    for family in ('SERIAL_FANIN','BALANCED_FANIN'):
        case=B.S.V2._case(family,3,6,seed=801411)
        case.update(id='layout_interaction_'+family.lower(),role='INTERACTION')
        for node in case['graph']:
            node['inputs']=['s'+str(int(v[1:])%4) if v.startswith('s') else v for v in node['inputs']]
        cases.append(case)
    return cases


def _replace(source,before,after):
    _require(source.count(before)==1,'canonical diagnostic anchor changed')
    return source.replace(before,after)


def _source(available):
    original=(_HERE/'loop_v1.cpp.in').read_text()
    start=original.index('class ThreadAffinity {');affinity=original[start:original.index('\n#endif',start)]
    start=original.index('class FloatingScope {');floating=original[start:original.index('\n};',start)+3]
    functions=[];pointers=[];loads=[]
    for case in _registry():
        for group in (1,2):
            name=case['id']+'_g'+str(group);body=[]
            for i,layout in enumerate(B.S.V2._layouts(case['array_loads'],100,group)):
                r,c=divmod(layout['offset'],layout['stride']);padding=layout['stride']-100
                body.append(f'const double x{i}=a.inputs[{layout["storage_group"]}][({r}+row)*(a.cols+{padding})+{c}+col];')
            for node in case['graph']:
                args=['a.scalars['+v[1:]+']' if v.startswith('s') else v for v in node['inputs']]
                body.append('const double '+node['id']+'=('+args[0]+dict(add='+',sub='-',mul='*')[node['op']]+args[1]+');')
            if case['role']=='DISPATCH':text='#pragma omp barrier\n'
            else:
                body.append('a.output[row*a.cols+col]='+case['output']+';')
                text='#pragma omp for schedule(static)\nfor(int64_t row=0;row<a.rows;++row)\nfor(int64_t col=0;col<a.cols;++col){\n'+'\n'.join(body)+'\n}\n'
            functions.append('__attribute__((noinline)) void '+name+'(const Args &a){\n'+text+'}\n')
            pointers.append('&'+name);loads.append(str(case['array_loads']))
    source=(_HERE/'cpu_batch_instruction_v1.cpp.in').read_text()
    for key,value in dict(CPU_SLOTS=str(max(available)+1),MAX_WORKERS=str(len(available)),
        AFFINITY_CLASS=affinity,FLOATING_CLASS=floating,CANONICAL_FUNCTIONS='\n'.join(functions),
        CANONICAL_POINTERS=', '.join(pointers),CANONICAL_LOADS=', '.join(loads)).items():
        source=source.replace('@'+key+'@',value)
    source=_replace(source,'scar_batch_instruction_probe_v1','scar_layout_diagnostic_probe_v1')
    source=_replace(source,'uint64_t *checksum,int32_t *observed_cpus){',
        'uint64_t *checksum,int32_t *observed_cpus,uint64_t *addresses,double *worker_samples){')
    source=_replace(source,'!layout||!samples||!checksum||!observed_cpus||primitive<0||',
        '!layout||!samples||!checksum||!observed_cpus||!addresses||!worker_samples||primitive<0||')
    source=_replace(source,'for(int i=0;i<8;++i)a.inputs[i]=bases[i].get();',
        'for(int i=0;i<8;++i){a.inputs[i]=bases[i].get();addresses[i]=reinterpret_cast<uintptr_t>(a.inputs[i]);}\n'
        '    addresses[8]=reinterpret_cast<uintptr_t>(a.output);')
    source=_replace(source,'if(iteration>=warmup)samples[iteration-warmup]=*std::max_element(elapsed.begin(),elapsed.end())/batch;',
        'if(iteration>=warmup){\n'
        '                    samples[iteration-warmup]=*std::max_element(elapsed.begin(),elapsed.end())/batch;\n'
        '                    for(int w=0;w<workers;++w)worker_samples[(iteration-warmup)*workers+w]=elapsed[w]/batch;\n'
        '                }')
    return source


def _configurations():
    rows=[]
    for index,case in enumerate(_registry()):
        shapes=((1,1),) if case['role']=='DISPATCH' else ((256,320),(427,587),(512,512),(622,764),(768,832))
        for group in ((1,) if case['array_loads']<2 else (1,2)):
            for r,c in shapes:
                rows.append(dict(id='layout-'+str(len(rows)),case_index=index,kernel_index=index*2+group-1,
                    role=case['role'],array_loads=case['array_loads'],groups=group,rows=r,cols=c,
                    layouts=B.S.V2._layouts(case['array_loads'],c,group)))
    random.Random(801577).shuffle(rows);return rows


def build_layout_probe(context_model,out):
    reference,model,_,_,_=B.read_context_model(context_model)
    context=model['resource_context'];_require(len(context['worker_cpu_ids'])>1,'parallel context required')
    tool=P._tool_contract(context);compiler=context['backend']['compiler'];out=Path(out).resolve()
    _require(not out.exists(),'fresh diagnostic pack required');out.mkdir(parents=True)
    source=out/'canonical.cpp';source.write_text(_source(context['caller_affinity_cpu_ids']))
    (out/'scar_abi_v1.h').write_bytes((_HERE/'scar_abi_v1.h').read_bytes())
    assembly=out/'canonical.s';library=out/'canonical.so'
    commands=[[*compiler['argv'],*compiler['options'],'-S','-fverbose-asm',str(source),'-o',str(assembly)],
        [*compiler['argv'],*compiler['options'],str(assembly),'-o',str(library)]]
    for i,command in enumerate(commands):
        p=subprocess.run(command,capture_output=True,text=True);(out/f'compile-{i}.log').write_text(p.stdout+p.stderr)
        _require(p.returncode==0,'diagnostic compilation failed: '+p.stderr[-800:])
    pack=dict(schema=PACK,schema_version=1,context_model=reference,resource_context=context,tool=tool,
        registry=_registry(),configurations=_configurations(),source=_receipt(source),assembly=_receipt(assembly),
        library=_receipt(library),compiler_commands=commands,
        implementation=[_receipt(Path(__file__)),_receipt(_HERE/'cpu_batch_instruction_v1.cpp.in'),
            _receipt(_HERE/'loop_v1.cpp.in'),_receipt(_HERE/'scar_abi_v1.h')],
        policy=dict(repetitions=3,warmup=1,batch=4,configuration_order='FIXED_SEED_801577',
            worker_times_and_addresses_recorded_outside_timing=True),
        scope='EXPLORATORY_CANONICAL_LAYOUT_AND_INTERACTION_DIAGNOSTICS_NOT_BLIND_QUALIFICATION',
        target_workload_executed=False,target_dag_provided=False,cost_selection_authorized=False)
    _save(out/'pack.json',pack);return artifact_reference(out/'pack.json',PACK)


def _read_pack(value):
    reference=value if isinstance(value,dict) else artifact_reference(value,PACK);pack=read_artifact(reference)
    _require(pack['schema_version']==1 and pack['registry']==_registry()
        and pack['configurations']==_configurations(),'diagnostic design changed')
    for row in (*pack['implementation'],pack['source'],pack['assembly'],pack['library']):
        _require(_receipt(Path(row['path']))==row,'diagnostic source/native bytes changed')
    context=pack['resource_context']
    _require(sorted(os.sched_getaffinity(0))==context['caller_affinity_cpu_ids']
        and all(os.environ.get(k)==v for k,v in context['native_thread_environment'].items())
        and P._tool_contract(context)==pack['tool'],'diagnostic runtime context changed')
    return reference,pack


def sample_layout_probe(pack,out):
    reference,document=_read_pack(pack);out=Path(out).resolve()
    _require(not out.exists(),'fresh diagnostic sample required');out.mkdir(parents=True)
    _save(out/'design.json',dict(schema='scar.cpu-layout-diagnostic-design',schema_version=1,
        pack=reference,policy=document['policy'],configurations=document['configurations']))
    design_ref=artifact_reference(out/'design.json','scar.cpu-layout-diagnostic-design')
    handle=ctypes.CDLL(document['library']['path']);context=document['resource_context'];cpus=context['worker_cpu_ids']
    _require(P._loaded_openmp(handle)==document['tool']['openmp_runtime'],'actual diagnostic OpenMP runtime changed')
    function=handle.scar_layout_diagnostic_probe_v1
    function.argtypes=[ctypes.c_int32,ctypes.POINTER(_Layout),ctypes.POINTER(B._Mask),
        ctypes.c_uint64,ctypes.c_uint64,ctypes.c_uint64,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_double)]
    function.restype=ctypes.c_int32
    cpu_array=(ctypes.c_int32*len(cpus))(*cpus);mask=B._Mask(len(cpus),0,cpu_array)
    records=[];started=time.perf_counter()
    for row in document['configurations']:
        layout=_Layout();layout.rows=row['rows'];layout.cols=row['cols'];layout.loads=row['array_loads']
        for i,value in enumerate(row['layouts']):
            layout.strides[i]=value['stride'];layout.offsets[i]=value['offset'];layout.groups[i]=value['storage_group']
        samples=(ctypes.c_double*3)();checksum=ctypes.c_uint64();observed=(ctypes.c_int32*(2+len(cpus)))()
        addresses=(ctypes.c_uint64*9)();worker_times=(ctypes.c_double*(3*len(cpus)))()
        status=function(row['kernel_index'],ctypes.byref(layout),ctypes.byref(mask),3,1,4,samples,
            ctypes.byref(checksum),observed,addresses,worker_times)
        _require(status==0 and list(observed)[2:]==cpus,'canonical layout/FP/affinity guard failed: '+str(status))
        values=list(samples);workers=[list(worker_times)[i*len(cpus):(i+1)*len(cpus)] for i in range(3)]
        _require(all(math.isfinite(x) and x>0 for x in values)
            and all(max(w)==v for w,v in zip(workers,values)),'per-worker maximum/raw samples differ')
        records.append(dict(**row,samples_ns=values,median_ns=statistics.median(values),
            per_worker_ns=workers,actual_worker_cpu_ids=list(observed)[2:],checksum=checksum.value,
            allocation_addresses=list(addresses),page_offsets=[x%os.sysconf('SC_PAGE_SIZE') if x else None for x in addresses],
            cache_line_offsets_64=[x%64 if x else None for x in addresses],
            allocation_identity_scope='THIS_PROBE_CALL_ONLY_NOT_CROSS_CALL_STORAGE_ID'))
    _read_pack(reference);_require(artifact_reference(out/'design.json',design_ref['schema'])==design_ref,'diagnostic design changed')
    result=dict(schema='scar.cpu-layout-diagnostic-observations',schema_version=1,pack=reference,design=design_ref,
        records=records,seconds=time.perf_counter()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        actual_worker_cpu_ids=cpus,page_size=os.sysconf('SC_PAGE_SIZE'),
        exploratory_diagnostic_only=True,holdout_qualification=False,model_fit_performed=False,
        target_workload_executed=False,target_variants_ranked=False,cost_selection_authorized=False)
    _save(out/'result.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('build');p.add_argument('--context-model',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('sample');p.add_argument('--pack',required=True);p.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    if args.command=='build':print(json.dumps(build_layout_probe(args.context_model,args.out)))
    else:
        r=sample_layout_probe(args.pack,args.out);print(json.dumps(dict(records=len(r['records']),seconds=r['seconds'],
            peak_rss_kib=r['peak_rss_kib'],cost_selection_authorized=False)))
    return 0


if __name__=='__main__':raise SystemExit(main())

"""Exact-mask OpenMP fused-stage model from independent instruction probes.

Worker execution, a once-per-stage barrier/dispatch floor, and contended warm
memory are separate facts. No serial duration is divided by worker count.
The original backend is only statically lowered when projecting a target.
"""
import argparse
import ast
from copy import deepcopy
import ctypes
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time

from . import cpu_instruction_cost as C
from .cpu_fused_cost import _HERE,_Layout,_hash,_llc,_receipt,_require,_save

SCHEMA='scar.cpu-parallel-instruction-cost-model'
_POLICY=dict(maximum_holdout_relative_error=.20,
    coefficient_sources='INDEPENDENT_EXACT_MASK_PARALLEL_MIXED_ALU_MEMORY_AND_BARRIER_PRIMITIVES',
    holdout_used_for_fitting=False,holdout_used_for_model_choice=False,
    composition='ONE_STAGE_BARRIER_FLOOR_PLUS_MAX_WORKER_MCA_AND_SHARED_MEMORY_PROXY',
    authority='ENGINEERING_ADMISSION_NOT_A_THEORETICAL_ERROR_BOUND')
_ASSUMPTIONS=[*C._ASSUMPTIONS,
    'MAX_WORKER_NATIVE_WALL_NOT_SUM_OF_WORKER_TIMES',
    'EFFECTIVE_MODEL_CYCLE_MAPPING_REMEASURED_UNDER_THIS_EXACT_ACTIVE_CPU_MASK',
    'MEMORY_PROXY_REMEASURED_WITH_THIS_SHARED_TEAM_NOT_SERIAL_TIME_DIVIDED_BY_WORKERS',
    'ONE_STAGE_FLOOR_INCLUDES_ITS_SINGLE_COMPLETION_BARRIER_AND_CANONICAL_DISPATCH',
    'TEAM_LIFECYCLE_AND_OTHER_STAGE_BARRIERS_ARE_SEPARATE_EXISTING_COMPONENTS',
    'STATIC_ROW_PARTITION_COUNTS_ARE_NOT_A_PHYSICAL_BANDWIDTH_SCALING_LAW',
    'NO_CPU_MASK_TRANSFER_WITHOUT_MATCHED_INDEPENDENT_CALIBRATION']


def _lex_assembly(assembly):
    """Instruction operands end at #; standalone source-location rows remain."""
    return '\n'.join(line if line.lstrip().startswith(('#','.')) else line.split('#',1)[0]
        for line in assembly.splitlines())+'\n'


def _lexer_evidence(assembly):
    return dict(rule='STRIP_INSTRUCTION_TRAILING_COMMENTS_KEEP_STANDALONE_SOURCE_MARKS_AND_LINE_COUNT',
        original_assembly_sha256=_hash(assembly.encode()),lexed_assembly_sha256=_hash(_lex_assembly(assembly).encode()),
        line_count_preserved=len(assembly.splitlines())==len(_lex_assembly(assembly).splitlines()),
        implementation_sha256=_hash(Path(__file__).read_bytes()))


def _analyze(assembly,tool,out,**kwargs):
    return dict(**C._analyze(_lex_assembly(assembly),tool,out,**kwargs),instruction_lexer=_lexer_evidence(assembly))


def _architecture_identity(original):
    names=('_registry','_configurations','_source','_compile','_derive','_predict','_qualify',
        '_tool_contract','_loaded_openmp','_partition_witnesses','_POLICY','_ASSUMPTIONS')
    def selected(source):
        result={}
        for node in ast.parse(source).body:
            name=node.name if isinstance(node,ast.FunctionDef) else (
                node.targets[0].id if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) else None)
            if name in names:result[name]=ast.dump(node,include_attributes=False)
        return result
    before=selected(original);after=selected(Path(__file__).read_text())
    _require(set(before)==set(names) and before==after,
        'primitive registry/harness/numerical model/policy changed beyond the lexical repair')
    return _hash(json.dumps(before,sort_keys=True).encode())


def _registry():
    rows=[deepcopy(r) for r in C._registry() if not r['holdout']]
    rows[0].update(id='parallel_stage_barrier_floor',completion_barriers=1)
    for i,(family,n,extra) in enumerate((('SERIAL_FANIN',5,10),('SERIAL_FANIN',7,6),
        ('BALANCED_FANIN',3,9),('BALANCED_FANIN',5,10),('PARALLEL_BRANCHES',5,10),
        ('PARALLEL_BRANCHES',7,6),('SHARED_BRANCHES',3,8),('SHARED_BRANCHES',5,10))):
        case=C.V2._case(family,n,extra,seed=541+i,holdout=True);pool=3 if i%2==0 else 6
        for node in case['graph']:
            node['inputs']=['s'+str(int(k[1:])%pool) if k.startswith('s') else k for k in node['inputs']]
        case.update(role='FRESH_HOLDOUT',operand_pool=pool);rows.append(case)
    return rows


def _configurations(context):
    rows=[];workers=len(context['worker_cpu_ids'])
    for i,case in enumerate(_registry()):
        shapes=((439,613),(618,752)) if case['holdout'] else ((256,320),(512,512),(768,832))
        groups=(1,2) if case['role'] in ('MEMORY','FRESH_HOLDOUT') else (1,)
        if case['role']=='DISPATCH':shapes=((1,1),)
        for group in groups:
            if group==2 and case['array_loads']==1:continue
            for r,c in shapes:
                q,remainder=divmod(r,workers)
                chunks=[q+(tid<remainder) for tid in range(workers)]
                rows.append(dict(id='parallel-instruction-configuration-'+str(len(rows)),case_index=i,
                    kernel_index=i*2+group-1,groups=group,rows=r,cols=c,workers=workers,
                    elements=max(chunks)*c,array_loads=case['array_loads'],
                    layouts=C.V2._layouts(case['array_loads'],c,group),role=case['role'],holdout=case['holdout'],
                    logical_worker_rows=chunks,partition='GOMP_STATIC_CONTIGUOUS_FIRST_REMAINDER_DECLARED_CONTEXT'))
    random.Random(671933).shuffle(rows);return rows


def _source(context):
    base=C._source(context);head=base[:base.index('__attribute__((noinline)) void instruction_')]
    tail=base[base.index('extern "C" SCAR_EXPORT_V1 int32_t scar_instruction_cost_probe_v1'):]
    functions=[];pointers=[];loads=[]
    for case in _registry():
        for group in (1,2):
            name=case['id']+'_g'+str(group);body=[]
            for i,layout in enumerate(C.V2._layouts(case['array_loads'],100,group)):
                pad=layout['stride']-100;r,c=divmod(layout['offset'],layout['stride'])
                body.append(f'const double x{i}=a.inputs[{layout["storage_group"]}][({r}+row)*(a.cols+{pad})+{c}+col];')
            for node in case['graph']:
                args=['a.scalars['+v[1:]+']' if v.startswith('s') else v for v in node['inputs']]
                body.append('const double '+node['id']+'=('+args[0]+dict(add='+',sub='-',mul='*')[node['op']]+args[1]+');')
            if case['role']=='DISPATCH':text='#pragma omp barrier\n'
            else:
                body.append('a.output[row*a.cols+col]='+case['output']+';')
                text='#pragma omp for schedule(static)\nfor(int64_t row=0;row<a.rows;++row)\nfor(int64_t col=0;col<a.cols;++col){\n'+'\n'.join(body)+'\n}\n'
            functions.append('__attribute__((noinline)) void '+name+'(const Args &a){\n'+text+'}\n')
            pointers.append('&'+name);loads.append(str(case['array_loads']))
    witness='''
extern "C" SCAR_EXPORT_V1 int32_t scar_parallel_partition_witness_v1(
    int64_t rows,int64_t *first,int64_t *counts,int32_t *cpus){
    if(rows<1||rows>1000000||!first||!counts||!cpus)return 1;
    constexpr int workers=sizeof(planned_cpu_ids)/sizeof(int);uint32_t guard=0;
    #pragma omp parallel num_threads(workers) reduction(|:guard)
    {
        const int tid=omp_get_thread_num();ThreadAffinity affinity;
        const bool ok=omp_get_num_threads()==workers&&affinity.bind(planned_cpu_ids[tid]);
        int64_t start=rows,count=0;
        #pragma omp for schedule(static)
        for(int64_t row=0;row<rows;++row){start=std::min(start,row);++count;}
        first[tid]=start;counts[tid]=count;cpus[tid]=sched_getcpu();
        const bool kept=affinity.bound(),restored=affinity.restore();
        guard|=static_cast<uint32_t>(!ok||!kept||!restored);
    }
    return guard?4:0;
}
'''
    return head+'\n'.join(functions)+'\nusing Kernel=void(*)(const Args &);\nconstexpr Kernel kernels[]={'+', '.join(pointers)+'};\nconstexpr int required_loads[]={'+', '.join(loads)+'};\n}\n'+tail+witness


def _tool_contract(context):
    tool=C._tool_contract(context);compiler=context['backend']['compiler']
    path=subprocess.run([*compiler['argv'],'-print-file-name=libgomp.so.1'],check=True,capture_output=True,text=True).stdout.strip()
    library=Path(path).resolve();_require(library.is_file(),'compiler-selected libgomp is unavailable')
    # The Python executable can have a conda RPATH distinct from the shell/GCC
    # linker search. Record both identities; never infer runtime from ldd alone.
    loaded=ctypes.CDLL('libgomp.so.1')
    return dict(**tool,compiler_openmp_runtime=_receipt(library),openmp_runtime=_loaded_openmp(loaded),
        openmp_runtime_contract='ACTUAL_PROCESS_LOADER_SYMBOL_IDENTITY_SEPARATE_FROM_COMPILER_LIBRARY')


def _loaded_openmp(handle):
    class DlInfo(ctypes.Structure):
        _fields_=[('name',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]
    loader=ctypes.CDLL(None);dladdr=loader.dladdr
    dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DlInfo)];dladdr.restype=ctypes.c_int
    result=DlInfo();address=ctypes.cast(handle.omp_get_num_threads,ctypes.c_void_p)
    _require(dladdr(address,ctypes.byref(result))!=0 and result.name,'cannot identify loaded OpenMP runtime')
    return _receipt(Path(result.name.decode()).resolve())


def _partition_witnesses(handle,context):
    workers=len(context['worker_cpu_ids']);function=handle.scar_parallel_partition_witness_v1
    function.argtypes=[ctypes.c_int64,ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int32)]
    function.restype=ctypes.c_int32;result=[]
    for rows in (256,439,512,513,618,768):
        first=(ctypes.c_int64*workers)();counts=(ctypes.c_int64*workers)();cpus=(ctypes.c_int32*workers)()
        status=function(rows,first,counts,cpus);_require(status==0,'independent static partition witness guard')
        q,r=divmod(rows,workers);expected=[q+(tid<r) for tid in range(workers)]
        starts=[tid*q+min(tid,r) for tid in range(workers)]
        _require(list(counts)==expected and list(first)==starts and list(cpus)==context['worker_cpu_ids'],
            'active OpenMP static partition differs from the declared row model')
        result.append(dict(rows=rows,first_rows=list(first),worker_rows=list(counts),actual_cpu_ids=list(cpus),
            largest_rows=max(counts),row_model='GOMP_STATIC_CONTIGUOUS_FIRST_REMAINDER',
            canonical_control_probe=True,target_body_executed=False,timing_measured=False,
            finite_witness_is_universal_partition_proof=False))
    return result


def _compile(context,directory):
    compiler=context['backend']['compiler'];cpus=context['worker_cpu_ids']
    _require(len(cpus)>1 and set(cpus)<=os.sched_getaffinity(0)
        and sorted(os.sched_getaffinity(0))==context['caller_affinity_cpu_ids'],'exact active parallel CPU context required')
    _require('-fopenmp' in compiler['options'],'OpenMP compiler contract required')
    identity=subprocess.run([*compiler['argv'],'--version'],check=True,capture_output=True,text=True).stdout.splitlines()[0]
    _require(identity==compiler['identity'] and 'g++' in identity,'exact GCC required')
    directory.mkdir();source=directory/'canonical.cpp';source.write_text(_source(context))
    (directory/'scar_abi_v1.h').write_bytes((_HERE/'scar_abi_v1.h').read_bytes())
    assembly=directory/'canonical.s';remarks=directory/'vectorization.txt';library=directory/'canonical.so'
    commands=[[*compiler['argv'],*compiler['options'],'-S','-fverbose-asm','-fopt-info-vec-optimized='+str(remarks),str(source),'-o',str(assembly)],
        [*compiler['argv'],*compiler['options'],str(assembly),'-o',str(library)]]
    for index,command in enumerate(commands):
        p=subprocess.run(command,capture_output=True,text=True);(directory/f'compile-{index}.log').write_text(p.stdout+p.stderr)
        _require(p.returncode==0,'parallel instruction probe compile failed: '+p.stderr[-1000:])
    handle=ctypes.CDLL(str(library));function=handle.scar_instruction_cost_probe_v1
    function.argtypes=[ctypes.c_int32,ctypes.POINTER(_Layout),ctypes.c_uint64,ctypes.c_uint64,ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_int32)]
    function.restype=ctypes.c_int32
    return handle,function,dict(compiler=compiler,commands=commands,source=_receipt(source),assembly=_receipt(assembly),
        library=_receipt(library),vectorization_remarks=_receipt(remarks),
        scope='MAX_PINNED_WORKER_WALL_ONE_NATIVE_STAGE_WITH_OWN_IMPLICIT_COMPLETION_BARRIER')


def _derive(observations,analyses):
    coefficients=C._derive(observations,analyses)
    coefficients['barrier_floor_ns']=coefficients.pop('dispatch_ns')
    coefficients['authority']='INDEPENDENT_EXACT_MASK_SHARED_TEAM_INSTRUCTION_MEMORY_AND_STAGE_BARRIER_FACTS'
    coefficients['serial_duration_divided_by_workers']=False
    return coefficients


def _predict(row,analysis,coefficients):
    translated={**coefficients,'dispatch_ns':coefficients['barrier_floor_ns']}
    value=C._predict(row,analysis,translated)
    value['barrier_floor_ns']=value.pop('dispatch_ns');value['composition']=_POLICY['composition']
    value['worker_aggregation']='LARGEST_LOGICAL_STATIC_CHUNK_AND_MAX_WORKER_WALL'
    value['completion_barriers_charged']=1;value['team_lifecycle_charged_here']=False
    return value


def _qualify(observations,analyses,coefficients):
    result={}
    for group in (1,2):
        holdout=[]
        for row in observations:
            if not row['holdout'] or row['groups']!=group:continue
            cost=_predict(row,analyses[str(row['kernel_index'])],coefficients)
            holdout.append(dict(configuration=row['id'],observed_ns=row['median_ns'],prediction=cost,
                relative_error=abs(cost['duration_ns']-row['median_ns'])/row['median_ns']))
        maximum=max(r['relative_error'] for r in holdout)
        result[str(group)]=dict(holdout=holdout,maximum_holdout_relative_error=maximum,
            admitted=maximum<=_POLICY['maximum_holdout_relative_error'],precision_policy=_POLICY)
    return result


def calibrate_parallel_instruction_cpu(resource_context,out,*,repetitions=5,warmup=2,batch=8):
    from scar.ir.execution_recipe import artifact_reference
    context=deepcopy(resource_context);out=Path(out).resolve();_require(not out.exists(),'fresh output required')
    _require(repetitions>=3 and warmup>=1 and batch>=1,'repeated warm samples required')
    _require(len(context['worker_cpu_ids'])>1,'parallel worker context required')
    _require(all(os.environ.get(k)==v for k,v in context['native_thread_environment'].items()),'thread environment differs')
    _,cache=C._llc(context);tool=_tool_contract(context);configs=_configurations(context)
    implementation={name:_hash((_HERE/name).read_bytes()) for name in
        ('cpu_parallel_instruction_cost.py','cpu_instruction_cost.py','cpu_fused_cost_v2.py','cpu_fused_cost_v2.cpp.in',
         'cpu_fused_cost_v3.py','cpu_fused_cost.py','loop_v1.cpp.in','scar_abi_v1.h')}
    out.mkdir(parents=True);start=time.time();before=list(os.getloadavg())
    design=dict(schema='scar.cpu-parallel-instruction-cost-design',schema_version=1,resource_context=context,tool=tool,
        registry=_registry(),configurations=configs,precision_policy=_POLICY,implementation_files=implementation,
        repetitions=repetitions,warmup=warmup,batch=batch,model_frozen_before_measurement=True,
        barrier_accounting='ONE_PRIMITIVE_STAGE_COMPLETION_BARRIER_PLUS_DISPATCH_NOT_TEAM_LIFECYCLE',
        target_workload_executed=False,target_dag_provided_to_probe=False,target_variants_ranked=False,
        serial_cost_values_reused=False,serial_duration_divided_by_workers=False)
    _save(out/'design.json',design);design_ref=artifact_reference(out/'design.json',design['schema'])
    handle,function,native=_compile(context,out/'native')
    _require(_loaded_openmp(handle)==tool['openmp_runtime'],'actual probe OpenMP runtime differs from compiler contract')
    witnesses=_partition_witnesses(handle,context)
    assembly=Path(native['assembly']['path']).read_text();analyses={}
    for row in configs:
        key=str(row['kernel_index'])
        if row['role']=='DISPATCH' or key in analyses:continue
        case=_registry()[row['case_index']]
        analyses[key]=_analyze(assembly,tool,out/'mca'/key,function=case['id']+'_g'+str(row['groups']))
    _save(out/'instruction-analysis.json',dict(schema='scar.cpu-parallel-instruction-analysis',schema_version=1,tool=tool,analyses=analyses))
    observations=[]
    for row in configs:
        layout=_Layout();layout.rows=row['rows'];layout.cols=row['cols'];layout.loads=row['array_loads']
        for i,p in enumerate(row['layouts']):layout.strides[i]=p['stride'];layout.offsets[i]=p['offset'];layout.groups[i]=p['storage_group']
        samples=(ctypes.c_double*repetitions)();sink=ctypes.c_uint64();cpus=(ctypes.c_int32*2)()
        status=function(row['kernel_index'],ctypes.byref(layout),repetitions,warmup,batch,samples,ctypes.byref(sink),cpus)
        _require(status==0,'independent parallel FP/affinity/allocation guard: '+str(status));values=list(samples)
        _require(all(math.isfinite(x) and x>0 for x in values),'positive finite stage samples required')
        observations.append(dict(**row,samples_ns=values,median_ns=statistics.median(values),min_ns=min(values),max_ns=max(values),
            stdev_ns=statistics.stdev(values),checksum=sink.value,caller_cpu_before=cpus[0],caller_cpu_after=cpus[1]))
    raw=dict(schema='scar.cpu-parallel-instruction-cost-observations',schema_version=1,frozen_design=design_ref,native=native,
        partition_witnesses=witnesses,observations=observations)
    _save(out/'raw-observations.json',raw);coefficients=_derive(observations,analyses)
    _require(sorted(os.sched_getaffinity(0))==context['caller_affinity_cpu_ids']
        and all(os.environ.get(k)==v for k,v in context['native_thread_environment'].items())
        and artifact_reference(out/'design.json',design_ref['schema'])==design_ref,'frozen context/design changed')
    model=dict(schema=SCHEMA,schema_version=1,resource_context=context,frozen_design=design_ref,
        raw_observations=artifact_reference(out/'raw-observations.json',raw['schema']),
        instruction_analysis=artifact_reference(out/'instruction-analysis.json','scar.cpu-parallel-instruction-analysis'),
        coefficients=coefficients,local_models=_qualify(observations,analyses,coefficients),tool=tool,native=native,
        partition_witnesses=witnesses,
        implementation_files=implementation,cache_geometry=cache,precision_policy=_POLICY,assumptions=_ASSUMPTIONS,
        calibration_seconds=time.time()-start,host_load=dict(before=before,after=list(os.getloadavg()),isolated=False),
        parallel_boundary=design['barrier_accounting'],target_workload_executed=False,target_variants_ranked=False,
        target_dag_compiled_for_static_analysis=False,serial_cost_values_reused=False,
        physical_optimality_proven=False,deployment_authorized=False)
    _save(out/'profile.json',model);return model


def _verify_model(path):
    from scar.ir.execution_recipe import artifact_reference,read_artifact
    reference=artifact_reference(path,SCHEMA);model=read_artifact(reference)
    design=read_artifact(model['frozen_design']);raw=read_artifact(model['raw_observations']);analysis=read_artifact(model['instruction_analysis'])
    _require(model['schema_version']==1 and model['precision_policy']==design['precision_policy']==_POLICY,'parallel policy differs')
    current={name:_hash((_HERE/name).read_bytes()) for name in model['implementation_files']}
    _require(model['implementation_files']==current,'current qualification implementation differs')
    if 'qualification_design' in model:
        qualification=read_artifact(model['qualification_design'])
        _require(qualification['schema']=='scar.parallel-instruction-requalification-design'
            and qualification['schema_version']==1 and qualification['frozen_source_design']==model['frozen_design']
            and qualification['source_observations']==model['raw_observations']
            and qualification['implementation_files']==current and qualification['precision_policy']==_POLICY,
            'explicit lexer-correction qualification contract differs')
        repair=read_artifact(qualification['correction_plan'])
        _require(repair['schema']=='scar.parallel-instruction-lexer-correction-plan'
            and repair['original_producer']==qualification['original_producer']
            and repair['holdout_timing_or_error_values_inspected_before_correction'] is False
            and repair['registry_or_harness_or_cost_formula_or_precision_policy_changed'] is False
            and repair['repeated_measurements'] is False,'blinded formal-repair provenance differs')
        original=qualification['original_producer']
        _require(_receipt(Path(original['path']))==original
            and original['sha256']==design['implementation_files']['cpu_parallel_instruction_cost.py']
            and {k:v for k,v in design['implementation_files'].items() if k!='cpu_parallel_instruction_cost.py'}
                =={k:v for k,v in current.items() if k!='cpu_parallel_instruction_cost.py'}
            and qualification['native_source_replayed_unchanged'] is True
            and qualification['holdout_timing_or_error_values_inspected_before_correction'] is False,
            'historical producer or nonnumerical correction identity differs')
        _require(_architecture_identity(Path(original['path']).read_text())==qualification['unchanged_architecture_ast_sha256'],
            'mechanical nonnumerical-repair AST identity differs')
    else:_require(current==design['implementation_files'],'physical calibration producer differs')
    _require(_tool_contract(model['resource_context'])==model['tool']==design['tool']==analysis['tool'],'MCA/OpenMP binary/CPU context differs')
    _require(raw['partition_witnesses']==model['partition_witnesses'],'partition evidence differs')
    _require([r['rows'] for r in model['partition_witnesses']]==[256,439,512,513,618,768],
        'complete fixed partition witness design required')
    for row in model['partition_witnesses']:
        workers=len(model['resource_context']['worker_cpu_ids']);q,r=divmod(row['rows'],workers)
        _require(row['worker_rows']==[q+(tid<r) for tid in range(workers)]
            and row['first_rows']==[tid*q+min(tid,r) for tid in range(workers)]
            and row['actual_cpu_ids']==model['resource_context']['worker_cpu_ids']
            and row['largest_rows']==(row['rows']+workers-1)//workers,'retained static partition witness differs')
    _require(design['registry']==_registry() and design['configurations']==_configurations(model['resource_context'])
        and raw['frozen_design']==model['frozen_design'] and raw['native']==model['native'],'fixed parallel design/native differs')
    _require(model['target_workload_executed'] is False and model['target_variants_ranked'] is False
        and model['serial_cost_values_reused'] is False and design['target_dag_provided_to_probe'] is False
        and design['serial_duration_divided_by_workers'] is False,'independent parallel authority differs')
    _require(len(raw['observations'])==len(design['configurations']),'incomplete parallel design')
    for row,config in zip(raw['observations'],design['configurations']):
        _require(all(row[k]==v for k,v in config.items()) and len(row['samples_ns'])==design['repetitions']
            and all(math.isfinite(x) and x>0 for x in row['samples_ns'])
            and statistics.median(row['samples_ns'])==row['median_ns'],'parallel sample/design differs')
    _require(Path(model['native']['source']['path']).read_text()==_source(model['resource_context']),'canonical source differs')
    for receipt in model['native'].values():
        if type(receipt) is dict and {'path','sha256','bytes'}<=set(receipt):
            _require(_receipt(Path(receipt['path']))==receipt,'native compiler/source receipt differs')
    handle=ctypes.CDLL(model['native']['library']['path'])
    _require(_loaded_openmp(handle)==model['tool']['openmp_runtime'],'current probe OpenMP runtime differs')
    assembly=Path(model['native']['assembly']['path']).read_text();analyses=analysis['analyses']
    for key,value in analyses.items():
        case=_registry()[int(key)//2];group=int(key)%2+1
        _require(value['instruction_lexer']==_lexer_evidence(assembly),'instruction lexical provenance differs')
        loops=C._natural_loops(_lex_assembly(assembly),function=case['id']+'_g'+str(group))
        _require(len(loops)==len(value['loops']),'parallel natural loop set differs')
        for original,retained in zip(loops,value['loops']):
            _require(_receipt(Path(retained['loop_source']['path']))==retained['loop_source']
                and Path(retained['loop_source']['path']).read_text()==original['text']
                and _receipt(Path(retained['result']['path']))==retained['result'],'MCA input/output differs')
            result=json.loads(Path(retained['result']['path']).read_text());summary=result['CodeRegions'][0]['SummaryView']
            _require(retained['cycles_per_element']==summary['BlockRThroughput']/original['width']
                and retained['block_throughput_cycles']==summary['BlockRThroughput']
                and result['TargetInfo']['CPUName']==model['tool']['cpu_model'],'MCA numerical summary differs')
    _require(_derive(raw['observations'],analyses)==model['coefficients']
        and _qualify(raw['observations'],analyses,model['coefficients'])==model['local_models'],'parallel model/holdout differs')
    _require(_llc(model['resource_context'])[1]==model['cache_geometry'],'cache evidence differs')
    return reference,model,raw,analyses


def requalify_parallel_instruction_cpu(source_directory,out,*,correction_plan):
    """Reparse fixed native instructions; preserve all physical observations."""
    from scar.ir.execution_recipe import artifact_reference,read_artifact
    source_directory=Path(source_directory).resolve();out=Path(out).resolve()
    _require(not out.exists(),'fresh requalification output required')
    source_design=artifact_reference(source_directory/'design.json','scar.cpu-parallel-instruction-cost-design')
    source_raw=artifact_reference(source_directory/'raw-observations.json','scar.cpu-parallel-instruction-cost-observations')
    design=read_artifact(source_design);raw=read_artifact(source_raw)
    correction_ref=artifact_reference(correction_plan,'scar.parallel-instruction-lexer-correction-plan');repair=read_artifact(correction_ref)
    _require(repair['holdout_timing_or_error_values_inspected_before_correction'] is False
        and repair['registry_or_harness_or_cost_formula_or_precision_policy_changed'] is False
        and repair['repeated_measurements'] is False,'nonnumerical blind correction contract required')
    original=repair['original_producer'];_require(_receipt(Path(original['path']))==original
        and original['sha256']==design['implementation_files']['cpu_parallel_instruction_cost.py'],
        'original physical-observation producer differs')
    _require(_receipt(source_directory/'raw-observations.json')==repair['raw_observations'],
        'fixed original physical observations differ')
    architecture=_architecture_identity(Path(original['path']).read_text())
    _require(design['registry']==_registry() and design['configurations']==_configurations(design['resource_context'])
        and design['precision_policy']==_POLICY and raw['frozen_design']==source_design,'original fixed design differs')
    native=raw['native'];_require(Path(native['source']['path']).read_text()==_source(design['resource_context']),
        'physical canonical native source changed during lexical repair')
    for receipt in native.values():
        if type(receipt) is dict and {'path','sha256','bytes'}<=set(receipt):
            _require(_receipt(Path(receipt['path']))==receipt,'original native/source receipt differs')
    tool=_tool_contract(design['resource_context']);_require(tool==design['tool'],'physical runtime/tool context changed')
    current={name:_hash((_HERE/name).read_bytes()) for name in design['implementation_files']}
    _require({k:v for k,v in current.items() if k!='cpu_parallel_instruction_cost.py'}=={
        k:v for k,v in design['implementation_files'].items() if k!='cpu_parallel_instruction_cost.py'},'shared implementation changed')
    out.mkdir(parents=True);started=time.time()
    qualification=dict(schema='scar.parallel-instruction-requalification-design',schema_version=1,
        frozen_source_design=source_design,source_observations=source_raw,original_producer=original,
        correction_plan=correction_ref,implementation_files=current,precision_policy=_POLICY,
        unchanged_architecture_ast_sha256=architecture,native_source_replayed_unchanged=True,
        lexical_rule=_lexer_evidence(Path(native['assembly']['path']).read_text())['rule'],
        holdout_timing_or_error_values_inspected_before_correction=False,
        numerical_model_or_precision_policy_changed=False,new_physical_measurements_collected=False)
    # Freeze the explicit parser amendment before deriving any qualification
    # error from the retained raw measurements.
    _save(out/'qualification-design.json',qualification)
    qualification_ref=artifact_reference(out/'qualification-design.json',qualification['schema'])
    assembly=Path(native['assembly']['path']).read_text();analyses={}
    for row in design['configurations']:
        key=str(row['kernel_index'])
        if row['role']=='DISPATCH' or key in analyses:continue
        case=_registry()[row['case_index']]
        analyses[key]=_analyze(assembly,tool,out/'mca'/key,function=case['id']+'_g'+str(row['groups']))
    _save(out/'instruction-analysis.json',dict(schema='scar.cpu-parallel-instruction-analysis',schema_version=1,tool=tool,analyses=analyses))
    coefficients=_derive(raw['observations'],analyses)
    model=dict(schema=SCHEMA,schema_version=1,resource_context=design['resource_context'],frozen_design=source_design,
        qualification_design=qualification_ref,raw_observations=source_raw,
        instruction_analysis=artifact_reference(out/'instruction-analysis.json','scar.cpu-parallel-instruction-analysis'),
        coefficients=coefficients,local_models=_qualify(raw['observations'],analyses,coefficients),tool=tool,native=native,
        partition_witnesses=raw['partition_witnesses'],implementation_files=current,cache_geometry=_llc(design['resource_context'])[1],
        precision_policy=_POLICY,assumptions=_ASSUMPTIONS,calibration_seconds=None,
        qualification_replay_seconds=time.time()-started,new_physical_measurements_collected=False,
        parallel_boundary=design['barrier_accounting'],target_workload_executed=False,target_variants_ranked=False,
        target_dag_compiled_for_static_analysis=False,serial_cost_values_reused=False,
        physical_optimality_proven=False,deployment_authorized=False)
    _save(out/'profile.json',model);return model


def predict_parallel_instruction_components(domain,candidate_id,profile,out):
    from scar.optimization.stateful_cpu_choices import _domain
    from scar.optimization.stateful_cpu_model import stateful_cpu_demand
    from scar.ir.execution_recipe import artifact_reference
    domain=_domain(domain);demand=stateful_cpu_demand(domain,candidate_id);reference,model,raw,calibrated=_verify_model(profile)
    _require(demand['resource_context']==model['resource_context'],'exact parallel resource/runtime/compiler/mask differs')
    out=Path(out).resolve();_require(not out.exists(),'fresh projection required');out.mkdir(parents=True)
    region,plan,source,assembly,lowering=C._target_static(domain,candidate_id,demand['resource_context'],out/'static',model['tool'])
    order={n.id:i for i,n in enumerate(region.nodes)};components=[];unsupported=[];proposals=[]
    capacity=model['cache_geometry']['reported_capacity_bytes'];maximum_stack=max(r['stack_memory_references'] for a in calibrated.values() for r in a['loops'])
    for term in demand['terms']:
        if term['component']!='FUSED_SCALAR_DAG_ROW_STAGE':continue
        proposal=None
        try:
            p=term['parameters'];stage_index=int(term['source_ref']['pointer'].rsplit('/',1)[1]);stage=plan['stages'][stage_index]
            _require(stage['kind']=='COMPUTE' and p['workers']>1,'unsupported parallel stage boundary')
            output=plan['assignment'][stage['output_payload']]['slot_id']
            inputs=[plan['assignment'][key]['slot_id'] for key in stage['input_payloads'] if key in plan['assignment']]
            _require(output not in inputs,'live workspace output/input alias')
            expected=C.V2._layouts(p['array_loads'],p['cols'],p['backing_storage_groups']);sort=lambda r:(r['storage_group'],r['stride'],r['offset'])
            _require(sorted(p['operand_layouts'],key=sort)==sorted(expected,key=sort),'uncalibrated shared memory geometry')
            _,geometry=C.V2.fused_features(p,capacity)
            _require(256<=p['rows']<=768 and 320<=p['cols']<=832
                and geometry['backing_bytes']+p['rows']*p['cols']*8<=capacity,'outside measured shape/reported cache extent scope')
            span=C._stage_source_range(source,order[stage['root_expr']])
            analysis=_analyze(assembly,model['tool'],out/'mca'/str(stage_index),source_range=span,source_path=lowering['source']['path'])
            _require(max(r['stack_memory_references'] for r in analysis['loops'])<=maximum_stack,'uncalibrated worker stack pressure')
            cost=_predict(dict(groups=p['backing_storage_groups'],rows=p['rows'],cols=p['cols'],workers=p['workers'],
                elements=((p['rows']+p['workers']-1)//p['workers'])*p['cols'],array_loads=p['array_loads']),analysis,model['coefficients'])
            qualified=model['local_models'][str(p['backing_storage_groups'])]
            proposal=dict(component=term['component'],parameters=p,prediction=cost,
                point_estimate_ns=cost['duration_ns'],maximum_holdout_relative_error=qualified['maximum_holdout_relative_error'],
                precision_policy=_POLICY,target_dag_compiled_for_static_analysis=True,measured_target_component=False,selectable=False)
            _require(qualified['admitted'],'fresh parallel holdout maximum relative error '+str(qualified['maximum_holdout_relative_error'])+' exceeds frozen0.20 policy')
            components.append(dict(component=term['component'],parameters=p,basis='DECLARED_COMPONENT_MODEL',median_ns=cost['duration_ns'],
                prediction=cost,source_model=dict(artifact=reference,locator='#/coefficients'),
                source_validation=dict(artifact=reference,locator='#/local_models/'+str(p['backing_storage_groups'])+'/holdout'),
                source_provenance=term['source_ref'],instruction_analysis=analysis,
                static_lowering=artifact_reference(out/'static/lowering.json',lowering['schema']),
                workspace_alias_authority=dict(output_slot=output,input_slots=inputs,live_slots_distinct=True),
                precision_policy=_POLICY,maximum_holdout_relative_error=qualified['maximum_holdout_relative_error'],
                target_dag_compiled_for_static_analysis=True,measured_target_component=False,
                physical_optimality_proven=False,assumptions=_ASSUMPTIONS))
        except (ValueError,StopIteration) as error:
            unsupported.append(dict(component=term['component'],parameters=term['parameters'],reason=str(error)))
            if proposal is not None:proposals.append(dict(**proposal,rejection_reason=str(error)))
    result=dict(schema='scar.stateful-resource-calibration',schema_version=1,resource_context=demand['resource_context'],
        components=components,unsupported_components=unsupported,proposed_estimates=proposals,source_model=reference,
        static_lowering=artifact_reference(out/'static/lowering.json',lowering['schema']),all_fused_components_covered=not unsupported,
        complete_method_cost_measured=False,target_dag_compiled_for_static_analysis=True,
        target_workload_executed=False,target_variants_ranked=False,physical_optimality_proven=False)
    _save(out/'observations.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('calibrate');c.add_argument('--context',required=True);c.add_argument('--out',required=True)
    c.add_argument('--repetitions',type=int,default=5);c.add_argument('--warmup',type=int,default=2);c.add_argument('--batch',type=int,default=8)
    c=sub.add_parser('project');c.add_argument('--domain',required=True);c.add_argument('--candidate-id',required=True)
    c.add_argument('--profile',required=True);c.add_argument('--out',required=True);a=p.parse_args(argv)
    if a.command=='project':
        r=predict_parallel_instruction_components(a.domain,a.candidate_id,a.profile,a.out)
        print(json.dumps(dict(components=len(r['components']),unsupported=r['unsupported_components']),sort_keys=True));return 0
    model=calibrate_parallel_instruction_cpu(json.loads(Path(a.context).read_text()),a.out,
        repetitions=a.repetitions,warmup=a.warmup,batch=a.batch)
    print(json.dumps(dict(profile=str(Path(a.out).resolve()/'profile.json'),local_models={k:dict(admitted=v['admitted'],
        maximum_holdout_relative_error=v['maximum_holdout_relative_error']) for k,v in model['local_models'].items()}),sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())

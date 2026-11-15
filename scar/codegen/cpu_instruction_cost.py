"""Independent CPU stage costs using compiler instruction/resource structure.

LLVM-MCA supplies a versioned instruction scheduling proxy, not cache or DRAM
timings. Independent canonical measurements supply an effective clock mapping
and a separate warm memory/fanin proxy. Target code is only statically lowered
for projection; neither calibration nor projection executes a target variant.
"""
import argparse
from collections import Counter
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

from . import cpu_fused_cost_v2 as V2
from .cpu_fused_cost_v3 import _interpolate
from .cpu_fused_cost import _HERE,_Layout,_hash,_json,_llc,_receipt,_require,_save

_MCA=Path('/opt/amd/aocc/5.1/bin/llvm-mca')
_POLICY=dict(maximum_holdout_relative_error=.20,
    coefficient_sources='FIXED_NON_TARGET_MIXED_ARITHMETIC_AND_MEMORY_PRIMITIVES',
    holdout_used_for_fitting=False,holdout_used_for_model_choice=False,
    composition='DISPATCH_PLUS_MAX_MCA_EXECUTION_AND_MEMORY_PROXY',
    authority='ENGINEERING_ADMISSION_NOT_A_THEORETICAL_ERROR_BOUND')
_ASSUMPTIONS=[
    'MCA_BROADWELL_STATIC_INSTRUCTION_RESOURCE_MODEL_NOT_HARDWARE_EXECUTION_OBSERVATION',
    'EFFECTIVE_NS_PER_MCA_CYCLE_IS_CALIBRATED_NOT_ACTUAL_CLOCK_FREQUENCY',
    'MCA_DOES_NOT_MODEL_CACHE_MISSES_OR_DRAM_BANDWIDTH',
    'WARM_MEMORY_FANIN_PROXY_IS_SEPARATE_NOT_A_DRAM_TRAFFIC_MEASUREMENT',
    'STACK_ACCESSES_ARE_INCLUDED_IN_MCA_WITH_L1_MODEL_NOT_A_SPILL_LATENCY_GUARANTEE',
    'SOURCE_OUTPUT_ALLOCATION_DISTINCT_FROM_READONLY_INPUTS_AUTHORIZES_NOALIAS_MODEL',
    'ROW_SETUP_AND_SCALAR_TAIL_USE_DECLARED_STATIC_LOOP_PROXY',
    'STATIC_VECTOR_PATH_NOT_A_SAMPLED_RUNTIME_BRANCH',
    'CANONICAL_NUMA_AND_CACHE_RESIDENCY_NOT_RESERVED',
    'FRESH_HOLDOUT_PRECISION_NOT_A_BOUND_ON_TARGET_ERROR',
    'COMPONENT_ADDITIVITY_AND_WHOLE_PROGRAM_PROFITABILITY_REQUIRE_SEPARATE_VALIDATION']


def _registry():
    rows=[dict(id='instruction_empty',role='DISPATCH',array_loads=0,graph=[],output=None,holdout=False)]
    for n in (1,2,4,6,8):
        case=V2._case('BALANCED_FANIN',n,0);case.update(id='instruction_memory_'+str(n),role='MEMORY')
        rows.append(case)
    patterns=(('ADD',('add',)),('SUB',('sub',)),('MUL',('mul',)),
        ('ADD_MUL',('add','mul')),('ADD_ADD_MUL',('add','add','mul')),
        ('ADD_MUL_MUL',('add','mul','mul')),('ADD_SUB',('add','sub')))
    for name,pattern in patterns:
        for chains,pool in ((1,4),(4,4),(4,64)):
            graph=[];values=[]
            for chain in range(chains):
                value='x0'
                for step in range(16):
                    key='v'+str(len(graph));scalar='s'+str((chain*16+step) if pool==64 else (chain+step)%pool)
                    graph.append(dict(id=key,op=pattern[step%len(pattern)],inputs=[value,scalar]));value=key
                values.append(value)
            while len(values)>1:
                key='v'+str(len(graph));graph.append(dict(id=key,op='add',inputs=values[:2]));values=[key,*values[2:]]
            rows.append(dict(id=f'instruction_math_{name}_{chains}_{pool}',role='MATH',array_loads=1,
                graph=graph,output=values[0],operand_pool=pool,holdout=False))
    for i,(family,n,extra) in enumerate((('SERIAL_FANIN',3,8),('SERIAL_FANIN',7,10),
        ('BALANCED_FANIN',5,6),('BALANCED_FANIN',7,8),('PARALLEL_BRANCHES',3,10),
        ('PARALLEL_BRANCHES',5,8),('SHARED_BRANCHES',5,6),('SHARED_BRANCHES',7,10))):
        case=V2._case(family,n,extra,seed=281+i,holdout=True)
        if i%2==0:
            for node in case['graph']:
                node['inputs']=['s'+str(int(k[1:])%4) if k.startswith('s') else k for k in node['inputs']]
        case.update(role='FRESH_HOLDOUT',operand_pool=4 if i%2==0 else extra);rows.append(case)
    return rows


def _configurations(context):
    result=[];workers=len(context['worker_cpu_ids'])
    for i,case in enumerate(_registry()):
        shapes=((432,560),(593,721)) if case['holdout'] else ((256,320),(512,512),(768,832))
        groups=(1,2) if case['role'] in ('MEMORY','FRESH_HOLDOUT') else (1,)
        if case['role']=='DISPATCH':shapes=((1,1),)
        for group in groups:
            if group==2 and case['array_loads']==1:continue
            for r,c in shapes:
                result.append(dict(id='instruction-configuration-'+str(len(result)),case_index=i,
                    kernel_index=i*2+group-1,groups=group,rows=r,cols=c,workers=workers,
                    elements=((r+workers-1)//workers)*c,array_loads=case['array_loads'],
                    layouts=V2._layouts(case['array_loads'],c,group),role=case['role'],holdout=case['holdout']))
    random.Random(391127).shuffle(result);return result


def _source(context):
    base=V2._source(context)
    head=base[:base.index('__attribute__((noinline)) void registry_')]
    head=head.replace('#include <omp.h>','#ifdef _OPENMP\n#include <omp.h>\n#endif')
    head=head.replace('double scalars[16]','double scalars[128]')
    head+='\nint probe_tid(){\n#ifdef _OPENMP\nreturn omp_get_thread_num();\n#else\nreturn 0;\n#endif\n}\n'
    head+='int probe_team(){\n#ifdef _OPENMP\nreturn omp_get_num_threads();\n#else\nreturn 1;\n#endif\n}\n'
    tail=base[base.index('extern "C" SCAR_EXPORT_V1 int32_t scar_fused_cost_probe_v2'):]
    tail=tail.replace('scar_fused_cost_probe_v2','scar_instruction_cost_probe_v1')
    tail=tail.replace('omp_get_thread_num()','probe_tid()').replace('omp_get_num_threads()','probe_team()')
    tail=tail.replace('i<16;++i)a.scalars','i<128;++i)a.scalars').replace('double[l.rows*l.cols]);','double[l.rows*l.cols]());')
    functions=[];pointers=[];loads=[]
    for case in _registry():
        for group in (1,2):
            name=case['id']+'_g'+str(group);body=[]
            for i,layout in enumerate(V2._layouts(case['array_loads'],100,group)):
                pad=layout['stride']-100;r,c=divmod(layout['offset'],layout['stride'])
                body.append(f'const double x{i}=a.inputs[{layout["storage_group"]}][({r}+row)*(a.cols+{pad})+{c}+col];')
            for node in case['graph']:
                args=['a.scalars['+v[1:]+']' if v.startswith('s') else v for v in node['inputs']]
                body.append('const double '+node['id']+'=('+args[0]+dict(add='+',sub='-',mul='*')[node['op']]+args[1]+');')
            if case['role']=='DISPATCH':text='asm volatile("" : : "g"(&a) : "memory");\n'
            else:
                body.append('a.output[row*a.cols+col]='+case['output']+';')
                text='#pragma omp for schedule(static)\nfor(int64_t row=0;row<a.rows;++row)\nfor(int64_t col=0;col<a.cols;++col){\n'+'\n'.join(body)+'\n}\n'
            functions.append('__attribute__((noinline)) void '+name+'(const Args &a){\n'+text+'}\n')
            pointers.append('&'+name);loads.append(str(case['array_loads']))
    return head+'\n'.join(functions)+'\nusing Kernel=void(*)(const Args &);\nconstexpr Kernel kernels[]={'+', '.join(pointers)+'};\nconstexpr int required_loads[]={'+', '.join(loads)+'};\n}\n'+tail


def _tool_contract(context):
    _require(_MCA.is_file(),'local LLVM-MCA unavailable')
    version=subprocess.run([str(_MCA),'--version'],check=True,capture_output=True,text=True).stdout
    info=Path('/proc/cpuinfo').read_text().split('\n\n')
    cpus={}
    for block in info:
        fields=dict(l.split(':',1) for l in block.splitlines() if ':' in l)
        fields={k.strip():v.strip() for k,v in fields.items()}
        if 'processor' in fields:cpus[int(fields['processor'])]=fields
    selected=[cpus[c] for c in context['worker_cpu_ids']]
    _require('Host CPU: broadwell' in version and all(r.get('vendor_id')=='GenuineIntel'
        and r.get('cpu family')=='6' and r.get('model')=='79' for r in selected),
        'this instruction contract requires Broadwell CPU model79 and its exact MCA model')
    return dict(executable=_receipt(_MCA),version=version,cpu_model='broadwell',iterations=1000,
        noalias=True,noalias_authority='FRESH_DISTINCT_OUTPUT_AND_READONLY_INPUT_ALLOCATIONS',
        cpu_identity=[{k:r.get(k) for k in ('processor','vendor_id','cpu family','model','stepping','model name')} for r in selected],
        model_is_actual_hardware_execution=False,cache_model_included=False)


def _compile(context,directory):
    compiler=context['backend']['compiler'];cpus=context['worker_cpu_ids']
    _require(bool(cpus) and set(cpus)<=os.sched_getaffinity(0)
        and sorted(os.sched_getaffinity(0))==context['caller_affinity_cpu_ids'],'exact caller and worker affinity required')
    _require(('-fopenmp' in compiler['options'])==(len(cpus)>1),'serial/parallel compiler contract differs')
    identity=subprocess.run([*compiler['argv'],'--version'],check=True,capture_output=True,text=True).stdout.splitlines()[0]
    _require(identity==compiler['identity'] and 'g++' in identity,'exact GCC required')
    directory.mkdir();source=directory/'canonical.cpp';source.write_text(_source(context))
    (directory/'scar_abi_v1.h').write_bytes((_HERE/'scar_abi_v1.h').read_bytes())
    assembly=directory/'canonical.s';remarks=directory/'vectorization.txt';library=directory/'canonical.so'
    commands=[[*compiler['argv'],*compiler['options'],'-S','-fverbose-asm','-fopt-info-vec-optimized='+str(remarks),str(source),'-o',str(assembly)],
        [*compiler['argv'],*compiler['options'],str(assembly),'-o',str(library)]]
    for index,command in enumerate(commands):
        p=subprocess.run(command,capture_output=True,text=True);(directory/f'compile-{index}.log').write_text(p.stdout+p.stderr)
        _require(p.returncode==0,'independent instruction probe compilation failed: '+p.stderr[-1000:])
    handle=ctypes.CDLL(str(library));function=handle.scar_instruction_cost_probe_v1
    function.argtypes=[ctypes.c_int32,ctypes.POINTER(_Layout),ctypes.c_uint64,ctypes.c_uint64,ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_int32)]
    function.restype=ctypes.c_int32
    return handle,function,dict(compiler=compiler,commands=commands,source=_receipt(source),assembly=_receipt(assembly),
        library=_receipt(library),vectorization_remarks=_receipt(remarks),
        scope='SERIAL_NATIVE_STAGE_NO_OPENMP' if len(cpus)==1 else 'PINNED_TEAM_STAGE_WITH_OWN_BARRIER')


def _natural_loops(assembly,function=None,source_range=None,source_path=None):
    lines=assembly.splitlines();start=0;end=len(lines)
    if function is not None:
        start=next(i for i,l in enumerate(lines) if l.startswith('_ZN') and function in l and l.endswith(':'))
        end=next(i for i in range(start+1,len(lines)) if lines[i].startswith('\t.size'))
    part=lines[start:end];labels={l.strip()[:-1]:i for i,l in enumerate(part) if re.match(r'^\.L\d+:$',l.strip())}
    source_lines=[];explicit_sources=[];current=None
    for line in part:
        m=re.match(r'# (.+?):(\d+):',line)
        matched=bool(m and (source_path is None or Path(m[1]).resolve()==Path(source_path).resolve()))
        if matched:current=int(m[2])
        source_lines.append(current)
        explicit_sources.append(current if matched else None)
    loops=[]
    for j,line in enumerate(part):
        match=re.match(r'\s*j\w+\s+(\.L\d+)',line)
        if not match or match[1] not in labels or labels[match[1]]>=j:continue
        a=labels[match[1]];body=part[a:j+1];store_widths=[]
        for text in body:
            m=re.match(r'\s*(movups|movupd|movaps|movapd|movsd)\s+%xmm\d+,\s*(.+)',text)
            if m and '(' in m[2] and '%rsp' not in m[2]:
                store_widths.append(1 if m[1]=='movsd' else 2)
        if not store_widths:continue
        # Out-of-line fallback labels inherit unrelated preceding debug lines.
        # Actual source marks inside the natural loop take precedence over that
        # carried label attribution; header lines never become program lines.
        sources={n for n in explicit_sources[a:j+1] if n is not None}
        if not sources:sources={n for n in source_lines[a:j+1] if n is not None}
        if source_range and (not sources or not all(source_range[0]<=n<=source_range[1] for n in sources)):continue
        instructions='\n'.join(l.split('#',1)[0] for l in body if not l.lstrip().startswith(('.p2align','.cfi')))+'\n'
        loops.append(dict(start=start+a+1,end=start+j+1,width=max(store_widths),text=instructions,
            source_lines=sorted(sources),stack_memory_references=len(re.findall(r'\(%rsp\)',instructions))))
    # Remove outer row/control loops containing any smaller store-bearing loop.
    return [loop for loop in loops if not any(other['start']>loop['start'] and other['end']<loop['end'] for other in loops)]


def _analyze(assembly,tool,out,*,function=None,source_range=None,source_path=None):
    loops=_natural_loops(assembly,function,source_range,source_path);out.mkdir(parents=True)
    _require(bool(loops),'no supported natural store loop in compiler output')
    results=[]
    for index,loop in enumerate(loops):
        path=out/f'loop-{index}.s';path.write_text(loop['text'])
        _require(not re.search(r'\bcall\w*\b',loop['text']),'opaque call in stage loop')
        command=[tool['executable']['path'],'-mcpu='+tool['cpu_model'],'-iterations='+str(tool['iterations']),
            '--noalias=true','--json',str(path)]
        p=subprocess.run(command,capture_output=True,text=True);(out/f'loop-{index}.stderr').write_text(p.stderr)
        _require(p.returncode==0,'MCA rejected compiler instruction loop: '+p.stderr[-500:])
        result=json.loads(p.stdout);(out/f'loop-{index}.mca.json').write_text(p.stdout)
        region=result['CodeRegions'][0];summary=region['SummaryView']
        _require(result['TargetInfo']['CPUName']==tool['cpu_model'],'MCA CPU differs')
        _require(not any(i['hasUnmodeledSideEffects'] for i in region['InstructionInfoView']['InstructionList']),
            'instruction with unmodeled side effects')
        results.append(dict(**{k:v for k,v in loop.items() if k!='text'},loop_source=_receipt(path),
            result=_receipt(out/f'loop-{index}.mca.json'),command=command,
            block_throughput_cycles=summary['BlockRThroughput'],
            simulated_cycles_per_iteration=summary['TotalCycles']/summary['Iterations'],
            cycles_per_element=summary['BlockRThroughput']/loop['width'],
            resource_pressure=region['ResourcePressureView'],runtime_path_observed=False))
    vectors=[r for r in results if r['width']==2];scalars=[r for r in results if r['width']==1]
    _require(bool(vectors) or bool(scalars),'supported instruction width missing')
    return dict(loops=results,vector=max(vectors,key=lambda r:r['cycles_per_element']) if vectors else None,
        scalar=max(scalars,key=lambda r:r['cycles_per_element']) if scalars else None,
        selection='MAX_STATIC_VECTOR_PATH_WHEN_NONALIAS_WORKSPACE_GUARD_HOLDS')


def _cycle_demand(analysis,rows,cols,workers):
    r=(rows+workers-1)//workers;vector=analysis['vector'];scalar=analysis['scalar']
    if vector:
        # A scalar alias-fallback loop is a proxy for the at-most-one tail.
        _require(not cols%2 or scalar is not None,'odd-width scalar tail instruction proxy missing')
        return r*((cols//2)*vector['block_throughput_cycles']+(cols%2)*(scalar['cycles_per_element'] if scalar else 0))
    _require(scalar is not None,'scalar execution loop missing')
    return r*cols*scalar['cycles_per_element']


def _derive(observations,analyses):
    overhead=next(r['median_ns'] for r in observations if r['role']=='DISPATCH')
    mappings=[]
    for row in observations:
        if row['role']!='MATH':continue
        cycles=_cycle_demand(analyses[str(row['kernel_index'])],row['rows'],row['cols'],row['workers'])
        _require(row['median_ns']>overhead and cycles>0,'invalid independent cycle mapping')
        mappings.append(dict(configuration=row['id'],cycles=cycles,effective_ns_per_mca_cycle=(row['median_ns']-overhead)/cycles))
    memory={}
    for group in (1,2):
        memory[str(group)]={}
        for loads in ((1,2,4,6,8) if group==1 else (2,4,6,8)):
            rows=[r for r in observations if r['role']=='MEMORY' and r['groups']==group and r['array_loads']==loads]
            _require(all(r['median_ns']>overhead for r in rows),'invalid memory proxy samples')
            memory[str(group)][str(loads)]=[dict(elements=r['elements'],proxy_ns_per_element=(r['median_ns']-overhead)/r['elements'],configuration=r['id']) for r in rows]
    return dict(dispatch_ns=overhead,effective_ns_per_mca_cycle=statistics.median(r['effective_ns_per_mca_cycle'] for r in mappings),
        cycle_mapping_observations=mappings,memory=memory,authority='INDEPENDENT_FIXED_INSTRUCTION_AND_MEMORY_PROXY_FACTS',
        actual_clock_frequency_measured=False,mca_cache_latency_measured=False)


def _predict(row,analysis,coefficients):
    points=[];elements=row['elements']
    for loads,rows in coefficients['memory'][str(row['groups'])].items():
        points.append((int(loads),_interpolate([(r['elements'],r['proxy_ns_per_element']) for r in rows],elements)))
    memory=_interpolate(points,row['array_loads'])*elements
    cycles=_cycle_demand(analysis,row['rows'],row['cols'],row['workers'])
    execution=cycles*coefficients['effective_ns_per_mca_cycle']
    return dict(duration_ns=coefficients['dispatch_ns']+max(memory,execution),dispatch_ns=coefficients['dispatch_ns'],
        mca_execution_proxy_ns=execution,memory_proxy_ns=memory,mca_cycle_demand=cycles,
        spill_accounting='ACTUAL_STACK_MEMORY_INSTRUCTIONS_INCLUDED_IN_MCA_L1_MODEL',
        composition=_POLICY['composition'],physical_bound=False)


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


def calibrate_instruction_cpu(resource_context,out,*,repetitions=5,warmup=2,batch=8):
    from scar.ir.execution_recipe import artifact_reference
    context=deepcopy(resource_context);out=Path(out).resolve();_require(not out.exists(),'fresh output required')
    _require(repetitions>=3 and warmup>=1 and batch>=1,'repeated warm samples required')
    _require(len(context['worker_cpu_ids'])==1,'first instruction-cost qualification supports serial CPU only')
    _require(all(os.environ.get(k)==v for k,v in context['native_thread_environment'].items()),'thread environment differs')
    _,cache=_llc(context);tool=_tool_contract(context);configs=_configurations(context)
    implementation={name:_hash((_HERE/name).read_bytes()) for name in
        ('cpu_instruction_cost.py','cpu_fused_cost_v2.py','cpu_fused_cost_v2.cpp.in','cpu_fused_cost_v3.py','cpu_fused_cost.py','loop_v1.cpp.in','scar_abi_v1.h')}
    out.mkdir(parents=True);start=time.time();before=list(os.getloadavg())
    design=dict(schema='scar.cpu-instruction-cost-design',schema_version=1,resource_context=context,tool=tool,
        registry=_registry(),configurations=configs,precision_policy=_POLICY,implementation_files=implementation,
        repetitions=repetitions,warmup=warmup,batch=batch,model_frozen_before_measurement=True,
        target_workload_executed=False,target_dag_provided_to_probe=False,target_variants_ranked=False)
    _save(out/'design.json',design);design_ref=artifact_reference(out/'design.json',design['schema'])
    handle,function,native=_compile(context,out/'native');assembly=Path(native['assembly']['path']).read_text();analyses={}
    for row in configs:
        key=str(row['kernel_index'])
        if row['role']=='DISPATCH' or key in analyses:continue
        case=_registry()[row['case_index']]
        analyses[key]=_analyze(assembly,tool,out/'mca'/key,function=case['id']+'_g'+str(row['groups']))
    _save(out/'instruction-analysis.json',dict(schema='scar.cpu-instruction-analysis',schema_version=1,tool=tool,analyses=analyses))
    observations=[]
    for row in configs:
        layout=_Layout();layout.rows=row['rows'];layout.cols=row['cols'];layout.loads=row['array_loads']
        for i,p in enumerate(row['layouts']):layout.strides[i]=p['stride'];layout.offsets[i]=p['offset'];layout.groups[i]=p['storage_group']
        samples=(ctypes.c_double*repetitions)();sink=ctypes.c_uint64();cpus=(ctypes.c_int32*2)()
        status=function(row['kernel_index'],ctypes.byref(layout),repetitions,warmup,batch,samples,ctypes.byref(sink),cpus)
        _require(status==0,'independent FP/affinity/allocation guard: '+str(status));values=list(samples)
        _require(all(math.isfinite(x) and x>0 for x in values),'positive finite stage samples required')
        observations.append(dict(**row,samples_ns=values,median_ns=statistics.median(values),min_ns=min(values),max_ns=max(values),
            stdev_ns=statistics.stdev(values),checksum=sink.value,caller_cpu_before=cpus[0],caller_cpu_after=cpus[1]))
    raw=dict(schema='scar.cpu-instruction-cost-observations',schema_version=1,frozen_design=design_ref,native=native,observations=observations)
    _save(out/'raw-observations.json',raw);coefficients=_derive(observations,analyses)
    _require(sorted(os.sched_getaffinity(0))==context['caller_affinity_cpu_ids']
        and artifact_reference(out/'design.json',design_ref['schema'])==design_ref,'frozen context/design changed')
    model=dict(schema='scar.cpu-instruction-cost-model',schema_version=1,resource_context=context,frozen_design=design_ref,
        raw_observations=artifact_reference(out/'raw-observations.json',raw['schema']),
        instruction_analysis=artifact_reference(out/'instruction-analysis.json','scar.cpu-instruction-analysis'),
        coefficients=coefficients,local_models=_qualify(observations,analyses,coefficients),tool=tool,native=native,
        implementation_files=implementation,cache_geometry=cache,precision_policy=_POLICY,assumptions=_ASSUMPTIONS,
        calibration_seconds=time.time()-start,host_load=dict(before=before,after=list(os.getloadavg()),isolated=False),
        serial_boundary='NO_OPENMP_TEAM_OR_STAGE_BARRIER_IN_SERIAL_NATIVE_SAMPLE',
        target_workload_executed=False,target_variants_ranked=False,target_dag_compiled_for_static_analysis=False,
        physical_optimality_proven=False,deployment_authorized=False)
    _save(out/'profile.json',model);return model


def _verify_model(path):
    from scar.ir.execution_recipe import artifact_reference,read_artifact
    reference=artifact_reference(path,'scar.cpu-instruction-cost-model');model=read_artifact(reference)
    design=read_artifact(model['frozen_design']);raw=read_artifact(model['raw_observations'])
    analysis=read_artifact(model['instruction_analysis']);analyses=analysis['analyses']
    _require(model['schema_version']==1 and model['precision_policy']==design['precision_policy']==_POLICY,
        'model precision policy differs')
    _require(model['implementation_files']==design['implementation_files']=={
        name:_hash((_HERE/name).read_bytes()) for name in model['implementation_files']},'current producer implementation differs')
    _require(_tool_contract(model['resource_context'])==model['tool']==design['tool']==analysis['tool'],
        'MCA binary/model/CPU contract differs')
    _require(design['registry']==_registry() and design['configurations']==_configurations(model['resource_context'])
        and raw['frozen_design']==model['frozen_design'] and raw['native']==model['native'],'canonical design/native differs')
    _require(model['target_workload_executed'] is False and model['target_variants_ranked'] is False
        and design['target_workload_executed'] is False and design['target_dag_provided_to_probe'] is False,
        'source-independent calibration claims differ')
    _require(len(raw['observations'])==len(design['configurations']),'incomplete fixed design')
    for row,config in zip(raw['observations'],design['configurations']):
        _require(all(row[k]==v for k,v in config.items()) and len(row['samples_ns'])==design['repetitions']
            and all(math.isfinite(x) and x>0 for x in row['samples_ns'])
            and statistics.median(row['samples_ns'])==row['median_ns'],'canonical samples differ')
    source=Path(model['native']['source']['path']).read_text()
    _require(source==_source(model['resource_context']),'canonical source text differs')
    for receipt in model['native'].values():
        if type(receipt) is dict and {'path','sha256','bytes'}<=set(receipt):
            _require(_receipt(Path(receipt['path']))==receipt,'canonical compiler/source/native receipt differs')
    assembly=Path(model['native']['assembly']['path']).read_text()
    for key,value in analyses.items():
        index=int(key)//2;group=int(key)%2+1;case=_registry()[index]
        loops=_natural_loops(assembly,function=case['id']+'_g'+str(group))
        _require(len(loops)==len(value['loops']),'instruction loop set differs')
        for original,retained in zip(loops,value['loops']):
            _require(_receipt(Path(retained['loop_source']['path']))==retained['loop_source']
                and Path(retained['loop_source']['path']).read_text()==original['text']
                and _receipt(Path(retained['result']['path']))==retained['result'],'MCA input/output receipt differs')
            result=json.loads(Path(retained['result']['path']).read_text());summary=result['CodeRegions'][0]['SummaryView']
            _require(retained['cycles_per_element']==summary['BlockRThroughput']/original['width']
                and retained['block_throughput_cycles']==summary['BlockRThroughput']
                and result['TargetInfo']['CPUName']==model['tool']['cpu_model'],'MCA numerical summary differs')
    _require(_derive(raw['observations'],analyses)==model['coefficients']
        and _qualify(raw['observations'],analyses,model['coefficients'])==model['local_models'],
        'coefficient/holdout replay differs')
    _require(_llc(model['resource_context'])[1]==model['cache_geometry'],'cache evidence differs')
    return reference,model,raw,analyses


def _target_static(domain,candidate_id,context,out,tool):
    """Emit the unchanged backend source and compile only to assembly."""
    from dataclasses import asdict
    from scar.ir.execution_recipe import read_artifact
    from scar.compiler.stateful_model import stateful_from_dict,stateful_digest
    from scar.compiler.model import CpuPlacement
    from scar.optimization.stateful_schedule import plan_stateful_region
    from scar.codegen import stateful
    region=stateful_from_dict(read_artifact(domain['region']))
    candidate=next(r for r in domain['candidates'] if r['id']==candidate_id)
    binding=dict(candidate['binding']);binding['cpu_ids']=tuple(binding['cpu_ids']);binding['evidence']=tuple(binding['evidence'])
    placement=CpuPlacement(**binding)
    plan=plan_stateful_region(region,input_shapes=domain['request']['input_shapes'],cpu_threads=len(placement.cpu_ids),
        memory_budget_bytes=placement.memory_budget_bytes,resource_binding=placement)
    region,plan,placement=stateful._admit(region,plan,placement)
    contract=dict(region_sha256=stateful_digest(region),plan_sha256=stateful._digest(plan),
        provider_runtime=context['backend']['provider_runtime'])
    source=stateful._source(region,plan,placement,contract);out.mkdir()
    path=out/'stateful.cpp';path.write_text(source)
    for name in ('stateful_abi_v1.h','scar_abi_v1.h'):(out/name).write_bytes((_HERE/name).read_bytes())
    assembly=out/'stateful.s';remarks=out/'vectorization.txt';compiler=context['backend']['compiler']
    command=[*compiler['argv'],*compiler['options'],'-S','-fverbose-asm','-fopt-info-vec-optimized='+str(remarks),str(path),'-o',str(assembly)]
    p=subprocess.run(command,capture_output=True,text=True);(out/'compile.log').write_text(p.stdout+p.stderr)
    _require(p.returncode==0,'original backend static instruction lowering failed: '+p.stderr[-1000:])
    document=dict(schema='scar.stateful-static-instruction-lowering',schema_version=1,
        source=_receipt(path),assembly=_receipt(assembly),vectorization=_receipt(remarks),compiler=compiler,command=command,
        region=domain['region'],plan=plan,resource_binding=asdict(placement),backend_implementation=stateful.implementation_sha256(),
        target_dag_compiled_for_static_analysis=True,target_workload_executed=False,target_variants_ranked=False,
        native_library_built=False)
    _save(out/'lowering.json',document);return region,plan,source,assembly.read_text(),document


def _stage_source_range(source,root_order):
    lines=source.splitlines();needle='const double value'+str(root_order)+' ='
    found=[i for i,l in enumerate(lines) if needle in l];_require(len(found)==1,'backend expression/source mapping differs')
    root=found[0];start=next(i for i in range(root,-1,-1) if re.search(r'for \(int64_t row = 0;',lines[i]))
    balance=0
    for end in range(start,len(lines)):
        balance+=lines[end].count('{')-lines[end].count('}')
        if balance==0 and end>start:return (start+1,end+1)
    raise ValueError('backend source row-loop boundary missing')


def predict_instruction_components(domain,candidate_id,profile,out):
    from scar.optimization.stateful_cpu_choices import _domain
    from scar.optimization.stateful_cpu_model import stateful_cpu_demand
    from scar.ir.execution_recipe import artifact_reference
    domain=_domain(domain);demand=stateful_cpu_demand(domain,candidate_id)
    reference,model,raw,calibrated_analyses=_verify_model(profile)
    _require(demand['resource_context']==model['resource_context'],'exact hardware/runtime/compiler/mask context differs')
    out=Path(out).resolve();_require(not out.exists(),'fresh projection output required');out.mkdir(parents=True)
    region,plan,source,assembly,lowering=_target_static(domain,candidate_id,demand['resource_context'],out/'static',model['tool'])
    order={n.id:i for i,n in enumerate(region.nodes)};components=[];unsupported=[];proposals=[];capacity=model['cache_geometry']['reported_capacity_bytes']
    calibrated_stack_max=max(r['stack_memory_references'] for a in calibrated_analyses.values() for r in a['loops'])
    for term in demand['terms']:
        if term['component']!='FUSED_SCALAR_DAG_ROW_STAGE':continue
        proposal=None
        try:
            p=term['parameters'];stage_index=int(term['source_ref']['pointer'].rsplit('/',1)[1]);stage=plan['stages'][stage_index]
            _require(stage['kind']=='COMPUTE' and p['workers']==1,'unsupported stage/parallel boundary')
            output=plan['assignment'][stage['output_payload']]['slot_id']
            input_slots=[plan['assignment'][key]['slot_id'] for key in stage['input_payloads'] if key in plan['assignment']]
            _require(output not in input_slots,'output workspace aliases a live input slot')
            expected=V2._layouts(p['array_loads'],p['cols'],p['backing_storage_groups'])
            key=lambda r:(r['storage_group'],r['stride'],r['offset'])
            _require(sorted(p['operand_layouts'],key=key)==sorted(expected,key=key),'uncalibrated root/axial-neighborhood memory geometry')
            _,geometry=V2.fused_features(p,capacity)
            _require(256<=p['rows']<=768 and 320<=p['cols']<=832
                and geometry['backing_bytes']+p['rows']*p['cols']*8<=capacity,'outside measured geometry/reported cache extent scope')
            source_range=_stage_source_range(source,order[stage['root_expr']])
            analysis=_analyze(assembly,model['tool'],out/'mca'/str(stage_index),source_range=source_range,
                source_path=lowering['source']['path'])
            _require(max(r['stack_memory_references'] for r in analysis['loops'])<=calibrated_stack_max,
                'target stack-memory instruction pressure exceeds independent calibration scope')
            cost=_predict(dict(groups=p['backing_storage_groups'],rows=p['rows'],cols=p['cols'],workers=p['workers'],
                elements=((p['rows']+p['workers']-1)//p['workers'])*p['cols'],array_loads=p['array_loads']),analysis,model['coefficients'])
            qualified=model['local_models'][str(p['backing_storage_groups'])]
            proposal=dict(component=term['component'],parameters=p,prediction=cost,
                maximum_holdout_relative_error=qualified['maximum_holdout_relative_error'],precision_policy=_POLICY,
                target_dag_compiled_for_static_analysis=True,measured_target_component=False,selectable=False)
            _require(qualified['admitted'],'fresh independent holdout maximum relative error '
                +str(qualified['maximum_holdout_relative_error'])+' exceeds frozen0.20 policy')
            components.append(dict(component=term['component'],parameters=p,basis='DECLARED_COMPONENT_MODEL',median_ns=cost['duration_ns'],
                prediction=cost,source_model=dict(artifact=reference,locator='#/coefficients'),
                source_validation=dict(artifact=reference,locator='#/local_models/'+str(p['backing_storage_groups'])+'/holdout'),
                source_provenance=term['source_ref'],instruction_analysis=analysis,
                static_lowering=artifact_reference(out/'static/lowering.json',lowering['schema']),
                workspace_alias_authority=dict(output_slot=output,input_slots=input_slots,live_slots_distinct=True),
                precision_policy=_POLICY,maximum_holdout_relative_error=qualified['maximum_holdout_relative_error'],
                target_dag_compiled_for_static_analysis=True,measured_target_component=False,physical_optimality_proven=False,
                assumptions=_ASSUMPTIONS))
        except (ValueError,StopIteration) as error:
            unsupported.append(dict(component=term['component'],parameters=term['parameters'],reason=str(error)))
            if proposal is not None:proposals.append(dict(**proposal,rejection_reason=str(error)))
    result=dict(schema='scar.stateful-resource-calibration',schema_version=1,resource_context=demand['resource_context'],
        components=components,unsupported_components=unsupported,proposed_estimates=proposals,source_model=reference,
        static_lowering=artifact_reference(out/'static/lowering.json',lowering['schema']),
        all_fused_components_covered=not unsupported,complete_method_cost_measured=False,
        target_dag_compiled_for_static_analysis=True,target_workload_executed=False,target_variants_ranked=False,physical_optimality_proven=False)
    _save(out/'observations.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('calibrate');c.add_argument('--context',required=True);c.add_argument('--out',required=True)
    c.add_argument('--repetitions',type=int,default=5);c.add_argument('--warmup',type=int,default=2);c.add_argument('--batch',type=int,default=8)
    c=sub.add_parser('project');c.add_argument('--domain',required=True);c.add_argument('--candidate-id',required=True)
    c.add_argument('--profile',required=True);c.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if a.command=='project':
        result=predict_instruction_components(a.domain,a.candidate_id,a.profile,a.out)
        print(json.dumps(dict(components=len(result['components']),unsupported=result['unsupported_components']),sort_keys=True));return 0
    model=calibrate_instruction_cpu(json.loads(Path(a.context).read_text()),a.out,
        repetitions=a.repetitions,warmup=a.warmup,batch=a.batch)
    print(json.dumps(dict(profile=str(Path(a.out).resolve()/'profile.json'),local_models={k:dict(admitted=v['admitted'],
        maximum_holdout_relative_error=v['maximum_holdout_relative_error']) for k,v in model['local_models'].items()}),sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())

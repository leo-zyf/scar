"""CPU instruction profiles, independent probe calibration and static cost projection."""

import argparse
import ast
from collections.abc import Callable
from dataclasses import dataclass
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
from .cpu_fused_cost import _HERE,_Layout,_hash,_llc,_receipt,_require,_save


_MCA=Path('/opt/amd/aocc/5.1/bin/llvm-mca')
_PARTITION_ROWS = (256, 439, 512, 513, 618, 768)
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


_PARALLEL_POLICY=dict(maximum_holdout_relative_error=.20,
    coefficient_sources='INDEPENDENT_EXACT_MASK_PARALLEL_MIXED_ALU_MEMORY_AND_BARRIER_PRIMITIVES',
    holdout_used_for_fitting=False,holdout_used_for_model_choice=False,
    composition='ONE_STAGE_BARRIER_FLOOR_PLUS_MAX_WORKER_MCA_AND_SHARED_MEMORY_PROXY',
    authority='ENGINEERING_ADMISSION_NOT_A_THEORETICAL_ERROR_BOUND')
_PARALLEL_ASSUMPTIONS=[*_ASSUMPTIONS,
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


def _parallel_analyze(assembly,tool,out,**kwargs):
    return dict(**_analyze(_lex_assembly(assembly),tool,out,**kwargs),instruction_lexer=_lexer_evidence(assembly))


def _parallel_registry():
    rows=[deepcopy(r) for r in _registry() if not r['holdout']]
    rows[0].update(id='parallel_stage_barrier_floor',completion_barriers=1)
    for i,(family,n,extra) in enumerate((('SERIAL_FANIN',5,10),('SERIAL_FANIN',7,6),
        ('BALANCED_FANIN',3,9),('BALANCED_FANIN',5,10),('PARALLEL_BRANCHES',5,10),
        ('PARALLEL_BRANCHES',7,6),('SHARED_BRANCHES',3,8),('SHARED_BRANCHES',5,10))):
        case=V2._case(family,n,extra,seed=541+i,holdout=True);pool=3 if i%2==0 else 6
        for node in case['graph']:
            node['inputs']=['s'+str(int(k[1:])%pool) if k.startswith('s') else k for k in node['inputs']]
        case.update(role='FRESH_HOLDOUT',operand_pool=pool);rows.append(case)
    return rows


def _parallel_configurations(context):
    rows=[];workers=len(context['worker_cpu_ids'])
    for i,case in enumerate(_parallel_registry()):
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
                    layouts=V2._layouts(case['array_loads'],c,group),role=case['role'],holdout=case['holdout'],
                    logical_worker_rows=chunks,partition='GOMP_STATIC_CONTIGUOUS_FIRST_REMAINDER_DECLARED_CONTEXT'))
    random.Random(671933).shuffle(rows);return rows


def _parallel_source(context):
    base=_source(context);head=base[:base.index('__attribute__((noinline)) void instruction_')]
    tail=base[base.index('extern "C" SCAR_EXPORT_V1 int32_t scar_instruction_cost_probe_v1'):]
    functions=[];pointers=[];loads=[]
    for case in _parallel_registry():
        for group in (1,2):
            name=case['id']+'_g'+str(group);body=[]
            for i,layout in enumerate(V2._layouts(case['array_loads'],100,group)):
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


def _parallel_tool_contract(context):
    tool=_tool_contract(context);compiler=context['backend']['compiler']
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
    for rows in _PARTITION_ROWS:
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


def _parallel_derive(observations,analyses):
    coefficients=_derive(observations,analyses)
    coefficients['barrier_floor_ns']=coefficients.pop('dispatch_ns')
    coefficients['authority']='INDEPENDENT_EXACT_MASK_SHARED_TEAM_INSTRUCTION_MEMORY_AND_STAGE_BARRIER_FACTS'
    coefficients['serial_duration_divided_by_workers']=False
    return coefficients


def _parallel_predict(row,analysis,coefficients):
    translated={**coefficients,'dispatch_ns':coefficients['barrier_floor_ns']}
    value=_predict(row,analysis,translated)
    value['barrier_floor_ns']=value.pop('dispatch_ns');value['composition']=_PARALLEL_POLICY['composition']
    value['worker_aggregation']='LARGEST_LOGICAL_STATIC_CHUNK_AND_MAX_WORKER_WALL'
    value['completion_barriers_charged']=1;value['team_lifecycle_charged_here']=False
    return value


def _static_stage_contract(domain, candidate_id, context):
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
    return region, plan, placement, contract


def _target_static(domain, candidate_id, context, out):
    """Emit the admitted backend source and compile only to assembly."""
    from dataclasses import asdict
    from scar.codegen import stateful

    region, plan, placement, contract = _static_stage_contract(domain, candidate_id, context)
    source = stateful._source(region, plan, placement, contract)
    out.mkdir()
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


@dataclass(frozen=True)
class InstructionCostProfile:
    mode: str
    registry: Callable
    configurations: Callable
    source: Callable
    tool_contract: Callable
    analyze: Callable
    derive: Callable
    predict: Callable
    precision_policy: dict
    assumptions: list

    @property
    def parallel(self):
        return self.mode == 'parallel'

    @property
    def prefix(self):
        return 'scar.cpu-parallel-instruction' if self.parallel else 'scar.cpu-instruction'

    def require_workers(self, count):
        _require(
            count > 1 if self.parallel else count == 1,
            f'invalid {self.mode} worker count: {count}',
        )


SERIAL = InstructionCostProfile(
    'serial', _registry, _configurations, _source, _tool_contract, _analyze,
    _derive, _predict, _POLICY, _ASSUMPTIONS,
)
PARALLEL = InstructionCostProfile(
    'parallel', _parallel_registry, _parallel_configurations, _parallel_source,
    _parallel_tool_contract, _parallel_analyze, _parallel_derive, _parallel_predict,
    _PARALLEL_POLICY, _PARALLEL_ASSUMPTIONS,
)
INSTRUCTION_COST_SCHEMAS = {
    profile.prefix + '-cost-model': profile for profile in (SERIAL, PARALLEL)
}
_SOURCE_FILE = Path(__file__).name
_BARRIER_ACCOUNTING = 'ONE_PRIMITIVE_STAGE_COMPLETION_BARRIER_PLUS_DISPATCH_NOT_TEAM_LIFECYCLE'


def instruction_cost_profile(mode):
    if mode == 'serial':
        return SERIAL
    if mode == 'parallel':
        return PARALLEL
    raise ValueError('serial or parallel compiler mode required')


def _implementation():
    names = (
        _SOURCE_FILE, 'cpu_fused_cost_v2.py', 'cpu_fused_cost_v2.cpp.in',
        'cpu_fused_cost_v3.py', 'cpu_fused_cost.py', 'loop_v1.cpp.in', 'scar_abi_v1.h',
    )
    return {name: _hash((_HERE / name).read_bytes()) for name in names}


def _compile_probe(context, directory, source_text):
    compiler = context['backend']['compiler']
    identity = subprocess.run([*compiler['argv'], '--version'], check=True,
        capture_output=True, text=True).stdout.splitlines()[0]
    _require(identity == compiler['identity'] and 'g++' in identity, 'exact GCC required')
    directory.mkdir(parents=True)
    source = directory / 'canonical.cpp'
    source.write_text(source_text)
    (directory / 'scar_abi_v1.h').write_bytes((_HERE / 'scar_abi_v1.h').read_bytes())
    assembly = directory / 'canonical.s'
    remarks = directory / 'vectorization.txt'
    library = directory / 'canonical.so'
    flags = [*compiler['argv'], *compiler['options']]
    commands = [
        [*flags, '-S', '-fverbose-asm', '-fopt-info-vec-optimized=' + str(remarks),
         str(source), '-o', str(assembly)],
        [*flags, str(assembly), '-o', str(library)],
    ]
    for index, command in enumerate(commands):
        result = subprocess.run(command, capture_output=True, text=True)
        (directory / f'compile-{index}.log').write_text(result.stdout + result.stderr)
        _require(result.returncode == 0,
            'independent instruction probe compilation failed: ' + result.stderr[-1000:])
    return dict(compiler=compiler, commands=commands, source=_receipt(source), assembly=_receipt(assembly),
        library=_receipt(library), vectorization_remarks=_receipt(remarks))


def _compile(context, directory, profile):
    cpus = context['worker_cpu_ids']
    profile.require_workers(len(cpus))
    affinity = os.sched_getaffinity(0)
    _require(set(cpus) <= affinity and sorted(affinity) == context['caller_affinity_cpu_ids'],
        'exact caller and worker affinity required')
    _require(('-fopenmp' in context['backend']['compiler']['options']) == profile.parallel,
        'serial/OpenMP compiler contract differs')
    native = _compile_probe(context, directory, profile.source(context))
    handle = ctypes.CDLL(native['library']['path'])
    function = handle.scar_instruction_cost_probe_v1
    function.argtypes = [
        ctypes.c_int32, ctypes.POINTER(_Layout), ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_uint64, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int32),
    ]
    function.restype = ctypes.c_int32
    native['scope'] = (
        'MAX_PINNED_WORKER_WALL_ONE_NATIVE_STAGE_WITH_OWN_IMPLICIT_COMPLETION_BARRIER'
        if profile.parallel else 'SERIAL_NATIVE_STAGE_NO_OPENMP'
    )
    return handle, function, native


def _probe_layout(row):
    layout = _Layout(rows=row['rows'], cols=row['cols'], loads=row['array_loads'])
    for index, entry in enumerate(row['layouts']):
        layout.strides[index] = entry['stride']
        layout.offsets[index] = entry['offset']
        layout.groups[index] = entry['storage_group']
    return layout


def _sample_summary(samples):
    values = list(samples)
    _require(all(math.isfinite(value) and value > 0 for value in values),
        'positive finite stage samples required')
    return dict(samples_ns=values, median_ns=statistics.median(values),
        min_ns=min(values), max_ns=max(values), stdev_ns=statistics.stdev(values))


def _observe(function, configs, repetitions, warmup, batch):
    observations = []
    for row in configs:
        layout = _probe_layout(row)
        samples = (ctypes.c_double * repetitions)()
        sink = ctypes.c_uint64()
        cpus = (ctypes.c_int32 * 2)()
        status = function(row['kernel_index'], ctypes.byref(layout), repetitions, warmup, batch,
            samples, ctypes.byref(sink), cpus)
        _require(status == 0, 'independent FP/affinity/allocation guard: ' + str(status))
        observations.append(dict(**row, **_sample_summary(samples), checksum=sink.value,
            caller_cpu_before=cpus[0], caller_cpu_after=cpus[1]))
    return observations


def _verify_observations(observations, configs, repetitions):
    _require(len(observations) == len(configs), 'incomplete fixed design')
    for row, config in zip(observations, configs):
        _require(
            all(row[key] == value for key, value in config.items())
            and len(row['samples_ns']) == repetitions
            and all(math.isfinite(value) and value > 0 for value in row['samples_ns'])
            and statistics.median(row['samples_ns']) == row['median_ns'],
            'canonical samples differ',
        )


def _qualify(observations, analyses, coefficients, profile):
    result = {}
    for group in (1, 2):
        holdout = []
        for row in observations:
            if not row['holdout'] or row['groups'] != group:
                continue
            analysis = analyses[str(row['kernel_index'])]
            predicted = profile.predict(row, analysis, coefficients)
            observed = row['median_ns']
            holdout.append(dict(configuration=row['id'], observed_ns=observed, prediction=predicted,
                relative_error=abs(predicted['duration_ns'] - observed) / observed))
        maximum = max(row['relative_error'] for row in holdout)
        policy = profile.precision_policy
        result[str(group)] = dict(holdout=holdout, maximum_holdout_relative_error=maximum,
            admitted=maximum <= policy['maximum_holdout_relative_error'], precision_policy=policy)
    return result


def _instruction_analyses(assembly, tool, configs, registry, profile, out):
    from scar.ir.execution_recipe import artifact_reference

    analyses = {}
    for row in configs:
        key = str(row['kernel_index'])
        if row['role'] == 'DISPATCH' or key in analyses:
            continue
        case = registry[row['case_index']]
        analyses[key] = profile.analyze(assembly, tool, out / 'mca' / key,
            function=case['id'] + '_g' + str(row['groups']))
    schema = profile.prefix + '-analysis'
    path = out / 'instruction-analysis.json'
    _save(path, dict(schema=schema, schema_version=1, tool=tool, analyses=analyses))
    return artifact_reference(path, schema), analyses


def _require_environment(context):
    _require(all(os.environ.get(key) == value for key, value in context['native_thread_environment'].items()),
        'thread environment differs')


def _fitted_model(profile, design_ref, design, raw_ref, raw, analysis_ref, analyses, implementation):
    observations = raw['observations']
    coefficients = profile.derive(observations, analyses)
    context = design['resource_context']
    model = dict(schema=profile.prefix + '-cost-model', schema_version=1, resource_context=context,
        frozen_design=design_ref, raw_observations=raw_ref, instruction_analysis=analysis_ref,
        coefficients=coefficients, local_models=_qualify(observations, analyses, coefficients, profile),
        tool=design['tool'], native=raw['native'], implementation_files=implementation,
        cache_geometry=_llc(context)[1], precision_policy=profile.precision_policy, assumptions=profile.assumptions,
        target_workload_executed=False, target_variants_ranked=False,
        target_dag_compiled_for_static_analysis=False, physical_optimality_proven=False, deployment_authorized=False)
    if profile.parallel:
        model.update(parallel_boundary=design['barrier_accounting'], partition_witnesses=raw['partition_witnesses'],
            serial_cost_values_reused=False)
    else:
        model['serial_boundary'] = 'NO_OPENMP_TEAM_OR_STAGE_BARRIER_IN_SERIAL_NATIVE_SAMPLE'
    return model


def calibrate_instruction_cpu(resource_context, out, *, mode='serial', repetitions=5, warmup=2, batch=8):
    from scar.ir.execution_recipe import artifact_reference

    profile = instruction_cost_profile(mode)
    context = deepcopy(resource_context)
    out = Path(out).resolve()
    _require(not out.exists(), 'fresh output required')
    _require(repetitions >= 3 and warmup >= 1 and batch >= 1, 'repeated warm samples required')
    profile.require_workers(len(context['worker_cpu_ids']))
    _require_environment(context)
    _llc(context)
    tool = profile.tool_contract(context)
    configs = profile.configurations(context)
    registry = profile.registry()
    implementation = _implementation()
    out.mkdir(parents=True)
    started = time.time()
    load_before = list(os.getloadavg())
    design = dict(schema=profile.prefix + '-cost-design', schema_version=1, resource_context=context, tool=tool,
        registry=registry, configurations=configs, precision_policy=profile.precision_policy,
        implementation_files=implementation, repetitions=repetitions, warmup=warmup, batch=batch,
        model_frozen_before_measurement=True, target_workload_executed=False,
        target_dag_provided_to_probe=False, target_variants_ranked=False)
    if profile.parallel:
        design.update(barrier_accounting=_BARRIER_ACCOUNTING, serial_cost_values_reused=False,
            serial_duration_divided_by_workers=False)
    design_path = out / 'design.json'
    _save(design_path, design)
    design_ref = artifact_reference(design_path, design['schema'])
    handle, function, native = _compile(context, out / 'native', profile)
    raw = dict(schema=profile.prefix + '-cost-observations', schema_version=1, frozen_design=design_ref, native=native)
    if profile.parallel:
        _require(_loaded_openmp(handle) == tool['openmp_runtime'], 'actual probe OpenMP runtime differs')
        raw['partition_witnesses'] = _partition_witnesses(handle, context)
    assembly = Path(native['assembly']['path']).read_text()
    analysis_ref, analyses = _instruction_analyses(assembly, tool, configs, registry, profile, out)
    raw['observations'] = _observe(function, configs, repetitions, warmup, batch)
    raw_path = out / 'raw-observations.json'
    _save(raw_path, raw)
    raw_ref = artifact_reference(raw_path, raw['schema'])
    _require_environment(context)
    _require(sorted(os.sched_getaffinity(0)) == context['caller_affinity_cpu_ids']
        and artifact_reference(design_path, design['schema']) == design_ref, 'frozen context/design changed')
    model = _fitted_model(profile, design_ref, design, raw_ref, raw, analysis_ref, analyses, implementation)
    model.update(calibration_seconds=time.time() - started,
        host_load=dict(before=load_before, after=list(os.getloadavg()), isolated=False))
    _save(out / 'profile.json', model)
    return model


def _architecture_identity(original):
    def semantics(source):
        tree = ast.parse(source)
        allowed = {'_lex_assembly', '_lexer_evidence'}
        tree.body = [node for node in tree.body
            if not (isinstance(node, ast.FunctionDef) and node.name in allowed)]
        return ast.dump(tree, include_attributes=False)

    before = semantics(original)
    _require(before == semantics(Path(__file__).read_text()),
        'instruction producer semantics changed beyond the lexical repair')
    return _hash(before.encode())


def _validate_repair(repair, original, design):
    _require(repair['schema'] == 'scar.parallel-instruction-lexer-correction-plan'
        and repair['schema_version'] == 1
        and repair['original_producer'] == original
        and repair['holdout_timing_or_error_values_inspected_before_correction'] is False
        and repair['registry_or_harness_or_cost_formula_or_precision_policy_changed'] is False
        and repair['repeated_measurements'] is False, 'blinded formal-repair provenance differs')
    path = Path(original['path'])
    _require(_receipt(path) == original and original['sha256'] == design['implementation_files'].get(_SOURCE_FILE),
        'original physical-observation producer differs')
    return path


def _unchanged_support(design, current):
    return {k: v for k, v in design['implementation_files'].items() if k != _SOURCE_FILE} == {
        k: v for k, v in current.items() if k != _SOURCE_FILE
    }


def _verify_qualification(model, design, current):
    from scar.ir.execution_recipe import read_artifact

    qualification = read_artifact(model['qualification_design'])
    _require(qualification['schema'] == 'scar.parallel-instruction-requalification-design'
        and qualification['schema_version'] == 1 and qualification['frozen_source_design'] == model['frozen_design']
        and qualification['source_observations'] == model['raw_observations']
        and qualification['implementation_files'] == current and qualification['precision_policy'] == PARALLEL.precision_policy
        and qualification['native_source_replayed_unchanged'] is True
        and qualification['holdout_timing_or_error_values_inspected_before_correction'] is False
        and _unchanged_support(design, current), 'explicit lexer-correction qualification contract differs')
    repair = read_artifact(qualification['correction_plan'])
    original = _validate_repair(repair, qualification['original_producer'], design)
    _require(_receipt(Path(model['raw_observations']['uri'])) == repair['raw_observations'],
        'fixed original physical observations differ')
    _require(_architecture_identity(original.read_text()) == qualification['unchanged_architecture_ast_sha256'],
        'mechanical nonnumerical-repair AST identity differs')


def _verify_native_source(native, context, profile):
    for receipt in native.values():
        if isinstance(receipt, dict) and {'path', 'sha256', 'bytes'} <= receipt.keys():
            _require(_receipt(Path(receipt['path'])) == receipt, 'native compiler/source receipt differs')
    _require(Path(native['source']['path']).read_text() == profile.source(context), 'canonical source differs')


def _verify_partitions(witnesses, context):
    _require([row['rows'] for row in witnesses] == list(_PARTITION_ROWS),
        'complete fixed partition witness design required')
    cpus = context['worker_cpu_ids']
    workers = len(cpus)
    for row in witnesses:
        count, remainder = divmod(row['rows'], workers)
        _require(row['worker_rows'] == [count + (tid < remainder) for tid in range(workers)]
            and row['first_rows'] == [tid * count + min(tid, remainder) for tid in range(workers)]
            and row['actual_cpu_ids'] == cpus and row['largest_rows'] == count + bool(remainder),
            'retained static partition witness differs')


def _verify_model(path, profile):
    from scar.ir.execution_recipe import artifact_reference, read_artifact

    reference = artifact_reference(path, profile.prefix + '-cost-model')
    model = read_artifact(reference)
    design = read_artifact(model['frozen_design'])
    raw = read_artifact(model['raw_observations'])
    analysis = read_artifact(model['instruction_analysis'])
    for document, suffix in ((design, '-cost-design'), (raw, '-cost-observations'), (analysis, '-analysis')):
        _require((document['schema'], document['schema_version']) == (profile.prefix + suffix, 1),
            'frozen calibration artifact schema differs')
    analyses = analysis['analyses']
    context = model['resource_context']
    profile.require_workers(len(context['worker_cpu_ids']))
    current = _implementation()
    _require(model['schema_version'] == 1 and model['precision_policy'] == design['precision_policy'] == profile.precision_policy
        and model['implementation_files'] == current, 'precision policy or current producer implementation differs')
    if profile.parallel and 'qualification_design' in model:
        _verify_qualification(model, design, current)
    else:
        _require(design['implementation_files'] == current, 'physical calibration producer differs')
    _require(profile.tool_contract(context) == model['tool'] == design['tool'] == analysis['tool'],
        'MCA/OpenMP binary/CPU context differs')
    registry = profile.registry()
    configs = profile.configurations(context)
    _require(design['resource_context'] == context and design['registry'] == registry and design['configurations'] == configs
        and raw['frozen_design'] == model['frozen_design'] and raw['native'] == model['native'], 'fixed design/native differs')
    _require(model['target_workload_executed'] is False and model['target_variants_ranked'] is False
        and design['target_workload_executed'] is False and design['target_dag_provided_to_probe'] is False,
        'source-independent calibration claims differ')
    _verify_observations(raw['observations'], configs, design['repetitions'])
    native = model['native']
    _verify_native_source(native, context, profile)
    if profile.parallel:
        _require(raw['partition_witnesses'] == model['partition_witnesses']
            and model['serial_cost_values_reused'] is False and design['serial_duration_divided_by_workers'] is False,
            'independent parallel authority differs')
        _verify_partitions(model['partition_witnesses'], context)
        handle = ctypes.CDLL(native['library']['path'])
        _require(_loaded_openmp(handle) == model['tool']['openmp_runtime'], 'current probe OpenMP runtime differs')
    assembly = Path(native['assembly']['path']).read_text()
    lexical_evidence = _lexer_evidence(assembly) if profile.parallel else None
    instruction_text = _lex_assembly(assembly) if profile.parallel else assembly
    expected = {str(row['kernel_index']) for row in configs if row['role'] != 'DISPATCH'}
    _require(set(analyses) == expected, 'calibration analysis coverage differs')
    for key, value in analyses.items():
        case = registry[int(key) // 2]
        group = int(key) % 2 + 1
        if profile.parallel:
            _require(value['instruction_lexer'] == lexical_evidence, 'instruction lexical provenance differs')
        loops = _natural_loops(instruction_text, function=case['id'] + '_g' + str(group))
        _require(len(loops) == len(value['loops']), 'instruction loop set differs')
        for original, retained in zip(loops, value['loops']):
            source = Path(retained['loop_source']['path'])
            result_path = Path(retained['result']['path'])
            _require(_receipt(source) == retained['loop_source'] and source.read_text() == original['text']
                and _receipt(result_path) == retained['result'], 'MCA input/output receipt differs')
            result = json.loads(result_path.read_text())
            throughput = result['CodeRegions'][0]['SummaryView']['BlockRThroughput']
            _require(retained['cycles_per_element'] == throughput / original['width']
                and retained['block_throughput_cycles'] == throughput
                and result['TargetInfo']['CPUName'] == model['tool']['cpu_model'], 'MCA numerical summary differs')
    _require(profile.derive(raw['observations'], analyses) == model['coefficients']
        and _qualify(raw['observations'], analyses, model['coefficients'], profile) == model['local_models'],
        'coefficient/holdout replay differs')
    _require(_llc(context)[1] == model['cache_geometry'], 'cache evidence differs')
    return reference, model, analyses


def requalify_parallel_instruction_cpu(source_directory, out, *, correction_plan):
    """Reparse a frozen OpenMP probe, preserving the original physical measurements."""
    from scar.ir.execution_recipe import artifact_reference, read_artifact

    source_directory = Path(source_directory).resolve()
    out = Path(out).resolve()
    _require(not out.exists(), 'fresh requalification output required')
    design_ref = artifact_reference(source_directory / 'design.json', PARALLEL.prefix + '-cost-design')
    raw_path = source_directory / 'raw-observations.json'
    raw_ref = artifact_reference(raw_path, PARALLEL.prefix + '-cost-observations')
    design, raw = read_artifact(design_ref), read_artifact(raw_ref)
    repair_ref = artifact_reference(correction_plan, 'scar.parallel-instruction-lexer-correction-plan')
    repair = read_artifact(repair_ref)
    original = repair['original_producer']
    original_file = _validate_repair(repair, original, design)
    _require(_receipt(raw_path) == repair['raw_observations'], 'fixed original physical observations differ')
    architecture = _architecture_identity(original_file.read_text())
    context = design['resource_context']
    configs, registry = PARALLEL.configurations(context), PARALLEL.registry()
    _require(design['configurations'] == configs and design['registry'] == registry
        and design['precision_policy'] == PARALLEL.precision_policy and raw['frozen_design'] == design_ref,
        'original fixed design differs')
    native = raw['native']
    _verify_native_source(native, context, PARALLEL)
    _verify_partitions(raw['partition_witnesses'], context)
    tool = PARALLEL.tool_contract(context)
    _require(tool == design['tool'], 'physical runtime/tool context changed')
    current = _implementation()
    _require(_unchanged_support(design, current), 'shared implementation changed')
    out.mkdir(parents=True)
    started = time.time()
    assembly = Path(native['assembly']['path']).read_text()
    qualification = dict(schema='scar.parallel-instruction-requalification-design', schema_version=1,
        frozen_source_design=design_ref, source_observations=raw_ref, original_producer=original,
        correction_plan=repair_ref, implementation_files=current, precision_policy=PARALLEL.precision_policy,
        unchanged_architecture_ast_sha256=architecture, native_source_replayed_unchanged=True,
        lexical_rule=_lexer_evidence(assembly)['rule'], holdout_timing_or_error_values_inspected_before_correction=False,
        numerical_model_or_precision_policy_changed=False, new_physical_measurements_collected=False)
    path = out / 'qualification-design.json'
    _save(path, qualification)
    qualification_ref = artifact_reference(path, qualification['schema'])
    analysis_ref, analyses = _instruction_analyses(assembly, tool, configs, registry, PARALLEL, out)
    _verify_observations(raw['observations'], configs, design['repetitions'])
    model = _fitted_model(PARALLEL, design_ref, design, raw_ref, raw, analysis_ref, analyses, current)
    model.update(qualification_design=qualification_ref, calibration_seconds=None,
        qualification_replay_seconds=time.time() - started, new_physical_measurements_collected=False)
    _save(out / 'profile.json', model)
    return model


def _projection_stage(plan, term, node_order, source, capacity, profile):
    parameters = term['parameters']
    index = int(term['source_ref']['pointer'].rsplit('/', 1)[1])
    stage = plan['stages'][index]
    profile.require_workers(parameters['workers'])
    _require(stage['kind'] == 'COMPUTE', 'unsupported fused stage boundary')
    output = plan['assignment'][stage['output_payload']]['slot_id']
    inputs = [plan['assignment'][key]['slot_id'] for key in stage['input_payloads'] if key in plan['assignment']]
    _require(output not in inputs, 'output workspace aliases a live input slot')
    groups, rows, cols = parameters['backing_storage_groups'], parameters['rows'], parameters['cols']
    expected = V2._layouts(parameters['array_loads'], cols, groups)

    def layout_identity(row):
        return row['storage_group'], row['stride'], row['offset']

    _require(sorted(parameters['operand_layouts'], key=layout_identity) == sorted(expected, key=layout_identity),
        'uncalibrated operand memory geometry')
    _, geometry = V2.fused_features(parameters, capacity)
    _require(256 <= rows <= 768 and 320 <= cols <= 832 and geometry['backing_bytes'] + rows * cols * 8 <= capacity,
        'outside measured geometry/reported cache extent scope')
    span = _stage_source_range(source, node_order[stage['root_expr']])
    witness = dict(output_slot=output, input_slots=inputs, live_slots_distinct=True)
    return index, span, witness


def _projection_cost(term, analysis, model, profile, maximum_stack):
    _require(max(row['stack_memory_references'] for row in analysis['loops']) <= maximum_stack,
        'target stack-memory instruction pressure exceeds independent calibration scope')
    p = term['parameters']
    fitted = dict(groups=p['backing_storage_groups'], rows=p['rows'], cols=p['cols'], workers=p['workers'],
        elements=((p['rows'] + p['workers'] - 1) // p['workers']) * p['cols'], array_loads=p['array_loads'])
    return profile.predict(fitted, analysis, model['coefficients']), model['local_models'][str(fitted['groups'])]


def _projection_component(term, analysis, cost, qualification, profile, reference, lowering_ref, witness):
    group = str(term['parameters']['backing_storage_groups'])
    return dict(component=term['component'], parameters=term['parameters'], basis='DECLARED_COMPONENT_MODEL',
        median_ns=cost['duration_ns'], prediction=cost, source_model=dict(artifact=reference, locator='#/coefficients'),
        source_validation=dict(artifact=reference, locator='#/local_models/' + group + '/holdout'),
        source_provenance=term['source_ref'], instruction_analysis=analysis, static_lowering=lowering_ref,
        workspace_alias_authority=witness, precision_policy=profile.precision_policy,
        maximum_holdout_relative_error=qualification['maximum_holdout_relative_error'],
        target_dag_compiled_for_static_analysis=True, measured_target_component=False,
        physical_optimality_proven=False, assumptions=profile.assumptions)


def predict_instruction_components(domain, candidate_id, profile_path, out):
    from scar.ir.execution_recipe import artifact_reference
    from scar.optimization.stateful_cpu_model import _domain, stateful_cpu_demand

    domain = _domain(domain)
    demand = stateful_cpu_demand(domain, candidate_id)
    profile_document = json.loads(Path(profile_path).read_bytes())
    profile = INSTRUCTION_COST_SCHEMAS.get(profile_document.get('schema'))
    _require(profile is not None, 'unsupported instruction cost profile')
    reference, model, calibrated = _verify_model(profile_path, profile)
    context = demand['resource_context']
    _require(context == model['resource_context'], 'exact hardware/runtime/compiler/mask context differs')
    out = Path(out).resolve()
    _require(not out.exists(), 'fresh projection output required')
    out.mkdir(parents=True)
    region, plan, source, assembly, lowering = _target_static(domain, candidate_id, context, out / 'static')
    lowering_ref = artifact_reference(out / 'static/lowering.json', lowering['schema'])
    order = {node.id: index for index, node in enumerate(region.nodes)}
    capacity = model['cache_geometry']['reported_capacity_bytes']
    maximum_stack = max(row['stack_memory_references'] for entry in calibrated.values() for row in entry['loops'])
    components, unsupported, proposals = [], [], []
    for term in demand['terms']:
        if term['component'] != 'FUSED_SCALAR_DAG_ROW_STAGE':
            continue
        proposal = None
        try:
            index, span, witness = _projection_stage(plan, term, order, source, capacity, profile)
            _require(term['parameters']['workers'] == len(context['worker_cpu_ids']), 'exact target worker mask differs')
            analysis = profile.analyze(assembly, model['tool'], out / 'mca' / str(index), source_range=span,
                source_path=lowering['source']['path'])
            cost, qualification = _projection_cost(term, analysis, model, profile, maximum_stack)
            proposal = dict(component=term['component'], parameters=term['parameters'], prediction=cost,
                maximum_holdout_relative_error=qualification['maximum_holdout_relative_error'],
                precision_policy=profile.precision_policy, target_dag_compiled_for_static_analysis=True,
                measured_target_component=False, selectable=False)
            if profile.parallel:
                proposal['point_estimate_ns'] = cost['duration_ns']
            _require(qualification['admitted'], 'fresh holdout exceeds the frozen precision policy')
            components.append(_projection_component(term, analysis, cost, qualification, profile, reference,
                lowering_ref, witness))
        except (ValueError, StopIteration) as error:
            unsupported.append(dict(component=term['component'], parameters=term['parameters'], reason=str(error)))
            if proposal is not None:
                proposals.append(dict(**proposal, rejection_reason=str(error)))
    result = dict(schema='scar.stateful-resource-calibration', schema_version=1, resource_context=context,
        components=components, unsupported_components=unsupported, proposed_estimates=proposals, source_model=reference,
        static_lowering=lowering_ref, all_fused_components_covered=not unsupported, complete_method_cost_measured=False,
        target_dag_compiled_for_static_analysis=True, target_workload_executed=False, target_variants_ranked=False,
        physical_optimality_proven=False)
    _save(out / 'observations.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    calibrate = commands.add_parser('calibrate')
    calibrate.add_argument('--context', required=True)
    calibrate.add_argument('--out', required=True)
    calibrate.add_argument('--mode', choices=('serial', 'parallel'), default='serial')
    calibrate.add_argument('--repetitions', type=int, default=5)
    calibrate.add_argument('--warmup', type=int, default=2)
    calibrate.add_argument('--batch', type=int, default=8)
    project = commands.add_parser('project')
    project.add_argument('--domain', required=True)
    project.add_argument('--candidate-id', required=True)
    project.add_argument('--profile', required=True)
    project.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.command == 'project':
        result = predict_instruction_components(args.domain, args.candidate_id, args.profile, args.out)
        print(json.dumps(dict(components=len(result['components']), unsupported=result['unsupported_components']),
            sort_keys=True))
    else:
        model = calibrate_instruction_cpu(json.loads(Path(args.context).read_text()), args.out, mode=args.mode,
            repetitions=args.repetitions, warmup=args.warmup, batch=args.batch)
        coverage = {key: dict(admitted=value['admitted'],
            maximum_holdout_relative_error=value['maximum_holdout_relative_error'])
            for key, value in model['local_models'].items()}
        print(json.dumps(dict(profile=str(Path(args.out).resolve() / 'profile.json'), local_models=coverage),
            sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Explicit CPU-team lowering of the existing scalar temporal Expr program.

Only row ownership, placement and worker environment change. Mathematical
Expr emission remains owned by temporal_invariant_loop/cpp. Private output
is published only after all workers restore their environments and succeed.
"""
from dataclasses import dataclass
import ctypes
import json
import os
from pathlib import Path
import shlex
import subprocess

from scar.codegen import cpp
from scar.compiler import temporal_invariant_loop as serial
from scar.compiler.model import document,plan_from_dict,region_from_dict
from scar.optimization.hardware import bind_cpu_placement,validate_cpu_placement
from scar.optimization.resource_solver import semantic_sha256

PARTITION='CONTIGUOUS_ROWS_FIRST_REMAINDER_EXPLICIT_V1'


def describe_parallel_temporal_capability(resource_binding=None):
    """Inspect exact compiler/runtime context without compiling target code."""
    compiler=shlex.split(os.environ.get('CXX','g++'))
    identity=subprocess.run([*compiler,'--version'],capture_output=True,text=True,check=True).stdout.splitlines()[0]
    if resource_binding is not None:validate_cpu_placement(resource_binding)
    return dict(schema='scar.parallel-temporal-backend-capability',schema_version=1,
        family='CPP_OPENMP_EXPLICIT_ROW_PARTITION_TEMPORAL',cxx_standard=20,
        operator_descriptor_abi_version=1,temporal_control_abi_version=1,
        compiler=dict(argv=compiler,identity=identity,options=[*cpp.COMPILER_OPTIONS,
            '-fopenmp','-fno-unroll-loops','-fno-peel-loops']),
        generators=_sources(),row_partition=PARTITION,
        workers=None if resource_binding is None else len(resource_binding.cpu_ids),
        cpu_ids=None if resource_binding is None else list(resource_binding.cpu_ids),
        worker_count_bounds=[1,len(os.sched_getaffinity(0))],
        shape_profile='EQUAL_RANK2_FLOAT64_READONLY_INPUTS_ONE_FRESH_OUTPUT',
        per_element_order='INVARIANT_PREHEADER_THEN_EXACT_ORDERED_CARRY_RECURRENCE',
        team_policy='EXACT_TEAM_SIZE_BIND_WORKER_I_TO_CPU_IDS_I_SAVE_RESTORE_FENV_AND_AFFINITY',
        joins='ONE_TEAM_COMPLETION_NO_PER_ITERATION_TEAM_BARRIER',
        selected_for_use=False,performance_qualified=False,physical_optimality_proven=False)


def _sources():return [*_serial_sources(),serial._receipt(__file__)]
def _serial_sources():return serial._sources()


def _plan(body,carry,interface,budget,binding):
    validate_cpu_placement(binding)
    if not 1<=len(binding.cpu_ids)<=len(os.sched_getaffinity(0)):
        raise ValueError('parallel temporal CPU team exceeds current affinity capacity')
    plan,execution=serial._plan(body,carry,interface,budget,None)
    execution=bind_cpu_placement(execution,binding)
    plan.update(schema='scar.parallel-temporal-invariant-loop-plan',execution_plan=document(execution),
        cpu_threads=len(binding.cpu_ids),row_partition=PARTITION,
        row_partition_formula=dict(begin='tid*(rows//W)+min(tid,rows%W)',
            end='begin+rows//W+(tid<rows%W)'),
        worker_affinity='EXACT_PLANNED_CPU_PER_WORKER_SAVE_BIND_VERIFY_RESTORE',
        worker_floating_environment='COPY_HELD_CALLER_FENV_THEN_RESTORE_EACH_PREVIOUS_WORKER_FENV')
    plan.pop('id',None);plan['id']='parallel-temporal-invariant-loop:'+semantic_sha256(plan)
    return plan,execution


def _replace(source,old,new):
    if source.count(old)!=1:raise ValueError('parallel source anchor differs from unchanged serial emitter')
    return source.replace(old,new,1)


def _source(plan,execution):
    raw,counts=serial._source(plan,execution);source=raw.decode();workers=execution.cpu_threads
    source='#ifndef _GNU_SOURCE\n#define _GNU_SOURCE\n#endif\n#include <omp.h>\n'+source
    source=_replace(source,'int32_t temporal_run(const scar_array_v1 *inputs, uint64_t count, uint64_t steps,\n'
        '                    scar_array_v1 *output, uint64_t *counters) {',
        'int32_t temporal_run(const scar_array_v1 *inputs, uint64_t count, uint64_t steps,\n'
        '                    scar_array_v1 *output, uint64_t *counters, uint64_t *observations, uint64_t observation_count) {')
    source=_replace(source,'    counters[0]=0; counters[1]=0;',
        f'    counters[0]=0; counters[1]=0;\n    if (observations && observation_count != {workers*8}) return SCAR_ARGUMENT_GUARD_V1;')
    source=_replace(source,'    if (!affinity.bind(planned_cpu_ids[0])) return SCAR_CPU_AFFINITY_GUARD_V1;\n','')
    begin=source.index('    for (int64_t row=0; row<output->shape[0]; ++row) {')
    end=source.index('\n    const int raised=',begin)
    rows=source[begin:end].replace('for (int64_t row=0; row<output->shape[0]; ++row)',
        'for (int64_t row=row_begin; row<row_end; ++row)').replace('++counters[0]','++preheaders').replace('++counters[1]','++iterations')
    parallel=f'''    fenv_t execution_environment;
    if (std::fegetenv(&execution_environment)!=0) {{ std::fesetenv(&saved); return SCAR_FLOATING_GUARD_V1; }}
    uint32_t worker_guard=0,placement_guard=0;
    uint64_t preheaders=0,iterations=0;
    #pragma omp parallel num_threads({workers}) reduction(|:nonfinite_input,worker_guard,placement_guard) reduction(+:preheaders,iterations)
    {{
        const int tid=omp_get_thread_num();
        const bool team_ready=omp_get_num_threads()=={workers};
        ThreadAffinity worker_affinity;
        const bool affinity_ready=team_ready && worker_affinity.bind(planned_cpu_ids[tid]);
        const int actual_cpu=affinity_ready ? sched_getcpu() : -1;
        fenv_t worker_saved;
        const bool saved_ok=std::fegetenv(&worker_saved)==0;
        const bool environment_ready=saved_ok && std::fesetenv(&execution_environment)==0;
        const int64_t base=output->shape[0]/{workers}, remainder=output->shape[0]%{workers};
        const int64_t row_begin=tid*base+(tid<remainder ? tid : remainder);
        const int64_t row_end=row_begin+base+(tid<remainder);
        if (affinity_ready && environment_ready) {{
{rows}
        }}
        const int worker_raised=environment_ready ? std::fetestexcept(FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW|FE_UNDERFLOW) : 0;
        const bool environment_restored=saved_ok && std::fesetenv(&worker_saved)==0;
        const bool kept=worker_affinity.bound();
        const bool restored=worker_affinity.restore();
        worker_guard |= static_cast<uint32_t>(!environment_ready || worker_raised || !environment_restored);
        placement_guard |= static_cast<uint32_t>(!affinity_ready || !kept || !restored);
        if (observations && tid<{workers}) {{
            uint64_t *record=observations+8*tid;
            record[0]=1; record[1]=planned_cpu_ids[tid]; record[2]=static_cast<uint64_t>(actual_cpu);
            record[3]=kept; record[4]=restored; record[5]=environment_restored;
            record[6]=row_begin; record[7]=row_end;
        }}
    }}
    counters[0]=preheaders; counters[1]=iterations;'''
    source=source[:begin]+parallel+source[end:]
    source=_replace(source,'    const bool kept = affinity.bound();\n    const bool affinity_restored = affinity.restore();\n    if (!kept || !affinity_restored) return SCAR_CPU_AFFINITY_GUARD_V1;',
        '    const bool affinity_restored = affinity.restore_original();\n    if (placement_guard || !affinity_restored) return SCAR_CPU_AFFINITY_GUARD_V1;')
    source=_replace(source,'return raised || restored || nonfinite_input ? SCAR_FLOATING_GUARD_V1 : SCAR_OK_V1;',
        'return raised || restored || nonfinite_input || worker_guard ? SCAR_FLOATING_GUARD_V1 : SCAR_OK_V1;')
    source=_replace(source,'return temporal_run(inputs,count,steps,output,counters);',
        'return temporal_run(inputs,count,steps,output,counters,nullptr,0);')
    source+='''\nextern "C" SCAR_EXPORT_V1 int32_t scar_parallel_temporal_observe_v1(
    const scar_array_v1 *inputs, uint64_t count, uint64_t steps, scar_array_v1 *output,
    uint64_t *counters, uint64_t *observations, uint64_t observation_count) {
    return temporal_run(inputs,count,steps,output,counters,observations,observation_count);
}\n'''
    return source.encode(),counts


@dataclass
class ParallelTemporalInvariantArtifact:
    plan: dict
    directory: Path
    library: object
    run_native: object
    observe_native: object
    report: dict

    def run(self,inputs,count,original_loop=None):
        observations=(ctypes.c_uint64*(self.plan['cpu_threads']*8))()
        def invoke(*args):return self.observe_native(*args,observations,len(observations))
        facade=serial.TemporalInvariantLoopArtifact(self.plan,self.directory,self.library,invoke,self.report)
        result=facade.run(inputs,count,original_loop=original_loop)
        self.report['last_execution']['workers']=[dict(worker=i,planned_cpu=int(observations[i*8+1]),
            observed_cpu=int(observations[i*8+2]),binding_kept=bool(observations[i*8+3]),
            original_affinity_restored=bool(observations[i*8+4]),original_fenv_restored=bool(observations[i*8+5]),
            row_begin=int(observations[i*8+6]),row_end=int(observations[i*8+7]))
            for i in range(self.plan['cpu_threads']) if observations[i*8]]
        return result


def build_parallel_temporal_invariant_loop(body,out,*,carry_parameter,loop_interface,memory_budget_bytes,resource_binding):
    plan,execution=_plan(body,carry_parameter,loop_interface,memory_budget_bytes,resource_binding)
    if serial._receipt(plan['body']['source']['path'])['sha256']!=plan['body']['source']['sha256']:
        raise ValueError('parallel temporal original body source changed')
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh parallel temporal output required')
    source,counts=_source(plan,execution);cap=describe_parallel_temporal_capability(resource_binding)
    out.mkdir(parents=True);(out/'plan.json').write_bytes(serial._json(plan));(out/'temporal.cpp').write_bytes(source)
    header=Path(cpp.__file__).with_name('scar_abi_v1.h');(out/header.name).write_bytes(header.read_bytes())
    command=[*cap['compiler']['argv'],*cap['compiler']['options'],str(out/'temporal.cpp'),'-o',str(out/'temporal.so')]
    process=subprocess.run(command,capture_output=True,text=True)
    if process.returncode:raise RuntimeError('parallel temporal compilation failed:\n'+process.stderr)
    report=dict(schema='scar.native-parallel-temporal-invariant-loop',schema_version=1,plan_id=plan['id'],
        cxx_standard=20,operator_descriptor_abi_version=1,temporal_control_abi_version=1,
        body_sha256=plan['body_sha256'],loop_interface=loop_interface,
        source=serial._receipt(out/'temporal.cpp'),library=serial._receipt(out/'temporal.so'),header=serial._receipt(out/header.name),
        generators=_sources(),compiler=cap['compiler'],capability=cap,
        emitted_arithmetic_per_element=counts,full_invariant_buffers=0,carry_pingpong_buffers=0,final_output_buffers=1,
        physical_register_allocation_proven=False,cpu_threads=execution.cpu_threads,cpu_ids=list(resource_binding.cpu_ids),
        source_replacement_authorized=False,last_execution=None,selected_for_use=False,
        physical_optimality_proven=False,cost_status='UNCALIBRATED')
    (out/'artifact.json').write_bytes(serial._json(report));return load_parallel_temporal_invariant_loop(out)


def load_parallel_temporal_invariant_loop(directory):
    directory=Path(directory).resolve();plan=json.loads((directory/'plan.json').read_bytes())
    report=json.loads((directory/'artifact.json').read_bytes())
    if (report.get('schema'),report.get('schema_version'))!=('scar.native-parallel-temporal-invariant-loop',1):
        raise ValueError('versioned parallel temporal artifact required')
    execution=plan_from_dict(plan['execution_plan'])
    expected,actual=_plan(region_from_dict(plan['body']),plan['carry_parameter'],plan['loop_interface'],
        plan['memory_budget_bytes'],execution.resource_binding)
    if semantic_sha256(expected)!=semantic_sha256(plan) or plan['id']!=report['plan_id']:
        raise ValueError('parallel temporal dependency/resource plan replay differs')
    if serial._receipt(plan['body']['source']['path'])['sha256']!=plan['body']['source']['sha256']:
        raise ValueError('parallel temporal original body source changed')
    if _sources()!=report['generators']:raise ValueError('parallel temporal generator receipts changed')
    for key,name in (('source','temporal.cpp'),('library','temporal.so'),('header','scar_abi_v1.h')):
        if serial._receipt(directory/name)!=report[key]:raise ValueError('parallel temporal artifact receipt changed')
    raw,_=_source(plan,actual)
    if raw!=(directory/'temporal.cpp').read_bytes():raise ValueError('parallel temporal native routing replay differs')
    library=ctypes.CDLL(str(directory/'temporal.so'));run=library.scar_temporal_invariant_loop_v1
    args=[ctypes.POINTER(cpp._Array),ctypes.c_uint64,ctypes.c_uint64,
        ctypes.POINTER(cpp._Array),ctypes.POINTER(ctypes.c_uint64)]
    run.argtypes=args;run.restype=ctypes.c_int32
    observed=library.scar_parallel_temporal_observe_v1
    observed.argtypes=[*args,ctypes.POINTER(ctypes.c_uint64),ctypes.c_uint64];observed.restype=ctypes.c_int32
    return ParallelTemporalInvariantArtifact(plan,directory,library,run,observed,report)


__all__=['describe_parallel_temporal_capability','build_parallel_temporal_invariant_loop',
    'load_parallel_temporal_invariant_loop','ParallelTemporalInvariantArtifact']

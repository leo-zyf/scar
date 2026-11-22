"""C++20 lowering for serial and OpenMP temporal loops.

Scalar locals express reuse across iterations; actual register assignment and
spills remain the C++ compiler's responsibility. This conditional lowering
requires the same original caller F4/runtime admission as invariant_loop.
"""
from dataclasses import asdict,dataclass
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
import subprocess

from scar.codegen import cpp
from scar.analysis.array_motion import derive_array_motion,verify_array_motion
from scar.compiler.model import document,plan_from_dict,region_from_dict
from scar.compiler.pipeline_frontend import prepare_pipeline_leaf
from scar.compiler.invariant_loop import plan_invariant_loop
from scar.optimization.optimizer import optimize_region
from scar.optimization.hardware import bind_cpu_placement,validate_cpu_placement
from scar.optimization.resource_solver import semantic_sha256


_SERIAL_PLAN = 'scar.temporal-invariant-loop-plan'
_OPENMP_PLAN = 'scar.parallel-temporal-invariant-loop-plan'
_SERIAL_ARTIFACT = 'scar.native-temporal-invariant-loop'
_OPENMP_ARTIFACT = 'scar.native-parallel-temporal-invariant-loop'
_ROW_PARTITION = 'CONTIGUOUS_ROWS_FIRST_REMAINDER_EXPLICIT_V1'


def _json(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def _receipt(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return dict(path=str(path),sha256=sha256(raw).hexdigest(),bytes=len(raw))
def _sources():
    return [_receipt(p) for p in (Path(__file__),Path(cpp.__file__),
        Path(cpp.__file__).with_name('kernel_v1.cpp.in'),Path(cpp.__file__).with_name('scar_abi_v1.h'),
        Path(__file__).with_name('invariant_loop.py'),
        Path(__file__).with_name('loop_admission.py'),
        Path(__file__).resolve().parents[1]/'analysis'/'array_motion.py',
        Path(__file__).resolve().parents[1]/'analysis'/'array_motion_legality.py',
        Path(__file__).resolve().parents[1]/'analysis'/'array_call_effects.py',
        Path(__file__).resolve().parents[1]/'analysis'/'composite_call_effect_binding.py')]


def _compiler(parallel):
    argv = shlex.split(os.environ.get('CXX', 'g++'))
    identity = subprocess.run([*argv, '--version'], capture_output=True, text=True,
                              check=True).stdout.splitlines()[0]
    options = list(cpp.COMPILER_OPTIONS)
    if parallel:
        options.append('-fopenmp')
    options.extend(('-fno-unroll-loops', '-fno-peel-loops'))
    return dict(argv=argv, identity=identity, options=options)


def describe_parallel_temporal_capability(resource_binding=None):
    """Inspect the compiler and worker contract without building a target."""
    if resource_binding is not None:
        validate_cpu_placement(resource_binding)
    return dict(schema='scar.parallel-temporal-backend-capability', schema_version=1,
        family='CPP_OPENMP_EXPLICIT_ROW_PARTITION_TEMPORAL', cxx_standard=20,
        operator_descriptor_abi_version=1, temporal_control_abi_version=1,
        compiler=_compiler(parallel=True), generators=_sources(), row_partition=_ROW_PARTITION,
        workers=None if resource_binding is None else len(resource_binding.cpu_ids),
        cpu_ids=None if resource_binding is None else list(resource_binding.cpu_ids),
        worker_count_bounds=[1,len(os.sched_getaffinity(0))],
        shape_profile='EQUAL_RANK2_FLOAT64_READONLY_INPUTS_ONE_FRESH_OUTPUT',
        per_element_order='INVARIANT_PREHEADER_THEN_EXACT_ORDERED_CARRY_RECURRENCE',
        team_policy='EXACT_TEAM_SIZE_BIND_WORKER_I_TO_CPU_IDS_I_SAVE_RESTORE_FENV_AND_AFFINITY',
        joins='ONE_TEAM_COMPLETION_NO_PER_ITERATION_TEAM_BARRIER',
        selected_for_use=False, performance_qualified=False, physical_optimality_proven=False)


def uses_openmp(plan):
    schema = plan.get('schema')
    if plan.get('schema_version') != 1 or schema not in (_SERIAL_PLAN, _OPENMP_PLAN):
        raise ValueError('versioned C++20 temporal lowering plan required')
    return schema == _OPENMP_PLAN


def _execution_plan(body,budget,binding):
    execution=prepare_pipeline_leaf(optimize_region(body,memory_budget_bytes=budget))
    scalar = cpp._supported(execution)
    if any(scalar[node.id] and node.op != 'const' for node in execution.region.nodes):
        raise ValueError('temporal lowering requires a separate guarded scalar setup before the count loop')
    if binding is not None:
        validate_cpu_placement(binding)
        execution=bind_cpu_placement(execution,binding)
    return execution


def _source_movement_gate(body,interface):
    """Consume a source certificate; its original F4 replay remains external."""
    from scar.analysis.array_motion_legality import verify_array_motion_legality
    from scar.ir.execution_recipe import read_artifact,_reference
    reference=_reference(interface['source_motion']);motion=read_artifact(reference)
    family=(motion.get('schema'),motion.get('schema_version'))
    families={('scar.source-loop-motion',2):('original_f4_proof','source-loop-motion:'),
        ('scar.source-composite-loop-motion',1):('original_composite_f4_proof','source-composite-loop-motion:')}
    if (type(motion.get('schema'))is not str or type(motion.get('schema_version'))is not int
            or family not in families):
        raise ValueError('current source-motion movement certificate required')
    proof_key,prefix=families[family]
    if {key for key in ('original_f4_proof','original_composite_f4_proof')if key in interface}!={proof_key}:
        raise ValueError('distinct original direct or composite source proof required')
    if motion.get('id')!=prefix+semantic_sha256({key:value for key,value in motion.items()if key!='id'}):
        raise ValueError('source-motion content identity differs')
    original=interface[proof_key];proof=read_artifact(motion['proof']);query={};extra={}
    if proof_key=='original_composite_f4_proof':
        from scar.compiler.pipeline_loop_frontend import _read_composite_invariant_loop_materials,COMPOSITE_PROOF_SCHEMA
        from scar.analysis.array_call_effects import verify_composite_call_effects
        if motion['proof']['schema']!=COMPOSITE_PROOF_SCHEMA:
            raise ValueError('composite source movement requires its separate original proof receipt')
        typed_proof,proof_ref=_read_composite_invariant_loop_materials(motion['proof']['uri'])
        if proof_ref!=motion['proof']:
            raise ValueError('composite original proof receipt changed during material replay')
        proof=asdict(typed_proof)
        effects=verify_composite_call_effects(motion['callee_effects'],typed_proof)
        query=dict(callee_effects=effects,callee_binding=motion['callee_binding'])
        extra=dict(callee_effects_sha256=semantic_sha256(effects),
            callee_binding_sha256=semantic_sha256(motion['callee_binding']))
    else:
        proof={key:value for key,value in proof.items()if key not in ('schema','schema_version')}
    canonical=dict(carry_parameter=interface['carry_parameter'],
        readonly_parameters=[name for name in body.parameters if name!=interface['carry_parameter']],
        **{proof_key:original})
    if (original is None or semantic_sha256(proof)!=semantic_sha256(original)
            or motion['proof_id']!=proof['id'] or motion['proof_sha256']!=semantic_sha256(proof)
            or motion['body_sha256']!=semantic_sha256(asdict(body))
            or motion['canonical_interface_sha256']!=semantic_sha256(canonical)):
        raise ValueError('source-motion proof/body/canonical interface differs from native input')
    legality=verify_array_motion_legality(motion['movement_legality'],body,
        carry_parameter=interface['carry_parameter'],loop_interface=canonical,source_window=motion['source_window'],**query)
    roots={row['expression']:row for row in legality['roots']}
    if (legality['status']!='CONDITIONAL'
            or legality['array_motion_sha256']!=semantic_sha256(motion['array_motion'])
            or legality['source_window_effects']['status']!='CONDITIONAL_NO_INPUT_CLOBBERS'
            or len(roots)!=len(legality['roots']) or set(roots)!=set(motion['array_motion']['invariant_roots'])
            or not roots or any(row['status']!='CONDITIONAL' for row in roots.values())
            or legality['source_window_effects']['blockers']
            or any(row[field]['status']!='CONDITIONAL' or row[field]['blockers'] for row in roots.values()
                for field in ('data_available','no_clobber','execution','effects'))):
        raise ValueError('source movement has unresolved roots or source-window blockers')
    stable=legality['stable_formal_inputs']
    if (type(stable)is not list or len(set(stable))!=len(stable)
            or set(stable)-set(canonical['readonly_parameters'])):
        raise ValueError('source movement stable formal input set differs from the original interface')
    return dict(source_motion=reference,source_motion_id=motion['id'],
        source_motion_sha256=semantic_sha256(motion),movement_legality_sha256=semantic_sha256(legality),
        source_window_sha256=semantic_sha256(motion['source_window']),
        canonical_interface_sha256=motion['canonical_interface_sha256'],source_body_sha256=motion['body_sha256'],
        source_invariant_roots=sorted(roots),stable_formal_inputs=stable,
        source_window_replay_required=True,source_replacement_authorized=False,**extra)


def _execution_movement_binding(body,execution,interface,budget,motion):
    if 'source_motion'not in interface:return None
    admitted=_source_movement_gate(body,interface)
    expected=_execution_plan(body,budget,execution.resource_binding)
    if semantic_sha256(document(expected))!=semantic_sha256(document(execution)):
        raise ValueError('movement binding requires exact existing optimized execution replay')
    stable=set(admitted['stable_formal_inputs']);placements={}
    for node in execution.region.nodes:
        placement=motion['placements'][node.id]['temporal_placement']
        dependencies=motion['dependencies'][node.id]
        if placement=='PRE_TEMPORAL_LOOP' and not set(dependencies)<=stable:
            raise ValueError('optimized preheader expression lacks source-window stable input evidence: '+node.id)
        placements[node.id]=dict(placement=placement,formal_dependencies=dependencies)
    return dict(schema='scar.temporal-source-movement-binding',schema_version=1,
        **admitted,optimized_body_sha256=semantic_sha256(asdict(execution.region)),
        execution_motion_sha256=semantic_sha256(motion),node_placements=placements,
        optimized_node_ids_are_source_root_ids=False,legality='CONDITIONAL_ON_REPLAYED_SOURCE_WINDOW_AND_ORIGINAL_RUNTIME_Q',
        runtime_guards_are_not_must_execute_proof=True,source_F4_replayed_here=False)


def _plan(body,carry_parameter,loop_interface,memory_budget_bytes,resource_binding):
    base=plan_invariant_loop(body,carry_parameter=carry_parameter,loop_interface=loop_interface,
        memory_budget_bytes=memory_budget_bytes)
    original=region_from_dict(base['body'])
    execution=_execution_plan(original,memory_budget_bytes,resource_binding)
    if resource_binding is not None:
        if len(resource_binding.cpu_ids)!=1:raise ValueError('temporal invariant profile supports exactly one explicitly bound CPU')
    # Existing optimization can fold/CSE nodes and change their IDs. Its exact
    # body needs its own report; the source body's report remains in base.
    execution_motion=derive_array_motion(execution.region,carry_parameter=carry_parameter,
        loop_interface=loop_interface)
    base.update(schema=_SERIAL_PLAN,execution_plan=document(execution),
        execution_motion_analysis=execution_motion,
        execution_movement_binding=_execution_movement_binding(original,execution,loop_interface,memory_budget_bytes,execution_motion),
        representation='ONE_SHARED_SCALAR_DAG_PER_ELEMENT_WITH_NATIVE_COUNT_LOOP',
        physical_array_buffers=dict(final_output=1,full_invariant_buffers=0,carry_pingpong_buffers=0),
        memory_scope='ONE_FRESH_FINAL_INCREMENTAL_ARRAY_PAYLOAD_EXCLUDES_CALLER_INPUTS_AND_COMPILER_STACK',
        scalar_storage='CXX_LOCAL_VALUES_COMPILER_MAY_SPILL_NO_HARDWARE_REGISTER_CLAIM',
        logical_iteration_count='RUNTIME_EXACT_POSITIVE_INT64_COUNT_NOT_UNROLLED',
        scalar_and_element_orders='ORIGINAL_DAG_OPERAND_ORDER_EACH_ELEMENT_ITERATION_ORDER',
        cross_element_order='REORDERED_ONLY_UNDER_NO_INPUT_WRITES_OR_INTERMEDIATE_OBSERVERS_Q',
        cpu_threads=1,placement_optimized=False)
    base.pop('id',None);base['id']='temporal-invariant-loop:'+semantic_sha256(base)
    return base,execution


def _parallel_plan(body, carry, interface, budget, binding):
    validate_cpu_placement(binding)
    if not 1 <= len(binding.cpu_ids) <= len(os.sched_getaffinity(0)):
        raise ValueError('parallel temporal CPU team exceeds current affinity capacity')
    plan, execution = _plan(body, carry, interface, budget, None)
    execution = bind_cpu_placement(execution, binding)
    plan.update(schema=_OPENMP_PLAN, execution_plan=document(execution),
        cpu_threads=len(binding.cpu_ids), row_partition=_ROW_PARTITION,
        row_partition_formula=dict(begin='tid*(rows//W)+min(tid,rows%W)',
            end='begin+rows//W+(tid<rows%W)'),
        worker_affinity='EXACT_PLANNED_CPU_PER_WORKER_SAVE_BIND_VERIFY_RESTORE',
        worker_floating_environment='COPY_HELD_CALLER_FENV_THEN_RESTORE_EACH_PREVIOUS_WORKER_FENV')
    plan.pop('id')
    plan['id'] = 'parallel-temporal-invariant-loop:' + semantic_sha256(plan)
    return plan, execution


def _source(plan,execution):
    # Keep the existing descriptor, pointer-bound, finite-load and optional
    # affinity implementations. Only the loop nesting/Expr placement changes.
    motion=verify_array_motion(plan['execution_motion_analysis'],execution.region,
        carry_parameter=plan['carry_parameter'],loop_interface=plan['loop_interface'])
    movement=_execution_movement_binding(region_from_dict(plan['body']),execution,
        plan['loop_interface'],plan['memory_budget_bytes'],motion)
    if movement!=plan['execution_movement_binding']:
        raise ValueError('native expression placement differs from source movement binding replay')
    support=cpp._source(execution,cpp._supported(execution)).split('int32_t run(',1)[0]
    body=execution.region;names={};pre=[];varying=[]
    carry=plan['carry_parameter'];invariant_ops=0;variant_ops=0
    for i,node in enumerate(body.nodes):
        placement=(motion['placements'][node.id]['temporal_placement']if movement is None else
            movement['node_placements'][node.id]['placement'])
        if placement=='ITERATION_CARRY':
            names[node.id]='carry';continue
        name=names[node.id]='value'+str(i)
        if node.op=='input':
            index=body.parameters.index(node.attrs['name'])
            expression=f'read_value(static_cast<const unsigned char *>(inputs[{index}].data) + row * static_cast<ptrdiff_t>(inputs[{index}].strides[0]) + col * static_cast<ptrdiff_t>(inputs[{index}].strides[1]), nonfinite_input)'
        elif node.op=='const':expression=float(node.attrs['value']).hex()
        elif node.op=='neg':expression='-'+names[node.inputs[0]]
        elif node.op=='copy':expression=names[node.inputs[0]]
        else:expression=f'{names[node.inputs[0]]} {cpp._BINARY_OPS[node.op]} {names[node.inputs[1]]}'
        line=f'            const double {name} = {expression};'
        if placement=='PRE_TEMPORAL_LOOP':
            pre.append(line)
            invariant_ops+=int(node.op not in ('input','const'))
        elif placement=='ITERATION_BODY':
            varying.append('    '+line)
            variant_ops+=1
        else:
            raise ValueError('temporal emission requires a supported unified node placement: '+node.id)
    source=body.source
    carry_index=body.parameters.index(carry)
    affinity_start=affinity_end=''
    if execution.resource_binding is not None:
        affinity_start='''    ThreadAffinity affinity;
    for (int cpu : planned_cpu_ids) if (!affinity.contains(cpu)) return SCAR_CPU_AFFINITY_GUARD_V1;
    if (!affinity.bind(planned_cpu_ids[0])) return SCAR_CPU_AFFINITY_GUARD_V1;
'''
        affinity_end='''    const bool kept = affinity.bound();
    const bool affinity_restored = affinity.restore();
    if (!kept || !affinity_restored) return SCAR_CPU_AFFINITY_GUARD_V1;
'''
    run=f'''
bool finite_bits(double value) {{
    return (std::bit_cast<uint64_t>(value) & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}}
int32_t temporal_run(const scar_array_v1 *inputs, uint64_t count, uint64_t steps,
                    scar_array_v1 *output, uint64_t *counters) {{
    if (!inputs || !output || !counters || count != {len(body.parameters)} || !steps || steps > INT64_MAX) return SCAR_ARGUMENT_GUARD_V1;
    counters[0]=0; counters[1]=0;
    for (uint64_t i=0; i<count; ++i) if (!valid_array(inputs[i])) return SCAR_ARGUMENT_GUARD_V1;
    if (!valid_array(*output)) return SCAR_ARGUMENT_GUARD_V1;
    for (uint64_t i=0; i<count; ++i) {{
        if (inputs[i].shape[0]!=output->shape[0] || inputs[i].shape[1]!=output->shape[1]) return SCAR_SHAPE_GUARD_V1;
        if (output->lower_bound<inputs[i].upper_bound && inputs[i].lower_bound<output->upper_bound) return SCAR_ARGUMENT_GUARD_V1;
    }}
    const __int128 bytes=__int128(output->shape[0])*output->shape[1]*8;
    if (bytes>{plan['memory_budget_bytes']}ULL || bytes>PTRDIFF_MAX ||
        (bytes && (output->strides[1]!=8 || output->strides[0]!=__int128(output->shape[1])*8))) return SCAR_ARGUMENT_GUARD_V1;
    if (__int128(output->shape[0])*output->shape[1]*steps>UINT64_MAX) return SCAR_ARGUMENT_GUARD_V1;
{affinity_start}    fenv_t saved;
    if (std::feholdexcept(&saved)!=0) return SCAR_FLOATING_GUARD_V1;
    uint32_t nonfinite_input=0;
    for (int64_t row=0; row<output->shape[0]; ++row) {{
        for (int64_t col=0; col<output->shape[1]; ++col) {{
            double carry=read_value(static_cast<const unsigned char *>(inputs[{carry_index}].data) +
                row*static_cast<ptrdiff_t>(inputs[{carry_index}].strides[0]) + col*static_cast<ptrdiff_t>(inputs[{carry_index}].strides[1]), nonfinite_input);
{chr(10).join(pre)}
            ++counters[0];
            if (nonfinite_input) continue;
            #pragma GCC unroll 1
            for (uint64_t iteration=0; iteration<steps; ++iteration) {{
{chr(10).join(varying)}
                carry={names[body.output]};
                ++counters[1];
                if (!finite_bits(carry)) {{ nonfinite_input|=1; break; }}
            }}
            static_cast<double *>(output->data)[row*output->shape[1]+col]=carry;
        }}
    }}
    const int raised=std::fetestexcept(FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW|FE_UNDERFLOW);
    const int restored=std::fesetenv(&saved);
{affinity_end}    return raised || restored || nonfinite_input ? SCAR_FLOATING_GUARD_V1 : SCAR_OK_V1;
}}
}} // original helper namespace
extern "C" SCAR_EXPORT_V1 int32_t scar_temporal_invariant_loop_v1(
    const scar_array_v1 *inputs, uint64_t count, uint64_t steps, scar_array_v1 *output, uint64_t *counters) {{
    return temporal_run(inputs,count,steps,output,counters);
}}
'''
    return (support+run).encode(),dict(invariant_ops=invariant_ops,variant_ops=variant_ops)


def _replace(source,old,new):
    if source.count(old)!=1:raise ValueError('parallel source anchor differs from unchanged serial emitter')
    return source.replace(old,new,1)


def _parallel_source(plan,execution):
    raw,counts=_source(plan,execution);source=raw.decode();workers=execution.cpu_threads
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
class TemporalInvariantLoopArtifact:
    plan: dict
    directory: Path
    library: object
    run_native: object
    report: dict

    def run(self,inputs,trip_count,original_loop=None):
        return self._run(self.run_native, inputs, trip_count, original_loop)

    def _run(self, invoke, inputs, trip_count, original_loop):
        import numpy as np
        if type(trip_count) is not int or not -(2**63)<=trip_count<2**63:
            raise ValueError('exact signed int64 loop count required')
        if trip_count<=0:
            self.report['last_execution']=dict(native_calls=0,element_preheaders=0,element_iterations=0,
                logical_iterations=0,status=0,output_alias='ORIGINAL_CARRY')
            return inputs[self.plan['carry_parameter']]
        counters=(ctypes.c_uint64*2)()
        try:
            parameters=self.plan['body']['parameters']
            if type(inputs) is not dict or set(inputs)!=set(parameters):raise cpp.GuardFailure('original body input mapping differs')
            arrays=[inputs[p] for p in parameters];descriptors=[cpp._descriptor(a) for a in arrays]
            shape=arrays[0].shape
            if any(a.shape!=shape for a in arrays):raise cpp.GuardFailure('temporal LICM requires equal original array shapes')
            size=shape[0]*shape[1]
            if size*8>self.plan['memory_budget_bytes'] or size*trip_count>=2**64:
                raise cpp.GuardFailure('temporal LICM output budget or observation counter extent exceeded')
            output=np.empty(shape,dtype=np.float64);descriptor=cpp._descriptor(output)
            status=invoke((cpp._Array*len(descriptors))(*descriptors),len(descriptors),trip_count,
                ctypes.byref(descriptor),counters)
            self.report['last_execution']=dict(native_calls=1,element_preheaders=counters[0],
                element_iterations=counters[1],logical_iterations=trip_count,status=status)
            if status:raise cpp.GuardFailure('temporal LICM private work rejected: '+str(status))
            return output
        except (cpp.GuardFailure,MemoryError) as error:
            self.report['last_execution']=dict(native_calls=int(bool(counters[0])),element_preheaders=counters[0],
                element_iterations=counters[1],logical_iterations=trip_count,
                fallback=original_loop is not None,reason=str(error))
            if original_loop is None:raise
            return original_loop()


@dataclass
class ParallelTemporalInvariantArtifact(TemporalInvariantLoopArtifact):
    observe_native: object

    def run(self, inputs, count, original_loop=None):
        workers = self.plan['cpu_threads']
        observations = (ctypes.c_uint64 * (workers * 8))()

        def invoke(*args):
            return self.observe_native(*args, observations, len(observations))

        result = self._run(invoke, inputs, count, original_loop)
        self.report['last_execution']['workers'] = [
            dict(worker=i, planned_cpu=int(observations[i*8+1]), observed_cpu=int(observations[i*8+2]),
                 binding_kept=bool(observations[i*8+3]), original_affinity_restored=bool(observations[i*8+4]),
                 original_fenv_restored=bool(observations[i*8+5]), row_begin=int(observations[i*8+6]),
                 row_end=int(observations[i*8+7]))
            for i in range(workers) if observations[i*8]
        ]
        return result


def build_temporal_invariant_loop(body,out,*,carry_parameter,loop_interface,memory_budget_bytes,resource_binding=None):
    plan,_=_plan(body,carry_parameter,loop_interface,memory_budget_bytes,resource_binding)
    return emit_temporal_invariant_loop(plan,out)


def build_parallel_temporal_invariant_loop(body,out,*,carry_parameter,loop_interface,memory_budget_bytes,resource_binding):
    plan,_=_parallel_plan(body,carry_parameter,loop_interface,memory_budget_bytes,resource_binding)
    return emit_temporal_invariant_loop(plan,out)


_ARTIFACT_FILES = dict(source='temporal.cpp', library='temporal.so', header='scar_abi_v1.h')


def _source_receipt(plan):
    source = plan['body']['source']
    if _receipt(source['path'])['sha256'] != source['sha256']:
        raise ValueError('temporal LICM original source changed')


def emit_temporal_invariant_loop(plan, out):
    """Build the selected C++20 temporal operator from its published lowering."""
    parallel = uses_openmp(plan)
    execution = plan_from_dict(plan['execution_plan'])
    binding = execution.resource_binding
    if parallel and binding is None:
        raise ValueError('OpenMP temporal lowering requires a scheduled CPU team')
    _source_receipt(plan)
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('fresh temporal LICM output required')
    render = _parallel_source if parallel else _source
    raw, counts = render(plan, execution)
    capability = describe_parallel_temporal_capability(binding) if parallel else None
    compiler = capability['compiler'] if parallel else _compiler(parallel=False)
    generators = capability['generators'] if parallel else _sources()
    paths = {key: out / name for key, name in _ARTIFACT_FILES.items()}
    out.mkdir(parents=True)
    (out / 'plan.json').write_bytes(_json(plan))
    paths['source'].write_bytes(raw)
    header = Path(cpp.__file__).with_name(_ARTIFACT_FILES['header'])
    paths['header'].write_bytes(header.read_bytes())
    command = [*compiler['argv'], *compiler['options'], str(paths['source']), '-o', str(paths['library'])]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('temporal LICM compilation failed:\n' + result.stderr)
    report = dict(schema=_OPENMP_ARTIFACT if parallel else _SERIAL_ARTIFACT, schema_version=1, plan_id=plan['id'],
        cxx_standard=20, operator_descriptor_abi_version=1, temporal_control_abi_version=1,
        body_sha256=plan['body_sha256'], loop_interface=plan['loop_interface'],
        **{key: _receipt(path) for key, path in paths.items()}, generators=generators, compiler=compiler,
        emitted_arithmetic_per_element=counts, full_invariant_buffers=0, carry_pingpong_buffers=0,
        final_output_buffers=1, physical_register_allocation_proven=False, cpu_threads=execution.cpu_threads,
        source_replacement_authorized=False, last_execution=None, selected_for_use=False,
        physical_optimality_proven=False, cost_status='UNCALIBRATED')
    if parallel:
        report.update(capability=capability, cpu_ids=list(binding.cpu_ids))
    else:
        report['cpu_placement'] = 'UNBOUND' if binding is None else 'EXPLICIT_SINGLE_CPU_NOT_OPTIMIZED'
    (out / 'artifact.json').write_bytes(_json(report))
    return load_temporal_invariant_loop(out)


def load_temporal_invariant_loop(directory):
    directory = Path(directory).resolve()
    plan = json.loads((directory / 'plan.json').read_bytes())
    report = json.loads((directory / 'artifact.json').read_bytes())
    parallel = uses_openmp(plan)
    expected_schema = _OPENMP_ARTIFACT if parallel else _SERIAL_ARTIFACT
    if (report.get('schema'), report.get('schema_version')) != (expected_schema, 1):
        raise ValueError('temporal artifact schema does not match its C++20 lowering')
    execution = plan_from_dict(plan['execution_plan'])
    replay = _parallel_plan if parallel else _plan
    expected, actual = replay(region_from_dict(plan['body']), plan['carry_parameter'], plan['loop_interface'],
                              plan['memory_budget_bytes'], execution.resource_binding)
    if semantic_sha256(expected) != semantic_sha256(plan) or plan['id'] != report['plan_id']:
        raise ValueError('temporal LICM plan differs from dependency/execution replay')
    _source_receipt(plan)
    if _sources() != report['generators']:
        raise ValueError('temporal LICM generator receipt changed')
    for key, name in _ARTIFACT_FILES.items():
        if _receipt(directory / name) != report[key]:
            raise ValueError('temporal LICM artifact changed: ' + key)
    render = _parallel_source if parallel else _source
    raw, _ = render(plan, actual)
    if raw != (directory / _ARTIFACT_FILES['source']).read_bytes():
        raise ValueError('temporal LICM native source routing changed')
    library = ctypes.CDLL(str(directory / _ARTIFACT_FILES['library']))
    run = library.scar_temporal_invariant_loop_v1
    args = [ctypes.POINTER(cpp._Array), ctypes.c_uint64, ctypes.c_uint64,
            ctypes.POINTER(cpp._Array), ctypes.POINTER(ctypes.c_uint64)]
    run.argtypes = args
    run.restype = ctypes.c_int32
    if not parallel:
        return TemporalInvariantLoopArtifact(plan, directory, library, run, report)
    observe = library.scar_parallel_temporal_observe_v1
    observe.argtypes = [*args, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64]
    observe.restype = ctypes.c_int32
    return ParallelTemporalInvariantArtifact(plan, directory, library, run, report, observe)


__all__=['build_temporal_invariant_loop','build_parallel_temporal_invariant_loop',
    'describe_parallel_temporal_capability','emit_temporal_invariant_loop', 'load_temporal_invariant_loop',
    'ParallelTemporalInvariantArtifact','TemporalInvariantLoopArtifact','uses_openmp']

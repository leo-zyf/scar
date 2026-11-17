"""Per-element temporal LICM over the existing strict CPU scalar Expr profile.

Scalar locals express reuse across iterations; actual register assignment and
spills remain the C++ compiler's responsibility. This conditional lowering
requires the same original caller F4/runtime admission as invariant_loop.
"""
from dataclasses import dataclass
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
from scar.compiler.pipeline_model import prepare_pipeline_leaf
from scar.compiler.invariant_loop import plan_invariant_loop
from scar.optimization.optimizer import optimize_region
from scar.optimization.hardware import bind_cpu_placement,validate_cpu_placement
from scar.optimization.resource_solver import semantic_sha256


def _json(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def _receipt(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return dict(path=str(path),sha256=sha256(raw).hexdigest(),bytes=len(raw))
def _sources():
    return [_receipt(p) for p in (Path(__file__),Path(cpp.__file__),
        Path(cpp.__file__).with_name('kernel_v1.cpp.in'),Path(cpp.__file__).with_name('scar_abi_v1.h'),
        Path(__file__).with_name('invariant_loop.py'),
        Path(__file__).resolve().parents[1]/'analysis'/'array_motion.py')]


def _plan(body,carry_parameter,loop_interface,memory_budget_bytes,resource_binding):
    base=plan_invariant_loop(body,carry_parameter=carry_parameter,loop_interface=loop_interface,
        memory_budget_bytes=memory_budget_bytes)
    original=region_from_dict(base['body'])
    execution=prepare_pipeline_leaf(optimize_region(original,memory_budget_bytes=memory_budget_bytes))
    if resource_binding is not None:
        validate_cpu_placement(resource_binding)
        if len(resource_binding.cpu_ids)!=1:raise ValueError('temporal invariant profile supports exactly one explicitly bound CPU')
        execution=bind_cpu_placement(execution,resource_binding)
    # Existing optimization can fold/CSE nodes and change their IDs. Its exact
    # body needs its own report; the source body's report remains in base.
    execution_motion=derive_array_motion(execution.region,carry_parameter=carry_parameter,
        loop_interface=loop_interface)
    base.update(schema='scar.temporal-invariant-loop-plan',execution_plan=document(execution),
        execution_motion_analysis=execution_motion,
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


def _source(plan,execution):
    # Keep the existing descriptor, pointer-bound, finite-load and optional
    # affinity implementations. Only the loop nesting/Expr placement changes.
    motion=verify_array_motion(plan['execution_motion_analysis'],execution.region,
        carry_parameter=plan['carry_parameter'],loop_interface=plan['loop_interface'])
    support=cpp._source(execution,cpp._supported(execution)).split('int32_t run(',1)[0]
    body=execution.region;names={};pre=[];varying=[]
    carry=plan['carry_parameter'];invariant_ops=0;variant_ops=0
    for i,node in enumerate(body.nodes):
        placement=motion['placements'][node.id]['temporal_placement']
        if placement=='ITERATION_CARRY':
            names[node.id]='carry';continue
        name=names[node.id]='value'+str(i)
        if node.op=='input':
            index=body.parameters.index(node.attrs['name'])
            expression=f'read_value(static_cast<const unsigned char *>(inputs[{index}].data) + row * static_cast<ptrdiff_t>(inputs[{index}].strides[0]) + col * static_cast<ptrdiff_t>(inputs[{index}].strides[1]), nonfinite_input)'
        elif node.op=='const':expression=float(node.attrs['value']).hex()
        elif node.op=='neg':expression='-'+names[node.inputs[0]]
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


@dataclass
class TemporalInvariantLoopArtifact:
    plan: dict
    directory: Path
    library: object
    run_native: object
    report: dict

    def run(self,inputs,trip_count,original_loop=None):
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
            status=self.run_native((cpp._Array*len(descriptors))(*descriptors),len(descriptors),trip_count,
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


def build_temporal_invariant_loop(body,out,*,carry_parameter,loop_interface,memory_budget_bytes,resource_binding=None):
    plan,execution=_plan(body,carry_parameter,loop_interface,memory_budget_bytes,resource_binding)
    source=plan['body']['source']
    if _receipt(source['path'])['sha256']!=source['sha256']:raise ValueError('original body source changed')
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh temporal LICM output required')
    raw,counts=_source(plan,execution);out.mkdir(parents=True)
    (out/'plan.json').write_bytes(_json(plan));(out/'temporal.cpp').write_bytes(raw)
    header=Path(cpp.__file__).with_name('scar_abi_v1.h');(out/header.name).write_bytes(header.read_bytes())
    compiler=shlex.split(os.environ.get('CXX','g++'))
    identity=subprocess.run([*compiler,'--version'],capture_output=True,text=True,check=True).stdout.splitlines()[0]
    options=[*cpp.COMPILER_OPTIONS,'-fno-unroll-loops','-fno-peel-loops']
    command=[*compiler,*options,str(out/'temporal.cpp'),'-o',str(out/'temporal.so')]
    result=subprocess.run(command,capture_output=True,text=True)
    if result.returncode:raise RuntimeError('temporal LICM compilation failed:\n'+result.stderr)
    report=dict(schema='scar.native-temporal-invariant-loop',schema_version=1,plan_id=plan['id'],
        cxx_standard=20,operator_descriptor_abi_version=1,temporal_control_abi_version=1,
        body_sha256=plan['body_sha256'],loop_interface=loop_interface,
        source=_receipt(out/'temporal.cpp'),library=_receipt(out/'temporal.so'),header=_receipt(out/header.name),
        generators=_sources(),compiler=dict(argv=compiler,identity=identity,options=options),
        emitted_arithmetic_per_element=counts,full_invariant_buffers=0,carry_pingpong_buffers=0,
        final_output_buffers=1,physical_register_allocation_proven=False,
        cpu_threads=1,cpu_placement='UNBOUND' if resource_binding is None else 'EXPLICIT_SINGLE_CPU_NOT_OPTIMIZED',
        source_replacement_authorized=False,last_execution=None,selected_for_use=False,
        physical_optimality_proven=False,cost_status='UNCALIBRATED')
    (out/'artifact.json').write_bytes(_json(report))
    return load_temporal_invariant_loop(out)


def load_temporal_invariant_loop(directory):
    directory=Path(directory).resolve();plan=json.loads((directory/'plan.json').read_bytes())
    report=json.loads((directory/'artifact.json').read_bytes())
    if (report.get('schema'),report.get('schema_version'))!=('scar.native-temporal-invariant-loop',1):
        raise ValueError('versioned temporal invariant artifact required')
    execution=plan_from_dict(plan['execution_plan'])
    expected,actual=_plan(region_from_dict(plan['body']),plan['carry_parameter'],plan['loop_interface'],
        plan['memory_budget_bytes'],execution.resource_binding)
    if semantic_sha256(expected)!=semantic_sha256(plan) or plan['id']!=report['plan_id']:
        raise ValueError('temporal LICM plan differs from dependency/execution replay')
    if _receipt(plan['body']['source']['path'])['sha256']!=plan['body']['source']['sha256']:
        raise ValueError('temporal LICM original source changed')
    if _sources()!=report['generators']:raise ValueError('temporal LICM generator receipt changed')
    for key,name in (('source','temporal.cpp'),('library','temporal.so'),('header','scar_abi_v1.h')):
        if _receipt(directory/name)!=report[key]:raise ValueError('temporal LICM artifact changed: '+key)
    raw,_=_source(plan,actual)
    if raw!=(directory/'temporal.cpp').read_bytes():raise ValueError('temporal LICM native source routing changed')
    library=ctypes.CDLL(str(directory/'temporal.so'));run=library.scar_temporal_invariant_loop_v1
    run.argtypes=[ctypes.POINTER(cpp._Array),ctypes.c_uint64,ctypes.c_uint64,
        ctypes.POINTER(cpp._Array),ctypes.POINTER(ctypes.c_uint64)]
    run.restype=ctypes.c_int32
    return TemporalInvariantLoopArtifact(plan,directory,library,run,report)


__all__=['build_temporal_invariant_loop','load_temporal_invariant_loop','TemporalInvariantLoopArtifact']

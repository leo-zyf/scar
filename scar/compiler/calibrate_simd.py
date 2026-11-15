"""Measure independent canonical SSE2 primitives, never a target-program variant."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import statistics
import subprocess

from scar.codegen.cpp import COMPILER_OPTIONS
from scar.compiler.model import read_plan
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import encode
from scar.optimization.hardware import validate_cpu_placement


def simd_compiler_options(plan):
    return [*COMPILER_OPTIONS, '-fno-unroll-loops', '-fno-peel-loops',
            '-msse2', '-mno-avx', '-mno-fma', *(['-fopenmp'] if plan.cpu_threads > 1 else [])]


def _source():
    functions = []
    names = ('add', 'sub', 'mul', 'div', 'neg', 'finite_mask')
    for name in names:
        for chains in (1, 8):
            variables = [f'v{i}' for i in range(chains)]
            declarations = '\n'.join(f'    __m128d {v} = _mm_set_pd(0.75+{i}*0.001,0.5+{i}*0.001);'
                                     for i, v in enumerate(variables))
            operations = []
            for value in variables:
                if name == 'neg': statement = f'{value} = _mm_xor_pd({value}, sign);'
                elif name == 'finite_mask':
                    statement = (f'mask |= _mm_movemask_pd(_mm_castsi128_pd(_mm_cmpeq_epi32('
                        f'_mm_and_si128(_mm_castpd_si128({value}),_mm_castpd_si128(exponent)),'
                        '_mm_castpd_si128(exponent))));')
                else: statement = f'{value} = _mm_{name}_pd({value}, operand);'
                operations.append(f'        {statement}\n        asm volatile("" : "+x"({value}));')
            stores = '\n'.join(f'    _mm_storeu_pd(sink+{2*i}, {v});' for i, v in enumerate(variables))
            functions.append(f'''__attribute__((noinline)) double {name}_{chains}(uint64_t count) {{
{declarations}
    const __m128d operand = _mm_set1_pd({'0.000000000001' if name in ('add','sub') else '1.000000000001'});
    const __m128d sign = _mm_set1_pd(-0.0);
    const __m128d exponent = _mm_castsi128_pd(_mm_set1_epi64x(INT64_C(0x7ff0000000000000)));
    int mask = 0;
    const auto started = std::chrono::steady_clock::now();
    #pragma GCC unroll 1
    for (uint64_t i=0;i<count;++i) {{
{chr(10).join(operations)}
    }}
    const auto stopped = std::chrono::steady_clock::now();
{stores}
    sink[31] = static_cast<double>(mask);
    return std::chrono::duration<double,std::nano>(stopped-started).count() / (count*{chains});
}}''')
    samples = []
    for name in names:
        for chains in (1, 8):
            samples.append(f'''    for (unsigned r=0;r<warmup;++r) {name}_{chains}(count);
    std::cout << "{name} {chains}";
    for (unsigned r=0;r<repetitions;++r) std::cout << ' ' << {name}_{chains}(count);
    std::cout << '\\n';''')
    return '''#include <emmintrin.h>
#include <cpuid.h>
#include <chrono>
#include <cfenv>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sched.h>
static_assert(sizeof(double)==8);
alignas(16) double sink[32];
''' + '\n'.join(functions) + '''
int main(int argc,char **argv) {
    if (argc!=5) return 2;
    const uint64_t count=std::strtoull(argv[1],nullptr,10);
    const unsigned repetitions=std::strtoul(argv[2],nullptr,10), warmup=std::strtoul(argv[3],nullptr,10);
    const int cpu=std::atoi(argv[4]);
    unsigned a=0,b=0,c=0,d=0;
    if (!__get_cpuid(1,&a,&b,&c,&d) || !(d & bit_SSE2) || sched_getcpu()!=cpu) return 3;
    fenv_t saved;
    if (std::feholdexcept(&saved)!=0) return 4;
    std::cout << std::setprecision(17) << "CPUID_SSE2 1 CPU " << cpu << '\\n';
''' + '\n'.join(samples) + '''
    if (sched_getcpu()!=cpu || std::fesetenv(&saved)!=0) return 5;
    return sink[31]!=0 ? 6 : 0;
}
'''


def calibrate_simd(plan_path, out, *, iterations=2000000, repetitions=5, warmup=1):
    plan = read_plan(plan_path)
    if plan.resource_binding is None: raise ValueError('recorded CPU allocation required')
    validate_cpu_placement(plan.resource_binding)
    if platform.machine().lower() not in ('x86_64','amd64'):
        raise ValueError('SSE2 layout calibration requires x86_64')
    if iterations < 10000 or repetitions < 3 or warmup < 1:
        raise ValueError('canonical calibration needs >=10000 iterations, >=3 repeats and >=1 warmup')
    out = Path(out).absolute()
    if out.exists(): raise ValueError('fresh primitive calibration directory required')
    out.mkdir(parents=True)
    cpu = plan.resource_binding.cpu_ids[0]
    if cpu not in os.sched_getaffinity(0): raise ValueError('measured CPU allocation unavailable')
    compiler = shlex.split(os.environ.get('CXX','g++'))
    if not compiler or not shutil.which(compiler[0]) or not shutil.which('taskset'):
        raise ValueError('C++ compiler and taskset required')
    identity = subprocess.check_output([*compiler,'--version'],text=True).splitlines()[0]
    options = simd_compiler_options(plan)
    calibration_options = [option for option in options if option not in ('-shared','-fopenmp')]
    source = out/'primitives.cpp'; source.write_text(_source())
    library = out/'primitives'
    subprocess.run([*compiler,*calibration_options,str(source),'-o',str(library),'-lm'],check=True,
                   capture_output=True,text=True)
    process = subprocess.run(['taskset','-c',str(cpu),str(library),str(iterations),
        str(repetitions),str(warmup),str(cpu)],check=True,capture_output=True,text=True)
    (out/'samples.txt').write_text(process.stdout)
    lines=process.stdout.splitlines()
    if lines[0] != 'CPUID_SSE2 1 CPU '+str(cpu): raise ValueError('native CPU/ISA evidence mismatch')
    measurements={}; samples={}
    for line in lines[1:]:
        name,chains,*values=line.split(); values=[float(value) for value in values]
        if len(values)!=repetitions or any(value<=0 for value in values):
            raise ValueError('invalid canonical timing samples')
        key='dependent_ns_per_vector' if chains=='1' else 'independent_ns_per_vector'
        samples.setdefault(name,{})[key]=dict(samples_ns_per_vector=values,
            median=statistics.median(values),min=min(values),max=max(values),
            stdev=statistics.stdev(values),independent_chains=int(chains))
        measurements.setdefault(name,{})[key]=statistics.median(values)
    cpuinfo=Path('/proc/cpuinfo').read_bytes(); (out/'cpuinfo.txt').write_bytes(cpuinfo)
    model=next(line.split(':',1)[1].strip() for line in cpuinfo.decode().splitlines() if line.startswith('model name'))
    context=dict(target_path=plan.resource_binding.target_path,target_sha256=plan.resource_binding.target_sha256,
        executor_id=plan.resource_binding.executor_id,cpu_ids=list(plan.resource_binding.cpu_ids),
        measured_cpu_id=cpu,cpu_model=model,compiler=dict(argv=compiler,identity=identity,options=options))
    isa=dict(architecture='x86_64',name='SSE2_X86_64',lanes=2,vector_registers=16,supported=True,
        support_authority='NATIVE_CPUID_ON_PINNED_CPU_AND_X86_64_ABI',
        cpuinfo_sha256=sha256(cpuinfo).hexdigest(),register_count_authority='X86_64_SSE2_ARCHITECTURAL_XMM_REGISTER_FILE')
    register_model=dict(reserved_vector_registers=2,max_vector_chains=8,
        scope='DECLARED_CODEGEN_REGISTER_BUDGET_NOT_AN_ACTUAL_COMPILER_SPILL_PROOF')
    observations=dict(schema='scar.cpu-simd-primitive-observations',schema_version=1,context=context,isa=isa,
        register_model=register_model,measurements=dict(ops={k:v for k,v in measurements.items() if k!='finite_mask'},
            finite_mask=dict(ns_per_vector=measurements['finite_mask']['independent_ns_per_vector'])),
        samples=samples,iterations=iterations,repetitions=repetitions,warmup=warmup,
        measurement_scope='EFFECTIVE_CANONICAL_PRIMITIVES_WITH_LOOP_AND_ASM_COMPILER_BARRIERS_NOT_PURE_INSTRUCTION_LATENCIES',
        finite_mask_implementation='SSE2_INTEGER_EXPONENT_AND_CMPEQ32_MOVEMASK_PD',
        calibration_options=calibration_options,link_options_differ_from_native_context=True,
        target_variants_executed=False,whole_program_duration_predicted=False,
        source_sha256=sha256(source.read_bytes()).hexdigest(),binary_sha256=sha256(library.read_bytes()).hexdigest())
    path=out/'observations.json'; path.write_text(json.dumps(observations,indent=2,allow_nan=False)+'\n')
    raw=path.read_bytes(); artifact=ProviderArtifact(str(path),sha256(raw).hexdigest(),len(raw),observations['schema'])
    def evidence(pointer): return [encode(ProviderFactReference(artifact,pointer))]
    profile=dict(schema='scar.cpu-simd-primitives',schema_version=1,context=dict(context,evidence=evidence('#/context')),
        isa=dict(isa,evidence=evidence('#/isa')),register_model=dict(register_model,evidence=evidence('#/register_model')),
        measurements=dict(ops={op:{name:dict(value=value,unit='ns/vector',evidence=evidence('#/measurements/ops/'+op+'/'+name))
            for name,value in row.items()} for op,row in observations['measurements']['ops'].items()},
            finite_mask=dict(ns_per_vector=dict(value=observations['measurements']['finite_mask']['ns_per_vector'],
                unit='ns/vector',evidence=evidence('#/measurements/finite_mask/ns_per_vector')))),
        model_scope='EFFECTIVE_CANONICAL_DEPENDENCY_THROUGHPUT_PROXY',
        whole_program_duration_predicted=False,physical_optimality_proven=False,
        profile_generator_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'profile.json').write_text(json.dumps(profile,indent=2,allow_nan=False)+'\n')
    return profile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--iterations',type=int,default=2000000)
    parser.add_argument('--repetitions',type=int,default=5);parser.add_argument('--warmup',type=int,default=1)
    args=parser.parse_args();profile=calibrate_simd(args.plan,args.out,iterations=args.iterations,
        repetitions=args.repetitions,warmup=args.warmup)
    print(json.dumps(dict(status='CALIBRATED_CANONICAL_PRIMITIVES',profile=str(Path(args.out)/'profile.json'),
        target_variants_executed=False,operations=list(profile['measurements']['ops']))))


if __name__=='__main__': main()

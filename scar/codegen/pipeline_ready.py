"""Emit dependency-ready control around unchanged, CPU-bound ABI1 operators."""
from pathlib import Path

from scar.compiler.pipeline_frontend import pipeline_digest
from scar.optimization.resource_solver import semantic_sha256

READY_FAMILY='DEPENDENCY_READY_CPU_SERIAL_LEAVES_V1'
_DIRECTORY=Path(__file__).resolve().parent


def render_ready_source(serial_source,pipeline,memory,report):
    marker='extern "C" SCAR_EXPORT_V1 const char *scar_pipeline_plan_sha256_v1()'
    if serial_source.count(marker)!=1:raise ValueError('original ABI1 plan/descriptor helpers required')
    prefix=serial_source.split(marker,1)[0]
    kernel=(_DIRECTORY/'kernel_v1.cpp.in').read_text()
    affinity='class ThreadAffinity {'+kernel.split('class ThreadAffinity {',1)[1].split('};\n#endif',1)[0]+'};'
    wave=(_DIRECTORY/'pipeline_parallel_v1.cpp.in').read_text()
    worker=wave[wave.index('// Save the complete environment'):wave.index('int32_t run_waves(')]
    dependencies=['    {{'+','.join('1'if other.id in report['call_predecessors'][call.id]else'0'
        for other in pipeline.calls)+'}},'for call in pipeline.calls]
    values=dict(PLAN_AND_DESCRIPTOR_HELPERS=prefix,AFFINITY_IMPLEMENTATION=affinity,
        WORKER_ENVIRONMENT_AND_ABI_CALL=worker,CPU_SLOTS=str(max(report['cpu_ids'])+1),
        CALL_COUNT=str(len(pipeline.calls)),LANE_COUNT=str(len(report['cpu_ids'])),
        LANE_CPUS=','.join(map(str,report['cpu_ids'])),
        CALL_CPUS=','.join(str(report['call_cpu_ids'][call.id])for call in pipeline.calls),
        CALL_PREDECESSORS='\n'.join(dependencies),PIPELINE_SHA256=pipeline_digest(pipeline),
        MEMORY_SHA256=semantic_sha256(memory),EXECUTION_SCHEDULE_SHA256=semantic_sha256(report))
    source=(_DIRECTORY/'pipeline_ready_v1.cpp.in').read_text()
    for name,value in values.items():source=source.replace('@'+name+'@',value)
    if '@'in source:raise ValueError('unresolved dependency-ready host template')
    return source


__all__=['READY_FAMILY','render_ready_source']

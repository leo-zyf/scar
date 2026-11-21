"""Render replayed CPU resource-event dependencies as C++20 control tables."""

from pathlib import Path

from scar.compiler.pipeline_model import pipeline_digest
from scar.optimization.resource_solver import semantic_sha256


RESOURCE_EVENTS_FAMILY = 'RESOURCE_EVENTS_CPU_SERIAL_LEAVES_V1'
DIAGNOSTIC_CAPACITY_PER_EVENT = 6
DIAGNOSTIC_KINDS = (
    'WORKSPACE_ALLOCATED', 'FINAL_STORAGE_ADOPTED', 'SLOT_ACQUIRED',
    'COMPUTE_START', 'COMPUTE_SUCCESS', 'COMPUTE_FAILURE', 'VALUE_READY',
    'SLOT_RETIRED', 'SUCCESS_FENCE', 'WORKSPACE_RELEASED', 'FINAL_PUBLISHED',
)
_DIRECTORY = Path(__file__).resolve().parent


def _flatten(groups):
    offsets, flat = [0], []
    for row in groups:
        flat.extend(row)
        offsets.append(len(flat))
    return offsets, flat


def render_resource_events_source(serial_source, pipeline, memory, report):
    marker = 'extern "C" SCAR_EXPORT_V1 const char *scar_pipeline_plan_sha256_v1()'
    if serial_source.count(marker) != 1 or report.get('execution_family') != RESOURCE_EVENTS_FAMILY:
        raise ValueError('ABI1 plan helpers and a resource-event schedule required')
    prefix = serial_source.split(marker, 1)[0]
    kernel = (_DIRECTORY / 'kernel_v1.cpp.in').read_text()
    affinity = 'class ThreadAffinity {' + kernel.split('class ThreadAffinity {', 1)[1].split('};\n#endif', 1)[0] + '};'
    wave = (_DIRECTORY / 'pipeline_parallel_v1.cpp.in').read_text()
    floating = wave[wave.index('// Save the complete environment'):wave.index('int32_t execute_task(')]
    rows, cpus = report['resource_events'], report['cpu_ids']
    events = {row['id']: index for index, row in enumerate(rows)}
    calls = {call.id: index for index, call in enumerate(pipeline.calls)}
    slots = {slot['id']: index for index, slot in enumerate(memory['workspace_slots'])}
    slots['final:output'] = len(slots)
    if len(events) != len(rows) or len(cpus) != len(set(cpus)) or not cpus:
        raise ValueError('unique event IDs and explicit distinct CPU lanes required')
    kinds = {'ALLOCATE': 'allocate', 'COMPUTE': 'compute', 'SUCCESS_FENCE': 'fence',
             'RELEASE': 'release', 'EXPORT': 'export_value'}
    declarations, successors, controls = [], [[] for _ in rows], []
    computes, allocate_slots, release_slots = {}, [], []
    for index, row in enumerate(rows):
        kind, dependencies = row['kind'], row['depends_on']
        if kind not in kinds or len(dependencies) != len(set(dependencies)) or any(
                name not in events or events[name] >= index for name in dependencies):
            raise ValueError('unsupported event or non-topological event dependency')
        for name in dependencies:
            successors[events[name]].append(index)
        subject = 0
        if kind == 'COMPUTE':
            call = row['call_id']
            if call not in calls or call in computes or row['cpu_id'] != report['call_cpu_ids'][call]:
                raise ValueError('resource compute and original call/CPU differ')
            computes[call] = index
            subject = calls[call]
        else:
            controls.append(index)
            if kind in ('ALLOCATE', 'RELEASE'):
                subject = slots[row['slot_id']]
                (allocate_slots if kind == 'ALLOCATE' else release_slots).append(subject)
        declarations.append(f'    {{EventKind::{kinds[kind]}, {subject}, {len(dependencies)}}},')
    if (set(computes) != set(calls) or sorted(allocate_slots) != list(range(len(slots)))
            or sorted(release_slots) != list(range(len(slots) - 1))
            or sum(row['kind'] == 'SUCCESS_FENCE' for row in rows) != 1
            or sum(row['kind'] == 'EXPORT' for row in rows) != 1):
        raise ValueError('event schedule must cover all original calls, slots and the final publication')
    lanes = [report['lane_call_order'][str(cpu)] for cpu in cpus]
    lane_offsets, lane_calls = _flatten(lanes)
    if (len(lane_calls) != len(calls) or set(lane_calls) != set(calls)
            or any(report['call_cpu_ids'][call] != cpu for cpu, lane in zip(cpus, lanes) for call in lane)):
        raise ValueError('each original call must belong to its certified lane exactly once')
    successor_offsets, successor_ids = _flatten(successors)

    def numbers(group):
        return ', '.join(map(str, group))

    replacements = dict(PLAN_AND_DESCRIPTOR_HELPERS=prefix, AFFINITY_IMPLEMENTATION=affinity,
        FLOATING_ENVIRONMENT=floating, CPU_SLOTS=str(max(cpus) + 1), LANE_COUNT=str(len(cpus)),
        LANE_CPUS=numbers(cpus), LANE_OFFSETS=numbers(lane_offsets),
        LANE_EVENT_IDS=numbers(computes[call] for call in lane_calls),
        EVENT_COUNT=str(len(rows)), EVENT_PLANS='\n'.join(declarations),
        SUCCESSOR_OFFSETS=numbers(successor_offsets), SUCCESSOR_COUNT=str(len(successor_ids)),
        SUCCESSOR_IDS=numbers(successor_ids), CONTROL_COUNT=str(len(controls)), CONTROL_IDS=numbers(controls),
        DIAGNOSTIC_CAPACITY_PER_EVENT=str(DIAGNOSTIC_CAPACITY_PER_EVENT),
        PIPELINE_SHA256=pipeline_digest(pipeline), MEMORY_SHA256=semantic_sha256(memory),
        EXECUTION_SCHEDULE_SHA256=semantic_sha256(report))
    source = (_DIRECTORY / 'pipeline_resource_events_v1.cpp.in').read_text()
    for name, value in replacements.items():
        source = source.replace('@' + name + '@', value)
    if '@' in source:
        raise ValueError('unresolved resource-event host template field')
    return source

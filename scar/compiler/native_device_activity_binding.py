"""Bind hybrid runtime sample indices to exact, archived physical activities."""

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import PurePosixPath
import re

from scar.codegen.cuda_copy_schedule import _HOST_ATEN, render_cuda_hybrid_schedule
from scar.ir.implementation_plan import PlanningProblem, problem_document, problem_from_dict
from scar.ir.program import codec


SCHEMA = 'scar.native-device-activity-binding'
_MEMBERS = {'program', 'problem', 'schedule', 'bindings', 'executable', 'launcher'}
_DIGEST = re.compile(r'[0-9a-f]{64}')
_COPY = 'CUDA_MEMCPY_ASYNC'


def _qualified_refs(references):
    if type(references) is not dict or set(references) != _MEMBERS:
        raise ValueError('activity calibration needs locks for every native plan and launch member')
    for row in references.values():
        if (type(row) is not dict or set(row) != {'path', 'sha256', 'size_bytes'}
                or type(row['path']) is not str or type(row['sha256']) is not str
                or _DIGEST.fullmatch(row['sha256']) is None
                or type(row['size_bytes']) is not int or row['size_bytes'] <= 0):
            raise ValueError('activity calibration member is missing its archive content identity')
        path = PurePosixPath(row['path'])
        if not path.parts or path.is_absolute() or '..' in path.parts or path.as_posix() != row['path']:
            raise ValueError('activity calibration references must be canonical archive-relative paths')
    if len({row['path'] for row in references.values()}) != len(_MEMBERS):
        raise ValueError('activity calibration member roles must identify distinct files')
    return {name: dict(row) for name, row in references.items()}


def _check_encoded(references, program, problem, schedule, bindings):
    expected = {'program': codec.dumps(program), **{role: json.dumps(value, indent=2, sort_keys=True)
                for role, value in (('problem', problem), ('schedule', schedule), ('bindings', bindings))}}
    for role, content in expected.items():
        raw = (content + '\n').encode()
        lock = references[role]
        if (len(raw), sha256(raw).hexdigest()) != (lock['size_bytes'], lock['sha256']):
            raise ValueError(f'activity calibration {role} differs from its archive content lock')


def _source_ids(operation, adjacent, reference, program_ids):
    result = set()
    for task in (operation, *adjacent):
        for row in task.source_refs:
            if 'operation' in row:
                name = row['operation']
                if name not in program_ids or any(row.get(key) != value for key, value in reference.items()):
                    raise ValueError('activity source operation differs from the archived typed program')
                result.add(name)
    if not adjacent and len(result) != 1:
        raise ValueError('timed compute requires exactly one original program operation')
    return sorted(result)


def _materials(mode, incoming, outgoing):
    inputs = [dict(edge=edge.id, material=edge.value, producer=edge.producer,
                   nbytes=edge.nbytes, memory=mode.input_locations[edge.value]) for edge in incoming]
    outputs = {}
    for edge in outgoing:
        row = outputs.setdefault(edge.value, dict(material=edge.value, nbytes=edge.nbytes,
                                                 memory=mode.output_locations[edge.value], consumers=[]))
        row['consumers'].append(dict(edge=edge.id, operation=edge.consumer))
    return inputs, list(outputs.values())


def derive_activity_binding(program, problem, schedule, bindings, *, references):
    """Derive timings' activity identities from a verified hybrid C++ lowering.

    The publisher/verifier checks the locked binary and launcher bytes. This
    pure projection regenerates the plan-side bindings and uses the renderer's
    exact step, queue, and wait order. It carries no measured times.
    """
    refs = _qualified_refs(references)
    if type(problem) is PlanningProblem:
        typed_problem, wire = problem, problem_document(problem)
    else:
        typed_problem, wire = problem_from_dict(problem), problem
    _check_encoded(refs, program, wire, schedule, bindings)
    if ('host_resource' not in bindings or not typed_problem.normal_completion_only
            or typed_problem.source_refs != (refs['program'],)):
        raise ValueError('activity calibration requires a source-locked, normal-completion hybrid plan')
    program_ids = {node.id.wire for node in program.operations}
    lowered = render_cuda_hybrid_schedule(typed_problem, schedule,
        device=bindings['device'], memory_kinds=bindings['memory_kinds'],
        stream_channels=bindings['stream_channels'], compute_resource=bindings['compute_resource'],
        host_compute_resource=bindings['host_resource'])

    selected = {row['operation']: row['option'] for row in schedule['selected']}
    tasks = {task.id: task for task in typed_problem.operations}
    incoming, outgoing = defaultdict(list), defaultdict(list)
    for edge in typed_problem.data_edges:
        incoming[edge.consumer].append(edge)
        outgoing[edge.producer].append(edge)
    steps = []
    for index, (name, lane, waits) in enumerate(zip(
            lowered.activity_operations, lowered.activity_streams, lowered.activity_waits, strict=True)):
        task = tasks[name]
        mode, = (item for item in task.options if item.id == selected[name])
        reads, writes = _materials(mode, incoming[name], outgoing[name])
        adjacent = ()
        transfer = None
        if mode.lowering == _COPY:
            adjacent = tuple(tasks[other] for other in {
                entry['producer'] for entry in reads if entry['producer'] is not None
            } | {target['operation'] for row in writes for target in row['consumers']})
            transfer = dict(logical_value=mode.parameters['logical_value'],
                            source_memory=mode.parameters['source_memory'],
                            destination_memory=mode.parameters['destination_memory'],
                            nbytes=reads[0]['nbytes'])
        row = dict(index=index, operation=name, option=mode.id, lowering=mode.lowering,
                   clock='HOST_STEADY' if mode.lowering == _HOST_ATEN else 'CUDA_EVENT',
                   execution_lane=lane, wait_for=list(waits),
                   source_operations=_source_ids(task, adjacent, refs['program'], program_ids),
                   inputs=reads, outputs=writes, parameters=deepcopy(mode.parameters))
        if transfer is not None:
            row['transfer'] = transfer
        steps.append(row)
    return dict(schema=SCHEMA, schema_version=1, references=refs, device=bindings['device'],
                problem_sha256=schedule['problem_sha256'],
                host_resource=bindings['host_resource'], compute_resource=bindings['compute_resource'],
                memory_kinds=dict(bindings['memory_kinds']), cuda_streams=list(lowered.stream_channels),
                activities=steps)


def replay_activity_binding(stored, program, problem, schedule, bindings, *, references):
    """Recompute a binding without importing Torch or initializing a device."""
    expected = derive_activity_binding(program, problem, schedule, bindings, references=references)
    if stored != expected:
        raise ValueError('archived activity binding differs from the exact native step replay')
    return expected

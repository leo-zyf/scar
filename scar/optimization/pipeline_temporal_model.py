"""L2 task/resource/cost projection; solving is a separate explicit L3 call.

The default contains no estimated durations or free control resources. The
first admitted numerical format is an explicit mathematical example, not a
calibration or deployment receipt. Original v2 physical slots are counted once
against the incremental payload budget; external inputs are not allocations.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from scar.compiler.pipeline_frontend import pipeline_digest
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.record_codec import loads
from scar.ir.versioned_schedule import VersionedTaskKind, validate_versioned_schedule
from scar.runtime.compiled_deployment import thread_environment
from .pipeline_memory import _pipeline, verify_pipeline_memory
from .pipeline_ready_schedule import verify_pipeline_ready_schedule
from .resource_solver import semantic_sha256
from .schedule import schedule_from_dict


PROJECTION_SCHEMA = 'scar.pipeline-temporal-model-projection'
SCHEMA = 'scar.pipeline-temporal-model'
COST_SCHEMA = 'scar.pipeline-task-cost-model'
ROLE = 'PIPELINE_NATIVE_TASK_DURATION_MODEL'
SCOPE = 'ONE_GUARDED_PIPELINE_HOST_CALL_WITH_FRESH_FINAL_ALLOCATION'
EXAMPLE = 'DECLARED_MODEL_EXAMPLE'
POOL = 'explicit-incremental-host-payload-budget'
CONTROL_RESOURCE = 'logical:PIPELINE_CALLER_CONTROL_LANE'


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _document(value):
    return loads(Path(value).read_bytes()) if isinstance(value, (str, Path)) else deepcopy(value)


def _inputs(pipeline, memory, ready, target):
    pipeline = _pipeline(pipeline)
    memory = verify_pipeline_memory(pipeline, memory)
    ready = verify_pipeline_ready_schedule(pipeline, memory, target, ready)
    graph = schedule_from_dict(ready['task_graph'])
    validate_versioned_schedule(graph)
    compute = {}
    for task in graph.tasks:
        if task.kind is VersionedTaskKind.COMPUTE:
            call = task.attrs.get('call_id')
            if call not in ready['call_resource_demands'] or call in compute.values():
                raise ValueError('one original COMPUTE task per bound pipeline call required')
            compute[task.id] = call
    if set(compute.values()) != {call.id for call in pipeline.calls}:
        raise ValueError('compute task projection omits an original pipeline call')
    return pipeline, memory, ready, graph, compute


def _context(pipeline, memory, ready, graph, tick):
    source = asdict(pipeline.source)
    source.pop('origins')
    binding = dict(cpu_ids=ready['cpu_ids'], call_cpu_ids=ready['call_cpu_ids'],
        resource_capacities=ready['resource_capacities'],
        call_resource_demands=ready['call_resource_demands'],
        hardware_bindings_sha256=semantic_sha256(ready['hardware_bindings']))
    return dict(role=ROLE, scope=SCOPE,
        pipeline_sha256=pipeline_digest(pipeline), memory_plan_sha256=semantic_sha256(memory),
        ready_plan_sha256=semantic_sha256(ready), task_graph_sha256=semantic_sha256(ready['task_graph']),
        target=deepcopy(ready['target']), binding=binding, binding_sha256=semantic_sha256(binding),
        input_shapes=deepcopy(memory['input_shapes']),
        incremental_payload_budget_bytes=memory['memory_budget_bytes'],
        scheduled_slot_capacities={slot.id: slot.capacity_bytes for slot in graph.storage_slots
            if slot.ownership == 'SCHEDULE' and slot.allocated_by is not None},
        control_resource=dict(id=CONTROL_RESOURCE, capacity=1,
            authority='DECLARED_ABSTRACT_SERIAL_CONTROL_SERVICE_MODEL',
            hardware_thread_binding=None, physical_reservation_proven=False),
        tick_seconds=deepcopy(tick), caller_affinity_cpu_ids=ready['caller_affinity_cpu_ids'],
        native_thread_environment=thread_environment(),
        original_evidence=dict(pipeline_source=source,
            source_origins_sha256=semantic_sha256(pipeline.source.origins),
            input_graph=deepcopy(ready['input_graph']),
            hardware_bindings_sha256=binding['hardware_bindings_sha256']),
        numerical_and_guard_contract='ORIGINAL_CLOSED_READONLY_FRESH_OUTPUT_LEAF_PROFILE',
        implementation_envelope=dict(
            allocations='ALL_SCHEDULE_SLOTS_PREALLOCATED_BEFORE_ANY_COMPUTE',
            final_allocation='PYTHON_WRAPPER_FRESH_NUMPY_EMPTY_BEFORE_CPP_HOST',
            workspace_retention='THROUGH_GLOBAL_SUCCESS_OR_FAILURE_JOIN_THEN_RAII_RELEASE',
            task_release='POST_FENCE_NO_EARLY_PHYSICAL_FREE_IMPLEMENTATION',
            source_task_allocation_dependencies_preserved=True,
            allocation_timing_optimization_implemented=False,
            complete_host_call_latency_modeled=False),
        cost_boundary='DECLARED_PER_TASK_SERVICE_MODEL_NOT_COMPLETE_CALL_OR_PYTHON_ENTRY_LATENCY',
        memory_policy='UNBOUND_NOT_NUMA_RESIDENCY')


def _cost_reference(value):
    if isinstance(value, (str, Path)):
        data = _document(value)
        if type(data) is not dict or type(data.get('schema')) is not str:
            raise ValueError('versioned task cost artifact required')
        value = artifact_reference(value, data['schema'])
    elif hasattr(value, '__dataclass_fields__'):
        value = asdict(value)
    value = _json(value)
    return value, read_artifact(value)


def _example(data, context, tasks, compute, capacities):
    """Return exact declarations, or reasons; never use a partial model."""
    issues = []
    allowed = {'schema', 'schema_version', 'basis', 'role', 'context', 'tick_seconds',
               'task_costs', 'control_resource_demands'}
    if (set(data) != allowed or data.get('schema') != COST_SCHEMA
            or type(data.get('schema_version')) is not int or data['schema_version'] != 1
            or data.get('basis') != EXAMPLE or data.get('role') != ROLE):
        return None, None, ['unsupported cost schema/basis/fields; independent measured-cost reader is not implemented']
    tick = data['tick_seconds']
    if (type(tick) is not dict or set(tick) != {'numerator', 'denominator'}
            or any(type(tick[key]) is not int or tick[key] <= 0 for key in tick)):
        return None, None, ['positive rational tick_seconds required']
    expected = deepcopy(context)
    expected['tick_seconds'] = tick
    if semantic_sha256(data['context']) != semantic_sha256(expected):
        issues.append('cost context differs from original pipeline/memory/ready/target/binding/shape/budget/tick/evidence')
    costs = data['task_costs']
    if type(costs) is not dict or set(costs) != set(tasks):
        issues.append('task_costs must cover every original v2 task exactly once')
    else:
        for key, record in costs.items():
            if (type(record) is not dict or set(record) != {'duration_ticks'}
                    or type(record['duration_ticks']) is not int or record['duration_ticks'] < 0):
                issues.append('explicit nonnegative integer duration_ticks required: ' + key)
    controls = data['control_resource_demands']
    if type(controls) is not dict or set(controls) != set(tasks) - set(compute):
        issues.append('control_resource_demands must explicitly cover all non-COMPUTE tasks')
    else:
        for key, demands in controls.items():
            if type(demands) is not dict or demands != {CONTROL_RESOURCE: 1}:
                issues.append('explicit abstract serial control-lane demand required: ' + key)
    return (None, None, issues) if issues else (costs, controls, [])


def project_pipeline_temporal_model(pipeline, memory, ready, target, *, cost_model=None):
    """L2 only: replay inputs and project constraints without invoking a solver."""
    pipeline, memory, ready, graph, compute = _inputs(pipeline, memory, ready, target)
    tasks = {task.id: task for task in graph.tasks}
    context = _context(pipeline, memory, ready, graph, None)
    capacities = deepcopy(ready['resource_capacities'])
    capacities[CONTROL_RESOURCE] = 1
    costs = {key: dict(status='UNKNOWN', duration_ticks=None) for key in tasks}
    placements = {task: deepcopy(ready['call_resource_demands'][call])
                  for task, call in compute.items()}
    reference = None
    issues = []
    admitted = False
    if cost_model is not None:
        reference, data = _cost_reference(cost_model)
        declared_costs, controls, issues = _example(data, context, tasks, compute, capacities)
        if not issues:
            admitted = True
            context['tick_seconds'] = deepcopy(data['tick_seconds'])
            costs = {key: dict(status=EXAMPLE, duration_ticks=row['duration_ticks'],
                evidence=dict(artifact=reference, locator='#/task_costs/' + key.replace('~', '~0').replace('/', '~1')))
                for key, row in declared_costs.items()}
            placements.update(deepcopy(controls))
    slots = {slot.id: slot for slot in graph.storage_slots
             if slot.ownership == 'SCHEDULE' and slot.allocated_by is not None}
    if sum(slot.capacity_bytes for slot in slots.values()) != memory['peak_allocation_bytes']:
        raise ValueError('physical-slot projection differs from original fixed pool plus fresh final bytes')
    projection = dict(task_costs=costs, task_placements=placements,
        resource_capacities=capacities,
        memory_capacities={POOL: memory['memory_budget_bytes']},
        materialization_sizes={key: slot.capacity_bytes for key, slot in slots.items()},
        memory_locations={key: POOL for key in slots},
        model_instantiation=dict(cost_context=context, cost_model=reference,
            numerical_cost_basis=EXAMPLE if admitted else 'UNKNOWN'))
    unresolved = [dict(field='cost_model', reason=issue) for issue in issues]
    for task in tasks:
        if costs[task]['status'] == 'UNKNOWN':
            unresolved.append(dict(field='task_costs.' + task, reason='duration is UNKNOWN; no zero substitution'))
        if task not in placements:
            unresolved.append(dict(field='task_placements.' + task,
                reason='non-COMPUTE service resource occupation is UNKNOWN; no free-control assumption'))
    report = dict(schema=PROJECTION_SCHEMA, schema_version=1,
        layer='L2_CONSTRAINT_AND_COST_PROJECTION_ONLY', status='COMPLETE_DECLARED_MODEL_EXAMPLE' if admitted else 'UNRESOLVED',
        pipeline_sha256=pipeline_digest(pipeline), memory_plan_sha256=semantic_sha256(memory),
        ready_plan_sha256=semantic_sha256(ready), task_graph_sha256=semantic_sha256(ready['task_graph']),
        target=deepcopy(ready['target']), binding_sha256=context['binding_sha256'],
        cost_context=context, cost_model=reference, compute_call_map=compute,
        task_kinds={key: task.kind.value for key, task in tasks.items()},
        control_task_ids=sorted(set(tasks)-set(compute)), projection=projection, unresolved=unresolved,
        input_projection_receipts=[dict(role=role, semantic_sha256=digest) for role, digest in (
            ('ORIGINAL_PIPELINE_IR', pipeline_digest(pipeline)), ('REPLAYED_MEMORY_PLAN', semantic_sha256(memory)),
            ('REPLAYED_READY_PLAN', semantic_sha256(ready)), ('ORIGINAL_VERSIONED_TASK_GRAPH', semantic_sha256(ready['task_graph'])))],
        physical_pool_scope='SCHEDULE_OWNED_PHYSICAL_SLOTS_ONCE_EXTERNAL_INPUTS_EXCLUDED',
        abstract_control_resource=deepcopy(context['control_resource']),
        encoded_graph_budget_used_as_incremental_budget=False,
        cost_authority=dict(basis=EXAMPLE if admitted else 'UNKNOWN', measured=False, scope=SCOPE,
            declared_control_resource_placement_only=admitted, deployment_authorized=False),
        solver_invoked=False, runtime_order_executed=False, model_optimum_proven=False,
        physical_optimality_proven=False, deployment_authorized=False,
        target_program_executed=False, performance_measured=False)
    return _json(report)


def verify_pipeline_temporal_projection(pipeline, memory, ready, target, report):
    value = _document(report)
    if value.get('schema') != PROJECTION_SCHEMA or type(value.get('schema_version')) is not int or value['schema_version'] != 1:
        raise ValueError('versioned L2 pipeline temporal projection required')
    expected = project_pipeline_temporal_model(pipeline, memory, ready, target, cost_model=value['cost_model'])
    if semantic_sha256(expected) != semantic_sha256(value):
        raise ValueError('temporal projection differs from original input/cost/resource replay')
    return value


def plan_pipeline_temporal_model(pipeline, memory, ready, target, *, cost_model=None, search_budget=100000):
    """Explicit L3 entry; L2 projection never calls this function implicitly."""
    from .resource_solver import solve_resource_schedule
    projected = project_pipeline_temporal_model(pipeline, memory, ready, target, cost_model=cost_model)
    graph = schedule_from_dict(_document(ready)['task_graph'])
    solver = solve_resource_schedule(graph, **projected['projection'], search_budget=search_budget)
    return _json(dict(schema=SCHEMA, schema_version=1,
        layer='L3_EXPLICIT_FIXED_RESOURCE_MAKESPAN_SEARCH', projection_report=projected,
        cost_context=projected['cost_context'], cost_model=projected['cost_model'],
        cost_authority=projected['cost_authority'], compute_call_map=projected['compute_call_map'],
        solver_report=solver, search_budget=search_budget, solver_invoked=True,
        runtime_order_executed=False, physical_optimality_proven=False,
        deployment_authorized=False, target_program_executed=False, performance_measured=False))


def verify_pipeline_temporal_model(pipeline, memory, ready, target, report):
    value = _document(report)
    if value.get('schema') != SCHEMA or type(value.get('schema_version')) is not int or value['schema_version'] != 1:
        raise ValueError('versioned L3 pipeline temporal model required')
    expected = plan_pipeline_temporal_model(pipeline, memory, ready, target,
        cost_model=value['cost_model'], search_budget=value['search_budget'])
    if semantic_sha256(expected) != semantic_sha256(value):
        raise ValueError('temporal model differs from original inputs and complete solver replay')
    return value


__all__ = ['PROJECTION_SCHEMA', 'SCHEMA', 'COST_SCHEMA', 'ROLE', 'SCOPE', 'EXAMPLE', 'CONTROL_RESOURCE',
    'project_pipeline_temporal_model', 'verify_pipeline_temporal_projection',
    'plan_pipeline_temporal_model', 'verify_pipeline_temporal_model']

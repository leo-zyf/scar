"""Replay source, storage and resource proofs for native CPU execution."""
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path

from scar.compiler.pipeline_frontend import pipeline_digest
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.record_codec import loads
from scar.ir.versioned_schedule import (
    VersionedTaskKind, schedule_from_dict, validate_versioned_schedule,
)
from .pipeline_ready_schedule import verify_pipeline_ready_schedule
from scar.optimization.pipeline_memory import verify_pipeline_memory
from .resource_solver import semantic_sha256


SCHEMA = 'scar.pipeline-execution-refinement'
FAMILY = 'SOLVER_ORDER_CPU_SERIAL_LEAVES_V1'
RESOURCE_EVENTS_FAMILY = 'RESOURCE_EVENTS_CPU_SERIAL_LEAVES_V1'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _acyclic(call_order, predecessors, lane_orders):
    dependencies = {name: set(predecessors[name]) for name in call_order}
    for lane in lane_orders.values():
        for before, after in zip(lane, lane[1:]):
            dependencies[after].add(before)
    completed = set()
    order = []
    while len(order) < len(call_order):
        ready = [name for name in call_order
                 if name not in completed and dependencies[name] <= completed]
        _need(ready, 'solver lane order and original true dependencies form a cycle')
        for name in ready:
            completed.add(name)
            order.append(name)
    return order


def _projection(pipeline, ready, temporal):
    solver = temporal['solver_report']
    _need(temporal.get('cost_model') is not None,
          'unknown task costs cannot select a native execution refinement')
    _need(solver.get('status') == 'OPTIMAL_WITHIN_MODEL'
          and solver.get('optimality') == 'MODEL_OPTIMUM_ESTABLISHED'
          and solver.get('search_budget_exhausted') is False
          and not solver.get('unresolved') and not solver.get('execution_unresolved')
          and solver.get('source_schedule_executable') is True
          and type(solver.get('solution')) is dict,
          'a complete replayed exact model solution is required')
    graph = schedule_from_dict(ready['task_graph'])
    topology = validate_versioned_schedule(graph)['topological_order']
    rank = {name: index for index, name in enumerate(topology)}
    tasks = {task.id: task for task in graph.tasks}
    timed = solver['solution']['tasks']
    _need(type(timed) is list and len(timed) == len(tasks)
          and len({row['task'] for row in timed}) == len(timed)
          and {row['task'] for row in timed} == set(tasks),
          'model solution must cover every original task exactly once')
    times = {row['task']: row for row in timed}
    for row in timed:
        _need(type(row['start_ticks']) is int and row['start_ticks'] >= 0
              and type(row['end_ticks']) is int and row['end_ticks'] >= row['start_ticks'],
              'model task endpoints must be nonnegative integer ticks')
        for predecessor in tasks[row['task']].depends_on:
            _need(times[predecessor]['end_ticks'] <= row['start_ticks'],
                  'model task starts before an original predecessor completes')
    mapping = temporal['compute_call_map']
    calls = [call.id for call in pipeline.calls]
    computes = {task.id for task in graph.tasks if task.kind is VersionedTaskKind.COMPUTE}
    _need(type(mapping) is dict and set(mapping) == computes
          and len(mapping) == len(calls) and set(mapping.values()) == set(calls),
          'one original COMPUTE task per original caller call is required')
    _need(all(tasks[name].attrs.get('call_id') == call for name, call in mapping.items()),
          'model COMPUTE mapping differs from original task call identity')
    # A global original-DAG topological tie rank handles zero-duration tasks
    # consistently across lanes; lexical task IDs are not a causal order.
    ordered = sorted(mapping, key=lambda name: (times[name]['start_ticks'],
        times[name]['end_ticks'] != times[name]['start_ticks'], rank[name]))
    lanes = {str(cpu): [] for cpu in ready['cpu_ids']}
    rows = []
    for name in ordered:
        call = mapping[name]
        cpu = ready['call_cpu_ids'][call]
        lanes[str(cpu)].append(call)
        rows.append(dict(task=name, call=call, cpu_id=cpu,
            start_ticks=times[name]['start_ticks'], end_ticks=times[name]['end_ticks'],
            original_task_topological_rank=rank[name]))
    by_call = {row['call']: row for row in rows}
    for lane in lanes.values():
        for before, after in zip(lane, lane[1:]):
            _need(by_call[before]['end_ticks'] <= by_call[after]['start_ticks'],
                  'selected model overlaps compute calls on the same CPU lane')
    topological_calls = _acyclic(calls, ready['call_predecessors'], lanes)
    return lanes, rows, topological_calls, [task.id for task in graph.tasks
        if task.kind is not VersionedTaskKind.COMPUTE]


def _resource_events(ready):
    graph = schedule_from_dict(ready['task_graph'])
    topology = validate_versioned_schedule(graph)['topological_order']
    tasks = {task.id: task for task in graph.tasks}
    materials = {material.id: material for material in graph.materializations}
    consumers = {name: [] for name in materials}
    events = []
    for name in topology:
        task = tasks[name]
        row = dict(id=name, kind=task.kind.value, depends_on=list(task.depends_on))
        if task.kind is VersionedTaskKind.COMPUTE:
            call = task.attrs['call_id']
            row.update(call_id=call, cpu_id=ready['call_cpu_ids'][call])
        elif task.kind in (VersionedTaskKind.ALLOCATE, VersionedTaskKind.RELEASE):
            slots = task.allocates or task.releases
            _need(len(slots) == 1, 'native lifetime event must address one physical slot')
            row['slot_id'], = slots
        else:
            _need(task.kind in (VersionedTaskKind.SUCCESS_FENCE, VersionedTaskKind.EXPORT),
                  'unsupported task in native resource-event profile')
        for material in task.reads:
            consumers[material].append(name)
        events.append(row)
    by_slot = {slot.id: [] for slot in graph.storage_slots}
    for material in graph.materializations:
        by_slot[material.region.slot].append(material)
    lifetime = []
    for slot in graph.storage_slots:
        versions = sorted(by_slot[slot.id], key=lambda material: material.version)
        lifetime.append(dict(slot_id=slot.id, byte_capacity=slot.capacity_bytes,
            ownership=slot.ownership, allocation=slot.allocated_by, release=slot.released_by,
            versions=[dict(materialization=material.id, logical_value=material.value,
                slot_version=material.version, producer=material.ready_task,
                consumers=consumers[material.id]) for material in versions]))
    return events, lifetime


@contextmanager
def _selected(original_domain, selected_case_id=None):
    from scar.compiler.solve_pipeline_temporal import SCHEMA as DOMAIN, verify_pipeline_temporal_domain
    from scar.compiler.project_pipeline_temporal import open_temporal_inputs
    from .pipeline_temporal_model import verify_pipeline_temporal_model
    if isinstance(original_domain, (str, Path)):
        path = Path(original_domain).resolve()
        original_domain = artifact_reference(path/'result.json' if path.is_dir() else path, DOMAIN)
    reference = deepcopy(original_domain)
    _need(reference.get('schema') == DOMAIN, 'original finite-domain ArtifactRef required')
    domain = verify_pipeline_temporal_domain(read_artifact(reference))
    decision = domain['decision']
    _need(decision['status'] == 'OPTIMAL_WITHIN_EXPLICIT_DECLARED_DOMAIN'
          and decision['selection_decision'] == 'MODEL_SELECTED_EXPLICIT_CASE_ONLY'
          and decision['all_cases_search_complete'] is True and decision['model_optimum_proven'] is True
          and not decision['unresolved_case_ids'] and decision['selected_case_id'] is not None,
          'only a complete selected original finite domain can produce an execution certificate')
    selected = decision['selected_case_id']
    _need(selected_case_id is None or selected_case_id == selected,
          'requested case differs from the original finite-domain decision')
    rows = [row for row in domain['cases'] if row['id'] == selected]
    _need(len(rows) == 1 and rows[0]['search_complete'] is True
          and rows[0]['proved_infeasible'] is False, 'one complete selected feasible case required')
    row = rows[0]
    with open_temporal_inputs(row['inputs']) as (pipeline, memory, ready, target):
        temporal = read_artifact(row['plan'])
        verify_pipeline_temporal_model(pipeline, memory, ready, target, temporal)
        refs = {name: deepcopy(row[name]) for name in (
            'preparation', 'inputs', 'projection', 'original_proof', 'memory', 'ready', 'plan', 'cost_model')}
        yield reference, selected, refs, pipeline, memory, ready, target, temporal
        for value in refs.values():
            if value is not None:
                read_artifact(value)
    read_artifact(reference)


def _derive(pipeline, memory, ready, temporal, reference, selected, refs, *, resource_events=False):
    lanes, rows, topology, envelope_tasks = _projection(pipeline, ready, temporal)
    report = dict(schema=SCHEMA, schema_version=1, execution_family=FAMILY,
        status='MODEL_COMPUTE_ORDER_IMPLEMENTABLE_UNDER_EXISTING_HOST_ENVELOPE',
        original_domain=deepcopy(reference), selected_case_id=selected,
        selected_case_references=deepcopy(refs),
        pipeline_sha256=pipeline_digest(pipeline), memory_plan_sha256=semantic_sha256(memory),
        target=deepcopy(ready['target']), cpu_ids=list(ready['cpu_ids']),
        call_predecessors=deepcopy(ready['call_predecessors']),
        call_cpu_ids=deepcopy(ready['call_cpu_ids']),
        requires_serial_leaf_operators=True,
        ready_plan_sha256=semantic_sha256(ready), temporal_plan_sha256=semantic_sha256(temporal),
        lane_call_order=lanes, model_compute_order=rows,
        dependency_and_lane_order_topological_calls=topology,
        tie_policy='MODEL_START_THEN_ZERO_DURATION_BEFORE_POSITIVE_THEN_ORIGINAL_FULL_TASK_DAG_TOPOLOGICAL_RANK',
        dispatch_policy='FIXED_CPU_LANE_HEAD_ONLY_WAIT_TRUE_PREDECESSORS_NO_TIMESTAMP_WAIT',
        dependency_authority='ORIGINAL_READY_PLAN_TRUE_RAW_AND_POOL_VERSION_EDGES',
        resource_order_role='EXECUTION_REFINEMENT_NOT_ORIGINAL_PROGRAM_DEPENDENCY',
        existing_host_envelope=dict(
            allocation='ALL_PRIVATE_SLOTS_BEFORE_WORKERS_NOT_MODEL_ALLOCATION_TIMESTAMPS',
            preflight='ORIGINAL_DESCRIPTOR_TABLE_AND_PAYLOAD_GUARDS',
            completion='ORIGINAL_SUCCESS_OR_FAILURE_JOIN_ALL_WORKERS',
            release='ENTRY_EXIT_RAII_NOT_MODEL_RELEASE_TIMESTAMPS',
            output='FRESH_PRIVATE_FINAL_EXPOSED_ONLY_AFTER_FULL_SUCCESS'),
        non_compute_model_tasks_not_individually_timed_by_host=envelope_tasks,
        implementation_scope='MODEL_SOLUTION_COMPUTE_ORDER_ONLY',
        full_model_timeline_implemented=False,
        task_model_makespan_is_native_runtime_prediction=False,
        model_solution_optimal_within_declared_model=True,
        native_compiled=False, runtime_order_executed=False,
        native_host_optimality_proven=False, physical_optimality_proven=False,
        duration_predicted=False, deployment_authorized=False,
        target_program_executed=False, performance_measured=False)
    if resource_events:
        report.update(_event_contract(ready), case_role='REPLAYED_FINITE_MODEL_SELECTION')
        del report['non_compute_model_tasks_not_individually_timed_by_host']
    return report


def _event_contract(ready):
    events, lifetime = _resource_events(ready)
    return dict(execution_family=RESOURCE_EVENTS_FAMILY,
        status='RESOURCE_EVENTS_IMPLEMENTABLE_UNDER_SOURCE_CONTRACT',
        implementation_scope='SOURCE_RESOURCE_EVENTS_FIXED_CPU_LANES',
        dispatch_policy='FIXED_CPU_LANE_HEAD_ONLY_WAIT_COMPLETED_RESOURCE_PREDECESSORS',
        existing_host_envelope=dict(
            allocation='ADOPT_PRIVATE_ABI1_FINAL_THEN_ALLOCATE_FIXED_WORKSPACE',
            preflight='ORIGINAL_DESCRIPTOR_TABLE_AND_PAYLOAD_GUARDS',
            completion='ORIGINAL_SUCCESS_OR_FAILURE_JOIN_ALL_WORKERS',
            release='PHYSICAL_WORKSPACE_FREE_AFTER_CERTIFIED_SUCCESS_FENCE_OR_FAILURE_CLEANUP',
            output='FRESH_PRIVATE_FINAL_EXPOSED_ONLY_AFTER_CERTIFIED_EXPORT'),
        resource_events=events, slot_lifetimes=lifetime,
        publication_task=next(row['id'] for row in events if row['kind'] == 'EXPORT'))


def _derive_resource_case(pipeline, memory, ready, refs, reference, selected, role):
    lanes = {str(cpu): [] for cpu in ready['cpu_ids']}
    for call in pipeline.calls:
        lanes[str(ready['call_cpu_ids'][call.id])].append(call.id)
    topology = _acyclic([call.id for call in pipeline.calls], ready['call_predecessors'], lanes)
    return dict(schema=SCHEMA, schema_version=1, **_event_contract(ready),
        case_role=role, original_domain=reference, selected_case_id=selected,
        selected_case_references=deepcopy(refs), pipeline_sha256=pipeline_digest(pipeline),
        memory_plan_sha256=semantic_sha256(memory), ready_plan_sha256=semantic_sha256(ready),
        target=deepcopy(ready['target']), cpu_ids=list(ready['cpu_ids']),
        call_predecessors=deepcopy(ready['call_predecessors']),
        call_cpu_ids=deepcopy(ready['call_cpu_ids']), requires_serial_leaf_operators=True,
        lane_call_order=lanes, dependency_and_lane_order_topological_calls=topology,
        tie_policy='ORIGINAL_CALL_ORDER_WITHIN_EACH_LANE',
        dependency_authority='ORIGINAL_READY_PLAN_TRUE_RAW_AND_POOL_VERSION_EDGES',
        resource_order_role='EXECUTION_REFINEMENT_NOT_ORIGINAL_PROGRAM_DEPENDENCY',
        full_model_timeline_implemented=False, task_model_makespan_is_native_runtime_prediction=False,
        model_solution_optimal_within_declared_model=False, native_compiled=False,
        runtime_order_executed=False, native_host_optimality_proven=False,
        physical_optimality_proven=False, duration_predicted=False, deployment_authorized=False,
        target_program_executed=False, performance_measured=False)


@contextmanager
def _resource_domain_certificate(domain, trial_case=None):
    from scar.compiler.solve_pipeline_temporal import open_pipeline_resource_event_case
    with open_pipeline_resource_event_case(domain, trial_case) as (
            reference, selected, case, pipeline, memory, ready, target):
        refs = {name: case[name] for name in ('inputs', 'original_proof', 'memory', 'ready')}
        role = 'COST_SELECTED_RESOURCE_CASE' if trial_case is None else 'EXPLICIT_DOMAIN_VALIDATION_TRIAL'
        certificate = _derive_resource_case(pipeline, memory, ready, refs, reference, selected, role)
        _need(certificate['lane_call_order'] == case['schedules']['lane_call_order'],
              'resource-domain lane order differs from source-replayed execution order')
        yield certificate, refs, pipeline, memory, ready, target


def refine_pipeline_resource_domain(domain, *, trial_case=None):
    """Use the cost-selected case, or explicitly request an admitted validation trial."""
    with _resource_domain_certificate(domain, trial_case) as (certificate, *_):
        return certificate


@contextmanager
def replay_pipeline_execution_refinement(value):
    _need(value.get('schema') == SCHEMA and type(value.get('schema_version')) is int
          and value['schema_version'] == 1, 'versioned pipeline execution refinement required')
    family = value.get('execution_family')
    _need(family in (FAMILY, RESOURCE_EVENTS_FAMILY), 'unsupported refinement execution family')

    def verify(expected):
        _need(semantic_sha256(value) == semantic_sha256(expected),
              'certificate differs from independently replayed source/model selection')

    role = value.get('case_role')
    if family == RESOURCE_EVENTS_FAMILY and role in ('COST_SELECTED_RESOURCE_CASE', 'EXPLICIT_DOMAIN_VALIDATION_TRIAL'):
        trial_case = value['selected_case_id'] if role == 'EXPLICIT_DOMAIN_VALIDATION_TRIAL' else None
        with _resource_domain_certificate(value['original_domain'], trial_case) as (
                expected, refs, pipeline, memory, ready, target):
            verify(expected)
            yield refs, pipeline, memory, ready, target
    else:
        with _selected(value['original_domain'], value['selected_case_id']) as (
                reference, selected, refs, pipeline, memory, ready, target, temporal):
            verify(_derive(pipeline, memory, ready, temporal, reference, selected, refs,
                           resource_events=family == RESOURCE_EVENTS_FAMILY))
            yield refs, pipeline, memory, ready, target


def refine_selected_pipeline_temporal(original_domain, *, resource_events=False):
    """Replay the entire original domain and independently reopen its winner."""
    with _selected(original_domain) as (reference, selected, refs, pipeline, memory, ready, _, temporal):
        return _derive(pipeline, memory, ready, temporal, reference, selected, refs,
                       resource_events=resource_events)


def plan_pipeline_execution_refinement(pipeline, memory_plan, ready_plan, target, temporal_plan,
                                      *, original_domain, selected_case_id=None):
    """Bind supplied lowering inputs to the independently replayed domain winner."""
    with _selected(original_domain, selected_case_id) as (
            reference, selected, refs, original, memory, ready, _, temporal):
        current_memory = verify_pipeline_memory(pipeline, memory_plan)
        current_ready = verify_pipeline_ready_schedule(pipeline, current_memory, target, ready_plan)
        current_temporal = loads(Path(temporal_plan).read_bytes()) if isinstance(
            temporal_plan, (str, Path)) else deepcopy(temporal_plan)
        _need(pipeline_digest(pipeline) == pipeline_digest(original)
              and semantic_sha256(current_memory) == semantic_sha256(memory)
              and semantic_sha256(current_ready) == semantic_sha256(ready)
              and semantic_sha256(current_temporal) == semantic_sha256(temporal),
              'lowering inputs differ from original finite-domain selected case references')
        return _derive(pipeline, memory, ready, temporal, reference, selected, refs)


def verify_pipeline_execution_refinement(pipeline, memory_plan, target, report):
    """Reconstruct the source/model/lane proof before native lowering or load."""
    value = loads(Path(report).read_bytes()) if isinstance(report, (str, Path)) else deepcopy(report)
    with replay_pipeline_execution_refinement(value) as (_, original, memory, ready, _):
        current_memory = verify_pipeline_memory(pipeline, memory_plan)
        current_ready = verify_pipeline_ready_schedule(pipeline, current_memory, target, ready)
        _need(pipeline_digest(pipeline) == pipeline_digest(original)
              and semantic_sha256(current_memory) == semantic_sha256(memory)
              and semantic_sha256(current_ready) == semantic_sha256(ready),
              'pipeline/memory/hardware differs from original finite-domain selected case')
    return value


__all__ = ['SCHEMA', 'FAMILY', 'RESOURCE_EVENTS_FAMILY', 'plan_pipeline_execution_refinement',
    'verify_pipeline_execution_refinement', 'refine_selected_pipeline_temporal',
    'refine_pipeline_resource_domain', 'replay_pipeline_execution_refinement']

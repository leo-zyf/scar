"""Source-qualified CPU implementation alternatives over an unchanged pipeline.

The point model covers serial ABI1 leaf services. The existing host envelope
is retained, with its uncalibrated allocation/dispatch/join costs explicit.
"""
from dataclasses import asdict
from hashlib import sha256
import os
from pathlib import Path
import shlex
import shutil

from scar.compiler.model import document as leaf_document
from scar.compiler.pipeline_frontend import pipeline_digest
from scar.ir.implementation_plan import (CostEstimate, ImplementationOption,
    PlanningOperation, PlanningProblem, PlanningResource, digest, validate_problem)
from scar.ir.program.providers import ProviderArtifact
from scar.optimization.hardware import propose_cpu_placements, bind_cpu_placement
from scar.optimization.joint_planner import verify_execution_plan
from scar.optimization.implementation_costs import CostQuery, query_key, resolve_cost
from scar.optimization.pipeline_schedule import _closed, _dependencies, _target
from scar.optimization.resource_solver import semantic_sha256

COST_SCHEMA = 'scar.pipeline-implementation-costs'
CERT_SCHEMA = 'scar.pipeline-implementation-refinement'
FAMILY = 'JOINT_IMPLEMENTATION_CPU_SERIAL_LEAVES_V1'
LOWERING = 'EXISTING_CPP_ABI1_SINGLE_CPU_LEAF'
SCOPE = 'ABI1_LEAF_SERVICE_ONLY_HOST_ENVELOPE_UNCALIBRATED'
ISOLATED_SERVICE_RESOURCE = 'model:isolated-service-regime'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _compiler():
    from scar.codegen.cpp import COMPILER_OPTIONS, runtime_implementation_sha256
    command = shlex.split(os.environ.get('CXX', 'g++'))
    executable = shutil.which(command[0]) if command else None
    _need(executable is not None, 'original GNU C++ compiler unavailable')
    return dict(command=command, executable=str(Path(executable).resolve()),
        executable_sha256=sha256(Path(executable).read_bytes()).hexdigest(),
        options=list(COMPILER_OPTIONS), backend_sha256=runtime_implementation_sha256())


def build_pipeline_problem(pipeline, memory_plan, target, *, cpu_ids,
        cost_model=None, cost_profile_ref=None, source_refs=(),
        host_lifecycle=False, host_cost_profile_ref=None):
    """Derive all call/CPU choices; missing exact-context estimates stay UNKNOWN.

    Source refs must be qualified by the caller's original F4 replay. This
    conditional model builder does not itself authorize source replacement.
    """
    from scar.codegen.pipeline import _memory
    _closed(pipeline)
    memory = _memory(pipeline, memory_plan)
    _, snapshot, cpus, _, hardware = _target(target, cpu_ids)
    _need(bool(source_refs), 'original source input references required')
    _need(cost_model is None or cost_profile_ref is None, 'cost model and empirical profile are mutually exclusive')
    _need(type(host_lifecycle)is bool and(host_lifecycle or host_cost_profile_ref is None),
        'host costs require the explicit source lifecycle model')
    compiler = _compiler()
    model = cost_model
    if model is not None:
        _need((model.get('schema'), model.get('schema_version')) == (COST_SCHEMA, 1)
            and model.get('pipeline_sha256') == pipeline_digest(pipeline)
            and model.get('memory_plan_sha256') == semantic_sha256(memory)
            and model.get('target') == snapshot and type(model.get('costs')) is dict,
            'cost model source/pool/target context differs')
    resources = [PlanningResource(row['thread']['record']['id']['wire'], 'EXECUTION', 1,
        (dict(target=row['thread']['target'], executor=row['executor']),)) for row in hardware]
    pool = 'formal:pipeline-private-payload-budget'
    # Capacity is the guarded formal budget, not available RAM or NUMA residency.
    resources.append(PlanningResource(pool, 'MEMORY', max(1, memory['memory_budget_bytes']),
        (dict(scope=memory['budget_scope'], physical_capacity_proven=False),)))
    threads = {cpu: resources[index].id for index, cpu in enumerate(cpus)}
    if cost_profile_ref is not None:
        resources.append(PlanningResource(ISOLATED_SERVICE_RESOURCE, 'BANDWIDTH_CLASS', 1,
            (dict(role='ISOLATED_SERVICE_COST_APPLICABILITY_MUTEX',
                profile=cost_profile_ref, physical_bandwidth_capacity_proven=False,
                reason='NO_CONCURRENT_SERVICE_COST_EVIDENCE'),)))
    predecessors, _ = _dependencies(pipeline, memory)
    values = {row['id']: row for row in memory['values']}
    operations = []
    seen_costs = set()
    for index, call in enumerate(pipeline.calls):
        options = []
        baseline_cpu = (call.plan.resource_binding.cpu_ids[0]
            if call.plan.resource_binding is not None and len(call.plan.resource_binding.cpu_ids) == 1
            and call.plan.resource_binding.cpu_ids[0] in cpus else cpus[index % len(cpus)])
        for cpu in cpus:
            placements = propose_cpu_placements(call.plan, target, affinity=[cpu],
                memory_budget_bytes=memory['memory_budget_bytes'], max_threads=1)
            binding = next((row for row in placements if row.cpu_ids == (cpu,)), None)
            _need(binding is not None, 'call has no original singleton CPU lowering')
            bound = bind_cpu_placement(call.plan, binding)
            context = dict(schema='scar.pipeline-implementation-cost-context', schema_version=1,
                pipeline_sha256=pipeline_digest(pipeline), memory_plan_sha256=semantic_sha256(memory),
                target=snapshot, call_id=call.id, source_leaf_plan_sha256=semantic_sha256(leaf_document(call.plan)),
                bound_leaf_plan_sha256=semantic_sha256(leaf_document(bound)), resource_binding=asdict(binding),
                inputs=[dict(parameter=name, shape=values[value]['shape'], nbytes=values[value]['nbytes'])
                    for name, value in zip(call.plan.region.parameters, call.inputs)],
                output=dict(shape=values[call.output]['shape'], nbytes=values[call.output]['nbytes']),
                input_layout='C_CONTIGUOUS', compiler=compiler, scope=SCOPE)
            query = CostQuery(SCOPE, context, tuple(source_refs))
            key = query_key(query)
            estimate = None if model is None else model['costs'].get(key)
            if estimate is None:
                cost = resolve_cost(query, cost_profile_ref)
            else:
                _need(semantic_sha256(estimate.get('context')) == semantic_sha256(context)
                    and estimate.get('basis') == 'DECLARED_MODEL'
                    and estimate.get('missing_terms') == [],
                    'only exact-context explicitly declared model estimates are admitted by this producer')
                cost = CostEstimate(estimate['estimate_ns'], estimate.get('lower_ns'), estimate.get('upper_ns'),
                    'DECLARED_MODEL', key, tuple(source_refs)+
                    (dict(model_sha256=semantic_sha256(model), locator='#/costs/'+key),), ())
                seen_costs.add(key)
            options.append(ImplementationOption(call.id+'@cpu:'+str(cpu), LOWERING, tuple(source_refs),
                {threads[cpu]: 1, **({ISOLATED_SERVICE_RESOURCE: 1} if cost_profile_ref is not None else {})},
                {}, {}, {}, cost,
                dict(cpu_id=cpu, resource_binding=asdict(binding), context=context,
                    bound_leaf_plan_sha256=semantic_sha256(leaf_document(bound)))))
        operations.append(PlanningOperation(call.id, tuple(source_refs), tuple(predecessors[call.id]),
            tuple(options), call.id+'@cpu:'+str(baseline_cpu)))
    if model is not None:
        _need(set(model['costs']) == seen_costs, 'cost model contains extra or nonmatching call/CPU contexts')
    problem=validate_problem(PlanningProblem(ProviderArtifact(**snapshot), tuple(source_refs),
        tuple(resources), tuple(operations), (), assumptions=(
            'ORIGINAL_F4_CALLER_AND_LEAF_PURITY_REPLAY_REQUIRED',
            'EXACT_READONLY_C_CONTIGUOUS_F64_INPUTS_STABLE_DURING_REGION',
            'ORIGINAL_TRUE_RAW_WAR_WAW_AND_POOL_ASSIGNMENT_PRESERVED', SCOPE,
            'HOST_SETUP_ALLOCATION_DISPATCH_JOIN_NOT_MODELED_AS_ZERO',
            ('ISOLATED_PROFILE_LOGICAL_SERIAL_REGIME_NOT_PHYSICAL_BANDWIDTH_CAPACITY'
                if cost_profile_ref is not None else
                'DECLARED_OR_UNKNOWN_COST_MODEL_THREAD_CAPACITIES_NOT_EMPIRICAL_CONCURRENCY_EVIDENCE')),
        reserved_memory={pool: memory['peak_allocation_bytes']}))
    if host_lifecycle:
        from scar.optimization.pipeline_lifecycle import attach_pipeline_lifecycle
        problem=attach_pipeline_lifecycle(problem,pipeline,memory,
            cost_profile_ref=host_cost_profile_ref,source_refs=tuple(source_refs))
    return problem


def cost_contexts(problem):
    contexts={mode.parameters.get('leaf_service_cost',{}).get('context',mode.cost.context):mode.parameters['context']
        for operation in problem.operations for mode in operation.options}
    if problem.activities:
        from scar.optimization.pipeline_lifecycle import lifecycle_cost_contexts
        contexts.update(lifecycle_cost_contexts(problem))
    return contexts


def refine_pipeline_implementation(pipeline, memory, target, problem, execution):
    """Verify selected modes and emit a fixed-lane order, never timestamp waits."""
    problem = validate_problem(problem)
    verify_execution_plan(problem, execution)
    _need(execution['domain_complete'] is True and execution['frontiers'] == []
        and execution['status'] in ('OPTIMAL_WITHIN_MODEL', 'FEASIBLE_WITHIN_MODEL')
        and execution['baseline_only'] is False, 'complete known implementation domain required')
    calls = [call.id for call in pipeline.calls]
    _need([op.id for op in problem.operations] == calls and not problem.data_edges,
        'fixed-pool call implementation problem required')
    selected = {row['operation']: row for row in execution['selected']}
    options = {op.id: next(mode for mode in op.options if mode.id == selected[op.id]['option'])
        for op in problem.operations}
    cpus = sorted({mode.parameters['cpu_id'] for mode in options.values()})
    _, snapshot, _, _, _ = _target(target, cpus)
    predecessors, reasons = _dependencies(pipeline, memory)
    _need(asdict(problem.target) == snapshot
        and all(tuple(predecessors[op.id]) == op.predecessors for op in problem.operations),
        'selected model differs from original data/pool predecessor authority')
    # Full source order is already a topological rank for this straight-line
    # profile. At one tick, zero-duration events precede positive intervals.
    order = sorted(calls, key=lambda name: (selected[name]['start_ns'],
        selected[name]['end_ns'] != selected[name]['start_ns'], calls.index(name)))
    resource_order = {name: [] for name in calls}
    regime = [row for row in problem.resources if row.id == ISOLATED_SERVICE_RESOURCE]
    if regime:
        _need(len(regime) == 1 and regime[0].kind == 'BANDWIDTH_CLASS' and regime[0].capacity == 1
            and all(mode.demands.get(ISOLATED_SERVICE_RESOURCE) == 1 for mode in options.values()),
            'isolated service regime requires one shared logical capacity claim per call')
        for before, after in zip(order, order[1:]):
            _need(selected[before]['end_ns'] <= selected[after]['start_ns'],
                'isolated profile services overlap in the selected model')
            resource_order[after].append(before)
    delivered_predecessors = {name: sorted(set(predecessors[name]) | set(resource_order[name]), key=calls.index)
        for name in calls}
    lanes = {str(cpu): [name for name in order if options[name].parameters['cpu_id'] == cpu] for cpu in cpus}
    union = {name: set(delivered_predecessors[name]) for name in calls}
    for lane in lanes.values():
        for before, after in zip(lane, lane[1:]):
            _need(selected[before]['end_ns'] <= selected[after]['start_ns'], 'lane intervals overlap')
            union[after].add(before)
    done = set()
    while len(done) < len(calls):
        ready = {name for name in calls if name not in done and union[name] <= done}
        _need(bool(ready), 'lane order and true predecessors form a cycle')
        done.update(ready)
    certificate=dict(schema=CERT_SCHEMA, schema_version=1, execution_family=FAMILY,
        problem_sha256=digest(problem), execution_plan_sha256=semantic_sha256(execution),
        pipeline_sha256=pipeline_digest(pipeline), memory_plan_sha256=semantic_sha256(memory), target=snapshot,
        cpu_ids=cpus, call_cpu_ids={name: options[name].parameters['cpu_id'] for name in calls},
        original_call_predecessors=predecessors,
        resource_order_dependencies=resource_order,
        resource_order_authority=('ISOLATED_SERVICE_COST_APPLICABILITY_IMPLEMENTATION_REFINEMENT'
            if regime else 'NO_ADDITIONAL_CROSS_CPU_RESOURCE_ORDER'),
        resource_order_is_original_semantic_dependency=False,
        call_predecessors=delivered_predecessors, dependency_reasons=reasons, lane_call_order=lanes,
        call_order=order, selected_options={name: selected[name]['option'] for name in calls},
        requires_serial_leaf_operators=True, solver_status=execution['status'],
        dispatch_policy='FIXED_CPU_LANE_HEAD_ONLY_WAIT_TRUE_PREDECESSORS_NO_TIMESTAMP_WAIT',
        implementation_scope='SELECTED_CPU_PLACEMENT_AND_COMPUTE_LANE_ORDER_ONLY',
        full_model_timeline_implemented=False, model_makespan_is_native_runtime_prediction=False,
        native_host_optimality_proven=False, existing_host_envelope='ALL_POOL_SLOTS_PREALLOCATED_JOIN_ALL_BEFORE_EXPORT',
        cost_scope=SCOPE, deployment_authorized=False, default_backend='NO_OP')
    if problem.activities:
        from scar.optimization.pipeline_lifecycle import lifecycle_cost_contexts
        certificate.update(implementation_scope='SELECTED_WORKERS_AND_SOURCE_HOST_LIFECYCLE_ORDER',
            cost_scope='ABI1_AND_COMPLETE_TASK_WRAPPER_PLUS_HOST_LIFECYCLE_POINT_MODEL',
            host_lifecycle=dict(problem_sha256=digest(problem),
                selected_activities=execution['activities'],
                contexts_sha256=semantic_sha256(lifecycle_cost_contexts(problem)),
                selected_worker_cpu_ids=cpus,
                caller_worker_physical_cpu_demands_proven=False,
                model_timestamps_are_runtime_guarantees=False))
    return certificate


__all__ = ['build_pipeline_problem', 'cost_contexts', 'refine_pipeline_implementation',
    'COST_SCHEMA', 'CERT_SCHEMA', 'FAMILY']

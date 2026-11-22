"""Conditional native caller/worker lifecycles on the shared planning IR."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from scar.compiler.pipeline_frontend import pipeline_digest
from scar.ir.execution_recipe import artifact_reference
from scar.ir.implementation_plan import (CostEstimate, ImplementationChoice, PlanningActivity,
    PlanningTimingConstraint, PlanningResource, digest, validate_problem)
from scar.ir.record_codec import decode, encode
from scar.optimization.implementation_costs import CostQuery, query_key, resolve_cost

SCHEMA = 'scar.pipeline-lifecycle-source'
PREFIX = 'pipeline:lifecycle:'
WRAPPER_SCOPE = 'NATIVE_TASK_WRAPPER_BEFORE_AND_AFTER_ABI_FORECAST'
_ADDED = ('leaf_service_cost', 'wrapper_cost_query', 'wrapper_cost', 'task_cost_context',
          'abi_service_demands', 'whole_task_physical_cpu_demand')
_HOST = ('codegen/pipeline_v1.cpp.in', 'codegen/pipeline_ready_v1.cpp.in', 'codegen/pipeline_parallel_v1.cpp.in',
         'codegen/pipeline.py', 'codegen/pipeline_implementation.py',
         'codegen/pipeline.py', 'codegen/pipeline.py', 'codegen/kernel_v1.cpp.in')
_SCOPES = dict(prefix='NATIVE_CALLER_POOL_CONTROL_PREFIX_FORECAST',
    submit='NATIVE_CALLER_EMPLACE_SUBMISSION_FORECAST',
    launch='NATIVE_WORKER_START_LATENCY_FORECAST', exit='NATIVE_WORKER_EXIT_AFTER_ALL_PUBLICATIONS_FORECAST',
    join='JOIN_AFTER_READY_TAIL_FORECAST', cleanup='NATIVE_CALLER_POST_JOIN_CLEANUP_FORECAST')


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _known(cost):
    return cost.estimate_ns is not None and cost.basis != 'UNKNOWN' and not cost.missing_terms


def _profile(reference):
    if isinstance(reference, (str, Path)):
        return artifact_reference(Path(reference).resolve(), 'scar.implementation-cost-profile')
    return None if reference is None else encode(reference)


def _complete_task(leaf, wrapper, context):
    key = 'pipeline-task-forecast:'+digest(context)
    refs = (*leaf.source_refs, *wrapper.source_refs,
        dict(schema='scar.pipeline-task-cost-composition', schema_version=1,
             composition=context['composition'], task_end_means=context['task_end_means'],
             leaf_context=leaf.context, wrapper_context=wrapper.context,
             whole_task_measured=False, empirical_ranges_are_guaranteed_bounds=False))
    if _known(leaf) and _known(wrapper):
        return CostEstimate(leaf.estimate_ns+wrapper.estimate_ns, None, None,
            'ANALYTICAL_MODEL', key, refs, ())
    missing = tuple(dict.fromkeys(reason for name, cost in (('LEAF_SERVICE', leaf), ('WRAPPER', wrapper))
        if not _known(cost) for reason in (name+'_FORECAST_INCOMPLETE',
            *(name+':'+term for term in cost.missing_terms))))
    return CostEstimate(None, None, None, 'UNKNOWN', key, refs, missing)


def attach_pipeline_lifecycle(problem, pipeline, memory, *, cost_profile_ref=None, source_refs=()):
    """Keep every source/CPU option; add guarded host phases and endpoint order.

    Source/target admission belongs to the caller. Only exact forecast query
    scopes are read; raw instrumented OBSERVED_SCOPE costs are not service
    forecasts. No source program, compiler, hardware probe or native is run.
    """
    base = validate_problem(problem)
    _need(not base.activities and not base.timing_constraints,
          'attach lifecycle once to an original leaf-service problem')
    _need(type(source_refs) in (tuple, list) and all(type(r) is dict and r for r in source_refs),
          'original nonempty source reference records required')
    calls = [call.id for call in pipeline.calls]
    _need([op.id for op in base.operations] == calls and not base.data_edges,
          'original fixed-pool pipeline calls must match the planning operations')
    cpus = sorted({m.parameters['cpu_id'] for op in base.operations for m in op.options})
    _need(cpus and all(type(cpu) is int and cpu >= 0 for cpu in cpus), 'original explicit CPU options required')
    resources = {r.id:r for r in base.resources}
    _need(not any(name.startswith(PREFIX) for name in resources), 'lifecycle resource identity collision')
    contexts = [m.parameters['context'] for op in base.operations for m in op.options]
    pipeline_sha = pipeline_digest(pipeline); memory_sha = digest(memory)
    _need(all(c['pipeline_sha256'] == pipeline_sha and c['memory_plan_sha256'] == memory_sha
        and c['target'] == encode(base.target) for c in contexts), 'source/pool/target leaf query context differs')
    root = Path(__file__).resolve().parents[1]
    common = dict(schema='scar.pipeline-lifecycle-cost-context', schema_version=1,
        pipeline_sha256=pipeline_sha, memory_plan_sha256=memory_sha, target=encode(base.target),
        input_shapes=deepcopy(memory['input_shapes']),
        private_workspace_slots=deepcopy(memory['workspace_slots']),
        final_bytes=memory['final_bytes'], memory_budget_bytes=memory['memory_budget_bytes'],
        call_argument_counts={c.id:len(c.inputs) for c in pipeline.calls},
        option_cpus=[dict(operation=o.id, option=m.id, cpu_id=m.parameters['cpu_id'])
            for o in base.operations for m in o.options],
        original_compilers=list({digest(c['compiler']):c['compiler'] for c in contexts}.values()),
        host_sources={name:artifact_reference(root/name, 'scar.backend-implementation-source') for name in _HOST},
        native_scope='NORMAL_CPP_READY_HOST_NOT_PYTHON_WRAPPER',
        active_worker_subset_cost_policy='FIXED_ROLE_CPU_FORECAST_ASSUMES_ACTIVE_SUBSET_AND_LANE_ORDER_INVARIANCE',
        single_configuration_observations_qualify_other_selections=False,
        dispatch_policy='FIXED_CPU_LANE_HEAD_ONLY_WAIT_TRUE_PREDECESSORS_NO_TIMESTAMP_WAIT',
        source_math_changed=False, observed_intervals_promoted_to_forecasts=False)
    profile = _profile(cost_profile_ref); refs = (*base.source_refs, *tuple(deepcopy(source_refs)))
    caller = PREFIX+'logical-caller'
    new_resources = [PlanningResource(caller, 'EXECUTION', 1,
        refs+(dict(logical_caller=True, physical_cpu_demand=False),))]
    for cpu in cpus:
        new_resources.append(PlanningResource(PREFIX+'lane:cpu:'+str(cpu), 'EXECUTION', 1,
            refs+(dict(logical_lane=True, cpu_parameter=cpu, physical_cpu_demand=False,
                       interpretation='NOT_PHYSICAL_CPU_DEMAND'),)))
    activity_queries = []; wrapper_queries = []; operations = []
    def query(scope, role, cpu=None, **details):
        context = dict(common, scope=scope, lifecycle_role=role, cpu_id=cpu,
            cpu_parameter_is_lane_target_not_host_phase_location=True, **details)
        q = CostQuery(scope, context, refs)
        cost=resolve_cost(q,profile)
        if cost.basis=='EMPIRICAL_MODEL':
            cost=CostEstimate(None,None,None,'UNKNOWN',cost.context,cost.source_refs,
                ('SINGLE_CONFIGURATION_SAMPLES_DO_NOT_QUALIFY_SUBSET_INVARIANT_HOST_FORECAST',))
        return dict(scope=scope, context=context, key=query_key(q)),cost
    for op in base.operations:
        modes = []
        for mode in op.options:
            _need(not set(mode.parameters).intersection(_ADDED), 'original mode already contains lifecycle composition')
            cpu = mode.parameters['cpu_id']; leaf = mode.cost
            q, wrapper = query(WRAPPER_SCOPE, 'TASK_WRAPPER', cpu, call_id=op.id,
                original_leaf_parameters=deepcopy(mode.parameters),
                normal_wrapper_scope='CLAIM_ENV_BIND_RESTORE_SUCCESSFUL_PUBLICATION_AND_MUTEX_COMPETITION',
                wrapper_excludes_original_abi_service=True)
            task_context = dict(schema='scar.pipeline-complete-task-forecast-context', schema_version=1,
                call_id=op.id, cpu_id=cpu, leaf_service_context=leaf.context,
                wrapper_cost_context=wrapper.context, source_context_sha256=digest(common),
                composition='ORDERED_DISJOINT_ABI_PLUS_ALL_NORMAL_WRAPPER_WORK',
                task_end_means='EXECUTE_TASK_RESTORE_AND_SUCCESSFUL_PUBLICATION_COMPLETE')
            params = deepcopy(mode.parameters)
            params.update(leaf_service_cost=encode(leaf), wrapper_cost_query=q, wrapper_cost=encode(wrapper),
                task_cost_context=task_context, abi_service_demands=deepcopy(mode.demands),
                whole_task_physical_cpu_demand='UNKNOWN_OUTSIDE_ABI')
            physical = {r for r in mode.demands if resources[r].kind == 'EXECUTION'}
            _need(physical, 'original ABI service hardware-thread demand required')
            demands = {r:u for r,u in mode.demands.items() if r not in physical}
            demands[PREFIX+'lane:cpu:'+str(cpu)] = 1
            modes.append(replace(mode, parameters=params, demands=demands,
                cost=_complete_task(leaf, wrapper, task_context)))
            wrapper_queries.append(dict(operation=op.id, option=mode.id, query=q))
        operations.append(replace(op, options=tuple(modes)))
    guards = {cpu:tuple(ImplementationChoice(op.id,m.id) for op in base.operations for m in op.options
        if m.parameters['cpu_id'] == cpu) for cpu in cpus}
    activities = []
    def activity(role, cpu=None):
        name = PREFIX+role+('' if cpu is None else ':cpu:'+str(cpu))
        q, cost = query(_SCOPES[role], role.upper(), cpu,
            includes_waiting=False if role == 'join' else role in ('launch',),
            tail_after_worker_and_caller_readiness=role == 'join')
        demands = {caller:1} if role in ('prefix','submit','cleanup') else {}
        activities.append(PlanningActivity(name, cost, demands,
            refs+(dict(schema=SCHEMA, role=role.upper(), cpu_id=cpu, query=q),),
            () if cpu is None else guards[cpu], 'WAIT_TAIL' if role == 'join' else 'FIXED'))
        activity_queries.append(dict(activity=name, role=role.upper(), cpu_id=cpu, query=q))
        return name, cost
    prefix, _ = activity('prefix'); cleanup, _ = activity('cleanup'); phases = {}
    for cpu in cpus:
        phases[cpu] = {role:activity(role,cpu) for role in ('submit','launch','exit','join')}
    constraints = []
    def edge(before, be, after, ae, label, *, when=(), minimum=0, maximum=None):
        data = dict(before=before,before_endpoint=be,after=after,after_endpoint=ae,
                    minimum_ns=minimum,maximum_ns=maximum,when=encode(when),label=label)
        constraints.append(PlanningTimingConstraint(PREFIX+'edge:'+digest(data), before, be, after, ae,
            minimum, maximum, when, refs+(dict(schema=SCHEMA, role=label),)))
    for cpu in cpus:
        p = phases[cpu]; submit, launch, exit, join = (p[r][0] for r in ('submit','launch','exit','join'))
        edge(prefix,'END',submit,'START','PREFIX_BEFORE_ACTIVE_SUBMIT')
        edge(submit,'START',launch,'START','ASYNC_START_IS_SUBMIT_START',maximum=0)
        for op in base.operations:
            edge(op.id,'END',exit,'START','ALL_PUBLISHED_CALLS_BEFORE_WORKER_EXIT')
            for mode in op.options:
                if mode.parameters['cpu_id'] == cpu:
                    edge(launch,'END',op.id,'START','SELECTED_WORKER_READY_BEFORE_CALL',
                         when=(ImplementationChoice(op.id,mode.id),))
        for other in cpus:
            edge(phases[other]['submit'][0],'END',join,'START','ALL_ACTIVE_SUBMITS_BEFORE_JOIN')
        tail = p['join'][1]
        edge(exit,'END',join,'END','WORKER_EXIT_BEFORE_JOIN_READY_TAIL',
             minimum=tail.estimate_ns if _known(tail) else 0)
        edge(join,'END',cleanup,'START','ALL_ACTIVE_JOINS_BEFORE_CLEANUP')
    for index, cpu in enumerate(cpus):
        for earlier in cpus[:index]:
            for role in ('submit','join'):
                edge(phases[earlier][role][0],'END',phases[cpu][role][0],'START',
                     'SORTED_ACTIVE_CPU_'+role.upper()+'_ORDER')
    proof = dict(schema=SCHEMA, schema_version=1, original_problem_sha256=digest(base),
        original_source_ref_count=len(base.source_refs), extra_source_refs=encode(source_refs),
        cost_profile_ref=profile, source_context=common, activity_queries=activity_queries,
        wrapper_queries=wrapper_queries, added_resources=[r.id for r in new_resources],
        original_hardware_resources_preserved=True, call_end_means_successful_publication=True,
        joins_are_wait_tails_not_fixed_wait_intervals=True, clean_costs_measured_here=False)
    return validate_problem(replace(base, operations=tuple(operations),
        resources=(*base.resources,*new_resources), activities=tuple(activities),
        timing_constraints=tuple(constraints), source_refs=(*refs,proof),
        assumptions=(*base.assumptions,'HOST_PHASE_FORECASTS_REQUIRE_ACTIVE_WORKER_SUBSET_INVARIANCE',
            'HOST_WRAPPER_PHYSICAL_CPU_DEMAND_UNRESOLVED_OUTSIDE_ABI')))


def lifecycle_cost_contexts(problem):
    """Return real host/wrapper profile query templates, never synthetic sums."""
    p = validate_problem(problem)
    rows = [r for r in p.source_refs if r.get('schema') == SCHEMA]
    _need(len(rows) == 1, 'one original lifecycle derivation required')
    return {r['query']['key']:r['query']['context']
            for r in (*rows[0]['activity_queries'],*rows[0]['wrapper_queries'])}


def verify_pipeline_lifecycle(problem, pipeline, memory):
    """Restore the original problem and reattach every query/phase/constraint."""
    p = validate_problem(problem)
    proofs = [r for r in p.source_refs if r.get('schema') == SCHEMA]
    _need(len(proofs) == 1, 'one lifecycle source record required')
    proof = proofs[0]; operations = []
    for op in p.operations:
        modes = []
        for mode in op.options:
            params = deepcopy(mode.parameters)
            leaf = decode(CostEstimate, params['leaf_service_cost']); demands = params['abi_service_demands']
            for key in _ADDED:params.pop(key)
            modes.append(replace(mode, parameters=params, cost=leaf, demands=demands))
        operations.append(replace(op,options=tuple(modes)))
    count = proof['original_source_ref_count']
    base = replace(p, source_refs=p.source_refs[:count], operations=tuple(operations),
        resources=tuple(r for r in p.resources if r.id not in proof['added_resources']),
        activities=(), timing_constraints=(),assumptions=p.assumptions[:-2])
    _need(digest(base) == proof['original_problem_sha256'], 'original leaf/source/resource problem differs')
    expected = attach_pipeline_lifecycle(base,pipeline,memory,cost_profile_ref=proof['cost_profile_ref'],
        source_refs=proof['extra_source_refs'])
    _need(digest(expected) == digest(p), 'conditional lifecycle/source/cost composition differs from exact replay')
    return problem


__all__ = ['SCHEMA','PREFIX','WRAPPER_SCOPE','attach_pipeline_lifecycle',
           'lifecycle_cost_contexts','verify_pipeline_lifecycle']

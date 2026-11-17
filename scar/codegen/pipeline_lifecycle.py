"""Source witness for conditional lifecycle models over the existing ready host.

The witness implements no new executor and does not turn model timestamps into
native timing guarantees. Original F4, cost-profile and full-source admission
remain the qualified producer's responsibility.
"""
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from scar.compiler.pipeline_model import pipeline_digest
from scar.ir.implementation_plan import CostEstimate, validate_problem, digest
from scar.ir.record_codec import decode
from scar.optimization.joint_planner import verify_execution_plan
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.pipeline-lifecycle-delivery-proof'
_PREFIX = 'pipeline:lifecycle:'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _span(source, start, end=None):
    _need(source.count(start) == 1, 'unique original lifecycle source marker required')
    begin = source.index(start)
    finish = begin+len(start)
    if end is not None:
        _need(source.count(end) == 1 and source.index(end) >= finish,
            'unique ordered original lifecycle end marker required')
        finish = source.index(end)+len(end)
    raw = source[begin:finish]
    return dict(start_line=source[:begin].count('\n')+1,
        end_line=source[:finish].count('\n')+1, sha256=sha256(raw.encode()).hexdigest())


def _source(pipeline, memory, options, certificate):
    from .pipeline import _source as serial_source
    from .pipeline_implementation import render_implementation_source
    # These are source descriptor rows, not runnable CppArtifact instances.
    # The original renderer consumes only output budget, plan SHA and whether
    # timed input guards are enabled. Actual operators are separately admitted.
    descriptors = [SimpleNamespace(
        plan=SimpleNamespace(cost={'memory_budget_bytes':options[call.id].parameters['resource_binding']['memory_budget_bytes']}),
        report={'plan_sha256':options[call.id].parameters['bound_leaf_plan_sha256']}, timed_admission=None)
        for call in pipeline.calls]
    return render_implementation_source(serial_source(pipeline,memory,descriptors),pipeline,memory,certificate)


def lifecycle_delivery_proof(pipeline, memory, problem, execution, certificate, source):
    """Check normal lifecycle selection and bind its roles to exact C++ bytes."""
    problem = validate_problem(problem)
    _need(problem.activities and problem.timing_constraints and problem.normal_completion_only,
        'explicit normal conditional host lifecycle model required')
    from scar.optimization.pipeline_lifecycle import verify_pipeline_lifecycle
    from scar.optimization.implementation_costs import CostQuery, query_key
    from scar.optimization.pipeline_implementation import SCOPE, LOWERING
    verify_pipeline_lifecycle(problem,pipeline,memory)
    verify_execution_plan(problem,execution)
    _need(type(source) is bytes and execution['domain_complete'] is True and not execution['frontiers'],
        'complete known model and original generated source bytes required')
    calls = [call.id for call in pipeline.calls]
    _need([op.id for op in problem.operations] == calls, 'original call domain differs')
    selections = {row['operation']:row for row in execution['selected']}
    selected = {(row['operation'],row['option']) for row in execution['selected']}
    options = {op.id:next(mode for mode in op.options if mode.id == selections[op.id]['option'])
        for op in problem.operations}
    resources = {resource.id:resource for resource in problem.resources}
    for name,mode in options.items():
        parameters = mode.parameters
        _need(mode.lowering == LOWERING, 'lifecycle delivery reuses only existing singleton-CPU ABI1 leaves')
        abi_demands = parameters['abi_service_demands']
        abi_execution = {resource for resource in abi_demands if resources[resource].kind == 'EXECUTION'}
        lane = _PREFIX+'lane:cpu:'+str(parameters['cpu_id'])
        _need(len(abi_execution) == 1 and lane in resources and resources[lane].capacity == 1
            and resources[lane].kind == 'EXECUTION'
            and any(ref.get('logical_lane') is True and ref.get('physical_cpu_demand') is False
                for ref in resources[lane].source_refs)
            and mode.demands == {**{resource:units for resource,units in abi_demands.items()
                if resource not in abi_execution},lane:1}
            and parameters['whole_task_physical_cpu_demand'] == 'UNKNOWN_OUTSIDE_ABI',
            'complete task must claim a logical lane, preserving physical ABI demands separately')
        leaf = decode(CostEstimate,parameters['leaf_service_cost'])
        wrapper = decode(CostEstimate,parameters['wrapper_cost'])
        query, context = parameters['wrapper_cost_query'], parameters['task_cost_context']
        _need(query['scope'] == 'NATIVE_TASK_WRAPPER_BEFORE_AND_AFTER_ABI_FORECAST'
            and query_key(CostQuery(query['scope'],query['context'])) == query['key'] == wrapper.context
            and query_key(CostQuery(SCOPE,parameters['context'])) == leaf.context,
            'task components require the exact original ABI and complete wrapper query scopes')
        _need(context.get('schema') == 'scar.pipeline-complete-task-forecast-context'
            and context.get('schema_version') == 1 and context['call_id'] == name
            and context['cpu_id'] == parameters['cpu_id']
            and context['leaf_service_context'] == leaf.context
            and context['wrapper_cost_context'] == wrapper.context
            and context['composition'] == 'ORDERED_DISJOINT_ABI_PLUS_ALL_NORMAL_WRAPPER_WORK'
            and context['task_end_means'] == 'EXECUTE_TASK_RESTORE_AND_SUCCESSFUL_PUBLICATION_COMPLETE',
            'ABI-only service cannot stand in for a complete published task')
        _need(all(part.basis != 'UNKNOWN' and part.estimate_ns is not None and not part.missing_terms
                for part in (leaf,wrapper)) and mode.cost.basis == 'ANALYTICAL_MODEL'
            and mode.cost.estimate_ns == leaf.estimate_ns+wrapper.estimate_ns
            and not mode.cost.missing_terms and mode.cost.lower_ns is None and mode.cost.upper_ns is None,
            'known disjoint ABI plus wrapper point forecasts required, without invented guaranteed bounds')
    cpus = sorted({mode.parameters['cpu_id'] for mode in options.values()})
    _need(certificate['cpu_ids'] == cpus
        and certificate['pipeline_sha256'] == pipeline_digest(pipeline)
        and certificate['memory_plan_sha256'] == semantic_sha256(memory)
        and certificate['problem_sha256'] == digest(problem)
        and certificate['execution_plan_sha256'] == semantic_sha256(execution)
        and certificate['call_cpu_ids'] == {name:mode.parameters['cpu_id'] for name,mode in options.items()},
        'lifecycle selection/source/pool/certificate identities differ')
    text = source.decode()
    _need(_source(pipeline,memory,options,certificate).encode() == source,
        'lifecycle host must be the exact current original renderer, including launch/join and guards')

    all_cpus = sorted({mode.parameters['cpu_id'] for op in problem.operations for mode in op.options})
    conditions = {cpu:{(op.id,mode.id) for op in problem.operations for mode in op.options
        if mode.parameters['cpu_id'] == cpu} for cpu in all_cpus}
    expected = {_PREFIX+'prefix':set(), _PREFIX+'cleanup':set()}
    for cpu in all_cpus:
        for role in ('submit','launch','exit','join'):
            expected[_PREFIX+role+':cpu:'+str(cpu)] = conditions[cpu]
    activities = {row.id:row for row in problem.activities}
    _need(set(activities) == set(expected), 'conditional lifecycle activity domain differs')
    for name, choices in expected.items():
        actual = {(choice.operation,choice.option) for choice in activities[name].when}
        _need(actual == choices and len(actual) == len(activities[name].when),
            'lifecycle worker condition must be exactly OR of all original options selecting its CPU')
    active = {name for name,choices in expected.items() if not choices or choices & selected}
    rows = execution.get('activities')
    _need(type(rows) is list and len(rows) == len(active)
        and {row['activity'] for row in rows} == active, 'selected lifecycle activities differ from original conditions')
    times = {row['activity']:row for row in rows}
    def time(name, endpoint):
        row = times[name] if name in times else selections[name]
        return row[endpoint+'_ns']
    def before(left, left_end, right, right_end):
        _need(time(left,left_end) <= time(right,right_end), 'selected model contradicts original C++ lifecycle order')
    prefix, cleanup = _PREFIX+'prefix', _PREFIX+'cleanup'
    for position,cpu in enumerate(cpus):
        names = {role:_PREFIX+role+':cpu:'+str(cpu) for role in ('submit','launch','exit','join')}
        before(prefix,'end',names['submit'],'start')
        _need(time(names['launch'],'start') == time(names['submit'],'start'), 'worker launch origin must be its actual emplace submission start')
        _need(activities[names['join']].duration_kind == 'WAIT_TAIL' and not activities[names['join']].demands,
            'join waiting must not be charged as a fixed busy-CPU service')
        for call in calls:
            before(call,'end',names['exit'],'start')
            if certificate['call_cpu_ids'][call] == cpu:
                before(names['launch'],'end',call,'start')
        for earlier in cpus[:position]:
            before(_PREFIX+'submit:cpu:'+str(earlier),'end',names['submit'],'start')
            before(_PREFIX+'join:cpu:'+str(earlier),'end',names['join'],'start')
        for submitted in cpus:
            before(_PREFIX+'submit:cpu:'+str(submitted),'end',names['join'],'start')
        before(names['exit'],'end',names['join'],'end')
        before(names['join'],'end',cleanup,'start')

    roles = {
        'prefix':dict(actor='CALLER_UNBOUND',source_fragments=[_span(text,'    FloatingEnvironment caller;'),
            _span(text,'    if (count != input_count || output == nullptr || (count && inputs == nullptr) ||'),
            _span(text,'    std::vector<std::unique_ptr<double[]>> slots(slot_capacities.size());'),
            _span(text,'        workers.reserve(lane_cpu_ids.size());')],
            meaning='CALLER_ENV_PREFLIGHT_POOL_AND_CONTROL_BEFORE_FIRST_SUBMISSION'),
        'submit':dict(actor='CALLER_UNBOUND',span=_span(text,'        for (int cpu : lane_cpu_ids) workers.emplace_back(worker, cpu);'),
            meaning='ORDERED_EMPLACE_SUBMISSION_MAY_OVERLAP_WORKER_ENTRY'),
        'launch':dict(actor='WORKER_ORIGIN_UNBOUND',span=_span(text,'    auto worker = [&](int cpu) {',
            '            std::unique_lock lock(mutex);'),
            meaning='ASYNCHRONOUS_SUBMISSION_TO_LAMBDA_ENTRY_NOT_SUBMIT_RETURN_BARRIER'),
        'call':dict(actor='WORKER_TASK',source_fragments=[
            _span(text,'            const size_t index = next_for_cpu(cpu);'),
            _span(text,'            const int32_t admission = admit_task(index);','            changed.notify_all();')],
            meaning='CLAIM_SELECTION_ADMISSION_EXECUTE_RESTORE_AND_SUCCESS_PUBLICATION',
            publication='MUTEX_RELEASE_AFTER_NOTIFY_NOT_ABI_RETURN', physical_CPU_for_entire_task_proven=False),
        'exit':dict(actor='WORKER_ORIGIN_UNBOUND',span=_span(text,'            if (stop || succeeded == call_plans.size()) return;'),
            meaning='NORMAL_LAMBDA_EXIT_REQUIRES_ALL_CALLS_SUCCESSFULLY_PUBLISHED'),
        'join':dict(actor='CALLER_WAIT',span=_span(text,'    for (auto &thread : workers) if (thread.joinable()) thread.join();'),
            meaning='CREATION_ORDER_JOIN_START_CAN_PRECEDE_WORKER_EXIT', duration_kind='WAIT_TAIL'),
        'cleanup':dict(actor='CALLER_UNBOUND',source_fragments=[
            _span(text,'    if (succeeded != call_plans.size() || !ready[final_value]) return lifetime_guard;'),
            _span(text,'        try { status = run_ready(inputs, count, output, operators, operator_count, failed_call, caller.entry()); }'),
            _span(text,'    return caller.restore() ? status : worker_guard;')],
            meaning='JOINED_LOCAL_RAII_AND_SLOTS_DESTRUCTION_BEFORE_CALLER_RESTORE_AND_ABI_RETURN',
            pool_release_order='STD_VECTOR_IMPLEMENTATION_NOT_ASSERTED_PER_SLOT_REVERSE'),
    }
    here = Path(__file__).resolve()
    return dict(schema=SCHEMA,schema_version=1,source_sha256=sha256(source).hexdigest(),
        implementation_sha256=sha256(here.read_bytes()).hexdigest(),problem_sha256=digest(problem),
        execution_plan_sha256=semantic_sha256(execution),certificate_sha256=semantic_sha256(certificate),
        selected_cpu_ids=cpus,selected_activities=[row['activity'] for row in rows],
        activities=[dict(id=name,source_role=name[len(_PREFIX):].split(':')[0],when=[dict(operation=a,option=b)
            for a,b in sorted(expected[name])]) for name in sorted(active)],source_roles=roles,
        source_reconstruction_equal=True,mathematical_operators_changed=False,thread_executor_changed=False,
        complete_task_forecast_includes_abi_and_normal_wrapper=True,
        normal_completion_only=True,model_time_is_runtime_guarantee=False,
        full_model_timeline_implemented=False,exact_clock_cut_correspondence_claimed=False,
        physical_resource_feasibility_proven=False,target_program_executed=False,
        deployment_authorized=False,default_backend='NO_OP',
        original_source_and_cost_authority='QUALIFIED_PARENT_PRODUCER_REPLAY_REQUIRED')


__all__ = ['lifecycle_delivery_proof','SCHEMA']

"""Finite CPU resource scenarios over replayed caller values and arena plans."""
from itertools import combinations

from scar.compiler.pipeline_model import pipeline_digest
from scar.ir.program.target_hardware import CpuCore, CpuExecutor, HardwareThread
from .hardware import _indexed_resources, _record_evidence, _target_materials
from .implementation_costs import CostQuery, query_key, resolve_cost
from .pipeline_memory import plan_pipeline_memory
from .pipeline_partial_order_memory import plan_pipeline_partial_order_memory
from .pipeline_ready_schedule import plan_pipeline_ready_schedule
from .resource_solver import semantic_sha256


def _eligible_cpu_sets(target, affinity):
    graph, snapshot, resources, _ = _target_materials(target)
    indexed = _indexed_resources(resources)
    online = {x.cpu_id: x for x in resources if type(x) is HardwareThread and x.online is not False}
    available = set(affinity) & online.keys()
    cores = {}
    for core in (x for x in resources if type(x) is CpuCore):
        for cpu in available.intersection(core.cpu_ids):
            cores.setdefault(cpu, []).append(core)
    executors = [x for x in resources if type(x) is CpuExecutor]
    per_cpu = {cpu: [executor for executor in executors if cpu in executor.cpu_ids]
               for cpu in sorted(available)}
    admitted = [cpu for cpu, owners in per_cpu.items() if len(owners) == 1]
    pair = next((pair for pair in combinations(admitted, 2)
        if all(len(cores.get(cpu, ())) == 1 for cpu in pair)
        and cores[pair[0]][0].id != cores[pair[1]][0].id
        and per_cpu[pair[0]][0].numa_node is not None
        and per_cpu[pair[0]][0].numa_node == per_cpu[pair[1]][0].numa_node), ())
    serial = pair[:1] or tuple(admitted[:1])
    def evidence(cpu):
        owner, = per_cpu[cpu]
        cpu_cores = cores.get(cpu, ())
        return dict(cpu_id=cpu, numa_node=owner.numa_node,
            thread=_record_evidence(graph, snapshot, online[cpu], indexed[str(online[cpu].id)][1]),
            executor=_record_evidence(graph, snapshot, owner, indexed[str(owner.id)][1]),
            core=_record_evidence(graph, snapshot, cpu_cores[0], indexed[str(cpu_cores[0].id)][1])
            if len(cpu_cores) == 1 else None)
    return serial, pair, {cpu: evidence(cpu) for cpu in set(serial) | set(pair)}


def _ordered_calls(ready):
    """The diagnostic host realizes fixed lane heads and true predecessors."""
    lanes = {str(cpu): [] for cpu in ready['cpu_ids']}
    ancestors, ranks, combined = {}, {}, {}
    previous = {}
    for call in ready['call_order']:
        cpu = str(ready['call_cpu_ids'][call])
        deps = set(ready['call_predecessors'][call])
        if cpu in previous:
            deps.add(previous[cpu])
        combined[call] = sorted(deps)
        ancestors[call] = deps.union(*(ancestors[dep] for dep in deps))
        ranks[call] = 1 + max((ranks[dep] for dep in deps), default=0)
        lanes[cpu].append(call)
        previous[cpu] = call
    incomparable = [[first, second] for first, second in combinations(ready['call_order'], 2)
        if ready['call_cpu_ids'][first] != ready['call_cpu_ids'][second]
        and first not in ancestors[second]]
    successors = {call: [] for call in ready['call_order']}
    for after, parents in ready['call_predecessors'].items():
        for before in parents:
            if ready['call_cpu_ids'][before] != ready['call_cpu_ids'][after]:
                successors[before].append(after)
    return dict(lane_call_order=lanes, schedule_predecessors=combined,
        longest_dependency_and_lane_path_call_count=max(ranks.values()),
        potentially_overlapping_cross_lane_calls=incomparable,
        cross_lane_predecessors={call: children for call, children in successors.items() if children})


def domain_cases(pipeline, source_memory, source_ready, target, source_values, *, search_budget):
    """Construct one retained original and the declared CPU × arena domain."""
    serial, pair, evidence = _eligible_cpu_sets(target, source_ready['caller_affinity_cpu_ids'])
    plans = {name: planner(pipeline, input_shapes=source_memory['input_shapes'],
        memory_budget_bytes=source_memory['memory_budget_bytes'], search_budget=search_budget)
        for name, planner in (('source', plan_pipeline_memory), ('dependencies', plan_pipeline_partial_order_memory))}
    yield dict(id='original', implementation='ORIGINAL_PYTHON', arena_policy=None,
        legal_status='ORIGINAL_EXECUTION', cpu_ids=None, hardware_bindings=[], legality_frontier=[],
        schedules=None, dispatch_policy='ORIGINAL_PYTHON_CALL_ORDER',
        call_order=source_ready['call_order'], peak_payload_bytes=None), None, None
    for implementation, cpus in (('cpu-serial', serial), ('cpu-ready', pair)):
        for arena, memory in plans.items():
            issues = []
            if not cpus:
                issues.append('NO_REPLAYED_SAME_NUMA_DISTINCT_PHYSICAL_CORE_PAIR' if implementation == 'cpu-ready'
                    else 'NO_REPLAYED_ONLINE_AFFINITY_CPU')
            if memory['status'].startswith('INFEASIBLE'):
                issues.append('ARENA_CAPACITY_LOWER_BOUND_EXCEEDS_DECLARED_PAYLOAD_BUDGET')
            elif memory['search_budget_exhausted'] or memory['frontier'] or not memory['budget_satisfied']:
                issues.append('ARENA_LIFETIMES_OR_FINITE_CAPACITY_SEARCH_UNRESOLVED')
            ready = None if issues else plan_pipeline_ready_schedule(pipeline, memory, target,
                cpu_ids=cpus, input_graph=source_values)
            yield dict(id=implementation+':'+arena, implementation=implementation, arena_policy=arena,
                legal_status=('ADMITTED_EXPLICIT_TRIAL' if ready is not None else 'UNRESOLVED'
                    if 'ARENA_LIFETIMES_OR_FINITE_CAPACITY_SEARCH_UNRESOLVED' in issues else 'INELIGIBLE'),
                cpu_ids=list(cpus), hardware_bindings=[evidence[cpu] for cpu in cpus],
                legality_frontier=issues, peak_payload_bytes=memory['peak_allocation_bytes'],
                dispatch_policy='FIXED_CPU_LANE_HEADS_WAIT_ON_CALL_AND_SLOT_PREDECESSORS',
                call_order=source_ready['call_order'],
                schedules=None if ready is None else _ordered_calls(ready)), memory, ready


_ORIGINAL_SCOPE = 'ORIGINAL_SINGLE_PROTECTED_CALL_WARM_COMPOSED_TOTAL'
_REGIME = 'FIXED_LANE_HEADS_WARM_GUARDED_CXX20_ABI1'


def _cost(role, query, profile, *, bases=('EMPIRICAL_MODEL', 'ANALYTICAL_MODEL')):
    estimate = resolve_cost(query, profile)
    missing = list(estimate.missing_terms)
    if estimate.basis not in bases and estimate.basis != 'UNKNOWN':
        missing.append('COST_BASIS_IS_NOT_QUALIFIED_FOR_PHYSICAL_COMPOSITION')
    if any(getattr(estimate, field) is None for field in ('estimate_ns', 'lower_ns', 'upper_ns')):
        if not missing:
            missing.append('FULL_CONDITIONAL_INTERVAL_REQUIRED')
    return dict(role=role, key=query_key(query), query=dict(scope=query.scope, context=query.context),
        basis=estimate.basis, estimate_ns=None if missing else estimate.estimate_ns,
        lower_ns=None if missing else estimate.lower_ns, upper_ns=None if missing else estimate.upper_ns,
        evidence=list(estimate.source_refs), missing_terms=missing)


def case_cost(pipeline, memory, ready, row, source_context, leaf_problem, profile):
    """Compose guarded scope terms and only count shared lifecycle work once."""
    if row['legal_status'] in ('UNRESOLVED', 'INELIGIBLE'):
        return dict(status='NO_ADMITTED_NATIVE_CASE', terms=[], predicted_ns=None)
    if row['implementation'] == 'ORIGINAL_PYTHON':
        context = dict(source_context, implementation='ORIGINAL_PYTHON',
            regime='UNCHANGED_WARM_CALLER_REGION', sample_unit='ONE_INVOCATION')
        term = _cost('original_region', CostQuery(_ORIGINAL_SCOPE, context), profile,
            bases=('ANALYTICAL_MODEL',))
        return _summarize_terms([term], None)

    machine = dict(cpu_ids=row['cpu_ids'], call_cpu_ids=ready['call_cpu_ids'],
        memory_policy=ready['memory_policy'], caller_affinity=ready['caller_affinity_cpu_ids'])
    context = dict(source_context, implementation=row['implementation'], regime=_REGIME,
        sample_unit='ONE_INVOCATION', cpu_resources=machine,
        memory_plan_sha256=semantic_sha256(memory), peak_payload_bytes=memory['peak_allocation_bytes'],
        workspace_slot_capacities=[x['capacity_bytes'] for x in memory['workspace_slots']],
        final_bytes=memory['final_bytes'])
    schedules = row['schedules']
    terms = []
    for role in ('entry_guards_and_ffi', 'fresh_final_allocation_and_publish'):
        terms.append(_cost(role, CostQuery(f'{_REGIME}/{role}',
            dict(context, scope_qualification='EXCLUSIVE_HOST_WORK_OUTSIDE_CALL_AND_LANE_CRITICAL_PATH')), profile))
    for role in ('workspace_arena_lifecycle', 'worker_start_and_join'):
        terms.append(_cost(role, CostQuery(f'{_REGIME}/{role}',
            dict(context, scope_qualification='EXCLUSIVE_CRITICAL_PATH_RESIDUAL_AFTER_ACCOUNTING_FOR_OVERLAP')),
            profile, bases=('ANALYTICAL_MODEL',)))
    if schedules['cross_lane_predecessors']:
        terms.append(_cost('dependency_handoffs', CostQuery(f'{_REGIME}/dependency_handoffs',
            dict(context, cross_lane_predecessors=schedules['cross_lane_predecessors'],
                scope_qualification='CRITICAL_PATH_RESIDUAL_BEYOND_ALL_ABI1_SERVICES_AND_WORKER_LIFECYCLE')),
            profile, bases=('ANALYTICAL_MODEL',)))
    possible_overlap = schedules['potentially_overlapping_cross_lane_calls']
    from .pipeline_implementation import SCOPE
    for operation in leaf_problem.operations:
        call = operation.id
        option = next(mode for mode in operation.options if mode.parameters['cpu_id'] == ready['call_cpu_ids'][call])
        leaf = option.parameters['context']
        partners = [pair[1-pair.index(call)] for pair in possible_overlap if call in pair]
        if partners:
            # A warmed isolated ABI1 probe cannot stand in for simultaneous
            # demand on the memory hierarchy. Its exact concurrent scope is separate.
            query = CostQuery(f'{_REGIME}/abi1_service_with_possible_overlap',
                dict(context, abi1_service=leaf, possibly_overlapping_calls=partners))
        else:
            query = CostQuery(SCOPE, leaf)
        terms.append(dict(_cost('abi1_service', query, profile), call=call))
        terms.append(dict(_cost('call_wrapper', CostQuery(f'{_REGIME}/call_wrapper',
            dict(context, call=call, scope_qualification='EXCLUDES_ABI1_SERVICE_AND_DEPENDENCY_WAIT')), profile),
            call=call))
    return _summarize_terms(terms, schedules['schedule_predecessors'])


def _summarize_terms(terms, predecessors):
    missing = [dict(role=term['role'], call=term.get('call'), key=term['key'],
        reasons=term['missing_terms']) for term in terms if term['missing_terms']]
    summary = None
    if not missing:
        summary = {}
        for field in ('estimate_ns', 'lower_ns', 'upper_ns'):
            shared = sum(term[field] for term in terms if 'call' not in term)
            core = {}
            weights = {}
            for term in (term for term in terms if 'call' in term):
                call = term['call']
                weights[call] = weights.get(call, 0) + term[field]
            for call, weight in weights.items():
                core[call] = weight + max((core[parent] for parent in predecessors[call]), default=0)
            summary[field] = shared + max(core.values(), default=0)
    return dict(status='CONDITIONAL_COMPONENT_MODEL' if summary else 'UNRESOLVED',
        terms=terms, missing=missing, predicted_ns=summary,
        composition='SHARED_EXCLUSIVE_HOST_TERMS_ONCE_PLUS_WEIGHTED_CALL_AND_LANE_CRITICAL_PATH',
        interval_is_physical_guarantee=False)


def cost_context(pipeline, source_memory, source_ready, source_inputs):
    source = pipeline.source
    return dict(pipeline_sha256=pipeline_digest(pipeline), target=source_ready['target'],
        input_shapes=source_memory['input_shapes'], input_layout='CPU_NUMPY_C_CONTIGUOUS_FLOAT64',
        memory_budget_bytes=source_memory['memory_budget_bytes'],
        caller_source=dict(path=source.path, sha256=source.sha256, qualname=source.qualname,
            line=source.line, column=source.column, code_key=source.code_key,
            origins_sha256=semantic_sha256(source.origins)), original_proof_sha256=source_inputs['proof']['sha256'])


def decide(cases):
    """Select only interval-separated conditional improvements over original."""
    eligible = [row for row in cases if row['legal_status'] != 'INELIGIBLE']
    unresolved = [row['id'] for row in eligible if row['cost']['predicted_ns'] is None]
    original = cases[0]['cost']['predicted_ns']
    priced = [row for row in eligible if row['cost']['predicted_ns'] is not None]
    best = min(priced, key=lambda row: (row['cost']['predicted_ns']['estimate_ns'], row['id'])) if priced else None
    selected = None
    if unresolved:
        reason = 'REQUIRED_COST_OR_LEGALITY_UNRESOLVED'
    elif best['id'] == 'original':
        reason = 'ORIGINAL_HAS_LOWEST_CONDITIONAL_POINT_COST'
    else:
        upper = best['cost']['predicted_ns']['upper_ns']
        alternative_lower = min(row['cost']['predicted_ns']['lower_ns'] for row in priced if row is not best)
        if upper >= alternative_lower or 100*upper > 95*original['lower_ns']:
            reason = 'COST_INTERVALS_NOT_SEPARATED_OR_MINIMUM_GAIN_NOT_PROVEN_WITHIN_MODEL'
        else:
            selected = best['id']
            reason = 'CONDITIONALLY_LEAST_COST_WITH_INTERVAL_SEPARATION_AND_MINIMUM_GAIN'
    return dict(deployment_default='NO_OP', selected_case_id=selected,
        selection_decision='COST_SELECTED_EXPLICIT_CASE_ONLY' if selected else 'NO_OP',
        reason=reason, unresolved_case_ids=unresolved, minimum_gain_fraction=dict(numerator=1, denominator=20),
        full_program_variant_search_performed=False, deployment_authorized=False)

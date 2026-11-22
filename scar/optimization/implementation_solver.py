"""Finite declared implementation choices with fixed materialization locations.

This searches a mathematical domain, never program variants. Implementation
legality and cost must have retained declarations; they are not inferred from
an implementation name. Unknown options remain explicit domain frontiers.
"""

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from itertools import product
from pathlib import Path

from scar.ir.program.providers import ProviderFactReference
from scar.ir.record_codec import decode, encode, loads
from .cost_model import _pointer
from .resource_solver import _prepare_fixed_schedule, _solve_fixed_schedule, semantic_sha256
from .schedule import validate_schedule


def task_implementation_contract(schedule, task_id) -> dict:
    """Bind a choice to exact work/effects/dependencies and unchanged storage."""
    validate_schedule(schedule)
    task = next((task for task in schedule.tasks if task.id == task_id), None)
    if task is None:
        raise ValueError('unknown task ID')
    return _contract_reader(schedule)(task)


def _contract_reader(schedule):
    # Keep declaration order without walking unrelated graph values per task.
    materials_by_id = {record.id: (position, record) for position, record in enumerate(schedule.materializations)}
    values_by_id = {record.id: (position, record) for position, record in enumerate(schedule.values)}

    def contract(task):
        materials = [record for _, record in sorted(materials_by_id[key] for key in set(task.reads + task.writes))]
        value_ids = {material.value for material in materials}
        values = [record for _, record in sorted(values_by_id[key] for key in value_ids)]
        return dict(schema='scar.task-implementation-contract', schema_version=1,
                    region_id=schedule.region_id, call_scope=schedule.call_scope,
                    task=encode(task), materializations=encode(materials), values=encode(values),
                    loop_axes=encode(schedule.loop_axes))

    return contract


def solve_implementation_choices(schedule, *, task_options, resource_capacities,
                                 memory_capacities=None, materialization_sizes=None,
                                 memory_locations=None, model_instantiation=None,
                                 search_budget=100000) -> dict:
    """Search a finite, sourced option domain plus its resource-task schedules.

    Each option has id, duration_ticks, resource_demands, contract_sha256,
    cost_evidence, legality_conditions and legality_evidence. The cost evidence
    must reference matching integer ticks. At least one legality record must
    bind the exact contract SHA, resource_demands and nonempty conditions.
    Conditions are retained external/model obligations, not global proofs.

    Materialization locations, memory sizes/budgets, task data and dependencies
    remain fixed. No COPY or other task is inserted. The global search budget
    counts choice branches plus fixed-solver states. Unknown options cannot
    yield full-domain optimality, even when a known-subset incumbent is found.
    """
    graph = _prepare_fixed_schedule(schedule)
    if type(search_budget) is not int or search_budget < 1:
        raise ValueError('search_budget must be a positive integer')
    tasks, ids = graph.tasks, graph.task_ids
    contract_for = _contract_reader(schedule)
    frontiers, model_issues, cache = [], [], {}
    if not isinstance(task_options, Mapping):
        task_options = {}
        model_issues.append(dict(field='task_options', reason='explicit finite option mapping required'))
    if not isinstance(resource_capacities, Mapping):
        resource_capacities = {}
        model_issues.append(dict(field='resource_capacities', reason='explicit resource capacities required'))
    for resource, capacity in resource_capacities.items():
        if not isinstance(resource, str) or not resource or type(capacity) is not int or capacity < 1:
            model_issues.append(dict(field='resource_capacities.' + str(resource), reason='positive integer capacity required'))
    for key in task_options:
        if key not in tasks:
            model_issues.append(dict(field='task_options.' + str(key), reason='option domain refers to an absent task'))

    def evidence(records, matcher):
        if not isinstance(records, (tuple, list)) or not records:
            return False
        matched = False
        for record in records:
            reference = decode(ProviderFactReference, record)
            artifact = reference.artifact
            key = (artifact.uri, artifact.sha256, artifact.size_bytes, artifact.schema)
            if key not in cache:
                path = Path(artifact.uri)
                if not path.is_absolute():
                    raise ValueError('absolute retained artifact required')
                raw = path.read_bytes()
                if len(raw) != artifact.size_bytes or sha256(raw).hexdigest() != artifact.sha256:
                    raise ValueError('retained option artifact SHA/size differs')
                data = loads(raw)
                if isinstance(data, dict) and data.get('schema') != artifact.schema:
                    raise ValueError('option artifact schema differs')
                cache[key] = data
            matched = matcher(_pointer(cache[key], reference.locator)) or matched
        return matched

    choices, incomplete = {}, set()
    allowed = {'id', 'duration_ticks', 'resource_demands', 'contract_sha256',
               'cost_evidence', 'legality_conditions', 'legality_evidence'}
    for key in ids:
        options = task_options.get(key)
        choices[key] = []
        if not isinstance(options, (tuple, list)) or not options:
            frontiers.append(dict(task=key, field='task_options.' + key, reason='nonempty finite option domain required'))
            incomplete.add(key)
            continue
        seen = set()
        contract_hash = semantic_sha256(contract_for(tasks[key]))
        for index, option in enumerate(options):
            field = f'task_options.{key}.{index}'
            reasons = []
            if not isinstance(option, Mapping):
                frontiers.append(dict(task=key, field=field, reason='option record required'))
                incomplete.add(key)
                continue
            option = deepcopy(dict(option))
            identifier = option.get('id')
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                reasons.append('nonempty unique option ID required')
            elif identifier:
                seen.add(identifier)
            if set(option) - allowed:
                reasons.append('options cannot add data, locations, effects, tasks or other unsupported constraints')
            duration = option.get('duration_ticks')
            if type(duration) is not int or duration < 0:
                reasons.append('explicit nonnegative duration_ticks required; unknown is not zero')
            demands = option.get('resource_demands')
            if not isinstance(demands, Mapping) or any(not isinstance(r, str) or not r or type(u) is not int or u < 1
                                                        for r, u in demands.items()):
                reasons.append('explicit resource-to-positive-units demands required')
            elif any(r not in resource_capacities for r in demands):
                reasons.append('each demanded resource needs an explicit capacity')
            if option.get('contract_sha256') != contract_hash:
                reasons.append('option must preserve the exact original task/storage/transport contract')
            attrs = tasks[key].attrs
            if 'cpu_threads' in attrs or 'executor_id' in attrs:
                executor, workers = attrs.get('executor_id'), attrs.get('cpu_threads')
                if (not isinstance(executor, str) or not executor or type(workers) is not int
                        or workers < 1 or not isinstance(demands, Mapping)
                        or dict(demands) != {executor: workers}):
                    reasons.append('bound CPU task attributes are fixed; resource alternatives require an explicitly unbound semantic task')
            for reference in tasks[key].resource_refs:
                if reference.get('kind') == 'cpu_executor_binding':
                    cpus = reference.get('cpu_ids')
                    executor, workers = attrs.get('executor_id'), attrs.get('cpu_threads')
                    refs = reference.get('refs', ())
                    declared_ids = [ref.get('resource', {}).get('wire') for ref in refs if isinstance(ref, Mapping)]
                    if (not isinstance(cpus, list) or not cpus or len(cpus) != workers
                            or executor not in declared_ids or resource_capacities.get(executor) != len(cpus)):
                        reasons.append('bound CPU resource references/capacity must preserve the recorded CPU allocation')
            conditions = option.get('legality_conditions')
            if not isinstance(conditions, list) or not conditions or any(not isinstance(c, str) or not c.strip() for c in conditions):
                reasons.append('explicit nonempty legality conditions required')
            try:
                def cost_matches(record):
                    value = record.get('duration_ticks') if isinstance(record, Mapping) else record
                    return type(value) is int and type(duration) is int and value == duration

                if not evidence(option.get('cost_evidence'), cost_matches):
                    reasons.append('matching retained duration-model evidence required')
                expected = dict(contract_sha256=contract_hash, resource_demands=demands, conditions=conditions)
                def legality_matches(record):
                    return (isinstance(record, Mapping) and all(name in record for name in expected)
                            and semantic_sha256({name: record[name] for name in expected}) == semantic_sha256(expected))

                if not evidence(option.get('legality_evidence'), legality_matches):
                    reasons.append('matching retained contract/resource/conditions evidence required')
            except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
                reasons.append('invalid retained option evidence: ' + str(error))
            if reasons:
                incomplete.add(key)
                frontiers.extend(dict(task=key, option=identifier, field=field, reason=reason) for reason in reasons)
            else:
                choices[key].append(option)
        choices[key].sort(key=lambda option: option['id'])

    def critical_bound(durations):
        finish = {}
        for key in graph.order:
            finish[key] = durations[key] + max((finish[d] for d in tasks[key].depends_on), default=0)
        return max(finish.values(), default=0)

    # A zero here is a mathematical nonnegativity relaxation for an unknown
    # frontier, never an imputed cost and never submitted to the fixed solver.
    relaxed = {key: (0 if key in incomplete or not choices[key] else
                     min(option['duration_ticks'] for option in choices[key])) for key in ids}
    global_lower = critical_bound(relaxed)
    constraints = dict(resource_capacities=deepcopy(dict(resource_capacities)),
                       memory_capacities=deepcopy(memory_capacities),
                       materialization_sizes=deepcopy(materialization_sizes),
                       memory_locations=deepcopy(memory_locations),
                       model_instantiation=deepcopy(model_instantiation))
    try:
        domain_hash = semantic_sha256(task_options)
        model_hash = semantic_sha256(dict(schedule=graph.document,
                                          option_domain=task_options, fixed_constraints=constraints))
    except (TypeError, ValueError):
        domain_hash, model_hash = None, None
        model_issues.append(dict(field='option_domain', reason='finite semantic JSON required'))
    result = dict(schema='scar.implementation-choice-solver', schema_version=1,
                  region_id=schedule.region_id, schedule_sha256=graph.sha256,
                  status='UNRESOLVED_OPTION_DOMAIN', optimality='NOT_ESTABLISHED',
                  objective='MINIMIZE_MAKESPAN', physical_optimality_proven=False,
                  time_unit='CALLER_DECLARED_COMMON_TICKS',
                  model_assumptions=['ALL_OPTION_COSTS_USE_THE_SAME_DECLARED_TICK_BASIS',
                                     'NONPREEMPTIVE_TASKS_AND_FIXED_MATERIALIZATION_LOCATIONS',
                                     'ONLY_SOURCED_DECLARED_LEGALITY_CONDITIONS_ARE_ASSUMED',
                                     'RESOURCE_CAPACITIES_AND_ALLOCATION_LIFETIMES_FOLLOW_THE_FIXED_SOLVER_MODEL'],
                  option_domain_sha256=domain_hash, global_model_sha256=model_hash,
                  option_domain=deepcopy(dict(task_options)), fixed_resource_constraints=constraints,
                  target_program_executed=False, deployment_authorized=False,
                  scope='FINITE_DECLARED_IMPLEMENTATION_RESOURCE_CHOICES_FIXED_MATERIALIZATIONS',
                  materialization_placement_optimized=False, implicit_transfers_inserted=False,
                  execution_plan_rewritten=False, backend_lowering_performed=False,
                  chosen_option_lowering_requires='TYPED_BACKEND_LEGALITY_AND_MATERIALIZATION_LOCATION_ADMISSION',
                  execution_unresolved=deepcopy(schedule.unresolved),
                  frontier=frontiers, unresolved=model_issues, chosen_option_ids=None,
                  chosen_options=None, fixed_solver_result=None, solution=None,
                  global_lower_bound_ticks=global_lower, gap_ticks=None, relative_gap=None,
                  lower_bound_basis='TASK_DAG_MINIMUM_DURATIONS_WITH_EXPLICIT_NONNEGATIVITY_RELAXATION_FOR_FRONTIERS_NOT_COST_IMPUTATION',
                  relaxed_duration_bounds=relaxed, states_explored=0, choice_branches=0,
                  search_budget=search_budget, search_budget_exhausted=False,
                  external_assumptions=deepcopy(schedule.proof.get('external_assumptions', [])),
                  optimality_scope='GIVEN_FINITE_OPTION_COST_LEGALITY_RESOURCE_AND_LIFETIME_MODEL_ONLY',
                  legality_authority='RETAINED_DECLARATIONS_UNDER_EXPLICIT_CONDITIONS_NOT_GLOBAL_EFFECT_PROOFS')
    if model_issues or any(not choices[key] for key in ids):
        if model_issues:
            result['status'] = 'UNRESOLVED_MODEL'
        return result
    best, used, branches, exhausted = None, 0, 0, False
    for combo in product(*(choices[key] for key in ids)):
        if used >= search_budget:
            exhausted = True
            break
        used += 1
        branches += 1
        selected = dict(zip(ids, combo))
        durations = {key: option['duration_ticks'] for key, option in selected.items()}
        if best is not None and critical_bound(durations) >= best['solution']['makespan_ticks']:
            continue
        if used >= search_budget:
            exhausted = True
            break
        fixed = _solve_fixed_schedule(
            graph, task_costs=durations,
            task_placements={key: option['resource_demands'] for key, option in selected.items()},
            resource_capacities=resource_capacities, memory_capacities=memory_capacities,
            materialization_sizes=materialization_sizes, memory_locations=memory_locations,
            model_instantiation=model_instantiation, search_budget=search_budget - used)
        used += fixed['states_explored']
        if fixed['status'] == 'UNRESOLVED':
            result['unresolved'] = fixed['unresolved']
            break
        if fixed['solution'] is not None and (best is None or fixed['solution']['makespan_ticks'] < best['solution']['makespan_ticks']):
            best = fixed
            result.update(chosen_option_ids={key: option['id'] for key, option in selected.items()},
                          chosen_options=deepcopy(selected))
        if fixed['status'] == 'SEARCH_BUDGET_EXHAUSTED':
            exhausted = True
            break
    result.update(states_explored=used, choice_branches=branches, search_budget_exhausted=exhausted)
    if best is not None:
        result.update(fixed_solver_result=best, solution=best['solution'])
    if result['unresolved']:
        result['status'] = 'UNRESOLVED_MODEL'
    elif frontiers:
        result['status'] = 'UNRESOLVED_OPTION_DOMAIN'
    elif exhausted:
        result['status'] = 'SEARCH_BUDGET_EXHAUSTED'
    elif best is None:
        result.update(status='INFEASIBLE_WITHIN_MODEL', optimality='INFEASIBILITY_ESTABLISHED')
    else:
        global_lower = best['solution']['makespan_ticks']
        result.update(status='OPTIMAL_WITHIN_FINITE_MODEL', optimality='FINITE_MODEL_OPTIMUM_ESTABLISHED')
    result['global_lower_bound_ticks'] = global_lower
    if best is not None:
        gap = best['solution']['makespan_ticks'] - global_lower
        result.update(gap_ticks=gap, relative_gap=gap / best['solution']['makespan_ticks'] if best['solution']['makespan_ticks'] else 0.0)
    return result


__all__ = ['task_implementation_contract', 'solve_implementation_choices']

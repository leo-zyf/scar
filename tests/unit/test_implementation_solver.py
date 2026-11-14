"""Finite resource choices are optimized jointly with model task ordering."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json

from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import encode
from scar.optimization.implementation_solver import solve_implementation_choices, task_implementation_contract
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.schedule import ResourceSchedule, Task, TaskKind


def options_for(schedule, tmp_path):
    definitions = {
        'flexible': [('fast_local', 2, {'cpu': 1}), ('slow_other', 3, {'aux': 1})],
        'critical': [('fixed', 5, {'cpu': 1})],
        'tail': [('fixed', 1, {'cpu': 1})]}
    records, options = {}, {}
    for task in schedule.tasks:
        contract = semantic_sha256(task_implementation_contract(schedule, task.id))
        records[task.id], options[task.id] = {}, []
        for identifier, duration, demands in definitions[task.id]:
            conditions = ['EXACT_ABSTRACT_TASK_SEMANTICS_UNDER_DECLARED_TEST_MODEL', 'UNCHANGED_MATERIALIZATION_LOCATIONS']
            records[task.id][identifier] = {
                'duration_ticks': duration,
                'legality': {'contract_sha256': contract, 'resource_demands': demands, 'conditions': conditions}}
            options[task.id].append(dict(id=identifier, duration_ticks=duration, resource_demands=demands,
                                         contract_sha256=contract, legality_conditions=conditions))
    path = tmp_path / 'declared-options.json'
    path.write_text(json.dumps({'schema': 'scar.declared-options-test', 'options': records,
                                'basis': 'UNIT_TEST_DECLARATIONS_NOT_PHYSICAL_COST_MEASUREMENTS'}))
    raw = path.read_bytes()
    artifact = ProviderArtifact(str(path), sha256(raw).hexdigest(), len(raw), 'scar.declared-options-test')
    for task, domain in options.items():
        for option in domain:
            base = f"#/options/{task}/{option['id']}"
            option['cost_evidence'] = [encode(ProviderFactReference(artifact, base + '/duration_ticks'))]
            option['legality_evidence'] = [encode(ProviderFactReference(artifact, base + '/legality'))]
    return options


def schedule():
    return ResourceSchedule('finite', (), (), (
        Task('flexible', TaskKind.COMPUTE), Task('critical', TaskKind.COMPUTE),
        Task('tail', TaskKind.COMPUTE, depends_on=('critical',))))


def test_locally_slower_resource_option_avoids_blocking_the_critical_chain(tmp_path):
    graph = schedule()
    options = options_for(graph, tmp_path)
    result = solve_implementation_choices(graph, task_options=options, resource_capacities={'cpu': 1, 'aux': 1})
    assert result['status'] == 'OPTIMAL_WITHIN_FINITE_MODEL'
    assert result['chosen_option_ids']['flexible'] == 'slow_other'
    assert result['solution']['makespan_ticks'] == 6
    assert result['global_lower_bound_ticks'] == 6 and result['gap_ticks'] == 0
    assert result['fixed_solver_result']['model']['resource_demands']['flexible'] == {'aux': 1}
    assert not result['target_program_executed'] and not result['execution_plan_rewritten']
    assert not result['materialization_placement_optimized'] and not result['implicit_transfers_inserted']
    limited = solve_implementation_choices(graph, task_options=options,
                                           resource_capacities={'cpu': 1, 'aux': 1}, search_budget=1)
    assert limited['status'] == 'SEARCH_BUDGET_EXHAUSTED' and limited['optimality'] == 'NOT_ESTABLISHED'


def test_unknown_option_cost_stays_frontier_and_bound_cpu_attributes_cannot_change(tmp_path):
    graph = schedule()
    options = options_for(graph, tmp_path)
    options['flexible'][0]['duration_ticks'] = None
    result = solve_implementation_choices(graph, task_options=options, resource_capacities={'cpu': 1, 'aux': 1})
    assert result['status'] == 'UNRESOLVED_OPTION_DOMAIN' and result['optimality'] == 'NOT_ESTABLISHED'
    assert result['frontier'] and result['solution']['makespan_ticks'] == 6
    assert result['relaxed_duration_bounds']['flexible'] == 0
    assert 'NOT_COST_IMPUTATION' in result['lower_bound_basis']
    assert result['fixed_solver_result']['model']['task_durations_ticks']['flexible'] == 3
    bound = replace(graph, tasks=(replace(graph.tasks[0], attrs={'cpu_threads': 1, 'executor_id': 'cpu'}),)
                    + graph.tasks[1:])
    bound_options = options_for(bound, tmp_path)
    changed = solve_implementation_choices(bound, task_options=bound_options,
                                           resource_capacities={'cpu': 1, 'aux': 1})
    assert changed['status'] == 'UNRESOLVED_OPTION_DOMAIN'
    assert any('bound CPU task attributes are fixed' in entry['reason'] for entry in changed['frontier'])
    assert changed['chosen_option_ids']['flexible'] == 'fast_local'

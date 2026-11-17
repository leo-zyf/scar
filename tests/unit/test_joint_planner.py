"""Small declared models exercise joint choices, placement and hard budgets.

Numbers are algorithm examples, never physical calibration or source admission.
The pure solver consumes already admitted source/legality references upstream.
"""
from copy import deepcopy

import pytest

pytest.importorskip('ortools', reason='joint planning uses the optional scar[planning] dependency')

from scar.ir.implementation_plan import (CostEstimate, ImplementationOption,
    PlanningDataEdge, PlanningOperation, PlanningProblem, PlanningResource, TransferOption)
from scar.ir.program.providers import ProviderArtifact
from scar.optimization.joint_planner import solve_implementations, verify_execution_plan


_REFS = ({'kind': 'DECLARED_UNIT_TEST_MODEL_NOT_PHYSICAL_MEASUREMENT'},)
_TARGET = ProviderArtifact('/declared/unit-test-target.json', '0'*64, 0, 'scar.target-hardware-graph')


def _cost(ns, *, basis=None, missing=()):
    return CostEstimate(ns, 0 if ns is None else ns, ns,
        ('UNKNOWN' if ns is None else 'DECLARED_MODEL') if basis is None else basis,
        'DECLARED_UNIT_TEST_NS_NOT_PHYSICAL_TIME', _REFS, tuple(missing))


def _option(name, ns, cores=1, workspace=0, *, inputs=None, outputs=None, cost=None, parameters=None):
    return ImplementationOption(name, 'DECLARED_TEST_IMPLEMENTATION', _REFS,
        {'cpu': cores}, {'host': workspace} if workspace else {}, inputs or {}, outputs or {},
        _cost(ns) if cost is None else cost, parameters or {})


def _operation(name, *options, predecessors=(), baseline=None):
    return PlanningOperation(name, _REFS, tuple(predecessors), tuple(options),
        options[0].id if baseline is None else baseline)


def _problem(operations, *, cores=4, memory=1024, edges=(), resources=()):
    return PlanningProblem(_TARGET, _REFS,
        (PlanningResource('cpu', 'EXECUTION', cores, _REFS),
         PlanningResource('host', 'MEMORY', memory,
            ({'capacity_scope': 'INCREMENTAL_PRIVATE_PAYLOAD_EXCLUDING_BORROWED_EXTERNALS'},)),
         *resources), tuple(operations), tuple(edges),
        assumptions=('DECLARED_FINITE_ALGORITHM_EXAMPLE',
            'MEMORY_CAPACITY_IS_INCREMENTAL_PRIVATE_PAYLOAD_NOT_TOTAL_RSS'))


def _solve(problem, **kwargs):
    result = solve_implementations(problem, max_seconds=5, workers=1, **kwargs)
    assert result['schema'] == 'scar.implementation-execution-plan'
    rows = {row['operation']: row for row in result['selected']}
    assert len(rows) == len(result['selected']) == len(problem.operations)
    assert set(rows) == {operation.id for operation in problem.operations}
    assert all(0 <= row['start_ns'] <= row['end_ns'] for row in rows.values())
    assert isinstance(verify_execution_plan(problem, result), dict)
    return result, rows


def _parallel_problem():
    return _problem(tuple(_operation(name,
        _option('four_cores', 60, cores=4), _option('two_cores', 70, cores=2)) for name in ('A', 'B')))


def _memory_problem():
    return _problem(tuple(_operation(name,
        _option('fast', 40, workspace=90), _option('lean', 70, workspace=20),
        _option('over_budget', 1, workspace=101)) for name in ('A', 'B')), cores=2, memory=100)


def _transfer_problem():
    producer = _operation('P', _option('host', 5, outputs={'v': 'host'}))
    consumer = _operation('C', ImplementationOption('device', 'DECLARED_DEVICE_REGION', _REFS,
        {'device': 1}, {}, {'v': 'device_memory'}, {}, _cost(7), {}))
    transfer = TransferOption('h2d', 'host', 'device_memory', {'link': 1}, _cost(3),
        'DECLARED_EXPLICIT_TRANSFER', _REFS)
    edge = PlanningDataEdge('P_to_C', 'v', 'P', 'C', 64, None, (transfer,), _REFS)
    return _problem((producer, consumer), memory=128, edges=(edge,), resources=(
        PlanningResource('device', 'EXECUTION', 1, _REFS),
        PlanningResource('link', 'CHANNEL', 1, _REFS),
        PlanningResource('device_memory', 'MEMORY', 128, _REFS)))


def test_joint_core_choice_prefers_two_parallel_regions_over_locally_fastest_modes():
    problem = _parallel_problem()
    result, rows = _solve(problem)
    assert {row['option'] for row in rows.values()} == {'two_cores'}
    assert result['makespan_ns'] == result['model_lower_bound_ns'] == 70
    assert result['relative_gap'] == 0 and result['domain_complete'] is True
    assert all(row['start_ns'] == 0 for row in rows.values())
    baseline, _ = _solve(problem, baseline_only=True)
    assert baseline['makespan_ns'] == 120
    assert baseline['optimization_domain'] == 'BASELINE_IMPLEMENTATIONS_ONLY'


def test_startup_cost_can_make_serial_baseline_better_than_parallel_implementation():
    parallel = _option('parallel_with_startup', 100, cores=4,
        parameters={'compute_ns': 20, 'startup_ns': 80})
    problem = _problem((_operation('region', _option('serial', 90), parallel),))
    result, rows = _solve(problem)
    assert rows['region']['option'] == 'serial' and result['makespan_ns'] == 90
    assert result['domain_complete'] is True and result['relative_gap'] == 0


def test_private_workspace_budget_changes_the_joint_implementation_choice():
    result, rows = _solve(_memory_problem())
    assert result['makespan_ns'] == 70
    assert {row['option'] for row in rows.values()} == {'lean'}
    assert result['domain_complete'] is True


def test_different_data_locations_require_an_explicit_causal_transfer():
    result, rows = _solve(_transfer_problem())
    assert result['makespan_ns'] == 15 and len(result['transfers']) == 1
    transfer = result['transfers'][0]
    assert transfer['edge'] == 'P_to_C' and transfer['option'] == 'h2d'
    assert (transfer['source_memory'], transfer['destination_memory']) == ('host', 'device_memory')
    assert transfer['start_ns'] >= rows['P']['end_ns']
    assert transfer['end_ns'] - transfer['start_ns'] == 3
    assert rows['C']['start_ns'] >= transfer['end_ns']


def test_produced_value_fork_shares_one_storage_and_external_roots_are_borrowed():
    producer = _operation('P', _option('produce', 2, inputs={'x': 'host'}, outputs={'v': 'host'}))
    consumers = tuple(_operation(name, _option('consume', 5, inputs={'v': 'host'})) for name in ('A', 'B'))
    edges = (PlanningDataEdge('external_to_P', 'x', None, 'P', 1024, 'host', (), _REFS),
        *(PlanningDataEdge('P_to_'+name, 'v', 'P', name, 64, None, (), _REFS) for name in ('A', 'B')))
    problem = _problem((producer, *consumers), cores=2, memory=64, edges=edges)
    result, rows = _solve(problem)
    assert result['makespan_ns'] == 7 and not result['transfers']
    assert rows['A']['start_ns'] == rows['B']['start_ns'] == rows['P']['end_ns']
    leases = [row for row in result['memory_leases'] if row['id'] == 'output:v']
    assert len(leases) == 1 and leases[0]['memory'] == 'host' and leases[0]['bytes'] == 64
    assert leases[0]['start_ns'] == rows['P']['start_ns']
    assert leases[0]['end_ns'] == max(rows['A']['end_ns'], rows['B']['end_ns'])
    assert result['external_input_bytes_not_charged'] == 1024
    assert len(result['memory_leases']) == 1


def test_unknown_or_incomplete_costs_cannot_claim_complete_domain_optimality():
    unresolved = (
        _option('unknown', None, cost=_cost(None, missing=('UNCALIBRATED',))),
        _option('no_estimate', None, cost=_cost(None, basis='ANALYTICAL_MODEL')),
        _option('missing_component', 1, cost=_cost(1, missing=('TEAM_STARTUP_NOT_ACCOUNTED',))))
    problem = _problem((_operation('region', _option('baseline', 40), *unresolved),))
    result, rows = _solve(problem)
    assert rows['region']['option'] == 'baseline' and result['makespan_ns'] == 40
    assert result['domain_complete'] is False
    assert result['model_lower_bound_ns'] is None and result['relative_gap'] is None
    assert result['status'] not in ('OPTIMAL', 'OPTIMAL_WITHIN_COMPLETE_DOMAIN')
    mutant = deepcopy(result); mutant['domain_complete'] = True
    with pytest.raises(ValueError): verify_execution_plan(problem, mutant)


def test_independent_verifier_rejects_numeric_resource_memory_and_materialization_tampering():
    problem = _parallel_problem(); original, _ = _solve(problem)
    mutant = deepcopy(original); mutant['selected'][0]['end_ns'] -= 1
    with pytest.raises(ValueError): verify_execution_plan(problem, mutant)
    mutant = deepcopy(original)
    for row in mutant['selected']:
        row.update(option='four_cores', start_ns=0, end_ns=60)
    mutant['makespan_ns'] = 60
    with pytest.raises(ValueError): verify_execution_plan(problem, mutant)
    problem = _memory_problem(); original, _ = _solve(problem); mutant = deepcopy(original)
    for row in mutant['selected']:
        row.update(option='fast', start_ns=0, end_ns=40)
    mutant['makespan_ns'] = 40
    for lease in mutant['memory_leases']:
        lease.update(bytes=90, start_ns=0, end_ns=40)
    with pytest.raises(ValueError): verify_execution_plan(problem, mutant)
    problem = _transfer_problem(); original, rows = _solve(problem); mutant = deepcopy(original)
    mutant['transfers'] = []
    for row in mutant['selected']:
        if row['operation'] == 'C': row.update(start_ns=5, end_ns=12)
    mutant['makespan_ns'] = 12
    with pytest.raises(ValueError): verify_execution_plan(problem, mutant)

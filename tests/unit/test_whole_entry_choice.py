"""Exact entry scope, interval dominance and resource-timeline replay."""

from copy import deepcopy
import json

import pytest

from scar.ir.execution_recipe import artifact_reference
from scar.ir.implementation_plan import (
    SCHEMA as PROBLEM, CostEstimate, ImplementationOption, PlanningOperation,
    PlanningProblem, PlanningResource, problem_document,
)
from scar.ir.program.providers import ProviderArtifact
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.whole_entry_choice import (
    DOMAIN, LIFECYCLE_PHASES, solve_whole_entry_choice, verify_whole_entry_choice,
)


_REFS = ({'kind': 'DECLARED_TEST_MODEL'},)


def _artifact(path, document):
    path.write_text(json.dumps(document), encoding='utf-8')
    return artifact_reference(path, document['schema'])


def _domain():
    return dict(schema=DOMAIN, schema_version=1, scope=dict(boundary='PROCESS_LAUNCH_THROUGH_EXIT',
        argv=['work=1'], target='test-domain'), baseline_id='original', options=[], coverage_frontier=[])


def _phase_coverage():
    return {phase: 'ACCOUNTED' if phase in ('startup', 'execution', 'teardown') else 'INAPPLICABLE'
            for phase in LIFECYCLE_PHASES}


def _interval(low, high):
    return dict(kind='COMPLETE_INTERVAL', basis='ANALYTICAL_MODEL', interval_ns=dict(lower=low, upper=high),
                model_gap_ns=0, lifecycle_coverage=_phase_coverage())


def _option(domain, tmp_path, name, cost=None, status='LEGAL'):
    fact = dict(option_id=name, scope_sha256=semantic_sha256(domain['scope']))
    ref = _artifact(tmp_path / (name + '.json'), dict(schema='test.whole-entry-evidence',
        legality=dict(fact, legal_status=status), cost=None if cost is None else dict(fact, **cost)))
    domain['options'].append(dict(id=name, legal_status=status,
        legality_evidence=[dict(artifact=ref, locator='#/legality')],
        cost=None if cost is None else dict(artifact=ref, locator='#/cost')))


def test_open_scope_and_unknown_cost_never_claim_model_optimality(tmp_path):
    domain = _domain()
    _option(domain, tmp_path, 'original')
    report = solve_whole_entry_choice(domain, expected_option_ids=['original'])
    assert report['decision'] == 'NO_OP' and report['selected_option_id'] is None
    assert report['cost_frontier'][0]['option_id'] == 'original'
    with pytest.raises(ValueError, match='independent enumeration'):
        solve_whole_entry_choice(domain, expected_option_ids=['original', 'omitted'])

    domain['options'].clear()
    observed = _interval(3, 3)
    observed['basis'] = 'EMPIRICAL_ENVELOPE'
    _option(domain, tmp_path, 'original', observed)
    report = solve_whole_entry_choice(domain, expected_option_ids=['original'])
    assert not report['model_optimality_proven']
    assert report['alternatives'][0]['cost_basis'] == 'EMPIRICAL_ENVELOPE_NOT_MODEL_BOUND'

    domain['options'].clear()
    _option(domain, tmp_path, 'original', _interval(110, 120))
    domain['coverage_frontier'].append({'reason': 'UNEXPLORED_REGION_COMPOSITIONS'})
    report = solve_whole_entry_choice(domain, expected_option_ids=['original'])
    assert not report['model_optimality_proven'] and report['known_incumbent_id'] == 'original'
    domain['scope']['argv'] = ['work=2']
    with pytest.raises(ValueError, match='exact-scope'):
        solve_whole_entry_choice(domain, expected_option_ids=['original'])


def test_pending_competitor_is_only_excluded_by_strict_mandatory_work_bound(tmp_path):
    domain = _domain()
    _option(domain, tmp_path, 'original', _interval(110, 120))
    _option(domain, tmp_path, 'native', _interval(90, 100))
    names = ['original', 'native', 'open']
    bound = dict(kind='LOWER_BOUND_ONLY', basis='ANALYTICAL_MODEL', required_work='startup', lower_ns=100)
    _option(domain, tmp_path, 'open', bound, 'UNRESOLVED')
    assert not solve_whole_entry_choice(domain, expected_option_ids=names)['model_optimality_proven']

    domain['options'].pop()
    bound['lower_ns'] = 101
    _option(domain, tmp_path, 'open', bound, 'UNRESOLVED')
    report = solve_whole_entry_choice(domain, expected_option_ids=names)
    assert report['selected_option_id'] == 'native' and report['selection_gap_ns'] == 0
    assert report['decision'] == 'CANDIDATE_FOR_VALIDATION'
    assert not report['deployment_authorized'] and not report['physical_optimality_proven']
    assert {row['option_id'] for row in report['dominated']} == {'original', 'open'}
    assert verify_whole_entry_choice(domain, report, expected_option_ids=names) == report
    forged = deepcopy(report)
    forged['decision'] = 'DEPLOY'
    with pytest.raises(ValueError, match='independent replay'):
        verify_whole_entry_choice(domain, forged, expected_option_ids=names)


def test_process_phase_contract_and_deterministic_exact_tie(tmp_path):
    domain = _domain()
    _option(domain, tmp_path, 'z-native', _interval(10, 10))
    _option(domain, tmp_path, 'a-native', _interval(10, 10))
    _option(domain, tmp_path, 'original', _interval(10, 10))
    names = ['original', 'z-native', 'a-native']
    report = solve_whole_entry_choice(domain, expected_option_ids=names)
    assert report['selected_option_id'] == 'original'
    assert report['proved_exact_tie_ids'] == ['a-native', 'z-native']
    domain['options'].pop()
    regional = _interval(50, 50)
    regional['lifecycle_coverage']['startup'] = 'INAPPLICABLE'
    _option(domain, tmp_path, 'original', regional)
    with pytest.raises(ValueError, match='process startup and exit'):
        solve_whole_entry_choice(domain, expected_option_ids=names)


def test_exact_resource_optimum_includes_launch_and_exit_and_replays(tmp_path):
    pytest.importorskip('ortools')
    domain = _domain()
    target = _artifact(tmp_path / 'target.json', {'schema': 'scar.target-hardware-graph'})
    domain['scope']['target'] = target

    def mode(name, ns, cores):
        return ImplementationOption(name, 'DECLARED', _REFS, {'cpu': cores}, {}, {}, {},
            CostEstimate(ns, ns, ns, 'DECLARED_MODEL', 'EXAMPLE_NS', _REFS), {})

    def stage(name, predecessors, *modes):
        return PlanningOperation(name, _REFS, predecessors, modes, modes[0].id)

    start = stage('launch', (), mode('start', 6, 1))
    workers = tuple(stage(name, ('launch',), mode('full', 60, 4), mode('half', 70, 2))
                    for name in ('x', 'y'))
    finish = stage('exit', ('x', 'y'), mode('finish', 4, 1))
    problem = PlanningProblem(ProviderArtifact(**target), _REFS,
        (PlanningResource('cpu', 'EXECUTION', 4, _REFS),), (start, *workers, finish), ())
    source = _artifact(tmp_path / 'problem.json', problem_document(problem))
    cost = dict(kind='OPTIMIZED_POINT_SCHEDULE', problem=source, lifecycle_coverage=_phase_coverage(),
        phase_bindings=dict(startup=['launch'], execution=['x', 'y'], teardown=['exit']),
        boundary_anchors={'entry': 'launch', 'exit': 'exit'}, seconds=5)
    _option(domain, tmp_path, 'original', _interval(125, 135))
    _option(domain, tmp_path, 'native', cost)
    names = ['original', 'native']
    report = solve_whole_entry_choice(domain, expected_option_ids=names)
    assert report['selected_option_id'] == 'native'
    schedule = report['alternatives'][0]['schedule_certificate']
    assert schedule['proof'] == 'CP_SAT_CLOSED_DOMAIN_ZERO_GAP'
    assert schedule['plan']['makespan_ns'] == 80
    assert {row['option'] for row in schedule['plan']['selected']} == {'start', 'half', 'finish'}
    assert verify_whole_entry_choice(domain, report, expected_option_ids=names) == report

    detached = stage('exit', (), mode('finish', 4, 1))
    invalid = PlanningProblem(problem.target, _REFS, problem.resources, (start, *workers, detached), ())
    domain['options'].pop()
    cost['problem'] = _artifact(tmp_path / 'detached.json', problem_document(invalid))
    _option(domain, tmp_path, 'native', cost)
    with pytest.raises(ValueError, match='ordered inside unconditional'):
        solve_whole_entry_choice(domain, expected_option_ids=names)

    serial = stage('y', ('x',), mode('full', 60, 4), mode('half', 70, 2))
    fixed = PlanningProblem(problem.target, _REFS, problem.resources, (start, workers[0], serial, finish), ())
    domain['options'].pop()
    cost['problem'] = _artifact(tmp_path / 'serial.json', problem_document(fixed))
    _option(domain, tmp_path, 'native', cost)
    report = solve_whole_entry_choice(domain, expected_option_ids=names)
    schedule = report['alternatives'][0]['schedule_certificate']
    assert schedule['proof'] == 'CONSTRUCTIVE_UPPER_MEETS_ANALYTIC_LOWER_BOUND'
    assert schedule['plan']['makespan_ns'] == 130
    assert not report['model_optimality_proven']

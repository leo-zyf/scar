"""Exact entry scope, interval dominance and resource-timeline replay."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from scar.ir.execution_recipe import artifact_reference
from scar.ir.implementation_plan import (
    SCHEMA as PROBLEM, CostEstimate, ImplementationOption, PlanningOperation,
    PlanningProblem, PlanningResource, PlanningTimingConstraint, problem_document,
)
from scar.ir.program.providers import ProviderArtifact
from scar.optimization.resource_solver import semantic_sha256
from scar.compiler import whole_entry_pipeline
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


@pytest.mark.parametrize('executed', (False, None, True))
def test_device_trial_allows_cpu_only_when_neither_entry_ran(monkeypatch, tmp_path, executed):
    generated = {device: dict(id=device, legal_status='UNRESOLVED', implementation_family=family)
                 for device, family in (('cpu', 'NVIDIA_CPU_MATRIX_FILE_ENTRY'),
                                        ('device', 'NVIDIA_DEVICE_MATRIX_FILE_ENTRY'))}
    attempted = []

    def emit(option, scope, folder, *, plan_only, verify_entry):
        name = option['id']
        attempted.append(name)
        assert not plan_only and verify_entry
        return dict(selected_id=name, target_executed=executed if name == 'device' else True,
                    validation=None, reason='test')

    monkeypatch.setattr(whole_entry_pipeline, '_emit_generated', emit)
    cpu, device, audit = whole_entry_pipeline._review_source_trials(
        {}, generated, {'options': list(generated.values())}, {}, tmp_path)
    assert device['target_executed'] is executed
    assert (cpu is not None) == (executed is False)
    assert attempted == (['device', 'cpu'] if executed is False else ['device'])
    assert [row['option_id'] for row in audit] == attempted


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


def test_mandatory_subgraph_prunes_with_capacity_even_with_unknown_launch(tmp_path):
    domain = _domain()
    target = _artifact(tmp_path/'target.json', {'schema': 'scar.target-hardware-graph'})
    domain['scope']['target'] = target
    scope = dict(option_id='native', scope_sha256=semantic_sha256(domain['scope']))
    proof = _artifact(tmp_path/'mandatory-proof.json', dict(schema='test.mandatory-model',
        cpu=dict(scope, kind='MANDATORY_RESOURCE_CAPACITY', resource_id='cpu', upper_capacity=1),
        **{name: dict(scope, kind='MANDATORY_ACTIVITY_LOWER_BOUND', activity_id=name, lower_ns=52)
           for name in ('x', 'y')}))

    def refs(name):
        return (dict(artifact=proof, locator=f'#/{name}'),)

    def stage(name, parents, lower):
        cost = CostEstimate(None, lower, None, 'UNKNOWN' if lower is None else 'ANALYTICAL_MODEL',
            name, () if lower is None else refs(name))
        mode = ImplementationOption('fixed', 'DECLARED', _REFS, {'cpu': 1}, {}, {}, {}, cost, {})
        return PlanningOperation(name, _REFS, parents, (mode,), mode.id)

    problem = PlanningProblem(ProviderArtifact(**target), _REFS, (PlanningResource('cpu', 'EXECUTION', 1, refs('cpu')),),
        (stage('launch', (), None), stage('x', ('launch',), 52), stage('y', ('launch',), 52)), ())
    _option(domain, tmp_path, 'original', _interval(90, 99))
    cost = dict(kind='MANDATORY_SUBGRAPH_LOWER_BOUND', basis='ANALYTICAL_MODEL',
        problem=_artifact(tmp_path/'mandatory.json', problem_document(problem)))
    _option(domain, tmp_path, 'native', cost, 'UNRESOLVED')
    names = ['original', 'native']
    result = solve_whole_entry_choice(domain, expected_option_ids=names)
    native, original = result['alternatives']
    assert result['selected_option_id'] == original['option_id'] == 'original'
    assert native['interval_ns'] == {'lower': 104, 'upper': None}
    assert native['schedule_certificate']['critical_path_ns'] == 52
    assert native['schedule_certificate']['activities_without_positive_bound'] == ['launch']
    assert verify_whole_entry_choice(domain, result, expected_option_ids=names) == result

    domain['options'].pop()
    widened = replace(problem, resources=(replace(problem.resources[0], capacity=2),))
    cost['problem'] = _artifact(tmp_path/'no-exact-capacity.json', problem_document(widened))
    _option(domain, tmp_path, 'native', cost, 'UNRESOLVED')
    result = solve_whole_entry_choice(domain, expected_option_ids=names)
    assert result['alternatives'][0]['interval_ns'] == {'lower': 52, 'upper': None}
    assert result['selected_option_id'] is None


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


def test_fixed_entry_interval_replays_overlapping_workers_and_lifecycle(tmp_path):
    domain = _domain()
    target = _artifact(tmp_path / 'target.json', {'schema': 'scar.target-hardware-graph'})
    domain['scope']['target'] = target
    evidence = _artifact(tmp_path / 'components.json', {'schema': 'test.component-cost-model'})
    refs = ({'artifact': evidence, 'locator': '#'},)

    def operation(name, parents, low, high, demands):
        duration = CostEstimate(None, low, high, 'ANALYTICAL_MODEL', name, refs)
        mode = ImplementationOption('fixed', 'LOWERED', _REFS, demands, {}, {}, {}, duration, {})
        return PlanningOperation(name, _REFS, parents, (mode,), 'fixed')

    operations = (operation('launch', (), 10, 15, {'cpu:0': 1}),
                  operation('x', ('launch',), 30, 40, {'cpu:0': 1}),
                  operation('y', ('launch',), 32, 45, {'cpu:1': 1}),
                  operation('exit', ('x', 'y'), 5, 8, {'cpu:0': 1}))
    resources = tuple(PlanningResource(f'cpu:{i}', 'EXECUTION', 1, _REFS) for i in range(2))
    problem = PlanningProblem(ProviderArtifact(**target), _REFS, resources, operations, ())
    grouped = _artifact(tmp_path/'grouped.json', dict(schema='test.grouped-phase', option_id='native',
        scope_sha256=semantic_sha256(domain['scope']), included_phase='allocation', charged_phase='execution'))
    coverage = dict(_phase_coverage(), allocation='ACCOUNTED')
    cost = dict(kind='BOUNDED_MONOTONE_SCHEDULE', lifecycle_coverage=coverage,
        problem=_artifact(tmp_path / 'problem.json', problem_document(problem)),
        phase_bindings=dict(startup=['launch'], execution=['x', 'y'], teardown=['exit']),
        included_phases=dict(allocation=dict(phase='execution', evidence=dict(artifact=grouped, locator='#'))),
        boundary_anchors={'entry': 'launch', 'exit': 'exit'})
    _option(domain, tmp_path, 'original', _interval(100, 130))
    _option(domain, tmp_path, 'native', cost)
    names = ['original', 'native']
    report = solve_whole_entry_choice(domain, expected_option_ids=names)
    assert report['selected_option_id'] == 'native'
    native = report['alternatives'][0]
    assert native['interval_ns'] == {'lower': 47, 'upper': 68}
    assert native['schedule_certificate']['plan']['makespan_ns'] == 68

    non_monotone = replace(problem, timing_constraints=(PlanningTimingConstraint(
        'deadline', 'x', 'END', 'exit', 'START', maximum_ns=4, source_refs=_REFS),))
    cost['problem'] = _artifact(tmp_path / 'bounded-lag.json', problem_document(non_monotone))
    domain['options'].pop()
    _option(domain, tmp_path, 'native', cost)
    with pytest.raises(ValueError, match='unconditional end-to-start'):
        solve_whole_entry_choice(domain, expected_option_ids=names)

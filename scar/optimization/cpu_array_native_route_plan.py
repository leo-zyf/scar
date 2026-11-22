"""Compare two admitted local C++ routes for one exact CPU array entry."""

from fractions import Fraction
from pathlib import Path

from scar.analysis.replay_transaction import replay_transaction
from scar.ir.execution_recipe import read_artifact
from scar.optimization.paired_entry_selection import SCHEMA as PAIRED_SELECTION

SCHEMA = 'scar.cpu-array-native-route-plan'
_PROOF = 'scar.cpu-array-native-route-proof'
_OBSERVATION = 'scar.cpu-array-native-route-observation'
_PROVEN = 'PROVEN_SCOPED_CPU_ARRAY_NATIVE_ROUTE'
_INCUMBENT = 'retained_process'
_CHALLENGER = 'grouped_openmp'
_MINIMUM = Fraction(1, 20)


def _require(ok, detail):
    if not ok:
        raise ValueError('CPU array native route plan: ' + detail)


def _routes(source):
    retained = read_artifact(source['retention'])
    workload = source['workload']
    local_ref, grouped_ref = source['retained_runtime'], source['baseline_grouped']
    local, grouped = read_artifact(local_ref), read_artifact(grouped_ref)
    executable = local['files'][local['binary']]
    local_env = dict(workload['original_environment'], **local['launch_environment'])
    grouped_env = dict(workload['original_environment'], **grouped['launch_environment'])
    _require(retained['retained_release'] == local_ref and retained['reused_binary'] == executable
             and grouped['runtime'] == source['grouped_runtime']
             and local['scope'] == grouped['scope'] == source['scope']
             and local['recipe'] == grouped['recipe'] == workload['recipe']
             and local['cpu_ids'] == grouped['cpu_ids'] == source['resources']['launch_cpu_ids']
             and local_env == workload['local_process_environment']
             and grouped_env == workload['openmp_grouped_environment']
             and local['python_runtime_required'] is False and grouped['python_runtime_required'] is False
             and local['cxx_standard'] == grouped['cxx_standard'] == 20
             and local['deployment_authorized'] is grouped['deployment_authorized'] is False,
             'routes differ from the paired scope or retained C++20 packages')
    return {
        _INCUMBENT: dict(package=local_ref, executable=executable, resource='#/resources/local_processes',
                         launch_environment=local['launch_environment']),
        _CHALLENGER: dict(package=grouped_ref, executable=grouped['executable'], resource='#/resources/openmp_grouped',
                          launch_environment=grouped['launch_environment']),
    }


def _choose(pairs):
    _require(pairs['schema'] == PAIRED_SELECTION and pairs['schema_version'] == 1
             and pairs['incumbent'] == _INCUMBENT and pairs['challenger'] == _CHALLENGER
             and pairs['policy'] == dict(minimum_gain=float(_MINIMUM), strict_observed_separation=True)
             and pairs['deployment_authorized'] is False, 'unsupported clean paired route comparison policy')
    incumbent, challenger = (pairs['metrics'][route] for route in (_INCUMBENT, _CHALLENGER))
    baseline_median, candidate_median = incumbent['median_ns_twice'], challenger['median_ns_twice']
    _require(type(baseline_median) is type(candidate_median) is int and min(baseline_median, candidate_median) > 0,
             'positive exact paired route medians required')
    gain = Fraction(baseline_median - candidate_median, baseline_median)
    separated = challenger['max_ns'] < incumbent['min_ns']
    nominated = gain >= _MINIMUM and separated
    verdict = 'ADVANCE_TO_RELEASE_REVIEW' if nominated else 'KEEP_INCUMBENT'
    _require(separated == pairs['observed_envelopes_separated']
             and pairs['verdict'] == verdict,
             'paired route envelope or verdict differs from replay')
    summary = dict(
        measured_pairs=pairs['measurement_pairs'], warmup_pairs=pairs['warmup_pairs'],
        retained_process_median_ns_twice=baseline_median, grouped_openmp_median_ns_twice=candidate_median,
        retained_process_min_ns=incumbent['min_ns'], retained_process_max_ns=incumbent['max_ns'],
        grouped_openmp_min_ns=challenger['min_ns'], grouped_openmp_max_ns=challenger['max_ns'],
        observed_envelopes_separated=separated, gain_fraction=float(gain),
        exact_gain_fraction=dict(numerator=gain.numerator, denominator=gain.denominator),
        source_pairing_verdict=pairs['verdict'])
    return (_CHALLENGER if nominated else _INCUMBENT), summary, pairs['reasons']


def _observation_scope(source, proof_ref, comparison):
    workload, resources, case = source['workload'], source['resources'], comparison['case']
    _require(source['status'] == _PROVEN and comparison['retention'] == source['retention']
             and comparison['grouped'] == source['baseline_grouped']
             and comparison.get('proof', proof_ref) == proof_ref
             and all(case[key] == workload[key] for key in ('argv', 'recipe', 'holdout'))
             and case['cpu_ids'] == resources['launch_cpu_ids']
             and comparison['deployment_authorized'] is False and comparison['target_executed'] is True,
             'comparison must share the exact original launch, CPU cohort and two route packages')


def _describe(comparison):
    chosen, summary, reasons = _choose(comparison['pairs'])
    states = comparison['state_diagnostics']
    summary.update(
        exact_round_state_checkpoints=states['exact_checkpoints'], exact_round_state_bytes=states['exact_bytes'],
        population_variance_ns2=comparison['population_variance_ns2'])
    return chosen, summary, reasons


def _checkpoint_oracle(comparison):
    # The observer already checks both arms against each other at each position.
    return tuple((Path(row['uri']).name, row['size_bytes'], row['sha256'])
                 for row in comparison['state_diagnostics']['checkpoints'][_INCUMBENT])


def _distinct_captures(comparison, paths, stdout_digests):
    for row in (*comparison['runs'], *comparison['state_diagnostics']['runs']):
        captures = {Path(row[key]['uri']).resolve(strict=True) for key in ('stdout', 'stderr', 'published')}
        digest = row['stdout']['sha256']
        _require(len(captures) == 3 and captures.isdisjoint(paths) and digest not in stdout_digests,
                 'campaigns must use disjoint raw process captures, including warmup and diagnostics')
        paths.update(captures)
        stdout_digests.add(digest)


def _replication(campaigns):
    paths, stdout_digests, outcomes = set(), set(), []
    oracle = _checkpoint_oracle(campaigns[0][1])
    for reference, comparison in campaigns:
        _require(_checkpoint_oracle(comparison) == oracle, 'replication changed the ordered round-state oracle')
        _distinct_captures(comparison, paths, stdout_digests)
        chosen, summary, reasons = _describe(comparison)
        outcomes.append(dict(observation=reference, gate_passed=chosen == _CHALLENGER,
                             summary=summary, reasons=reasons))
    passed = sum(row['gate_passed'] for row in outcomes)
    unanimous = passed == len(outcomes)
    disagree = 0 < passed < len(outcomes)
    reasons = list(dict.fromkeys(reason for row in outcomes for reason in row['reasons']))
    if disagree:
        reasons.append('CAMPAIGN_VERDICTS_DISAGREE')
    result = dict(
        status=('ALL_CAMPAIGNS_PASS' if unanimous else 'CAMPAIGN_VERDICTS_DISAGREE' if disagree
                else 'NO_CAMPAIGN_PASSES'),
        policy='REQUIRE_EXISTING_PAIRED_GATE_IN_EVERY_CAMPAIGN',
        campaign_count=len(outcomes), passing_campaigns=passed, campaigns=outcomes,
        pooled_samples=False, same_checkpoint_oracle_verified=True, disjoint_process_captures_verified=True)
    return (_CHALLENGER if unanimous else _INCUMBENT), reasons, result


def derive_cpu_array_native_route_plan(proof_ref, comparison_ref, *, replications=()):
    """Nominate a native route only when every supplied clean campaign passes."""
    from scar.analysis.cpu_array_native_route import verify_cpu_array_native_route
    from scar.compiler.cpu_array_native_route_trial import verify_cpu_array_native_route_observation

    _require(proof_ref['schema'] == _PROOF and comparison_ref['schema'] == _OBSERVATION
             and type(replications) in (tuple, list), 'scoped route proof and clean paired observations required')
    with replay_transaction():
        source = verify_cpu_array_native_route(proof_ref)
        comparison = verify_cpu_array_native_route_observation(comparison_ref)
        _observation_scope(source, proof_ref, comparison)
        campaigns = [(comparison_ref, comparison)]
        paths = {Path(comparison_ref['uri']).resolve(strict=True)}
        digests = {comparison_ref['sha256']}
        for reference in replications:
            path = Path(reference['uri']).resolve(strict=True)
            _require(reference['schema'] == _OBSERVATION and path not in paths and reference['sha256'] not in digests,
                     'replication requires a distinct verified observation')
            measured = verify_cpu_array_native_route_observation(reference)
            _observation_scope(source, proof_ref, measured)
            campaigns.append((reference, measured))
            paths.add(path)
            digests.add(reference['sha256'])

    workload, resources = source['workload'], source['resources']
    routes = _routes(source)
    chosen, summary, reasons = _describe(comparison)
    replication = None
    if replications:
        chosen, reasons, replication = _replication(campaigns)
    nominated = chosen == _CHALLENGER
    report = dict(
        schema=SCHEMA, schema_version=1, proof=proof_ref, comparison=comparison_ref,
        retention=source['retention'], grouped=source['baseline_grouped'], scope=source['scope'],
        workload=workload, resources=resources, effect_contract=source['effects'],
        routes=routes, selected_route=chosen, fallback_route=_INCUMBENT,
        selected_executable=routes[chosen]['executable'], fallback_executable=routes[_INCUMBENT]['executable'],
        status='SCOPED_NATIVE_CHALLENGER_NOMINATED' if nominated else 'SCOPED_NATIVE_INCUMBENT_RETAINED',
        reasons=reasons, summary=summary,
        empirical_evidence=dict(
            pairing='#/pairs', state_validation='#/state_diagnostics',
            basis='OBSERVED_CLEAN_NATIVE_WALL_TIME_ENVELOPES',
            minimum_gain=float(_MINIMUM), inclusive_minimum_gain=True,
            strict_observed_separation=True, serves_as_analytic_hardware_bound=False),
        coverage_frontier=[
            dict(role='RESOURCE_PLACEMENT', status='EXACT_SCOPED_ROUTE_REQUIREMENTS', evidence='#/resources'),
            dict(role='TIMING', status='EMPIRICAL_PAIRED_VALIDATION', blocker='ANALYTIC_END_TO_END_ROUTE_COST_OPEN'),
            dict(role='DELIVERY', status='OPT_IN_RELEASE_REVIEW_ONLY', blocker='ORIGINAL_DEPLOYMENT_AUTHORITY_NOT_GRANTED')],
        analytical_cost_bounds_ns=dict(retained_process=None, grouped_openmp=None),
        forward_cost_proven=False, target_executed=False, original_default_retained=True,
        execution_default='NO_OP', transformation_authorized=False,
        whole_entry_effect_equivalence_proven=False, deployment_authorized=False)
    if replication is not None:
        report.update(replications=list(replications), replication=replication)
    return report


def verify_cpu_array_native_route_plan(reference):
    """Cold-replay both native routes, the empirical receipts and the scoped decision."""
    saved = read_artifact(reference)
    _require(reference['schema'] == SCHEMA and type(saved['schema_version']) is int and saved['schema_version'] == 1,
             'unsupported CPU array native route plan')
    replay = derive_cpu_array_native_route_plan(
        saved['proof'], saved['comparison'], replications=saved.get('replications', ()))
    _require(saved == replay, 'saved native route plan differs at cold replay')
    return replay

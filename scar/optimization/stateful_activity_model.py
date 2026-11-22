"""Compose source-accounted serial activities into a conditional entry model.

The existing component table stays unchanged. This bridge sums each final
subject once under explicit canonical-transfer assumptions; it neither fits
target timings nor turns component ranges into physical latency bounds.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import json

from scar.ir.activity_costs import CostBasis
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.record_codec import encode
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.stateful-activity-entry-model'
FACTS = 'scar.stateful-activity-entry-facts'
BASIS = 'SOURCE_ACCOUNTED_SERIAL_ACTIVITY_POINT_MODEL'
PROFILE = 'scar.stateful-cpu-cost-profile'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _reference(path, schema):
    return path if type(path) is dict else artifact_reference(Path(path).resolve(), schema)


def _facts(reference):
    from scar.compiler.cost_activities import replay_activity_costs
    from scar.ir.versioned_schedule import TemporalScopeKind
    from scar.optimization.stateful_cpu_model import _domain, stateful_cpu_cost_context

    saved = read_artifact(reference)
    report, table, derivation, schedule, _ = replay_activity_costs(reference['uri'])
    _need(saved == report and report['domain'] is not None, 'original source-accounted activity receipt required')
    domain = _domain(read_artifact(report['domain']))
    candidate = next((row for row in domain['candidates'] if row['id'] == report['candidate_id']), None)
    _need(candidate is not None and candidate['legal_under_guards'] and candidate['worker_count'] == 1,
          'conditional entry composition requires an admitted standalone serial candidate')
    resources = read_artifact(report['resources'])
    time = read_artifact(resources['time_schedule'])
    envelope = schedule.proof.get('implementation_envelope', {})
    _need(time['source_kind'] == 'stateful_recipe'
          and envelope.get('invocation_scope') == 'ONE_STANDALONE_STATEFUL_METHOD_CALL_NOT_REPEAT_ADAPTER'
          and 'implementation_scope_cleanup' in schedule.proof
          and envelope.get('phase_order') == [task.id for task in schedule.tasks]
          and len(schedule.temporal.scopes) == 1
          and schedule.temporal.scopes[0].kind is TemporalScopeKind.ROOT
          and schedule.temporal.scopes[0].iteration_symbol is None,
          'complete normal-success single-root source scope required; loops and repeat adapters need another model')
    _need(all(previous.id in following.depends_on
              for previous, following in zip(schedule.tasks, schedule.tasks[1:])),
          'source activities must form a complete serial dependency chain before costs can be added')
    _need((table.tick_numerator, table.tick_denominator) == (1, 10**9),
          'exact integer nanosecond activity costs required')
    ledger = derivation.get('scope_accounting', {})
    subjects = {row.subject for row in table.costs}
    _need(ledger.get('schema') == 'scar.stateful-scope-accounting'
          and not ledger['numeric_unknown_subjects'] and not ledger['source_frontiers']
          and ledger['original_demand_terms_accounted'] is True
          and ledger['original_activity_ownership_exactly_once'] is True
          and len(ledger['active_subjects']) == len(subjects)
          and set(ledger['active_subjects']) == subjects,
          'final source ownership and numeric activity coverage must be closed')
    represented = {ledger['milestone']['subject']}
    for term in ledger['terms']:
        _need(term['disposition'] in ('ADDITIVE_COMPONENT', 'SUPERSEDED_COARSE_DESCRIPTION')
              and term['cost_charged_separately'] is False
              and term['legacy_numeric_estimate_added'] is False,
              'old demand descriptions cannot add separate cost or conceal a source frontier')
        represented.update(term['active_subjects'])
    _need(represented == subjects, 'original demand ownership must reach every final cost subject')
    _need(all(row['additional_work'] is False and row['cost_charged'] is False
              for row in (*ledger['residuals'], *ledger['subject_residuals'])),
          'transfer residuals cannot be silently converted to zero-cost work')
    activities = {row.id: index for index, row in enumerate(schedule.temporal.activities)}
    charged, charge_keys, conditions, residual_terms = [], set(), [], set()
    total = 0
    for row in sorted(table.costs, key=lambda item: min(activities[key] for key in item.activities)):
        _need(type(row.estimate_ticks) is int and row.estimate_ticks >= 0,
              'all final subjects need a nonnegative integer point estimate')
        indices = sorted(activities[key] for key in row.activities)
        _need(indices == list(range(indices[0], indices[-1]+1)),
              'collective costs must cover a contiguous original serial activity group')
        if row.basis is CostBasis.SOURCE_CONTROL_MILESTONE:
            _need(row.subject == ledger['milestone']['subject'] and row.estimate_ticks == 0
                  and not row.charge_keys and not row.missing_terms,
                  'only the source-certified nonadditive control milestone may omit charge keys')
        else:
            _need(row.basis is CostBasis.DECLARED_COMPONENT_MODEL and row.charge_keys
                  and not charge_keys.intersection(row.charge_keys),
                  'unique occurrence charges and declared canonical components required, not observed scopes')
        charge_keys.update(row.charge_keys)
        conditions.extend(row.conditions)
        residual_terms.update(row.missing_terms)
        total += row.estimate_ticks
        charged.append(dict(subject=row.subject, activities=list(row.activities),
            estimate_ticks=row.estimate_ticks, basis=row.basis.value,
            original_coverage=row.coverage.value, component_refs=encode(row.component_refs),
            charge_keys=list(row.charge_keys), conditions=list(row.conditions),
            transfer_residuals=list(row.missing_terms), charged_once=True))
    context = stateful_cpu_cost_context(domain, report['candidate_id'])
    _need(context['scope']['call_scope'] == 'ONE_ORIGINAL_METHOD_CALL_SITE'
          and context['scope']['reuse_count'] == 1 and context['scope']['boundary'] == 'WARM_SESSION',
          'original complete warm method-call context required')
    assumptions = list(dict.fromkeys([
        'CONDITIONAL_POINT_PREDICTION_FROM_CANONICAL_COMPONENT_MEDIANS_NOT_TARGET_OBSERVATION',
        'COMPLETE_SERIAL_SOURCE_ACTIVITIES_ADD_ONCE_WITHOUT_HIDDEN_PARALLEL_OVERLAP',
        'CANONICAL_COMPONENT_TRANSFER_ASSUMED_FOR_THIS_SOURCE_SHAPE_CPU_AND_DATA_REGIME',
        'TRANSFER_RESIDUALS_ARE_MODEL_UNCERTAINTY_NOT_UNCOUNTED_EXTRA_WORK',
        'NO_GUARANTEED_ERROR_BOUND_NO_PHYSICAL_RESOURCE_OR_DEPLOYMENT_AUTHORIZATION',
        *conditions]))
    return dict(schema=FACTS, schema_version=1, context=context,
        estimate=dict(duration_ticks=total, tick_seconds=1e-9),
        derivation=dict(origin='DECLARED_FORWARD_MODEL', basis=BASIS,
            target_variant_trials=False, assumptions=assumptions,
            activity_cost_receipt=reference,
            scope_accounting_ref=dict(artifact=report['derivation'], locator='#/scope_accounting'),
            charged_subjects=charged, serial_task_order=[row.id for row in schedule.tasks],
            residuals=deepcopy(ledger['residuals']), subject_residuals=deepcopy(ledger['subject_residuals']),
            transfer_residual_terms=sorted(residual_terms),
            canonical_transfer_validated=False, original_component_table_relabelled=False,
            component_ranges_summed_as_bounds=False, whole_entry_physical_cost_closed=False,
            implementation_source=artifact_reference(Path(__file__).resolve(), 'scar.python-source')))


def _profile(facts, reference):
    fact = lambda locator: dict(artifact=reference, locator=locator)
    return dict(schema=PROFILE, schema_version=1, context=deepcopy(facts['context']),
        context_evidence=[fact('#/context')],
        estimate={**deepcopy(facts['estimate']), 'evidence':[fact('#/estimate')]},
        derivation={**deepcopy(facts['derivation']), 'evidence':[fact('#/derivation')]})


def verify_activity_entry_profile(profile):
    """Replay a common profile's source composition, not just its JSON hashes."""
    _need(profile.get('derivation', {}).get('basis') == BASIS, 'source-accounted activity entry basis required')
    reference = profile['derivation']['activity_cost_receipt']
    expected = _facts(reference)
    facts_reference = profile['context_evidence'][0]['artifact']
    _need(read_artifact(facts_reference) == expected and profile == _profile(expected, facts_reference),
          'activity entry profile differs from exact source accounting and component replay')
    return expected


def _summary(reference, facts_reference, profile_reference, facts):
    return dict(schema=SCHEMA, schema_version=1, activity_costs=reference,
        facts=facts_reference, profile=profile_reference,
        duration_ns=facts['estimate']['duration_ticks'], subjects=len(facts['derivation']['charged_subjects']),
        activities=sum(len(row['activities']) for row in facts['derivation']['charged_subjects']),
        conditional_point_estimate_available=True, canonical_transfer_validated=False,
        source_coverage_accounted=True, component_table_unchanged=True,
        complete_target_implementation_cost=False, physical_resource_qualification='SEPARATE_UNRESOLVED',
        target_program_executed=False, target_versions_ranked=False, solver_invoked=False,
        performance_measured=False, deployment_authorized=False, default_backend='NO_OP')


def prepare_activity_entry_model(activity_costs, out):
    reference = _reference(activity_costs, 'scar.activity-cost-preparation')
    facts = _facts(reference)
    directory = Path(out).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        path = directory/name
        with path.open('x') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
        return artifact_reference(path, value['schema'])
    facts_reference = write('facts.json', facts)
    profile_reference = write('profile.json', _profile(facts, facts_reference))
    result = _summary(reference, facts_reference, profile_reference, facts)
    write('result.json', result)
    return result


def read_activity_entry_model(path):
    reference = _reference(path, SCHEMA)
    report = read_artifact(reference)
    facts = _facts(report['activity_costs'])
    _need(read_artifact(report['facts']) == facts
          and read_artifact(report['profile']) == _profile(facts, report['facts'])
          and report == _summary(report['activity_costs'], report['facts'], report['profile'], facts),
          'entry model differs from source-accounted serial point composition')
    read_artifact(reference)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--activities'); source.add_argument('--check')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check:
        if args.out: parser.error('check consumes the retained model only')
        report = read_activity_entry_model(args.check)
    else:
        if not args.out: parser.error('--activities requires --out')
        report = prepare_activity_entry_model(args.activities, args.out)
    print(json.dumps({key:report[key] for key in ('duration_ns', 'subjects', 'activities',
        'conditional_point_estimate_available', 'canonical_transfer_validated', 'profile',
        'deployment_authorized')}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Select a borrowed-root launch after source, resource and cost replay."""

import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import (
    CostEstimate, ImplementationOption, PlanningOperation, PlanningProblem,
    PlanningResource, problem_document, validate_problem,
)
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import decode
from scar.optimization.constructive_planner import construct_implementations


SCHEMA = 'scar.snapshot-launch-selection'
QUALIFICATION = 'scar.snapshot-entry-cost-qualification'
NATIVE = 'LOAD_NATIVE_CANDIDATE'
PYTHON = 'NO_OP'


def _need(condition, detail):
    if not condition:
        raise ValueError(detail)


def _ref(value, schema):
    return value if type(value) is dict else artifact_reference(Path(value).resolve(), schema)


def _write(path, document):
    with path.open('x') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, document['schema'])


def _derive(qualification):
    from scar.compiler.compile_workspace_borrowed_roots import SCHEMA as BUILD, check as check_build
    from scar.compiler.snapshot_entry_calibration import check as check_cost

    qualified = check_cost(qualification['uri'])
    _need(qualification['schema'] == QUALIFICATION and qualified == read_artifact(qualification),
        'same independently replayed snapshot entry cost qualification required')
    context = qualified['context']
    candidate = context['candidate']
    _need(candidate['schema'] == BUILD, 'original compiled borrowed-root candidate required')
    build = check_build(candidate['uri'])
    _need(build == read_artifact(candidate) and build['scope_admission'] == context['scope_admission'],
        'cost evidence and C++ build must use the same guarded scope admission')
    _need(context['entry']['schema'] == 'scar.python-source'
        and type(context['argv']) is list and all(type(arg) is str for arg in context['argv'])
        and type(context['output_token']) is str and bool(context['output_token'])
        and context['argv'].count(context['output_token']) == 1,
        'one exact structured launch and its output token required')
    read_artifact(context['entry'], json_document=False)

    inner = read_artifact(build['snapshot_planning'])
    demand = read_artifact(inner['demand'])
    resources = demand['resources']
    cpu_ids = resources['cpu_ids']
    caller_cpus = context['caller_cpu_ids']
    fits = (bool(cpu_ids) and set(cpu_ids) <= set(caller_cpus)
        and resources['private_payload_bytes'] + resources['borrowed_payload_bytes'] <= resources['budget_bytes'])
    offered = qualified['qualifies'] is True and qualified['decision'] == NATIVE and fits
    comparison = qualified['prediction']
    lane, pool = 'entry-call', 'original-native-array-payload'
    source = ({'artifact':qualification}, {'artifact':candidate}, {'artifact':inner['demand']})
    durations = ((PYTHON, 'original_ns', {}), (NATIVE, 'native_ns', {pool:resources['private_payload_bytes']}))
    options = []
    for mode, cost_key, workspace in durations:
        estimate = comparison[cost_key] if offered else None
        if offered:
            _need(type(estimate) is int and estimate > 0, 'positive qualified full-entry estimates required')
        cost = CostEstimate(estimate, None, None, 'EMPIRICAL_MODEL' if offered else 'UNKNOWN',
            qualification['sha256']+':complete-launch', source,
            () if offered else ('UNQUALIFIED_COMPLETE_ENTRY_COST_OR_PAYLOAD',))
        options.append(ImplementationOption(mode, mode, source, {lane:1}, workspace, {}, {}, cost,
            dict(scope='COMPLETE_ORIGINAL_ENTRY_LAUNCH', native_cpu_ids=cpu_ids if workspace else [],
                native_payload_scope_only=True)))
    problem = validate_problem(PlanningProblem(decode(ProviderArtifact, resources['target']), source,
        (PlanningResource(lane, 'EXECUTION', 1, source),
         PlanningResource(pool, 'MEMORY', resources['budget_bytes'], source)),
        (PlanningOperation('snapshot-entry:'+context['entry']['sha256'], source, (), tuple(options), PYTHON),),
        (), assumptions=('ONE_COMPLETE_ENTRY_WITH_QUALIFIED_CONTEXT', 'ORIGINAL_NATIVE_PAYLOAD_BUDGET_ONLY',
            'NO_EXCLUSIVE_RESOURCE_RESERVATION', 'SOURCE_AND_RUNTIME_GUARDS_REMAIN_REQUIRED'),
        reserved_memory={pool:resources['borrowed_payload_bytes']}))
    plan = construct_implementations(problem)
    selected = plan['selected']
    decision = NATIVE if offered and selected and selected[0]['option'] == NATIVE else PYTHON
    body = dict(schema=SCHEMA, schema_version=1, qualification=qualification, candidate=candidate,
        bundle=build['bundle'], scope_admission=build['scope_admission'], context=context,
        source_planning=build['snapshot_planning'], work_demand=inner['demand'], decision=decision,
        expected_times_ns=comparison, original_native_cpu_ids=cpu_ids,
        original_payload_budget_fits=fits, requires_exclusive_inputs=True,
        declared_context_only=True, default_backend=PYTHON, deployment_authorized=False)
    return body, problem_document(problem), plan


def prepare(qualification, out):
    body, problem, plan = _derive(_ref(qualification, QUALIFICATION))
    directory = Path(out).resolve()
    directory.mkdir(parents=True)
    body['problem'] = _write(directory/'problem.json', problem)
    body['plan'] = _write(directory/'plan.json', plan)
    return _write(directory/'result.json', body)


def check(result):
    """Cold-replay the source graph, C++ bytes, fit, holdout and decision."""
    saved = read_artifact(_ref(result, SCHEMA))
    body, problem, plan = _derive(saved['qualification'])
    saved_plan = read_artifact(saved['plan'])
    _need(read_artifact(saved['problem']) == problem
        and {k:v for k,v in saved_plan.items() if k != 'elapsed_seconds'} ==
            {k:v for k,v in plan.items() if k != 'elapsed_seconds'}
        and saved == {**body, 'plan':saved['plan'], 'problem':saved['problem']},
        'selected snapshot launch differs from independent model and source replay')
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--qualification')
    mode.add_argument('--check')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check:
        result = check(args.check)
        print(json.dumps(dict(decision=result['decision'], planning_replayed=True, deployment_authorized=False)))
    elif args.out:
        print(json.dumps(prepare(args.qualification, args.out), sort_keys=True))
    else:
        parser.error('--qualification requires a fresh --out directory')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

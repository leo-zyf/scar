"""Compare an expanded worker team with a previously qualified team."""

import argparse
import json
from pathlib import Path
from statistics import median

from scar.analysis.sparse_parallel_scaling_scope import SCHEMA as SOURCE, check_sparse_parallel_scaling_scope
from scar.compiler.plan_sparse_parallel import _peak, _ref, _require, _write
from scar.ir.execution_recipe import read_artifact
from scar.ir.implementation_plan import (CostEstimate, ImplementationOption, PlanningOperation,
    PlanningProblem, PlanningResource, problem_document)
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import decode
from scar.optimization.constructive_planner import construct_implementations
from scar.optimization.sparse_parallel_scaling_cost import SCHEMA as COST, check
from scar.runtime.sparse_parallel_entry import SCHEMA as CHILD
from scar.runtime.sparse_parallel_scaling import EXECUTION, SCHEMA

VALIDATION = 'scar.sparse-parallel-scaling-validation'
MODES = ('ESTABLISHED', 'EXPANDED')
COST_MODES = dict(zip(MODES, ('baseline', 'expanded'), strict=True))


def _problem(scope, source, profile, calibration, teams):
    refs = (dict(artifact=source, locator='#/resource_edges'),
            dict(artifact=calibration, locator='#/projected_section_ns'))
    owner, worker, memory = 'current-owner', 'independent-worker-cores', 'declared-address-space-and-private-buffers'
    capacity = teams['EXPANDED']['memory_floor_bytes']
    options = []
    for mode in MODES:
        team = teams[mode]
        bounds = profile['projected_section_ns'][COST_MODES[mode]]
        estimate = CostEstimate(bounds['estimate_ns'], bounds['lower_ns'], bounds['upper_ns'],
            'EMPIRICAL_MODEL', calibration['sha256'], refs)
        options.append(ImplementationOption(mode, mode, refs,
            {owner:1, worker:team['workers']['window']}, {memory:team['memory_floor_bytes']}, {}, {},
            estimate, dict(owner_cpu=scope['owner_cpu'], workers=team['workers'], same_ordered_carrier=True)))
    target = read_artifact(teams['ESTABLISHED']['problem'])['target']
    resources = (PlanningResource(owner, 'EXECUTION', 1, refs),
        PlanningResource(worker, 'EXECUTION', teams['EXPANDED']['workers']['window'], refs),
        PlanningResource(memory, 'MEMORY', capacity, refs))
    op = PlanningOperation('sparse-capacity:'+source['sha256'], refs, (), tuple(options), 'ESTABLISHED')
    return PlanningProblem(decode(ProviderArtifact,target), refs, resources, (op,), (),
        assumptions=('UNCHANGED_ORIGINAL_PUBLICATION_ORDER', 'MODELED_PHASES_EXCLUDE_WHOLE_ENTRY_REMAINDER',
                     'WORKER_ADDRESS_SPACE_CAP_IS_NOT_DRAM_COMMITMENT'))


def prepare(scope_path, expanded, cost, out):
    scope_ref = _ref(scope_path, SOURCE)
    scope = check_sparse_parallel_scaling_scope(scope_ref)
    cost_ref = _ref(cost, COST)
    profile = check(cost_ref['uri'])
    refs = dict(ESTABLISHED=scope['released_selection'], EXPANDED=_ref(expanded, CHILD))
    teams = {name:read_artifact(ref) for name, ref in refs.items()}
    base, large = (teams[mode] for mode in MODES)
    comparable = ('entry', 'source', 'context', 'count', 'owner_cpu', 'numa_node', 'matrix',
                  'native', 'serial_launch', 'serial_study', 'symbolic_budget_bytes')
    _require(all(base[key] == large[key] for key in comparable)
             and base['source'] == scope['parallel_scope'] and base['entry'] == scope['entry']['source']
             and base['decision'] == 'PARALLEL' and base['study_parallel'] is False
             and large['study_parallel'] is True and large['launch_cpu_ids'] == scope['launch_cpu_ids']
             and large['cost'] == profile['costs']['expanded'] and profile['source_scope'] == scope['source_scope']
             and profile['owner_cpu_id'] == scope['owner_cpu'] and profile['output_bitwise'] is True
             and base['workers'] == scope['previous_workers'] and large['workers'] == scope['selected_workers']
             and read_artifact(base['recipe']) == read_artifact(large['recipe']),
             'expanded capacity must reuse the same original loop, numeric implementation and ordered publication')
    for name, team in teams.items():
        observed = profile['teams'][COST_MODES[name]]
        clock = read_artifact(profile['costs'][COST_MODES[name]])
        _require(observed['cpu_ids'] == team['workers']['cpu_ids'] and observed['window'] == team['workers']['window']
                 and clock['native'] == team['native'] and clock['context']['cpu_ids'] == team['launch_cpu_ids']
                 and {**clock['context'], 'cpu_ids':[team['owner_cpu']]} == team['context'],
                 'paired timing does not use the intended isolated core team')
    directory = Path(out).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    problem = _problem(scope, scope_ref, profile, cost_ref, teams)
    problem_ref = _write(directory,'problem.json',problem_document(problem))
    plan_ref = _write(directory,'plan.json',construct_implementations(problem))
    runtime = Path(__file__).resolve().parents[1]/'runtime'/'sparse_parallel_scaling.py'
    return _write(directory,'selection.json',dict(schema=SCHEMA, schema_version=1, source=scope_ref, cost=cost_ref,
        teams=refs, entry=base['entry'], context=scope['context'], owner_cpu=scope['owner_cpu'],
        launch_cpu_ids=scope['launch_cpu_ids'], count=scope['count'], problem=problem_ref, plan=plan_ref,
        runtime=_ref(runtime,'scar.python-source'), study_expanded=True, decision='ESTABLISHED'))


def _validate_run(row, name, study_ref, study, team):
    report = read_artifact(row['launch'])
    _require(row['launch']['schema'] == EXECUTION and report['scaling']['fingerprint'] == study_ref['sha256']
             and report['scaling']['applied'] == name and report['applied'] == 'PARALLEL'
             and report['selection_sha256'] == study['teams'][name]['sha256']
             and report['entry_path'] == study['entry']['uri'] and report['argv'] == study['context']['argv']
             and report['exit_code'] == 0 and report['statistics']['fallback_calls'] == 0
             and report['statistics']['native_calls'] == study['count'], 'run executed a different team or scope')
    parallel = report['parallel']
    workers = parallel['workers']
    count = study['count']
    window = team['workers']['window']
    cap = team['workers']['address_space_limit_bytes']
    _require(parallel['matched'] == workers['submissions'] == count
             and parallel['serial_calls'] == parallel['rejected'] == 0 and parallel['inputs_captured']
             and parallel['peak_window'] == window and _peak(workers['tasks']) == window
             and workers['closed'] is True and workers['retained_after_close_bytes'] == workers['pending_snapshot_bytes'] == 0
             and [item['issue'] for item in workers['publications']] == list(range(count))
             and sorted(item['cpu'] for item in workers['workers']) == team['workers']['cpu_ids']
             and all(item['address_space_limit_bytes'] == cap for item in workers['workers']),
             'parallel overlap, order, memory cap or cleanup differs from the compiled team')
    _require(type(row['wall_ns']) is int and row['wall_ns'] > 0, 'positive process time required')


def qualify(study_path, validation, out):
    study_ref = _ref(study_path, SCHEMA)
    study = read_artifact(study_ref)
    validation_ref = _ref(validation, VALIDATION)
    evidence = read_artifact(validation_ref)
    teams = {name:read_artifact(ref) for name,ref in study['teams'].items()}
    _require(evidence['schema_version'] == 1 and evidence['selection'] == study_ref and evidence['clean'] is True
             and evidence['context'] == {**study['context'], 'cpu_ids':study['launch_cpu_ids']},
             'full entry results do not bind the exact team comparison')
    warm, repeats = evidence['warmup_pairs'], evidence['measurement_pairs']
    indexed = {(row['pair'],row['mode'].upper()):row for row in evidence['runs']}
    _require(type(warm) is int and warm >= 0 and type(repeats) is int and repeats >= 3
             and set(indexed) == {(n,mode) for n in range(warm+repeats) for mode in MODES}
             and len(indexed) == len(evidence['runs']), 'matched clean process pairs required')
    _require(len({(row['stdout_sha256'],row['stderr_sha256']) for row in indexed.values()}) == 1,
             'team expansion changed an observable output')
    clocks = {mode:[] for mode in MODES}
    for (pair,mode), row in sorted(indexed.items()):
        _validate_run(row, mode, study_ref, study, teams[mode])
        if pair >= warm:
            clocks[mode].append(row['wall_ns'])
    costs = check(study['cost']['uri'])
    plan = read_artifact(study['plan'])
    preferred = any(row['option'] == 'EXPANDED' for row in plan['selected'])
    win = max(clocks['EXPANDED']) < min(clocks['ESTABLISHED'])
    enabled = preferred and win and costs['output_bitwise'] is True
    directory = Path(out).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return _write(directory,'selection.json',{**study, 'study':study_ref, 'validation':validation_ref,
        'study_expanded':False, 'decision':'EXPANDED' if enabled else 'ESTABLISHED',
        'full_entry_median_ns':{mode:median(values) for mode,values in clocks.items()},
        'expanded_all_samples_faster':win, 'positive_point_section_plan':preferred})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope')
    parser.add_argument('--expanded')
    parser.add_argument('--cost')
    parser.add_argument('--study')
    parser.add_argument('--validation')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if all((args.scope,args.expanded,args.cost)) and not args.study:
        result = prepare(args.scope,args.expanded,args.cost,args.out)
    elif args.study and args.validation and not args.scope:
        result = qualify(args.study,args.validation,args.out)
    else:
        parser.error('use --scope/--expanded/--cost or --study/--validation')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

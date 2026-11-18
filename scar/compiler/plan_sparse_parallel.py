"""Build and qualify an ordered sparse solve plan for a finite worker team."""

import argparse
import json
from pathlib import Path
from statistics import median
import sys

from scar.analysis.sparse_parallel_recipe import derive_recipe
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import (CostEstimate, ImplementationOption, PlanningOperation,
    PlanningProblem, PlanningResource, problem_document)
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import decode
from scar.optimization.constructive_planner import construct_implementations
from scar.optimization.sparse_parallel_cost import SCHEMA as COST, check
from scar.runtime.sparse_parallel_entry import SCHEMA
from scar.runtime.sparse_solve_launch import LAUNCH as SERIAL_LAUNCH

SCOPE = 'scar.sparse-parallel-scope'
RECIPE = 'scar.sparse-parallel-detached-recipe'
VALIDATION = 'scar.sparse-parallel-entry-validation'
EXECUTION = 'scar.sparse-parallel-entry-execution'
FIELDS = ('estimate_ns', 'lower_ns', 'upper_ns')


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(), schema)


def _write(directory, name, value):
    path = directory/name
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return _ref(path, value['schema'])


def _planning(source, cost, cost_ref, target, count, memory_floor, lanes):
    refs = (dict(artifact=source, locator='#/dependences'), dict(artifact=cost_ref, locator='#/projections'))
    serial = tuple(cost['serial']['fixed_lifecycle_ns'][key] + (count-1)*cost['serial']['warm_ns'][key]
                   for key in FIELDS)
    parallel = tuple(cost['projections'][key]['elapsed_projection_ns'] for key in FIELDS)
    owner, team, cap = 'owner-thread', 'isolated-worker-cores', 'declared-address-space-and-owned-payload'
    before = CostEstimate(*serial, 'EMPIRICAL_MODEL', cost_ref['sha256'], refs)
    after = CostEstimate(*parallel, 'EMPIRICAL_MODEL', cost_ref['sha256'], refs)
    options = (
        ImplementationOption('SERIAL', 'SERIAL', refs, {owner:1}, {}, {}, {}, before, dict(call_count=count)),
        ImplementationOption('PARALLEL', 'PARALLEL', refs, {owner:1,team:lanes}, {cap:memory_floor}, {}, {},
            after, dict(call_count=count, ordered_window=lanes, retained_effect_order=True)),
    )
    issue = PlanningOperation('ordered-sparse-solve:'+source['sha256'], refs, (), options, 'SERIAL')
    resources = (PlanningResource(owner, 'EXECUTION', 1, refs), PlanningResource(team, 'EXECUTION', lanes, refs),
                 PlanningResource(cap, 'MEMORY', memory_floor, refs))
    return PlanningProblem(decode(ProviderArtifact,target), refs, resources, (issue,), (),
        assumptions=('CAPPED_WORKER_ADDRESS_SPACE_NOT_PHYSICAL_RESERVATION', 'OPAQUE_ORIGINAL_EFFECTS_ON_OWNER',
                     'EMPIRICAL_SERVICE_PROJECTION_INCLUDES_BOOTSTRAP_QUEUE_AND_CLOSE'))


def prepare(scope_path, serial_path, cost_path, out):
    from scar.analysis.sparse_parallel_scope import check_sparse_parallel_scope

    scope_ref = _ref(scope_path, SCOPE)
    scope = check_sparse_parallel_scope(scope_ref)
    serial_ref = _ref(serial_path, SERIAL_LAUNCH)
    serial = read_artifact(serial_ref)
    attached = read_artifact(serial['study'])
    cost_ref = _ref(cost_path, COST)
    cost = check(cost_ref['uri'])
    design = read_artifact(cost['design'])
    entry = scope['entry']['source']
    recipe = derive_recipe(read_artifact(entry, json_document=False), scope['source_body'],
                          scope['preparation'], optimize=sys.flags.optimize)
    owner, workers, window = cost['owner_cpu_id'], cost['worker_cpu_ids'], cost['window']
    cpus = sorted({owner, *workers})
    choice = read_artifact(cost['choice'])
    count = choice['dimensions']['call_count']
    source_count = read_artifact(scope['source_scope'])['observed']['consumer_completions']
    _require(entry == serial['entry'] == attached['entry'] and serial['decision'] == 'LOAD_NATIVE_CANDIDATE'
             and cost['parallel_scope'] == scope_ref and cost['source_scope'] == scope['source_scope']
             and attached['source'] == scope['solve_scope'] and attached['native'] == cost['native']
             and {**cost['context'], 'cpu_ids':[owner]} == serial['context']
             and cpus == cost['context']['cpu_ids'] and recipe == design['recipe']
             and count == source_count and len(workers) == window > 1 and cost['all_outputs_bitwise'],
             'the source, serial execution, resources and calibration must refer to one exact scope')
    shape = read_artifact(scope['solve_scope'])['sampled_values']['first_structure']
    matrix = dict(size=shape['indptr']['shape'][0]-1, nnz=shape['indices']['shape'][0])
    cap = design['worker_address_space_bytes']
    worker_budget = window*cap
    _require(all(row['worker_virtual_address_caps_bytes'] == worker_budget for row in cost['resources']),
             'measured workers did not enforce their planned address-space limit')
    observed_owner_payload = max(row['owner_peak_pending_input_bytes']+row['owner_pending_result_bound_bytes']
                                 + row['owner_structure_bytes'] for row in cost['resources'])
    base_copy = 12*matrix['nnz'] + 28*matrix['size'] + 4
    planned_memory = worker_budget + observed_owner_payload + base_copy
    directory = Path(out).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    recipe_ref = _write(directory, 'recipe.json', dict(recipe,schema=RECIPE,schema_version=1,source=scope_ref))
    target = read_artifact(serial['problem'])['target']
    problem = _planning(scope_ref, cost, cost_ref, target, count, planned_memory, window)
    problem_ref = _write(directory, 'problem.json', problem_document(problem))
    plan_ref = _write(directory, 'plan.json', construct_implementations(problem))
    runtime = Path(__file__).resolve().parents[1]/'runtime'
    members = [*design['runtime'].values(), _ref(runtime/'sparse_parallel_entry.py', 'scar.python-source')]
    return _write(directory, 'selection.json', dict(schema=SCHEMA, schema_version=1, entry=entry,
        decision='SERIAL', study_parallel=True, source=scope_ref, serial_launch=serial_ref,
        serial_study=serial['study'], recipe=recipe_ref, cost=cost_ref, native=cost['native'],
        problem=problem_ref, plan=plan_ref, context=serial['context'], launch_cpu_ids=cpus, owner_cpu=owner,
        numa_node=choice['choice']['cpu_numa_node'], count=count, matrix=matrix, runtime_members=members,
        workers=dict(cpu_ids=workers, window=window, address_space_limit_bytes=cap),
        symbolic_budget_bytes=attached['budget_bytes'], memory_floor_bytes=planned_memory))


def _peak(tasks):
    events = [(time, delta) for task in tasks for time,delta in (
        (task['started_ns'], 1), (task['ended_ns'], -1))]
    current = peak = 0
    for _, delta in sorted(events):
        current += delta
        peak = max(peak, current)
    return peak


def qualify(study_path, evidence_path, out):
    study_ref = _ref(study_path, SCHEMA)
    study = read_artifact(study_ref)
    evidence_ref = _ref(evidence_path, VALIDATION)
    evidence = read_artifact(evidence_ref)
    cost = check(study['cost']['uri'])
    _require(evidence['schema_version'] == 1 and evidence['selection'] == study_ref and evidence['clean'] is True
             and evidence['context'] == cost['context'], 'fresh-process evidence does not bind the frozen worker plan')
    warm, repeats = evidence['warmup_pairs'], evidence['measurement_pairs']
    modes = ('serial','parallel')
    observations = {(row['pair'],row['mode']):row for row in evidence['runs']}
    _require(type(warm) is int and warm >= 0 and type(repeats) is int and repeats >= 3
             and set(observations) == {(pair,mode) for pair in range(warm+repeats) for mode in modes}
             and len(observations) == len(evidence['runs']), 'matched serial and parallel samples required')
    _require(len({(row['stdout_sha256'],row['stderr_sha256']) for row in observations.values()}) == 1,
             'published outputs changed with the worker team')
    count = study['count']
    clocks = {mode: [] for mode in modes}
    for (pair,mode), row in sorted(observations.items()):
        raw = read_artifact(row['launch'])
        _require(row['launch']['schema'] == EXECUTION and raw['selection_sha256'] == study_ref['sha256']
                 and raw['entry_path'] == study['entry']['uri'] and raw['argv'] == study['context']['argv']
                 and raw['exit_code'] == 0 and raw['statistics']['native_calls'] == count
                 and raw['statistics']['fallback_calls'] == 0 and raw['applied'] == mode.upper(),
                 'run did not fully exercise the intended guarded mode')
        if mode == 'parallel':
            offers = raw['parallel']
            team = offers['workers']
            _require(offers['matched'] == count and offers['serial_calls'] == offers['rejected'] == 0
                     and offers['inputs_captured'] and offers['peak_window'] == study['workers']['window']
                     and team['native_calls'] == count and team['closed']
                     and team['retained_after_close_bytes'] == 0 and team['pending_snapshot_bytes'] == 0
                     and team['submissions'] == count
                     and [item['issue'] for item in team['publications']] == list(range(count))
                     and sorted(worker['cpu'] for worker in team['workers']) == study['workers']['cpu_ids']
                     and _peak(team['tasks']) == study['workers']['window'], 'parallel workers or order differed')
        elif raw['parallel'] is not None:
            raise ValueError('serial comparison launched the worker team')
        _require(type(row['wall_ns']) is int and row['wall_ns'] > 0, 'positive process time required')
        if pair >= warm:
            clocks[mode].append(row['wall_ns'])
    plan = read_artifact(study['plan'])
    model_qualified = (cost['projections']['upper_ns']['elapsed_projection_ns'] <
        cost['serial']['fixed_lifecycle_ns']['lower_ns'] + (count-1)*cost['serial']['warm_ns']['lower_ns'])
    eligible = (model_qualified and max(clocks['parallel']) < min(clocks['serial'])
                and any(row['option'] == 'PARALLEL' for row in plan['selected']))
    directory = Path(out).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return _write(directory, 'selection.json', {**study, 'study_parallel':False,
        'decision':'PARALLEL' if eligible else 'SERIAL', 'study':study_ref, 'validation':evidence_ref,
        'entry_medians_ns':{mode:median(vals) for mode,vals in clocks.items()}})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope')
    parser.add_argument('--serial-launch')
    parser.add_argument('--cost')
    parser.add_argument('--study')
    parser.add_argument('--validation')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if all((args.scope,args.serial_launch,args.cost)) and not args.study:
        result = prepare(args.scope,args.serial_launch,args.cost,args.out)
    elif args.study and args.validation and not args.scope:
        result = qualify(args.study,args.validation,args.out)
    else:
        parser.error('use --scope/--serial-launch/--cost or --study/--validation')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
